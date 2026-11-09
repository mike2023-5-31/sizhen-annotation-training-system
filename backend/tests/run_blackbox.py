"""Run the V1.2 acceptance clients against isolated Uvicorn and a built frontend."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import httpx


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
STEPS = {
    "data": "blackbox_data.py",
    "training": "blackbox_training.py",
    "browser": "blackbox_browser.py",
}
VITE = ["node", str(FRONTEND / "node_modules" / "vite" / "bin" / "vite.js")]


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def environment(work: Path, base: str, backend: str, args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    for key in (
        "MODEL_DIR", "EXPORT_DIR", "UPLOAD_DIR", "LABELS_PATH", "BLACKBOX_SCREENSHOT_DIR",
        "BLACKBOX_FIXTURE_PATH", "BLACKBOX_GPU_SMOKE", "BLACKBOX_EXPECT_GPU_COUNT",
    ):
        env.pop(key, None)
    data = work / "data"
    env.update(
        DATA_DIR=str(data),
        DB_URL="sqlite:///" + (data / "db.sqlite").as_posix(),
        COOKIE_SECURE="false",
        AUTH_ALLOWED_ORIGINS=base,
        CORS_ORIGINS=base,
        VITE_API_TARGET=backend,
        VITE_API_BASE_URL="/api",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
        PYTHONIOENCODING="utf-8",
        PYTHONDONTWRITEBYTECODE="1",
        BLACKBOX_BASE_URL=base,
        BLACKBOX_USERNAME="blackbox-admin",
        BLACKBOX_PASSWORD=secrets.token_urlsafe(24),
        BLACKBOX_EXPECT_DEVICE=args.device,
    )
    if args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    if args.screenshots:
        env["BLACKBOX_SCREENSHOT_DIR"] = str(args.screenshots.resolve())
    if args.gpu_smoke:
        env["BLACKBOX_GPU_SMOKE"] = "1"
        env["BLACKBOX_EXPECT_GPU_COUNT"] = str(args.gpu_smoke)
    elif "training" in args.steps:
        env["BLACKBOX_FIXTURE_PATH"] = str(work / "training-fixture.json")
    return env


def run(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True, timeout=1200)


def prefetch_backbones(env: dict[str, str], *, include_resnet50: bool) -> None:
    """Warm the torchvision ImageNet cache outside the timed training step.

    A cold resnet50 download is ~100 MB and can take minutes on a slow link;
    doing it inside the acceptance job used to trip the per-job timeout. The
    cache lives under XDG_CACHE_HOME, so the server process below reuses it.
    """
    code = (
        "import torchvision.models as m\n"
        "m.resnet18(weights=m.ResNet18_Weights.IMAGENET1K_V1)\n"
    )
    if include_resnet50:
        code += "m.resnet50(weights=m.ResNet50_Weights.IMAGENET1K_V1)\n"
    run([sys.executable, "-c", code], BACKEND, env)
    print("Pretrained backbone cache ready", flush=True)


def stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        if os.name == "nt":
            # A Windows venv launcher may have a Python child that owns the test DB.
            subprocess.run(
                ["taskkill.exe", "/pid", str(process.pid), "/t", "/f"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            )
        else:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


@contextmanager
def server(command: list[str], cwd: Path, env: dict[str, str], log: Path):
    with log.open("wb") as output:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT)
        try:
            yield process
        finally:
            stop(process)


def ready(process: subprocess.Popen, url: str) -> None:
    deadline = time.monotonic() + 60
    with httpx.Client(timeout=2, trust_env=False) as client:
        while process.poll() is None and time.monotonic() < deadline:
            try:
                response = client.get(url)
                if response.status_code == 200 and response.json() == {"status": "ok"}:
                    return
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(0.25)
    raise RuntimeError(f"Service did not become ready at {url} (exit: {process.poll()})")


def verify_http_contract(base: str, backend: str, env: dict[str, str]) -> None:
    """Exercise the built SPA, same-origin proxy and API authentication contract."""
    with httpx.Client(base_url=base, timeout=10, trust_env=False) as client:
        spa = client.get("/datasets")
        assert spa.status_code == 200 and "text/html" in spa.headers["content-type"]
        assert client.get("/api/datasets").status_code == 401
        assert client.get("/static/missing.png").status_code == 401
        credentials = {"username": env["BLACKBOX_USERNAME"], "password": env["BLACKBOX_PASSWORD"]}
        assert client.post("/api/auth/login", json=credentials).status_code == 403
        login = client.post("/api/auth/login", json=credentials, headers={
            "Origin": base, "X-Requested-With": "XMLHttpRequest",
        })
        login.raise_for_status()
        assert client.get("/api/datasets").status_code == 200
        assert client.post("/api/auth/logout").status_code == 403
        response = client.get(f"{backend}/openapi.json")
        response.raise_for_status()
        schema = response.json()
        assert schema["components"]["securitySchemes"]["sessionCookie"]["name"] == "sizhen_session"
        assert schema["paths"]["/api/health"]["get"]["security"] == []
        operations = schema["paths"]["/api/datasets"]
        assert operations["get"]["security"] == [{"sessionCookie": []}]
        assert operations["post"]["security"] == [{"sessionCookie": [], "csrfHeader": []}]
        assert client.post("/api/auth/logout", headers={
            "Origin": base, "X-CSRF-Token": login.json()["csrf_token"],
        }).status_code == 200
        assert client.get("/api/datasets").status_code == 401


def run_acceptance(args: argparse.Namespace, work: Path) -> None:
    backend_port = free_port()
    backend = f"http://127.0.0.1:{backend_port}"
    base = backend
    if not args.gpu_smoke:
        frontend_port = free_port()
        while frontend_port == backend_port:
            frontend_port = free_port()
        base = f"http://127.0.0.1:{frontend_port}"
    env = environment(work, base, backend, args)
    out_dir = work / "frontend"
    run([sys.executable, "-m", "app.migrations"], BACKEND, env)
    run(
        [sys.executable, "-c",
         "import os; from app.services.auth_service import create_user; "
         "create_user(os.environ['BLACKBOX_USERNAME'], os.environ['BLACKBOX_PASSWORD'], 'admin')"],
        BACKEND, env,
    )
    if not args.gpu_smoke:
        run([*VITE, "build", "--outDir", str(out_dir)], FRONTEND, env)
    if "training" in args.steps:
        prefetch_backbones(env, include_resnet50=not args.gpu_smoke)

    with ExitStack() as stack:
        app = stack.enter_context(server(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
             "--port", str(backend_port), "--workers", "1"],
            BACKEND, env, work / "backend.log",
        ))
        ready(app, f"{backend}/api/health")
        if args.gpu_smoke:
            print(f"Isolated GPU test API ready at {backend}", flush=True)
        else:
            frontend = stack.enter_context(server(
                [*VITE, "preview", "--host", "127.0.0.1", "--port", str(frontend_port),
                 "--strictPort", "--outDir", str(out_dir)],
                FRONTEND, env, work / "frontend.log",
            ))
            ready(frontend, f"{base}/api/health")
            print(f"Isolated production build and API ready at {base}", flush=True)
            verify_http_contract(base, backend, env)
            print("HTTP contract, login, CSRF, protected proxy and logout passed", flush=True)
        for step in args.steps:
            print(f"\n=== V1.2 black-box: {step} ===", flush=True)
            run([sys.executable, str(Path(__file__).with_name(STEPS[step]))], BACKEND, env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", nargs="+", choices=STEPS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpu-smoke", type=int, choices=(1, 2),
                        help="Run only the real HTTP AMP smoke test, expecting this many visible GPUs")
    parser.add_argument("--screenshots", type=Path, help="Optional output directory for browser screenshots")
    args = parser.parse_args()
    if args.gpu_smoke:
        if args.steps or args.screenshots:
            parser.error("--gpu-smoke is separate from the full suite")
        args.steps, args.device = ["training"], "cuda"
    else:
        args.steps = args.steps or list(STEPS)
        if ("training" in args.steps and "browser" in args.steps
                and args.steps.index("training") > args.steps.index("browser")):
            parser.error("run training before browser when both steps are selected")
    if not __debug__:
        parser.error("run without -O to keep acceptance assertions enabled")
    with TemporaryDirectory(prefix="sizhen-blackbox-", ignore_cleanup_errors=True) as temporary:
        work = Path(temporary)
        try:
            run_acceptance(args, work)
        except (RuntimeError, subprocess.SubprocessError):
            for log in ("backend.log", "frontend.log"):
                path = work / log
                if path.is_file():
                    print(f"\n--- {log} ---\n{path.read_text(encoding='utf-8', errors='replace')[-10000:]}",
                          file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
