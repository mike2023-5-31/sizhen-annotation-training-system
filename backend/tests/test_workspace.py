from __future__ import annotations

from fastapi.testclient import TestClient

from app import config
from app.main import app
from app.services.inference_service import BatchInferenceJobs


def test_models_lists_checkpoints(tmp_path, monkeypatch, login_admin) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "best.pth").write_bytes(b"checkpoint")
    (tmp_path / "nested" / "latest.pt").write_bytes(b"checkpoint")
    (tmp_path / "ignore.txt").write_text("not a checkpoint", encoding="utf-8")
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path)

    client = login_admin(TestClient(app))
    response = client.get("/api/models")
    assert response.status_code == 200, response.text
    items = response.json()["items"]
    assert {item["model_path"] for item in items} == {"best.pth", "nested/latest.pt"}
    assert all(item["size_bytes"] > 0 and item["modified_at"] for item in items)


def test_onnx_lists_exports_and_source_digest(tmp_path, monkeypatch, login_admin) -> None:
    digest = "a" * 64
    (tmp_path / f"model-{digest}.onnx").write_bytes(b"onnx")
    (tmp_path / "plain.onnx").write_bytes(b"onnx")
    monkeypatch.setattr(config, "EXPORT_DIR", tmp_path)

    client = login_admin(TestClient(app))
    response = client.get("/api/onnx")
    assert response.status_code == 200, response.text
    items = {item["onnx_name"]: item for item in response.json()["items"]}
    assert items[f"model-{digest}.onnx"]["source_sha256"] == digest
    assert items["plain.onnx"]["source_sha256"] is None
    assert items[f"model-{digest}.onnx"]["size_bytes"] == 4


def test_batch_jobs_list_omits_predictions() -> None:
    jobs = BatchInferenceJobs()
    with jobs._lock:
        jobs._jobs["job-a"] = {
            "job_id": "job-a",
            "status": "completed",
            "processed": 2,
            "total": 2,
            "predictions": [{"filename": "a.png"}],
            "error": None,
            "created_at": "2026-11-09T00:00:00+00:00",
        }

    items = jobs.list()
    assert len(items) == 1
    assert items[0]["job_id"] == "job-a"
    assert items[0]["prediction_count"] == 1
    assert "predictions" not in items[0]


def test_import_rejects_oversized_upload(monkeypatch, login_admin) -> None:
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 8)
    client = login_admin(TestClient(app))
    response = client.post(
        "/api/datasets/1/import",
        files={"file": ("big.zip", b"0123456789", "application/zip")},
    )
    assert response.status_code == 413
