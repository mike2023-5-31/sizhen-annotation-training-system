"""Exercise the model pipeline against a disposable running server over HTTP and WS.

Set BLACKBOX_BASE_URL, BLACKBOX_USERNAME and BLACKBOX_PASSWORD. The server
must have an initialized test database and an administrator account. This
script creates datasets, uploads trusted test code and writes model artifacts.
Set BLACKBOX_GPU_SMOKE=1 with BLACKBOX_EXPECT_GPU_COUNT to run one bounded
ResNet18 AMP job on a CUDA server instead of the full CPU acceptance flow.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import time
import uuid
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlsplit, urlunsplit
from xml.etree import ElementTree
from zipfile import ZipFile

import httpx
import numpy as np
import onnx
import onnxruntime
from PIL import Image, ImageDraw
from websockets.sync.client import connect


CUSTOM_MODEL = b'''\
from torch import nn


class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.heads = nn.ModuleList(nn.Linear(3, count) for count in num_classes)

    def forward(self, x):
        features = self.pool(x).flatten(1)
        return tuple(head(features) for head in self.heads)
'''


def request(client, method, path, status=200, **kwargs):
    response = client.request(method, f"/api{path}", **kwargs)
    assert response.status_code == status, f"{method} {path}: {response.status_code} {response.text[:1000]}"
    return response


def make_images(prefix, count, offset):
    images = {}
    for index in range(count):
        shade = index + offset
        picture = Image.new("RGB", (52, 44), (shade * 7 % 256, shade * 19 % 256, shade * 13 % 256))
        ImageDraw.Draw(picture).rectangle((5 + index, 5, 32 + index, 35), fill=(220, 100, 55))
        buf = io.BytesIO()
        picture.save(buf, format="PNG")
        images[f"{prefix}-{index:02d}.png"] = buf.getvalue()
    return images


def create_dataset(client, name, images):
    dataset_id = request(client, "POST", "/datasets", 201, json={"name": name}).json()["id"]
    if images:
        buf = io.BytesIO()
        with ZipFile(buf, "w") as archive:
            for filename, content in images.items():
                archive.writestr(f"images/{filename}", content)
        report = request(client, "POST", f"/datasets/{dataset_id}/import",
                         files={"file": ("images.zip", buf.getvalue(), "application/zip")}).json()
        assert report["imported"] == len(images) and not report["errors"], report
    return dataset_id


def label_path(dataset_id, filename):
    return f"/datasets/{dataset_id}/image/{quote(filename)}/labels"


def label_image(client, dataset_id, filename, labels, quality="合格"):
    path = label_path(dataset_id, filename)
    before = request(client, "GET", path).json()
    return request(client, "PUT", path, json={
        "labels": labels, "quality": quality, "expected_revision": before["revision"],
    }).json()


def labels_for(categories, index):
    return [{"category": cat["name"], "name": cat["labels"][index % min(2, len(cat["labels"]))]}
            for cat in categories]


def training_ws(client):
    url = urlsplit(str(client.base_url))
    ws_url = urlunsplit(("wss" if url.scheme == "https" else "ws", url.netloc,
                         "/api/training/ws/training", "", ""))
    handshake = client.build_request("GET", "/api/training/status")
    return connect(ws_url, origin=f"{url.scheme}://{url.netloc}",
                   additional_headers={"Cookie": handshake.headers["cookie"]}, close_timeout=3)


def wait_job(client, job_id, timeout=600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        jobs = request(client, "GET", "/training/status", params={"job_id": job_id}).json()["jobs"]
        assert len(jobs) == 1, jobs
        if jobs[0]["status"] not in {"created", "running"}:
            return jobs[0]
        time.sleep(0.2)
    raise AssertionError(f"Training timed out: {job_id}")


def start_training(client, config):
    return request(client, "POST", "/training/start", json=config).json()["job_id"]


def check_job(job, epochs, backbone, expect_amp=False):
    assert job["status"] == "completed", job
    assert job["config"]["backbone"] == backbone, job
    expected = os.environ.get("BLACKBOX_EXPECT_DEVICE", "cpu")
    assert job["device"].split(":", 1)[0] == expected, job
    if expected == "cpu":
        assert job["gpu_count"] == 0, job
    assert job["use_amp"] is expect_amp, job
    assert [metric["epoch"] for metric in job["metrics"]] == epochs, job
    assert job["current_epoch"] == job["total_epochs"] == epochs[-1], job
    for metric in job["metrics"]:
        assert math.isfinite(metric["train_loss"]), metric
        assert 0 <= metric["train_acc"] <= 1, metric


def checkpoint_name(path):
    return PurePosixPath(path.replace("\\", "/")).name


def train_models(client, categories, primary, secondary, empty):
    options = {"datasets": [primary, secondary], "backbone": "resnet18", "epochs": 1,
               "batch_size": 4, "val_split": 0.5, "fp16": True, "lr": 0.0002,
               "optimizer": "sgd", "dropout": 0.1, "loss": "focalloss",
               "augmentation": ["flip", "rotation", "colorjitter", "crop"]}
    before = request(client, "GET", "/training/status").json()["jobs"]
    for invalid in ({"lr": 0}, {"batch_size": 0}, {"epochs": 0}, {"optimizer": "other"},
                    {"loss": "other"}, {"dropout": 1}, {"datasets": [empty]}):
        request(client, "POST", "/training/start", 400, json={**options, **invalid})
    assert request(client, "GET", "/training/status").json()["jobs"] == before

    with training_ws(client) as ws:
        snapshot = json.loads(ws.recv(timeout=10))
        assert snapshot["type"] == "snapshot" and isinstance(snapshot["jobs"], list), snapshot
        first_id = start_training(client, options)
        assert first_id not in {item["job_id"] for item in snapshot["jobs"]}
        deadline = time.monotonic() + 180
        events = []
        while True:
            frame = json.loads(ws.recv(timeout=max(0.01, deadline - time.monotonic())))
            if frame.get("job_id") != first_id:
                continue
            if frame["type"] == "metrics":
                events.extend(frame["metrics"])
            if frame["type"] == "status" and frame["status"] in {"completed", "stopped", "failed"}:
                assert frame["status"] == "completed", frame
                break
    first = wait_job(client, first_id)
    check_job(first, [1], "resnet18", expect_amp=os.environ.get("BLACKBOX_EXPECT_DEVICE") == "cuda")
    assert first["metrics"] == events, (first, events)
    with training_ws(client) as ws:
        refreshed = json.loads(ws.recv(timeout=10))
        assert refreshed["type"] == "snapshot"
        assert next(job for job in refreshed["jobs"] if job["job_id"] == first_id)["metrics"] == events
    config = first["config"]
    for key in ("datasets", "epochs", "batch_size", "lr", "optimizer", "dropout", "loss"):
        assert config[key] == options[key], config
    assert config["augmentation"] == ["randomhorizontalflip", "randomrotation", "colorjitter", "randomcrop"]
    assert events[0]["train_valid_per_head"] == events[0]["val_valid_per_head"] == [4, 4, 4, 0], events
    assert math.isfinite(events[0]["val_loss"]) and 0 <= events[0]["val_acc"] <= 1
    assert events[0]["eta_seconds"] == 0 and events[0]["lr"] > 0 and events[0]["recorded_at"]
    if first["device"] == "cpu":
        assert events[0]["gpu_metrics"] == []

    evaluation = request(client, "GET", f"/training/eval/{first_id}").json()
    absent = categories[-1]["name"]
    assert absent not in evaluation["confusion"] and absent not in evaluation["roc"], evaluation
    assert evaluation["details"][absent]["valid_count"] == 0, evaluation
    for cat in categories[:-1]:
        name = cat["name"]
        matrix = evaluation["confusion"][name]
        assert len(matrix) == len(cat["labels"]) and all(len(row) == len(matrix) for row in matrix)
        assert sum(map(sum, matrix)) == evaluation["details"][name]["valid_count"] == 4
        assert 0 <= evaluation["roc"][name]["auc"] <= 1, evaluation["roc"][name]
    assert evaluation["checkpoint"]["path"] == first["checkpoint_best"]
    print(f"PASS .6/.7/.8 CPU fallback/.9/.10/.12/.13 ResNet18 + combined datasets + live WS: {first_id}", flush=True)

    for bad in ("../outside.pth", "missing.pth"):
        request(client, "POST", "/training/resume", 400, json={"checkpoint_path": bad})
    resume_id = request(client, "POST", "/training/resume", json={
        "checkpoint_path": checkpoint_name(first["checkpoint_latest"]), "epochs": 1,
    }).json()["job_id"]
    resumed = wait_job(client, resume_id)
    check_job(resumed, [2], "resnet18", expect_amp=first["use_amp"])
    assert resumed["start_epoch"] == 1 and resumed["resume_from"] == first["checkpoint_latest"], resumed
    assert resumed["config"]["datasets"] == options["datasets"], resumed
    assert math.isclose(resumed["metrics"][0]["lr"], first["metrics"][0]["lr"] * 0.1), resumed
    resumed_eval = request(client, "GET", f"/training/eval/{resume_id}").json()
    assert resumed_eval["confusion"] and resumed_eval["checkpoint"]["epoch"] in (1, 2), resumed_eval
    print(f"PASS .11 ResNet18 checkpoint resumes at epoch 2: {resume_id}", flush=True)

    no_val = {**options, "fp16": False, "val_split": 0.0, "augmentation": []}
    resnet50 = wait_job(client, start_training(client, {**no_val, "backbone": "resnet50"}))
    check_job(resnet50, [1], "resnet50")
    assert request(client, "GET", f"/training/eval/{resnet50['job_id']}").json() == {"confusion": {}, "roc": {}}

    custom = request(client, "POST", "/models/custom",
                     files={"file": ("blackbox_model.py", CUSTOM_MODEL, "text/x-python")}).json()["model_path"]
    assert hashlib.sha256(CUSTOM_MODEL).hexdigest() in custom, custom
    custom_job = wait_job(client, start_training(client, {**no_val, "backbone": "custom", "custom_model_path": custom}))
    check_job(custom_job, [1], "custom")
    long_job_id = start_training(client, {**no_val, "epochs": 100, "backbone": "custom", "custom_model_path": custom})
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        jobs = request(client, "GET", "/training/status", params={"job_id": long_job_id}).json()["jobs"]
        if jobs[0]["metrics"] or jobs[0]["status"] != "running":
            break
        time.sleep(0.05)
    assert jobs[0]["metrics"] and jobs[0]["status"] == "running", jobs[0]
    assert request(client, "POST", f"/training/stop/{long_job_id}").json() == {"ok": True}
    stopped = wait_job(client, long_job_id)
    assert stopped["status"] == "stopped" and 0 < stopped["current_epoch"] < 100, stopped
    restarted_id = request(client, "POST", "/training/resume", json={
        "checkpoint_path": checkpoint_name(stopped["checkpoint_latest"]), "epochs": 1,
    }).json()["job_id"]
    check_job(wait_job(client, restarted_id), [stopped["current_epoch"] + 1], "custom")
    print(f"PASS .11/.12 stopped custom job at epoch {stopped['current_epoch']} and resumed from the saved epoch", flush=True)
    models = request(client, "GET", "/inference/models").json()["models"]
    for job in (first, resumed, resnet50, custom_job):
        assert checkpoint_name(job["checkpoint_latest"]) in models, models
    assert checkpoint_name(first["checkpoint_best"]) in models, models
    assert checkpoint_name(resnet50["checkpoint_best"]) not in models, models
    print(f"PASS .7/.14 ResNet50 + trusted custom training and model enumeration: {resnet50['job_id']}, {custom_job['job_id']}", flush=True)
    return first, resnet50, custom_job, evaluation


def verify_capabilities(client):
    capabilities = request(client, "GET", "/training/capabilities").json()
    if os.environ.get("BLACKBOX_EXPECT_DEVICE", "cpu") == "cpu":
        assert capabilities == {"device": "cpu", "gpu_count": 0, "parallel_mode": "cpu", "devices": []}, capabilities
        print("PASS .8 CPU capability snapshot with zero CUDA devices", flush=True)
    else:
        count = int(os.environ.get("BLACKBOX_EXPECT_GPU_COUNT", capabilities["gpu_count"]))
        assert count > 0 and capabilities["gpu_count"] == count, capabilities
        assert capabilities["device"] == "cuda:0"
        assert capabilities["parallel_mode"] == ("data_parallel" if count > 1 else "single_gpu"), capabilities
        assert [device["index"] for device in capabilities["devices"]] == list(range(count)), capabilities
        for device in capabilities["devices"]:
            assert device["name"] and device["total_bytes"] > 0 and device["status"] == "detected", device
            temperature = device["temperature_c"]
            utilization = device["utilization_percent"]
            assert temperature is None or isinstance(temperature, int) and temperature > -273, device
            assert utilization is None or 0 <= utilization <= 100, device
        print(f"PASS .8 CUDA capability snapshot: {capabilities['devices']}", flush=True)
    return capabilities


def gpu_smoke(client, primary, secondary, capabilities):
    count = int(os.environ["BLACKBOX_EXPECT_GPU_COUNT"])
    assert count > 0 and os.environ.get("BLACKBOX_EXPECT_DEVICE") == "cuda"
    job = wait_job(client, start_training(client, {
        "datasets": [primary, secondary], "backbone": "resnet18", "epochs": 1,
        "batch_size": 4, "val_split": 0.5, "lr": 0.0002, "optimizer": "sgd", "fp16": True,
    }))
    check_job(job, [1], "resnet18", expect_amp=True)
    assert job["gpu_count"] == count, job
    assert job["metrics"][0]["train_valid_per_head"] == [4, 4, 4, 0], job
    assert job["metrics"][0]["val_valid_per_head"] == [4, 4, 4, 0], job
    devices = job["metrics"][0]["gpu_metrics"]
    assert [gpu["index"] for gpu in devices] == list(range(count)), devices
    assert all(0 < gpu["used_bytes"] <= gpu["total_bytes"] for gpu in devices), devices
    assert all(0 < gpu["peak_allocated_bytes"] <= gpu["total_bytes"] for gpu in devices), devices
    assert all(0 <= gpu["utilization_percent"] <= 100 for gpu in devices), devices
    assert [gpu["total_bytes"] for gpu in devices] == [gpu["total_bytes"] for gpu in capabilities["devices"]], devices
    print(f"PASS .8 real CUDA ResNet18 AMP on {count} visible device(s): {job['job_id']}; "
          f"allocator peaks {[gpu['peak_allocated_bytes'] for gpu in devices]}", flush=True)


def verify_inference(client, categories, model_name, custom_name, dataset_id, empty_id, image_names, manual):
    payload = {"model_path": model_name, "dataset_id": dataset_id}
    request(client, "POST", "/inference/predict", 404, json={**payload, "model_path": "missing.pth"})
    request(client, "POST", "/inference/predict", 400, json={**payload, "model_path": "../outside.pth"})
    request(client, "POST", "/inference/predict", 404, json={**payload, "dataset_id": 2147483647})
    assert request(client, "POST", "/inference/predict", json={**payload, "dataset_id": empty_id}).json() == {"predictions": []}

    results = request(client, "POST", "/inference/predict", json=payload).json()["predictions"]
    assert [row["filename"] for row in results] == image_names, results
    for row in results:
        assert [pred["category"] for pred in row["predictions"]] == [cat["name"] for cat in categories]
        for pred, category in zip(row["predictions"], categories):
            assert pred["label"] in category["labels"] and 0 <= pred["confidence"] <= 1, pred
    accepted = request(client, "POST", "/inference/jobs", 202, json=payload)
    status_path = f"/inference/jobs/{accepted.json()['job_id']}"
    assert accepted.headers["location"] == f"/api{status_path}"
    processed = 0
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        status = request(client, "GET", status_path).json()
        assert processed <= status["processed"] <= status["total"] == len(image_names), status
        processed = status["processed"]
        if status["status"] != "running":
            break
        time.sleep(0.1)
    assert status["status"] == "completed" and processed == status["total"], status
    assert status["predictions"] == results, status
    custom = request(client, "POST", "/inference/predict", json={**payload, "model_path": custom_name}).json()
    assert [row["filename"] for row in custom["predictions"]] == image_names

    preannotation = "/inference/preannotation"
    assert max(p["confidence"] for row in results[1:] for p in row["predictions"]) < 1
    assert request(client, "POST", preannotation, json={**payload, "confidence_threshold": 1}).json() == {"generated": 0}
    assert request(client, "POST", preannotation, json=payload).json() == {"generated": len(image_names) - 1}
    assert request(client, "POST", preannotation, json=payload).json() == {"generated": 0}
    assert request(client, "GET", label_path(dataset_id, image_names[0])).json() == manual
    exported = request(client, "GET", f"/datasets/{dataset_id}/export").content
    with ZipFile(io.BytesIO(exported)) as archive:
        for row in results[1:]:
            filename = row["filename"]
            xml = ElementTree.fromstring(archive.read(f"annotations/{Path(filename).stem}.xml"))
            assert xml.findtext("annotation_source") == "model" and xml.findtext("filename") == filename
            actual = [(label.findtext("category"), label.findtext("name"), float(label.findtext("confidence")))
                      for label in xml.findall("./labels/label")]
            expected = [(pred["category"], pred["label"], pred["confidence"]) for pred in row["predictions"]]
            assert actual == expected, (actual, expected)
    print(f"PASS .15/.16/.17/.18 {len(results)} images across batches, confidence filter and XML export", flush=True)


def verify_onnx(client, categories, model_name, custom_name, dataset_id, images, source_sha256):
    request(client, "POST", "/onnx/convert", 400, json={"model_path": custom_name})
    converted = request(client, "POST", "/onnx/convert", json={"model_path": model_name}).json()
    onnx_name = converted["onnx_name"]
    assert converted["source_sha256"] == source_sha256 and source_sha256 in onnx_name, converted
    assert converted["input_shape"] == [1, 3, 224, 224] and converted["opset"] >= 11, converted
    report = request(client, "POST", "/onnx/compare", json={
        "model_path": model_name, "onnx_name": onnx_name, "dataset_id": dataset_id,
    }).json()
    assert report["sample_type"] == "dataset" and report["sample_count"] == 8, report
    assert report["sample_filenames"] == list(images)[:8] and report["source_sha256"] == source_sha256, report
    assert report["overall_pass"] and [item["category"] for item in report["results"]] == [cat["name"] for cat in categories]
    assert all(item["pass"] and item["max_abs_diff"] < 1e-3 and item["top1_agreement"] == 1
               for item in report["results"]), report

    path = f"/onnx/{quote(onnx_name)}/download"
    with httpx.Client(base_url=client.base_url, trust_env=False) as anonymous:
        request(anonymous, "GET", path, 401)
    downloaded = request(client, "GET", path)
    assert onnx_name in downloaded.headers["content-disposition"]
    assert int(downloaded.headers["content-length"]) == len(downloaded.content) > 0
    assert downloaded.headers["last-modified"] and converted["exported_at"].endswith("+00:00")
    assert converted["size_bytes"] == len(downloaded.content)
    graph = onnx.load_model_from_string(downloaded.content)
    onnx.checker.check_model(graph)
    assert graph.opset_import[0].version == converted["opset"]
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    session = onnxruntime.InferenceSession(downloaded.content, sess_options=options, providers=["CPUExecutionProvider"])
    metadata = session.get_modelmeta().custom_metadata_map
    assert metadata["sizhen.source_sha256"] == source_sha256
    contract = json.loads(metadata["sizhen.inference_contract"])
    assert contract["input"]["shape"] == converted["input_shape"]
    assert [item["labels"] for item in contract["outputs"]] == [cat["labels"] for cat in categories]
    assert session.get_inputs()[0].shape == converted["input_shape"]
    assert [output.name for output in session.get_outputs()] == [f"head_{index}" for index in range(len(categories))]
    with Image.open(io.BytesIO(next(iter(images.values())))) as first:
        pixels = np.asarray(first.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32)
    batch = np.transpose(pixels / 255, (2, 0, 1))[None, ...]
    outputs = session.run(None, {"input": batch})
    for output, category, head in zip(outputs, categories, report["image_results"][0]["heads"]):
        assert output.shape == (1, len(category["labels"])) and np.isfinite(output).all()
        assert int(output.argmax()) == head["onnx_top1_index"], head
    print(f"PASS .19/.20/.21 ONNX validation, 8 comparison images and authenticated runtime download: "
          f"{len(downloaded.content)} bytes, max abs diff {max(item['max_abs_diff'] for item in report['results']):.3g}", flush=True)


def main():
    base = os.environ["BLACKBOX_BASE_URL"]
    prefix = f"V1.2-ML-{uuid.uuid4().hex[:8]}"
    with httpx.Client(base_url=base, timeout=120, trust_env=False) as client:
        session = request(client, "POST", "/auth/login",
                          headers={"Origin": base.rstrip("/"), "X-Requested-With": "XMLHttpRequest"},
                          json={"username": os.environ["BLACKBOX_USERNAME"],
                                "password": os.environ["BLACKBOX_PASSWORD"]}).json()
        client.headers.update({"X-CSRF-Token": session["csrf_token"]})
        capabilities = verify_capabilities(client)
        categories = request(client, "GET", "/labels").json()["categories"]
        primary_images, secondary_images = make_images("primary", 5, 1), make_images("aux", 5, 9)
        primary = create_dataset(client, f"{prefix}-primary", primary_images)
        secondary = create_dataset(client, f"{prefix}-auxiliary", secondary_images)
        empty = create_dataset(client, f"{prefix}-empty", {})
        for dataset_id, images in ((primary, primary_images), (secondary, secondary_images)):
            for index, filename in enumerate(images):
                quality = "合格" if index < 4 else "不合格"
                label_image(client, dataset_id, filename, labels_for(categories[:-1], index), quality)

        if os.environ.get("BLACKBOX_GPU_SMOKE") == "1":
            gpu_smoke(client, primary, secondary, capabilities)
            return

        first, resnet50, custom, evaluation = train_models(client, categories, primary, secondary, empty)
        images = make_images("inference", 18, 17)
        name = f"{prefix}-inference"
        dataset_id = create_dataset(client, name, images)
        image_names = list(images)
        label_image(client, dataset_id, image_names[0], labels_for(categories, 0))
        manual = request(client, "GET", label_path(dataset_id, image_names[0])).json()
        model_name = checkpoint_name(first["checkpoint_best"])
        custom_name = checkpoint_name(custom["checkpoint_latest"])
        verify_inference(client, categories, model_name, custom_name, dataset_id, empty, image_names, manual)
        verify_onnx(client, categories, model_name, custom_name, dataset_id, images, evaluation["checkpoint"]["sha256"])
        resnet50_export = request(client, "POST", "/onnx/convert", json={
            "model_path": checkpoint_name(resnet50["checkpoint_latest"]),
        }).json()
        assert resnet50_export["source_sha256"] != evaluation["checkpoint"]["sha256"]
        request(client, "POST", "/onnx/compare", 400, json={
            "model_path": model_name, "onnx_name": resnet50_export["onnx_name"], "dataset_id": dataset_id,
        })
        print("PASS .19 ResNet50 conversion and cross-source artifact rejection", flush=True)

        fixture = os.environ.get("BLACKBOX_FIXTURE_PATH")
        if fixture:
            path = Path(fixture)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"job_id": first["job_id"], "model_name": model_name,
                                             "dataset_id": dataset_id, "dataset_name": name, "image_filenames": image_names,
                                             "primary_dataset_id": primary, "secondary_dataset_id": secondary}), encoding="utf-8")
            temporary.replace(path)


if __name__ == "__main__":
    main()
