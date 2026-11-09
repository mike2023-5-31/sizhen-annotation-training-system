"""V1.2 .1–.5 HTTP acceptance against a disposable authenticated server.

Set BLACKBOX_BASE_URL, BLACKBOX_USERNAME and BLACKBOX_PASSWORD for a test
administrator. All images and annotations in this script are synthetic.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import io
import os
from pathlib import PurePosixPath
import secrets
from urllib.parse import quote
from xml.etree import ElementTree as ET
from zipfile import ZipFile, ZipInfo

import httpx
from PIL import Image


def request(client, method, path, status=200, **kwargs):
    response = client.request(method, f"/api{path}", **kwargs)
    assert response.status_code == status, f"{method} {path}: {response.status_code} {response.text[:500]}"
    return response


def login(client, username, password):
    response = request(
        client, "POST", "/auth/login", json={"username": username, "password": password},
        headers={"X-Requested-With": "XMLHttpRequest"},
    ).json()
    assert client.post("/api/auth/logout").status_code == 403
    client.headers["X-CSRF-Token"] = response["csrf_token"]
    return response["user"]


def png(color):
    output = io.BytesIO()
    Image.new("RGB", (19, 11), color).save(output, "PNG")
    return output.getvalue()


def annotation(filename, timestamp, quality="", labels=(), source=None, collector=None):
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = PurePosixPath(filename).name
    ET.SubElement(root, "timestamp").text = timestamp
    ET.SubElement(root, "quality").text = quality
    labels_node = ET.SubElement(root, "labels")
    for label in labels:
        node = ET.SubElement(labels_node, "label")
        for field, value in label.items():
            ET.SubElement(node, field).text = str(value)
    if source:
        ET.SubElement(root, "annotation_source").text = source
    if collector is not None:
        ET.SubElement(root, "collector_alias").text = collector
    return ET.tostring(root, encoding="utf-8")


def archive(entries):
    output = io.BytesIO()
    with ZipFile(output, "w") as bundle:
        for name, content in entries.items():
            bundle.writestr(ZipInfo(name, date_time=(2039, 1, 1, 12, 0, 0)), content)
    return output.getvalue()


def import_archive(client, dataset, content, status=200):
    return request(client, "POST", f"{dataset}/import", status,
                   files={"file": ("验收合成数据.zip", content, "application/zip")}).json()


def export_archive(client, dataset):
    response = request(client, "GET", f"{dataset}/export")
    assert response.headers["content-type"] == "application/zip"
    assert "filename*=UTF-8''" in response.headers["content-disposition"]
    return response.content


def exported_entries(content):
    with ZipFile(io.BytesIO(content)) as bundle:
        return {name: bundle.read(name) for name in bundle.namelist()}


def dataset_path(client, name, modality=None):
    data = request(client, "POST", "/datasets", 201, json={"name": name, "modality": modality}).json()
    assert data["name"] == name
    assert data["modality"] == modality
    assert data["created_at"] and data["updated_at"]
    return f"/datasets/{data['id']}"


def label_path(dataset, filename):
    return f"{dataset}/image/{quote(filename, safe='')}/labels"


def test_history_isolation(admin, name, label):
    previous = dataset_path(admin, name)
    first_zip = archive({"样本.png": png("red")})
    second_zip = archive({"样本.png": png("blue")})
    import_archive(admin, previous, first_zip)
    url = label_path(previous, "样本.png")
    history_url = url.removesuffix("/labels") + "/history"
    revision = request(admin, "GET", url).json()["revision"]
    request(admin, "PUT", url, json={"labels": [label], "expected_revision": revision})
    assert len(request(admin, "GET", history_url).json()["items"]) == 1
    request(admin, "DELETE", previous)
    request(admin, "GET", history_url, 404)

    replacement = dataset_path(admin, name)
    assert replacement != previous
    assert import_archive(admin, replacement, second_zip)["files"][0]["filename"] == "样本.png"
    url = label_path(replacement, "样本.png")
    image_url = url.removesuffix("/labels")
    history_url = image_url + "/history"
    assert request(admin, "GET", history_url).json() == {"items": [], "has_more": False}
    revision = request(admin, "GET", url).json()["revision"]
    request(admin, "PUT", url, json={"labels": [label], "expected_revision": revision})
    request(admin, "DELETE", image_url)
    request(admin, "GET", history_url, 404)
    request(admin, "GET", f"{replacement}/images/{quote('样本.png')}", 404)
    assert admin.get(f"/static/{replacement.rsplit('/', 1)[-1]}/{quote('样本.png')}").status_code == 404
    assert exported_entries(export_archive(admin, replacement)) == {}
    assert import_archive(admin, replacement, first_zip)["files"][0]["filename"] == "样本.png"
    assert request(admin, "GET", history_url).json() == {"items": [], "has_more": False}
    request(admin, "DELETE", replacement)


def test_data_flow(admin, member, prefix, actor_ids):
    categories = request(admin, "GET", "/labels").json()["categories"]
    assert len(categories) == 4
    category = categories[0]
    first = {"category": category["name"], "name": category["labels"][0]}
    second = {"category": category["name"], "name": category["labels"][1]}
    timestamp = datetime(2042, 1, 1, tzinfo=timezone.utc) + timedelta(
        seconds=int(prefix[-8:], 16) % (365 * 86400)
    )
    stamp = timestamp.isoformat()
    test_history_isolation(admin, prefix + "-历史隔离", first)
    print(".1/.4 已验证删除并重建数据集和图片时的历史隔离", flush=True)

    name = prefix + "-原始数据"
    dataset = dataset_path(admin, name, "舌")
    dataset_id = int(dataset.rsplit("/", 1)[-1])
    request(admin, "POST", "/datasets", 409, json={"name": f" {name} "})
    renamed = request(admin, "PATCH", dataset, json={"name": f"{name}-改名"}).json()
    assert renamed["modality"] == "舌"
    assert exported_entries(export_archive(admin, dataset)) == {}
    stats = request(admin, "GET", "/datasets", params={"include_stats": "true"}).json()
    empty = next(item for item in stats if item["id"] == dataset_id)
    assert empty["image_count"] == empty["annotated_count"] == 0
    assert empty["name"] == f"{name}-改名" and empty["modality"] == "舌"
    assert request(admin, "GET", f"{dataset}/imports").json() == {"items": [], "has_more": False}

    image = "舌诊_%样本.PNG"
    collector = "COL-aB1234cd"
    invalid_collector = "未脱敏测试姓名"
    images = {
        f"现场/{image}": png("gold"),
        "诊室甲/同名.png": png("green"),
        "诊室乙/同名.png": png("blue"),
        "现场/补采.png": png("white"),
        "现场/待标注.png": png("orange"),
    }
    xmls = {
        f"现场/{image}": annotation(
            image, stamp, labels=[{**first, "confidence": 0.61}], source="model", collector=collector,
        ),
        "诊室甲/同名.png": annotation("同名.png", stamp, "合格", [second]),
        "诊室乙/同名.png": annotation("同名.png", stamp, "不合格"),
        "现场/补采.png": annotation("补采.png", stamp),
    }
    source_archive = archive({**images, **{key + ".xml": value for key, value in xmls.items()},
                              "不可解码.png": b"invalid image",
                              "错误代号.png": png("pink"),
                              "错误代号.xml": annotation("错误代号.png", stamp, collector=invalid_collector)})
    import_archive(admin, dataset, b"not a zip", 400)
    import_archive(admin, dataset, archive({"../escape.png": png("red")}), 400)
    result = import_archive(admin, dataset, source_archive)
    assert result["imported"] == 5 and result["initialized"] == 1 and len(result["errors"]) == 2
    assert any("采集人脱敏代号" in error for error in result["errors"])
    assert all(invalid_collector not in error for error in result["errors"])
    names = {item["source"]: item["filename"] for item in result["files"]}
    assert names["诊室甲/同名.png"] != names["诊室乙/同名.png"]
    assert names[f"现场/{image}"] == image
    duplicate = import_archive(admin, dataset, source_archive)
    assert duplicate["imported"] == 0 and duplicate["duplicates"] == 5 and len(duplicate["errors"]) == 2
    page = request(member, "GET", f"{dataset}/imports", params={"limit": 1}).json()
    assert page["has_more"] and len(page["items"]) == 1
    batch = page["items"][0]
    assert batch["filename"] == "验收合成数据.zip"
    assert batch["source"] == "zip" and batch["actor_id"] == actor_ids[0]
    assert batch["created_at"].endswith("Z")
    assert (batch["imported"], batch["skipped"], batch["errors"]) == (0, 5, 2)
    older = request(member, "GET", f"{dataset}/imports", params={"offset": 1}).json()
    assert not older["has_more"] and len(older["items"]) == 1
    assert (older["items"][0]["imported"], older["items"][0]["errors"]) == (5, 2)

    stats = request(admin, "GET", "/datasets", params={"include_stats": "true"}).json()
    counts = next(item for item in stats if item["id"] == dataset_id)
    assert (counts["image_count"], counts["annotated_count"]) == (5, 2)
    assert request(admin, "GET", f"{dataset}/images").json()["images"] == sorted(names.values())
    for query, expected in [("舌诊", [image]), ("%", [image]), ("_", [image]), (".png", sorted(names.values())),
                            ("匹配不到", [])]:
        assert request(admin, "GET", f"{dataset}/images/search", params={"q": query}).json()["images"] == expected
    for source_name, filename in names.items():
        assert request(admin, "GET", f"{dataset}/images/{quote(filename, safe='')}").content == images[source_name]
    print(".1–.3 已验证 CRUD、计数、安全 ZIP、同名文件和字面量检索", flush=True)

    pulled = dataset_path(admin, prefix + "-按时段拉取")
    # 保留目标中的同名图片，拉取服务必须另存一个名称。
    assert import_archive(admin, pulled, archive({"同名.png": png("purple")}))["imported"] == 1
    pull_body = {"target_dataset_id": int(pulled.rsplit("/", 1)[-1]), "start_time": stamp, "end_time": stamp}
    request(admin, "POST", "/datasets/pull", 400, json={**pull_body, "quality": "未知"})
    request(admin, "POST", "/datasets/pull", 400,
            json={**pull_body, "end_time": (timestamp - timedelta(seconds=1)).isoformat()})
    assert request(admin, "POST", "/datasets/pull", json=pull_body).json()["pulled"] == 1
    assert request(admin, "POST", "/datasets/pull", json=pull_body).json()["pulled"] == 0
    qualified = exported_entries(export_archive(admin, pulled))
    assert sorted(qualified[name] for name in qualified if name.startswith("images/")) == sorted(
        [png("purple"), png("green")]
    )
    assert request(admin, "POST", "/datasets/pull", json={**pull_body, "quality": "全部"}).json()["pulled"] == 3
    assert request(admin, "POST", "/datasets/pull", json={**pull_body, "quality": "全部"}).json()["pulled"] == 0
    history = request(admin, "GET", f"{pulled}/imports").json()["items"]
    assert [batch["source"] for batch in history] == ["pull", "pull", "pull", "pull", "zip"]
    assert [batch["imported"] for batch in history] == [0, 3, 0, 1, 1]
    assert history[0]["start_time"] == history[0]["end_time"] == stamp.replace("+00:00", "Z")
    assert history[0]["quality"] == "全部" and history[0]["skipped"] == 4
    request(admin, "DELETE", pulled)
    request(admin, "GET", f"{pulled}/imports", 404)
    print(".2 已验证闭区间、质量过滤、重复拉取、同名保护及导入记录", flush=True)

    url = label_path(dataset, image)
    current = request(admin, "GET", url).json()
    assert current["file_size_bytes"] == len(images[f"现场/{image}"])
    assert current["record_timestamp"] == stamp.replace("+00:00", "Z")
    assert current["collector_alias"] == collector
    assert request(admin, "GET", label_path(dataset, "待标注.png")).json()["collector_alias"] is None
    assert current["annotation_source"] == "model"
    request(admin, "PUT", url, 428, json={"labels": [first]})
    request(admin, "PUT", url, 400, json={"labels": [first, second], "expected_revision": current["revision"]})
    quality_only = request(admin, "PUT", url, json={
        "labels": [first], "quality": "合格", "expected_revision": current["revision"],
    }).json()
    assert quality_only["annotation_source"] == "model" and quality_only["labels"] == current["labels"]
    assert quality_only["quality"] == "合格"
    assert quality_only["collector_alias"] == collector
    member_revision = request(member, "GET", url).json()["revision"]
    assert member_revision == quality_only["revision"]
    saved = request(admin, "PUT", url, json={
        "labels": [second], "quality": "合格", "expected_revision": member_revision,
    }).json()
    assert saved["annotation_source"] == "human"
    assert saved["labels"] == [{**second, "confidence": 1.0}]
    assert saved["collector_alias"] == collector
    conflict = request(member, "PUT", url, 409, json={
        "labels": [first], "quality": "", "expected_revision": member_revision,
    }).json()
    assert conflict["detail"]["current"]["revision"] == saved["revision"]
    assert request(member, "GET", url).json()["labels"] == saved["labels"]
    cleared = request(member, "PUT", url, json={
        "labels": [], "quality": "", "expected_revision": saved["revision"],
    }).json()
    assert cleared["annotation_state"] == "未标注" and cleared["annotation_source"] == "human"
    assert cleared["collector_alias"] == collector
    assert request(admin, "GET", url).json()["revision"] == cleared["revision"]
    history_url = url.removesuffix("/labels") + "/history"
    page = request(admin, "GET", history_url, params={"limit": 2}).json()
    assert page["has_more"] and [item["actor_id"] for item in page["items"]] == [actor_ids[1], actor_ids[0]]
    final_page = request(admin, "GET", history_url, params={"offset": 2}).json()
    assert not final_page["has_more"] and len(final_page["items"]) == 1
    assert final_page["items"][0]["after"]["labels"] == current["labels"]
    print(".4 已验证两账号写入、版本冲突、质量来源、持久回读和历史分页", flush=True)

    clone = dataset_path(admin, prefix + "-ZIP 往返")
    exported = export_archive(admin, dataset)
    original = exported_entries(exported)
    expected = {f"images/{filename}" for filename in names.values()} | {
        f"annotations/{PurePosixPath(filename).stem}.xml" for filename in names.values()
    }
    assert set(original) == expected
    assert all(ET.fromstring(original[name]).tag == "annotation" for name in expected if name.endswith(".xml"))
    assert ET.fromstring(original[f"annotations/{PurePosixPath(image).stem}.xml"]).findtext("collector_alias") == collector
    result = import_archive(admin, clone, exported)
    assert result["imported"] == len(images) and not result["errors"]
    assert request(admin, "GET", label_path(clone, image)).json()["collector_alias"] == collector
    assert exported_entries(export_archive(admin, clone)) == original
    print(".3–.5 已验证脱敏采集人代号校验、标注保留和配对 ZIP 往返字节一致", flush=True)
    request(admin, "DELETE", clone)
    request(admin, "DELETE", dataset)


def main():
    base_url = os.environ["BLACKBOX_BASE_URL"]
    username = os.environ["BLACKBOX_USERNAME"]
    password = os.environ["BLACKBOX_PASSWORD"]
    suffix = secrets.token_hex(4)
    prefix = f"blackbox-data-{suffix}"
    member_name = f"blackbox-member-{suffix}"
    member_password = secrets.token_urlsafe(24)
    options = {"base_url": base_url, "timeout": 30, "trust_env": False}
    with httpx.Client(**options) as admin, httpx.Client(**options) as member:
        request(admin, "GET", "/datasets", 401)
        admin_user = login(admin, username, password)
        assert admin_user["role"] == "admin"
        request(admin, "POST", "/auth/users", 201,
                json={"username": member_name, "password": member_password, "role": "member"})
        member_user = login(member, member_name, member_password)
        try:
            request(member, "GET", "/auth/users", 403)
            test_data_flow(admin, member, prefix, (admin_user["id"], member_user["id"]))
        finally:
            request(member, "POST", "/auth/logout")
            request(admin, "PATCH", f"/auth/users/{member_user['id']}", json={"active": False})
            request(admin, "POST", "/auth/logout")


if __name__ == "__main__":
    main()
