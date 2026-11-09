"""Run Playwright against a disposable, same-origin deployment started by the runner."""
from __future__ import annotations

import io
import json
import os
import re
import uuid
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

from PIL import Image, ImageDraw
from playwright.sync_api import Browser, Page, Response, expect, sync_playwright


FIRST_IMAGE = "00_舌%_样例.png"
TRAIN_IMAGES = [FIRST_IMAGE, *(f"{index:02d}_样例.png" for index in range(1, 9))]
PREDICT_IMAGE = "待预标注.png"
COLLECTOR_ALIAS = "COL-A012BCDE"
HEADS = ("舌", "面", "脉", "声音")
ALTERNATE_LABELS = ("舌苔薄白", "面色红润", "浮", "清晰")
LOGIN_HEADING = "登录四诊工作台"
TINY_MODEL = '''
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


def image_zip(filenames: list[str], *, labeled: bool = False) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        for index, filename in enumerate(filenames):
            image = Image.new("RGB", (320, 240), (180 - 10 * index, 210, 236))
            ImageDraw.Draw(image).ellipse((75, 40, 245, 200), fill=(90, 110 + 10 * index, 155))
            png = io.BytesIO()
            image.save(png, "PNG")
            archive.writestr(filename, png.getvalue())
            if labeled and index:
                annotation = ET.Element("annotation")
                ET.SubElement(annotation, "filename").text = filename
                ET.SubElement(annotation, "quality").text = "合格"
                ET.SubElement(annotation, "collector_alias").text = COLLECTOR_ALIAS
                labels = ET.SubElement(annotation, "labels")
                for category, alternate in zip(HEADS, ALTERNATE_LABELS):
                    label = ET.SubElement(labels, "label")
                    ET.SubElement(label, "category").text = category
                    ET.SubElement(label, "name").text = alternate if index % 2 else "正常/无"
                archive.writestr(Path(filename).with_suffix(".xml").name, ET.tostring(annotation, encoding="utf-8"))
    return payload.getvalue()


def api_response(path: str, method: str = "POST"):
    def matches(response: Response) -> bool:
        return unquote(urlparse(response.url).path) == path and response.request.method == method

    return matches


def response_json(response: Response) -> dict:
    assert response.ok, f"{response.status} {response.url}: {response.text()}"
    return response.json()


def button(container, name: str):
    return container.get_by_role("button", name=name, exact=True)


def navigate(page: Page, name: str) -> None:
    page.get_by_role("navigation", name="主导航").get_by_role("link", name=name, exact=True).click()
    expect(page.get_by_role("heading", level=1, name=name)).to_be_visible()
    expect(page.get_by_role("navigation", name="当前位置").locator('[aria-current="page"]')).to_have_text(name)


def select(page: Page, placeholder: str, value: str) -> None:
    selection = page.locator(".el-select").filter(has=page.get_by_text(placeholder, exact=True))
    selection.locator(".el-select__wrapper").click()
    page.get_by_role("option", name=value, exact=True).click()


def login(page: Page, username: str, password: str) -> None:
    expect(page.get_by_role("heading", name=LOGIN_HEADING)).to_be_visible()
    page.get_by_label("账号", exact=True).fill(username)
    page.get_by_label("密码", exact=True).fill(password)
    with page.expect_response(api_response("/api/auth/login")) as login_result:
        button(page, "登录").click()
    assert login_result.value.ok, login_result.value.status
    expect(page.get_by_role("heading", name="数据集管理", exact=True)).to_be_visible()


def create_dataset(page: Page, name: str) -> int:
    button(page, "新建数据集").first.click()
    dialog = page.get_by_role("dialog", name="新建数据集")
    dialog.get_by_placeholder("请输入数据集名称").fill(name)
    dialog.get_by_text("未填写", exact=True).click()
    page.get_by_role("option", name="舌", exact=True).click()
    with page.expect_response(api_response("/api/datasets")) as result:
        button(dialog, "确定").click()
    dataset = response_json(result.value)
    assert dataset["name"] == name and dataset["modality"] == "舌", dataset
    expect(page.get_by_role("link", name=f"标注数据集：{name}")).to_be_visible()
    return dataset["id"]


def dataset_row(page: Page, name: str):
    return page.get_by_role("row").filter(has=page.get_by_role("link", name=f"标注数据集：{name}"))


def import_dataset(page: Page, dataset_id: int, name: str, archive: bytes, count: int) -> None:
    with page.expect_response(api_response(f"/api/datasets/{dataset_id}/import")) as result:
        with page.expect_file_chooser() as chooser:
            button(dataset_row(page, name), "导入 ZIP").click()
        chooser.value.set_files({"name": "browser-fixture.zip", "mimeType": "application/zip", "buffer": archive})
    imported = response_json(result.value)
    assert imported["imported"] == count and imported["errors"] == [], imported
    dialog = page.get_by_role("dialog", name="ZIP 导入结果")
    expect(dialog.get_by_test_id("import-imported")).to_contain_text(f"{count} 条")
    button(dialog, "确定").click()
    expect(dialog).to_be_hidden()
    button(dataset_row(page, name), "导入记录").click()
    history = page.get_by_role("dialog", name=f"导入记录 · {name}")
    expect(history.get_by_role("cell", name="browser-fixture.zip", exact=True)).to_be_visible()
    expect(history.get_by_role("cell", name=f"成功 {count} · 跳过 0 · 错误 0", exact=True)).to_be_visible()
    button(history, "关闭").click()
    expect(history).to_be_hidden()


def verify_image(page: Page, filename: str) -> None:
    expect(page.get_by_role("img", name=filename, exact=True)).to_be_visible()
    page.wait_for_function(
        "(filename) => { const img = document.querySelector('.main-image');"
        " return img?.alt === filename && img.complete && img.naturalWidth === 320; }",
        arg=filename,
    )
    expect(button(page, "保存 / 重试")).to_be_enabled()


def annotate(page: Page, dataset_id: int, name: str) -> None:
    page.get_by_role("link", name=f"标注数据集：{name}").click()
    verify_image(page, FIRST_IMAGE)
    metadata = page.locator(".image-meta > div")
    for field, value in (("记录时间", r"\d"), ("文件大小", r"\d+\.\d KB"), ("采集人（脱敏）", r"^未知$")):
        entry = metadata.filter(has=page.get_by_text(field, exact=True))
        expect(entry.locator("dd")).to_have_text(re.compile(value))

    search = page.get_by_placeholder("按文件名搜索")
    search.fill("没有对应图片")
    search.press("Enter")
    expect(page.get_by_text("无匹配图片", exact=True)).to_be_visible()
    search.fill("舌%_")
    search.press("Enter")
    expect(page.get_by_text("1 / 1", exact=True)).to_be_visible()
    verify_image(page, FIRST_IMAGE)
    button(page, "重置").click()
    expect(page.get_by_text(f"1 / {len(TRAIN_IMAGES)}", exact=True)).to_be_visible()
    verify_image(page, FIRST_IMAGE)

    slider = page.get_by_role("slider")
    slider.press("End")
    verify_image(page, TRAIN_IMAGES[-1])
    expect(metadata.filter(has=page.get_by_text("采集人（脱敏）", exact=True)).locator("dd")).to_have_text(COLLECTOR_ALIAS)
    slider.press("ArrowLeft")
    verify_image(page, TRAIN_IMAGES[-2])
    slider.press("Home")
    verify_image(page, FIRST_IMAGE)
    for category in HEADS:
        expect(page.get_by_role("treeitem", name=category, exact=True)).to_be_visible()

    page.get_by_text("舌苔黄", exact=True).click()
    expect(page.get_by_text("舌：舌苔黄", exact=True)).to_be_visible()
    label_api = f"/api/datasets/{dataset_id}/image/{FIRST_IMAGE}/labels"
    with page.expect_response(lambda response: (
        api_response(label_api, "PUT")(response)
        and response.request.post_data_json["quality"] == "合格"
    )) as saved:
        page.keyboard.press("q")
    label_result = response_json(saved.value)
    assert [(label["category"], label["name"]) for label in label_result["labels"]] == [("舌", "舌苔黄")], (
        label_api, saved.value.request.post_data_json, label_result
    )
    expect(page.get_by_text("已保存", exact=True)).to_be_visible()
    page.keyboard.press("ArrowRight")
    verify_image(page, TRAIN_IMAGES[1])
    with page.expect_response(api_response(label_api, "GET")) as loaded:
        page.keyboard.press("ArrowLeft")
    assert response_json(loaded.value)["labels"] == label_result["labels"], loaded.value.text()
    verify_image(page, FIRST_IMAGE)
    expect(page.get_by_text("舌：舌苔黄", exact=True)).to_be_visible()

    button(page, "修改记录").click()
    history = page.get_by_role("dialog", name="标注修改记录")
    expect(history.get_by_role("cell", name=os.environ["BLACKBOX_USERNAME"], exact=True).first).to_be_visible()
    expect(history.get_by_role("cell", name="舌：舌苔黄；合格", exact=True).first).to_be_visible()
    button(history, "关闭此对话框").click()
    expect(history).to_be_hidden()
    expect(page).to_have_url(re.compile(rf"/annotate\?dataset={dataset_id}$"))


def verify_export(page: Page, dataset_name: str) -> None:
    navigate(page, "数据集管理")
    with page.expect_download() as downloaded:
        button(dataset_row(page, dataset_name), "导出").click()
    download = downloaded.value
    assert download.suggested_filename.endswith(".zip")
    with zipfile.ZipFile(download.path()) as archive:
        pngs = {Path(name).name for name in archive.namelist() if name.endswith(".png")}
        assert pngs == set(TRAIN_IMAGES), pngs
        xml_name = Path(FIRST_IMAGE).with_suffix(".xml").name
        entry = next(name for name in archive.namelist() if Path(name).name == xml_name)
        root = ET.fromstring(archive.read(entry))
        assert root.findtext("filename") == FIRST_IMAGE
        assert root.findtext("quality") == "合格"
        assert root.findtext("labels/label/name") == "舌苔黄"


def verify_editor_conflict(browser: Browser, page: Page, base: str, dataset_name: str, capture) -> None:
    navigate(page, "账号")
    member = f"editor-{uuid.uuid4().hex[:8]}"
    password = uuid.uuid4().hex
    page.get_by_label("新账号", exact=True).fill(member)
    page.get_by_label("初始密码", exact=True).fill(password)
    button(page, "创建账号").click()
    expect(page.get_by_role("cell", name=member, exact=True)).to_be_visible()
    capture("account")
    navigate(page, "数据集管理")
    link_name = f"标注数据集：{dataset_name}"
    page.get_by_role("link", name=link_name).click()
    verify_image(page, FIRST_IMAGE)

    with browser.new_context() as context:
        other = context.new_page()
        other.goto(f"{base}/datasets")
        login(other, member, password)
        navigate(other, "账号")
        expect(other.get_by_role("heading", name="成员管理")).to_have_count(0)
        navigate(other, "数据集管理")
        other.get_by_role("link", name=link_name).click()
        verify_image(other, FIRST_IMAGE)
        other.get_by_text("舌质红", exact=True).click()
        expect(other.get_by_text("已保存", exact=True)).to_be_visible()

    page.get_by_text("舌苔薄白", exact=True).click()
    button(page, "核对冲突").click()
    conflict = page.get_by_role("dialog", name="核对标注冲突")
    expect(conflict).to_contain_text("舌质红")
    expect(conflict).to_contain_text("舌苔薄白")
    capture("conflict")
    button(conflict, "采用我的草稿").click()
    expect(page.get_by_text("已保存", exact=True)).to_be_visible()
    button(page, "修改记录").click()
    history = page.get_by_role("dialog", name="标注修改记录")
    expect(history.get_by_role("cell", name=member, exact=True)).to_be_visible()
    button(history, "关闭此对话框").click()
    expect(history).to_be_hidden()


def train_and_evaluate(page: Page, dataset_name: str, dataset_id: int, capture) -> str:
    frames = []

    def track_websocket(socket):
        if urlparse(socket.url).path == "/api/training/ws/training":
            socket.on("framereceived", lambda frame: frames.append(json.loads(frame)))

    page.on("websocket", track_websocket)
    navigate(page, "模型训练")
    expect(page.get_by_text("实时推送（WebSocket）", exact=True)).to_be_visible()
    select(page, "主训练集（必选）", dataset_name)
    page.get_by_text("自定义上传", exact=True).click()
    with page.expect_response(api_response("/api/models/custom")) as uploaded:
        page.get_by_test_id("custom-model-file-input").set_input_files({
            "name": "blackbox_model.py", "mimeType": "text/x-python", "buffer": TINY_MODEL.encode(),
        })
    model_path = response_json(uploaded.value)["model_path"]
    assert model_path.startswith("custom/") and model_path.endswith("/blackbox_model.py"), model_path
    expect(page.locator(".custom-row .el-tag")).to_contain_text(model_path)
    epochs = page.get_by_role("spinbutton", name="训练轮数 epochs")
    epochs.fill("2")
    epochs.press("Tab")
    page.get_by_text("随机水平翻转", exact=True).click()
    with page.expect_response(api_response("/api/training/start")) as started:
        button(page, "启动训练").click()
    job_id = response_json(started.value)["job_id"]
    request = started.value.request.post_data_json
    assert request["epochs"] == 2 and request["datasets"] == [dataset_id] and request["backbone"] == "custom", request
    expect(page.locator('[data-testid="job-meta"]')).to_contain_text("第 2 / 2 轮", timeout=120_000)
    expect(page.locator('[data-testid="job-meta"]')).to_contain_text("已完成", timeout=30_000)
    expect(page.get_by_test_id("loss-chart").locator("canvas").first).to_be_visible()
    expect(page.get_by_test_id("acc-chart").locator("canvas").first).to_be_visible()
    assert any(frame["type"] == "snapshot" for frame in frames), frames
    assert any(frame["type"] == "metrics" and frame["job_id"] == job_id for frame in frames), frames
    page.get_by_test_id("loss-chart").scroll_into_view_if_needed()
    capture("training")

    with page.expect_response(api_response(f"/api/training/eval/{job_id}", "GET"), timeout=120_000) as evaluated:
        button(page, "前往评估 →").click()
    results = response_json(evaluated.value)
    assert "舌" in results["confusion"], results
    expect(page.get_by_role("heading", name="验证集指标")).to_be_visible()
    expect(page.get_by_test_id("heatmap-舌").locator("canvas").first).to_be_visible()
    capture("evaluation")
    return job_id


def predict(page: Page, model_name: str, dataset_name: str, dataset_id: int, capture) -> None:
    navigate(page, "批量推理")
    select(page, "选择模型", model_name)
    select(page, "选择数据集", dataset_name)
    with page.expect_response(api_response("/api/inference/jobs")) as started:
        button(page, "批量预测").click()
    assert started.value.status == 202, started.value.status
    job_id = response_json(started.value)["job_id"]
    progress = page.get_by_role("progressbar", name="批量推理进度")
    expect(progress).to_have_attribute("aria-valuenow", "100", timeout=120_000)
    expect(page.get_by_text("已处理 1 / 1 张", exact=True)).to_be_visible()
    assert started.value.request.post_data_json == {"model_path": model_name, "dataset_id": dataset_id}
    result_row = page.get_by_role("row").filter(has_text=PREDICT_IMAGE)
    expect(result_row.get_by_role("cell")).to_have_count(5)
    expect(result_row.locator(".cat-pct")).to_have_count(4)
    status = page.request.get(f"/api/inference/jobs/{job_id}")
    assert status.ok, status.text()
    assert [row["filename"] for row in status.json()["predictions"]] == [PREDICT_IMAGE]

    threshold = page.get_by_role("spinbutton", name="预标注最低置信度（百分比）")
    threshold.fill("0")
    threshold.press("Tab")
    with page.expect_response(api_response("/api/inference/preannotation"), timeout=120_000) as generated:
        button(page, "生成预标注 XML").click()
    assert generated.value.request.post_data_json["confidence_threshold"] == 0
    assert response_json(generated.value)["generated"] == 1
    expect(page.get_by_text("预标注已生成 1 个 XML", exact=True)).to_be_visible()
    capture("inference")
    navigate(page, "数据集管理")
    page.get_by_role("link", name=f"标注数据集：{dataset_name}").click()
    verify_image(page, PREDICT_IMAGE)
    expect(page.locator(".sel-tag")).to_have_count(4)
    expect(page).to_have_url(re.compile(rf"/annotate\?dataset={dataset_id}$"))


def convert_and_download(page: Page, fixture: dict, capture) -> None:
    import onnx

    navigate(page, "模型转换")
    select(page, "选择模型", fixture["model_name"])
    select(page, "验证数据集（可选）", fixture["dataset_name"])
    with page.expect_response(api_response("/api/onnx/compare"), timeout=120_000) as compared:
        button(page, "一键转换并对比").click()
    result = response_json(compared.value)
    assert result["overall_pass"] and len(result["results"]) == 4, result
    assert result["sample_type"] == "dataset"
    expected_files = sorted(fixture["image_filenames"])[:8]
    assert result["sample_filenames"] == expected_files, result
    page.get_by_text("查看参与对比的图片", exact=True).click()
    expect(page.locator(".compare-samples li")).to_have_text(expected_files)
    expect(page.get_by_role("cell", name="通过", exact=True)).to_have_count(4)
    capture("onnx")
    with page.expect_download() as downloaded:
        button(page, "下载 .onnx").click()
    download = downloaded.value
    assert download.suggested_filename.endswith(".onnx")
    model = onnx.load(download.path())
    onnx.checker.check_model(model)
    assert len(model.graph.output) == 4


def mobile(browser: Browser, page: Page, base: str, name: str, capture, errors: list[str]) -> None:
    with browser.new_context(
        viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True,
        storage_state=page.context.storage_state(),
    ) as context:
        narrow = context.new_page()
        narrow.on("pageerror", lambda error: errors.append(str(error)))
        narrow.goto(f"{base}/datasets")
        expect(narrow.get_by_role("heading", name="数据集管理")).to_be_visible()
        toggle = button(narrow, "展开导航")
        expect(toggle).to_have_attribute("aria-expanded", "false")
        toggle.click()
        expect(narrow.get_by_role("navigation", name="主导航")).to_be_visible()
        first_link = narrow.get_by_role("link", name="数据集管理", exact=True)
        first_link.focus()
        first_link.press("Escape")
        expect(toggle).to_be_focused()
        expect(toggle).to_have_attribute("aria-expanded", "false")

        narrow.get_by_role("link", name=f"标注数据集：{name}").click()
        verify_image(narrow, FIRST_IMAGE)
        assert narrow.evaluate("document.documentElement.scrollWidth === window.innerWidth")
        assert narrow.evaluate(
            "const bounds = document.querySelector('.app-main'); bounds.scrollWidth <= bounds.clientWidth + 1"
        )
        capture("mobile", narrow)


def run() -> None:
    base = os.environ["BLACKBOX_BASE_URL"].rstrip("/")
    user = os.environ["BLACKBOX_USERNAME"]
    password = os.environ["BLACKBOX_PASSWORD"]
    fixture_path = os.environ.get("BLACKBOX_FIXTURE_PATH")
    fixture = json.loads(Path(fixture_path).read_text(encoding="utf-8")) if fixture_path else None
    screenshot_dir = os.environ.get("BLACKBOX_SCREENSHOT_DIR")
    suffix = uuid.uuid4().hex[:8]
    train_name, predict_name = f"浏览器训练-{suffix}", f"浏览器预标注-{suffix}"
    errors: list[str] = []

    with sync_playwright() as playwright:
        chromium_installed = Path(playwright.chromium.executable_path).is_file()
        launch_options = {"channel": "msedge"} if os.name == "nt" and not chromium_installed else {}
        browser = playwright.chromium.launch(**launch_options)
        try:
            page = browser.new_page(base_url=base, viewport={"width": 1440, "height": 1000})
            page.set_default_timeout(10_000)
            page.on("pageerror", lambda error: errors.append(str(error)))

            def capture(label: str, target: Page = page) -> None:
                if screenshot_dir:
                    output = Path(screenshot_dir)
                    output.mkdir(parents=True, exist_ok=True)
                    target.screenshot(path=output / f"sizhen-{label}-blackbox.png", full_page=True)

            page.goto(f"{base}/datasets")
            page.wait_for_load_state("networkidle")
            capture("login")
            login(page, user, password)
            train_id = create_dataset(page, train_name)
            import_dataset(page, train_id, train_name, image_zip(TRAIN_IMAGES, labeled=True), len(TRAIN_IMAGES))
            predict_id = create_dataset(page, predict_name)
            import_dataset(page, predict_id, predict_name, image_zip([PREDICT_IMAGE]), 1)
            capture("datasets")
            annotate(page, train_id, train_name)
            page.get_by_role("main").evaluate("element => element.scrollTop = 0")
            capture("annotate")
            verify_export(page, train_name)
            print("PASS browser login, dataset creation/import, image bytes/metadata, search, slider, shortcuts, autosave/history, ZIP export")
            verify_editor_conflict(browser, page, base, train_name, capture)
            print("PASS browser member access and annotation conflict resolution across two accounts")

            job_id = train_and_evaluate(page, train_name, train_id, capture)
            print(f"PASS browser custom training, WebSocket snapshot and metrics, evaluation ({job_id})")
            predict(page, f"{job_id}_best.pth", predict_name, predict_id, capture)
            print("PASS browser batch inference and preannotation")
            if fixture is not None:
                convert_and_download(page, fixture, capture)
                print("PASS browser sampled ONNX conversion, comparison, and download")
            else:
                print("SKIP browser ONNX checks: BLACKBOX_FIXTURE_PATH was not provided")

            mobile(browser, page, base, train_name, capture, errors)
            navigate(page, "工作台总览")
            expect(page.get_by_text("最近导入", exact=True)).to_be_visible()
            expect(page.get_by_text("最近任务", exact=True)).to_be_visible()
            expect(page.get_by_text(train_name, exact=False).first).to_be_visible()
            capture("overview")
            navigate(page, "作业中心")
            expect(page.get_by_text("训练任务", exact=True).first).to_be_visible()
            expect(page.get_by_text("批量推理任务", exact=True)).to_be_visible()
            expect(page.get_by_text(job_id).first).to_be_visible()
            expect(page.locator(".el-tag").first).to_be_visible()
            capture("jobs")
            navigate(page, "模型与产物")
            expect(page.get_by_text("模型存档", exact=True)).to_be_visible()
            expect(page.get_by_text("ONNX 产物", exact=True)).to_be_visible()
            expect(page.get_by_text(job_id, exact=False).first).to_be_visible()
            capture("artifacts")
            navigate(page, "数据集管理")
            page.get_by_role("tab", name="导入记录").click()
            expect(page.get_by_text("ZIP 导入", exact=True).first).to_be_visible()
            capture("imports")
            print("PASS browser overview, job center, artifact repository and global import history")
            navigate(page, "账号")
            expect(page.get_by_role("heading", name="成员管理")).to_be_visible()
            button(page, "退出登录").click()
            expect(page.get_by_role("heading", name=LOGIN_HEADING)).to_be_visible()
            page.goto(f"{base}/datasets")
            expect(page.get_by_role("heading", name=LOGIN_HEADING)).to_be_visible()
            assert not errors, errors
            print("PASS browser mobile navigation/layout and logout protection; no uncaught page errors")
        finally:
            browser.close()


if __name__ == "__main__":
    run()
