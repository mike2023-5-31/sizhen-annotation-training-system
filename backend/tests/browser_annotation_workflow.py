"""通过隔离服务器运行；不读取或修改用户数据。见docs/实施进度-v2.md。"""
import io
import json
import os
from pathlib import Path
import sys
import uuid
import zipfile

from PIL import Image
from playwright.sync_api import sync_playwright, expect

BASE = os.environ.get("BROWSER_TEST_URL", "http://127.0.0.1:15178")
OUTPUT = Path(os.environ["BROWSER_TEST_OUTPUT"])
OUTPUT.mkdir(parents=True, exist_ok=True)
(OUTPUT / "result.json").write_text('{"status":"running"}', encoding="utf-8")


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1000})
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        login = context.request.post(f"{BASE}/api/auth/login", data={
            "username": os.environ["BROWSER_TEST_USERNAME"], "password": os.environ["BROWSER_TEST_PASSWORD"],
        }, headers={"X-Requested-With": "XMLHttpRequest"})
        assert login.ok, login.text()
        csrf_headers = {"X-CSRF-Token": login.json()["csrf_token"]}
        response = context.request.post(f"{BASE}/api/datasets", data={"name": f"browser-{uuid.uuid4().hex[:8]}"}, headers=csrf_headers)
        assert response.status == 201, response.text()
        dataset_id = response.json()["id"]
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            for name, color in [("a.png", (190, 100, 90)), ("b.png", (100, 170, 130))]:
                image = io.BytesIO()
                Image.new("RGB", (320, 240), color).save(image, format="PNG")
                archive.writestr(name, image.getvalue())
        response = context.request.post(f"{BASE}/api/datasets/{dataset_id}/import", headers=csrf_headers, multipart={
            "file": {"name": "images.zip", "mimeType": "application/zip", "buffer": payload.getvalue()},
        })
        assert response.ok and response.json()["initialized"] == 2, response.text()
        page.goto(f"{BASE}/annotate?dataset={dataset_id}")
        page.wait_for_load_state("networkidle")
        expect(page.get_by_role("button", name="保存 / 重试")).to_be_enabled()
        expect(page.get_by_text("a.png", exact=True)).to_be_visible()
        page.screenshot(path=str(OUTPUT / "annotation-initial.png"), full_page=True)
        (OUTPUT / "initial-dom.html").write_text(page.content(), encoding="utf-8")
        print(page.locator("body").inner_text()[:2200])
        if "--inspect" in sys.argv:
            browser.close()
            return

        # selectors are checked against the initial rendered page before interaction.
        page.get_by_text("舌苔黄", exact=True).click()
        page.get_by_role("button", name="下一张", exact=True).click()
        expect(page.get_by_text("2 / 2", exact=True)).to_be_visible()
        first_url = f"{BASE}/api/datasets/{dataset_id}/image/a.png/labels"
        saved = context.request.get(first_url).json()
        assert saved["labels"][0]["name"] == "舌苔黄", saved
        second_url = f"{BASE}/api/datasets/{dataset_id}/image/b.png/labels"
        assert context.request.get(second_url).json()["labels"] == []
        page.get_by_role("button", name="上一张", exact=True).click()
        expect(page.get_by_text("1 / 2", exact=True)).to_be_visible()
        expect(page.get_by_role("button", name="保存 / 重试")).to_be_enabled()

        # Another editor writes the same revision; the current page then becomes stale.
        latest = context.request.get(first_url).json()
        other = context.request.put(first_url, headers=csrf_headers, data={"labels": [{"category": "舌", "name": "舌质红"}],
                                                     "quality": "合格", "expected_revision": latest["revision"]})
        assert other.ok, other.text()
        page.get_by_text("舌苔薄白", exact=True).click()
        expect(page.get_by_role("button", name="核对冲突", exact=True)).to_be_visible()
        assert context.request.get(first_url).json()["labels"][0]["name"] == "舌质红"
        page.get_by_role("button", name="核对冲突", exact=True).click()
        expect(page.get_by_text("核对标注冲突", exact=True)).to_be_visible()
        page.screenshot(path=str(OUTPUT / "annotation-conflict.png"), full_page=True)
        page.get_by_role("button", name="采用我的草稿", exact=True).click()
        expect(page.get_by_text("已保存", exact=True)).to_be_visible()
        assert context.request.get(first_url).json()["labels"][0]["name"] == "舌苔薄白"

        # Abort PUT only: GET still works after reload and restores the pending draft.
        def abort_put(route):
            if route.request.method == "PUT":
                route.abort("failed")
            else:
                route.continue_()
        page.route("**/labels", abort_put)
        page.get_by_text("舌苔黄腻", exact=True).click()
        expect(page.get_by_text("保存失败", exact=True)).to_be_visible()
        page.on("dialog", lambda dialog: dialog.accept())
        page.reload()
        page.wait_for_load_state("networkidle")
        expect(page.get_by_text("舌：舌苔黄腻", exact=True)).to_be_visible()
        page.unroute("**/labels", abort_put)
        page.get_by_role("button", name="保存 / 重试", exact=True).click()
        expect(page.get_by_text("已保存", exact=True)).to_be_visible()
        assert context.request.get(first_url).json()["labels"][0]["name"] == "舌苔黄腻"
        page.get_by_role("button", name="修改记录", exact=True).click()
        expect(page.get_by_role("heading", name="标注修改记录", exact=True)).to_be_visible()
        expect(page.get_by_role("cell", name=os.environ["BROWSER_TEST_USERNAME"], exact=True).first).to_be_visible()
        history = context.request.get(first_url.replace("/labels", "/history")).json()
        assert history["items"][0]["username"] == os.environ["BROWSER_TEST_USERNAME"]
        page.screenshot(path=str(OUTPUT / "annotation-history.png"), full_page=True)
        page.get_by_role("button", name="Close this dialog").click()
        assert not errors, errors
        page.screenshot(path=str(OUTPUT / "annotation-restored.png"), full_page=True)
        (OUTPUT / "result.json").write_text(json.dumps({
            "status": "passed", "browser": browser.version,
            "checks": ["unlabelled import", "save before navigation", "revision conflict", "explicit rebase", "offline draft reload", "authenticated edit history"],
            "page_errors": errors,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        browser.close()


if __name__ == "__main__":
    run()
