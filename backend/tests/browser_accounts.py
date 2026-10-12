"""通过真实页面创建四人团队、登录/退出；只运行于显式隔离测试环境。"""
import json
import os
from pathlib import Path
import uuid

from playwright.sync_api import sync_playwright, expect

base = os.environ["BROWSER_TEST_URL"]
output = Path(os.environ["BROWSER_TEST_OUTPUT"])
output.mkdir(parents=True, exist_ok=True)
(output / "accounts-result.json").write_text('{"status":"running"}', encoding="utf-8")

with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    context = browser.new_context(viewport={"width": 1440, "height": 1000})
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(base + "/datasets")
    page.wait_for_load_state("networkidle")
    print(page.locator("body").inner_text())
    page.screenshot(path=str(output / "login.png"))
    expect(page.get_by_role("heading", name="登录平台")).to_be_visible()
    page.get_by_role("textbox", name="账号", exact=True).fill(os.environ["BROWSER_TEST_USERNAME"])
    page.get_by_role("textbox", name="密码", exact=True).fill(os.environ["BROWSER_TEST_PASSWORD"])
    page.get_by_role("button", name="登录", exact=True).click()
    expect(page).to_have_url(base + "/datasets")
    page.get_by_role("link", name="账号", exact=True).click()
    expect(page.get_by_role("heading", name="成员管理")).to_be_visible()
    page.wait_for_load_state("networkidle")
    print(page.locator("body").inner_text())
    names = [f"member-{uuid.uuid4().hex[:8]}" for _ in range(3)]
    for name in names:
        page.get_by_role("textbox", name="新账号", exact=True).fill(name)
        page.get_by_role("textbox", name="初始密码", exact=True).fill("member-test-password")
        page.get_by_role("button", name="创建账号", exact=True).click()
        expect(page.get_by_role("cell", name=name, exact=True)).to_be_visible()
    expect(page.get_by_role("button", name="创建账号", exact=True)).to_be_enabled()
    page.locator(".el-loading-mask").wait_for(state="hidden")
    page.screenshot(path=str(output / "accounts.png"), full_page=True)
    page.get_by_role("button", name="退出登录", exact=True).click()
    expect(page.get_by_role("heading", name="登录平台")).to_be_visible()
    page.get_by_role("textbox", name="账号", exact=True).fill(names[0])
    page.get_by_role("textbox", name="密码", exact=True).fill("member-test-password")
    page.get_by_role("button", name="登录", exact=True).click()
    expect(page).to_have_url(base + "/datasets")
    page.get_by_role("link", name="账号", exact=True).click()
    expect(page.get_by_text(names[0] + " · 成员", exact=True)).to_be_visible()
    expect(page.get_by_role("heading", name="成员管理")).to_have_count(0)
    assert context.request.get(base + "/api/auth/users").status == 403
    page.get_by_role("button", name="退出登录", exact=True).click()
    expect(page.get_by_role("heading", name="登录平台")).to_be_visible()
    assert context.request.get(base + "/api/datasets").status == 401
    assert not errors, errors
    (output / "accounts-result.json").write_text(json.dumps({"status": "passed", "members_created": 3,
        "checks": ["login redirect", "admin creates members", "logout", "member login", "member authority", "anonymous blocked"],
        "page_errors": errors}, ensure_ascii=False, indent=2), encoding="utf-8")
    browser.close()
