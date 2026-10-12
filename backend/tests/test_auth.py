"""生产 ASGI 登录边界：接口/静态图/下载/WS，撤销和角色。"""
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from starlette.websockets import WebSocketDisconnect

from app import config
from app.db import Base
from app.main import app
from app.models.orm import User, LoginSession
from app.services import auth_service as service


@pytest.fixture
def auth_env(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'auth.sqlite'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(service, "SessionLocal", factory)
    monkeypatch.setattr(config, "COOKIE_SECURE", False)
    admin = service.create_user("admin", "a-long-admin-password", "admin")
    member = service.create_user("member", "a-long-member-password")
    yield factory, admin, member
    engine.dispose()


def sign_in(client, username="admin", password="a-long-admin-password"):
    result = client.post("/api/auth/login", json={"username": username, "password": password},
                         headers={"X-Requested-With": "XMLHttpRequest"})
    assert result.status_code == 200, result.text
    client.headers["X-CSRF-Token"] = result.json()["csrf_token"]
    return result


def test_anonymous_cannot_read_any_data_or_stream(auth_env):
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        for url in ["/api/datasets", "/static/1/a.png", "/api/onnx/a.onnx/download", "/docs", "/api/training/status"]:
            assert client.get(url).status_code == 401
        with pytest.raises(WebSocketDisconnect) as error:
            with client.websocket_connect("/api/training/ws/training"):
                pass
        assert error.value.code == 4401


def test_login_cookie_csrf_logout_and_account_authority(auth_env):
    with TestClient(app) as client:
        response = sign_in(client)
        cookie = response.headers["set-cookie"].lower()
        assert "httponly" in cookie and "samesite=strict" in cookie
        assert response.headers["cache-control"] == "no-store"
        assert client.get("/api/auth/me").json()["user"]["username"] == "admin"
        assert len(client.get("/api/auth/users").json()) == 2
        assert client.post("/api/auth/logout", headers={"X-CSRF-Token": "wrong"}).status_code == 403
        assert client.post("/api/auth/logout").status_code == 200
        assert client.get("/api/auth/me").status_code == 401
        sign_in(client, "member", "a-long-member-password")
        assert client.get("/api/auth/users").status_code == 403
        assert client.post("/api/auth/users", json={"username": "bad", "password": "a-long-password"}).status_code == 403
        assert client.post("/api/models/custom").status_code == 403


def test_reset_disable_expiry_revoke_sessions(auth_env):
    factory, admin, member = auth_env
    with TestClient(app) as manager, TestClient(app) as client:
        sign_in(manager)
        sign_in(client, "member", "a-long-member-password")
        assert manager.patch(f"/api/auth/users/{member['id']}", json={"password": "replacement-password"}).status_code == 200
        assert client.get("/api/auth/me").status_code == 401
        sign_in(client, "member", "replacement-password")
        assert manager.patch(f"/api/auth/users/{member['id']}", json={"active": False}).status_code == 200
        assert client.get("/api/auth/me").status_code == 401
        assert manager.patch(f"/api/auth/users/{admin['id']}", json={"active": False}).status_code == 400
        with factory() as db:
            for session in db.scalars(select(LoginSession)):
                session.expires_at = service.now() - timedelta(seconds=1)
            db.commit()
        assert manager.get("/api/auth/me").status_code == 401


def test_password_hash_unique_lockout_and_origin(auth_env):
    factory, _, _ = auth_env
    assert service.hash_password("identical-password") != service.hash_password("identical-password")
    with factory() as db:
        stored = db.scalar(select(User).where(User.username == "admin")).password_hash
        assert "a-long-admin-password" not in stored
        assert service.verify_password("a-long-admin-password", stored)
    with TestClient(app) as client:
        data = {"username": "admin", "password": "a-long-admin-password"}
        assert client.post("/api/auth/login", json=data).status_code == 403
        assert client.post("/api/auth/login", json=data, headers={"X-Requested-With": "XMLHttpRequest", "Origin": "https://evil.example"}).status_code == 403
        for _ in range(5):
            assert client.post("/api/auth/login", json={**data, "password": "wrong"}, headers={"X-Requested-With": "XMLHttpRequest"}).status_code == 401
        assert client.post("/api/auth/login", json=data, headers={"X-Requested-With": "XMLHttpRequest"}).status_code == 401


def test_cross_tab_account_switch_blocks_old_draft(auth_env):
    _, admin, _ = auth_env
    with TestClient(app) as client:
        sign_in(client, "member", "a-long-member-password")
        response = client.put("/api/datasets/1/image/a.png/labels", json={}, headers={"X-Annotation-Owner": admin["id"]})
        assert response.status_code == 409


def test_websocket_origin_and_revocation(auth_env, monkeypatch):
    with TestClient(app) as client:
        sign_in(client)
        with pytest.raises(WebSocketDisconnect) as error:
            with client.websocket_connect("/api/training/ws/training", headers={"Origin": "https://evil.example"}):
                pass
        assert error.value.code == 4403
        from app.services import train_service
        manager = train_service.train_manager
        monkeypatch.setattr(manager, "get_metrics", lambda: {"jobs": []})
        monkeypatch.setattr(manager, "list_jobs", lambda: [])
        with client.websocket_connect("/api/training/ws/training") as ws:
            assert ws.receive_json()["type"] == "snapshot"
            client.post("/api/auth/logout")
            monkeypatch.setattr(manager, "list_jobs", lambda: [{"job_id": "revoked", "metrics": [], "status": "done", "error": None, "finished_at": None}])
            manager.metrics_event.set()
            with pytest.raises(WebSocketDisconnect) as revoked:
                ws.receive_json()
            assert revoked.value.code == 4401


def test_annotation_history_uses_authenticated_actor(auth_env, tmp_path, monkeypatch):
    factory, admin, _ = auth_env
    from app.models.orm import Dataset, Record
    from app.services import annotation_service, image_service, xml_io
    monkeypatch.setattr(annotation_service, "SessionLocal", factory)
    monkeypatch.setattr(image_service, "SessionLocal", factory)
    image = tmp_path / "a.png"
    xml_io.write_xml(image.with_suffix(".xml"), {"labels": [], "quality": ""})
    with factory() as db:
        dataset = Dataset(name="audited")
        db.add(dataset)
        db.flush()
        db.add(Record(dataset_id=dataset.id, filename="a.png", image_path=str(image)))
        db.commit()
        dataset_id = dataset.id
    url = f"/api/datasets/{dataset_id}/image/a.png"
    with TestClient(app) as client:
        sign_in(client)
        revision = client.get(url + "/labels").json()["revision"]
        response = client.put(url + "/labels", json={"expected_revision": revision, "labels": [], "quality": "合格", "actor_id": "forged"})
        assert response.status_code == 200
        history = client.get(url + "/history").json()
        assert history["items"][0]["actor_id"] == admin["id"]
        assert history["items"][0]["username"] == "admin"
        assert history["items"][0]["before"]["quality"] == ""
        assert history["items"][0]["after"]["quality"] == "合格"


def test_four_accounts_edit_distinct_samples_concurrently(auth_env, tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import ExitStack
    from app.models.orm import Dataset, Record
    from app.services import annotation_service, image_service, xml_io
    factory, admin, member = auth_env
    monkeypatch.setattr(annotation_service, "SessionLocal", factory)
    monkeypatch.setattr(image_service, "SessionLocal", factory)
    users = [admin, member, service.create_user("third", "member-test-password"), service.create_user("fourth", "member-test-password")]
    passwords = ["a-long-admin-password", "a-long-member-password", "member-test-password", "member-test-password"]
    with factory() as db:
        dataset = Dataset(name="four-users")
        db.add(dataset)
        db.flush()
        for i in range(4):
            path = tmp_path / f"{i}.png"
            xml_io.write_xml(path.with_suffix(".xml"), {"labels": [], "quality": ""})
            db.add(Record(dataset_id=dataset.id, filename=path.name, image_path=str(path)))
        db.commit()
        dataset_id = dataset.id
    with ExitStack() as stack:
        clients = [stack.enter_context(TestClient(app)) for _ in users]
        for client, user, password in zip(clients, users, passwords):
            sign_in(client, user["username"], password)
        def edit(i):
            url = f"/api/datasets/{dataset_id}/image/{i}.png"
            revision = clients[i].get(url + "/labels").json()["revision"]
            response = clients[i].put(url + "/labels", json={"expected_revision": revision, "labels": [], "quality": "合格"})
            assert response.status_code == 200, response.text
            return clients[i].get(url + "/history").json()["items"][0]["actor_id"]
        with ThreadPoolExecutor(max_workers=4) as pool:
            assert list(pool.map(edit, range(4))) == [u["id"] for u in users]
