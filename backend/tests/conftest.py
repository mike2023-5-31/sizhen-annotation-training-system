"""需要生产应用路由的功能测试显式登录，禁止绕过认证中间件。"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app import config
from app.services import auth_service


@pytest.fixture
def login_admin(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'test_accounts.sqlite'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    monkeypatch.setattr(auth_service, "SessionLocal", sessionmaker(bind=engine, expire_on_commit=False))
    monkeypatch.setattr(config, "COOKIE_SECURE", False)
    auth_service.create_user("test-admin", "test-admin-password", "admin")
    def authenticate(client):
        result = client.post("/api/auth/login", json={"username": "test-admin", "password": "test-admin-password"},
                             headers={"X-Requested-With": "XMLHttpRequest"})
        assert result.status_code == 200, result.text
        client.headers["X-CSRF-Token"] = result.json()["csrf_token"]
        return client
    yield authenticate
    engine.dispose()
