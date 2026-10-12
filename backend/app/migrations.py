"""执行 python -m app.migrations：备份SQLite，再添加v2表；不清除既有数据。"""
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from sqlalchemy import inspect

from app import config
from app.db import Base, engine
from app.models import orm  # noqa: F401


def _check_existing_schema() -> None:
    """此版本只增加表；旧表不兼容时停止，不能把 create_all 当列迁移。"""
    inspector = inspect(engine)
    existing = set(inspector.get_table_names())
    for table in Base.metadata.sorted_tables:
        if table.name not in existing:
            continue
        actual = {column["name"] for column in inspector.get_columns(table.name)}
        missing = set(table.columns.keys()) - actual
        if missing:
            raise RuntimeError(f"表 {table.name} 缺少列 {sorted(missing)}；需要专门迁移，未修改数据库")


def _check_sqlite(connection: sqlite3.Connection) -> None:
    if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
        raise RuntimeError("SQLite 完整性检查失败，未执行升级")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise RuntimeError("SQLite 外键检查失败，未执行升级")


def migrate() -> Path | None:
    config.ensure_dirs()
    backup = None
    database = engine.url.database
    existed = engine.dialect.name == "sqlite" and database and Path(database).is_file()
    _check_existing_schema()
    if existed:
        destination = config.DATA_DIR / "backups"
        destination.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = destination / f"before-v2-{stamp}.sqlite"
        with sqlite3.connect(database) as source, sqlite3.connect(backup) as target:
            _check_sqlite(source)
            source.backup(target)
            _check_sqlite(target)
    with engine.begin() as connection:
        Base.metadata.create_all(connection)
    return backup


if __name__ == "__main__":
    backup = migrate()
    print(f"Schema ready; backup: {backup or 'new database'}")
