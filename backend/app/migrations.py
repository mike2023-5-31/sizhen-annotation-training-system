"""执行 python -m app.migrations：备份并升级 SQLite；保留既有数据。"""
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from sqlalchemy import inspect
from sqlalchemy.schema import CreateTable

from app import config
from app.db import Base, engine
from app.models.orm import Dataset, Record


def _check_existing_schema() -> None:
    """旧表缺少必要列时停止，不能把 create_all 当列迁移。"""
    inspector = inspect(engine)
    existing = set(inspector.get_table_names())
    for table in Base.metadata.sorted_tables:
        if table.name not in existing:
            continue
        actual = {column["name"] for column in inspector.get_columns(table.name)}
        missing = set(table.columns.keys()) - actual
        if engine.dialect.name == "sqlite" and table.name == "datasets":
            missing.discard("modality")  # 此可空字段通过下方专用迁移添加。
        if missing:
            raise RuntimeError(f"表 {table.name} 缺少列 {sorted(missing)}；需要专门迁移，未修改数据库")


def _check_sqlite(connection: sqlite3.Connection) -> None:
    if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
        raise RuntimeError("SQLite 完整性检查失败，未执行升级")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise RuntimeError("SQLite 外键检查失败，未执行升级")


def _upgrade_sqlite(connection: sqlite3.Connection) -> None:
    schemas = dict(connection.execute("SELECT name, sql FROM sqlite_master WHERE type='table'"))
    tables = [table for table in (Dataset.__table__, Record.__table__)
              if table.name in schemas and "AUTOINCREMENT" not in schemas[table.name].upper()]
    add_modality = "datasets" in schemas and "modality" not in {
        row[1] for row in connection.execute("PRAGMA table_info(datasets)")
    }
    if not tables and not add_modality:
        return

    # SQLite 修改主键定义需要重建表。关闭外键后重建并在同一事务内核验，
    # 避免删除旧 datasets 表时触发 ON DELETE CASCADE。
    connection.execute("PRAGMA foreign_keys=OFF")
    with connection:
        connection.execute("BEGIN IMMEDIATE")
        if add_modality:
            connection.execute("ALTER TABLE datasets ADD COLUMN modality VARCHAR(16)")
        for table in tables:
            name = table.name
            columns = list(table.columns.keys())
            current_columns = {row[1] for row in connection.execute(f"PRAGMA table_info({name})")}
            if current_columns != set(columns):
                raise RuntimeError(f"表 {name} 列与迁移结构不一致，未修改数据库")
            related = connection.execute(
                "SELECT sql FROM sqlite_master WHERE tbl_name=? AND type IN ('index','trigger') AND sql IS NOT NULL",
                (name,),
            ).fetchall()
            new_name = f"_upgrade_{name}"
            ddl = str(CreateTable(table).compile(dialect=engine.dialect))
            connection.execute(ddl.replace(f"CREATE TABLE {name}", f"CREATE TABLE {new_name}", 1))
            column_list = ", ".join(columns)
            connection.execute(f"INSERT INTO {new_name} ({column_list}) SELECT {column_list} FROM {name}")
            connection.execute(f"DROP TABLE {name}")
            connection.execute(f"ALTER TABLE {new_name} RENAME TO {name}")
            for (sql,) in related:
                connection.execute(sql)

            # sqlite_sequence 已包含存活行。额外跳过历史和旧目录中可见的已删除 ID。
            retired_id = 0
            if "annotation_edits" in schemas:
                key = "dataset_id" if name == "datasets" else "record_id"
                retired_id = connection.execute(f"SELECT COALESCE(MAX({key}), 0) FROM annotation_edits").fetchone()[0]
            if name == "datasets":
                data = config.DATA_DIR / "datasets"
                if data.is_dir():
                    retired_id = max(retired_id, max(
                        (int(path.name) for path in data.iterdir()
                         if path.is_dir() and path.name.isascii() and path.name.isdecimal()
                         and len(path.name) <= 19 and int(path.name) < 2**63 - 1),
                        default=0,
                    ))
            updated = connection.execute("UPDATE sqlite_sequence SET seq=MAX(seq, ?) WHERE name=?",
                                         (retired_id, name))
            if not updated.rowcount:
                connection.execute("INSERT INTO sqlite_sequence (name, seq) VALUES (?, ?)", (name, retired_id))
        _check_sqlite(connection)


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
        with closing(sqlite3.connect(database)) as source, closing(sqlite3.connect(backup)) as target:
            _check_sqlite(source)
            source.backup(target)
            _check_sqlite(target)
            _upgrade_sqlite(source)
    with engine.begin() as connection:
        Base.metadata.create_all(connection)
    return backup


if __name__ == "__main__":
    backup = migrate()
    print(f"Schema ready; backup: {backup or 'new database'}")
