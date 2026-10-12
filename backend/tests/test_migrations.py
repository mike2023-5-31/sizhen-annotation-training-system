"""旧库增量升级、备份可恢复性及失败时不继续建表。"""
import sqlite3

import pytest
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session

from app import migrations
from app.models.orm import AnnotationEdit, Dataset, Record


@pytest.fixture
def migration_env(tmp_path, monkeypatch):
    database = tmp_path / "db.sqlite"
    engine = create_engine(f"sqlite:///{database}")
    monkeypatch.setattr(migrations, "engine", engine)
    monkeypatch.setattr(migrations.config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(migrations.config, "ensure_dirs", lambda: None)
    yield tmp_path, database, engine
    engine.dispose()


def test_upgrade_preserves_records_and_backup_restores_old_schema(migration_env):
    root, database, engine = migration_env
    Dataset.__table__.create(engine)
    Record.__table__.create(engine)
    with Session(engine) as db:
        dataset = Dataset(name="旧数据集")
        db.add(dataset)
        db.flush()
        db.add(Record(dataset_id=dataset.id, filename="a.png", image_path="unchanged/a.png", quality="合格"))
        db.commit()
    backup = migrations.migrate()
    assert backup and backup.is_file()
    assert "annotation_edits" in inspect(engine).get_table_names()
    with Session(engine) as db:
        assert db.scalar(select(Dataset)).name == "旧数据集"
        assert db.scalar(select(Record)).image_path == "unchanged/a.png"
        assert list(db.scalars(select(AnnotationEdit))) == []
    # 恢复到独立目标并实际查询，不能仅以备份文件存在判定可恢复。
    with sqlite3.connect(backup) as source, sqlite3.connect(root / "restored.sqlite") as restored:
        source.backup(restored)
        assert restored.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert restored.execute("SELECT name FROM datasets").fetchall() == [("旧数据集",)]
        assert restored.execute("SELECT quality FROM records").fetchall() == [("合格",)]
        assert restored.execute("SELECT name FROM sqlite_master WHERE name='annotation_edits'").fetchall() == []
    second = migrations.migrate()
    assert second != backup and backup.is_file()


def test_new_database_gets_all_tables(migration_env):
    _, _, engine = migration_env
    assert migrations.migrate() is None
    assert {"datasets", "records", "annotation_edits"} <= set(inspect(engine).get_table_names())


def test_backup_failure_prevents_schema_mutation(migration_env, monkeypatch):
    _, _, engine = migration_env
    Dataset.__table__.create(engine)
    Record.__table__.create(engine)
    def fail(*args, **kwargs):
        raise OSError("backup unavailable")
    monkeypatch.setattr(migrations.sqlite3, "connect", fail)
    with pytest.raises(OSError, match="backup unavailable"):
        migrations.migrate()
    assert "annotation_edits" not in inspect(engine).get_table_names()


def test_incompatible_old_table_is_not_reported_as_upgraded(migration_env):
    _, _, engine = migration_env
    with engine.begin() as db:
        db.exec_driver_sql("CREATE TABLE datasets (id INTEGER PRIMARY KEY, name TEXT)")
    with pytest.raises(RuntimeError, match="缺少列"):
        migrations.migrate()
    assert inspect(engine).get_table_names() == ["datasets"]


def test_broken_foreign_keys_stop_upgrade(migration_env):
    _, _, engine = migration_env
    Dataset.__table__.create(engine)
    Record.__table__.create(engine)
    with Session(engine) as db:
        db.add(Record(dataset_id=999, filename="orphan.png", image_path="orphan.png"))
        db.commit()
    with pytest.raises(RuntimeError, match="外键检查失败"):
        migrations.migrate()
    assert "annotation_edits" not in inspect(engine).get_table_names()
