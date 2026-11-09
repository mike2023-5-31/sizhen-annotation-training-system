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


def test_upgrade_does_not_reuse_legacy_dataset_and_record_ids(migration_env):
    root, database, engine = migration_env
    with sqlite3.connect(database) as db:
        db.executescript("""
            CREATE TABLE datasets (
                id INTEGER PRIMARY KEY, name VARCHAR(255) NOT NULL UNIQUE,
                created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL
            );
            CREATE TABLE records (
                id INTEGER PRIMARY KEY, dataset_id INTEGER NOT NULL, filename VARCHAR(255) NOT NULL,
                image_path VARCHAR(1024) NOT NULL, timestamp DATETIME, quality VARCHAR(32),
                created_at DATETIME NOT NULL, FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
            );
            CREATE INDEX ix_records_dataset_id ON records (dataset_id);
            INSERT INTO datasets VALUES (1, '旧数据集', '2026-09-01 12:00:00', '2026-09-01 12:00:00');
            INSERT INTO records (id, dataset_id, filename, image_path, created_at)
                VALUES (1, 1, 'a.png', 'unchanged/a.png', '2026-09-01 12:00:00');
        """)
    AnnotationEdit.__table__.create(engine)
    with Session(engine) as db:
        db.add(AnnotationEdit(record_id=51, dataset_id=41, previous_revision="missing",
                              revision="r1", before={}, after={}))
        db.commit()
    # 数据集删除保留磁盘文件。即便这个目录没有对应的审计，也不能分配相同 ID。
    (root / "datasets" / "57").mkdir(parents=True)
    backup = migrations.migrate()
    assert backup is not None
    with sqlite3.connect(backup) as old:
        assert old.execute("SELECT id, dataset_id FROM records").fetchall() == [(1, 1)]
        assert old.execute("SELECT record_id, dataset_id FROM annotation_edits").fetchall() == [(51, 41)]
        assert "modality" not in {row[1] for row in old.execute("PRAGMA table_info(datasets)")}
    assert [index["name"] for index in inspect(engine).get_indexes("records")] == ["ix_records_dataset_id"]

    with Session(engine) as db:
        assert db.get(Record, 1).image_path == "unchanged/a.png"
        assert db.get(Dataset, 1).modality is None
        first = Dataset(name="新数据集")
        db.add(first)
        db.flush()
        record = Record(dataset_id=first.id, filename="b.png", image_path="new/b.png")
        db.add(record)
        db.flush()
        assert first.id > 57
        assert record.id > 51
        deleted_ids = first.id, record.id
        db.delete(first)
        db.commit()
    migrations.migrate()
    with Session(engine) as db:
        second = Dataset(name="重新创建")
        db.add(second)
        db.flush()
        record = Record(dataset_id=second.id, filename="b.png", image_path="new/b.png")
        db.add(record)
        db.flush()
        assert second.id > deleted_ids[0]
        assert record.id > deleted_ids[1]
        assert len(list(db.scalars(select(AnnotationEdit)))) == 1
    with sqlite3.connect(database) as upgraded:
        assert upgraded.execute("PRAGMA foreign_key_check").fetchall() == []
