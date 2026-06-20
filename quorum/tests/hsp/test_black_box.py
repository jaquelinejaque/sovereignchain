import json
import os
import sqlite3
import tempfile
from pathlib import Path
import pytest


@pytest.fixture
def tmp_audit_db(monkeypatch, tmp_path):
    db = tmp_path / "audit_test.db"
    monkeypatch.setenv("QUORUM_AUDIT_DB", str(db))
    # Re-import to pick up env
    import importlib
    from quorum.hsp import black_box
    importlib.reload(black_box)
    yield db, black_box


def test_append_returns_hash(tmp_audit_db):
    _, bb = tmp_audit_db
    h = bb.append({"event": "test", "value": 1})
    assert h is not None
    assert len(h) == 64  # sha256 hex


def test_chain_verifies_when_intact(tmp_audit_db):
    _, bb = tmp_audit_db
    for i in range(5):
        bb.append({"event": "test", "i": i})
    ok, broken = bb.verify_chain()
    assert ok is True
    assert broken is None


def test_tampering_payload_detected(tmp_audit_db):
    db, bb = tmp_audit_db
    for i in range(3):
        bb.append({"event": "test", "i": i})
    # Tamper id=2's payload directly
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE audit_log SET payload_json='{\"tampered\":true}' WHERE id=2")
    conn.commit()
    conn.close()
    ok, broken = bb.verify_chain()
    assert ok is False
    assert broken == 2


def test_deleting_row_detected(tmp_audit_db):
    db, bb = tmp_audit_db
    for i in range(5):
        bb.append({"event": "test", "i": i})
    # Delete id=3 — id=4's prev_hash now points to nonexistent
    conn = sqlite3.connect(str(db))
    conn.execute("DELETE FROM audit_log WHERE id=3")
    conn.commit()
    conn.close()
    ok, broken = bb.verify_chain()
    assert ok is False
    # Should break at id=4 because its prev_hash references id=3 hash
    assert broken == 4


def test_export_jsonl_count(tmp_audit_db, tmp_path):
    _, bb = tmp_audit_db
    for i in range(7):
        bb.append({"event": "test", "i": i})
    out = tmp_path / "export.jsonl"
    n = bb.export_jsonl(out_path=str(out))
    assert n == 7
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 7
    for line in lines:
        rec = json.loads(line)
        assert "this_hash" in rec
        assert "prev_hash" in rec


def test_append_never_raises_on_disk_error(tmp_audit_db, monkeypatch):
    _, bb = tmp_audit_db
    # Force open() to fail
    monkeypatch.setattr(bb, "_open", lambda: (_ for _ in ()).throw(OSError("disk full")))
    # Should return None, NOT raise
    h = bb.append({"event": "test"})
    assert h is None
