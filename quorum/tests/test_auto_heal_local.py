"""Tests for auto_heal local-error pickup.

Before 2026-06-28 auto_heal only read Cloud Run logs via gcloud. It missed
112 EMFILE traces sitting in ~/.quorum/launchagent-stderr.log for 26h
because the local logs were never queried. These tests pin the new
behaviour: local stderr files are scanned, tracebacks split, and frame
paths from three deploy shapes (Cloud Run container, macOS framework
site-packages, editable source tree) all map to the same repo-relative
target file.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


# auto_heal is a script (not a package), so import it by file path.
_HEAL_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "quorum-self-modify" / "auto_heal.py"
)
_spec = importlib.util.spec_from_file_location("auto_heal_under_test", _HEAL_PATH)
auto_heal = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["auto_heal_under_test"] = auto_heal
_spec.loader.exec_module(auto_heal)  # type: ignore[union-attr]


# ---------- parse_traceback: 3 deploy shapes --------------------------------


class TestParseTraceback:
    """Each shape must resolve to the same ``src/quorum/...`` target so
    self_modify sees a single canonical file to propose against."""

    def test_cloud_run_container_path(self) -> None:
        tb = (
            'Traceback (most recent call last):\n'
            '  File "/opt/venv/lib/python3.12/site-packages/quorum/server/main.py", line 847, in stripe_webhook\n'
            "    return process(payload)\n"
            "KeyError: 'customer.subscription.created'\n"
        )
        info = auto_heal.parse_traceback(tb)
        assert info is not None
        assert info["file"] == "src/quorum/server/main.py"
        assert info["line"] == 847
        assert info["exc_type"] == "KeyError"

    def test_macos_framework_site_packages(self) -> None:
        tb = (
            "Traceback (most recent call last):\n"
            '  File "/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/quorum/agents/drafts.py", line 468, in sell_quorum\n'
            "    path.write_text(...)\n"
            "OSError: [Errno 24] Too many open files\n"
        )
        info = auto_heal.parse_traceback(tb)
        assert info is not None
        assert info["file"] == "src/quorum/agents/drafts.py"
        assert info["line"] == 468
        assert info["exc_type"] == "OSError"

    def test_editable_source_tree(self) -> None:
        tb = (
            "Traceback (most recent call last):\n"
            '  File "/Users/facec/sovereignchain/quorum/src/quorum/core/consensus.py", line 528, in consensus\n'
            "    embedder = EmbeddingProvider.from_env()\n"
            "RuntimeError: no embedder available\n"
        )
        info = auto_heal.parse_traceback(tb)
        assert info is not None
        assert info["file"] == "src/quorum/core/consensus.py"
        assert info["line"] == 528
        assert info["exc_type"] == "RuntimeError"

    def test_no_quorum_frames_returns_none(self) -> None:
        # A traceback that only references stdlib must not produce a target.
        tb = (
            "Traceback (most recent call last):\n"
            '  File "/usr/lib/python3.12/json/decoder.py", line 355, in raw_decode\n'
            "    obj, end = self.scan_once(s, idx)\n"
            "json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)\n"
        )
        assert auto_heal.parse_traceback(tb) is None

    def test_deepest_frame_wins(self) -> None:
        # A traceback with multiple quorum frames must target the deepest
        # (last) one — that is where the exception actually fired.
        tb = (
            "Traceback (most recent call last):\n"
            '  File "/opt/venv/lib/python3.12/site-packages/quorum/cli.py", line 100, in run\n'
            "    do_thing()\n"
            '  File "/opt/venv/lib/python3.12/site-packages/quorum/core/consensus.py", line 700, in consensus\n'
            "    bad()\n"
            "TypeError: unsupported operand\n"
        )
        info = auto_heal.parse_traceback(tb)
        assert info is not None
        assert info["file"] == "src/quorum/core/consensus.py"
        assert info["line"] == 700


# ---------- _split_tracebacks: multi-error log -------------------------------


class TestSplitTracebacks:
    def test_two_back_to_back_tracebacks(self) -> None:
        log_blob = (
            "Some unrelated stdout line\n"
            "Traceback (most recent call last):\n"
            '  File "/opt/venv/lib/python3.12/site-packages/quorum/a.py", line 10, in x\n'
            "    pass\n"
            "ValueError: first\n"
            "Some interleaved noise\n"
            "Traceback (most recent call last):\n"
            '  File "/opt/venv/lib/python3.12/site-packages/quorum/b.py", line 20, in y\n'
            "    pass\n"
            "KeyError: 'second'\n"
        )
        blocks = auto_heal._split_tracebacks(log_blob)
        assert len(blocks) == 2
        assert "ValueError: first" in blocks[0]
        assert "KeyError: 'second'" in blocks[1]
        assert "first" not in blocks[1]  # leakage guard

    def test_no_tracebacks_returns_empty(self) -> None:
        blob = "Just a normal log line.\nAnother log line.\n"
        assert auto_heal._split_tracebacks(blob) == []

    def test_rich_boxed_frames_dont_terminate_block_prematurely(self) -> None:
        # rich.traceback wraps "ExcType: msg" inside a box with ``│``. A
        # naive splitter would close the block at the BOXED ExcType and
        # miss the real terminator a few lines down.
        blob = (
            "Traceback (most recent call last):\n"
            '  File "/opt/venv/lib/python3.12/site-packages/quorum/a.py", line 10\n'
            "│   ValueError: not really, this is decoration   │\n"
            "    bad()\n"
            "OSError: [Errno 24] Too many open files\n"
        )
        blocks = auto_heal._split_tracebacks(blob)
        assert len(blocks) == 1
        assert "OSError" in blocks[0]


# ---------- fetch_local_errors: integration -------------------------------


class TestFetchLocalErrors:
    def test_picks_up_stderr_log(self, tmp_path: Path, monkeypatch) -> None:
        # Build a fake ~/.quorum with a single stderr log containing one
        # rich-style EMFILE traceback (the real-world shape that caused
        # the 26h blind spot on 2026-06-28).
        log = tmp_path / "launchagent-stderr.log"
        log.write_text(
            "Traceback (most recent call last):\n"
            '  File "/Library/Frameworks/Python.framework/Versions/3.14/lib/'
            'python3.14/site-packages/quorum/agents/drafts.py", line 468, in sell_quorum\n'
            "    path.write_text(content)\n"
            "OSError: [Errno 24] Too many open files\n"
        )
        monkeypatch.setattr(auto_heal, "LOCAL_LOG_DIR", tmp_path)
        errors = auto_heal.fetch_local_errors()
        assert len(errors) == 1
        assert errors[0]["_source"] == "local"
        # _path is where the log lives (so cron can show which file had
        # the error), NOT the file referenced by the traceback.
        assert errors[0]["_path"].endswith("launchagent-stderr.log")
        info = auto_heal.parse_traceback(errors[0]["textPayload"])
        assert info is not None
        assert info["file"] == "src/quorum/agents/drafts.py"
        assert info["line"] == 468
        assert info["exc_type"] == "OSError"

    def test_missing_log_dir_returns_empty(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(auto_heal, "LOCAL_LOG_DIR", tmp_path / "does_not_exist")
        assert auto_heal.fetch_local_errors() == []

    def test_empty_log_dir_returns_empty(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(auto_heal, "LOCAL_LOG_DIR", tmp_path)
        assert auto_heal.fetch_local_errors() == []


# ---------- _tail_bytes: big-log safety ------------------------------------


class TestTailBytes:
    def test_truncates_to_last_n_bytes(self, tmp_path: Path) -> None:
        p = tmp_path / "big.log"
        # Write 100 KB so the default 200 KB cap doesn't truncate, then
        # write 300 KB and verify only the trailing 200 KB comes back.
        p.write_bytes(b"A" * 300_000)
        out = auto_heal._tail_bytes(p, n=200_000)
        assert len(out) == 200_000
        assert out.startswith("A")

    def test_missing_file_returns_empty_string(self, tmp_path: Path) -> None:
        assert auto_heal._tail_bytes(tmp_path / "nope.log") == ""
