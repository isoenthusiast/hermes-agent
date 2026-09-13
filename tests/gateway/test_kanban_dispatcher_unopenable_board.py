"""The dispatcher skips a board it cannot open — once — and keeps the pass.

Regression for ``tick failed on board zz-checkctl``: a board store whose
connect-time migration failed logged one ERROR every 60 s for as long as the
gateway ran (101 matching lines in 24 h on the box). A board the dispatcher
cannot open is news once; the rest of the pass must still dispatch.
"""

from __future__ import annotations

import logging
import sqlite3
from types import SimpleNamespace

import pytest

from gateway.kanban_watchers_dispatcher import _DispatcherSettings, _KanbanDispatcher
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd

_FIXTURE_SQL = "CREATE TABLE tasks (id TEXT, assignee TEXT, status TEXT, created_at INTEGER)"


def _settings() -> _DispatcherSettings:
    return _DispatcherSettings(
        interval=60.0,
        max_spawn=None,
        max_in_progress=None,
        failure_limit=kbd.DEFAULT_FAILURE_LIMIT,
        stale_timeout_seconds=0,
        reconcile_orphans=False,
        default_assignee=None,
        max_in_progress_per_profile=None,
    )


def _write_store(slug: str, ddl: str = _FIXTURE_SQL, *, replace: bool = False):
    """Write a foreign (non-board) SQLite store where board *slug* expects one."""
    path = kb.kanban_db_path(board=slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    if replace:
        for suffix in ("", "-wal", "-shm"):
            path.with_name(path.name + suffix).unlink(missing_ok=True)
    con = sqlite3.connect(path)
    try:
        con.execute(ddl)
        con.commit()
    finally:
        con.close()
    return path


def _columns(path) -> list[str]:
    con = sqlite3.connect(path)
    try:
        return [row[1] for row in con.execute("pragma table_info(tasks)")]
    finally:
        con.close()


def _result():
    return SimpleNamespace(
        spawned=[], reclaimed=0, crashed=[], timed_out=[], promoted=0, auto_blocked=[],
    )


def _dispatch_recording(seen: list):
    def _dispatch(conn, board=None, **kwargs):
        seen.append(board)
        return _result()
    return _dispatch


@pytest.fixture
def good_board():
    """A real, healthy board next to the foreign store."""
    kb.init_db(board="good")
    return "good"


def test_unopenable_board_is_reported_once_and_other_boards_keep_ticking(
    monkeypatch, caplog, good_board,
):
    fixture = _write_store("zz-checkctl")
    seen: list = []
    monkeypatch.setattr(kbd, "dispatch_once", _dispatch_recording(seen))
    dispatcher = _KanbanDispatcher(kb, _settings())

    with caplog.at_level(logging.ERROR, logger="gateway.run"):
        passes = [dispatcher.tick_once() for _ in range(3)]

    # The healthy board is dispatched on every pass; the refused one never is.
    assert seen.count("good") == 3
    assert "zz-checkctl" not in seen
    for result in passes:
        by_slug = dict(result)
        assert by_slug["zz-checkctl"] is None
        assert by_slug["good"] is not None

    # Reported by name ONCE, with the reason, without a traceback.
    faults = [record.getMessage() for record in caplog.records
              if "zz-checkctl" in record.getMessage()]
    assert len(faults) == 1
    assert "is not a kanban task store" in faults[0]
    assert "tasks lacks body,created_by,title" in faults[0]
    assert not [record for record in caplog.records if record.exc_info]
    assert not any("tick failed on board" in record.getMessage() for record in caplog.records)

    # Three passes half-migrated nothing: the foreign store is untouched.
    assert _columns(fixture) == ["id", "assignee", "status", "created_at"]


def test_a_different_fault_is_reported_again(monkeypatch, caplog, good_board):
    """Once per fault, not once per process: a changed store speaks up again."""
    _write_store("zz-checkctl")
    monkeypatch.setattr(kbd, "dispatch_once", _dispatch_recording([]))
    dispatcher = _KanbanDispatcher(kb, _settings())

    def _faults() -> list[str]:
        return [record.getMessage() for record in caplog.records
                if "is not a kanban task store" in record.getMessage()]

    with caplog.at_level(logging.ERROR, logger="gateway.run"):
        dispatcher.tick_once()
        assert len(_faults()) == 1
        dispatcher.tick_once()
        assert len(_faults()) == 1  # same file, same fault: silent
        # The operator swaps in a different foreign store (new fingerprint).
        _write_store("zz-checkctl", "CREATE TABLE tasks (id TEXT, status TEXT, note TEXT)",
                     replace=True)
        dispatcher.tick_once()
    assert len(_faults()) == 2


def test_other_tick_failures_are_also_suppressed_until_they_change(
    monkeypatch, caplog, good_board,
):
    """Any repeating per-board fault gets one report + a traceback, not a pulse."""
    def _boom(conn, board=None, **kwargs):
        raise RuntimeError("worker spawn backend unavailable")

    monkeypatch.setattr(kbd, "dispatch_once", _boom)
    dispatcher = _KanbanDispatcher(kb, _settings())

    with caplog.at_level(logging.ERROR, logger="gateway.run"):
        for _ in range(3):
            dispatcher.tick_once()

    faults = [record for record in caplog.records
              if "tick failed on board good" in record.getMessage()]
    assert len(faults) == 1
    assert faults[0].exc_info  # a real traceback is still recorded, once


def test_recovery_makes_the_next_fault_news_again(monkeypatch, caplog, good_board):
    """A successful tick clears that board's fault state."""
    state = {"fail": True}

    def _flaky(conn, board=None, **kwargs):
        if board == "good" and state["fail"]:
            raise RuntimeError("transient")
        return _result()

    monkeypatch.setattr(kbd, "dispatch_once", _flaky)
    dispatcher = _KanbanDispatcher(kb, _settings())

    with caplog.at_level(logging.ERROR, logger="gateway.run"):
        dispatcher.tick_once()   # fault -> reported
        state["fail"] = False
        dispatcher.tick_once()   # recovered -> fault state cleared
        state["fail"] = True
        dispatcher.tick_once()   # same fault text, but news again

    faults = [record for record in caplog.records
              if "tick failed on board good" in record.getMessage()]
    assert len(faults) == 2
