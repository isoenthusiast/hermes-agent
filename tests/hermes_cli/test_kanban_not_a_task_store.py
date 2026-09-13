"""A ``kanban.db`` that is not a task store is refused, not half-migrated.

Regression for the ``zz-checkctl`` board (an audit-harness fixture): a 4-column
``tasks`` table under ``boards/zz-checkctl/`` whose additive migration could
never complete — ``_backfill_legacy_inflight_runs`` selects ``claim_lock``, a
column the additive lists cannot add — so every connect aborted and the gateway
dispatcher logged a traceback per 60 s tick, forever.

The refusal is the same shape rule ``_not_a_store()`` applies in
``~/.hermes/scripts/kanban_card_lint.py``: prove the store is not a board, skip
it by name, and leave it alone.
"""

from __future__ import annotations

import sqlite3

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

# The audit harness fixture's shape: no title/body/created_by.
_FIXTURE_SQL = "CREATE TABLE tasks (id TEXT, assignee TEXT, status TEXT, created_at INTEGER)"

# Every column of the FIRST released schema that no additive migration adds.
# This is what a real board last written by an old release looks like on disk,
# and it must still migrate — the refusal must not catch it.
_LEGACY_BOARD_SQL = """
CREATE TABLE tasks (
    id                   TEXT PRIMARY KEY,
    title                TEXT NOT NULL,
    body                 TEXT,
    assignee             TEXT,
    status               TEXT NOT NULL,
    priority             INTEGER DEFAULT 0,
    created_by           TEXT,
    created_at           INTEGER NOT NULL,
    started_at           INTEGER,
    completed_at         INTEGER,
    workspace_kind       TEXT NOT NULL DEFAULT 'scratch',
    workspace_path       TEXT,
    claim_lock           TEXT,
    claim_expires        INTEGER
)
"""


def _columns(path) -> list[str]:
    con = sqlite3.connect(path)
    try:
        return [row[1] for row in con.execute("pragma table_info(tasks)")]
    finally:
        con.close()


def _tables(path) -> list[str]:
    con = sqlite3.connect(path)
    try:
        return sorted(row[0] for row in con.execute(
            "select name from sqlite_master where type='table' order by name"))
    finally:
        con.close()


def _make_store(slug: str, ddl: str, insert: str | None = None):
    path = kb.kanban_db_path(board=slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    try:
        con.execute(ddl)
        if insert:
            con.execute(insert)
        con.commit()
    finally:
        con.close()
    return path


def test_connect_refuses_the_audit_fixture_before_migrating_it():
    """The fixture is refused by name, and NOTHING is written to it."""
    path = _make_store(
        "zz-checkctl", _FIXTURE_SQL,
        "INSERT INTO tasks (id, assignee, status, created_at) "
        "VALUES ('chk-1', 'maintenance', 'ready', 1)",
    )

    with pytest.raises(kbc.KanbanDbNotATaskStoreError) as excinfo:
        kbc.connect(board="zz-checkctl")

    message = str(excinfo.value)
    assert "zz-checkctl" in message
    assert "tasks lacks body,created_by,title" in message

    # Refused INTACT: no additive columns, no sibling tables, fixture row kept.
    assert _columns(path) == ["id", "assignee", "status", "created_at"]
    assert _tables(path) == ["tasks"]
    con = sqlite3.connect(path)
    try:
        assert con.execute("select count(*) from tasks").fetchone()[0] == 1
    finally:
        con.close()


def test_connect_still_migrates_a_legacy_board_missing_only_additive_columns():
    """No false refusal: a real old board still gets its additive pass."""
    path = _make_store(
        "legacy", _LEGACY_BOARD_SQL,
        "INSERT INTO tasks (id, title, status, created_at) "
        "VALUES ('t_legacy', 'old card', 'ready', 1)",
    )

    with kbc.connect_closing(board="legacy") as conn:
        assert conn.execute("select count(*) from tasks").fetchone()[0] == 1

    columns = _columns(path)
    assert "block_recurrences" in columns          # additive pass ran
    assert "consecutive_failures" in columns
    assert "task_runs" in _tables(path)


def test_connect_still_creates_a_fresh_board():
    """A missing/empty file is not a shape defect — init is unchanged."""
    with kbc.connect_closing(board="fresh") as conn:
        assert conn.execute("select count(*) from tasks").fetchone()[0] == 0


def test_not_a_task_store_reason_matches_the_card_linter_rule():
    """One fleet-wide rule: the same column set the linter proves with."""
    assert set(kbc._BASELINE_TASK_COLUMNS) == {
        "id", "title", "body", "created_by", "status", "created_at"}
