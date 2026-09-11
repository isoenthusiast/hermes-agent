"""Create-time guard: never put a second worker into a busy workspace.

``dispatch_hold`` (t_677e4fd2) fixed the *shape* of the duplicate-worker
incident — a ``ready``+assigned card filed for work that is already in flight is
exactly what the next tick claims — but only for a filer who remembered to pass
``--hold``, and the tool surface did not expose the flag at all. These tests pin
the part that has to work without anyone remembering anything:

* a card created into a workspace that already has an active run is held, and
  the following tick spawns exactly one worker per workspace;
* a *stale* claim does not hold anything (a crashed worker must not park new
  cards forever), and a plain ``scratch`` card is never held;
* the tool surface can hold on request, and an automatic hold names the run it
  was taken against;
* ``unblock`` is still the release verb for both.

The first test is the incident reproducer: on a base without the create-time
probe the follower card is claimed and a second worker starts in the same
checkout.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


def _row(conn, tid):
    return conn.execute(
        "SELECT status, assignee, workspace_path, dispatch_hold FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()


def _run_in(conn, path: Path, title: str = "agent working inline") -> str:
    """A card that is *right now* running inside ``path`` (live claim)."""
    tid = kb.create_task(
        conn, title=title, assignee="default",
        workspace_kind="dir", workspace_path=str(path))
    assert kb.claim_task(conn, tid) is not None
    return tid


def _dispatch(conn) -> list[str]:
    """One real tick; returns the task ids it spawned, in order."""
    spawned: list[str] = []
    kbd.dispatch_once(
        conn, spawn_fn=lambda task, workspace, **_: spawned.append(task.id) or 4242)
    return spawned


def test_no_second_worker_for_a_workspace_with_an_active_run(
    kanban_home: Path, tmp_path: Path,
) -> None:
    """The incident: a filer inside a shared checkout files a card for it.

    Only one worker may end up in that checkout. Cards for other workspaces —
    and a plain scratch card — stay dispatchable, so the hold is per-workspace
    and never a board-wide stall.
    """
    shared, elsewhere = tmp_path / "shared", tmp_path / "elsewhere"
    shared.mkdir()
    elsewhere.mkdir()
    with kbc.connect() as conn:
        busy = _run_in(conn, shared)
        follower = kb.create_task(
            conn, title="follow-up in the same checkout", assignee="default",
            workspace_kind="dir", workspace_path=str(shared))
        other_dir = kb.create_task(
            conn, title="a different checkout", assignee="default",
            workspace_kind="dir", workspace_path=str(elsewhere))
        scratch = kb.create_task(conn, title="plain scratch card", assignee="default")

        # The whole incident in one line: the follower was created *ready*,
        # *assigned* and dispatchable even though `busy` was mid-run in the
        # very same workspace.
        assert _row(conn, follower)["dispatch_hold"] == 1, (
            f"card {follower} filed into a workspace with an active run "
            f"({busy}) was left dispatchable — the next tick spawns a second "
            "worker onto the same checkout")

        assert sorted(_dispatch(conn)) == sorted([other_dir, scratch])

        # Release is still `unblock` (no new verb): the auto-held card keeps its
        # status and assignee and becomes dispatchable in place.
        assert kb.unblock_task(conn, follower) is True
        row = _row(conn, follower)
        assert (row["status"], row["dispatch_hold"]) == ("ready", 0)
        assert _dispatch(conn) == [follower]


def test_a_stale_claim_does_not_hold_a_new_card(
    kanban_home: Path, tmp_path: Path,
) -> None:
    """The probe reads the dispatcher's live-claim bookkeeping, not "a row
    exists": a claim past its TTL is a crashed worker, and parking every card
    filed afterwards would be worse than the race it guards against."""
    shared = tmp_path / "shared"
    shared.mkdir()
    with kbc.connect() as conn:
        crashed = kb.create_task(
            conn, title="crashed mid-run", assignee="default",
            workspace_kind="dir", workspace_path=str(shared))
        kb.claim_task(conn, crashed)
        conn.execute(
            "UPDATE tasks SET claim_expires = ? WHERE id = ?",
            (int(time.time()) - 60, crashed))

        follower = kb.create_task(
            conn, title="next card for the same checkout", assignee="default",
            workspace_kind="dir", workspace_path=str(shared))
        assert _row(conn, follower)["dispatch_hold"] == 0
        # The stale claim is the reclaim sweep's business, not the hold's: the
        # tick clears it (and requeues the row) instead of leaving it as a
        # phantom that parks every card filed afterwards.
        _dispatch(conn)
        assert _row(conn, follower)["dispatch_hold"] == 0


def test_create_tool_takes_an_explicit_hold(kanban_home: Path, tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """``hold=true`` on the tool surface parks a card the same way ``--hold``
    does on the CLI, and the payload says so."""
    from tools import kanban_tools as kt

    monkeypatch.setenv("HERMES_PROFILE", "default")
    with kbc.connect() as conn:
        out = json.loads(kt._handle_create({
            "title": "customise the greeting copy", "assignee": "default", "hold": True}))
        tid = out["task_id"]
        assert _row(conn, tid)["dispatch_hold"] == 1
        assert out.get("dispatch_hold") is True
        assert "unblock" in str(out.get("hold_reason"))
        assert [row["id"] for row in kbd._lane_rows(conn, "ready")] == []
        assert kb.unblock_task(conn, tid) is True
        assert [row["id"] for row in kbd._lane_rows(conn, "ready")] == [tid]


def test_create_tool_auto_hold_names_the_busy_run(kanban_home: Path, tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """A dispatcher-owned worker filing a child into its own workspace gets a
    held card and a reason naming who is still in there — no silent parking and
    no ``hold`` argument required from the model."""
    from tools import kanban_tools as kt

    shared = tmp_path / "shared"
    shared.mkdir()
    with kbc.connect() as conn:
        filer = _run_in(conn, shared, title="orchestrator at work")
        monkeypatch.setenv("HERMES_KANBAN_TASK", filer)
        monkeypatch.setenv("HERMES_PROFILE", "default")

        out = json.loads(kt._handle_create({
            "title": "child in the shared checkout", "assignee": "default",
            "workspace_kind": "dir", "workspace_path": str(shared)}))
        child = out["task_id"]

        assert _row(conn, child)["dispatch_hold"] == 1
        assert out["dispatch_hold"] is True
        assert filer in out["hold_reason"], out
        assert kb.dispatch_hold_conflict(conn, child) == filer
        assert [row["id"] for row in kbd._lane_rows(conn, "ready")] == []
