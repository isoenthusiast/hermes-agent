"""``dispatch_hold`` — the "work is already in flight" guardrail.

A card filed for work an agent is about to do by hand (or is already doing
inline) used to be creatable *ready* and *assigned*, which is exactly the
shape auto-dispatch looks for: the next tick (the gateway ticks continuously,
~35s on the reporting board) claimed it and spawned a second worker onto the
same work. ``dispatch_hold`` parks such a card in its lane until an explicit
release, so the dispatcher's queue predicate (``_lane_rows``) never sees it.

Invariants pinned here:

* A held card is invisible to BOTH dispatch lanes — and the hold is per-card,
  so an unheld sibling row in the same lane stays dispatchable.
* Releasing the hold (``hermes kanban unblock``) leaves the card exactly where
  it was (status untouched, still assigned) and the very next tick picks it up.
"""

from __future__ import annotations

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


def _fake_spawn(*_args, **_kwargs):
    return 12345


def _row(conn, tid):
    return conn.execute(
        "SELECT status, assignee, dispatch_hold FROM tasks WHERE id = ?", (tid,)
    ).fetchone()


def test_held_card_is_never_selected_by_auto_dispatch(kanban_home: Path) -> None:
    """Both lanes: a held row is filtered out of the dispatch queue while its
    unheld sibling in the same lane is still picked up."""
    with kbc.connect() as conn:
        held_ready = kb.create_task(
            conn, title="inline work in flight", assignee="default", dispatch_hold=True)
        control = kb.create_task(conn, title="nothing in flight", assignee="default")
        # A held card that reached the review lane stays parked there too.
        held_review = kb.create_task(
            conn, title="held review", assignee="default", dispatch_hold=True)
        kb.claim_task(conn, held_review)
        assert kb.request_review(
            conn, held_review, summary="impl done, held for inline follow-up",
            expected_run_id=kb.get_task(conn, held_review).current_run_id) is True
        assert _row(conn, held_review)["status"] == "review"

        assert [row["id"] for row in kbd._lane_rows(conn, "ready")] == [control]
        assert [row["id"] for row in kbd._lane_rows(conn, "review")] == []

        # The hold is a dispatch filter, not a state change: both cards keep
        # their status and assignee, so the board still shows whose work it is.
        assert (_row(conn, held_ready)["status"], _row(conn, held_ready)["assignee"]) == (
            "ready", "default")


def test_release_makes_the_held_card_dispatchable_again(kanban_home: Path) -> None:
    """``unblock`` is the paired release verb: it clears the hold in place and
    the next tick dispatches the card it was already carrying."""
    with kbc.connect() as conn:
        held = kb.create_task(
            conn, title="inline work in flight", assignee="default", dispatch_hold=True)

        # Nothing to do this tick: the only row is held.
        first = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=True)
        assert first.spawned == []
        # A held card is not "work waiting for a worker" either: the gateway's
        # tick trigger and the stuck-dispatcher telemetry read this probe, and
        # counting a parked card would make both fire on every interval.
        assert kbd.has_spawnable_ready(conn) is False

        assert kb.unblock_task(conn, held) is True

        row = _row(conn, held)
        assert (row["status"], row["dispatch_hold"]) == ("ready", 0)
        assert kbd.has_spawnable_ready(conn) is True

        second = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=True)
        assert [entry[0] for entry in second.spawned] == [held]
