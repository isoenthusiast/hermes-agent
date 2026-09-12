"""Focused lock-handle behavior for hosted Group Chat reads.

The room read path is the one a web backend runs unconditionally at boot and then on every
idle poll: it must neither enter a write transaction nor hold a WRITE handle (``O_RDWR``) on
the shared ``state.db``, so a read-only surface cannot be the thing that keeps a second
writer open on a store another process owns.
"""

import os
import sqlite3
import sys
from pathlib import Path

import pytest

from gateway import hosted_rooms
from gateway.hosted_room_policy_checkpoint import HostedRoomPolicyCheckpoint


def _store_fds(db) -> list[str]:
    """Paths of this process's open descriptors for ``db`` and its ``-wal``/``-shm``."""
    if not sys.platform.startswith("linux"):
        pytest.skip("descriptor introspection needs /proc")
    prefix = Path(db).name
    found = []
    for fd in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if Path(target).name.startswith(prefix):
            found.append(target)
    return sorted(found)


def _create_room(db, room_id="room-1"):
    hosted_rooms.create_room(
        db,
        room_id=room_id,
        name="Release room",
        members=[{"profile": "default", "handle": "hermes"}],
        authority_gateway_id="gateway-a",
    )


def _link_record(**overrides):
    record = {
        "room_id": "room-1",
        "member_id": "member-1",
        "target_url": "https://example.invalid/webhook",
        "target_profile": "default",
        "grant": "{}",
        "catalog_json": "{}",
        "cancellation_scope_id": "scope-1",
        "trace_id": "trace-1",
        "transport_security": "strict",
        "status": "ready",
        "updated_at": 1.0,
    }
    record.update(overrides)
    return record


def _reject_write_transaction(*_args, **_kwargs):
    raise AssertionError("this read path must not open a write transaction")


def test_list_rooms_does_not_enter_a_write_transaction_or_prune(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    hosted_rooms.create_room(
        db,
        room_id="room-1",
        name="Release room",
        members=[{"profile": "default", "handle": "hermes"}],
        authority_gateway_id="gateway-a",
    )

    def reject_prune(*_args, **_kwargs):
        raise AssertionError("list_rooms must not prune retention state")

    monkeypatch.setattr(hosted_rooms, "_transaction", _reject_write_transaction)
    monkeypatch.setattr(
        hosted_rooms,
        "_prune_disbanded_rooms_locked",
        reject_prune,
    )

    rows = hosted_rooms.list_rooms(db)

    assert [row["room_id"] for row in rows] == ["room-1"]


def test_room_read_path_is_read_only_and_still_bootstraps_a_missing_store(tmp_path):
    """Reads never hold write capability, but a store that does not exist yet is still created."""
    db = tmp_path / "state.db"
    assert hosted_rooms.list_rooms(db) == []
    assert db.is_file()

    _create_room(db)
    conn = hosted_rooms._read_connection(db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE probe_write_attempt (value INTEGER)")
        assert conn.execute("SELECT COUNT(*) FROM hosted_rooms").fetchone()[0] == 1
    finally:
        conn.close()


def test_prune_takes_no_write_transaction_when_no_room_is_disbanded(tmp_path, monkeypatch):
    """Nothing to prune means no ``BEGIN IMMEDIATE`` on the shared store at backend boot."""
    db = tmp_path / "state.db"
    _create_room(db)

    monkeypatch.setattr(hosted_rooms, "_transaction", _reject_write_transaction)

    assert hosted_rooms.prune_disbanded_rooms(db) == 0

    monkeypatch.undo()
    hosted_rooms.disband_room(
        db, room_id="room-1", expected_gateway_id="gateway-a", expected_epoch=1, now=20
    )
    # The no-op fast path must not swallow real retention work.
    assert hosted_rooms.prune_disbanded_rooms(
        db, now=20 + hosted_rooms.DISBANDED_ROOM_RETENTION_SECONDS + 1
    ) == 1


def test_list_room_link_records_takes_no_write_transaction(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    _create_room(db)
    hosted_rooms.upsert_room_link_record(db, record=_link_record(), max_links=10)

    monkeypatch.setattr(hosted_rooms, "_transaction", _reject_write_transaction)

    rows = hosted_rooms.list_room_link_records(db)

    assert [(row["room_id"], row["member_id"]) for row in rows] == [("room-1", "member-1")]


def test_policy_checkpoint_holds_no_store_handle_after_an_operation(tmp_path):
    """``with conn:`` only commits: every checkpoint operation must close its connection."""
    db = tmp_path / "state.db"
    checkpoint = HostedRoomPolicyCheckpoint(db)
    _create_room(db)
    assert _store_fds(db) == []

    checkpoint.publication_exists(room_id="room-1", task_id="task-1", status="settled", execution_generation=1)
    checkpoint.events_for_task(room_id="room-1", source_event_seq=1)
    checkpoint.compact_completed(room_id="room-1")
    assert _store_fds(db) == []
