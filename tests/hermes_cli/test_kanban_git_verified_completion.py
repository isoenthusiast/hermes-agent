"""Tests for the git-verified completion guardrail (post-mortem 2026-09-07).

Covers the two dispatch/completion levers that stop a repo-editing card from
reaching `done` with uncommitted work or from sharing a checkout:

  1. Worktree isolation (`_maybe_upgrade_repo_scratch`): a card dispatched as
     `scratch` whose body references a git repo is auto-upgraded to an isolated
     per-task worktree, so two concurrent cards never share one working tree.
  2. Mandatory skill wiring (`_default_spawn`): every default-spawned worker
     force-loads `kanban-git-verified-completion`, so it commits + pushes before
     `kanban_complete` (avoiding the kernel-level gate rejection).

The dispatch machinery moved to ``hermes_cli.kanban_db_dispatch`` in the
Sep 2026 decomposition, so this file imports it as ``kbd`` (and the connect
layer as ``kbc``) rather than reaching them through ``kanban_db``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _make_git_repo(base: Path, name: str = "repo") -> str:
    repo = base / name
    repo.mkdir(parents=True, exist_ok=True)

    def git(*args, **kw):
        return subprocess.run(
            ["git"] + list(args), cwd=repo, capture_output=True, text=True, **kw
        )

    git("init", "-q")
    git("config", "user.email", "t@t.t")
    git("config", "user.name", "t")
    return str(repo)


def _mk(conn, *, title="card", body=None, workspace_kind="scratch", **kw):
    return kb.create_task(
        conn, title=title, body=body,
        workspace_kind=workspace_kind, assignee="conan", **kw,
    )


def test_repo_referencing_scratch_card_upgraded_to_worktree(kanban_home, tmp_path):
    """Acceptance: a card that edits a repo must never be dispatched as a shared
    `scratch` checkout. `_maybe_upgrade_repo_scratch` auto-upgrades it to an
    isolated per-task worktree."""
    repo = _make_git_repo(tmp_path)
    conn = kbc.connect()
    try:
        tid = _mk(conn, body=f"implement the fix in {repo}/src/main.py")
        task = kb.get_task(conn, tid)
        assert task.workspace_kind == "scratch"

        upgraded = kbd._maybe_upgrade_repo_scratch(conn, task, board=None)

        assert upgraded.workspace_kind == "worktree"
        # The worktree must anchor on the actual repo the card edits.
        assert Path(upgraded.workspace_path).resolve() == Path(repo).resolve()
        assert upgraded.branch_name == f"wt/{tid}"

        # The upgrade is observable in the event log.
        evs = kb.list_events(conn, tid)
        kinds = [e.kind for e in evs]
        assert "workspace_upgraded_to_worktree" in kinds
    finally:
        conn.close()


def test_non_repo_scratch_card_left_alone_with_warning(kanban_home, tmp_path):
    """A scratch card that references no git repo stays scratch (we cannot
    isolate on a repo that doesn't exist), but the hazard is surfaced as an
    event so an operator can pin it to a project if it edits a repo."""
    conn = kbc.connect()
    try:
        tid = _mk(conn, body="update the SSOT rules per the incident")
        task = kb.get_task(conn, tid)

        result = kbd._maybe_upgrade_repo_scratch(conn, task, board=None)

        assert result.workspace_kind == "scratch"
        evs = kb.list_events(conn, tid)
        kinds = [e.kind for e in evs]
        assert "scratch_repo_reference_warning" in kinds
    finally:
        conn.close()


def test_default_spawn_force_loads_git_verify_skill(kanban_home, monkeypatch):
    """Acceptance: the mandatory git-verified-completion skill is wired into the
    default dispatch path — even a worker with no task.skills gets it."""
    repo = _make_git_repo(kanban_home.parent)
    conn = kbc.connect()
    try:
        tid = kb.create_task(
            conn, title="spawn", assignee="conan",
            body=f"do it in {repo}", workspace_kind="scratch",
        )
        task = kb.get_task(conn, tid)
    finally:
        conn.close()

    # Task was upgraded by dispatch policy? For the spawn test we just need a
    # task object; the skill wiring is independent of workspace_kind.
    assert task.id == tid

    captured = {}

    class _FakeProc:
        pid = 4242

    real_popen = subprocess.Popen

    def fake_popen(args, **kw):
        captured["args"] = list(args)
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    # Resolve the assignee's profile without touching a real one: these are
    # imported inside `_default_spawn` from hermes_cli.profiles.
    monkeypatch.setattr(
        "hermes_cli.profiles.normalize_profile_name", lambda name: name
    )
    monkeypatch.setattr(
        "hermes_cli.profiles.resolve_profile_env", lambda name: str(kanban_home)
    )

    try:
        pid = kbd._default_spawn(task, workspace=repo, board=None)
    finally:
        subprocess.Popen = real_popen

    assert pid == 4242
    args = captured["args"]
    assert "--skills" in args
    i = args.index("--skills")
    assert i + 1 < len(args)
    assert args[i + 1] == kbd.DEFAULT_GIT_VERIFY_SKILL


# ---------------------------------------------------------------------------
# Anchor source: a board-default anchor is a guess, and must say so
# (card t_31f36f49 — a card targeting gamified-plant was anchored, via the
# board default_workdir fallback, to the sibling repo sams-app).
# ---------------------------------------------------------------------------


def _payloads(conn, tid, kind):
    return [e.payload for e in kb.list_events(conn, tid) if e.kind == kind]


def test_prose_anchor_records_anchor_source(kanban_home, tmp_path):
    """Invariant: the upgrade event names the lever that picked the repo."""
    repo = _make_git_repo(tmp_path)
    conn = kbc.connect()
    try:
        tid = _mk(conn, body=f"implement the fix in {repo}/src/main.py")
        task = kb.get_task(conn, tid)

        kbd._maybe_upgrade_repo_scratch(conn, task, board=None)

        payloads = _payloads(conn, tid, "workspace_upgraded_to_worktree")
        assert len(payloads) == 1
        assert payloads[0]["anchor_source"] == kbd.ANCHOR_SOURCE_PROSE
        assert payloads[0]["repo"] == str(Path(repo).resolve())
        # The prose itself named the repo — nothing to flag.
        assert _payloads(conn, tid, "workspace_anchor_fallback") == []
    finally:
        conn.close()


def test_board_default_anchor_without_prose_confirmation_is_flagged(kanban_home, tmp_path):
    """The regression card t_31f36f49 measured: prose names no repo path and no
    repo name, so the board's default_workdir decides — the worktree lands on a
    sibling repo the card never mentions. That guess must be visible on the
    board, not just in the workspace path."""
    repo = _make_git_repo(tmp_path)
    kb.create_board("sibling-probe", default_workdir=repo)
    conn = kbc.connect(board="sibling-probe")
    try:
        tid = _mk(conn, body="Apply the same .worktrees/ ignore rule to gamified-plant")
        task = kb.get_task(conn, tid)

        upgraded = kbd._maybe_upgrade_repo_scratch(conn, task, board="sibling-probe")

        assert upgraded.workspace_kind == "worktree"
        assert Path(upgraded.workspace_path).resolve() == Path(repo).resolve()
        payloads = _payloads(conn, tid, "workspace_upgraded_to_worktree")
        assert payloads[0]["anchor_source"] == kbd.ANCHOR_SOURCE_BOARD_DEFAULT

        flagged = _payloads(conn, tid, "workspace_anchor_fallback")
        assert len(flagged) == 1
        assert flagged[0]["repo"] == str(Path(repo).resolve())
        assert flagged[0]["anchor_source"] == kbd.ANCHOR_SOURCE_BOARD_DEFAULT
        assert flagged[0]["board"] == "sibling-probe"
        assert flagged[0]["card_names_repo_in_prose"] is False
        assert flagged[0]["sibling_repo_named_in_prose"] is None
    finally:
        conn.close()


def test_sibling_repo_named_in_prose_is_flagged(kanban_home, tmp_path):
    """The measured shape of t_ef94e814: the card is about gamified-plant and
    mentions the board repo in passing (``sams-app.sh``). Anchoring on sams-app
    must still be flagged — a sibling repo is what the card is actually about,
    and the passing mention is not confirmation."""
    ws = tmp_path / "workspace"
    sams = _make_git_repo(ws, "sams-app")
    _make_git_repo(ws, "gamified-plant")
    kb.create_board("sams-probe", default_workdir=sams)
    conn = kbc.connect(board="sams-probe")
    try:
        tid = _mk(
            conn,
            body="Port the ignore rules to gamified-plant, same sweep as sams-app.sh",
        )
        task = kb.get_task(conn, tid)

        kbd._maybe_upgrade_repo_scratch(conn, task, board="sams-probe")

        flagged = _payloads(conn, tid, "workspace_anchor_fallback")
        assert len(flagged) == 1
        assert flagged[0]["repo"] == str(Path(sams).resolve())
        assert flagged[0]["sibling_repo_named_in_prose"] == "gamified-plant"
        # "sams-app.sh" is not the repo name — the prose never names the anchor.
        assert flagged[0]["card_names_repo_in_prose"] is False
    finally:
        conn.close()


def test_prose_path_gone_from_disk_is_flagged(kanban_home, tmp_path):
    """A card naming a *deleted* sibling worktree inside a repo resolves through
    the enclosing repo, so the worktree lands on a repo the card never meant.
    Measured on t_31f36f49 itself, whose prose pointed at
    ``.../sams-app/.worktrees/t_ef94e814`` after that worktree was gone."""
    repo = _make_git_repo(tmp_path)
    (Path(repo) / ".worktrees").mkdir(exist_ok=True)
    gone = Path(repo) / ".worktrees" / "t_ef94e814"
    assert not gone.exists()
    conn = kbc.connect(board="default")
    try:
        tid = _mk(conn, body=f"Continue the audit in {gone}/docs (checkout was there)")
        task = kb.get_task(conn, tid)

        kbd._maybe_upgrade_repo_scratch(conn, task, board="default")

        # The anchor is still the enclosing repo — but it says why it is a guess.
        upgraded = _payloads(conn, tid, "workspace_upgraded_to_worktree")
        assert upgraded[0]["anchor_source"] == kbd.ANCHOR_SOURCE_PROSE
        assert upgraded[0]["repo"] == str(Path(repo).resolve())
        unverified = _payloads(conn, tid, "workspace_anchor_unverified")
        assert len(unverified) == 1
        assert unverified[0]["prose_worktree"] == str(gone)
        assert unverified[0]["repo"] == str(Path(repo).resolve())
    finally:
        conn.close()


def test_live_prose_path_is_not_flagged(kanban_home, tmp_path):
    """A file the card is about to create is a normal prose path, not a hazard —
    flagging it would be noise an operator learns to ignore."""
    repo = _make_git_repo(tmp_path)
    (Path(repo) / "docs").mkdir(exist_ok=True)
    conn = kbc.connect(board="default")
    try:
        tid = _mk(conn, body=f"Edit {repo}/docs/new-file.md in place")
        task = kb.get_task(conn, tid)

        kbd._maybe_upgrade_repo_scratch(conn, task, board="default")

        assert _payloads(conn, tid, "workspace_anchor_unverified") == []
        assert _payloads(conn, tid, "workspace_anchor_fallback") == []
    finally:
        conn.close()


def test_existing_worktree_path_is_not_flagged(kanban_home, tmp_path):
    """A card that names a worktree which *is* on disk resolves cleanly."""
    repo = _make_git_repo(tmp_path)
    live = Path(repo) / ".worktrees" / "t_alive"
    live.mkdir(parents=True)
    conn = kbc.connect(board="default")
    try:
        tid = _mk(conn, body=f"The checkout is at {live} — keep going")
        task = kb.get_task(conn, tid)

        kbd._maybe_upgrade_repo_scratch(conn, task, board="default")

        assert _payloads(conn, tid, "workspace_anchor_unverified") == []
    finally:
        conn.close()


def test_board_default_anchor_confirmed_by_prose_is_not_flagged(kanban_home, tmp_path):
    """A board-default anchor is fine when the card's own prose names that repo;
    flagging it would be noise an operator learns to ignore."""
    repo = _make_git_repo(tmp_path)
    kb.create_board("self-probe", default_workdir=repo)
    conn = kbc.connect(board="self-probe")
    try:
        tid = _mk(conn, body="Update the repo README and .gitignore")
        task = kb.get_task(conn, tid)

        kbd._maybe_upgrade_repo_scratch(conn, task, board="self-probe")

        payloads = _payloads(conn, tid, "workspace_upgraded_to_worktree")
        assert payloads[0]["anchor_source"] == kbd.ANCHOR_SOURCE_BOARD_DEFAULT
        assert _payloads(conn, tid, "workspace_anchor_fallback") == []
    finally:
        conn.close()


def test_prose_names_repo_is_word_bounded(tmp_path):
    """Basename matching must not fire on a longer or dotted name."""
    repo = Path(tmp_path) / "sams-app"
    assert kbd._prose_names_repo("fix sams-app now", None, repo)
    assert kbd._prose_names_repo(None, "SAMS-APP deploy notes", repo)
    assert not kbd._prose_names_repo("fix sams-app-old now", None, repo)
    assert not kbd._prose_names_repo("see x.sams-app for detail", None, repo)


# ---------------------------------------------------------------------------
# Two path shapes the prose anchor used to miss (card t_7ad7badc):
#   A. the token names an *existing file* — the commonest card shape
#      ("edit <repo>/.gitignore"); `git -C <file>` fails, so the card fell
#      through to the board default and anchored a worktree on a sibling repo.
#   B. the token is markdown inline code — the closing backtick was captured
#      into the path, so the repo path in the body resolved to nothing.
# ---------------------------------------------------------------------------


def test_prose_path_naming_an_existing_file_anchors_its_repo(kanban_home, tmp_path):
    """Invariant: a path token naming a file that exists resolves through the
    file's directory to the repo that holds it, and beats the board default."""
    ws = tmp_path / "workspace"
    plant = _make_git_repo(ws, "gamified-plant")
    sams = _make_git_repo(ws, "sams-app")
    (Path(plant) / ".gitignore").write_text("node_modules\n")
    kb.create_board("file-probe", default_workdir=sams)
    conn = kbc.connect(board="file-probe")
    try:
        tid = _mk(
            conn,
            body=f"Apply the same .worktrees/ ignore rule to {plant}/.gitignore",
        )
        task = kb.get_task(conn, tid)

        upgraded = kbd._maybe_upgrade_repo_scratch(conn, task, board="file-probe")

        assert Path(upgraded.workspace_path).resolve() == Path(plant).resolve()
        payloads = _payloads(conn, tid, "workspace_upgraded_to_worktree")
        assert payloads[0]["anchor_source"] == kbd.ANCHOR_SOURCE_PROSE
        assert payloads[0]["repo"] == str(Path(plant).resolve())
        # The prose itself named the repo — the board default never decided.
        assert _payloads(conn, tid, "workspace_anchor_fallback") == []
    finally:
        conn.close()


def test_prose_path_naming_a_file_that_does_not_exist_still_anchors(kanban_home, tmp_path):
    """Regression: a path the card is about to *create* resolves through its
    nearest existing ancestor to the same repo, as it always did."""
    repo = _make_git_repo(tmp_path)
    anchor = kbd._card_repo_anchor("t", f"create {repo}/src/main.py", board=None)
    assert anchor is not None
    assert anchor.source == kbd.ANCHOR_SOURCE_PROSE
    assert Path(anchor.repo).resolve() == Path(repo).resolve()


def test_backtick_wrapped_prose_path_anchors_its_repo(kanban_home, tmp_path):
    """Invariant: markdown inline code must not capture its closing backtick.
    t_c7131f54 was warned as "card body references no resolvable git repo"
    while the repo path sat in the body wrapped in backticks."""
    repo = _make_git_repo(tmp_path)
    anchor = kbd._card_repo_anchor(
        "t", f"- Repo = `{repo}`, has origin but no worktrees", board=None,
    )
    assert anchor is not None
    assert anchor.source == kbd.ANCHOR_SOURCE_PROSE
    assert Path(anchor.repo).resolve() == Path(repo).resolve()
    assert anchor.prose_path == str(repo)


def test_backtick_wrapped_path_beats_the_board_default(kanban_home, tmp_path):
    """End-to-end: a backtick-wrapped path naming a sibling repo wins over the
    board default, and is not reported as an unconfirmed guess."""
    ws = tmp_path / "workspace"
    sams = _make_git_repo(ws, "sams-app")
    littlepanda = _make_git_repo(ws, "littlepanda")
    kb.create_board("sams-probe", default_workdir=sams)
    conn = kbc.connect(board="sams-probe")
    try:
        tid = _mk(conn, body=f"- Repo = `{littlepanda}`, has origin but no worktrees")
        task = kb.get_task(conn, tid)

        upgraded = kbd._maybe_upgrade_repo_scratch(conn, task, board="sams-probe")

        assert Path(upgraded.workspace_path).resolve() == Path(littlepanda).resolve()
        payloads = _payloads(conn, tid, "workspace_upgraded_to_worktree")
        assert payloads[0]["anchor_source"] == kbd.ANCHOR_SOURCE_PROSE
        assert payloads[0]["repo"] == str(Path(littlepanda).resolve())
        assert _payloads(conn, tid, "workspace_anchor_fallback") == []
    finally:
        conn.close()


def test_unreadable_absolute_path_in_prose_does_not_kill_the_tick(kanban_home, tmp_path):
    """Invariant: resolving the anchor must never raise on an unstat-able path.

    A card body may quote a path under a directory this process cannot traverse
    (`/root/.hermes/...` from a card about credentials). The walk-up exists only
    to find the nearest existing ancestor — an unreadable path must advance to
    its parent, not abort the dispatcher's whole tick. Measured: t_3c7928ec,
    where `Path.is_dir()` raised EACCES inside `_card_repo_anchor`, the tick died
    before any worker spawned, and the card re-claimed every 15 min forever.
    """
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "google_token.json").write_text("{}")
    locked.chmod(0o000)
    try:
        # No repo is reachable from here, so the correct answer is "no anchor" —
        # but the decisive part is that this returns at all instead of raising.
        assert kbd._card_repo_anchor(
            "t", f"the dispatcher stats {locked}/google_token.json and dies", board=None,
        ) is None
    finally:
        locked.chmod(0o700)


def test_scratch_card_quoting_an_unreadable_path_stays_scratch(kanban_home, tmp_path):
    """End-to-end: the upgrade path survives the same card shape intact."""
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "google_token.json").write_text("{}")
    locked.chmod(0o000)
    try:
        conn = kbc.connect()
        try:
            tid = _mk(conn, body=f"see {locked}/google_token.json for the token")
            task = kb.get_task(conn, tid)
            upgraded = kbd._maybe_upgrade_repo_scratch(conn, task, board=None)
            assert upgraded.workspace_kind == "scratch"
        finally:
            conn.close()
    finally:
        locked.chmod(0o700)


# ---------------------------------------------------------------------------
# Title scope vs prose order (card t_31a9fee9). Two cards dispatched 2026-09-18
# anchored on a repo they do not edit, because the first *resolvable* prose path
# token won: t_c788df73 (`littlepanda: ...`) landed on
# /srv/hermes/workspaces/ellen-production, and t_eaa1a1f7 (`ellen-production: ...`)
# landed on /home/edward/.orrery — both titles open with the repo they edit, so
# the title's scope word is the lever that gets these right.
# ---------------------------------------------------------------------------


def test_title_scope_beats_the_first_prose_token(kanban_home, tmp_path):
    """Frame-for-frame t_c788df73: the card is about littlepanda and quotes the
    *retired* ellen-production path first, so the first resolvable token names a
    repo the card does not own. The title's `littlepanda:` scope must win, and the
    displacement must be recorded."""
    ws = tmp_path / "workspace"
    ellen = _make_git_repo(ws, "ellen-production")
    panda = _make_git_repo(ws, "littlepanda")
    gone = tmp_path / "gone" / "ellen-production"      # the retired path: not on disk
    assert not gone.exists()
    conn = kbc.connect()
    try:
        tid = _mk(
            conn,
            title=(
                "littlepanda: load_quizzes.py + make_book_frames.py still reach "
                f"into the retired {gone} path"
            ),
            body=(
                f"- load_quizzes.py:8 -> {ellen}/platform/static/book_frames\n"
                f"- make_book_frames.py:8 -> {panda}/packs\n"
            ),
        )
        task = kb.get_task(conn, tid)

        upgraded = kbd._maybe_upgrade_repo_scratch(conn, task, board=None)

        assert Path(upgraded.workspace_path).resolve() == Path(panda).resolve()
        payloads = _payloads(conn, tid, "workspace_upgraded_to_worktree")
        assert payloads[0]["anchor_source"] == kbd.ANCHOR_SOURCE_PROSE
        assert payloads[0]["repo"] == str(Path(panda).resolve())

        overrides = _payloads(conn, tid, "workspace_anchor_scope_override")
        assert len(overrides) == 1
        assert overrides[0]["repo"] == str(Path(panda).resolve())
        assert overrides[0]["title_scope"] == "littlepanda"
        assert overrides[0]["first_prose_candidate"] == str(Path(ellen).resolve())
        assert overrides[0]["first_prose_path"] == f"{ellen}/platform/static/book_frames"
    finally:
        conn.close()


def test_scope_word_naming_no_candidate_keeps_the_prose_order(kanban_home, tmp_path):
    """A lane word that is not a repo name (`box:`, `fleet:`, `conan:`) must change
    nothing — prose order still decides, and no override is reported."""
    ws = tmp_path / "workspace"
    sams = _make_git_repo(ws, "sams-app")
    plant = _make_git_repo(ws, "gamified-plant")
    conn = kbc.connect()
    try:
        tid = _mk(
            conn,
            title="box: audit both apps",
            body=f"start in {sams}/src, then {plant}/src",
        )
        task = kb.get_task(conn, tid)

        upgraded = kbd._maybe_upgrade_repo_scratch(conn, task, board=None)

        assert Path(upgraded.workspace_path).resolve() == Path(sams).resolve()
        assert _payloads(conn, tid, "workspace_anchor_scope_override") == []
    finally:
        conn.close()


def test_scope_word_matches_a_repo_name_only_as_a_whole_word(kanban_home, tmp_path):
    """`littlepanda-old:` must not select `littlepanda` — the same word-boundary
    rule `_prose_names_repo` already uses for `sams-app-old` vs `sams-app`."""
    ws = tmp_path / "workspace"
    panda = _make_git_repo(ws, "littlepanda")
    other = _make_git_repo(ws, "othello-trainer-concept")
    conn = kbc.connect()
    try:
        tid = _mk(
            conn,
            title=f"littlepanda-old: sweep {other}",
            body=f"start with {other}/a.py, then {panda}/b.py",
        )
        task = kb.get_task(conn, tid)

        upgraded = kbd._maybe_upgrade_repo_scratch(conn, task, board=None)

        assert Path(upgraded.workspace_path).resolve() == Path(other).resolve()
        assert _payloads(conn, tid, "workspace_anchor_scope_override") == []
    finally:
        conn.close()


def test_scope_matching_the_first_candidate_is_not_an_override(kanban_home, tmp_path):
    """When the title's scope word names the repo prose order already picked, the
    pick is unchanged and nothing is flagged — the override event is for a
    displacement only, not for agreement."""
    ws = tmp_path / "workspace"
    panda = _make_git_repo(ws, "littlepanda")
    ellen = _make_git_repo(ws, "ellen-production")
    conn = kbc.connect()
    try:
        tid = _mk(
            conn,
            title=f"littlepanda: sweep {panda} and {ellen}",
            body=f"start in {panda}/a.py, then {ellen}/b.py",
        )
        task = kb.get_task(conn, tid)

        upgraded = kbd._maybe_upgrade_repo_scratch(conn, task, board=None)

        assert Path(upgraded.workspace_path).resolve() == Path(panda).resolve()
        assert _payloads(conn, tid, "workspace_anchor_scope_override") == []
    finally:
        conn.close()


def test_scope_match_stops_at_the_first_same_named_clone(kanban_home, tmp_path):
    """The same repo often exists at more than one path (canonical checkout plus a
    bot's `.hermes/profiles/<bot>/workspace/<repo>` clone). When the prose-order
    pick already carries the title's scope name, the rule must not scan past it to
    a later clone — measured on real cards t_4996a046 and t_4b45bd45, where a
    look-past loop moved the anchor from the canonical `littlepanda` to a clone."""
    ws = tmp_path / "workspace"
    canonical = _make_git_repo(ws, "littlepanda")
    clone = _make_git_repo(tmp_path / "bots" / "ellen", "littlepanda")
    assert Path(canonical).name == Path(clone).name == "littlepanda"
    conn = kbc.connect()
    try:
        tid = _mk(
            conn,
            title="littlepanda: commit the two-file retired-path fix",
            body=f"draft in {canonical}/packs, mirror into {clone}/packs",
        )
        task = kb.get_task(conn, tid)

        upgraded = kbd._maybe_upgrade_repo_scratch(conn, task, board=None)

        assert Path(upgraded.workspace_path).resolve() == Path(canonical).resolve()
        assert _payloads(conn, tid, "workspace_anchor_scope_override") == []
    finally:
        conn.close()


def test_anchor_records_the_displaced_candidate(kanban_home, tmp_path):
    """Unit shape of the lever: the winning token is kept as ``prose_path`` and the
    displaced first candidate as ``overrode_repo``/``overrode_prose_path``."""
    ws = tmp_path / "workspace"
    ellen = _make_git_repo(ws, "ellen-production")
    panda = _make_git_repo(ws, "littlepanda")

    overridden = kbd._card_repo_anchor(
        "littlepanda: the sweep",
        f"first {ellen}/platform, then {panda}/packs",
        board=None,
    )
    assert overridden is not None
    assert Path(overridden.repo).resolve() == Path(panda).resolve()
    assert overridden.prose_path == f"{panda}/packs"
    assert overridden.title_scope == "littlepanda"
    assert Path(overridden.overrode_repo).resolve() == Path(ellen).resolve()
    assert overridden.overrode_prose_path == f"{ellen}/platform"

    # Control: no scope word, so the first resolvable token stands and nothing is
    # recorded as an override.
    plain = kbd._card_repo_anchor("the sweep", f"first {ellen}/platform, then {panda}/packs", board=None)
    assert plain is not None
    assert Path(plain.repo).resolve() == Path(ellen).resolve()
    assert plain.title_scope is None
    assert plain.overrode_repo is None


def test_title_scope_word_reads_the_leading_lane(kanban_home, tmp_path):
    """The scope is the word that abuts the title's first colon, any case."""
    assert kbd._title_scope_word("littlepanda: load_quizzes.py") == "littlepanda"
    assert kbd._title_scope_word("  ellen-production: 105 files") == "ellen-production"
    assert kbd._title_scope_word("LittlePanda: sweep") == "LittlePanda"
    # A date-prefixed lane (`fleet-delta 2026-09-18: ...`) declares no scope word.
    assert kbd._title_scope_word("fleet-delta 2026-09-18: the verifier gate") is None
    assert kbd._title_scope_word("no colon here") is None
    assert kbd._title_scope_word(None) is None

