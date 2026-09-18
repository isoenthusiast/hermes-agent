"""A ``case`` pattern list is not a command list, and a refusal names why (#2).

Two contracts:

1. The benign read-only shape from issue #2 (sample another process's fds for SQLite handles) must
   pass. The guard's tokenizer split the ``case "$t" in /a/state.db|/a/state.db-wal)`` alternative
   list on ``|``, landed each literal pattern at command position, and the referenced-script walk
   then read an 843-MiB SQLite database as a "referenced script", failed closed on its size and
   refused the whole (read-only) command. Measured on the shipped guard: the walk yielded
   ``/home/edward/.hermes/state.db``, size 882827264, ``read_unsafe=True``, while the identical
   command written with glob patterns passed. A pattern list executes nothing, so it is not a
   script reference — and the bodies between ``)`` and ``;;`` must stay scanned.
2. A refusal must name the specific condition that failed — the matched text, the path and its
   size, or the exhausted budget — so an operator does not debug an action when the trigger is a
   string.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import cron.lifecycle_guard as lifecycle_guard

guard = lifecycle_guard.contains_gateway_lifecycle_command_or_referenced_script

_CAP = lifecycle_guard._MAX_REFERENCED_SCRIPT_BYTES


def _explain(command: str, **kwargs):
    """Indirection so the behavioural tests fail on the verdict, not on a missing name."""
    return lifecycle_guard.explain_gateway_lifecycle_refusal(command, **kwargs)


def _oversized_data_file(tmp_path: Path, name: str = "state.db") -> Path:
    """A data file too large for the referenced-script scanner, like the reporter's state.db."""
    data = tmp_path / name
    data.write_bytes(b"SQLite format 3\x00" + b"\x00" * (_CAP + 64))
    return data


def _benign_fd_sampling_loop(targets: list[str]) -> str:
    """The read-only shape from issue #2: sample another process's fds for SQLite handles."""
    return (
        "P=4082633\n"
        "i=0\n"
        'while [ "$i" -lt 45 ]; do\n'
        "  for fd in /proc/$P/fd/*; do\n"
        "    n=${fd##*/}\n"
        '    t=$(readlink "$fd" 2>/dev/null) || continue\n'
        '    case "$t" in\n'
        f"      {'|'.join(targets)})\n"
        '        echo "$n"\n'
        "        ;;\n"
        "    esac\n"
        "  done\n"
        "  i=$((i+1))\n"
        "done\n"
    )


def test_absolute_paths_in_a_case_list_are_not_referenced_scripts(tmp_path):
    """The benign read-only shape must pass: a case PATTERN list is not a command list."""
    data = _oversized_data_file(tmp_path)
    loop = _benign_fd_sampling_loop([str(data), f"{data}-wal", f"{data}-shm"])

    assert guard(loop, cwd=str(tmp_path)) is False


def test_case_pattern_list_is_never_read_as_a_script(tmp_path, monkeypatch):
    """The patterns are dropped before the walk, so the data file is never opened as a script."""
    data = _oversized_data_file(tmp_path)
    loop = _benign_fd_sampling_loop([str(data)])

    real_open = lifecycle_guard.os.open

    def reject_data_open(path, flags, *args, **kwargs):
        if str(path) == str(data):
            pytest.fail("lifecycle guard opened a case-list pattern as a referenced script")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(lifecycle_guard.os, "open", reject_data_open)

    assert guard(loop, cwd=str(tmp_path)) is False


def test_case_body_commands_are_still_scanned(tmp_path):
    """Dropping the pattern list must not blind the body: commands after `)` stay scanned."""
    hostile = tmp_path / "hostile.sh"
    hostile.write_text("hermes gateway restart\n", encoding="utf-8")
    hostile.chmod(0o755)

    assert guard(f'case "$t" in *) {hostile} ;; esac') is True


def test_case_words_in_argument_position_do_not_swallow_later_references(tmp_path):
    """`grep -n case in file` only mentions the keywords; dropping from there could hide a script."""
    hostile = tmp_path / "hostile.sh"
    hostile.write_text("hermes gateway restart\n", encoding="utf-8")
    hostile.chmod(0o755)

    assert guard(f"grep -n case in /dev/null; {hostile}") is True


def test_lifecycle_text_in_a_pattern_list_is_still_caught_by_the_raw_pass(tmp_path):
    """The blunt raw-text pass still sees the whole command, pattern lists included."""
    assert guard('case "$t" in *"hermes gateway restart"*) true ;; esac') is True


def test_executable_oversized_script_still_fails_closed_with_a_named_condition(tmp_path):
    """Fail-closed is kept where it belongs, and the refusal names the file and its size."""
    script = tmp_path / "restart-loop.sh"
    script.write_bytes(b"#!/bin/bash\n#" + b"x" * (_CAP + 64) + b"\n")
    script.chmod(0o755)

    refusal = _explain(str(script))

    assert refusal is not None
    assert refusal.condition == "oversized_referenced_script"
    assert str(script) in refusal.detail
    assert str(script.stat().st_size) in refusal.detail


def test_sourcing_a_non_executable_script_stays_scanned(tmp_path):
    """A sourced file needs no execute bit, so the dot-source branch must keep reading it."""
    clean = tmp_path / "env.sh"
    clean.write_text("export A=1\n", encoding="utf-8")
    assert guard(f". {clean}") is False

    hostile = tmp_path / "hostile.sh"
    hostile.write_text("hermes gateway restart\n", encoding="utf-8")
    assert guard(f". {hostile}") is True


def test_shell_invocation_of_a_non_executable_script_stays_scanned(tmp_path):
    """Same for an interpreter invocation: `bash <path>` needs no execute bit either."""
    hostile = tmp_path / "hostile.sh"
    hostile.write_text("hermes gateway restart\n", encoding="utf-8")
    assert guard(f"bash {hostile}") is True


def test_direct_command_refusal_names_the_matched_text():
    refusal = _explain("hermes gateway restart")

    assert refusal is not None
    assert refusal.condition == "gateway_lifecycle_command"
    assert "hermes gateway restart" in refusal.detail


def test_quoted_pattern_refusal_says_the_trigger_is_the_string():
    """Still blocked (the guard stays blunt), but the operator is told it is a string, not an action."""
    refusal = _explain('echo "hermes gateway restart" is what a human would run')

    assert refusal is not None
    assert refusal.condition == "gateway_lifecycle_text"
    assert "quoted" in refusal.detail


def test_budget_refusal_names_the_budget(monkeypatch):
    monkeypatch.setattr(lifecycle_guard, "_MAX_LIFECYCLE_SCAN_BYTES", 8)
    monkeypatch.setattr(lifecycle_guard, "_MAX_LIFECYCLE_SCAN_LINE_BYTES", 8)

    refusal = _explain("x" * 9)

    assert refusal is not None
    assert refusal.condition == "scan_budget_exhausted"
    assert "_MAX_LIFECYCLE_SCAN_BYTES" in refusal.detail


def test_check_gateway_lifecycle_names_the_unscannable_script(tmp_path):
    """The cron path must not claim a lifecycle command when the real cause is an unscannable file."""
    script = tmp_path / "nightly.sh"
    script.write_bytes(b"#!/bin/bash\n#" + b"x" * (_CAP + 64) + b"\n")

    with pytest.raises(lifecycle_guard.GatewayLifecycleBlocked) as excinfo:
        lifecycle_guard.check_gateway_lifecycle("nightly job", str(script))

    message = str(excinfo.value)
    assert str(script) in message
    assert str(script.stat().st_size) in message
