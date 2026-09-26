"""Offline tests for live-status-writer's FLEET_STATUS.md rendering.

No network, no git: build_markdown() is fed a /shadow/status-shaped doc.

L1  alias ghost merged  -- REGRESSION (2026-09-26): the pre-identity-fix
                           `DLAPTOP-4.local` row (state=running, V3-EXQ-906c,
                           last seen 2026-08-09) rendered beside the live
                           `DLAPTOP` row, so the snapshot said the laptop was
                           running an experiment that finished weeks ago.
L2  freshest wins       -- if the ALIAS row is the fresher one, it is kept
L3  cloud fleet intact  -- ree-cloud-1..5 and <box>-metaworker never collapse
L4  stale label         -- a stale row that still says running is flagged
L5  non-vacuity         -- the table really has one line per expected machine

ASCII-only.
"""

import importlib.util
import os
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location(
    "live_status_writer", os.path.join(HERE, "live-status-writer.py"))
lsw = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lsw)

NOW = datetime(2026, 9, 26, 15, 30, tzinfo=timezone.utc)


def row(machine, state, last_seen, exq=None, lifecycle="live"):
    return {"machine": machine, "state": state, "current_exq": exq,
            "last_seen": last_seen, "lifecycle_state": lifecycle,
            "progress": {}}


# The two Mac rows exactly as the coordinator returned them on 2026-09-26.
LIVE_MAC = row("DLAPTOP", "offline", "2026-09-25T16:10:30Z",
               lifecycle="gracefully_offline")
GHOST_MAC = row("DLAPTOP-4.local", "running", "2026-08-09T19:49:28Z",
                exq="V3-EXQ-906c", lifecycle="stale")
CLOUD_2 = row("ree-cloud-2", "running", "2026-09-26T15:29:50Z",
              exq="V3-EXQ-1109")


def table_rows(md):
    """Machine names in the Workers table, in order."""
    out, in_table = [], False
    for line in md.splitlines():
        if line.startswith("| Machine |"):
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            if line.startswith("|---"):
                continue
            out.append(line.split("|")[1].strip())
    return out


def render(machines):
    return lsw.build_markdown({"machines": machines}, [], NOW)


def test_l1_alias_ghost_is_merged_into_live_row():
    md = render([LIVE_MAC, GHOST_MAC, CLOUD_2])
    assert table_rows(md) == ["DLAPTOP", "ree-cloud-2"]
    assert "V3-EXQ-906c" not in md.split("_Older rows")[0]
    assert "## Workers (2 total, 1 running)" in md
    # ...and the snapshot says what it hid rather than dropping it silently.
    assert "DLAPTOP-4.local" in md and "-> DLAPTOP" in md


def test_l2_fresher_alias_row_wins():
    fresh_alias = row("DLAPTOP-4.local", "running", "2026-09-26T15:20:00Z",
                      exq="V3-EXQ-2000")
    md = render([LIVE_MAC, fresh_alias])
    assert table_rows(md) == ["DLAPTOP-4.local"]
    assert "V3-EXQ-2000" in md


def test_l3_cloud_fleet_and_metaworker_rows_do_not_collapse():
    names = ["ree-cloud-1", "ree-cloud-2", "ree-cloud-3", "ree-cloud-4",
             "ree-cloud-4-metaworker", "ree-cloud-5", "ree-worker-1"]
    md = render([row(n, "offline", "2026-09-26T15:00:00Z") for n in names])
    assert table_rows(md) == sorted(names)
    assert "_Older rows" not in md


def test_l4_stale_running_row_is_flagged_not_rewritten():
    stale = row("ree-cloud-3", "running", "2026-09-20T00:00:00Z",
                exq="V3-EXQ-1", lifecycle="stale")
    md = render([stale])
    assert "| ree-cloud-3 | running (stale) |" in md
    offline_stale = row("ree-cloud-3", "offline", "2026-09-20T00:00:00Z",
                        lifecycle="stale")
    assert "| ree-cloud-3 | offline |" in render([offline_stale])


def test_l5_merge_is_not_vacuous():
    # Guard the guard: the machine_identity import must have succeeded, or
    # merge_alias_rows silently passes rows through and L1 can only fail.
    assert lsw.machine_identity is not None
    rows, hidden = lsw.merge_alias_rows([LIVE_MAC, GHOST_MAC, CLOUD_2])
    assert [r["machine"] for r in rows] == ["DLAPTOP", "ree-cloud-2"]
    assert hidden == [("DLAPTOP-4.local", "DLAPTOP", "2026-08-09T19:49:28Z")]
