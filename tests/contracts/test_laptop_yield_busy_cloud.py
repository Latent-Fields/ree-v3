"""Contract: the laptop yields an 'any'-affinity item to a cloud worker ONLY
when that worker is fresh AND idle -- never to an alive-but-busy worker.

Regression for the 2026-06-03 incident: DLAPTOP-4.local sat idle on a depth-9
queue because _should_yield_to_cloud yielded every item to ree-cloud-1 purely
because its heartbeat was fresh, even though cloud-1..4 were each saturated on
a long behavioural run. With the whole fleet busy and no worker free to claim,
the five pending items starved. The yield decision now requires the cloud
worker to be idle (state == 'idle', no current_exq), so a busy fleet falls
through to the laptop running the item locally.
"""
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import experiment_runner as er  # noqa: E402


def _hb_dir(base: Path) -> Path:
    d = base / "evidence" / "experiments" / "runner_heartbeats"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_hb(base: Path, host: str, *, age_min: float, state: str,
              current_exq):
    tick = datetime.now(timezone.utc) - timedelta(minutes=age_min)
    payload = {
        "machine": host,
        "last_tick_utc": tick.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "state": state,
        "current_exq": current_exq,
    }
    (_hb_dir(base) / f"{host}.json").write_text(json.dumps(payload))


ITEM = {"queue_id": "V3-EXQ-TEST", "machine_affinity": "any"}
FRESH = 35  # minutes, matches the default --laptop-yield-freshness-min


def test_fresh_idle_cloud_triggers_yield(tmp_path):
    _write_hb(tmp_path, "ree-cloud-1", age_min=1, state="idle",
              current_exq=None)
    yield_, host = er._should_yield_to_cloud(ITEM, FRESH, tmp_path)
    assert yield_ is True
    assert host == "ree-cloud-1"


def test_fresh_but_running_cloud_does_not_yield(tmp_path):
    # The exact incident shape: every cloud worker alive but mid-run.
    for host in ("ree-cloud-1", "ree-cloud-2", "ree-cloud-3", "ree-cloud-4"):
        _write_hb(tmp_path, host, age_min=1, state="running",
                  current_exq="V3-EXQ-463b")
    yield_, host = er._should_yield_to_cloud(ITEM, FRESH, tmp_path)
    assert yield_ is False
    assert host is None


def test_idle_state_but_lingering_current_exq_does_not_yield(tmp_path):
    # Defensive: state says idle but current_exq still populated -> not free.
    _write_hb(tmp_path, "ree-cloud-1", age_min=1, state="idle",
              current_exq="V3-EXQ-999")
    yield_, _ = er._should_yield_to_cloud(ITEM, FRESH, tmp_path)
    assert yield_ is False


def test_stale_idle_cloud_does_not_yield(tmp_path):
    _write_hb(tmp_path, "ree-cloud-1", age_min=FRESH + 10, state="idle",
              current_exq=None)
    yield_, _ = er._should_yield_to_cloud(ITEM, FRESH, tmp_path)
    assert yield_ is False


def test_mixed_fleet_yields_to_the_one_idle_worker(tmp_path):
    _write_hb(tmp_path, "ree-cloud-1", age_min=1, state="running",
              current_exq="V3-EXQ-610e")
    _write_hb(tmp_path, "ree-cloud-2", age_min=1, state="running",
              current_exq="V3-EXQ-460b")
    _write_hb(tmp_path, "ree-cloud-3", age_min=1, state="idle",
              current_exq=None)
    yield_, host = er._should_yield_to_cloud(ITEM, FRESH, tmp_path)
    assert yield_ is True
    assert host == "ree-cloud-3"


def test_pinned_item_never_yields(tmp_path):
    _write_hb(tmp_path, "ree-cloud-1", age_min=1, state="idle",
              current_exq=None)
    pinned = {"queue_id": "V3-EXQ-PIN", "machine_affinity": "DLAPTOP-4.local"}
    yield_, _ = er._should_yield_to_cloud(pinned, FRESH, tmp_path)
    assert yield_ is False
