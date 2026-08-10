"""Event-window inspectability browser for 906-lineage full-life episode logs.

Two independent same-lineage reviews of the V3-EXQ-906 "Fishtank" full-life
observational runs each proposed the same small tool from two angles, and each
had already built and used an ad hoc, uncommitted, one-off version of it:

  - organism_lifespan_development_review_906_lineage_2026-08-10.md Section 10
    item 5: a "top-N surprise-peak inspectability tool" -- extract the top-N
    residue_surprise peaks from a run and show the surrounding trajectory, to
    test whether REE's surprise responses track its own prediction error
    rather than an experimenter's judgement of event importance. (The ad hoc
    version found that top-8 peaks in both 906b/906c are dominated by
    reef-boundary crossings and resource-consumption clusters, with zero of 16
    combined top peaks corresponding to experimenter-injected
    limb_damage/external_hazard/world_rule_shift events.)
  - sleep_transition_investigation_906_lineage_2026-08-10.md Section 8: a
    sibling tool using the same window-extraction operation but a different
    event-selection rule -- N steps either side of every episode boundary
    where the next episode's `sleep_cycle_fired_before_this_segment` is true,
    for inspecting the one real sleep-cycle firing in each of 906b/906c.

This module builds one reusable tool covering both, so future reviews do not
repeat either script ad hoc a third time.

Input format
------------
An episode-log JSON of the shape produced by the 906-lineage full-stack
observational fishtank experiments:

    {"seeds": [{"seed": N, "episodes": [{"ep": ..., "steps": [...],
                                          "sleep_cycle_fired_before_this_segment": bool,
                                          "sleep_cycle_detail": {...} | None, ...}, ...]}, ...],
     ...}

Each step dict carries (among many other fields) `pos`, `action`, `mode`,
`transition_type`, `harm_event`, `action_blocked`, `residue_surprise`, and the
step's own ground-truth `hazards` / `resources` lists of [x, y] grid cells.

Event-selection modes
----------------------
  --mode surprise --top-n N
      Selects the N steps with the globally largest `residue_surprise` value
      across the WHOLE RUN (all seeds, all episodes combined, not per-episode
      -- this matches "top-N surprise peaks from a run" in both source
      reviews and is what the ad hoc predecessor scripts did). To avoid
      returning N near-duplicate windows off the same surprise plateau, peak
      selection is greedy-with-suppression: candidates are considered in
      descending residue_surprise order and skipped if they fall within
      `--window` steps of an already-selected peak in the same
      (seed, episode) -- so N peaks means N distinct events, not N adjacent
      samples of one event.

  --mode sleep
      Selects every episode boundary where the NEXT episode's
      `sleep_cycle_fired_before_this_segment` is true. The window straddles
      the boundary: the last W steps of the PRIOR episode plus the first W
      steps of the episode the sleep cycle preceded (there is no `--top-n`
      for this mode -- it takes every such boundary in the run, which is
      typically 0-2 given how rarely the sleep gate fires in current runs).

Distance fields
----------------
`hazard_dist` / `resource_dist` are the Chebyshev (grid/king-move) distance
from the step's own `pos` to the nearest entry in that step's own `hazards` /
`resources` ground-truth list -- the exact approach used ad hoc in the
sleep-transition investigation, reused here rather than reinvented.

Output format
-------------
Plain markdown tables (one per selected event), matching the ad hoc appendix
tables both source reviews already produced by hand. A richer HTML view was
considered and deliberately skipped -- the reviews only ever needed a table
scannable in a terminal or a markdown viewer, and an HTML/JS dashboard would
be scope beyond a single inspectability script.

Usage
-----
    python3 experiments/_lib/event_window_browser.py <episode_log.json> \\
        --mode surprise --top-n 8 --window 15

    python3 experiments/_lib/event_window_browser.py <episode_log.json> \\
        --mode sleep --window 15

    python3 experiments/_lib/event_window_browser.py <episode_log.json> \\
        --mode surprise --top-n 8 --out surprise_report.md

Or as a library:

    from experiments._lib.event_window_browser import (
        load_episode_log, select_surprise_peaks, select_sleep_boundaries,
        render_event, build_report,
    )
    log = load_episode_log(path)
    events = select_surprise_peaks(log, top_n=8, window=15)
    print(build_report(log, events, window=15))
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Distance helper (reused approach, not reinvented -- see module docstring)
# ---------------------------------------------------------------------------

def chebyshev_distance(pos: Sequence[int], points: Sequence[Sequence[int]]) -> Optional[float]:
    """Chebyshev distance from pos to the nearest point in points.

    Returns None if points is empty (no hazards/resources present on that step
    -- distinct from 0.0, which means the agent is standing on one).
    """
    if not points:
        return None
    px, py = pos[0], pos[1]
    return float(min(max(abs(px - qx), abs(py - qy)) for qx, qy in points))


# ---------------------------------------------------------------------------
# Event selection
# ---------------------------------------------------------------------------

@dataclass
class SelectedEvent:
    seed: Any
    seed_index: int
    episode_index: int          # index into seeds[i]["episodes"]
    ep: Any                     # the episode's own "ep" field
    step_index: Optional[int]   # step index within the episode for surprise mode; None for sleep mode
    value: Optional[float]      # residue_surprise for surprise mode; None for sleep mode
    kind: str                   # "surprise" | "sleep"
    meta: Dict[str, Any] = field(default_factory=dict)


def load_episode_log(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def select_surprise_peaks(log: dict, top_n: int, window: int) -> List[SelectedEvent]:
    """Top-N globally-largest residue_surprise steps across the whole run.

    Greedy selection in descending residue_surprise order, suppressing any
    candidate within `window` steps of an already-selected peak in the same
    (seed, episode) -- see module docstring for why.
    """
    candidates: List[SelectedEvent] = []
    for seed_idx, seed_obj in enumerate(log.get("seeds", [])):
        seed = seed_obj.get("seed", seed_idx)
        for ep_idx, episode in enumerate(seed_obj.get("episodes", [])):
            steps = episode.get("steps", [])
            for step_idx, step in enumerate(steps):
                rs = step.get("residue_surprise")
                if rs is None:
                    continue
                candidates.append(SelectedEvent(
                    seed=seed, seed_index=seed_idx, episode_index=ep_idx,
                    ep=episode.get("ep", ep_idx), step_index=step_idx,
                    value=float(rs), kind="surprise",
                ))

    candidates.sort(key=lambda e: e.value, reverse=True)

    selected: List[SelectedEvent] = []
    for cand in candidates:
        if len(selected) >= top_n:
            break
        suppressed = False
        for chosen in selected:
            if (chosen.seed_index == cand.seed_index
                    and chosen.episode_index == cand.episode_index
                    and abs(chosen.step_index - cand.step_index) <= window):
                suppressed = True
                break
        if not suppressed:
            selected.append(cand)

    # Re-sort selected events into run order (seed, episode, step) for a
    # readable report rather than pure surprise-descending order.
    selected.sort(key=lambda e: (e.seed_index, e.episode_index, e.step_index))
    return selected


def select_sleep_boundaries(log: dict) -> List[SelectedEvent]:
    """Every episode boundary where the NEXT episode's sleep flag is true."""
    selected: List[SelectedEvent] = []
    for seed_idx, seed_obj in enumerate(log.get("seeds", [])):
        seed = seed_obj.get("seed", seed_idx)
        episodes = seed_obj.get("episodes", [])
        for ep_idx, episode in enumerate(episodes):
            if episode.get("sleep_cycle_fired_before_this_segment"):
                selected.append(SelectedEvent(
                    seed=seed, seed_index=seed_idx, episode_index=ep_idx,
                    ep=episode.get("ep", ep_idx), step_index=None, value=None,
                    kind="sleep", meta={"sleep_cycle_detail": episode.get("sleep_cycle_detail")},
                ))
    return selected


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_COLUMNS = [
    "step", "pos", "action", "mode", "transition_type", "harm_event",
    "hazard_dist", "resource_dist", "action_blocked", "residue_surprise",
]


def _step_row(step: dict, step_index: int, marker: str = "") -> List[str]:
    pos = step.get("pos", [None, None])
    hazard_dist = chebyshev_distance(pos, step.get("hazards", []))
    resource_dist = chebyshev_distance(pos, step.get("resources", []))
    values = {
        "step": f"{step_index}{marker}",
        "pos": str(tuple(pos)),
        "action": step.get("action"),
        "mode": step.get("mode"),
        "transition_type": step.get("transition_type"),
        "harm_event": step.get("harm_event"),
        "hazard_dist": hazard_dist,
        "resource_dist": resource_dist,
        "action_blocked": step.get("action_blocked"),
        "residue_surprise": (
            f"{step['residue_surprise']:.4f}" if step.get("residue_surprise") is not None else None
        ),
    }
    return [str(values[c]) for c in _COLUMNS]


def _markdown_table(rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(_COLUMNS) + " |"]
    lines.append("|" + "|".join(["---"] * len(_COLUMNS)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_event(log: dict, event: SelectedEvent, window: int) -> str:
    seed_obj = log["seeds"][event.seed_index]
    episodes = seed_obj["episodes"]
    episode = episodes[event.episode_index]

    if event.kind == "surprise":
        steps = episode["steps"]
        lo = max(0, event.step_index - window)
        hi = min(len(steps), event.step_index + window + 1)
        rows = [
            _step_row(steps[i], i, marker=" <== PEAK" if i == event.step_index else "")
            for i in range(lo, hi)
        ]
        header = (
            f"### surprise peak -- seed {event.seed}, episode {event.episode_index} "
            f"(ep={event.ep}), step {event.step_index}, residue_surprise={event.value:.4f}"
        )
        return header + "\n\n" + _markdown_table(rows)

    # kind == "sleep"
    before_rows: List[List[str]] = []
    if event.episode_index > 0:
        prior_episode = episodes[event.episode_index - 1]
        prior_steps = prior_episode["steps"]
        start = max(0, len(prior_steps) - window)
        before_rows = [_step_row(prior_steps[i], i) for i in range(start, len(prior_steps))]

    after_steps = episode["steps"]
    after_end = min(len(after_steps), window)
    after_rows = [_step_row(after_steps[i], i) for i in range(after_end)]

    detail = event.meta.get("sleep_cycle_detail") or {}
    detail_line = ", ".join(f"{k}={v}" for k, v in detail.items()) if detail else "(no sleep_cycle_detail)"

    header = (
        f"### sleep boundary -- seed {event.seed}, before episode_index "
        f"{event.episode_index} (ep={event.ep})"
    )
    if event.episode_index == 0:
        note = "(no prior episode in this log -- showing after-window only)"
        body = note + "\n\n" + _markdown_table(after_rows)
    else:
        body = (
            _markdown_table(before_rows)
            + "\n\n---- SLEEP BOUNDARY ----\n\n"
            + _markdown_table(after_rows)
        )
    return header + "\n\nsleep_cycle_detail: " + detail_line + "\n\n" + body


def build_report(log: dict, events: List[SelectedEvent], window: int) -> str:
    if not events:
        return "(no events selected)"
    return "\n\n\n".join(render_event(log, e, window) for e in events)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Event-window inspectability browser for 906-lineage episode logs.",
    )
    parser.add_argument("episode_log", help="Path to a *_episode_log.json file.")
    parser.add_argument("--mode", choices=["surprise", "sleep"], required=True)
    parser.add_argument("--top-n", type=int, default=8, help="surprise mode only.")
    parser.add_argument("--window", type=int, default=15, help="Steps either side of each event.")
    parser.add_argument("--out", default=None, help="Write report here instead of stdout.")
    args = parser.parse_args(argv)

    log = load_episode_log(args.episode_log)

    if args.mode == "surprise":
        events = select_surprise_peaks(log, top_n=args.top_n, window=args.window)
    else:
        events = select_sleep_boundaries(log)

    report = build_report(log, events, window=args.window)

    if args.out:
        Path(args.out).write_text(report)
        print(f"wrote {len(events)} event window(s) to {args.out}")
    else:
        print(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
