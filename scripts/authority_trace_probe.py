"""authority_trace_probe -- where does each live modulatory signal's effect DIE?

WHAT THIS IS
------------
A paired ON/OFF ablation over the substrate's default-ENABLED `use_*` flags,
classifying each by the LAST pipeline stage at which its effect is still
observable:

    SCORING -> ELIGIBILITY -> ARBITRATION -> COMMITTED -> REALISED

WHY IT IS NOT A TRACE
---------------------
A forward trace of a signal through those stages is a chain of true statements
that answers the wrong question. The confirmed failure cluster this probe is
built from is exactly the case where the trace is LOUD and the effect is NIL:

  V3-EXQ-931/932 (2026-08-16..19, CEM modulatory authority cluster)
      80.3% of genuine elite argmins FLIPPED while mean_resource_proximity was
      BIT-IDENTICAL to ablation. Authority does not imply throughput.
  EXP-0155 (2026-08-18, SD-016/MECH-151 action_bias)
      MECH-151's registered notes assert action_bias elevates/suppresses
      candidates. `e3_selector.py` contains ZERO occurrences of action_bias:
      the signal has no causal channel to the ranking path at all.
  V3-EXQ-689d (2026-07-20, MATCHED_ENTROPY_TEMPERATURE)
      Knob present in config_slice and the arm fingerprint, never reached a
      sampling step. Two "declared distinct" arms were the same arm, and a
      conjunctive PASS went silently vacuous.

The discriminator in all three was a COUNTERFACTUAL, not a trace. So every
column here is an ablation contrast, and the reported per-stage magnitude is a
SPREAD RATIO in the sense of `e3_selector.authority_spread_ratio`: the signal's
cross-candidate spread against the dominant term's, because "nonzero" is not
the same as "competitive" (V3-EXQ-931 read ~0.0037 against a 0.1 floor).

THE NULL CONTROL IS MANDATORY, NOT OPTIONAL
-------------------------------------------
Every flag is measured twice: OFF-vs-ON, and ON-vs-ON. The second is a null
control on the harness itself. If ON-vs-ON is not bit-identical, the two arms
desynchronised for a reason unrelated to the flag (an RNG stream that advances
a different number of times, most likely), and the flag's verdict is reported
as UNMEASURABLE rather than as a finding. Without this, a desync is
indistinguishable from authority and every verdict is inflated.

THE from_dims SWALLOW TRAP
--------------------------
`REEConfig.from_dims(**kwargs)` silently ignores an unknown kwarg. A typo'd
flag name therefore produces an OFF arm identical to the ON arm and reports
`inert` -- a false negative that looks like a finding. Every flag is
precondition-checked: after construction the value must actually have landed,
on EVERY config object that declares it.

E3 CADENCE
----------
`heartbeat.e3_steps_per_tick` defaults to 10 and `agent.py` returns the HELD
action on a non-E3 tick BEFORE `e3.select()` is reached. Reading `e3.last_*`
per env step therefore replicates one genuine selection ~10 times -- the
hold-weighted readout defect (699 / 689d autopsies). All stage comparisons are
gated on `experiments._lib.fresh_select`'s sentinel.

ASCII-only output. Read-only with respect to the substrate: this script
constructs its own agents and never mutates ree_core.

Usage:
    /opt/local/bin/python3 scripts/authority_trace_probe.py --seeds 3 --steps 120
    /opt/local/bin/python3 scripts/authority_trace_probe.py --flags use_bla_analog --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REE_V3 = os.path.dirname(_HERE)
if _REE_V3 not in sys.path:
    sys.path.insert(0, _REE_V3)

import torch  # noqa: E402

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiments._harness import StepHarness, StepHooks  # noqa: E402
from experiments._lib.fresh_select import FreshSelectProbe  # noqa: E402

# The five stages, in pipeline order. A verdict names the LAST stage at which
# the ablation contrast is still nonzero.
STAGES = ("scoring", "eligibility", "arbitration", "committed", "realised")


# --------------------------------------------------------------------------
# flag discovery
# --------------------------------------------------------------------------

_DISCOVERY_SEED = 11


def _config_objects(cfg: Any) -> Dict[str, Any]:
    """Top-level config plus every attribute that is itself a *Config."""
    out = {type(cfg).__name__: cfg}
    for name in dir(cfg):
        if name.startswith("_"):
            continue
        try:
            v = getattr(cfg, name)
        except Exception:
            continue
        if type(v).__name__.endswith("Config"):
            out.setdefault(type(v).__name__, v)
    return out


def flag_map(cfg: Any) -> Dict[str, Dict[str, bool]]:
    """flag name -> {ConfigClass: value} over a LIVE config tree."""
    vals: Dict[str, Dict[str, bool]] = {}
    for cls, obj in _config_objects(cfg).items():
        for name in dir(obj):
            if not name.startswith("use_"):
                continue
            try:
                v = getattr(obj, name)
            except Exception:
                continue
            if isinstance(v, bool):
                vals.setdefault(name, {})[cls] = v
    return vals


def discover_default_on_flags() -> Dict[str, List[str]]:
    """Flags TRUE on a bare default-constructed config -> declaring classes.

    INTROSPECTION, NOT SOURCE PARSING, AND THE DIFFERENCE IS LOAD-BEARING.
    Two source-parse traps were hit building this and both inflate the
    population with flags that are not actually live:

      * a `from_dims` / `enable_goal_stream` SIGNATURE default (8-space indent)
        reads identically to a dataclass field (4-space) under a `^\\s*` regex,
        double-counting every flag that has both;
      * `use_resource_encoder` declares `False` as a dataclass field and `True`
        as an `enable_goal_stream` parameter, so its "default" depends entirely
        on the construction path. No parse of the declaration can settle that.

    The operating point measured here is the BARE default: `REEConfig.from_dims`
    with no profile method called. Profile methods (`enable_goal_stream` and
    friends) flip whole clusters of flags and constitute a SEPARATE operating
    point that this probe does not sweep -- see the module docstring.
    """
    _env, _agent, cfg = build(_DISCOVERY_SEED, {})
    return {
        name: sorted(cls for cls, v in per.items() if v)
        for name, per in flag_map(cfg).items()
        if any(per.values())
    }


def declaring_classes(flag: str) -> List[str]:
    """Every live config object carrying `flag`, at any value."""
    _env, _agent, cfg = build(_DISCOVERY_SEED, {})
    return sorted(flag_map(cfg).get(flag, {}))


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------

WORLD_DIM = 16

# The OPERATING POINT. Every rollout, and every discovery/precondition call,
# is built on top of this. Empty = the bare default. Set from --base.
#
# This axis is not cosmetic. `use_bla_analog` / `use_cea_analog` are default-ON
# consumers of `z_harm` / `z_harm_a`, but `LatentStack` only constructs
# `harm_encoder` / `affective_harm_encoder` when `use_harm_stream` /
# `use_affective_harm_stream` are set, and both default OFF. At the bare
# default those two flags are therefore inert BECAUSE THEIR PRODUCER DOES NOT
# EXIST, which is a different finding from being inert with the producer live.
# Distinguishing the two requires re-probing at a second operating point.
BASE_OVERRIDES: Dict[str, Any] = {}


def build(seed: int, overrides: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    merged = dict(BASE_OVERRIDES)
    merged.update(overrides)
    overrides = merged
    torch.manual_seed(seed)
    env = CausalGridWorldV2(
        seed=seed, size=5, num_hazards=1, num_resources=1, use_proxy_fields=True
    )
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        self_dim=16,
        world_dim=WORLD_DIM,
        **overrides,
    )
    agent = REEAgent(cfg)
    agent.reset()
    return env, agent, cfg


def live_default(flag: str) -> Optional[bool]:
    """The flag's value on a bare default-constructed config, or None."""
    _env, _agent, cfg = build(_DISCOVERY_SEED, {})
    per = flag_map(cfg).get(flag)
    if not per:
        return None
    return any(per.values())


def precondition_flag_lands(flag: str, seed: int, target: bool) -> Tuple[bool, str]:
    """Confirm `flag=target` actually reaches every config object carrying it.

    Guards the from_dims silent-kwarg-swallow trap: an unknown kwarg produces
    an OFF arm identical to ON, which reads as `inert`.
    """
    _env, _agent, cfg = build(seed, {flag: target})
    per = flag_map(cfg).get(flag)
    if not per:
        return False, "flag is not an attribute of any live config object"
    misses = sorted(cls for cls, v in per.items() if v is not target)
    if misses:
        return False, "override did not land on: " + ",".join(misses)
    return True, "landed on %d config object(s) as %s" % (len(per), target)


# --------------------------------------------------------------------------
# per-tick capture
# --------------------------------------------------------------------------

@dataclass
class TickCapture:
    step: int
    raw_scores: Optional[List[float]]
    scores: Optional[List[float]]
    selected_idx: Optional[int]
    n_eligible: Optional[int]
    arbitration_ran: bool
    action: Optional[List[float]]


class Recorder:
    """Captures E3 stage state on FRESH E3 ticks only."""

    def __init__(self, agent: Any, namespace: str) -> None:
        self.agent = agent
        self.probe = FreshSelectProbe(namespace)
        self.ticks: List[TickCapture] = []
        self.actions: List[Tuple[float, ...]] = []
        self.harm_sum: float = 0.0
        self.n_steps: int = 0
        self._step = 0

    def hooks(self) -> StepHooks:
        return StepHooks(on_sense=self._on_sense, on_action=self._on_action)

    def _on_sense(self, **kw: Any) -> None:
        self.probe.mark_stale(self.agent)

    def _on_action(self, **kw: Any) -> None:
        fresh = self.probe.is_fresh(self.agent)
        self._step += 1
        if not fresh:
            return
        e3 = getattr(self.agent, "e3", None)
        diag = getattr(e3, "last_score_diagnostics", None) or {}

        def _vec(name: str) -> Optional[List[float]]:
            t = getattr(e3, name, None)
            if t is None:
                return None
            try:
                return [round(float(x), 10) for x in t.detach().reshape(-1).tolist()]
            except Exception:
                return None

        n_elig = diag.get("modulatory_shortlist_size")
        if n_elig is None:
            n_elig = diag.get("f_eligibility_envelope_size")
        self.ticks.append(
            TickCapture(
                step=self._step,
                raw_scores=_vec("last_raw_scores"),
                scores=_vec("last_scores"),
                selected_idx=(
                    int(e3.last_selected_idx)
                    if getattr(e3, "last_selected_idx", None) is not None
                    else None
                ),
                n_eligible=int(n_elig) if n_elig is not None else None,
                arbitration_ran=getattr(e3, "last_arbitration", None) is not None,
                action=None,
            )
        )


def rollout(seed: int, overrides: Dict[str, Any], steps: int, namespace: str) -> Recorder:
    env, agent, _cfg = build(seed, overrides)
    rec = Recorder(agent, namespace)
    harness = StepHarness(agent, env, train_mode=False, hooks=rec.hooks(), seed=seed)
    flat, od = env.reset()
    agent.reset()
    harness.reset()
    with torch.no_grad():
        for _ in range(steps):
            res = harness.step(od)
            od = res.next_obs_dict
            rec.actions.append(tuple(
                round(float(x), 10) for x in res.action.detach().reshape(-1).tolist()
            ))
            rec.harm_sum += float(res.harm_signal)
            rec.n_steps += 1
            if res.done:
                flat, od = env.reset()
                agent.reset()
                harness.reset()
    return rec


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------

def _spread(v: List[float]) -> float:
    return max(v) - min(v) if v else 0.0


def _pairwise_stage_delta(a: Recorder, b: Recorder) -> Dict[str, Any]:
    """Compare two rollouts tick-by-tick, but ONLY while still state-identical.

    Once the executed actions diverge the two agents are in different world
    states, so a per-tick score comparison is comparing different problems.
    `n_comparable` is the honest denominator.
    """
    # First divergent executed action bounds the comparable prefix.
    div_step: Optional[int] = None
    for i, (x, y) in enumerate(zip(a.actions, b.actions)):
        if x != y:
            div_step = i + 1
            break
    if len(a.actions) != len(b.actions) and div_step is None:
        div_step = min(len(a.actions), len(b.actions)) + 1

    ta = {t.step: t for t in a.ticks}
    tb = {t.step: t for t in b.ticks}
    common = sorted(set(ta) & set(tb))
    if div_step is not None:
        common = [s for s in common if s < div_step]

    n_scoring_diff = 0
    n_committed_diff = 0
    n_elig_diff = 0
    n_arb_diff = 0
    ratios: List[float] = []
    max_abs = 0.0
    for s in common:
        x, y = ta[s], tb[s]
        if x.raw_scores and y.raw_scores and len(x.raw_scores) == len(y.raw_scores):
            d = [p - q for p, q in zip(y.raw_scores, x.raw_scores)]
            if any(abs(v) > 0.0 for v in d):
                n_scoring_diff += 1
                max_abs = max(max_abs, max(abs(v) for v in d))
                base = _spread(x.raw_scores)
                if base > 0.0:
                    ratios.append(_spread(d) / base)
        if x.selected_idx != y.selected_idx:
            n_committed_diff += 1
        if x.n_eligible != y.n_eligible:
            n_elig_diff += 1
        if x.arbitration_ran != y.arbitration_ran:
            n_arb_diff += 1

    return {
        "n_comparable_ticks": len(common),
        "divergence_step": div_step,
        "n_scoring_diff": n_scoring_diff,
        "n_eligibility_diff": n_elig_diff,
        "n_arbitration_diff": n_arb_diff,
        "n_committed_diff": n_committed_diff,
        "authority_spread_ratio_mean": (
            round(sum(ratios) / len(ratios), 6) if ratios else 0.0
        ),
        "max_abs_score_delta": round(max_abs, 10),
        "eligibility_stage_active": any(
            t.n_eligible is not None for t in list(ta.values()) + list(tb.values())
        ),
        "arbitration_stage_active": any(
            t.arbitration_ran for t in list(ta.values()) + list(tb.values())
        ),
        "realised_actions_differ": a.actions != b.actions,
        "realised_harm_delta": round(b.harm_sum - a.harm_sum, 8),
    }


def classify(d: Dict[str, Any], control_clean: bool) -> str:
    if not control_clean:
        return "UNMEASURABLE"
    if d["n_scoring_diff"] == 0 and not d["realised_actions_differ"]:
        return "INERT"
    if d["realised_actions_differ"] or d["n_committed_diff"] > 0:
        return "REALISED"
    if d["n_arbitration_diff"] > 0:
        return "ARBITRATION"
    if d["n_eligibility_diff"] > 0:
        return "ELIGIBILITY"
    if d["n_scoring_diff"] > 0:
        return "SCORING_ONLY"
    return "INERT"


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def probe_flag(flag: str, seeds: List[int], steps: int) -> Dict[str, Any]:
    """Contrast the BARE DEFAULT against the same config with `flag` FLIPPED.

    Direction follows the live default, so the probe is meaningful on a
    default-off flag (contrast = turn it ON) as well as a default-on one
    (contrast = ablate it). That symmetry is what makes a POSITIVE CONTROL
    possible: a default-off flag known to change scoring must NOT come back
    INERT, and if it does the harness is not measuring anything.
    """
    cur = live_default(flag)
    if cur is None:
        return {"flag": flag, "verdict": "PRECONDITION_FAIL",
                "reason": "not an attribute of any live config object",
                "default": None}
    target = not cur
    ok, why = precondition_flag_lands(flag, seeds[0], target)
    if not ok:
        return {"flag": flag, "verdict": "PRECONDITION_FAIL", "reason": why,
                "default": cur}

    per_seed: List[Dict[str, Any]] = []
    for sd in seeds:
        on = rollout(sd, {}, steps, "atp_on")
        # NULL CONTROL: a second independent ON build. Any difference here is
        # harness/RNG desync, not the flag.
        ctl = rollout(sd, {}, steps, "atp_ctl")
        off = rollout(sd, {flag: target}, steps, "atp_off")
        c = _pairwise_stage_delta(on, ctl)
        control_clean = (
            c["n_scoring_diff"] == 0
            and c["n_committed_diff"] == 0
            and not c["realised_actions_differ"]
        )
        d = _pairwise_stage_delta(on, off)
        d["verdict"] = classify(d, control_clean)
        d["control_clean"] = control_clean
        d["seed"] = sd
        d["n_fresh_ticks_on"] = len(on.ticks)
        per_seed.append(d)

    verdicts = [p["verdict"] for p in per_seed]
    order = {v: i for i, v in enumerate(
        ["UNMEASURABLE", "INERT", "SCORING_ONLY", "ELIGIBILITY", "ARBITRATION", "REALISED"]
    )}
    # Report the FURTHEST stage reached on any seed -- a signal with throughput
    # on one seed and none on another has throughput.
    agg = max(verdicts, key=lambda v: order.get(v, 0))
    if "UNMEASURABLE" in verdicts and len(set(verdicts)) > 1:
        agg = agg + "*"  # at least one seed could not be measured
    return {
        "flag": flag,
        "verdict": agg,
        "reason": why,
        "default": cur,
        "contrast": "%s -> %s" % (cur, target),
        "per_seed": per_seed,
        "declared_on": None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=2, help="number of seeds (from --seed0)")
    ap.add_argument("--seed0", type=int, default=11)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--flags", nargs="*", default=None,
                    help="explicit flag names; default = every default-ON use_* flag. "
                         "A named default-OFF flag is contrasted in the ON direction, "
                         "which is how positive controls are run.")
    ap.add_argument("--base", nargs="*", default=None, metavar="FLAG=BOOL",
                    help="operating point: extra overrides applied to EVERY arm "
                         "(e.g. --base use_harm_stream=true use_affective_harm_stream=true). "
                         "Contrast direction is recomputed against this base.")
    ap.add_argument("--json", default=None, help="write full result JSON here")
    ap.add_argument("--list", action="store_true", help="list the flag population and exit")
    args = ap.parse_args()

    for kv in (args.base or []):
        if "=" not in kv:
            print("bad --base entry (want FLAG=BOOL): %s" % kv)
            return 2
        k, v = kv.split("=", 1)
        lv = v.strip().lower()
        if lv not in ("true", "false"):
            print("bad --base value (want true/false): %s" % kv)
            return 2
        BASE_OVERRIDES[k.strip()] = (lv == "true")
    if BASE_OVERRIDES:
        # Fail loudly if an operating-point override did not land -- silently
        # falling back to the bare default would mislabel the whole table.
        _e, _a, _c = build(_DISCOVERY_SEED, {})
        fm = flag_map(_c)
        for k, want in BASE_OVERRIDES.items():
            per = fm.get(k)
            if not per or any(v is not want for v in per.values()):
                print("--base %s=%s did NOT land (per-object: %s)" % (k, want, per))
                return 2
        print("operating point: %s" % BASE_OVERRIDES)

    default_on = discover_default_on_flags()
    if args.list:
        print("default-True use_* flags: %d declarations, %d unique names"
              % (sum(len(v) for v in default_on.values()), len(default_on)))
        for name in sorted(default_on):
            print("  %-42s %s" % (name, ",".join(default_on[name])))
        return 0

    flags = args.flags if args.flags else sorted(default_on)
    seeds = [args.seed0 + i for i in range(args.seeds)]

    print("authority_trace_probe -- %d flag(s), seeds=%s, steps=%d"
          % (len(flags), seeds, args.steps))
    print("stages: " + " -> ".join(s.upper() for s in STAGES))
    print()

    results: List[Dict[str, Any]] = []
    for i, fl in enumerate(flags, 1):
        print("[%2d/%2d] %s ..." % (i, len(flags), fl), flush=True)
        r = probe_flag(fl, seeds, args.steps)
        r["declared_on"] = default_on.get(fl, declaring_classes(fl))
        results.append(r)
        ps = r.get("per_seed") or [{}]
        print("        verdict=%-14s spread_ratio=%s committed_diff=%s comparable=%s"
              % (r["verdict"],
                 ps[0].get("authority_spread_ratio_mean"),
                 sum(p.get("n_committed_diff", 0) for p in ps),
                 sum(p.get("n_comparable_ticks", 0) for p in ps)))

    print()
    print("=" * 100)
    print("%-42s %-14s %10s %9s %9s %8s" % (
        "flag", "dies_at", "spread", "commit", "realised", "ticks"))
    print("-" * 100)
    for r in sorted(results, key=lambda x: x["verdict"]):
        ps = r.get("per_seed") or []
        sr = max((p.get("authority_spread_ratio_mean", 0.0) for p in ps), default=0.0)
        cd = sum(p.get("n_committed_diff", 0) for p in ps)
        rd = any(p.get("realised_actions_differ") for p in ps)
        nt = sum(p.get("n_comparable_ticks", 0) for p in ps)
        print("%-42s %-14s %10.4f %9d %9s %8d" % (
            r["flag"][:42], r["verdict"], sr, cd, "yes" if rd else "no", nt))
    print("=" * 100)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"seeds": seeds, "steps": args.steps, "results": results}, fh, indent=2)
        print("wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
