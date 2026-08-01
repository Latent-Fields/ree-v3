#!/opt/local/bin/python3
"""
V3-EXQ-862: Q-040.c -- is dACC behavioural-adjustment magnitude proportional
to V_s precision-modulated forward-PE under MECH-269b=ON?

Supersedes V3-EXQ-475b. See
REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-475b_2026-08-01.md
(confirmed): 475b FAILed at precondition P2 (dACC engagement) on 0/6 runs --
`n_dacc_fires = 0` in every seed x arm cell, i.e. `agent._dacc_last_bundle`
was None on every tick. Root cause, code-traced: `ree_core/agent.py` gates
dACC's forward pass with `if self.dacc is not None and z_harm_a is not
None:` (agent.py:6117). `self.dacc` is instantiated (use_dacc=True is the
GAP-4 cohort default), so the failure is `z_harm_a is None` on every tick.
475b's `_build_arm_config` called `_lib.goal_pipeline_tier1.build_config(env,
gap4_arm)` WITHOUT `enable_affective_harm_stream=True` -- and `build_config`
only wires `use_harm_stream` / `use_affective_harm_stream` / `z_harm_a_dim`
into the config INSIDE an `if enable_affective_harm_stream:` block, which
defaults False. z_harm_a therefore stayed structurally None for the whole
run. This is the exact failure mode `build_config`'s own docstring names by
precedent (the V3-EXQ-620/625 measurement artifact diagnosed 2026-06-01).

THE FIX (the only substantive change from 475b): `_build_arm_config` below
passes `enable_affective_harm_stream=True` to `build_config(...)`, and (as a
direct, narrowly-scoped consequence of that change actually taking effect)
`run_seed_arm` now stamps `env._exq_env_kwargs` before building the config --
`build_config`'s `enable_affective_harm_stream=True` branch reads
`limb_damage_enabled` off `env_kwargs_or_default(env)`, which falls back to
the module-level `ENV_FISHTANK_KWARGS` default when `_exq_env_kwargs` is
unset. 475b never set it (relying on the fact that its caller always passed
`ENV_FISHTANK_KWARGS` anyway, so the fallback coincidentally matched); this
version sets it explicitly so the value actually used is never coincidental.
Everything else -- the P1 MECH-269b V_s-gating precondition, the correlation
statistic (per-seed Spearman rho of |precision-weighted PE| vs dACC
score-bias-vector magnitude), the 3 seeds (42/7/19), the 2-arm
(ARM_vs_off/ARM_vs_on) structure, the acceptance thresholds -- is unchanged
from 475b.

PRE-FLIGHT SMOKE GATE (new, per the autopsy's learning #3: "A cheap
smoke-test assertion (n_dacc_fires > 0) before committing to the full 6-run
design would have caught this in minutes"). Before the full 3-seed x 2-arm
design runs, `_preflight_dacc_engagement_check()` below runs a short pilot
rollout (ARM_vs_on, the pilot seed, a handful of warmup episodes + eval
steps) and asserts dACC actually fired at least once
(`agent._dacc_last_bundle` was populated on >=1 tick). If it did not, the
script raises `PreflightDaccNotFiring` uncaught -- a fail-fast crash (ERROR),
not a silent full-budget FAIL -- rather than repeating 475b's failure mode of
discovering `n_dacc_fires=0` only after all 6 runs completed. This gate runs
on every invocation (dry-run and real), not only under --dry-run, because the
autopsy's point is that the check must happen BEFORE the full design commits
compute, and a real run's config could in principle regress independently of
what --dry-run exercised.

MECHANISM. ree_core/cingulate/dacc.py's DACCAdaptiveControl.forward()
computes a precision-weighted affective-pain PE (MECH-258, bundle["pe"]) each
tick and DACCtoE3Adapter converts the bundle into a per-candidate E3 score
bias (agent._dacc_last_bias). The "dACC weight-delta" this experiment reads
is the MAGNITUDE of that per-step bias vector (its L2 norm across
candidates) -- the behavioural-adjustment dACC actually applies. The claim
under test is architectural: MECH-269b (V_s gating E1/E2 forward rollouts,
ree_core/agent.py, IMPLEMENTED 2026-04-26) determines whether the z_harm_a
forward prediction dACC's PE is computed against is fresh or stale. If V_s
gating is OFF, E2 keeps grounding harm-rollouts on stale streams, so any PE
dACC sees is decoupled from genuine environmental precision -- the bias
magnitude should track it weakly if at all. If V_s gating is ON, PE should
track a live precision-modulated signal, and the bias magnitude (which is
built directly from mode_ev, itself pe * dacc_effort_cost-scaled) should
track it detectably.

ARMS (factorial on ONE variable, matching the V3-EXQ-490b/490c convention of
holding the rest of the V_s invalidation circuit ON in both arms so only
use_vs_rollout_gating differs):
  ARM_vs_off: gap4-475a-conditions substrate + full V_s invalidation circuit
    (use_per_stream_vs/use_event_segmenter/use_invalidation_trigger/
    use_anchor_sets/use_per_region_vs/use_staleness_accumulator/
    use_mech284_hysteresis/use_vs_commit_release all True) but
    use_vs_rollout_gating=False. AFFECTIVE HARM STREAM ON (the 862 fix).
  ARM_vs_on: identical, use_vs_rollout_gating=True, plus the 490b smoke-scale
    gate threshold override (vs_gate_snapshot_refresh_threshold=0.95,
    vs_gate_e1_threshold=0.85, vs_gate_e2_threshold=0.85) -- without this the
    gate rarely crosses its hold trigger under typical Phase-1 V_s dynamics
    at this harness's episode/step scale (490/490a both FAILed the C1
    gate-firing precondition at default thresholds; 490b's fix is the
    reference).

HARD PRECONDITIONS (mirrors Q-040's own "Absent these the run self-routes
substrate-not-ready, not a verdict"):
  P1 (gate-firing): agent.vs_rollout_gate diagnostics show >0 total held
     (e1+e2) on >=2/3 seeds in ARM_vs_on. If not met, MECH-269b was never
     actually engaged this run and no verdict is possible.
  P2 (dACC engagement): dacc_bias_nonzero_steps > 0 on >=2/3 seeds in BOTH
     arms. If dACC never fires, there is nothing to correlate. (This is the
     precondition that caught 475b's defect -- correctly, but only at full
     scale. The pre-flight gate above is the same check run cheaply first.)
  Both are checked BEFORE the primary DV is read; failing either self-routes
  evidence_direction="non_contributory" (substrate_not_ready_requeue), never
  a weakens.

PRIMARY DV. Per (seed, arm), the Spearman rank correlation (via the
degeneracy-safe experiments/_lib/stats.py::spearman, NOT a hand-rolled
Pearson/rank helper -- see that module's docstring for the corpus-wide
degenerate-input bug it exists to prevent) between the per-eval-step
precision-weighted PE (bundle["pe"], already non-negative) and the per-step
dACC bias magnitude (||agent._dacc_last_bias||_2), over every eval step where
both are defined (i.e. dACC actually fired that tick).

PASS = (>=2/3 ARM_vs_on seeds have a DEFINED rho with |rho| >= 0.3) AND
       (>=2/3 ARM_vs_off seeds have a DEFINED rho with |rho| < 0.15).
A seed with an UNDEFINED (degenerate-input) rho is excluded from both counts,
never coerced to 0 -- coercing degenerate-to-null is exactly the corpus bug
_lib/stats.py's spearman() was written to stop (a constant bias-magnitude
series in a low-firing OFF arm is an expected, uninformative degeneracy, not
evidence of "no correlation").

DECLARED NULL. A FAIL here (no clean ON/OFF dissociation) does NOT reopen
Q-040a/b or goal_pipeline:GAP-4 -- those are independently settled by the 490
cohort and GAP-4 stays closed per governance_2026_06_09. A FAIL means only
that the fine-grained PE-magnitude coupling this sub-question asks about is
not detectable at this harness's scale/design; it leaves Q-040 as a whole
narrowed to "necessity falsified, modulatory reading stands, quantification
sub-question answered no" rather than "quantification untested".

GOV-REUSE-1: the decisive readout (dACC bias-magnitude vs precision-weighted
PE correlation, WITH dACC actually engaging) is not recorded in any prior
manifest. `reanalysis_query.py query --readout rho_pe_vs_bias_magnitude
--claim Q-040` returns exactly one manifest (475b's own), and that run's
n_dacc_fires=0 in every cell -- there is nothing to reanalyze. Not
recoverable -> proceed to author (this script).

Re-derive brake: 0 prior `substrate_ceiling` autopsies tag Q-040 (checked via
the standard grep-count method over
REE_assembly/evidence/planning/failure_autopsy_*.json) -- does not fire. This
is a non_contributory driver-bug category (measurement/instrumentation), not
a substrate_ceiling.

SLEEP DRIVER: not applicable (no sleep-cycle flags set).

claim_ids: [Q-040]
experiment_purpose: evidence
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from _lib.stats import spearman  # noqa: E402
from _lib.goal_pipeline_tier1 import (  # noqa: E402
    ArmSpec,
    ENV_FISHTANK_KWARGS,
    EVAL_EPISODES_DEFAULT,
    SEEDS_DEFAULT,
    STEPS_PER_EPISODE_DEFAULT,
    WARMUP_EPISODES_DEFAULT,
    build_config,
    make_env,
    warmup_train,
)
from experiments._metrics import check_degeneracy  # noqa: E402
from experiments._harness import StepHarness, StepHooks  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_862_q040c_dacc_pe_weight_delta_correlation"
QUEUE_ID = "V3-EXQ-862"
CLAIM_IDS = ["Q-040"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES_RUN_ID = "v3_exq_475b_q040c_dacc_pe_weight_delta_correlation"
SUPERSEDES_QUEUE_ID = "V3-EXQ-475b"

VS_ON_ARM = "ARM_vs_on"
VS_OFF_ARM = "ARM_vs_off"

# V3-EXQ-490b smoke-scale gate threshold override (carried verbatim -- see
# that script's rationale). Without this the gate's hold trigger rarely
# crosses under typical Phase-1 V_s dynamics at this harness's episode count.
VS_GATE_SNAPSHOT_REFRESH_THRESHOLD = 0.95
VS_GATE_E1_THRESHOLD = 0.85
VS_GATE_E2_THRESHOLD = 0.85

# Acceptance thresholds for the primary DV (Spearman rho of |PE| vs dACC
# bias-vector magnitude, per seed).
CORR_DETECT_THRESH = 0.3   # ON arm: |rho| at/above this counts as "detected"
CORR_NULL_THRESH = 0.15    # OFF arm: |rho| below this counts as "~zero"
SEEDS_PASS_MIN = 2         # of 3, matching the cohort's TIER1_SEEDS_PASS_MIN

# Pre-flight smoke-gate scale -- deliberately much smaller than even the
# --dry-run scale (1 seed, 8 warmup / 2 eval episodes, 40 steps). Just enough
# ticks to observe whether dACC engages at all; this is a readiness check,
# not a measurement.
PREFLIGHT_WARMUP_EPISODES = 3
PREFLIGHT_EVAL_EPISODES = 1
PREFLIGHT_STEPS_PER_EPISODE = 20
PREFLIGHT_SEED = SEEDS_DEFAULT[0]


class PreflightDaccNotFiring(RuntimeError):
    """Raised when the pre-flight pilot rollout observes zero dACC engagement.

    This is the fail-fast gate the V3-EXQ-475b autopsy's learning #3 asks
    for: catch a structurally-disabled dACC (agent._dacc_last_bundle staying
    None on every tick, e.g. because z_harm_a never got wired up) in the time
    it takes to run a handful of steps, rather than discovering it only after
    the full 3-seed x 2-arm design has already run to completion (475b's
    failure mode -- n_dacc_fires=0 in all 6 cells, ~90 min of compute spent
    on a structurally dead run).
    """


def _build_arm_config(env, gap4_arm: ArmSpec, *, vs_rollout_gating: bool):
    """475a-conditions gap4-operating config + full V_s invalidation circuit.

    Reuses _lib.goal_pipeline_tier1.build_config for the shared GAP-4
    substrate (gaba decay + PAG freeze + unconditional use_dacc=True), then
    layers the MECH-269b V_s-circuit knobs directly onto cfg.hippocampal --
    these live on the nested HippocampalConfig object (see
    ree_core/utils/config.py:7434-7451), not on REEConfig itself, so
    ArmSpec.extra_config's flat setattr(cfg, key, val) cannot reach them.

    enable_affective_harm_stream=True is the 862 fix (see module docstring):
    without it, build_config's gap4_operating branch never wires
    use_harm_stream / use_affective_harm_stream / z_harm_a_dim into the
    config, z_harm_a stays None on every tick, and agent.dacc's forward-pass
    guard (`self.dacc is not None and z_harm_a is not None`, agent.py:6117)
    never passes -- exactly 475b's confirmed root cause.
    """
    cfg = build_config(env, gap4_arm, enable_affective_harm_stream=True)
    hc = cfg.hippocampal
    # Full V_s invalidation circuit ON in BOTH arms (490b/490c convention):
    # isolates use_vs_rollout_gating as the sole toggled variable.
    hc.use_per_stream_vs = True
    hc.use_event_segmenter = True
    hc.use_invalidation_trigger = True
    hc.use_anchor_sets = True
    hc.use_per_region_vs = True
    hc.use_staleness_accumulator = True
    hc.use_mech284_hysteresis = True
    hc.use_vs_commit_release = True
    hc.use_vs_rollout_gating = bool(vs_rollout_gating)
    hc.vs_gate_snapshot_refresh_threshold = VS_GATE_SNAPSHOT_REFRESH_THRESHOLD
    hc.vs_gate_e1_threshold = VS_GATE_E1_THRESHOLD
    hc.vs_gate_e2_threshold = VS_GATE_E2_THRESHOLD
    return cfg


def _pe_and_bias_magnitude(agent: REEAgent) -> Tuple[Optional[float], Optional[float]]:
    """Read (precision-weighted PE, dACC bias-vector L2 magnitude) for this tick.

    Reads agent._dacc_last_bundle / agent._dacc_last_bias directly (NOT a
    `._last_bundle` attribute -- no substrate object defines one; see
    validate_experiments.py::dacc_last_bundle_lint). Returns (None, None)
    when dACC did not fire this tick (bundle absent, e.g. z_harm_a was None).
    """
    bundle = getattr(agent, "_dacc_last_bundle", None)
    if bundle is None:
        return None, None
    pe = bundle.get("pe")
    bias = getattr(agent, "_dacc_last_bias", None)
    if pe is None or bias is None:
        return None, None
    try:
        bias_mag = float(torch.as_tensor(bias).norm().item())
    except Exception:
        return None, None
    return float(pe), bias_mag


def _preflight_dacc_engagement_check(env_kwargs: Optional[Dict[str, Any]] = None) -> int:
    """Pre-flight smoke gate: assert dACC engages at all before the full design runs.

    Builds ARM_vs_on (vs_rollout_gating=True -- the arm most likely to
    exercise dACC, since it is the arm the substrate is expected to actually
    engage under) at a fixed pilot seed, runs a short warmup + a handful of
    eval steps, and counts ticks where agent._dacc_last_bundle populated
    (dACC actually fired). Raises PreflightDaccNotFiring if that count is
    zero. Returns the count on success (informational).

    Deliberately cheap: PREFLIGHT_WARMUP_EPISODES=3 / EVAL_EPISODES=1 /
    STEPS_PER_EPISODE=20 -- an order of magnitude below even the --dry-run
    scale. Runs unconditionally (both --dry-run and real invocations) since
    the point is to catch a structurally-disabled dACC before ANY compute
    commits, not merely to keep the smoke test itself honest.
    """
    import random

    import numpy as np

    seed = PREFLIGHT_SEED
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    kwargs = dict(env_kwargs or ENV_FISHTANK_KWARGS)
    env = make_env(seed, kwargs)
    env._exq_env_kwargs = kwargs
    gap4_arm = ArmSpec(
        "gap4_475a_conditions",
        gap4_operating=True,
        use_gabaergic_decay=True,
        use_pag_freeze_gate=True,
        use_broadcast_override=True,
    )
    cfg = _build_arm_config(env, gap4_arm, vs_rollout_gating=True)
    agent = REEAgent(cfg)

    warmup_train(
        agent,
        env,
        num_episodes=PREFLIGHT_WARMUP_EPISODES,
        steps_per_episode=PREFLIGHT_STEPS_PER_EPISODE,
        label="preflight_dacc_check",
        progress_total_episodes=PREFLIGHT_WARMUP_EPISODES + PREFLIGHT_EVAL_EPISODES,
    )

    n_dacc_fires = 0

    def on_post_step(*, agent, **kwargs) -> None:
        nonlocal n_dacc_fires
        pe, bias_mag = _pe_and_bias_magnitude(agent)
        if pe is not None and bias_mag is not None:
            n_dacc_fires += 1

    hooks = StepHooks(on_post_step=on_post_step)
    harness = StepHarness(agent, env, train_mode=False, hooks=hooks, seed=seed)
    agent.eval()

    for _ep in range(PREFLIGHT_EVAL_EPISODES):
        _, obs_dict = env.reset()
        agent.reset()
        harness.reset()
        for _ in range(PREFLIGHT_STEPS_PER_EPISODE):
            result = harness.step(obs_dict)
            obs_dict = result.next_obs_dict
            if result.done:
                break

    print(
        f"[preflight] dACC engagement check: n_dacc_fires={n_dacc_fires} "
        f"over {PREFLIGHT_EVAL_EPISODES * PREFLIGHT_STEPS_PER_EPISODE} pilot eval steps",
        flush=True,
    )

    if n_dacc_fires == 0:
        raise PreflightDaccNotFiring(
            "Pre-flight pilot rollout observed ZERO dACC engagement "
            f"(n_dacc_fires=0 over {PREFLIGHT_EVAL_EPISODES * PREFLIGHT_STEPS_PER_EPISODE} "
            "eval steps, ARM_vs_on, seed="
            f"{seed}). agent._dacc_last_bundle never populated -- either "
            "self.dacc is None (use_dacc not set) or z_harm_a is None "
            "(affective harm stream not wired into the config). This is "
            "the exact V3-EXQ-475b failure mode "
            "(REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-475b_2026-08-01.md); "
            "the full 3-seed x 2-arm design would burn its whole compute "
            "budget on a structurally dead run. Refusing to proceed -- fix "
            "the config wiring before re-running."
        )
    return n_dacc_fires


def run_seed_arm(
    seed: int,
    *,
    vs_rollout_gating: bool,
    arm_id: str,
    env_kwargs: Optional[Dict[str, Any]] = None,
    warmup_episodes: int = WARMUP_EPISODES_DEFAULT,
    eval_episodes: int = EVAL_EPISODES_DEFAULT,
    steps_per_episode: int = STEPS_PER_EPISODE_DEFAULT,
) -> Dict[str, Any]:
    import random

    import numpy as np

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    kwargs = dict(env_kwargs or ENV_FISHTANK_KWARGS)
    env = make_env(seed, kwargs)
    # Explicit -- build_config's enable_affective_harm_stream=True branch
    # reads limb_damage_enabled off env_kwargs_or_default(env), which falls
    # back to the module-level ENV_FISHTANK_KWARGS default when
    # _exq_env_kwargs is unset. Stamping it here (rather than relying on the
    # caller always passing ENV_FISHTANK_KWARGS anyway) makes the value
    # actually used explicit rather than coincidental -- directly relevant
    # now that this flag controls whether z_harm_a gets wired up at all.
    env._exq_env_kwargs = kwargs
    gap4_arm = ArmSpec(
        "gap4_475a_conditions",
        gap4_operating=True,
        use_gabaergic_decay=True,
        use_pag_freeze_gate=True,
        use_broadcast_override=True,
    )
    cfg = _build_arm_config(env, gap4_arm, vs_rollout_gating=vs_rollout_gating)
    agent = REEAgent(cfg)
    label = f"seed={seed} arm={arm_id}"
    print(f"Seed {seed} Condition {arm_id}", flush=True)

    warmup_train(
        agent,
        env,
        num_episodes=warmup_episodes,
        steps_per_episode=steps_per_episode,
        label=label,
        progress_total_episodes=warmup_episodes + eval_episodes,
    )

    pe_series: List[float] = []
    bias_series: List[float] = []
    dacc_nonzero_steps = 0
    total_eval_steps = 0

    def on_post_step(*, agent, **kwargs) -> None:
        nonlocal dacc_nonzero_steps, total_eval_steps
        total_eval_steps += 1
        pe, bias_mag = _pe_and_bias_magnitude(agent)
        if pe is not None and bias_mag is not None:
            pe_series.append(pe)
            bias_series.append(bias_mag)
            if bias_mag > 1e-6:
                dacc_nonzero_steps += 1

    hooks = StepHooks(on_post_step=on_post_step)
    harness = StepHarness(agent, env, train_mode=False, hooks=hooks, seed=seed)
    agent.eval()

    for _ep in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        harness.reset()
        for _ in range(steps_per_episode):
            result = harness.step(obs_dict)
            obs_dict = result.next_obs_dict
            if result.done:
                break
        if (_ep + 1) == eval_episodes:
            print(
                f"  [train] {label} ep {warmup_episodes + _ep + 1}/{warmup_episodes + eval_episodes}",
                flush=True,
            )

    vs_gate_total_held = 0
    vs_gate_diag: Dict[str, Any] = {}
    gate = getattr(agent, "vs_rollout_gate", None)
    if gate is not None:
        vs_gate_diag = dict(gate.get_diagnostics())
        vs_gate_total_held = int(vs_gate_diag.get("vs_gate_total_held_e1", 0)) + int(
            vs_gate_diag.get("vs_gate_total_held_e2", 0)
        )

    rho = spearman(pe_series, bias_series)

    n_dacc_fires = len(pe_series)
    # Progress-instrumentation verdict line -- absent from 475b (it never
    # printed one), which the runner needs to advance runs_done / ETA. Local
    # criterion: this cell observed dACC engagement at all (n_dacc_fires>0).
    # The RUN-level pass/fail verdict (P1/P2 preconditions + the correlation
    # criteria) is computed once over all six cells in evaluate_q040c() below
    # and is what actually determines the manifest outcome.
    cell_ok = n_dacc_fires > 0
    print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)

    return {
        "seed": int(seed),
        "arm": arm_id,
        "vs_rollout_gating": bool(vs_rollout_gating),
        "total_eval_steps": int(total_eval_steps),
        "n_dacc_fires": n_dacc_fires,
        "dacc_bias_nonzero_steps": int(dacc_nonzero_steps),
        "vs_gate_total_held": int(vs_gate_total_held),
        "vs_gate_diagnostics": vs_gate_diag,
        "pe_series": pe_series,
        "bias_magnitude_series": bias_series,
        "rho_pe_vs_bias_magnitude": rho,
    }


def evaluate_q040c(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    on_rows = [r for r in rows if r["arm"] == VS_ON_ARM]
    off_rows = [r for r in rows if r["arm"] == VS_OFF_ARM]

    # P1: gate-firing precondition (ON arm only -- OFF arm has the gate
    # instantiated but never consulted, so it never accrues holds).
    on_gate_fired_seeds = sum(1 for r in on_rows if r["vs_gate_total_held"] > 0)
    p1_pass = on_gate_fired_seeds >= SEEDS_PASS_MIN

    # P2: dACC engagement precondition (both arms).
    on_dacc_fired_seeds = sum(1 for r in on_rows if r["dacc_bias_nonzero_steps"] > 0)
    off_dacc_fired_seeds = sum(1 for r in off_rows if r["dacc_bias_nonzero_steps"] > 0)
    p2_pass = on_dacc_fired_seeds >= SEEDS_PASS_MIN and off_dacc_fired_seeds >= SEEDS_PASS_MIN

    preconditions_met = bool(p1_pass and p2_pass)

    on_valid = [r["rho_pe_vs_bias_magnitude"] for r in on_rows if r["rho_pe_vs_bias_magnitude"] is not None]
    off_valid = [r["rho_pe_vs_bias_magnitude"] for r in off_rows if r["rho_pe_vs_bias_magnitude"] is not None]
    on_detect_seeds = sum(1 for rho in on_valid if abs(rho) >= CORR_DETECT_THRESH)
    off_null_seeds = sum(1 for rho in off_valid if abs(rho) < CORR_NULL_THRESH)

    c3_on_detects = len(on_valid) >= SEEDS_PASS_MIN and on_detect_seeds >= SEEDS_PASS_MIN
    c4_off_null = len(off_valid) >= SEEDS_PASS_MIN and off_null_seeds >= SEEDS_PASS_MIN

    verdict_pass = bool(preconditions_met and c3_on_detects and c4_off_null)

    if not preconditions_met:
        self_route = "substrate_not_ready_requeue"
    else:
        self_route = None

    # Non-degeneracy self-report: the underlying per-step bias-magnitude
    # readout must have genuine spread within each (seed, arm) group, else
    # any rho computed over it is a measurement artifact, not a finding.
    degeneracy = check_degeneracy(
        {
            "dacc_bias_magnitude_series": {
                "groups": [r["bias_magnitude_series"] for r in rows if r["bias_magnitude_series"]],
            },
        }
    )

    return {
        "pass": verdict_pass,
        "preconditions_met": preconditions_met,
        "self_route": self_route,
        "p1_gate_firing_pass": bool(p1_pass),
        "p1_on_gate_fired_seeds": int(on_gate_fired_seeds),
        "p2_dacc_engagement_pass": bool(p2_pass),
        "p2_on_dacc_fired_seeds": int(on_dacc_fired_seeds),
        "p2_off_dacc_fired_seeds": int(off_dacc_fired_seeds),
        "on_valid_rho_seeds": len(on_valid),
        "off_valid_rho_seeds": len(off_valid),
        "on_detect_seeds": int(on_detect_seeds),
        "off_null_seeds": int(off_null_seeds),
        "c3_on_detects_correlation": bool(c3_on_detects),
        "c4_off_correlation_null": bool(c4_off_null),
        "corr_detect_thresh": CORR_DETECT_THRESH,
        "corr_null_thresh": CORR_NULL_THRESH,
        "seeds_pass_min": SEEDS_PASS_MIN,
        **degeneracy,
    }


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main(dry_run: bool = False) -> Tuple[str, Path] | int:
    seeds = [SEEDS_DEFAULT[0]] if dry_run else SEEDS_DEFAULT
    warmup = 8 if dry_run else WARMUP_EPISODES_DEFAULT
    eval_eps = 2 if dry_run else EVAL_EPISODES_DEFAULT
    steps = 40 if dry_run else STEPS_PER_EPISODE_DEFAULT

    # Pre-flight smoke gate -- runs BEFORE the full seeds x arms design,
    # every invocation (dry-run and real). Raises uncaught on failure (fast
    # ERROR, not a full-budget FAIL). See module docstring and
    # PreflightDaccNotFiring.
    n_preflight_fires = _preflight_dacc_engagement_check(env_kwargs=ENV_FISHTANK_KWARGS)

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for seed in seeds:
        for arm_id, vs_on in ((VS_OFF_ARM, False), (VS_ON_ARM, True)):
            rows.append(
                run_seed_arm(
                    seed,
                    vs_rollout_gating=vs_on,
                    arm_id=arm_id,
                    env_kwargs=ENV_FISHTANK_KWARGS,
                    warmup_episodes=warmup,
                    eval_episodes=eval_eps,
                    steps_per_episode=steps,
                )
            )

    acceptance = evaluate_q040c(rows)
    acceptance["preflight_n_dacc_fires"] = int(n_preflight_fires)
    elapsed = time.time() - t0

    if acceptance["self_route"] is not None:
        outcome = "FAIL"
        evidence_direction = "non_contributory"
    else:
        outcome = "PASS" if acceptance["pass"] else "FAIL"
        evidence_direction = "supports" if outcome == "PASS" else "weakens"

    if dry_run:
        print(
            f"[{EXPERIMENT_TYPE}] dry-run outcome={outcome} "
            f"self_route={acceptance['self_route']} pass={acceptance['pass']} "
            f"n_dacc_fires_per_cell={[r['n_dacc_fires'] for r in rows]}",
            flush=True,
        )
        return 0

    # Strip the raw per-step series from the top-level acceptance echo (kept
    # per-row in per_run for full auditability) so the manifest root stays
    # small; the degeneracy check has already consumed them.
    per_run_rows = rows

    run_id = f"{EXPERIMENT_TYPE}_{_utc_compact()}_v3"
    out_dir = (
        REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": _utc_compact(),
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "acceptance": acceptance,
        "per_run": per_run_rows,
        "supersedes": SUPERSEDES_RUN_ID,
        "supersedes_queue_id": SUPERSEDES_QUEUE_ID,
        "elapsed_seconds": elapsed,
    }
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=list(seeds),
        script_path=Path(__file__),
    )
    return outcome, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)
    if result == 0:
        sys.exit(0)
    outcome, out_path = result
    emit_outcome(outcome=outcome, manifest_path=out_path, dry_run=bool(args.dry_run))
    sys.exit(0)
