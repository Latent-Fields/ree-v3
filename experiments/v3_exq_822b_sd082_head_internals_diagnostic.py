"""V3-EXQ-822b -- SD-082 HEAD-INTERNALS DIAGNOSTIC, ROOT-CAUSING V3-EXQ-822a.

WHY THIS RUN. V3-EXQ-822a (2026-07-26) re-ran V3-EXQ-822 with the SD-082 read-out
consumer correctly engaged (lateral_pfc_rule_readout_consumer=True,
lateral_pfc_train_rule_bias_head=True) and a strong, non-degenerate upstream chain
(on_rule_state_diff_mean=0.644, c1_pass on the pool side), but readiness gate (d)
propagation_non_vacuity STILL failed at exactly 0.0 for both arms -- the SECOND
consecutive identical failure of that gate (V3-EXQ-822 failed it before SD-082
existed). Autopsy failure_autopsy_batch-822a-826-817a-827_2026-07-26 Section 2
(confirmed) routed this to a substrate AMEND rather than a third blind re-queue
letter: the manifest carried NO head-internals diagnostics to confirm or rule out
the leading hypothesis (a dead-ReLU / insensitivity point in rule_bias_head's
UNTOUCHED first Linear+ReLU layer, ree_core/pfc/lateral_pfc_analog.py, absorbing
the rule_state signal before it reaches the SD-082-instrumented last layer). That
amend (ree-v3 63267dcf5d, "SD-082 AMEND: head-internals instrumentation") landed
LateralPFCConfig.capture_head_diagnostics (+ REEConfig.lateral_pfc_capture_head_
diagnostics) -- a no-op-default, bit-identical-when-on diagnostic read of
hidden_dead_relu_frac and rule_summary_magnitude_ratio via get_state(), plus four
unconditional weight-norm fields for per-phase tracking. THIS run is the follow-on
the amend's own note calls for: "a follow-on consumer-validation queue-experiment
session (V3-EXQ-822b ... with lateral_pfc_capture_head_diagnostics=True on the
trained arm) to actually root-cause the dead-ReLU/insensitivity hypothesis this
instrumentation now makes measurable."

EXPERIMENT_PURPOSE = diagnostic (not evidence). This run's job is to DISCRIMINATE
WHY propagation_non_vacuity reads exactly 0.0, not to re-litigate whether SD-078's
centering differentiates the executive rule_state (822a already asked that
question and, precisely because capture_head_diagnostics is a documented no-op on
the computed bias -- "bias_raw is bit-identical to the single-call form ...
diagnostic-only, no computation change" -- this run's PRIMARY metrics (rule_state_
diff, prop_delta_mean, readiness a-c) will deterministically reproduce 822a's
values bit-for-bit given the SAME seeds and design. Excluded from governance
confidence/conflict scoring per EXPERIMENT_PURPOSE; the evidence-bearing SD-078
question stays exactly where 822a left it. claim_ids retained for traceability
(this IS the intended SD-082 root-cause probe) but is not scored.

GOV-REUSE-1 (Step 2.4): the decisive readouts are hidden_dead_relu_frac and
rule_summary_magnitude_ratio at the P2 measurement window and at P0/P1/P2 phase
boundaries. capture_head_diagnostics did not exist before 2026-07-27 (this
session's own AMEND), so no manifest on any substrate_hash carries these fields.
Not recoverable -> run. Checked: no prior manifest under experiment_type
v3_exq_822* or v3_exq_785* etc. carries hidden_dead_relu_frac /
rule_summary_magnitude_ratio (grep of evidence/experiments/ confirms absence).

RE-DERIVE BRAKE (Step 2.5b): both prior SD-078 autopsies (816c-822, batch-822a-...)
carry re_derive_brake.fired=False -- their categories are precondition_unmet /
competence_implementation_gap, neither is substrate_ceiling under R3, so the
formal brake does not fire. Independently exempt regardless: this run is
diagnostic-purpose, discriminating WHY the ceiling/gate holds, which the skill's
own brake carve-out excludes from the re-derive loop.

DESIGN. Structurally IDENTICAL to V3-EXQ-822a (same 2 arms x 3 seeds x P0/P1/P2
schedule, same env, same thresholds) -- this is a controlled instrumentation-only
replication, not a new manipulation. The ONLY change: both arms now build with
lateral_pfc_capture_head_diagnostics=True (capture is a no-op on the bias
computation itself; it only causes compute_bias() to record two additional
latched diagnostic fields via an identical layer-by-layer unroll of the same
nn.Sequential). "Trained arm(s)" = BOTH arms, since 822a's P1 REINFORCE trains
each arm's own bias head identically (no shared weights) -- so both ARM_OFF and
ARM_ON get the diagnostic capture, letting this run also compare dead-ReLU
incidence between the collapsed (OFF) and differentiated (ON) rule-pool regimes.

INSTRUMENTATION ADDED OVER 822a:
  1. Per-P2-active-tick head diagnostics, aligned to the EXACT compute_bias() call
     whose delta feeds prop_delta_mean. hidden_dead_relu_frac /
     rule_summary_magnitude_ratio are LATCHED fields (get_state() returns
     whatever the most recent compute_bias() call wrote) -- the same latch class
     as e3_selector.last_*. compute_bias() is called TWICE per active tick inside
     the propagation-counterfactual (once on the real field rule_state, once on
     the zeroed one); reading get_state() after the SECOND call would silently
     attribute the zeroed-rule_state call's diagnostics to the real one. This
     script reads get_state() immediately after the FIRST (real-rule_state) call,
     before the zeroing counterfactual overwrites the latch -- see
     _prop_delta_and_flip_with_diag. No new sample-size hazard: this read rides
     the SAME E3-active-tick cadence 822a already uses for prop_delta_mean itself
     (one read per existing sample, not a new independent per-step read).
  2. Phase-boundary weight-norm snapshots (init / post_p0 / post_p1 / post_p2).
     The four weight-norm fields in get_state() are UNCONDITIONAL fresh parameter
     reads (no latching -- get_state() computes them from
     self.rule_bias_head[...].weight.norm() at call time), so these are safe to
     read at any point regardless of what the last compute_bias() call was. The
     "init" snapshot uses a synthetic positive-control forward pass (3 random
     candidate summaries) immediately after agent construction, before any
     training -- this doubles as the P0 readiness-assert positive control
     confirming the capture path returns finite, in-range numbers before P2
     aggregates are trusted.

READINESS (self-route substrate_not_ready_requeue if unmet -- mirrors 822a's a-c
so this diagnostic is only interpreted in the SAME regime that produced the
confirmed failure; propagation_non_vacuity itself (822a's gate (d)) is
deliberately NOT a gating precondition here -- it is the TARGET PHENOMENON this
run diagnoses, not a precondition for the diagnosis being interpretable):
  (a) z_world common-mode cone present (== 822a).
  (b) ARM_ON pool differentiated / ARM_OFF pinned (== 822a).
  (c) ARM_ON rule active on >= frac floor of P2 ticks (== 822a).
  (e) head-diagnostics capture positive-control: capture_head_diagnostics echoes
      True and returns finite, in-range values on the synthetic init probe, for
      every cell.
  (f) head-diagnostics P2 sample sufficiency: worst-cell (min across the 6
      arm x seed cells) count of per-tick head-diagnostic reads clears a floor,
      so the P2 aggregate is not starved.

LOAD-BEARING CRITERION (diagnostic purpose: PASS = the probe produced an
interpretable finding, independent of WHICH hypothesis it confirms):
  C1_head_diagnostics_interpretable = readiness (a,b,c,e,f) all met.

DIAGNOSTIC ROUTING (informational, reported once C1 holds -- not gating, not
mutually exclusive; all component booleans are recorded so a human/autopsy can
read the full picture, not just the priority-ordered headline label):
  - dead_relu_confirmed_root_cause: ON-arm P2 hidden_dead_relu_frac mean clears
    the high floor (>=0.90) -- most hidden units are dead, blocking the signal
    before the instrumented last layer.
  - head_untrained_last_layer_static: rule_bias_head_last_linear_weight_norm
    barely moved from init to post_p1 (delta < floor) -- REINFORCE never
    engaged the head, a DIFFERENT root cause than dead-ReLU.
  - magnitude_imbalance_confirmed_contributor: rule_summary_magnitude_ratio
    (finite samples) sits outside a broad sanity band -- the rule signal is
    negligible or overwhelming relative to the (SD-082-centered) candidate
    summaries.
  - propagation_vacuity_unexplained_by_head_internals: none of the above --
    points elsewhere (e.g. output-side tanh saturation, or a converged
    near-invariance that head-internals alone do not explain).

DV-SYMMETRY (Step 3 mandatory declaration). This run adds no new manipulation
over 822a (capture_head_diagnostics is a documented bit-identical no-op on the
computed bias), so 822a's own DV-symmetry declaration is unchanged and still
holds for rule_state_diff / prop_delta / rule_flip_frac. The NEW diagnostic DVs
(hidden_dead_relu_frac, rule_summary_magnitude_ratio, weight norms) are read
directly off model internals (a boolean ReLU-sign test and two L2 norms) with no
manipulation applied to them at all in this script -- there is no symmetry-group
question to declare because nothing here transforms them; they are passive reads.

EVIDENCE DIRECTIONS: diagnostic purpose, excluded from claim scoring.
evidence_direction_per_claim is reported for continuity with 822a's bookkeeping
(both "unknown" -- readiness gate (d) is not re-litigated as a verdict here) but
does not drive governance.

See ree-v3/CLAUDE.md "SD-082 AMEND: head-internals instrumentation" entry, SD-078,
SD-082, SD-033a, ARC-063;
experiments/v3_exq_822a_sd078_rule_selection_consumer.py (structurally identical
parent this instruments);
REE_assembly/evidence/planning/failure_autopsy_batch-822a-826-817a-827_2026-07-26.md
Section 2 (the routing this run executes).
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_822b_sd082_head_internals_diagnostic"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS = ["SD-078", "SD-082"]

# head_diagnostics_capture_active is a config-echo + finiteness check, not a
# hand-tuned numeric scoring predicate: `capture_head_diagnostics=True` is passed
# explicitly in _make_agent (both arms), so `get_state()["capture_head_diagnostics"]`
# is True by construction unless the three-site plumbing itself is broken (which
# validate_experiments.py / the SD-082 AMEND's own contracts, not this precondition,
# are the right place to catch). The `dead_relu_frac`/`magnitude_ratio` finiteness
# checks are likewise reachable by construction -- get_state() always returns a
# float from a completed forward pass on the synthetic positive-control input,
# never a value that depends on training having "worked". No hand-written scoring
# predicate narrower than the state it anchors to is involved.
ANCHOR_REACHABILITY_EXEMPT = (
    "head_diagnostics_capture_active is a config echo (capture_head_diagnostics="
    "True is set explicitly, not learned/tuned) plus a finiteness check on a "
    "deterministic forward pass; reachable by construction, no scoring predicate."
)
SUPERSEDES = None  # not a re-test of 822a's SD-078 question -- an instrumentation
                    # add-on run alongside it; 822a's own manifest is untouched.

SEEDS = [101, 202, 303]  # identical to 822a: controlled replication, not a new draw.
ARMS = ["ARM_OFF", "ARM_ON"]
P0_WARMUP_EPISODES = 60
P1_BIAS_TRAIN_EPISODES = 70
P2_MEASUREMENT_EPISODES = 40
STEPS_PER_EPISODE = 48

# --- Pre-registered thresholds (identical readiness a-c to 822a). ---
CONE_MIN_COSINE_FLOOR = 0.90
DIST_FLOOR = 0.05
MIN_LIVE_RULES = 2
OFF_MAX_LIVE_CEIL = 1
FRAC_ACTIVE_FLOOR = 0.10
SEED_PASS_FRACTION = 2.0 / 3.0
# 822a's gate (d) floor, reported for continuity (NOT a gating precondition here).
PROP_NONVAC_FLOOR = 1e-3

# --- NEW: head-diagnostics readiness + interpretation thresholds. ---
HEAD_DIAG_SAMPLE_FLOOR = 5          # worst-cell min P2 diagnostic reads.
DEAD_RELU_HIGH_FLOOR = 0.90         # >=90% of hidden units dead -> strong H1 support.
DEAD_RELU_MED_FLOOR = 0.50          # >=50% -> partial/weak H1 support.
LAST_LAYER_WEIGHT_DELTA_FLOOR = 1e-3  # below this, the head effectively never moved.
MAGNITUDE_RATIO_LOW = 1e-3           # outside [LOW, HIGH] -> magnitude-imbalance flag.
MAGNITUDE_RATIO_HIGH = 1e3

# P1 REINFORCE (mirror V3-EXQ-598b / 654f / 822a).
LR_LPFC_BIAS = 5e-4
REINFORCE_BATCH_SIZE = 32
OUTCOME_BUF_MAX = 512
POLICY_TEMPERATURE = 1.0
ADV_MIN_THRESHOLD = 0.005
EMA_DECAY = 0.9

ENV_KWARGS = dict(size=8, num_hazards=2, num_resources=6, use_proxy_fields=True)


def _build_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, centering: bool) -> REEAgent:
    """Matched-stack agent, identical to 822a's _make_agent plus
    lateral_pfc_capture_head_diagnostics=True on both arms (the diagnostic
    add-on; documented no-op on the computed bias itself)."""
    return REEAgent(REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        alpha_world=0.9,
        use_lateral_pfc_analog=True,
        lateral_pfc_train_rule_bias_head=True,
        lateral_pfc_rule_readout_consumer=True,
        lateral_pfc_capture_head_diagnostics=True,  # <-- the only new flag vs 822a.
        use_gated_policy=True,
        use_candidate_rule_field=True,
        crf_persist_rules_across_episode_reset=True,
        crf_mature_pool_dynamics=True,
        crf_availability_maintenance=True,
        crf_maintenance_floor=0.45,
        crf_maintenance_couple_to_theta=True,
        crf_tolerance_conflict_cap=3,
        crf_cue_centering=centering,
        crf_cue_baseline_alpha=0.02,
    ))


def _config_slice(centering: bool) -> Dict[str, Any]:
    return {
        "env": dict(ENV_KWARGS),
        "schedule": {"p0": P0_WARMUP_EPISODES, "p1": P1_BIAS_TRAIN_EPISODES,
                     "p2": P2_MEASUREMENT_EPISODES, "steps": STEPS_PER_EPISODE},
        "alpha_world": 0.9,
        "use_lateral_pfc_analog": True,
        "lateral_pfc_train_rule_bias_head": True,
        "lateral_pfc_rule_readout_consumer": True,
        "lateral_pfc_capture_head_diagnostics": True,
        "use_gated_policy": True,
        "use_candidate_rule_field": True,
        "crf_persist_rules_across_episode_reset": True,
        "crf_mature_pool_dynamics": True,
        "crf_availability_maintenance": True,
        "crf_maintenance_floor": 0.45,
        "crf_maintenance_couple_to_theta": True,
        "crf_tolerance_conflict_cap": 3,
        "crf_cue_centering": centering,
        "crf_cue_baseline_alpha": 0.02,
    }


def _cone_min_cosine(vecs: List[torch.Tensor]) -> float:
    if len(vecs) < 2:
        return 1.0
    m = torch.nn.functional.normalize(torch.stack(vecs).float(), dim=-1)
    c = m @ m.t()
    n = c.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    return float(c[iu[0], iu[1]].min())


def _mean_pairwise_one_minus_cosine(vecs: List[torch.Tensor]) -> float:
    vecs = [v for v in vecs if v is not None and float(v.norm()) > 1e-9]
    if len(vecs) < 2:
        return 0.0
    m = torch.nn.functional.normalize(torch.stack(vecs).float(), dim=-1)
    c = m @ m.t()
    n = c.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    return float((1.0 - c[iu[0], iu[1]]).mean())


def _candidate_summaries(agent: REEAgent, candidates) -> Optional[torch.Tensor]:
    summ = agent._candidate_world_summaries(candidates)
    return summ.detach() if summ is not None else None


def _head_diag_snapshot(agent: REEAgent) -> Dict[str, float]:
    """Unconditional (always-fresh, never-latched) weight-norm read."""
    st = agent.lateral_pfc.get_state()
    return {
        "capture_head_diagnostics": bool(st.get("capture_head_diagnostics", False)),
        "first_linear_weight_norm": float(st["rule_bias_head_first_linear_weight_norm"]),
        "first_linear_bias_norm": float(st["rule_bias_head_first_linear_bias_norm"]),
        "last_linear_weight_norm": float(st["rule_bias_head_last_linear_weight_norm"]),
        "last_linear_bias_norm": float(st["rule_bias_head_last_linear_bias_norm"]),
    }


def _init_positive_control(agent: REEAgent) -> Dict[str, Any]:
    """P0 readiness-assert positive control: force one real compute_bias() call
    with capture_head_diagnostics=True on a synthetic 3-candidate input BEFORE any
    training, so the phase-boundary "init" snapshot also confirms the capture path
    returns finite, in-range numbers (independent of whether training later moves
    anything)."""
    lpfc = agent.lateral_pfc
    wd = agent.config.latent.world_dim
    synth = torch.randn(3, wd, device=agent.device)
    with torch.no_grad():
        lpfc.compute_bias(synth)
        st = lpfc.get_state()
    snap = _head_diag_snapshot(agent)
    snap["dead_relu_frac"] = float(st["hidden_dead_relu_frac"])
    snap["magnitude_ratio"] = float(st["rule_summary_magnitude_ratio"])
    snap["dead_relu_frac_finite_in_range"] = (
        math.isfinite(snap["dead_relu_frac"]) and 0.0 <= snap["dead_relu_frac"] <= 1.0)
    snap["magnitude_ratio_finite"] = math.isfinite(snap["magnitude_ratio"])
    return snap


def _prop_delta_and_flip_with_diag(
    agent: REEAgent, summaries: torch.Tensor
) -> Optional[Tuple[float, bool, Dict[str, float]]]:
    """Mirrors 822a._prop_delta_and_flip, plus a head-diagnostics snapshot taken
    IMMEDIATELY after the real-rule_state compute_bias() call and BEFORE the
    zeroed counterfactual call overwrites the latched hidden_dead_relu_frac /
    rule_summary_magnitude_ratio fields (see module docstring, instrumentation
    point 1 -- the latch-ordering hazard this function is written to avoid)."""
    lpfc = getattr(agent, "lateral_pfc", None)
    if lpfc is None or summaries is None:
        return None
    try:
        with torch.no_grad():
            bias_field = lpfc.compute_bias(summaries).detach().clone()
            # Snapshot diagnostics from the REAL rule_state call NOW, before the
            # zeroed counterfactual below overwrites the latch.
            diag_state = lpfc.get_state()
            head_diag = {
                "dead_relu_frac": float(diag_state["hidden_dead_relu_frac"]),
                "magnitude_ratio": float(diag_state["rule_summary_magnitude_ratio"]),
            }
            saved = lpfc.rule_state.detach().clone()
            lpfc.rule_state.zero_()
            bias_zero = lpfc.compute_bias(summaries).detach().clone()
            lpfc.rule_state.copy_(saved)
        delta = float((bias_field - bias_zero).abs().mean().item())
        flip = int(bias_field.argmax().item()) != int(bias_zero.argmax().item())
        if not math.isfinite(delta):
            return None
        return delta, bool(flip), head_diag
    except Exception:
        return None


def _lpfc_reinforce_loss(agent: REEAgent,
                         outcome_buf: List[Tuple[torch.Tensor, int, float]],
                         baseline: float, device) -> torch.Tensor:
    if agent.lateral_pfc is None or len(outcome_buf) < 2:
        return torch.zeros(1, device=device)
    n = len(outcome_buf)
    idxs = np.random.choice(n, size=min(REINFORCE_BATCH_SIZE, n), replace=False)
    terms: List[torch.Tensor] = []
    for i in idxs:
        cand_features, sel_idx, ep_return = outcome_buf[int(i)]
        adv = ep_return - baseline
        if abs(adv) < ADV_MIN_THRESHOLD:
            continue
        bias = agent.lateral_pfc.compute_bias(cand_features.to(device))
        log_p = F.log_softmax(-bias / POLICY_TEMPERATURE, dim=0)
        terms.append(-adv * log_p[min(sel_idx, bias.shape[0] - 1)])
    if not terms:
        return torch.zeros(1, device=device)
    return torch.stack(terms).mean()


def _run_cell(arm: str, seed: int) -> Dict[str, Any]:
    centering = (arm == "ARM_ON")
    print(f"Seed {seed} Condition {arm}", flush=True)
    with arm_cell(seed, config_slice=_config_slice(centering),
                  script_path=Path(__file__)) as cell:
        env = _build_env(seed)
        agent = _make_agent(env, centering)
        crf = agent.candidate_rule_field
        wd = agent.config.latent.world_dim
        bias_opt = torch.optim.Adam(
            list(agent.lateral_pfc.bias_head_parameters()), lr=LR_LPFC_BIAS)

        head_diag_by_phase: Dict[str, Any] = {"init": _init_positive_control(agent)}

        total_eps = P0_WARMUP_EPISODES + P1_BIAS_TRAIN_EPISODES + P2_MEASUREMENT_EPISODES
        p2_start = P0_WARMUP_EPISODES + P1_BIAS_TRAIN_EPISODES

        reinforce_baseline = 0.0
        outcome_buf: List[Tuple[torch.Tensor, int, float]] = []

        zworlds: List[torch.Tensor] = []
        rule_state_samples: List[torch.Tensor] = []
        prop_deltas: List[float] = []
        flips: List[int] = []
        p2_dead_relu_samples: List[float] = []
        p2_magnitude_ratio_samples: List[float] = []
        n_magnitude_ratio_inf = 0
        p2_ticks = 0
        p2_active_ticks = 0
        max_live = 0

        for ep in range(total_eps):
            is_p1 = (P0_WARMUP_EPISODES <= ep < p2_start)
            is_p2 = (ep >= p2_start)
            phase = "P2" if is_p2 else ("P1" if is_p1 else "P0")
            _, obs = env.reset()
            agent.reset()
            ep_reward = 0.0
            ep_buf: List[Tuple[torch.Tensor, int]] = []

            for _step in range(STEPS_PER_EPISODE):
                latent = agent.sense(obs["body_state"], obs["world_state"])
                ticks = agent.clock.advance()
                e1 = (agent._e1_tick(latent) if ticks.get("e1_tick")
                      else torch.zeros(1, wd, device=agent.device))
                candidates = agent.generate_trajectories(latent, e1, ticks)

                p1_snap: Optional[torch.Tensor] = None
                if is_p1 and candidates and len(candidates) >= 2:
                    cs = _candidate_summaries(agent, candidates)
                    if cs is not None and torch.isfinite(cs).all():
                        p1_snap = cs.clone()

                action = agent.select_action(candidates, ticks)
                if action is None:
                    idx = int(np.random.randint(0, 4))
                    action = torch.zeros(1, 4, device=agent.device)
                    action[0, idx] = 1.0
                    agent._last_action = action
                committed_class = int(action[0].argmax().item())

                if is_p1 and p1_snap is not None:
                    sel = 0
                    for ci, c in enumerate(candidates):
                        if (getattr(c, "actions", None) is not None
                                and c.actions.shape[1] >= 1
                                and int(c.actions[:, 0, :].argmax(-1).reshape(-1)[0].item())
                                == committed_class):
                            sel = min(ci, p1_snap.shape[0] - 1)
                            break
                    ep_buf.append((p1_snap, sel))

                if is_p2 and ticks.get("e3_tick", False):
                    p2_ticks += 1
                    if len(zworlds) < 200:
                        zworlds.append(latent.z_world.detach().reshape(-1).clone())
                    st = crf.get_state()
                    n_active = int(st.get("crf_n_active_last", 0))
                    max_live = max(max_live, int(st.get("crf_n_live", len(crf._rules))))
                    if n_active >= 1:
                        p2_active_ticks += 1
                        rs = agent.lateral_pfc.rule_state.detach().reshape(-1).clone()
                        if float(rs.norm()) > 1e-9:
                            rule_state_samples.append(rs)
                        if candidates and len(candidates) >= 2:
                            summ = _candidate_summaries(agent, candidates)
                            if summ is not None and torch.isfinite(summ).all():
                                pf = _prop_delta_and_flip_with_diag(agent, summ)
                                if pf is not None:
                                    delta, flip, head_diag = pf
                                    prop_deltas.append(delta)
                                    flips.append(1 if flip else 0)
                                    p2_dead_relu_samples.append(head_diag["dead_relu_frac"])
                                    mr = head_diag["magnitude_ratio"]
                                    if math.isfinite(mr):
                                        p2_magnitude_ratio_samples.append(mr)
                                    else:
                                        n_magnitude_ratio_inf += 1

                _, _h, done, _info, obs = env.step(int(action.argmax(dim=-1).item()))
                if is_p1:
                    ep_reward += float(_h)
                if done:
                    break

            if is_p1:
                reinforce_baseline = (EMA_DECAY * reinforce_baseline
                                      + (1.0 - EMA_DECAY) * ep_reward)
                for cand_features, sel in ep_buf:
                    outcome_buf.append((cand_features, sel, ep_reward))
                if len(outcome_buf) > OUTCOME_BUF_MAX:
                    outcome_buf = outcome_buf[-OUTCOME_BUF_MAX:]
                l_loss = _lpfc_reinforce_loss(
                    agent, outcome_buf, reinforce_baseline, agent.device)
                if l_loss.requires_grad:
                    bias_opt.zero_grad()
                    l_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.lateral_pfc.bias_head_parameters(), 1.0)
                    bias_opt.step()

            # Phase-boundary weight-norm snapshots (always-fresh, never latched).
            if (ep + 1) == P0_WARMUP_EPISODES:
                head_diag_by_phase["post_p0"] = _head_diag_snapshot(agent)
            elif (ep + 1) == p2_start:
                head_diag_by_phase["post_p1"] = _head_diag_snapshot(agent)
            elif (ep + 1) == total_eps:
                head_diag_by_phase["post_p2"] = _head_diag_snapshot(agent)

            if (ep + 1) % 20 == 0 or (ep + 1) == total_eps:
                print(f"  [train] crf seed={seed} arm={arm} phase={phase} "
                      f"ep {ep+1}/{total_eps} live={len(crf._rules)} "
                      f"minted={crf._n_minted}", flush=True)

        st = crf.get_state()
        rule_state_diff = _mean_pairwise_one_minus_cosine(rule_state_samples)
        prop_delta_mean = statistics.fmean(prop_deltas) if prop_deltas else 0.0
        rule_flip_frac = (sum(flips) / len(flips)) if flips else 0.0
        frac_active = (p2_active_ticks / p2_ticks) if p2_ticks else 0.0

        dead_relu_frac_p2_mean = (statistics.fmean(p2_dead_relu_samples)
                                   if p2_dead_relu_samples else 0.0)
        dead_relu_frac_p2_worst = max(p2_dead_relu_samples) if p2_dead_relu_samples else 0.0
        magnitude_ratio_p2_mean = (statistics.fmean(p2_magnitude_ratio_samples)
                                    if p2_magnitude_ratio_samples else 0.0)
        magnitude_ratio_p2_median = (statistics.median(p2_magnitude_ratio_samples)
                                      if p2_magnitude_ratio_samples else 0.0)
        last_layer_weight_delta_init_to_p1 = float(
            head_diag_by_phase.get("post_p1", {}).get("last_linear_weight_norm", 0.0)
            - head_diag_by_phase["init"]["last_linear_weight_norm"])

        row = {
            "arm": arm,
            "seed": seed,
            "rule_state_diff": float(rule_state_diff),
            "prop_delta_mean": float(prop_delta_mean),
            "rule_flip_frac": float(rule_flip_frac),
            "crf_max_pairwise_rule_dist": float(crf.max_pairwise_rule_distance()),
            "crf_max_live_rules": int(max_live),
            "crf_live_rules_final": int(len(crf._rules)),
            "crf_n_minted": int(crf._n_minted),
            "crf_frac_active_p2": float(frac_active),
            "crf_baseline_allocated": crf._baseline is not None,
            "n_rule_state_samples": len(rule_state_samples),
            "n_prop_samples": len(prop_deltas),
            "zworld_cone_min_cosine": _cone_min_cosine(zworlds),
            "n_zworld_sampled": len(zworlds),
            # --- NEW: head-internals diagnostics ---
            "n_head_diag_samples": len(p2_dead_relu_samples),
            "hidden_dead_relu_frac_p2_mean": float(dead_relu_frac_p2_mean),
            "hidden_dead_relu_frac_p2_worst": float(dead_relu_frac_p2_worst),
            "rule_summary_magnitude_ratio_p2_mean": float(magnitude_ratio_p2_mean),
            "rule_summary_magnitude_ratio_p2_median": float(magnitude_ratio_p2_median),
            "n_magnitude_ratio_inf": int(n_magnitude_ratio_inf),
            "head_diag_by_phase": head_diag_by_phase,
            "last_layer_weight_delta_init_to_p1": last_layer_weight_delta_init_to_p1,
        }
        cell.stamp(row)
    passed = (row["rule_state_diff"] > 0.05) if centering else \
             (row["crf_max_live_rules"] <= OFF_MAX_LIVE_CEIL)
    print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)
    return row


def _worst_cell(rows: List[Dict[str, Any]], key: str, mode: str = "min"):
    fn = min if mode == "min" else max
    r = fn(rows, key=lambda x: x[key])
    return float(r[key]), f"{r['arm']}/seed{r['seed']}"


def run_experiment() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        for arm in ARMS:
            rows.append(_run_cell(arm, seed))

    off = [r for r in rows if r["arm"] == "ARM_OFF"]
    on = [r for r in rows if r["arm"] == "ARM_ON"]

    # READINESS (a)-(c): identical to 822a (worst-cell / seed-fraction gates).
    cone_worst, cone_cell = _worst_cell(rows, "zworld_cone_min_cosine", mode="min")
    cone_present = cone_worst >= CONE_MIN_COSINE_FLOOR
    on_diff_pool = (sum(1 for r in on if r["crf_max_pairwise_rule_dist"] > DIST_FLOOR
                        and r["crf_max_live_rules"] >= MIN_LIVE_RULES) / len(on)
                    ) >= SEED_PASS_FRACTION
    off_pinned_pool = (sum(1 for r in off if r["crf_max_live_rules"] <= OFF_MAX_LIVE_CEIL)
                       / len(off)) >= SEED_PASS_FRACTION
    on_active = (sum(1 for r in on if r["crf_frac_active_p2"] >= FRAC_ACTIVE_FLOOR)
                 / len(on)) >= SEED_PASS_FRACTION

    # READINESS (e): head-diagnostics capture positive control -- worst cell.
    init_flags = [r["head_diag_by_phase"]["init"] for r in rows]
    capture_worst = min(
        1.0 if (f["capture_head_diagnostics"] and f["dead_relu_frac_finite_in_range"]
                and f["magnitude_ratio_finite"]) else 0.0
        for f in init_flags)
    capture_active = capture_worst >= 1.0
    capture_offending = next(
        (f"{r['arm']}/seed{r['seed']}" for r, f in zip(rows, init_flags)
         if not (f["capture_head_diagnostics"] and f["dead_relu_frac_finite_in_range"]
                 and f["magnitude_ratio_finite"])), "none")

    # READINESS (f): P2 head-diagnostic sample sufficiency -- worst cell.
    n_diag_worst, n_diag_cell = _worst_cell(rows, "n_head_diag_samples", mode="min")
    samples_sufficient = n_diag_worst >= HEAD_DIAG_SAMPLE_FLOOR

    ready = (cone_present and on_diff_pool and off_pinned_pool and on_active
             and capture_active and samples_sufficient)

    # Gate (d), reported for continuity with 822a -- NOT gating here.
    off_by_seed = {r["seed"]: r for r in off}
    prop_nonvac_seeds = 0
    for r in on:
        o = off_by_seed.get(r["seed"])
        if o is None:
            continue
        if (r["prop_delta_mean"] > PROP_NONVAC_FLOOR
                and abs(r["prop_delta_mean"] - o["prop_delta_mean"]) > PROP_NONVAC_FLOOR):
            prop_nonvac_seeds += 1
    prop_nonvac = (prop_nonvac_seeds / len(on)) >= SEED_PASS_FRACTION

    # --- Diagnostic routing (informational; component booleans all recorded). ---
    on_dead_relu_p2_mean = statistics.fmean(r["hidden_dead_relu_frac_p2_mean"] for r in on)
    off_dead_relu_p2_mean = statistics.fmean(r["hidden_dead_relu_frac_p2_mean"] for r in off)
    on_last_layer_delta = statistics.fmean(r["last_layer_weight_delta_init_to_p1"] for r in on)

    finite_ratios_on = [x for r in on for x in [r["rule_summary_magnitude_ratio_p2_median"]]
                        if x > 0.0]
    magnitude_extreme = any(
        (x < MAGNITUDE_RATIO_LOW or x > MAGNITUDE_RATIO_HIGH) for x in finite_ratios_on
    ) if finite_ratios_on else False

    dead_relu_confirmed = on_dead_relu_p2_mean >= DEAD_RELU_HIGH_FLOOR
    dead_relu_partial = (not dead_relu_confirmed) and (on_dead_relu_p2_mean >= DEAD_RELU_MED_FLOOR)
    head_untrained = abs(on_last_layer_delta) < LAST_LAYER_WEIGHT_DELTA_FLOOR

    diagnostic_flags = {
        "dead_relu_confirmed": dead_relu_confirmed,
        "dead_relu_partial_contributor": dead_relu_partial,
        "head_untrained_last_layer_static": head_untrained,
        "magnitude_imbalance_confirmed_contributor": magnitude_extreme,
    }

    if not ready:
        label = "substrate_not_ready_requeue"
    elif dead_relu_confirmed:
        label = "dead_relu_confirmed_root_cause"
    elif head_untrained:
        label = "head_untrained_last_layer_static"
    elif magnitude_extreme:
        label = "magnitude_imbalance_confirmed_contributor"
    elif dead_relu_partial:
        label = "dead_relu_partial_contributor"
    else:
        label = "propagation_vacuity_unexplained_by_head_internals"

    # C1: load-bearing criterion for THIS diagnostic's own PASS/FAIL.
    c1 = ready
    overall = c1

    # Non-degeneracy: capture must be genuinely engaged (not vacuously true) --
    # every cell echoed the flag AND showed at least some variation in the P2
    # dead-ReLU samples (a value that never varies across hundreds of ticks and
    # every cell would be a wiring-bug smell, not physically implausible on its
    # own but worth flagging rather than silently trusting).
    on_dead_relu_all_samples = [r["hidden_dead_relu_frac_p2_worst"] for r in rows]
    shows_variation = len(set(round(x, 6) for x in on_dead_relu_all_samples)) > 1
    c1_non_degenerate = bool(
        capture_active and all(r["n_head_diag_samples"] > 0 for r in rows)
        and shows_variation)

    metrics = {
        "on_rule_state_diff_mean": statistics.fmean(r["rule_state_diff"] for r in on),
        "off_rule_state_diff_mean": statistics.fmean(r["rule_state_diff"] for r in off),
        "on_prop_delta_mean": statistics.fmean(r["prop_delta_mean"] for r in on),
        "off_prop_delta_mean": statistics.fmean(r["prop_delta_mean"] for r in off),
        "readiness_cone_present": cone_present,
        "readiness_on_diff_pool": on_diff_pool,
        "readiness_off_pinned_pool": off_pinned_pool,
        "readiness_on_active": on_active,
        "readiness_capture_active": capture_active,
        "readiness_samples_sufficient": samples_sufficient,
        "readiness_prop_nonvac_reported_only": prop_nonvac,
        "zworld_cone_min_cosine_worst": cone_worst,
        "zworld_cone_worst_cell": cone_cell,
        "n_head_diag_samples_worst": n_diag_worst,
        "n_head_diag_samples_worst_cell": n_diag_cell,
        "on_hidden_dead_relu_frac_p2_mean": on_dead_relu_p2_mean,
        "off_hidden_dead_relu_frac_p2_mean": off_dead_relu_p2_mean,
        "on_last_layer_weight_delta_init_to_p1_mean": on_last_layer_delta,
        "on_rule_summary_magnitude_ratio_p2_median_values": finite_ratios_on,
        "diagnostic_flags": diagnostic_flags,
        "c1_pass": c1,
    }
    result: Dict[str, Any] = {
        "outcome": "PASS" if overall else "FAIL",
        "evidence_direction": "unknown",
        "evidence_direction_per_claim": {"SD-078": "unknown", "SD-082": "unknown"},
        "metrics": metrics,
        "per_seed_rows": rows,
        "arm_results": rows,
        "interpretation": {
            "label": label,
            "preconditions": [
                {"name": "zworld_common_mode_cone_present",
                 "description": "min pairwise z_world cosine clears the cone floor (== 822a).",
                 "measured": cone_worst, "threshold": CONE_MIN_COSINE_FLOOR,
                 "direction": "lower", "offending_cell": cone_cell, "met": cone_present},
                {"name": "on_pool_differentiated",
                 "description": "ARM_ON differentiated / ARM_OFF pinned on majority seeds (== 822a).",
                 "measured": float(sum(1 for r in on
                                       if r["crf_max_pairwise_rule_dist"] > DIST_FLOOR
                                       and r["crf_max_live_rules"] >= MIN_LIVE_RULES)),
                 "threshold": float(math.ceil(SEED_PASS_FRACTION * len(on))),
                 "direction": "lower", "met": on_diff_pool},
                {"name": "on_rule_active_p2",
                 "description": "ARM_ON rule active on >= frac floor of P2 ticks (== 822a).",
                 "measured": statistics.fmean(r["crf_frac_active_p2"] for r in on),
                 "threshold": FRAC_ACTIVE_FLOOR, "direction": "lower", "met": on_active},
                {"name": "head_diagnostics_capture_active",
                 "description": ("capture_head_diagnostics echoes True and the synthetic "
                                 "init positive-control returns finite, in-range values, "
                                 "for every cell."),
                 "measured": capture_worst, "threshold": 1.0, "direction": "lower",
                 "control": "synthetic 3-candidate forward pass immediately post-construction",
                 "offending_cell": capture_offending, "met": capture_active},
                {"name": "head_diag_samples_sufficient",
                 "description": "worst-cell count of P2 per-tick head-diagnostic reads clears the floor.",
                 "measured": n_diag_worst, "threshold": HEAD_DIAG_SAMPLE_FLOOR,
                 "direction": "lower", "offending_cell": n_diag_cell,
                 "met": samples_sufficient},
            ],
            "criteria": [
                {"name": "C1_head_diagnostics_interpretable", "load_bearing": True,
                 "passed": c1},
            ],
            "criteria_non_degenerate": {
                "C1_head_diagnostics_interpretable": c1_non_degenerate,
            },
            "diagnostic_flags": diagnostic_flags,
        },
    }
    if not ready:
        result["non_degenerate"] = False
        reasons = []
        if not cone_present:
            reasons.append("z_world common-mode cone absent")
        if not on_diff_pool:
            reasons.append("ON pool not differentiated")
        if not off_pinned_pool:
            reasons.append("OFF pool not pinned")
        if not on_active:
            reasons.append("ON rule not active enough in P2")
        if not capture_active:
            reasons.append("head-diagnostics capture positive control failed")
        if not samples_sufficient:
            reasons.append("P2 head-diagnostic samples starved")
        result["degeneracy_reason"] = (
            "readiness unmet ({}); head-internals diagnosis is not interpretable in "
            "this run; substrate_not_ready_requeue".format("; ".join(reasons)))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    global SEEDS, P0_WARMUP_EPISODES, P1_BIAS_TRAIN_EPISODES, P2_MEASUREMENT_EPISODES
    if args.dry_run:
        SEEDS = [101]
        P0_WARMUP_EPISODES = 3
        P1_BIAS_TRAIN_EPISODES = 3
        P2_MEASUREMENT_EPISODES = 3

    result = run_experiment()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    full_config = {
        "seeds": SEEDS, "arms": ARMS,
        "p0": P0_WARMUP_EPISODES, "p1": P1_BIAS_TRAIN_EPISODES,
        "p2": P2_MEASUREMENT_EPISODES, "steps_per_episode": STEPS_PER_EPISODE,
        "cone_min_cosine_floor": CONE_MIN_COSINE_FLOOR,
        "dist_floor": DIST_FLOOR, "min_live_rules": MIN_LIVE_RULES,
        "off_max_live_ceil": OFF_MAX_LIVE_CEIL, "frac_active_floor": FRAC_ACTIVE_FLOOR,
        "prop_nonvac_floor": PROP_NONVAC_FLOOR,
        "head_diag_sample_floor": HEAD_DIAG_SAMPLE_FLOOR,
        "dead_relu_high_floor": DEAD_RELU_HIGH_FLOOR,
        "dead_relu_med_floor": DEAD_RELU_MED_FLOOR,
        "last_layer_weight_delta_floor": LAST_LAYER_WEIGHT_DELTA_FLOOR,
        "magnitude_ratio_low": MAGNITUDE_RATIO_LOW,
        "magnitude_ratio_high": MAGNITUDE_RATIO_HIGH,
        "seed_pass_fraction": SEED_PASS_FRACTION,
        "lr_lpfc_bias": LR_LPFC_BIAS,
        "lateral_pfc_capture_head_diagnostics": True,
        "arm_config_slice_off": _config_slice(False),
        "arm_config_slice_on": _config_slice(True),
        "supersedes_note": ("not a supersession of v3_exq_822a_sd078_rule_selection_consumer "
                             "-- an instrumentation-only diagnostic companion run"),
    }
    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "evidence_direction": result["evidence_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "outcome": result["outcome"],
        "timestamp_utc": ts,
        "metrics": result["metrics"],
        "per_seed_rows": result["per_seed_rows"],
        "arm_results": result["arm_results"],
        "interpretation": result["interpretation"],
    }
    if "non_degenerate" in result:
        manifest["non_degenerate"] = result["non_degenerate"]
        manifest["degeneracy_reason"] = result["degeneracy_reason"]

    out_path = write_flat_manifest(
        manifest,
        Path(__file__).resolve().parents[2] / "REE_assembly" / "evidence" / "experiments",
        dry_run=args.dry_run,
        config=full_config,
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=t0,
    )
    m = result["metrics"]
    print(f"outcome: {result['outcome']}", flush=True)
    print(f"label: {result['interpretation']['label']}", flush=True)
    print(f"diagnostic_flags: {m['diagnostic_flags']}", flush=True)
    print(f"ON dead_relu_p2={m['on_hidden_dead_relu_frac_p2_mean']:.4f} "
          f"OFF dead_relu_p2={m['off_hidden_dead_relu_frac_p2_mean']:.4f} "
          f"ON last_layer_delta={m['on_last_layer_weight_delta_init_to_p1_mean']:.6f}",
          flush=True)
    print(f"C1={m['c1_pass']} ready(cone={m['readiness_cone_present']} "
          f"diffpool={m['readiness_on_diff_pool']} pinned={m['readiness_off_pinned_pool']} "
          f"active={m['readiness_on_active']} capture={m['readiness_capture_active']} "
          f"samples={m['readiness_samples_sufficient']}) "
          f"prop_nonvac_reported_only={m['readiness_prop_nonvac_reported_only']}", flush=True)
    print(f"wrote: {out_path}", flush=True)
    return result, out_path, args.dry_run


if __name__ == "__main__":
    _result, _out_path, _dry_run = main()
    _outcome_raw = str(_result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(_out_path),
        dry_run=_dry_run,
    )
