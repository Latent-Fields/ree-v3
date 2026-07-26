"""V3-EXQ-822 -- SD-078 DOWNSTREAM CONSUMER: does centering the CandidateRuleField
context key let the agent's EXECUTIVE RULE-SELECTION become context-conditioned and
REACH the action-bias, so the rule channel can steer behaviour differently in different
contexts (which a one-rule pool cannot)?

PROMOTION GATE (experiment_purpose = evidence). SD-078 is candidate_substrate_landed:
its INTERNAL cue geometry was validated in isolation by V3-EXQ-806 (raw z_world caps the
ARC-063 CandidateRuleField at ONE rule; centering recovers a differentiated pool --
on_dist_mean 1.67 vs 0.0, on_max_live 7.67 vs 1.0). What that run did NOT establish is
that the differentiated pool changes an AGENT-LEVEL decision. This run supplies exactly
that missing link and is the gate to promoting SD-078 candidate_substrate_landed ->
provisional. It does NOT edit claims.yaml; the run + review does.

WHAT IS NEW OVER V3-EXQ-806 (why this is not recoverable from it). 806 measured a POOL
property: how different the minted rules are FROM EACH OTHER (crf_max_pairwise_rule_dist)
and the max concurrent live count. It did NOT exercise the CONSUMER step -- gate_and_select
recruiting DIFFERENT active rules for DIFFERENT contexts, the resulting SD-033a rule_state
that GOVERNS the lateral-PFC action bias, and whether that reaches and moves the
per-candidate action score. Those readouts are absent from every 806 manifest.
GOV-REUSE-1 (Step 2.4): the decisive readouts are cross-context rule_state differentiation,
the rule-attributable action-bias it induces, and the fraction of ticks it flips the
bias-preferred action, under a CENTERED context key with a TRAINED bias head.
crf_cue_centering did not exist before 2026-07-22 and 806 recorded none of these, so no
manifest on any substrate_hash can carry them. Not recoverable -> run.

THE BEHAVIOURAL BOTTLENECK. Under SD-008 the raw z_world context sits in a ~0.98-cosine
cone, so the mint-block fires against the first rule for every later context -- the pool
is structurally capped at ONE rule (806). The SD-033a lateral-PFC rule_state is an EMA
over the active-rule stack, so with one rule it is CONTEXT-INVARIANT: the agent applies
the SAME executive bias everywhere and cannot condition action on context through the
rule channel. Centering the context key de-collapses the pool; distinct contexts then
recruit distinct active rules -> a context-varying rule_state -> a context-varying action
bias. This run tests whether that chain actually fires.

DESIGN. Two arms, same seeds; single swept variable = crf_cue_centering (ARM_OFF False =
the pinned SD-078 defect one-rule pool, ARM_ON True = the differentiated pool). Both arms
run the identical full agent with the CandidateRuleField built, matured across episodes
(crf_persist + mature-pool + availability-maintenance levers), the SD-033a
LateralPFCAnalog wired to consume the field's rule_state (crf_source), and the bias head
UN-ZEROED + trained. Phases:
  P0 (warmup): mature + maintain the rule pool; bias head not trained.
  P1 (bias-head train, encoder frozen): outcome-coupled REINFORCE on the SD-033a
     rule_bias_head (the V3-EXQ-598b / 654f pattern) so the matured rule_state reaches a
     NON-ZERO per-candidate E3 score-bias. No measurement.
  P2 (all frozen): the measurement window.
In P2, on each tick the rule-attributable action bias is isolated by the propagation
counterfactual (654f pattern): delta = mean_k |compute_bias(field rule_state)_k -
compute_bias(zeroed rule_state)_k| -- the per-candidate action-bias the rule_state
contributes THIS tick -- and a bias-argmax FLIP is recorded when zeroing the rule_state
changes which candidate the bias channel prefers. The rule_state vector itself is sampled
on active ticks to measure cross-context differentiation.

DEPENDENT VARIABLES (decision-level):
  * PRIMARY (C1): cross-context rule_state differentiation = mean pairwise (1 - cosine)
    of the rule_state vectors sampled across P2 ticks. ARM_OFF's one-rule rule_state is
    context-invariant (low differentiation); ARM_ON's context-varying active sets give
    high differentiation. Gated as a paired-by-seed EFFECT SIZE (ON - OFF) plus an
    absolute floor.
  * BEHAVIOURAL REACH (readiness / non-vacuity): ARM_ON prop_delta_mean > floor AND
    differs from ARM_OFF -- the differentiated rule_state actually MOVES the per-candidate
    action bias (else the executive differentiation is a dead internal signal). Mirrors
    654f's C1d propagation non-vacuity.
  * DECISION CONSEQUENCE (secondary, reported): rule_flip_frac = fraction of P2 ticks the
    rule_state changes the bias-preferred candidate action-class.

READINESS (self-route substrate_not_ready_requeue, never a false does_not_support):
  (a) z_world common-mode cone present (min pairwise cosine >= floor -- the SAME statistic
      the SD-008 hazard is DEFINED by, on the run's own z_world stream). Below it there is
      no common mode to subtract and a null says nothing about centering.
  (b) ARM_ON pool differentiated (crf_max_pairwise_rule_dist > floor AND >= 2 live rules)
      while ARM_OFF is pinned (<= 1 live rule) -- centering did its upstream job, so a
      behavioural null is attributable to the consumer, not a dead upstream.
  (c) ARM_ON rule active on >= frac floor of P2 ticks -- the rule_state is sampled often
      enough for the differentiation DV to be non-starved.
  (d) propagation non-vacuity above -- the trained head + matured field reach the action
      bias. If the head did not train (e.g. no reward signal), this fails and the run
      self-routes rather than reporting a false null.

DV-SYMMETRY (Step 3 mandatory declaration, per arm). ARM_OFF / ARM_ON DV =
rule_state_differentiation, prop_delta, rule_flip_frac. Symmetry group of the primary:
any orthogonal rotation / permutation of the rule_state coordinate frame (mean pairwise
1 - cosine is invariant to a shared rotation of all sampled vectors) and permutation of
the P2 tick order (the pairwise aggregate is symmetric over samples). The manipulation is
NOT invariant under it: centering changes WHICH rules exist and WHICH gate-activate per
context, hence the SET of rule_state vectors sampled -- and a differentiation aggregate
over a set is not preserved by a manipulation that changes the set's cardinality and
directions. The manipulation is NOT a broadcast constant on a rank/argmax readout either:
the flip DV compares argmax(compute_bias(field)) vs argmax(compute_bias(zeroed)), and
zeroing the rule_state is not a uniform additive shift on the per-candidate bias (the
head is a nonlinear function of concat([rule_state, candidate_summary])), so the accept/
reject of the flip genuinely changes.

EVIDENCE DIRECTIONS (both declared):
  PASS  (readiness met + C1 paired rule_state-differentiation lift) -> SUPPORTS SD-078:
        centering produces context-conditioned executive rule-selection that reaches and
        moves the action bias -- a real downstream behavioural consequence.
  FAIL, readiness MET (C1 fails) -> DOES_NOT_SUPPORT: the cue geometry is fixed (806) but
        the differentiated pool does NOT yield a context-differentiated executive state
        reaching the action -- the downstream benefit is absent on this DV.
  FAIL, readiness UNMET (cone absent / pool not differentiated / not active / head did not
        reach the bias) -> substrate_not_ready_requeue, non_contributory
        (non_degenerate=false); re-queue at an adequate regime. Never a weakens.

See ree-v3/CLAUDE.md SD-078 + SD-033a + ARC-063 entries;
experiments/v3_exq_806_sd078_centered_rule_field_context_key.py (internal-geometry
validation this consumes);
experiments/v3_exq_654f_arc062_gapb_rule_apprehension_behavioural_falsifier.py (the
bias-head REINFORCE + propagation-counterfactual patterns reused here).
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

EXPERIMENT_TYPE = "v3_exq_822_sd078_rule_selection_consumer"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS = ["SD-078"]

SEEDS = [101, 202, 303]
ARMS = ["ARM_OFF", "ARM_ON"]
P0_WARMUP_EPISODES = 60      # mature + maintain the rule pool; bias head not trained
P1_BIAS_TRAIN_EPISODES = 70  # frozen-encoder bias-head REINFORCE (598b/654f pattern);
                             # long enough to un-zero the head so prop_delta > floor
P2_MEASUREMENT_EPISODES = 40 # all frozen; behavioural measurement
STEPS_PER_EPISODE = 48

# --- Pre-registered thresholds (defined HERE, never inferred post-hoc). ---
# READINESS (a): the common-mode cone SD-078 repairs must be present.
CONE_MIN_COSINE_FLOOR = 0.90
# READINESS (b): ON pool differentiated / OFF pinned.
DIST_FLOOR = 0.05            # ON crf_max_pairwise_rule_dist must exceed this
MIN_LIVE_RULES = 2           # ON must hold >= this many concurrent rules
OFF_MAX_LIVE_CEIL = 1        # OFF must stay pinned at <= this many rules
# READINESS (c): ON rule active often enough to sample rule_state.
FRAC_ACTIVE_FLOOR = 0.10
# READINESS (d): propagation non-vacuity -- the rule_state reaches the action bias.
PROP_NONVAC_FLOOR = 1e-3
# C1 (PRIMARY): paired-by-seed rule_state differentiation lift + absolute floor.
DIFF_DELTA_MARGIN = 0.05     # ON - OFF cross-context rule_state (1-cosine) differentiation
DIFF_FLOOR = 0.05            # AND ON absolute differentiation must clear this
SEED_PASS_FRACTION = 2.0 / 3.0

# P1 REINFORCE (mirror V3-EXQ-598b / 654f).
LR_LPFC_BIAS = 5e-4
REINFORCE_BATCH_SIZE = 32
OUTCOME_BUF_MAX = 512
POLICY_TEMPERATURE = 1.0
ADV_MIN_THRESHOLD = 0.005
EMA_DECAY = 0.9

# Env: light hazard load so the REINFORCE reward (per-step harm signal) has variance to
# un-zero the bias head, while z_world still collapses (SD-008) so centering has work.
ENV_KWARGS = dict(size=8, num_hazards=2, num_resources=6, use_proxy_fields=True)


def _build_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, centering: bool) -> REEAgent:
    """Matched-stack agent; the ONLY varied flag is crf_cue_centering.

    Both arms build the CandidateRuleField (matured + maintained across episodes) and the
    SD-033a LateralPFCAnalog with the bias head UN-ZEROED + trainable, so the rule_state
    reaches a non-zero per-candidate E3 score-bias and the head can be trained in P1.
    """
    return REEAgent(REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        alpha_world=0.9,  # SD-008: z_world fidelity (matches 806)
        # SD-033a lateral-PFC consumer + trainable bias head.
        use_lateral_pfc_analog=True,
        lateral_pfc_train_rule_bias_head=True,
        use_gated_policy=True,
        # CandidateRuleField, matured + maintained (matches 806 + general maintenance
        # levers so the ON pool actually gate-activates in P2; inert in the OFF one-rule
        # pool). NOT the e2-world-forward context source -- centering the DEFAULT raw
        # z_world context key is exactly the SD-078 manipulation 806 validated.
        use_candidate_rule_field=True,
        crf_persist_rules_across_episode_reset=True,
        crf_mature_pool_dynamics=True,
        crf_availability_maintenance=True,
        crf_maintenance_floor=0.45,
        crf_maintenance_couple_to_theta=True,
        crf_tolerance_conflict_cap=3,
        # --- The ONLY swept variable ---
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
    """Cross-context differentiation of the sampled rule_state vectors."""
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


def _prop_delta_and_flip(agent: REEAgent, summaries: torch.Tensor
                         ) -> Optional[Tuple[float, bool]]:
    """Isolate the rule_state's contribution to the per-candidate action bias.

    Returns (mean |bias(field) - bias(zeroed)|, argmax(field) != argmax(zeroed)).
    Mirrors 654f._propagation_counterfactual_delta; best-effort (None on failure).
    """
    lpfc = getattr(agent, "lateral_pfc", None)
    if lpfc is None or summaries is None:
        return None
    try:
        with torch.no_grad():
            bias_field = lpfc.compute_bias(summaries).detach().clone()
            saved = lpfc.rule_state.detach().clone()
            lpfc.rule_state.zero_()
            bias_zero = lpfc.compute_bias(summaries).detach().clone()
            lpfc.rule_state.copy_(saved)
        delta = float((bias_field - bias_zero).abs().mean().item())
        flip = int(bias_field.argmax().item()) != int(bias_zero.argmax().item())
        if not math.isfinite(delta):
            return None
        return delta, bool(flip)
    except Exception:
        return None


def _lpfc_reinforce_loss(agent: REEAgent,
                         outcome_buf: List[Tuple[torch.Tensor, int, float]],
                         baseline: float, device) -> torch.Tensor:
    """REINFORCE on the SD-033a bias head (mirror 654f._lpfc_reinforce_loss)."""
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

        total_eps = P0_WARMUP_EPISODES + P1_BIAS_TRAIN_EPISODES + P2_MEASUREMENT_EPISODES
        p2_start = P0_WARMUP_EPISODES + P1_BIAS_TRAIN_EPISODES

        # P1 REINFORCE state.
        reinforce_baseline = 0.0
        outcome_buf: List[Tuple[torch.Tensor, int, float]] = []

        # P2 accumulators.
        zworlds: List[torch.Tensor] = []
        rule_state_samples: List[torch.Tensor] = []
        prop_deltas: List[float] = []
        flips: List[int] = []
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

                # P1: snapshot candidate summaries + selected index for REINFORCE.
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

                # Measure ONLY on E3 ticks: the CRF gate + SD-033a rule_state update
                # fire inside the E3 selection path (heartbeat.e3_steps_per_tick cadence,
                # default 10). Reading crf.get_state() / rule_state on a held (non-E3)
                # tick re-samples LATCHED values -- the pseudo-replication hazard. (Here
                # duplicates would only DEFLATE rule_state_diff, so this is conservative,
                # but E3-gating keeps the denominator honest.)
                if is_p2 and ticks.get("e3_tick", False):
                    p2_ticks += 1
                    if len(zworlds) < 200:
                        zworlds.append(latent.z_world.detach().reshape(-1).clone())
                    st = crf.get_state()
                    n_active = int(st.get("crf_n_active_last", 0))
                    max_live = max(max_live, int(st.get("crf_n_live", len(crf._rules))))
                    if n_active >= 1:
                        p2_active_ticks += 1
                        # Sample the rule_state on active ticks (non-zero, context-keyed).
                        rs = agent.lateral_pfc.rule_state.detach().reshape(-1).clone()
                        if float(rs.norm()) > 1e-9:
                            rule_state_samples.append(rs)
                        # Isolate + measure the rule-attributable action bias.
                        if candidates and len(candidates) >= 2:
                            summ = _candidate_summaries(agent, candidates)
                            if summ is not None and torch.isfinite(summ).all():
                                pf = _prop_delta_and_flip(agent, summ)
                                if pf is not None:
                                    prop_deltas.append(pf[0])
                                    flips.append(1 if pf[1] else 0)

                _, _h, done, _info, obs = env.step(int(action.argmax(dim=-1).item()))
                if is_p1:
                    ep_reward += float(_h)
                if done:
                    break

            # P1 end-of-episode REINFORCE update on the bias head.
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

            if (ep + 1) % 20 == 0 or (ep + 1) == total_eps:
                print(f"  [train] crf seed={seed} arm={arm} phase={phase} "
                      f"ep {ep+1}/{total_eps} live={len(crf._rules)} "
                      f"minted={crf._n_minted}", flush=True)

        st = crf.get_state()
        rule_state_diff = _mean_pairwise_one_minus_cosine(rule_state_samples)
        prop_delta_mean = statistics.fmean(prop_deltas) if prop_deltas else 0.0
        rule_flip_frac = (sum(flips) / len(flips)) if flips else 0.0
        frac_active = (p2_active_ticks / p2_ticks) if p2_ticks else 0.0

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
        }
        cell.stamp(row)
    # Per-cell verdict for progress display.
    passed = (row["rule_state_diff"] > DIFF_FLOOR) if centering else \
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

    # READINESS (a): the common-mode cone SD-078 repairs must be present (worst cell).
    cone_worst, cone_cell = _worst_cell(rows, "zworld_cone_min_cosine", mode="min")
    cone_present = cone_worst >= CONE_MIN_COSINE_FLOOR
    # READINESS (b): ON pool differentiated / OFF pinned (majority of seeds).
    on_diff_pool = (sum(1 for r in on if r["crf_max_pairwise_rule_dist"] > DIST_FLOOR
                        and r["crf_max_live_rules"] >= MIN_LIVE_RULES) / len(on)
                    ) >= SEED_PASS_FRACTION
    off_pinned_pool = (sum(1 for r in off if r["crf_max_live_rules"] <= OFF_MAX_LIVE_CEIL)
                       / len(off)) >= SEED_PASS_FRACTION
    # READINESS (c): ON rule active often enough to sample rule_state (majority).
    on_active = (sum(1 for r in on if r["crf_frac_active_p2"] >= FRAC_ACTIVE_FLOOR)
                 / len(on)) >= SEED_PASS_FRACTION
    # READINESS (d): propagation non-vacuity -- ON rule_state reaches the action bias AND
    # differs from OFF (paired by seed), the 654f C1d guard.
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

    ready = (cone_present and on_diff_pool and off_pinned_pool and on_active
             and prop_nonvac)

    # C1 (PRIMARY): paired-by-seed cross-context rule_state differentiation lift.
    c1_seed_pass = 0
    for r in on:
        o = off_by_seed.get(r["seed"])
        if o is None:
            continue
        delta = r["rule_state_diff"] - o["rule_state_diff"]
        if delta > DIFF_DELTA_MARGIN and r["rule_state_diff"] > DIFF_FLOOR:
            c1_seed_pass += 1
    frac_c1 = c1_seed_pass / len(on)
    c1 = frac_c1 >= SEED_PASS_FRACTION

    overall = ready and c1
    if not ready:
        label = "substrate_not_ready_requeue"
        direction = "unknown"
    elif overall:
        label = "sd078_centering_yields_context_conditioned_executive_bias"
        direction = "supports"
    else:
        label = "sd078_centering_did_not_differentiate_executive_bias"
        direction = "does_not_support"

    metrics = {
        "on_rule_state_diff_mean": statistics.fmean(r["rule_state_diff"] for r in on),
        "off_rule_state_diff_mean": statistics.fmean(r["rule_state_diff"] for r in off),
        "on_prop_delta_mean": statistics.fmean(r["prop_delta_mean"] for r in on),
        "off_prop_delta_mean": statistics.fmean(r["prop_delta_mean"] for r in off),
        "on_rule_flip_frac_mean": statistics.fmean(r["rule_flip_frac"] for r in on),
        "off_rule_flip_frac_mean": statistics.fmean(r["rule_flip_frac"] for r in off),
        "on_max_live_mean": statistics.fmean(r["crf_max_live_rules"] for r in on),
        "off_max_live_mean": statistics.fmean(r["crf_max_live_rules"] for r in off),
        "on_frac_active_p2_mean": statistics.fmean(r["crf_frac_active_p2"] for r in on),
        "frac_c1_rule_state_diff": frac_c1,
        "c1_pass": c1,
        "readiness_cone_present": cone_present,
        "readiness_on_diff_pool": on_diff_pool,
        "readiness_off_pinned_pool": off_pinned_pool,
        "readiness_on_active": on_active,
        "readiness_prop_nonvac": prop_nonvac,
        "zworld_cone_min_cosine_worst": cone_worst,
        "zworld_cone_worst_cell": cone_cell,
    }
    result: Dict[str, Any] = {
        "outcome": "PASS" if overall else "FAIL",
        "evidence_direction": direction,
        "metrics": metrics,
        "per_seed_rows": rows,
        "arm_results": rows,
        "interpretation": {
            "label": label,
            "preconditions": [
                {"name": "zworld_common_mode_cone_present",
                 "description": ("min pairwise z_world cosine clears the cone floor -- the "
                                 "geometry SD-078 exists to repair."),
                 "measured": cone_worst, "threshold": CONE_MIN_COSINE_FLOOR,
                 "direction": "lower",
                 "control": ("worst cell across all arms/seeds ({}), on the run's own "
                             "z_world stream".format(cone_cell)),
                 "offending_cell": cone_cell, "met": cone_present},
                {"name": "on_pool_differentiated",
                 "description": ("ARM_ON must recover a differentiated pool (dist>floor, "
                                 ">=2 live) on a majority of seeds -- centering did its "
                                 "upstream job so a behavioural null is the consumer's."),
                 "measured": float(sum(1 for r in on
                                       if r["crf_max_pairwise_rule_dist"] > DIST_FLOOR
                                       and r["crf_max_live_rules"] >= MIN_LIVE_RULES)),
                 "threshold": float(math.ceil(SEED_PASS_FRACTION * len(on))),
                 "direction": "lower", "met": on_diff_pool},
                {"name": "on_rule_active_p2",
                 "description": ("ARM_ON rule active on >= frac floor of P2 ticks so the "
                                 "rule_state differentiation DV is not starved."),
                 "measured": statistics.fmean(r["crf_frac_active_p2"] for r in on),
                 "threshold": FRAC_ACTIVE_FLOOR, "direction": "lower", "met": on_active},
                {"name": "propagation_non_vacuity",
                 "description": ("ARM_ON rule_state reaches the per-candidate action bias "
                                 "(prop_delta>floor) AND differs from ARM_OFF paired by "
                                 "seed -- the trained head made the executive state "
                                 "behavioural."),
                 "measured": statistics.fmean(r["prop_delta_mean"] for r in on),
                 "threshold": PROP_NONVAC_FLOOR, "direction": "lower", "met": prop_nonvac},
            ],
            "criteria": [
                {"name": "C1_on_rule_state_differentiates", "load_bearing": True, "passed": c1},
            ],
            "criteria_non_degenerate": {
                # C1 discriminates only if OFF is genuinely pinned (one-rule) and ran the
                # OFF path (no baseline allocated), and ON allocated a baseline.
                "C1_on_rule_state_differentiates": bool(
                    off_pinned_pool
                    and all(not r["crf_baseline_allocated"] for r in off)
                    and all(r["crf_baseline_allocated"] for r in on)),
            },
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
        if not prop_nonvac:
            reasons.append("ON rule_state did not reach the action bias (head untrained)")
        result["degeneracy_reason"] = (
            "readiness unmet ({}); the executive-bias DV says nothing about centering; "
            "substrate_not_ready_requeue".format("; ".join(reasons)))
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
        "diff_delta_margin": DIFF_DELTA_MARGIN, "diff_floor": DIFF_FLOOR,
        "seed_pass_fraction": SEED_PASS_FRACTION,
        "lr_lpfc_bias": LR_LPFC_BIAS,
        "arm_config_slice_off": _config_slice(False),
        "arm_config_slice_on": _config_slice(True),
    }
    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "evidence_direction": result["evidence_direction"],
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
    print(f"ON rs_diff={m['on_rule_state_diff_mean']:.4f} prop={m['on_prop_delta_mean']:.4f} "
          f"flip={m['on_rule_flip_frac_mean']:.3f} live={m['on_max_live_mean']:.2f} | "
          f"OFF rs_diff={m['off_rule_state_diff_mean']:.4f} prop={m['off_prop_delta_mean']:.4f} "
          f"live={m['off_max_live_mean']:.2f}", flush=True)
    print(f"C1={m['c1_pass']} ready(cone={m['readiness_cone_present']} "
          f"diffpool={m['readiness_on_diff_pool']} pinned={m['readiness_off_pinned_pool']} "
          f"active={m['readiness_on_active']} prop={m['readiness_prop_nonvac']}) "
          f"cone_worst={m['zworld_cone_min_cosine_worst']:.4f}", flush=True)
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
