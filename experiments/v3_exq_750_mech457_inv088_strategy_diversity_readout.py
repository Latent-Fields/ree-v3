#!/opt/local/bin/python3
"""V3-EXQ-750 -- MECH-457 / INV-088 single-step STRATEGY-DIVERSITY readout on the GOV-FANOUT-1
representation x reward-density 2x2 factorial (742 / 747 / 748 / 749).

DIAGNOSTIC discrimination probe (experiment_purpose=diagnostic; claim_ids tag relevance only
-> excluded from governance confidence/conflict scoring). PROMOTES / DEMOTES NOTHING. Routes to
/failure-autopsy for adjudication before any governance action. MECH-457 stays candidate /
v3_pending; INV-088 stays candidate / pending_substrate_reconfirmation.

WHY. The GOV-FANOUT-1 portfolio measured only COMPETENCE (foraging yield). Its 2x2 (742 =
(z_world, sparse)):
                     sparse foraging RL          dense teacher (shaped RL)
    z_world  (R0)    742: FAIL                    748: PASS
    raw 5x5  (R1)    747: FAIL                    749: PASS
showed competence recovery is gated by the TEACHER-DENSITY axis (dense clears the 1.0 floor),
NOT the representation axis. That leaves the monostrategy question the competence readout cannot
see (memory: monostrategy-representation-ceiling-root): INV-088 (world_goal_evaluator bounded by
z_world differentiation) predicts an under-differentiated representation caps how many DISTINCT
strategies the policy can express (Le Lan generalize-vs-approximate ceiling) -- a CEILING that
diversity pressure cannot break. The competence legs cannot distinguish "z_world buys strategy
diversity" from "z_world is merely a competent input", because 748 vs 749 forage the same.

THIS PROBE. Rerun the four RL cells verbatim through the shared mech457_fanout machinery
(deterministic under fixed seeds; the completed manifests saved only aggregate competence, no
per-step actions and no policy weights, and experiments/_lib/arm_fingerprint is Phase-0
emit-only, so nothing was reusable -- a fresh rerun is the minimal faithful path), and co-emit,
per cell, a SINGLE-STEP strategy-diversity readout ALONGSIDE competence:
  * H_greedy  -- Shannon entropy (bits) of the empirical distribution of the greedy action
                 taken across all eval steps (order-1 single-step strategy mixture).
  * effective_actions = 2 ** H_greedy (order-1 Hill number; the "distinct-strategy count").
  * distinct_actions_used -- count of actions with non-zero frequency (order-0 Hill number).
  * mean_softmax_entropy -- mean per-step policy softmax entropy (bits); a HEDGING control that
                 separates state-differentiated strategy (varied argmax across states) from mere
                 per-step stochastic hedging (a flat softmax that still argmaxes the same way).

SINGLE-STEP ONLY (per feedback_dont_queue_commitment_dependent_behavioural): every metric is a
marginal over single-step action choices. NO multi-step committed-trajectory / segment-diversity
metric -- that is premature while the substrate cannot sustain multi-step action-commitment.

DECISIVE TEST (the 748-vs-749 matched-competence pair). 748 (z_world, dense) and 749 (raw-view,
dense) both PASS competence, so they are matched on the competence axis and differ ONLY in
representation. Does strategy diversity rise along the representation axis at MATCHED competence?
  * repr gates diversity (|H_greedy(z_world_dense) - H_greedy(raw_dense)| > margin, majority of
    seeds) -> a representation-ceiling CHANNEL for monostrategy is real, decoupled from the
    competence gain -> SELF-ROUTE representation_gates_diversity_independent_of_competence; ESCALATE
    to registering an ARC-065 child (representation-ceiling channel) or an INV-088 behavioural
    consequence in claims.yaml (adjudicated by /failure-autopsy first).
  * repr does NOT gate diversity at matched competence -> the representation-ceiling root is NOT
    supported by this readout; diversity (if it varies) tracks the teacher/competence axis, not
    representation -> SELF-ROUTE diversity_not_gated_by_representation_at_matched_competence.
The full 2x2 diversity table + representation/teacher main-effect deltas are recorded so the
autopsy can read the decoupling directly (this probe does not fix the direction of the INV-088
prediction for this substrate pairing -- it MEASURES it).

READINESS (P0 readiness-assert; SAME statistic the verdict routes on). The load-bearing criterion
routes on a between-condition SPREAD of H_greedy, so readiness asserts a between-policy H_greedy
SPREAD on known-different positive controls: random_walk (maximally action-diverse by construction)
vs local_view_greedy (a competent, context-dependent forager -> structured, lower diversity). If
that anchor spread is below floor the INSTRUMENT cannot register a diversity difference -> self-route
substrate_not_ready_requeue (NEVER a diversity verdict). A second readiness leg keeps the foraging
env solvable (local_view_greedy + greedy_oracle clear the 1.0 forage floor, as in 742/747). The
decisive contrast additionally requires the dense pair to be matched-competent (both clear the
floor) -- recorded as a precondition; if a rerun diverges and a dense cell is sub-floor the probe
routes matched_competence_precondition_unmet, never a diversity verdict.

evidence_direction = "unknown" (a DIAGNOSTIC that discriminates WHETHER a representation-ceiling
channel exists; it does not directly score MECH-457 or INV-088 -- the discrimination verdict lives
in interpretation.label / discrimination_verdict, adjudicated by /failure-autopsy).

ethics_preflight:
  involves_negative_valence: false
  involves_suffering_like_state: false
  involves_self_model: false
  involves_inescapability_or_helplessness: false
  involves_offline_replay_over_harm: false
  involves_social_mind_or_language: false
  involves_human_data_or_clinical_context: false
  decision: allow

SLEEP DRIVER: none (no sleep loop; use_sleep_loop / sws_enabled / rem_enabled all OFF).

Shared machinery: experiments/_lib/mech457_fanout.py (z_world + raw-view AC trainers, anchors,
readiness scaffolding); the eval is instrumented by a recorder that mirrors each leg's eval
policy argmax EXACTLY (single forward per step -> competence byte-comparable to 742/747/748/749)
while capturing the per-step greedy action + softmax entropy. ASCII-only in all runtime strings.
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.capability_eval import (  # noqa: E402
    COMPETENCE_RESOURCE_FLOOR,
    LocalViewGreedyPolicy,
    OraclePolicy,
    Policy,
    RandomPolicy,
    evaluate_seed,
)
from experiments._metrics import check_degeneracy  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
import experiments._lib.mech457_fanout as fan  # noqa: E402
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734  # noqa: E402
import experiments.v3_exq_742_mech457_actor_critic_onoff as x742  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_750_mech457_inv088_strategy_diversity_readout"
QUEUE_ID = "V3-EXQ-750"
CLAIM_IDS: List[str] = ["MECH-457", "INV-088"]
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# --- pre-registered thresholds (constants, NOT derived from the run's own statistics) --------
DIVERSITY_MARGIN_BITS = 0.20          # min |delta H_greedy| to call a representation effect present
# Instrument-calibration readiness: H_greedy(random_walk ~= max) - H_greedy(constant-action == 0)
# must exceed this. random_walk is maximally action-diverse by construction, the constant-action
# calibration control is exactly monostrategy (H_greedy == 0), so a working instrument yields a
# spread ~= log2(action_dim) ~= 2.32; only a pinned/degenerate H_greedy instrument falls below.
# This is the SAME statistic the verdict routes on (a between-policy H_greedy spread) on controls
# with KNOWN-different single-step diversity, so it cannot false-fail on legitimately-similar arms.
READINESS_DIVERSITY_SPREAD_FLOOR = 1.0   # bits
ACTION_DIM = 5                        # causal grid world action space (up/down/left/right/stay)

# --- the four RL cells of the representation x teacher-density 2x2 (the RL arm from each leg) --
# rep in {z_world, raw_view}; teacher in {sparse, dense}; shaping_coef 0.0 = sparse, SHAPING_COEF = dense.
TREATMENT_CELLS: Tuple[Dict[str, Any], ...] = (
    {"arm_id": "ac_zworld_sparse_rl", "rep": "z_world",  "teacher": "sparse",
     "path": "zworld", "shaping_coef": 0.0,             "reference_leg": "V3-EXQ-742"},
    {"arm_id": "ac_zworld_dense_rl",  "rep": "z_world",  "teacher": "dense",
     "path": "zworld", "shaping_coef": fan.SHAPING_COEF, "reference_leg": "V3-EXQ-748"},
    {"arm_id": "ac_rawview_sparse_rl", "rep": "raw_view", "teacher": "sparse",
     "path": "raw",    "shaping_coef": 0.0,             "reference_leg": "V3-EXQ-747"},
    {"arm_id": "ac_rawview_dense_rl",  "rep": "raw_view", "teacher": "dense",
     "path": "raw",    "shaping_coef": fan.SHAPING_COEF, "reference_leg": "V3-EXQ-749"},
)
TREATMENT_IDS: Tuple[str, ...] = tuple(c["arm_id"] for c in TREATMENT_CELLS)
ARM_ORDER: Tuple[str, ...] = TREATMENT_IDS + fan.ANCHOR_ARMS


# ---------------------------------------------------------------------------
# Single-step diversity metrics over a recorded greedy-action sequence.
# ---------------------------------------------------------------------------
def _entropy_bits(counts: List[int]) -> float:
    """Shannon entropy (bits) of an empirical count histogram; 0.0 for empty/degenerate."""
    total = float(sum(counts))
    if total <= 0.0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def _diversity_from_actions(actions: List[int], step_softmax_ent_bits: List[float]) -> Dict[str, Any]:
    """Single-step strategy-diversity readout for one cell/seed.

    * H_greedy: entropy (bits) of the empirical greedy-action distribution (order-1 single-step
      strategy mixture across visited states).
    * effective_actions = 2 ** H_greedy (order-1 Hill number).
    * distinct_actions_used: order-0 Hill number (non-zero action bins).
    * mean_softmax_entropy: mean per-step policy softmax entropy (bits); None for anchors that
      expose no logits.
    """
    counts = [0] * ACTION_DIM
    for a in actions:
        if 0 <= int(a) < ACTION_DIM:
            counts[int(a)] += 1
    h_greedy = _entropy_bits(counts)
    mean_sm = (float(sum(step_softmax_ent_bits) / len(step_softmax_ent_bits))
               if step_softmax_ent_bits else None)
    return {
        "n_steps": int(sum(counts)),
        "action_histogram": counts,
        "h_greedy_bits": round(h_greedy, 6),
        "effective_actions": round(2.0 ** h_greedy, 6),
        "distinct_actions_used": int(sum(1 for c in counts if c > 0)),
        "mean_softmax_entropy_bits": (round(mean_sm, 6) if mean_sm is not None else None),
    }


# ---------------------------------------------------------------------------
# Diversity-recording eval policy. Wraps a leg eval policy; records the per-step greedy action
# and (for the AC arms) the per-step softmax entropy. For the AC arms a single `step_fn` mirrors
# the leg eval policy's argmax logic EXACTLY (one forward per step) so competence stays
# byte-comparable; the anchor policies expose no logits, so their diversity is action-histogram
# only. reset / post_step / name delegate to the inner leg policy (identical eval semantics).
# ---------------------------------------------------------------------------
class _DiversityRecorder(Policy):
    def __init__(self, inner: Policy, label: str,
                 step_fn: Optional[Any] = None) -> None:
        self.inner = inner
        self.name = label
        self._step_fn = step_fn          # (env, obs_dict) -> (action_int, logits|None); None for anchors
        self.actions: List[int] = []
        self.step_softmax_ent_bits: List[float] = []

    def reset(self, env: Any) -> None:
        self.inner.reset(env)

    def post_step(self, env: Any, info: Dict[str, Any], obs_dict: Dict[str, Any]) -> None:
        self.inner.post_step(env, info, obs_dict)

    def act(self, env: Any, obs_dict: Dict[str, Any]) -> int:
        if self._step_fn is None:
            # anchor: the real policy decides; record only the realised action.
            action = int(self.inner.act(env, obs_dict))
            self.actions.append(action)
            return action
        with torch.no_grad():
            action, logits = self._step_fn(env, obs_dict)
        action = int(action)
        self.actions.append(action)
        if logits is not None:
            p = torch.softmax(logits.reshape(-1).float(), dim=-1)
            ent_bits = float(-(p * torch.log2(p.clamp_min(1e-12))).sum().item())
            self.step_softmax_ent_bits.append(ent_bits)
        return action


def _zworld_step_fn(agent: Any):
    """Mirror x742.ActorCriticEvalPolicy.act EXACTLY (single _sense + one deterministic AC step)."""
    def _step(env: Any, obs_dict: Dict[str, Any]) -> Tuple[int, Optional[torch.Tensor]]:
        latent = x742._sense(agent, obs_dict)
        step = agent.actor_critic_step(latent, deterministic=True)
        logits = step.logits
        if logits is not None and not torch.isfinite(logits).all():
            return int(np.random.randint(0, int(env.action_dim))), None
        return int(step.action.reshape(-1)[0].item()), logits
    return _step


def _rawview_step_fn(ac: Any):
    """Mirror fan.RawViewACEvalPolicy.act EXACTLY (single forward on the raw 5x5 view)."""
    def _step(env: Any, obs_dict: Dict[str, Any]) -> Tuple[int, Optional[torch.Tensor]]:
        logits, _v, _phi, _psi = ac.forward(fan._rawview_tensor(obs_dict))
        if not torch.isfinite(logits).all():
            return int(np.random.randint(0, int(env.action_dim))), None
        return int(torch.argmax(logits, dim=-1).reshape(-1)[0].item()), logits
    return _step


# ---------------------------------------------------------------------------
# Cell runners: train (verbatim leg recipe) then diversity-instrumented eval.
# ---------------------------------------------------------------------------
def _run_treatment_cell(cell: Dict[str, Any], env_kwargs: Dict[str, Any], seed: int,
                        p0: int, rl_eps: int, eval_eps: int, steps: int) -> Dict[str, Any]:
    arm_id = cell["arm_id"]
    if cell["path"] == "zworld":
        warm_env = x734._make_env(seed, env_kwargs)
        agent = fan.make_zworld_agent(warm_env)
        fan.warmup_zworld(agent, warm_env, seed=seed, p0=p0, steps=steps)
        train_env = x734._make_env(seed, env_kwargs)
        guard = fan.train_zworld_ac_shaped(
            agent, train_env, seed=seed, n_episodes=rl_eps, steps=steps,
            arm_label=arm_id, denom=rl_eps, shaping_coef=float(cell["shaping_coef"]),
        )
        inner = x742.ActorCriticEvalPolicy(agent, arm_id)
        step_fn = _zworld_step_fn(agent)
    else:  # raw view
        ac = fan.make_rawview_ac()
        train_env = x734._make_env(seed, env_kwargs)
        guard = fan.train_rawview_ac_rl(
            ac, train_env, seed=seed, n_episodes=rl_eps, steps=steps,
            arm_label=arm_id, denom=rl_eps, shaping_coef=float(cell["shaping_coef"]),
        )
        inner = fan.RawViewACEvalPolicy(ac, arm_id)
        step_fn = _rawview_step_fn(ac)

    eval_env = x734._make_env(seed, env_kwargs)
    recorder = _DiversityRecorder(inner, arm_id, step_fn=step_fn)
    row = evaluate_seed(recorder, eval_env, eval_eps, steps)
    row["mean_train_forage_recent"] = guard.get("mean_train_forage_recent", 0.0)
    row["diversity"] = _diversity_from_actions(recorder.actions, recorder.step_softmax_ent_bits)
    return row


class _ConstantActionPolicy(Policy):
    """Instrument-calibration control: always returns a fixed action -> H_greedy == 0 by
    construction (a perfectly monostrategy policy). Used ONLY to prove the diversity instrument
    registers the LOW end of the H_greedy range; never a scored scientific arm."""

    def __init__(self, action: int = 0) -> None:
        self._action = int(action)
        self.name = "constant_action_calibration"

    def act(self, env: Any, obs_dict: Dict[str, Any]) -> int:
        return int(self._action)


def _calibration_diversity(env_kwargs: Dict[str, Any], seed: int, eval_eps: int,
                           steps: int) -> Dict[str, Any]:
    """Run the constant-action calibration control (no training, not an arm) and return its
    single-step diversity readout -- H_greedy MUST be 0."""
    cal_env = x734._make_env(seed, env_kwargs)
    recorder = _DiversityRecorder(_ConstantActionPolicy(0), "constant_action_calibration", step_fn=None)
    evaluate_seed(recorder, cal_env, eval_eps, steps)
    return _diversity_from_actions(recorder.actions, recorder.step_softmax_ent_bits)


def _anchor_policy(arm_id: str, seed: int) -> Policy:
    """Construct the anchor eval policy for an anchor arm_id -- EXACTLY as fan.run_anchor_cell
    (LocalViewGreedyPolicy(seed) / OraclePolicy() [no seed] / RandomPolicy(seed))."""
    if arm_id == "local_view_greedy":
        return LocalViewGreedyPolicy(seed=seed)
    if arm_id == "greedy_oracle":
        return OraclePolicy()
    if arm_id == "random_walk":
        return RandomPolicy(seed)
    raise ValueError(f"unknown anchor arm_id: {arm_id}")


def _run_anchor_cell(arm_id: str, env_kwargs: Dict[str, Any], seed: int,
                     eval_eps: int, steps: int) -> Dict[str, Any]:
    """Anchor eval, mirroring fan.run_anchor_cell but through the diversity recorder so we
    capture the action histogram (anchors expose no logits -> softmax entropy stays empty)."""
    anchor_env = x734._make_env(seed, env_kwargs)
    recorder = _DiversityRecorder(_anchor_policy(arm_id, seed), arm_id, step_fn=None)
    row = evaluate_seed(recorder, anchor_env, eval_eps, steps)
    row["diversity"] = _diversity_from_actions(recorder.actions, recorder.step_softmax_ent_bits)
    return row


def _arm_config_slice(arm_id: str, env_kwargs: Dict[str, Any], p0: int, rl_eps: int,
                      eval_eps: int, steps: int) -> Dict[str, Any]:
    base = {
        "arm_id": arm_id, "rung_id": fan.RUNG_ID, "env_kwargs": dict(env_kwargs),
        "eval_episodes": int(eval_eps), "steps_per_episode": int(steps),
    }
    cell = next((c for c in TREATMENT_CELLS if c["arm_id"] == arm_id), None)
    if cell is not None:
        base.update({
            "kind": ("zworld_actor_critic" if cell["path"] == "zworld" else "rawview_actor_critic"),
            "representation": ("z_world_cotrain" if cell["path"] == "zworld"
                               else "raw_5x5_resource_field_view"),
            "teacher": ("sparse_foraging" if cell["teacher"] == "sparse"
                        else "foraging_plus_potential_shaping"),
            "shaping_coef": float(cell["shaping_coef"]), "rl_episodes": int(rl_eps),
            "actor_critic_hidden": fan.ACTOR_CRITIC_HIDDEN,
        })
        if cell["path"] == "zworld":
            base.update({"cotrain_encoder": True, "use_sf_critic": False,
                         "p0_warmup_episodes": int(p0)})
        else:
            base.update({"world_dim": fan.RAW_VIEW_DIM})
    else:
        base.update({"kind": "anchor"})
    return base


def _mean(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def _strict_majority(flags: List[bool]) -> bool:
    return bool(sum(1 for f in flags if f) >= (len(flags) + 1) // 2) if flags else False


def run_experiment(seeds: List[int], p0: int, rl_eps: int, eval_eps: int,
                   steps: int) -> Dict[str, Any]:
    print(
        f"MECH-457/INV-088 single-step strategy-diversity readout on the 2x2 factorial "
        f"({len(ARM_ORDER)} arms x 1 rung [{fan.RUNG_ID}] x {len(seeds)} seeds; "
        f"p0={p0}, RL={rl_eps}, eval={eval_eps}, steps={steps})",
        flush=True,
    )
    env_kwargs = x734._env_kwargs_for_rung(fan.RUNG)
    per_arm_forage: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
    per_arm_hgreedy: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
    per_arm_effective: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
    per_arm_sment: Dict[str, List[Optional[float]]] = {a: [] for a in ARM_ORDER}
    per_arm_supra: Dict[str, List[bool]] = {a: [] for a in ARM_ORDER}
    all_cells: List[Dict[str, Any]] = []

    def _run_cell(arm_id: str, seed: int) -> Dict[str, Any]:
        print(f"Seed {seed} Condition {fan.RUNG_ID}:{arm_id}", flush=True)
        slice_cfg = _arm_config_slice(arm_id, env_kwargs, p0, rl_eps, eval_eps, steps)
        with arm_cell(seed, config_slice=slice_cfg, script_path=Path(__file__),
                      config_slice_declared=True, include_driver_script_in_hash=False) as cell:
            if arm_id in TREATMENT_IDS:
                spec = next(c for c in TREATMENT_CELLS if c["arm_id"] == arm_id)
                row = _run_treatment_cell(spec, env_kwargs, seed, p0, rl_eps, eval_eps, steps)
            else:
                row = _run_anchor_cell(arm_id, env_kwargs, seed, eval_eps, steps)
            row["rung_id"] = fan.RUNG_ID
            row["arm_id"] = arm_id
            row["seed"] = int(seed)
            cell.stamp(row)
        div = row["diversity"]
        per_arm_forage[arm_id].append(float(row["foraging_competence"]))
        per_arm_hgreedy[arm_id].append(float(div["h_greedy_bits"]))
        per_arm_effective[arm_id].append(float(div["effective_actions"]))
        per_arm_sment[arm_id].append(div["mean_softmax_entropy_bits"])
        per_arm_supra[arm_id].append(bool(row["competence_supra_floor"]))
        all_cells.append(row)
        print(
            f"verdict: {'PASS' if row['competence_supra_floor'] else 'FAIL'} "
            f"(arm={arm_id} seed={seed} forage/ep={row['foraging_competence']} "
            f"H_greedy={div['h_greedy_bits']} eff_actions={div['effective_actions']})",
            flush=True,
        )
        return row

    # --- anchors FIRST (readiness assert before the expensive treatment training) -----------
    for arm_id in fan.ANCHOR_ARMS:
        for seed in seeds:
            _run_cell(arm_id, seed)

    local_view_forage = _mean(per_arm_forage["local_view_greedy"])
    oracle_forage = _mean(per_arm_forage["greedy_oracle"])
    readiness_forage_met = bool(
        local_view_forage >= COMPETENCE_RESOURCE_FLOOR and oracle_forage >= COMPETENCE_RESOURCE_FLOOR
    )

    hg_random = _mean(per_arm_hgreedy["random_walk"])
    hg_localview = _mean(per_arm_hgreedy["local_view_greedy"])
    # instrument calibration: constant-action control (H_greedy == 0) vs random_walk (~max).
    cal_hg_per_seed = [
        _calibration_diversity(env_kwargs, seed, eval_eps, steps)["h_greedy_bits"]
        for seed in seeds
    ]
    hg_constant = _mean(cal_hg_per_seed)
    instrument_diversity_spread = hg_random - hg_constant
    readiness_diversity_met = bool(instrument_diversity_spread > READINESS_DIVERSITY_SPREAD_FLOOR)
    readiness_met = bool(readiness_forage_met and readiness_diversity_met)

    # --- treatment cells (only if the instrument + env are ready) ---------------------------
    if readiness_met:
        for cell in TREATMENT_CELLS:
            for seed in seeds:
                _run_cell(cell["arm_id"], seed)
    else:
        print(
            f"readiness UNMET (forage local={round(local_view_forage,3)} oracle="
            f"{round(oracle_forage,3)}; instrument_diversity_spread="
            f"{round(instrument_diversity_spread,3)} floor={READINESS_DIVERSITY_SPREAD_FLOOR}); "
            f"skipping treatment training -> substrate_not_ready_requeue", flush=True,
        )

    def _cell_hg_mean(arm_id: str) -> float:
        return _mean(per_arm_hgreedy[arm_id])

    def _cell_supra_majority(arm_id: str) -> bool:
        return _strict_majority(per_arm_supra[arm_id])

    # --- the decisive matched-competence dense pair (748 vs 749) ----------------------------
    z_dense, r_dense = "ac_zworld_dense_rl", "ac_rawview_dense_rl"
    dense_pair_matched = bool(_cell_supra_majority(z_dense) and _cell_supra_majority(r_dense))

    # per-seed representation effect on diversity at matched competence (dense pair)
    zd_hg, rd_hg = per_arm_hgreedy[z_dense], per_arm_hgreedy[r_dense]
    n_pair = min(len(zd_hg), len(rd_hg))
    repr_effect_dense_per_seed = [round(zd_hg[i] - rd_hg[i], 6) for i in range(n_pair)]
    repr_effect_dense_mean = round(_mean(repr_effect_dense_per_seed), 6) if n_pair else 0.0
    repr_gates_diversity = _strict_majority(
        [abs(d) > DIVERSITY_MARGIN_BITS for d in repr_effect_dense_per_seed]
    ) if n_pair else False

    # --- 2x2 main effects on H_greedy (recorded for the autopsy decoupling) -----------------
    def _cells(rep: Optional[str], teacher: Optional[str]) -> List[str]:
        return [c["arm_id"] for c in TREATMENT_CELLS
                if (rep is None or c["rep"] == rep) and (teacher is None or c["teacher"] == teacher)]

    def _grp_hg(arm_ids: List[str]) -> float:
        vals = [v for a in arm_ids for v in per_arm_hgreedy[a]]
        return _mean(vals)

    representation_main_effect = round(
        _grp_hg(_cells("z_world", None)) - _grp_hg(_cells("raw_view", None)), 6)
    teacher_main_effect = round(
        _grp_hg(_cells(None, "dense")) - _grp_hg(_cells(None, "sparse")), 6)

    # --- self-route (hypotheses; /failure-autopsy adjudicates before governance use) --------
    if not readiness_met:
        outcome, label = "FAIL", "substrate_not_ready_requeue"
    elif not dense_pair_matched:
        outcome, label = "FAIL", "matched_competence_precondition_unmet"
    elif repr_gates_diversity:
        outcome, label = "PASS", "representation_gates_diversity_independent_of_competence"
    else:
        outcome, label = "FAIL", "diversity_not_gated_by_representation_at_matched_competence"

    # --- non-degeneracy net: H_greedy must spread across the compared cells + anchors --------
    degeneracy = check_degeneracy({
        "h_greedy_across_cells": {
            "values": [_cell_hg_mean(a) for a in TREATMENT_IDS]
                      + [hg_localview, hg_random]
        }
    })

    # --- diversity table (per cell: competence + all single-step diversity readouts) --------
    diversity_table: Dict[str, Any] = {}
    for c in TREATMENT_CELLS:
        a = c["arm_id"]
        sm_vals = [v for v in per_arm_sment[a] if v is not None]
        diversity_table[a] = {
            "representation": c["rep"], "teacher": c["teacher"],
            "reference_leg": c["reference_leg"],
            "foraging_competence_mean": round(_mean(per_arm_forage[a]), 6),
            "foraging_competence_per_seed": [round(v, 6) for v in per_arm_forage[a]],
            "competence_majority_supra_floor": _cell_supra_majority(a),
            "h_greedy_bits_mean": round(_cell_hg_mean(a), 6),
            "h_greedy_bits_per_seed": [round(v, 6) for v in per_arm_hgreedy[a]],
            "effective_actions_mean": round(_mean(per_arm_effective[a]), 6),
            "effective_actions_per_seed": [round(v, 6) for v in per_arm_effective[a]],
            "mean_softmax_entropy_bits_mean": (round(_mean(sm_vals), 6) if sm_vals else None),
            "mean_softmax_entropy_bits_per_seed": per_arm_sment[a],
        }
    for a in fan.ANCHOR_ARMS:
        diversity_table[a] = {
            "representation": "anchor", "teacher": "anchor",
            "foraging_competence_mean": round(_mean(per_arm_forage[a]), 6),
            "h_greedy_bits_mean": round(_cell_hg_mean(a), 6),
            "h_greedy_bits_per_seed": [round(v, 6) for v in per_arm_hgreedy[a]],
            "effective_actions_mean": round(_mean(per_arm_effective[a]), 6),
        }

    interpretation = {
        "label": label,
        "preconditions": [
            {"name": "local_view_greedy_clears_forage_floor_at_d3", "kind": "readiness",
             "description": "Foraging env is solvable from the 5x5 local view (competence anchor).",
             "control": "local_view_greedy foraging_competence @D3 vs the 1.0 floor",
             "measured": round(local_view_forage, 6), "threshold": float(COMPETENCE_RESOURCE_FLOOR),
             "met": bool(local_view_forage >= COMPETENCE_RESOURCE_FLOOR)},
            {"name": "greedy_oracle_clears_forage_floor_at_d3", "kind": "readiness",
             "description": "Env is floor-achievable with global info (achievability anchor).",
             "control": "greedy_oracle foraging_competence @D3 vs the 1.0 floor",
             "measured": round(oracle_forage, 6), "threshold": float(COMPETENCE_RESOURCE_FLOOR),
             "met": bool(oracle_forage >= COMPETENCE_RESOURCE_FLOOR)},
            {"name": "instrument_diversity_spread_registerable", "kind": "readiness",
             "description": ("The instrument registers a between-policy H_greedy SPREAD (the SAME "
                            "statistic the verdict routes on) on controls with KNOWN-different "
                            "single-step diversity: random_walk (maximally action-diverse, "
                            "H_greedy ~= log2(action_dim)) vs a constant-action calibration control "
                            "(monostrategy, H_greedy == 0). Cannot false-fail on legitimately-similar "
                            "arms."),
             "control": "H_greedy(random_walk) - H_greedy(constant_action_calibration), bits",
             "measured": round(instrument_diversity_spread, 6),
             "threshold": float(READINESS_DIVERSITY_SPREAD_FLOOR),
             "met": bool(readiness_diversity_met)},
            {"name": "dense_pair_matched_competent", "kind": "precondition",
             "description": ("The decisive representation contrast requires the dense pair (748 vs "
                            "749) to be matched-competent -- both clear the forage floor -- so any "
                            "diversity difference there is NOT explained by competence."),
             "control": "min(majority_supra_floor(z_world_dense), majority_supra_floor(raw_dense))",
             "measured": int(dense_pair_matched), "threshold": 1, "met": bool(dense_pair_matched)},
        ],
        "criteria": [
            {"name": "C_representation_gates_diversity_at_matched_competence", "load_bearing": True,
             "passed": bool(repr_gates_diversity),
             "description": ("|H_greedy(z_world_dense) - H_greedy(raw_dense)| > "
                            f"{DIVERSITY_MARGIN_BITS} bits on a strict majority of seeds, with the "
                            "dense pair matched-competent.")},
        ],
        "criteria_non_degenerate": {
            "instrument_diversity_spread_registerable": bool(readiness_diversity_met),
            "dense_pair_matched_competent": bool(dense_pair_matched),
            "h_greedy_spread_across_cells": bool(degeneracy["non_degenerate"]),
        },
    }

    result: Dict[str, Any] = {
        "outcome": outcome,
        "interpretation": interpretation,
        "interpretation_label": label,
        "discrimination_verdict": label,
        "evidence_direction": "unknown",
        "evidence_direction_per_claim": {"MECH-457": "unknown", "INV-088": "unknown"},
        "readiness": {
            "readiness_met": readiness_met,
            "local_view_greedy_forage_d3": round(local_view_forage, 6),
            "greedy_oracle_forage_d3": round(oracle_forage, 6),
            "h_greedy_random_walk_bits": round(hg_random, 6),
            "h_greedy_local_view_greedy_bits": round(hg_localview, 6),
            "h_greedy_constant_calibration_bits": round(hg_constant, 6),
            "instrument_diversity_spread_bits": round(instrument_diversity_spread, 6),
            "instrument_diversity_spread_floor": float(READINESS_DIVERSITY_SPREAD_FLOOR),
        },
        "headline": {
            "decisive_pair": "748 (z_world,dense) vs 749 (raw_view,dense) -- matched competence",
            "dense_pair_matched_competent": dense_pair_matched,
            "repr_effect_dense_h_greedy_bits_mean": repr_effect_dense_mean,
            "repr_effect_dense_h_greedy_bits_per_seed": repr_effect_dense_per_seed,
            "repr_gates_diversity": repr_gates_diversity,
            "diversity_margin_bits": float(DIVERSITY_MARGIN_BITS),
            "representation_main_effect_h_greedy_bits": representation_main_effect,
            "teacher_main_effect_h_greedy_bits": teacher_main_effect,
            "h_greedy_bits_by_cell": {a: round(_cell_hg_mean(a), 6) for a in TREATMENT_IDS},
            "effective_actions_by_cell": {a: round(_mean(per_arm_effective[a]), 6)
                                          for a in TREATMENT_IDS},
        },
        "diversity_table": diversity_table,
        "per_arm_forage_mean": {a: round(_mean(per_arm_forage[a]), 6) for a in ARM_ORDER},
        "denominators": {
            "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
            "max_h_greedy_bits": round(math.log2(ACTION_DIM), 6),
            "action_dim": int(ACTION_DIM),
        },
        "arm_results": all_cells,
        "non_degenerate": bool(degeneracy["non_degenerate"]),
        "degeneracy_reason": degeneracy["degeneracy_reason"],
        "degenerate_metrics": degeneracy["degenerate_metrics"],
    }
    return result


def _build_manifest(result: Dict[str, Any], timestamp_utc: str, dry_run: bool,
                    cfg: Dict[str, Any]) -> Dict[str, Any]:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    return {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "timestamp_utc": timestamp_utc,
        "dry_run": bool(dry_run),
        "outcome": result["outcome"],
        "evidence_direction": result["evidence_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "interpretation": result["interpretation"],
        "interpretation_label": result["interpretation_label"],
        "discrimination_verdict": result["discrimination_verdict"],
        "readiness": result["readiness"],
        "headline": result["headline"],
        "diversity_table": result["diversity_table"],
        "per_arm_forage_mean": result["per_arm_forage_mean"],
        "denominators": result["denominators"],
        "arm_results": result["arm_results"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "degenerate_metrics": result["degenerate_metrics"],
        "portfolio": {
            "gov_fanout_1": "MECH-457 competence-discrimination portfolio (742 autopsy)",
            "readout": "single-step strategy-diversity (H_greedy / effective_actions / softmax-entropy)",
            "factorial": "representation (z_world | raw_5x5) x teacher-density (sparse | dense)",
            "cells": {c["arm_id"]: c["reference_leg"] for c in TREATMENT_CELLS},
            "decisive_pair": "748 (z_world,dense) vs 749 (raw_view,dense) [matched competence]",
        },
        "config": cfg,
        "load_bearing_dv": (
            "Representation effect on single-step strategy diversity at MATCHED competence: "
            "|H_greedy(z_world,dense) - H_greedy(raw_view,dense)| (bits) > "
            f"{DIVERSITY_MARGIN_BITS}, strict majority of seeds, dense pair both supra the 1.0 "
            "forage floor. Readiness = a registerable between-policy H_greedy spread "
            "(random_walk vs local_view_greedy) + env foraging-solvable."
        ),
        "notes": (
            "DIAGNOSTIC (excluded from scoring); PROMOTES/DEMOTES NOTHING; route to "
            "/failure-autopsy before any governance action. Adds a single-step strategy-diversity "
            "readout to the GOV-FANOUT-1 2x2 (742/747/748/749); the four cells are rerun verbatim "
            "through mech457_fanout (deterministic; the completed manifests saved only aggregate "
            "competence and arm_fingerprint is Phase-0 emit-only, so no reuse was possible). "
            "SINGLE-STEP ONLY -- no multi-step committed-trajectory metric (substrate cannot "
            "sustain multi-step commitment). Tests whether z_world differentiation buys strategy "
            "diversity INDEPENDENT of the competence gain (INV-088 representation-ceiling root of "
            "monostrategy vs ARC-065 no-diversity-pressure root). MECH-457 stays candidate/"
            "v3_pending; INV-088 stays candidate/pending_substrate_reconfirmation. A PASS ESCALATES "
            "to registering an ARC-065 child (representation-ceiling channel) or an INV-088 "
            "behavioural-consequence claim, adjudicated by /failure-autopsy first."
        ),
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-750 MECH-457/INV-088 single-step strategy-diversity readout"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)

    if args.dry_run:
        seeds = list(fan.DRY_SEEDS)
        p0, rl_eps, eval_eps, steps = fan.DRY_P0, fan.DRY_RL, fan.DRY_EVAL, fan.DRY_STEPS
    else:
        seeds = list(fan.SEEDS)
        p0, rl_eps, eval_eps, steps = (
            fan.P0_WARMUP_EPISODES, fan.RL_EPISODES, fan.EVAL_EPISODES, fan.STEPS_PER_EPISODE
        )

    result = run_experiment(seeds=seeds, p0=p0, rl_eps=rl_eps, eval_eps=eval_eps, steps=steps)

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cfg = {
        "seeds": seeds, "rung": fan.RUNG_ID, "arms": list(ARM_ORDER),
        "treatment_cells": [
            {"arm_id": c["arm_id"], "representation": c["rep"], "teacher": c["teacher"],
             "shaping_coef": float(c["shaping_coef"]), "reference_leg": c["reference_leg"]}
            for c in TREATMENT_CELLS
        ],
        "p0_warmup_episodes": p0, "rl_episodes": rl_eps, "eval_episodes": eval_eps,
        "steps_per_episode": steps, "actor_critic_hidden": fan.ACTOR_CRITIC_HIDDEN,
        "ac_lr": fan.AC_LR, "ac_gamma": fan.AC_GAMMA,
        "diversity_margin_bits": float(DIVERSITY_MARGIN_BITS),
        "readiness_diversity_spread_floor": float(READINESS_DIVERSITY_SPREAD_FLOOR),
        "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
        "diversity_metrics": ["h_greedy_bits", "effective_actions", "distinct_actions_used",
                              "mean_softmax_entropy_bits"],
    }
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run), cfg=cfg)

    out_dir = Path(args.out_dir) if args.out_dir is not None else (
        REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    )
    out_path = write_flat_manifest(
        manifest, out_dir, dry_run=args.dry_run, config=cfg, seeds=seeds,
        script_path=Path(__file__),
        elapsed_seconds=(datetime.now(timezone.utc) - started).total_seconds(),
    )

    print(f"manifest: {out_path}", flush=True)
    hl = result["headline"]
    print(
        f"outcome: {result['outcome']} label={result['interpretation_label']} "
        f"readiness_met={result['readiness']['readiness_met']} "
        f"non_degenerate={result['non_degenerate']}", flush=True,
    )
    print(
        f"  decisive: dense_matched={hl['dense_pair_matched_competent']} "
        f"repr_effect_dense_H={hl['repr_effect_dense_h_greedy_bits_mean']} "
        f"(margin={hl['diversity_margin_bits']}) repr_gates={hl['repr_gates_diversity']}",
        flush=True,
    )
    print(
        f"  main-effects H_greedy: representation={hl['representation_main_effect_h_greedy_bits']} "
        f"teacher={hl['teacher_main_effect_h_greedy_bits']}", flush=True,
    )

    if args.dry_run:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass

    outcome_norm = str(result["outcome"]).upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    manifest_for_sentinel = str(out_path) if not args.dry_run else None
    return outcome_emit, manifest_for_sentinel, bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _manifest_path, _dry_run = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path, dry_run=_dry_run)
