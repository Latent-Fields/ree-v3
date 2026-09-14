"""V3-EXQ-1027 -- SD-082 fan-out leg 1 (H-replay-rule-state-mismatch, axis
credit-assignment): does scoring each replayed REINFORCE sample at ITS OWN
action-time rule_state raise step-direction persistence of the rule->bias
readout above the faithful-to-822f/1020 path, at matched budget?

RED-TEAM: see RED_TEAM_VERDICT below (queue-experiment Step 4.5).

================================================================================
WHERE THIS COMES FROM
================================================================================
GOV-FANOUT-1 four-leg portfolio routed by the CONFIRMED
failure_autopsy_V3-EXQ-1020_2026-09-11 (REE_assembly 92751187b1), ratified by
governance cycle gov-20260911-1612, all four legs user-selected at that
autopsy's Step 8 gate; pre-registered in hypothesis_space_registry.v1.json
under qid sd082_candidate_discriminating_readout_locus (NOT re-registered here).
Chip: chip-20260911-sd082-fanout-portfolio-v2.

Sibling legs queued with this one:
  V3-EXQ-1028  H-learning-signal-sign (measurement) + H-learning-signal-noisy
               (process) -- one extended-P1 run, because both legs' manipulation
               IS the same P1 budget extension (two runs would duplicate compute).
  V3-EXQ-1029  H-selection-authority-bounded (intrinsic-architecture).

THE HYPOTHESIS (registry, verbatim label): "The REINFORCE replay scores every
sampled tuple at ONE shared end-of-episode lpfc.rule_state, so the rule half of
the head input is mismatched to the credit on most samples, randomising the
rule-attributable gradient BY CONSTRUCTION." Prediction: "Restoring each
sample's own rule_state at update time raises step-direction persistence
materially above the faithful-to-822f path, at matched budget."

WHY IT IS THE HIGHEST-VALUE LEG. V3-EXQ-1020's own F2 disposition declared it an
unexcluded rival to H-learning-signal-noisy ("Do NOT read a C2 PASS as
establishing H-learning-signal-noisy over this rival"), and the autopsy's
biological triage independently favours it: brains bind credit to the state
active AT ACTION TIME (eligibility traces), never to a single current state.
Per-sample rule_state at update time is the eligibility-trace analogue.

================================================================================
DESIGN -- two replay arms, everything else byte-identical to 1020's ARM_ON
================================================================================
  REPLAY_FAITHFUL        replay tuple index 4 = None -> every drawn sample is
                         scored at the live end-of-episode rule_state (1020/822f).
  REPLAY_OWN_RULE_STATE  replay tuple index 4 = the rule_state snapshot taken
                         IMMEDIATELY AFTER that tick's select_action, i.e. the
                         rule_state lateral_pfc.compute_bias actually used to bias
                         THAT selection (agent.py: lateral_pfc.update() precedes
                         compute_bias inside select_action). The live rule_state
                         is saved and restored around every update, so the arm
                         never perturbs the agent's own state.
Seeds 822f/1020's [611, 622, 633, 644, 655]; crf_cue_centering=True (1020 ARM_ON);
P0 60 / P1 70 / 48 steps; lr, batch, buffer, temperature, advantage filter and
EMA baseline all IMPORTED from the 1020 module, never re-typed. The two in-run
controls (dense-synthetic-credit positive control, pure-noise Adam negative
control) are 1020's, imported, run per cell at the same T=70.

THE COMPARISON IS PAIRED BY CONSTRUCTION, and that is measured, not assumed:
with use_modulatory_selection_authority False (1020's configuration) the
+/-0.1 tanh-bounded bias has never been observed to change the committed argmin
(1020: per_episode_returns bit-identical across arms), so both arms should see
the IDENTICAL episode stream and the IDENTICAL replay buffer, differing ONLY in
the rule_state each sample is scored at. Every cell records its per-episode
returns and the manifest records trajectory_identical_per_seed; if a seed's two
arms diverge, the pairing for that seed is weaker than stated and is flagged.

================================================================================
DECISIVE READOUT, PRE-REGISTERED CRITERIA
================================================================================
DV: persistence_pooled = ||sum of Adam steps|| / sum of ||steps|| on
rule_bias_head over the 70 P1 updates (1020's _StepAccumulator). Per cell the
bracket is [noise floor, positive-control ceiling], both measured in-run.

  R1 readiness (1020's, unchanged): the dense-synthetic-credit control learns
     its task and persists in >= 0.8 of cells (both arms).
  R2 manipulation reaches the DV (non-vacuity, OWN arm): the buffer-mean DIRECTION
     mismatch mean(1 - cos(rs_sample, rs_live)) at update time, averaged over
     updates, must be >= MISMATCH_FLOOR (0.10) on the worst seed. A norm-inclusive
     mismatch is recorded too but not gated: rule_state is zeroed each episode and
     EMA-ramps, so norms differ with an unchanged direction, which never presents
     "credit bound to the wrong rule". Probe (seed 633): direction mismatch 0.674.
  R3 headroom + pairing, PER SEED (not whole-run): a seed is ELIGIBLE iff FAITHFUL
     persistence <= ceiling - lift_margin (the DV has room to rise by the amount C1
     requires) AND both arms replayed a byte-identical P1 buffer (SHA-256 over
     summaries, action-time rule_state, selected index, fresh flag, return). A seed
     failing either is scoped out, never scored as null.
     Need >= MIN_ELIGIBLE_SEEDS (3) eligible seeds, else not-ready.
  C1 (load-bearing): lift = pers(OWN) - pers(FAITHFUL) >= lift_margin AND the
     first layer's SUMMARY-input-column persistence also rises, on
     >= SEED_MAJORITY (3) ELIGIBLE seeds, where
     lift_margin = LIFT_MARGIN_FRACTION (0.25) x (ceiling - floor) for that seed.
     1020's measured brackets (floor 0.449, ceilings 0.72-0.76) put the margin at
     ~0.07, below every ARM_ON seed's recorded headroom (0.08-0.32).
  C2 (recorded, not load-bearing): lift <= -lift_margin on >= 3 eligible seeds.
  combination: interpretation.criteria_aggregation = "any" (the 1020 autopsy's
  forward-only declaration fix); PASS iff R1 AND R2 AND R3 AND C1.

VERDICT GRID
  R1 or R2 or R3 unmet          -> substrate_not_ready_requeue (FAIL). Never a
                                   verdict on an unready or vacuous manipulation.
  C1                            -> H_replay_rule_state_mismatch_supported (PASS).
                                   The shared-state replay was randomising the
                                   rule-attributable gradient; 822f/1020's
                                   low-persistence readings are (at least partly)
                                   a credit-assignment artefact of the driver.
  not C1, C2 (lift negative)    -> H_replay_rule_state_mismatch_refuted_own_state_lowers
                                   (FAIL, informative): own-state scoring makes the
                                   gradient LESS consistent.
  not C1, not C2                -> H_replay_rule_state_mismatch_not_supported (FAIL,
                                   informative null): restoring per-sample
                                   conditioning does not materially move
                                   persistence at this budget; the mismatch rival
                                   does not explain 1020's low persistence (it
                                   weakens, not removes, the rival to 1028's
                                   H-learning-signal-noisy reading).
WHAT A NULL DOES NOT MEAN: it is not evidence against SD-082's readout or
SD-078's rule pool, and not a substrate ceiling. It removes one explanation of
a driver-level optimisation reading. Directions pinned "unknown" (diagnostic).

DV-SYMMETRY DECLARATION (per arm; same DV in both):
  persistence is invariant under a GLOBAL rotation of every step and under a
  uniform rescaling of all steps. The manipulation changes WHICH rule_state each
  sample's gradient is taken at, i.e. the per-sample gradient DIRECTION -- not a
  global rotation, not a rescaling. Not invariant, in either arm. It is not an
  argmax/rank DV, so broadcast-scalar and monotone annihilations do not apply.

A TRIVIAL-LIFT PATH, recorded rather than ignored: if the live end-of-episode
rule_state were often exactly zero, FAITHFUL would zero the rule half of the
input on every sample and OWN would "lift" merely by giving those weights a
gradient at all. Every cell therefore records the fraction of updates whose live
rule_state is zero (live_rule_state_zero_frac), and the per-tensor persistence
of the first Linear layer, so a reader can see whether a lift is carried by the
rule columns becoming live rather than by consistency. 1020 recorded
rule_state_norm_at_update means of 0.21-0.37 on these seeds (nonzero).
A second non-credit route, also recorded: with ONE shared rule_state every
sample's rule-column gradient is collinear within an update, so persistence can
move through input-distribution structure alone. Each cell therefore splits the
first layer's persistence into RULE-input columns and SUMMARY-input columns; a
lift carried only by the rule columns, with the summary columns flat, reads as
input-structure rather than credit correctness and must be adjudicated as such.
Authoring probe (seed 633, P0 60 / P1 30, darwin-arm64): trajectories bit-identical
across arms, buffer mismatch 1.16 (floor 0.10), persistence FAITHFUL 0.7595 vs
OWN 0.7486 -- the manipulation reaches the DV, in an as-yet undetermined direction.

================================================================================
QUEUE-EXPERIMENT GATE DISPOSITIONS (2026-09-14)
================================================================================
Step 2.4 GOV-REUSE-1: decisive readout = rule_bias_head persistence under
per-sample rule_state replay. No run has ever replayed with index 4 populated
on the real path (1020/822f: None always) -> not recoverable, run.
Step 2.5 / 2.5a: substrate unchanged from 1020 (lateral_pfc train_rule_bias_head,
rule_readout_consumer, CRF); the only new mechanic is the driver's replay tuple.
_reinforce_step already honours index 4 (the 1020 synthetic control uses it).
Step 2.5b brake: SD-082 counts 3 (822b/822c/822d), released by
failure_autopsy_V3-EXQ-822f_2026-09-09 on its own record; the 1020 autopsy
refuses further LETTERED iterations of the 822/1020 design. This is a new EXQ
number on a new axis (credit-assignment), a diagnostic discriminating WHY
persistence is low -- the skill's "Not braked" clause.
Step 2.5c: identical module footprint to V3-EXQ-1020, whose call-trace
disposition is carried forward verbatim (ContextMemory.write not executed;
tonic_vigor / blocked_agency / selection_entropy_floor not executed; open
corrupting entries not in path). SD-082's own substrate_paths are empty by the
2026-09-09 governance decision.
Step 2.6 ethics: all-false / allow.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest, flat_readout  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.readiness_anchor import assert_anchor_reachable  # noqa: E402
import experiments.v3_exq_1020_sd082_learning_signal_probe as x1020  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_1027_sd082_replay_own_rule_state_credit_assignment"
QUEUE_ID = "V3-EXQ-1027"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS = ["SD-082"]
FANOUT_QID = "sd082_candidate_discriminating_readout_locus"
FANOUT_HYPOTHESES = ["H-replay-rule-state-mismatch"]
FANOUT_AXIS = "credit-assignment"
FANOUT_SOURCE_AUTOPSY = "failure_autopsy_V3-EXQ-1020_2026-09-11"
SOURCE_CHIP_REF = "chip-20260911-sd082-fanout-portfolio-v2"
RED_TEAM_VERDICT = (
    "red-team (fable): CONTESTED -> 5 findings; 4 FIXED, 1 noted. F1 R2 mismatch was "
    "norm-dominated (within-episode EMA ramp) and could pass with cos=1 -> R2 now gates "
    "DIRECTION mismatch mean(1-cos) >= 0.10 (probe 0.674). F2 C2 fired on any negative "
    "sign (coin-flip under a null) -> same margin as C1, eligible seeds only. F3 a "
    "rule-column-only lift mapped to supported -> C1 also requires a positive "
    "summary-column persistence lift; pooled-only count recorded. F4 pairing witness "
    "compared returns only and could not fail -> per-cell SHA-256 of the P1 replay "
    "buffer (summaries, action-time rule_state, sel, fresh, return); a seed whose arms "
    "differ is scoped out (probe: identical). F5 OWN-arm rule_state_norm_at_update "
    "records the last replayed sample's norm, not the live one -- noted, not gated.")

SEEDS = list(x1020.SEEDS)
CENTERING = True
ARMS = ["REPLAY_FAITHFUL", "REPLAY_OWN_RULE_STATE"]
P0_WARMUP_EPISODES = x1020.P0_WARMUP_EPISODES
P1_BIAS_TRAIN_EPISODES = x1020.P1_BIAS_TRAIN_EPISODES
STEPS_PER_EPISODE = x1020.STEPS_PER_EPISODE

# Pre-registered (constants; not derived from this run).
MISMATCH_FLOOR = 0.10
LIFT_MARGIN_FRACTION = 0.25
SEED_MAJORITY = 3
MIN_ELIGIBLE_SEEDS = 3
RULE_STATE_ZERO_EPS = 1e-9

# R3 (persistence_headroom_eligible_seeds) IS this driver's dv_headroom gate: it
# measures, per seed, whether the FAITHFUL persistence sits at least one lift margin
# below that seed's own in-run ceiling -- the realised room the DV has to move by the
# amount C1 requires -- and scopes out seeds without it.
CRITERION_ACHIEVABLE_RANGE_EXEMPT = (
    "R3 persistence_headroom_eligible_seeds is the per-seed dv_headroom check against "
    "the same lift margin C1 uses; seeds without headroom are scoped out, and < 3 "
    "eligible seeds self-routes substrate_not_ready_requeue")

# Reference cells for the anchor guards, scored with the SHIPPED predicates.
# Headroom: V3-EXQ-1020's recorded ARM_ON cells (the FAITHFUL path, same seeds).
HEADROOM_REFERENCE_CELLS = [
    {"persistence_pooled": 0.652, "persistence_ceiling": 0.733, "persistence_floor": 0.449},
    {"persistence_pooled": 0.406, "persistence_ceiling": 0.729, "persistence_floor": 0.449},
    {"persistence_pooled": 0.558, "persistence_ceiling": 0.757, "persistence_floor": 0.449},
    {"persistence_pooled": 0.481, "persistence_ceiling": 0.723, "persistence_floor": 0.449},
    {"persistence_pooled": 0.620, "persistence_ceiling": 0.736, "persistence_floor": 0.449},
]
HEADROOM_REFERENCE_SOURCE = ("v3_exq_1020_sd082_learning_signal_probe_20260911T003146Z_v3 "
                             "ARM_ON cells, seeds 611/622/633/644/655")
# Direction mismatch: authoring probe 2026-09-14 (darwin-arm64), seed 633, P0 60 / P1 30.
DIRECTION_PROBE_VALUE = 0.6739  # seed 633, P0 60 / P1 25, both arms (identical buffers)
MISMATCH_REFERENCE_CELLS = [{"buffer_direction_mismatch_mean": DIRECTION_PROBE_VALUE}]
MISMATCH_REFERENCE_SOURCE = "V3-EXQ-1027 authoring probe, seed 633, p0=60 p1=30"


def _seed_eligible(c: Dict[str, Any]) -> bool:
    """THE SHIPPED headroom predicate (used live and by the anchor guard)."""
    lo = float(c["persistence_floor"])
    hi = float(c["persistence_ceiling"])
    margin = LIFT_MARGIN_FRACTION * max(hi - lo, 0.0)
    return bool(hi > lo and margin > 0.0 and (hi - float(c["persistence_pooled"])) >= margin)


def _mismatch_ok(c: Dict[str, Any]) -> bool:
    """THE SHIPPED non-vacuity predicate (direction, not norm)."""
    v = c.get("buffer_direction_mismatch_mean")
    return bool(v is not None and v >= MISMATCH_FLOOR)


def _assert_anchors() -> List[Dict[str, Any]]:
    return [
        assert_anchor_reachable(
            anchor_name="persistence_headroom_eligible_seeds",
            reference_cells=HEADROOM_REFERENCE_CELLS, score_fn=_seed_eligible,
            threshold=MIN_ELIGIBLE_SEEDS / len(HEADROOM_REFERENCE_CELLS),
            reference_source=HEADROOM_REFERENCE_SOURCE),
        assert_anchor_reachable(
            anchor_name="own_rule_state_manipulation_non_vacuous",
            reference_cells=MISMATCH_REFERENCE_CELLS, score_fn=_mismatch_ok,
            threshold=1.0, reference_source=MISMATCH_REFERENCE_SOURCE),
    ]


def _config_slice(arm: str) -> Dict[str, Any]:
    s = x1020._config_slice(CENTERING)
    s["replay_mode"] = arm
    s["schedule"] = {"p0": P0_WARMUP_EPISODES, "p1": P1_BIAS_TRAIN_EPISODES,
                     "steps": STEPS_PER_EPISODE}
    return s


def _update_with_restore(lpfc, opt, buf, baseline, device, acc, tel) -> bool:
    """1020's _reinforce_step, with the live rule_state saved and restored
    around it. For REPLAY_FAITHFUL (index 4 None) the restore is a no-op, so that
    arm is the 1020 path exactly. For REPLAY_OWN_RULE_STATE it stops the last
    replayed sample's rule_state leaking into the live agent."""
    saved = lpfc.rule_state.detach().clone()
    try:
        return x1020._reinforce_step(lpfc, opt, buf, baseline, device, acc, tel)
    finally:
        with torch.no_grad():
            lpfc.rule_state.copy_(saved)


def _buffer_mismatch(lpfc, buf) -> Tuple[Optional[float], Optional[float]]:
    """At the moment of an update, over buffer entries: (a) NORM-inclusive mismatch
    mean ||rs_sample - rs_live|| / ||rs_live||, and (b) DIRECTION mismatch
    mean(1 - cos(rs_sample, rs_live)) over entries with a nonzero rs_sample.
    rs_sample is the recorded action-time rule_state (index 6, stored in BOTH arms);
    rs_live is what FAITHFUL scores every sample at. (a) is dominated by the
    within-episode EMA norm ramp (rule_state is zeroed each episode), so R2 gates on
    (b): only a DIRECTIONAL difference presents the credit-to-the-wrong-rule
    hypothesis at all (red-team F1)."""
    live = lpfc.rule_state.detach().reshape(-1)
    ln = float(live.norm().item())
    if ln <= RULE_STATE_ZERO_EPS or not buf:
        return None, None
    tot = 0.0
    n = 0
    dtot = 0.0
    dn = 0
    for e in buf:
        rs = e[6].reshape(-1)
        tot += float((rs - live).norm().item()) / ln
        n += 1
        rn = float(rs.norm().item())
        if rn > RULE_STATE_ZERO_EPS:
            dtot += 1.0 - float(torch.dot(rs, live).item()) / (rn * ln)
            dn += 1
    return (tot / n if n else None), (dtot / dn if dn else None)


class _SplitAccumulator(x1020._StepAccumulator):
    """1020's accumulator plus the first Linear layer's weight split into its
    RULE-input columns and its SUMMARY-input columns (joined = cat([rule_state,
    summaries]) in lateral_pfc.compute_bias, so rule columns come first). A shared
    live rule_state makes every sample's rule-column gradient collinear with ONE
    vector within an update, which can move persistence for a reason unrelated to
    credit correctness; the split lets a reader see where a lift or drop lives."""

    def __init__(self, names: List[str], rule_dim: int):
        super().__init__(names)
        self.rule_dim = int(rule_dim)
        self._split_vec: Dict[str, Optional[torch.Tensor]] = {"rule_cols": None, "summary_cols": None}
        self._split_norm: Dict[str, float] = {"rule_cols": 0.0, "summary_cols": 0.0}

    def observe(self, deltas: Dict[str, torch.Tensor]) -> None:
        super().observe(deltas)
        w = deltas.get("0.weight")
        if w is None:
            return
        w = w.detach()
        for key, part in (("rule_cols", w[:, :self.rule_dim]),
                          ("summary_cols", w[:, self.rule_dim:])):
            v = part.reshape(-1).clone()
            self._split_vec[key] = v if self._split_vec[key] is None else self._split_vec[key] + v
            self._split_norm[key] += float(v.norm().item())

    def split_result(self) -> Dict[str, float]:
        return {k: self._ratio(self._split_vec[k], self._split_norm[k]) for k in self._split_vec}


def _first_layer_persistence(acc_res: Dict[str, Any]) -> Optional[float]:
    per = acc_res.get("persistence_per_tensor") or {}
    return per.get("0.weight")


def _run_cell(arm: str, seed: int, episodes: Dict[str, int],
              zg: ZGoalStreamAccumulator) -> Dict[str, Any]:
    own = (arm == "REPLAY_OWN_RULE_STATE")
    p0, p1 = episodes["p0"], episodes["p1"]
    total_eps = p0 + p1
    with arm_cell(seed, config_slice=_config_slice(arm), script_path=Path(__file__),
                  extra_substrate_paths=[Path(x1020.__file__)]) as cell:
        env = x1020._build_env(seed)
        agent = x1020._make_agent(env, CENTERING)
        wd = agent.config.latent.world_dim
        lpfc = agent.lateral_pfc
        device = agent.device

        init_head = copy.deepcopy(lpfc.rule_bias_head)
        for p in init_head.parameters():
            p.requires_grad_(False)

        bias_opt = torch.optim.Adam(list(lpfc.bias_head_parameters()), lr=x1020.LR_LPFC_BIAS)
        names = [n for n, _ in lpfc.rule_bias_head.named_parameters()]
        acc = _SplitAccumulator(names, int(lpfc.rule_state.shape[-1]))
        tel: Dict[str, Any] = {"grad_norms": [], "sample_flip_adv": [],
                               "n_updates": 0, "n_adv_filtered": 0,
                               "n_grad_nonfinite": 0, "sample_flip_ret": [],
                               "rule_state_norm_at_update": []}
        summary_counters = {"dispatch": 0, "fallback": 0}

        outcome_buf: List[Tuple] = []
        baseline = 0.0
        ep_returns: List[float] = []
        ep_flip_rate: List[float] = []
        n_flip_ticks = 0
        n_measured_ticks = 0
        n_rule_live_ticks = 0
        n_fresh_ticks = 0
        n_fresh_flip_ticks = 0
        n_latched_ticks = 0
        mismatch_per_update: List[float] = []
        direction_mismatch_per_update: List[float] = []
        buf_hash = hashlib.sha256()
        n_live_zero_updates = 0
        n_updates_attempted = 0
        stored_rs_norms: List[float] = []

        print(f"Seed {seed} Condition {arm}", flush=True)

        for ep in range(total_eps):
            is_p1 = (ep >= p0)
            _, obs = env.reset()
            agent.reset()
            ep_reward = 0.0
            ep_buf: List[Tuple] = []
            ep_ticks = 0
            ep_flips = 0
            last_flip = False
            last_flip_valid = False

            for _step in range(episodes["steps"]):
                latent = agent.sense(obs["body_state"], obs["world_state"])
                ticks = agent.clock.advance()
                is_fresh = bool(ticks.get("e3_tick", False))
                e1 = (agent._e1_tick(latent) if ticks.get("e1_tick")
                      else torch.zeros(1, wd, device=device))
                candidates = agent.generate_trajectories(latent, e1, ticks)

                snap: Optional[torch.Tensor] = None
                was_flip = False
                if is_p1 and candidates and len(candidates) >= 2:
                    cs = x1020._candidate_summaries(agent, candidates, summary_counters)
                    if cs is not None and torch.isfinite(cs).all():
                        snap = cs.clone()
                        rule_live = float(lpfc.rule_state.norm()) > x1020.RULE_STATE_LIVE_FLOOR
                        if rule_live:
                            n_rule_live_ticks += 1
                        if rule_live and is_fresh:
                            rf = x1020._raw_ratio_and_flip(lpfc, snap)
                            if rf is not None:
                                last_flip = bool(rf[1] > 0.5)
                                last_flip_valid = True
                            else:
                                last_flip_valid = False
                            if last_flip_valid:
                                n_fresh_ticks += 1
                                if last_flip:
                                    n_fresh_flip_ticks += 1
                        if rule_live and last_flip_valid:
                            was_flip = last_flip
                            ep_ticks += 1
                            n_measured_ticks += 1
                            if not is_fresh:
                                n_latched_ticks += 1
                            if was_flip:
                                ep_flips += 1
                                n_flip_ticks += 1

                action = agent.select_action(candidates, ticks)
                if action is None:
                    action = torch.zeros(1, 4, device=device)
                    action[0, int(np.random.randint(0, 4))] = 1.0
                    agent._last_action = action
                committed_class = int(action[0].argmax().item())

                if snap is not None:
                    sel = 0
                    for ci, c in enumerate(candidates):
                        if (getattr(c, "actions", None) is not None
                                and c.actions.shape[1] >= 1
                                and int(c.actions[:, 0, :].argmax(-1).reshape(-1)[0].item())
                                == committed_class):
                            sel = min(ci, snap.shape[0] - 1)
                            break
                    # Action-time rule_state: lateral_pfc.update() runs inside
                    # select_action BEFORE compute_bias, so the post-call value is
                    # the one that biased THIS selection.
                    rs_now = lpfc.rule_state.detach().clone()
                    ep_buf.append((snap, sel, was_flip, is_fresh, rs_now))

                _, _h, done, _info, obs = env.step(int(action.argmax(dim=-1).item()))
                if is_p1:
                    ep_reward += float(_h)
                if done:
                    break

            if is_p1:
                baseline = x1020.EMA_DECAY * baseline + (1.0 - x1020.EMA_DECAY) * ep_reward
                for cand_features, sel, flip_lbl, fresh_lbl, rs_now in ep_buf:
                    outcome_buf.append((cand_features, sel, ep_reward, flip_lbl,
                                        (rs_now if own else None), fresh_lbl, rs_now))
                    stored_rs_norms.append(float(rs_now.norm().item()))
                    # Pairing witness (red-team F4): everything the replay depends on
                    # EXCEPT the head-dependent flip label.
                    buf_hash.update(cand_features.detach().cpu().numpy().tobytes())
                    buf_hash.update(rs_now.detach().cpu().numpy().tobytes())
                    buf_hash.update(repr((int(sel), bool(fresh_lbl), float(ep_reward))).encode("ascii"))
                if len(outcome_buf) > x1020.OUTCOME_BUF_MAX:
                    outcome_buf = outcome_buf[-x1020.OUTCOME_BUF_MAX:]
                n_updates_attempted += 1
                if float(lpfc.rule_state.norm().item()) <= RULE_STATE_ZERO_EPS:
                    n_live_zero_updates += 1
                mm, dmm = _buffer_mismatch(lpfc, outcome_buf)
                if mm is not None and math.isfinite(mm):
                    mismatch_per_update.append(mm)
                if dmm is not None and math.isfinite(dmm):
                    direction_mismatch_per_update.append(dmm)
                _update_with_restore(lpfc, bias_opt, outcome_buf, baseline, device, acc, tel)
                ep_returns.append(float(ep_reward))
                ep_flip_rate.append(float(ep_flips) / ep_ticks if ep_ticks > 0 else 0.0)

            if (ep + 1) % 25 == 0 or (ep + 1) == total_eps:
                print(f"  [train] {arm} seed={seed} ep {ep+1}/{total_eps} "
                      f"updates={tel['n_updates']}", flush=True)

        control = x1020._synthetic_credit_control(agent, init_head, wd, device)
        noise_ctrl = x1020._noise_persistence_control(init_head, device)
        zg.observe(agent)

        acc_res = acc.result()
        row: Dict[str, Any] = {"arm_id": arm, "seed": seed, "centering": CENTERING,
                               "replay_mode": arm}
        row.update(x1020._summarise_cell(tel, acc_res, ep_returns, ep_flip_rate,
                                         n_flip_ticks, n_measured_ticks))
        row.update(control)
        row.update(noise_ctrl)
        lo = float(noise_ctrl["noise_persistence_pooled"])
        hi = float(control["synth_persistence_pooled"])
        row["persistence_floor"] = lo
        row["persistence_ceiling"] = hi
        row["persistence_bracket_valid"] = bool(hi > lo)
        row["lift_margin"] = LIFT_MARGIN_FRACTION * max(hi - lo, 0.0)
        row["first_layer_weight_persistence"] = _first_layer_persistence(acc_res)
        _split = acc.split_result()
        row["first_layer_rule_column_persistence"] = _split["rule_cols"]
        row["first_layer_summary_column_persistence"] = _split["summary_cols"]
        row["buffer_mismatch_mean"] = (float(np.mean(mismatch_per_update))
                                       if mismatch_per_update else None)
        row["buffer_mismatch_per_update"] = [float(v) for v in mismatch_per_update]
        row["buffer_direction_mismatch_mean"] = (float(np.mean(direction_mismatch_per_update))
                                                 if direction_mismatch_per_update else None)
        row["buffer_direction_mismatch_per_update"] = [float(v) for v in
                                                       direction_mismatch_per_update]
        row["p1_buffer_sha256"] = buf_hash.hexdigest()
        row["live_rule_state_zero_frac"] = (float(n_live_zero_updates) / n_updates_attempted
                                            if n_updates_attempted else None)
        row["stored_rule_state_norm_mean"] = (float(np.mean(stored_rs_norms))
                                              if stored_rs_norms else None)
        row["stored_rule_state_zero_frac"] = (
            float(sum(1 for v in stored_rs_norms if v <= RULE_STATE_ZERO_EPS)) / len(stored_rs_norms)
            if stored_rs_norms else None)
        row["n_rule_live_ticks"] = n_rule_live_ticks
        row["n_fresh_select_ticks"] = n_fresh_ticks
        row["n_latched_ticks"] = n_latched_ticks
        row["n_fresh_flip_ticks"] = n_fresh_flip_ticks
        row["summary_dispatch_calls"] = summary_counters["dispatch"]
        row["summary_fallback_calls"] = summary_counters["fallback"]
        row["control_ready"] = bool(
            row["synth_task_gain"] > x1020.SYNTH_TASK_GAIN_FLOOR
            and row["synth_persistence_pooled"] >= x1020.SYNTH_PERSISTENCE_FLOOR)
        cell.stamp(row)
    print(f"verdict: {'PASS' if row['control_ready'] else 'FAIL'}", flush=True)
    return row


def _cell(rows, arm, seed):
    return next((r for r in rows if r["arm_id"] == arm and r["seed"] == seed), None)


def run_experiment(episodes: Dict[str, int], dry_run: bool) -> Dict[str, Any]:
    t0 = time.perf_counter()
    anchor_reachability = x1020._assert_readiness_anchors_reachable() + _assert_anchors()
    zg = ZGoalStreamAccumulator()
    rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        for arm in ARMS:
            rows.append(_run_cell(arm, seed, episodes, zg))

    n_cells = max(len(rows), 1)
    gain_frac = sum(1 for r in rows if r["synth_task_gain"] > x1020.SYNTH_TASK_GAIN_FLOOR) / n_cells
    pers_frac = sum(1 for r in rows
                    if r["synth_persistence_pooled"] >= x1020.SYNTH_PERSISTENCE_FLOOR) / n_cells
    r1 = bool(gain_frac >= x1020.CONTROL_READY_FRACTION_FLOOR
              and pers_frac >= x1020.CONTROL_READY_FRACTION_FLOOR)

    own_mm = [(r["seed"], r["buffer_direction_mismatch_mean"]) for r in rows
              if r["arm_id"] == "REPLAY_OWN_RULE_STATE"]
    own_mm_vals = [v for _s, v in own_mm if v is not None]
    worst_mm = min(own_mm_vals) if len(own_mm_vals) == len(own_mm) and own_mm_vals else None
    worst_mm_seed = (min(own_mm, key=lambda t: t[1])[0]
                     if worst_mm is not None else None)
    r2 = bool(worst_mm is not None
              and all(_mismatch_ok(r) for r in rows if r["arm_id"] == "REPLAY_OWN_RULE_STATE"))

    per_seed: List[Dict[str, Any]] = []
    for seed in SEEDS:
        f = _cell(rows, "REPLAY_FAITHFUL", seed)
        o = _cell(rows, "REPLAY_OWN_RULE_STATE", seed)
        lift = float(o["persistence_pooled"]) - float(f["persistence_pooled"])
        margin = float(f["lift_margin"])
        headroom = float(f["persistence_ceiling"]) - float(f["persistence_pooled"])
        traj_identical = (f["per_episode_returns"] == o["per_episode_returns"])
        buffer_identical = (f["p1_buffer_sha256"] == o["p1_buffer_sha256"])
        eligible = bool(_seed_eligible(f) and buffer_identical)
        summary_col_lift = (
            float(o["first_layer_summary_column_persistence"])
            - float(f["first_layer_summary_column_persistence"])
            if (o["first_layer_summary_column_persistence"] is not None
                and f["first_layer_summary_column_persistence"] is not None) else None)
        per_seed.append({
            "seed": seed, "persistence_faithful": f["persistence_pooled"],
            "persistence_own": o["persistence_pooled"], "lift": lift,
            "lift_margin": margin, "headroom": headroom, "eligible": eligible,
            "buffer_identical": bool(buffer_identical),
            "summary_column_lift": summary_col_lift,
            "lift_clears_margin": bool(eligible and lift >= margin
                                       and summary_col_lift is not None and summary_col_lift > 0.0),
            "lift_clears_margin_pooled_only": bool(eligible and lift >= margin),
            "lift_negative": bool(eligible and lift <= -margin),
            "first_layer_persistence_faithful": f["first_layer_weight_persistence"],
            "first_layer_persistence_own": o["first_layer_weight_persistence"],
            "rule_column_persistence_faithful": f["first_layer_rule_column_persistence"],
            "rule_column_persistence_own": o["first_layer_rule_column_persistence"],
            "summary_column_persistence_faithful": f["first_layer_summary_column_persistence"],
            "summary_column_persistence_own": o["first_layer_summary_column_persistence"],
            "trajectory_identical": bool(traj_identical),
            "live_rule_state_zero_frac": f["live_rule_state_zero_frac"],
            "buffer_mismatch_own": o["buffer_mismatch_mean"],
        })
    n_eligible = sum(1 for p in per_seed if p["eligible"])
    r3 = bool(n_eligible >= MIN_ELIGIBLE_SEEDS)
    n_lift = sum(1 for p in per_seed if p["lift_clears_margin"])
    n_neg = sum(1 for p in per_seed if p["lift_negative"])
    c1 = bool(n_lift >= SEED_MAJORITY)
    c2 = bool(n_neg >= SEED_MAJORITY)
    ready = bool(r1 and r2 and r3)

    if not ready:
        label = "substrate_not_ready_requeue"
    elif c1:
        label = "H_replay_rule_state_mismatch_supported"
    elif c2:
        label = "H_replay_rule_state_mismatch_refuted_own_state_lowers"
    else:
        label = "H_replay_rule_state_mismatch_not_supported"
    outcome = "PASS" if (ready and c1) else "FAIL"

    worst_head_seed = min(per_seed, key=lambda p: p["headroom"] - p["lift_margin"])
    preconditions = [
        {"name": "synth_credit_control_ready", "kind": "readiness",
         "description": ("1020's dense-synthetic-credit positive control learns its task AND "
                         "persists in >= 0.8 of cells; min of the two fractions reported"),
         "control": "copy of each cell's own init head on dense synthetic credit (1020, imported)",
         "measured": float(min(gain_frac, pers_frac)),
         "threshold": x1020.CONTROL_READY_FRACTION_FLOOR, "direction": "lower",
         "met": r1},
        {"name": "own_rule_state_manipulation_non_vacuous", "kind": "readiness",
         "description": ("OWN arm: buffer-mean DIRECTION mismatch, mean(1 - cos) between "
                         "each sample's action-time rule_state and the live update-time "
                         "rule_state; the WORST seed is reported, matching the all-seeds "
                         "quantifier. The norm-inclusive mismatch is recorded per cell but "
                         "not gated (the within-episode EMA norm ramp dominates it)"),
         "control": "the live end-of-episode rule_state FAITHFUL scores every sample at",
         "measured": worst_mm, "threshold": MISMATCH_FLOOR, "direction": "lower",
         "offending_cell": (f"REPLAY_OWN_RULE_STATE::seed{worst_mm_seed}"
                            if worst_mm_seed is not None else None),
         "met": r2},
        {"name": "persistence_headroom_eligible_seeds", "kind": "readiness",
         "description": ("count of seeds whose FAITHFUL persistence sits at least one lift "
                         "margin below that seed's positive-control ceiling (the DV has room "
                         "to rise by the amount C1 requires) AND whose two arms replayed a "
                         "byte-identical P1 buffer (pairing witness); others are scoped out"),
         "control": "per-seed in-run bracket [noise floor, synthetic-credit ceiling]",
         "measured": float(n_eligible), "threshold": float(MIN_ELIGIBLE_SEEDS),
         "direction": "lower",
         "worst_seed_headroom_minus_margin": float(worst_head_seed["headroom"]
                                                   - worst_head_seed["lift_margin"]),
         "offending_cell": f"REPLAY_FAITHFUL::seed{worst_head_seed['seed']}",
         "met": r3},
    ]
    criteria = [
        {"name": "C1_own_rule_state_lifts_persistence", "load_bearing": True,
         "passed": c1, "measured": float(n_lift), "threshold": float(SEED_MAJORITY),
         "comparator": ">=", "seeds_eligible": n_eligible,
         "per_seed_lift": {str(p["seed"]): p["lift"] for p in per_seed},
         "per_seed_margin": {str(p["seed"]): p["lift_margin"] for p in per_seed},
         "detail": ("count of ELIGIBLE seeds with pers(OWN) - pers(FAITHFUL) >= "
                    "LIFT_MARGIN_FRACTION x (ceiling - floor) AND a positive lift in the "
                    "first layer's SUMMARY-input-column persistence (a rule-column-only lift "
                    "is input structure, not credit correctness)")},
        {"name": "C2_own_rule_state_lowers_persistence", "load_bearing": False,
         "passed": c2, "measured": float(n_neg), "threshold": float(SEED_MAJORITY),
         "comparator": ">=",
         "detail": ("count of ELIGIBLE seeds with lift <= -margin (same margin as C1, so a "
                    "null does not route to a substantive label by sign alone)")},
    ]
    combination_rule = ("PASS iff readiness (synthetic control, non-vacuous manipulation, "
                        ">= 3 headroom-eligible seeds) AND C1. C2 only selects between the "
                        "two FAIL labels. No cross-arm contrast other than the paired "
                        "per-seed lift.")
    n_lift_pooled_only = sum(1 for p in per_seed if p["lift_clears_margin_pooled_only"])
    non_degen = {
        "C1_own_rule_state_lifts_persistence": bool(r2 and r3),
        "C2_own_rule_state_lowers_persistence": bool(r2),
    }
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "unknown",
        "evidence_direction_note": "diagnostic fan-out leg; directions pinned unknown",
        "outcome": outcome,
        "timestamp_utc": ts,
        "dry_run": bool(dry_run),
        "fanout_qid": FANOUT_QID,
        "fanout_hypotheses": FANOUT_HYPOTHESES,
        "fanout_axis": FANOUT_AXIS,
        "fanout_source_autopsy": FANOUT_SOURCE_AUTOPSY,
        "source_chip_ref": SOURCE_CHIP_REF,
        "red_team_verdict": RED_TEAM_VERDICT,
        "criteria": criteria,
        "combination_rule": combination_rule,
        "interpretation": {
            "label": label,
            "criteria_aggregation": "any",
            "preconditions": preconditions,
            "criteria_non_degenerate": non_degen,
            "combination_rule": combination_rule,
        },
        "per_seed": per_seed,
        "arm_results": rows,
        "diagnostics": {"anchor_reachability": anchor_reachability,
                        "trajectory_identical_all_seeds": all(p["trajectory_identical"]
                                                              for p in per_seed)},
    }
    manifest["readout"] = flat_readout({
        "readiness_synth_control": r1,
        "readiness_manipulation_non_vacuous": r2,
        "readiness_eligible_seeds": r3,
        "n_eligible_seeds": n_eligible,
        "worst_buffer_mismatch_own": worst_mm,
        "n_seeds_lift_clears_margin": n_lift,
        "n_seeds_lift_negative": n_neg,
        "n_seeds_lift_clears_margin_pooled_only": n_lift_pooled_only,
        "mean_lift": float(np.mean([p["lift"] for p in per_seed])),
        "c1_own_rule_state_lifts_persistence": c1,
        "c2_own_rule_state_lowers_persistence": c2,
        "trajectory_identical_all_seeds": manifest["diagnostics"]["trajectory_identical_all_seeds"],
        **{f"lift_seed{p['seed']}": p["lift"] for p in per_seed},
        **{f"persistence_faithful_seed{p['seed']}": p["persistence_faithful"] for p in per_seed},
        **{f"persistence_own_seed{p['seed']}": p["persistence_own"] for p in per_seed},
    })
    manifest["_zg"] = zg
    manifest["_t0"] = t0
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    episodes = ({"p0": 4, "p1": 6, "steps": 12} if args.dry_run
                else {"p0": P0_WARMUP_EPISODES, "p1": P1_BIAS_TRAIN_EPISODES,
                      "steps": STEPS_PER_EPISODE})
    manifest = run_experiment(episodes, args.dry_run)
    zg = manifest.pop("_zg")
    t0 = manifest.pop("_t0")
    out_path = write_flat_manifest(
        manifest,
        dry_run=args.dry_run,
        config={"arms": ARMS, "episodes": episodes, **_config_slice("REPLAY_OWN_RULE_STATE"),
                "mismatch_floor": MISMATCH_FLOOR,
                "lift_margin_fraction": LIFT_MARGIN_FRACTION},
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=zg.stats(),
    )
    print(f"manifest: {out_path}", flush=True)
    print(f"label: {manifest['interpretation']['label']}", flush=True)
    print(f"outcome: {manifest['outcome']}", flush=True)
    print(f"per_seed: {[(p['seed'], round(p['lift'], 4), p['eligible']) for p in manifest['per_seed']]}",
          flush=True)
    return manifest, out_path, args.dry_run


if __name__ == "__main__":
    _manifest, _out_path, _dry = main()
    _o = str(_manifest["outcome"]).upper()
    emit_outcome(
        outcome=_o if _o in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        dry_run=_dry,
    )
