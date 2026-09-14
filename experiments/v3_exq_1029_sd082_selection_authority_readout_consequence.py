"""V3-EXQ-1029 -- SD-082 fan-out leg 3 (H-selection-authority-bounded, axis
intrinsic-architecture): is the tanh-bounded +/-0.1 rule->bias readout
behaviourally INERT at committed selection without modulatory selection
authority, and does it become consequential with authority ON?

RED-TEAM (fable, Step 4.5): CONTESTED -> 6 fixed, 1 scope note; see RED_TEAM_VERDICT.

================================================================================
WHERE THIS COMES FROM
================================================================================
GOV-FANOUT-1 portfolio routed by CONFIRMED failure_autopsy_V3-EXQ-1020_2026-09-11
(REE_assembly 92751187b1), ratified by governance gov-20260911-1612, all legs
user-selected at the autopsy's Step 8 gate; pre-registered in
hypothesis_space_registry.v1.json qid sd082_candidate_discriminating_readout_locus
(NOT re-registered). Chip: chip-20260911-sd082-fanout-portfolio-v2.
Siblings: V3-EXQ-1027 (H-replay-rule-state-mismatch), V3-EXQ-1028
(H-learning-signal-sign + H-learning-signal-noisy).

THE HYPOTHESIS (registry label): "The readout locus is not the binding constraint
at all: a tanh-bounded +/-0.1 bias added to much larger primary E3 scores cannot
change the committed argmin, so no readout design can express a behavioural
consequence in this configuration." Prediction: "With
use_modulatory_selection_authority=True (or a head-ablated contrast), a
behavioural difference attributable to the readout becomes measurable where it
is currently inert."
Substrate facts the autopsy verified: use_modulatory_selection_authority
defaults False (ree_core/utils/config.py:1273) and 1020 never set it; bias_scale
0.1 tanh-bounded (lateral_pfc_analog.py compute_bias); config.py:1262-1263
"fixed small bias magnitudes (~0.05-0.1) added to primary scores whose
raw_score_range is much larger never change the argmin". Authoring probe on this
configuration: raw E3 score range ~5.7 at fresh selects.

WHY THIS LEG MATTERS FOR THE WHOLE PORTFOLIO. If the readout is inert without
authority, no behavioural successor on the 822/1020 path can ever read anything,
whatever 1027/1028 find about the learning signal (autopsy sec 5.1: "any
behavioural probe on this path is inert until selection authority is enabled").

================================================================================
DESIGN -- a 2x2 (authority x rule-readout ablation) as ONE YOKED GROUP per seed
================================================================================
The V3-EXQ-949 yoked design (validated instrument), on V3-EXQ-1020's agent
configuration (imported kwargs, centering True) plus the authority knob:
  AUTH_OFF_INTACT      reference; its committed action drives the environment
  AUTH_OFF_ABLATED     same, rule-readout ablated
  AUTH_ON_INTACT       use_modulatory_selection_authority=True (gain 0.5, basis range)
  AUTH_ON_ABLATED      same, rule-readout ablated
  MAG_INTACT           authority OFF, lateral_pfc bias multiplied by MAG_GAIN (50)
                       AFTER its tanh bound -- raises the readout's own magnitude
                       without rescaling any co-summed channel (gated_policy etc.)
  MAG_ABLATED          same, rule-readout ablated
  AUTH_OFF_INTACT_SELF instrument control: identical to AUTH_OFF_INTACT
  AUTH_ON_INTACT_SELF  instrument control: identical to AUTH_ON_INTACT
WHY THE MAG PAIR (red-team F2). Authority rescales the WHOLE summed modulatory
vector (gated_policy bias + lateral_pfc bias) by its spread, so ablating the rule
part also changes the factor applied to the random-init gated_policy bias, and a
scale-free rescale would amplify any direction difference. A PASS on the ON pair
alone therefore says "direction matters once magnitude is normalised", not "the
readout's magnitude is what was binding". The MAG pair isolates the readout's own
magnitude; it suffixes the PASS label.
E3 NEVER COMMITS IN THIS CONFIGURATION (probe: committed_fraction 0.0 on every
runner), so every behavioural selection is a multinomial draw and the argmin is
the agent's top PREFERENCE, not an executed choice. The load-bearing DV is that
preference; the sampled-action divergence (probe: ON pair 5/71, OFF pair 0/71) is
the behavioural consequence and is recorded alongside. A PASS reads "the readout
changes which candidate E3 ranks first once authority is on".
MAG_GAIN is fixed at 50; authority's own per-tick scale factor is recorded
(authority_scale_factor_median). If authority rescaled by more than MAG_GAIN, a
MAG null is labelled magnitude_test_underpowered, not magnitude_alone_insufficient.
SCOPE (red-team F3): the probe measured |lateral_pfc bias| ~0.0225 against the 0.1
bias_scale, i.e. tanh in its linear region. What this leg can show is that the
readout's RAW magnitude is inert against a ~5.5 E3 score range; it does not show
the tanh ceiling itself binding, which is analytic here, not measured.
Every runner is stepped on the reference's observation at every tick, so every
comparison is at an IDENTICAL world state. Each runner owns a PRIVATE torch +
numpy + python-random stream (swapped in and out around every call), because
six agents interleaving one global stream diverge for stream-position reasons
alone (the 947/949 negative-control finding). The self-yoke pairs run for the
WHOLE run, not a short side-check, and must diverge on exactly 0 ticks.

RULE-READOUT ABLATION = the counterfactual 1020/822f define a "flip" by: the
ablated runner's lateral_pfc.compute_bias is evaluated with rule_state zeroed
for that call only (saved and restored), i.e. bias = head(0, summaries) instead
of head(rule_state, summaries). rule_state itself keeps evolving, so every
other consumer of it is untouched. The head is at its train_rule_bias_head=True
initialisation (random last layer, NOT zero) and is NOT trained here: the
hypothesis is about the magnitude bound bias_scale * tanh(.), which applies to
any trained head identically, and 822f measured the init head flipping MORE
often than a trained one, so init is the stronger positive control. That scope
is stated, not hidden: this leg does not test a trained head's influence.
Warm-up: 60 yoked episodes (CRF maturation; 1020 found flips appear only with
p0 = 60), then 60 scored yoked episodes, 48 steps.

================================================================================
DECISIVE READOUT, PRE-REGISTERED CRITERIA
================================================================================
DV per pair: ARGMIN divergence = among scored ticks on which BOTH runners ran a
genuinely fresh E3 select() (E3-cadence latches cleared to None before every
select_action, counted only when repopulated) AND their raw E3 scores are
bit-identical (torch.equal on last_raw_scores), the fraction on which
argmin(last_scores) -- the post-bias, post-authority score vector -- differs.
WHY ARGMIN, NOT THE COMMITTED ACTION (red-team F1): on every uncommitted fresh
tick E3 commits by torch.multinomial over softmax(-scores / temperature)
(e3_selector.py, uncommitted branch; select_action temperature 1.0), so under a
shared private RNG stream any nonzero bias delta can move the sampled action with
no argmin change. The hypothesis is about the argmin. Committed-action divergence
(all ticks, raw-identical co-fresh ticks, both-committed ticks) and the
committed fraction per runner are recorded as secondary readings.
WHY RAW-IDENTICAL (red-team F4): after a first divergence a runner's own history
(CRF prev-action key, commitment state) can drift from its partner's; counting
only ticks whose raw scores are still identical keeps every counted tick a
same-state comparison. raw_identical_fraction_of_cofresh is recorded.
Readiness, PER SEED:
  R1 instrument: both self-yoke pairs diverge on 0 of all scored ticks
     (threshold 1e-9, upper).
  R2 manipulation non-degenerate: on each intact runner, the fraction of its
     fresh selects on which the head's OWN argmax changes when rule_state is
     zeroed (x1020._raw_ratio_and_flip) >= FLIP_FRACTION_FLOOR (0.01). This is the
     DV-symmetry guard: a rule-attributable bias that were uniform across
     candidates could never move an argmin in either authority condition, and
     the null would be arithmetic. Same statistic family as the DV (an argmax
     change), measured one stage upstream.
  R3 authority engages: on both AUTH_ON runners, the fraction of fresh selects
     with E3's modulatory_authority_active True >= AUTHORITY_ACTIVE_FLOOR (0.5).
  R4 sample size: >= MIN_RAW_IDENTICAL_CO_FRESH (50) raw-identical co-fresh ticks
     for each of the on / off / mag pairs.
  R5 authority reaches selection (red-team F6): AUTH_OFF_INTACT vs AUTH_ON_INTACT
     argmin divergence >= DIV_FLOOR. If authority changes no selection at all, an
     ON-pair null is uninformative.
  A seed failing readiness is scoped out; >= 3 ready seeds are required.
Criteria over ready seeds:
  C1 (load-bearing): AUTH_ON pair co-fresh divergence >= DIV_FLOOR (0.02) on >= 3.
  C2 (load-bearing): AUTH_OFF pair argmin divergence <= NULL_CEILING (0.005)
     on >= 3. Stated plainly (red-team F5): at ~280 raw-identical co-fresh ticks
     per seed (60 scored episodes) this admits at most ONE divergent tick.
  C4 (recorded, suffixes PASS): MAG pair argmin divergence >= DIV_FLOOR on >= 3.
  C3 (recorded): ON minus OFF divergence >= INTERACTION_MARGIN (0.02) on >= 3.
  combination: PASS iff >= 3 ready seeds AND C1 AND C2 (criteria_aggregation all).

VERDICT GRID
  < 3 ready seeds             -> substrate_not_ready_requeue (FAIL)
  C1 and C2                   -> H_selection_authority_bounded_supported_<m> (PASS):
                                 inert at native magnitude, consequential under
                                 authority; <m> = magnitude_alone_suffices (C4) or
                                 magnitude_alone_insufficient (not C4).
  C2 only                     -> readout_inert_even_with_authority (FAIL,
                                 informative): the bound is not the only binding
                                 constraint -- authority does not rescue it.
  C1 only                     -> readout_consequential_without_authority (FAIL,
                                 informative): the +/-0.1 bound is NOT binding;
                                 H refuted for this configuration.
  neither                     -> anomaly_off_divergent_on_null (FAIL).
Directions pinned "unknown" (diagnostic). A PASS licenses a design rule for
successors (enable authority before reading behaviour); it is not evidence for
or against SD-082's readout content.

DV-SYMMETRY DECLARATION, per runner pair:
  AUTH_OFF pair: the manipulation changes the per-candidate bias by
    head(rs, s_k) - head(0, s_k), non-uniform across k through the ReLU (R2
    measures it); argmax is invariant only under a UNIFORM additive shift, so the
    DV is not invariant by construction -- a zero divergence is a magnitude fact.
  AUTH_ON pair: authority rescales the COMBINED modulatory vector by its own
    spread; a uniform rule-attributable shift would leave spread, scale and
    argmin unchanged, so again only a non-uniform part (R2) can move the DV. Not
    invariant by construction.
  Self-yoke pairs: manipulation is identity; divergence must be 0 (R1).

================================================================================
QUEUE-EXPERIMENT GATE DISPOSITIONS (2026-09-14)
================================================================================
Step 2.4 GOV-REUSE-1: decisive readout = rule-ablation committed-action
divergence at authority ON vs OFF on the SD-082 path. 1020 has no ablation arm
and never set authority; 949 measured authority on MECH-314b's channel, not the
lateral_pfc readout -> not recoverable, run.
Step 2.5/2.5a: authority rescale is built (e3_selector.py, modulatory
authority block; the lateral_pfc bias enters score_bias -> _modulatory_accum).
Authoring probes (seed 611, this config): (a) a 6-episode cold-start yoke:
self-yoke 0, AUTH_OFF pair 0/29 co-fresh, AUTH_ON pair 3/29; (b) THIS driver's
_run_seed at warmup 20 / score 15: both self-yokes 0/649 ticks, AUTH_OFF pair
0/71 co-fresh, AUTH_ON pair 5/71 (0.070), head-level flip fraction 0.21-0.29,
authority active on 100% of AUTH_ON fresh selects. Every readiness gate is
reachable, and the pre-registered PASS branch is reachable -- which is also why
this leg's prior is lopsided; the FAIL branches remain live (a near-tie regime
can flip the OFF pair; a different seed can null the ON pair).
Step 2.5b brake: SD-082 counts 3, released by the 822f autopsy; new EXQ number on
the intrinsic-architecture axis the 1020 autopsy licensed, diagnostic.
Step 2.5c: module footprint = 1020's plus the e3_selector authority block; no
open corrupting substrate_queue entry names e3_selector; 1020's call-trace
disposition for the rest is carried forward. The authority flag itself is
implemented_pending_validation in substrate_queue (degrading class at most) and
is disclosed here as the manipulated variable, not a hidden confound.
Step 2.6 ethics: all-false / allow.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
from experiments._lib.arm_fingerprint import arm_cell, reset_all_rng  # noqa: E402
from experiments._lib.readiness_anchor import assert_anchor_reachable  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
import experiments.v3_exq_1020_sd082_learning_signal_probe as x1020  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_1029_sd082_selection_authority_readout_consequence"
QUEUE_ID = "V3-EXQ-1029"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS = ["SD-082"]
FANOUT_QID = "sd082_candidate_discriminating_readout_locus"
FANOUT_HYPOTHESES = ["H-selection-authority-bounded"]
FANOUT_AXIS = "intrinsic-architecture"
FANOUT_SOURCE_AUTOPSY = "failure_autopsy_V3-EXQ-1020_2026-09-11"
SOURCE_CHIP_REF = "chip-20260911-sd082-fanout-portfolio-v2"
RED_TEAM_VERDICT = (
    "red-team (fable): CONTESTED -> 7 findings; 6 FIXED, 1 recorded as scope. "
    "F1 uncommitted E3 ticks commit by torch.multinomial, so committed-action divergence "
    "is not an argmin fact -> load-bearing DV is now argmin(last_scores) divergence on "
    "co-fresh RAW-IDENTICAL ticks; committed-action divergence recorded as secondary. "
    "F2 authority rescales gated_policy's bias together with the readout and is "
    "scale-free -> added MAG pair (authority OFF, lateral_pfc bias x MAG_GAIN) so a "
    "PASS says whether magnitude alone suffices; PASS label text narrowed. F3 bias sits "
    "at ~0.02, in tanh's linear region, far below the 0.1 bound -> stated as scope "
    "(the binding quantity is raw readout magnitude, not the bound). F4 post-divergence "
    "state echo -> DV counted only on ticks where the pair's raw E3 scores are "
    "identical, raw-identical fraction recorded. F5 thresholds are single-event tests "
    "-> SCORE_EPISODES 40 -> 60, NULL_CEILING stated as <= 1 divergent tick in ~280. "
    "F6 authority-active is not reach -> authority_only pair argmin divergence added "
    "as readiness R5. F7 over-reaching label text removed.")

SEEDS = list(x1020.SEEDS)
CENTERING = True
WARMUP_EPISODES = 60
SCORE_EPISODES = 60
TOTAL_EPISODES = WARMUP_EPISODES + SCORE_EPISODES
STEPS_PER_EPISODE = x1020.STEPS_PER_EPISODE
MODULATORY_AUTHORITY_GAIN = 0.5
# Post-bound multiplier on the lateral_pfc bias for the MAG pair (authority OFF).
# Probe: lateral_pfc |bias| mean ~0.0225 vs E3 raw score range ~5.5; x50 puts the
# readout at ~1.1, i.e. ~0.2 of the raw range -- same order as authority's
# 0.5 x raw_range target but WITHOUT rescaling any other channel.
MAG_GAIN = 50.0

# name -> (authority, rule_readout_ablated, bias_gain)
RUNNERS = {
    "AUTH_OFF_INTACT": (False, False, 1.0),
    "AUTH_OFF_ABLATED": (False, True, 1.0),
    "AUTH_ON_INTACT": (True, False, 1.0),
    "AUTH_ON_ABLATED": (True, True, 1.0),
    "MAG_INTACT": (False, False, MAG_GAIN),
    "MAG_ABLATED": (False, True, MAG_GAIN),
    "AUTH_OFF_INTACT_SELF": (False, False, 1.0),
    "AUTH_ON_INTACT_SELF": (True, False, 1.0),
}
REFERENCE = "AUTH_OFF_INTACT"
PAIRS = {
    "off_pair": ("AUTH_OFF_INTACT", "AUTH_OFF_ABLATED"),
    "on_pair": ("AUTH_ON_INTACT", "AUTH_ON_ABLATED"),
    "mag_pair": ("MAG_INTACT", "MAG_ABLATED"),
    "self_off": ("AUTH_OFF_INTACT", "AUTH_OFF_INTACT_SELF"),
    "self_on": ("AUTH_ON_INTACT", "AUTH_ON_INTACT_SELF"),
    "authority_only": ("AUTH_OFF_INTACT", "AUTH_ON_INTACT"),
    "magnitude_only": ("AUTH_OFF_INTACT", "MAG_INTACT"),
}

# Pre-registered.
FLIP_FRACTION_FLOOR = 0.01
AUTHORITY_ACTIVE_FLOOR = 0.5
MIN_RAW_IDENTICAL_CO_FRESH = 50
DIV_FLOOR = 0.02
NULL_CEILING = 0.005      # at ~280 raw-identical co-fresh ticks: at most ONE divergent tick
INTERACTION_MARGIN = 0.02
SEED_MAJORITY = 3
MIN_READY_SEEDS = 3

ANCHOR_SOURCE = ("V3-EXQ-1029 authoring probe seed 611 (warmup 20, score 15) + "
                 "v3_exq_1020_sd082_learning_signal_probe_20260911T003146Z_v3 ARM_ON "
                 "n_fresh_flip_ticks / n_fresh_select_ticks")
SELF_YOKE_REFERENCE_CELLS = [{"r1_self_yoke_max_divergent_ticks": 0.0}]
HEAD_FLIP_REFERENCE_CELLS = [
    {"r2_min_head_flip_fraction": 0.2326},
    {"r2_min_head_flip_fraction": 0.0300},
    {"r2_min_head_flip_fraction": 0.1022},
    {"r2_min_head_flip_fraction": 0.1283},
    {"r2_min_head_flip_fraction": 0.2252},
    {"r2_min_head_flip_fraction": 0.0604},
]
AUTHORITY_REFERENCE_CELLS = [{"r3_min_authority_active_fraction": 1.0}]
# authority_only pair, probe: 12 of 71 co-fresh committed actions differed.
AUTHORITY_REACH_REFERENCE_CELLS = [{"r5_authority_only_argmin_divergence": 12.0 / 71.0}]


def _r1_ok(c: Dict[str, Any]) -> bool:
    v = c.get("r1_self_yoke_max_divergent_ticks")
    return bool(v is not None and v <= 1e-9)


def _r2_ok(c: Dict[str, Any]) -> bool:
    v = c.get("r2_min_head_flip_fraction")
    return bool(v is not None and v >= FLIP_FRACTION_FLOOR)


def _r3_ok(c: Dict[str, Any]) -> bool:
    v = c.get("r3_min_authority_active_fraction")
    return bool(v is not None and v >= AUTHORITY_ACTIVE_FLOOR)


def _r5_ok(c: Dict[str, Any]) -> bool:
    v = c.get("r5_authority_only_argmin_divergence")
    return bool(v is not None and v >= DIV_FLOOR)


def _assert_anchors() -> List[Dict[str, Any]]:
    return [
        assert_anchor_reachable(anchor_name="self_yoke_bit_identical",
                                reference_cells=SELF_YOKE_REFERENCE_CELLS, score_fn=_r1_ok,
                                threshold=1.0, reference_source=ANCHOR_SOURCE),
        assert_anchor_reachable(anchor_name="rule_readout_ablation_moves_head_argmax",
                                reference_cells=HEAD_FLIP_REFERENCE_CELLS, score_fn=_r2_ok,
                                threshold=MIN_READY_SEEDS / 5.0, reference_source=ANCHOR_SOURCE),
        assert_anchor_reachable(anchor_name="authority_engages_on_auth_on_runners",
                                reference_cells=AUTHORITY_REFERENCE_CELLS, score_fn=_r3_ok,
                                threshold=1.0, reference_source=ANCHOR_SOURCE),
        assert_anchor_reachable(anchor_name="authority_reaches_selection",
                                reference_cells=AUTHORITY_REACH_REFERENCE_CELLS, score_fn=_r5_ok,
                                threshold=1.0, reference_source=ANCHOR_SOURCE),
    ]


def _make_agent(env, authority: bool) -> REEAgent:
    """1020's _make_agent kwargs verbatim (centering True) plus the authority knob."""
    return REEAgent(REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        alpha_world=0.9,
        use_lateral_pfc_analog=True,
        lateral_pfc_train_rule_bias_head=True,
        lateral_pfc_rule_readout_consumer=True,
        lateral_pfc_capture_head_diagnostics=True,
        candidate_summary_source="proposer_post_action",
        use_gated_policy=True,
        use_candidate_rule_field=True,
        crf_persist_rules_across_episode_reset=True,
        crf_mature_pool_dynamics=True,
        crf_availability_maintenance=True,
        crf_maintenance_floor=0.45,
        crf_maintenance_couple_to_theta=True,
        crf_tolerance_conflict_cap=3,
        crf_cue_centering=CENTERING,
        crf_cue_baseline_alpha=0.02,
        use_modulatory_selection_authority=bool(authority),
        modulatory_authority_gain=MODULATORY_AUTHORITY_GAIN,
    ))


def _config_slice() -> Dict[str, Any]:
    s = x1020._config_slice(CENTERING)
    s.pop("reinforce", None)
    s["schedule"] = {"warmup": WARMUP_EPISODES, "score": SCORE_EPISODES,
                     "steps": STEPS_PER_EPISODE}
    s["runners"] = {k: {"authority": v[0], "rule_readout_ablated": v[1], "bias_gain": v[2]}
                    for k, v in RUNNERS.items()}
    s["modulatory_authority_gain"] = MODULATORY_AUTHORITY_GAIN
    s["mag_gain"] = MAG_GAIN
    s["head_trained"] = False
    return s


class _Runner:
    """One agent on a private RNG stream (torch + numpy + python random)."""

    def __init__(self, env, authority: bool, ablated: bool, bias_gain: float) -> None:
        self.agent = _make_agent(env, authority)
        self.authority = bool(authority)
        self.ablated = bool(ablated)
        self.bias_gain = float(bias_gain)
        self._rng = (torch.get_rng_state(), np.random.get_state(), random.getstate())
        self.counters = {"dispatch": 0, "fallback": 0}
        lpfc = self.agent.lateral_pfc
        if ablated or self.bias_gain != 1.0:
            orig: Callable = lpfc.compute_bias

            def _wrapped_compute_bias(summaries, _orig=orig, _lpfc=lpfc,
                                      _ablate=self.ablated, _gain=self.bias_gain):
                if _ablate:
                    saved = _lpfc.rule_state.detach().clone()
                    with torch.no_grad():
                        _lpfc.rule_state.zero_()
                    try:
                        out = _orig(summaries)
                    finally:
                        with torch.no_grad():
                            _lpfc.rule_state.copy_(saved)
                else:
                    out = _orig(summaries)
                return out * _gain if _gain != 1.0 else out

            lpfc.compute_bias = _wrapped_compute_bias
        self.n_fresh = 0
        self.n_committed = 0
        self.n_head_flip_measured = 0
        self.n_head_flip = 0
        self.n_auth_active = 0
        self.raw_ranges: List[float] = []
        self.post_ranges: List[float] = []
        self.bias_abs: List[float] = []
        self.auth_scale: List[float] = []

    def _swap(self, fn):
        amb = (torch.get_rng_state(), np.random.get_state(), random.getstate())
        torch.set_rng_state(self._rng[0])
        np.random.set_state(self._rng[1])
        random.setstate(self._rng[2])
        try:
            return fn()
        finally:
            self._rng = (torch.get_rng_state(), np.random.get_state(), random.getstate())
            torch.set_rng_state(amb[0])
            np.random.set_state(amb[1])
            random.setstate(amb[2])

    def reset_episode(self) -> None:
        self._swap(self.agent.reset)

    def choose(self, obs: Dict[str, Any], scoring: bool) -> Dict[str, Any]:
        return self._swap(lambda: self._choose(obs, scoring))

    def _choose(self, obs: Dict[str, Any], scoring: bool) -> Dict[str, Any]:
        agent = self.agent
        lpfc = agent.lateral_pfc
        wd = agent.config.latent.world_dim
        latent = agent.sense(obs["body_state"], obs["world_state"])
        ticks = agent.clock.advance()
        e1 = (agent._e1_tick(latent) if ticks.get("e1_tick")
              else torch.zeros(1, wd, device=agent.device))
        candidates = agent.generate_trajectories(latent, e1, ticks)
        head_flip: Optional[bool] = None
        if candidates and len(candidates) >= 2 and bool(ticks.get("e3_tick", False)):
            cs = x1020._candidate_summaries(agent, candidates, self.counters)
            if (cs is not None and torch.isfinite(cs).all()
                    and float(lpfc.rule_state.norm()) > x1020.RULE_STATE_LIVE_FLOOR):
                rf = x1020._raw_ratio_and_flip(lpfc, cs)
                if rf is not None:
                    head_flip = bool(rf[1] > 0.5)
        agent.e3.last_raw_scores = None
        agent.e3.last_scores = None
        agent.e3.last_score_diagnostics = None
        agent._last_e3_selection_result = None
        action = agent.select_action(candidates, ticks)
        raw = agent.e3.last_raw_scores
        post = agent.e3.last_scores
        diag = agent.e3.last_score_diagnostics
        sel = agent._last_e3_selection_result
        fresh = raw is not None and post is not None
        if action is None:
            action = torch.zeros(1, 4, device=agent.device)
            action[0, int(np.random.randint(0, 4))] = 1.0
            agent._last_action = action
        committed = bool(getattr(sel, "committed", False)) if (fresh and sel is not None) else None
        out = {"action": int(action[0].argmax().item()), "fresh": bool(fresh),
               "argmin": (int(post.argmin().item()) if fresh and post.numel() > 0 else None),
               "raw": (raw.detach().clone() if fresh else None),
               "committed": committed}
        if scoring and fresh:
            self.n_fresh += 1
            if committed:
                self.n_committed += 1
            if raw.numel() > 1:
                self.raw_ranges.append(float((raw.max() - raw.min()).item()))
                self.post_ranges.append(float((post.max() - post.min()).item()))
            if isinstance(diag, dict) and bool(diag.get("modulatory_authority_active", False)):
                self.n_auth_active += 1
                sf = diag.get("modulatory_authority_scale_factor")
                if sf is not None and math.isfinite(float(sf)):
                    self.auth_scale.append(float(sf))
            if head_flip is not None:
                self.n_head_flip_measured += 1
                self.n_head_flip += int(head_flip)
            self.bias_abs.append(float(getattr(lpfc, "_last_bias_abs_mean", 0.0)))
        return out

    def summary(self) -> Dict[str, Any]:
        def _m(xs):
            return float(np.mean(xs)) if xs else None
        return {
            "authority": self.authority, "rule_readout_ablated": self.ablated,
            "bias_gain": self.bias_gain,
            "n_fresh_scored": self.n_fresh,
            "committed_fraction": (self.n_committed / self.n_fresh if self.n_fresh else None),
            "head_flip_fraction": (self.n_head_flip / self.n_head_flip_measured
                                   if self.n_head_flip_measured else None),
            "n_head_flip_measured": self.n_head_flip_measured,
            "authority_active_fraction": (self.n_auth_active / self.n_fresh
                                          if self.n_fresh else None),
            "raw_score_range_mean": _m(self.raw_ranges),
            "post_score_range_mean": _m(self.post_ranges),
            "lpfc_bias_abs_mean_prebias_gain": _m(self.bias_abs),
            "authority_scale_factor_median": (float(np.median(self.auth_scale))
                                              if self.auth_scale else None),
            "summary_fallback_calls": self.counters["fallback"],
        }


def _empty_pair() -> Dict[str, Any]:
    return {"n_tick": 0, "d_action_tick": 0, "n_cofresh": 0, "n_raw_identical": 0,
            "d_argmin": 0, "d_action_raw_identical": 0, "n_both_committed": 0,
            "d_action_both_committed": 0, "first_argmin_div_tick": None,
            "argmin_divergence_runs": 0, "_in_run": False}


def _run_seed(seed: int, episodes: Dict[str, int], zg: ZGoalStreamAccumulator) -> Dict[str, Any]:
    warm, score, steps = episodes["warmup"], episodes["score"], episodes["steps"]
    total = warm + score
    print(f"Seed {seed} Condition yoked_authority_ablation_magnitude", flush=True)
    with arm_cell(seed, config_slice=_config_slice(), script_path=Path(__file__),
                  extra_substrate_paths=[Path(x1020.__file__)]) as cell:
        env = x1020._build_env(seed)
        runners: Dict[str, _Runner] = {}
        for name, (auth, abl, gain) in RUNNERS.items():
            reset_all_rng(seed)
            runners[name] = _Runner(env, auth, abl, gain)
        reset_all_rng(seed)
        env = x1020._build_env(seed)
        stats = {p: _empty_pair() for p in PAIRS}
        scored_tick = 0
        for ep in range(total):
            scoring = ep >= warm
            _, obs = env.reset()
            for r in runners.values():
                r.reset_episode()
            for _s in range(steps):
                outs = {k: r.choose(obs, scoring) for k, r in runners.items()}
                if scoring:
                    for p, (a, b) in PAIRS.items():
                        st = stats[p]
                        oa, ob = outs[a], outs[b]
                        st["n_tick"] += 1
                        st["d_action_tick"] += int(oa["action"] != ob["action"])
                        if oa["fresh"] and ob["fresh"]:
                            st["n_cofresh"] += 1
                            if torch.equal(oa["raw"], ob["raw"]):
                                st["n_raw_identical"] += 1
                                div = oa["argmin"] != ob["argmin"]
                                st["d_argmin"] += int(div)
                                st["d_action_raw_identical"] += int(oa["action"] != ob["action"])
                                if div and st["first_argmin_div_tick"] is None:
                                    st["first_argmin_div_tick"] = scored_tick
                                if div and not st["_in_run"]:
                                    st["argmin_divergence_runs"] += 1
                                st["_in_run"] = bool(div)
                            if oa["committed"] and ob["committed"]:
                                st["n_both_committed"] += 1
                                st["d_action_both_committed"] += int(oa["action"] != ob["action"])
                    scored_tick += 1
                _, _h, done, _info, obs = env.step(outs[REFERENCE]["action"])
                if done:
                    break
            if (ep + 1) % 10 == 0 or (ep + 1) == total:
                print(f"  [train] yoked seed={seed} ep {ep+1}/{total} "
                      f"on={stats['on_pair']['d_argmin']}/{stats['on_pair']['n_raw_identical']} "
                      f"off={stats['off_pair']['d_argmin']}/{stats['off_pair']['n_raw_identical']} "
                      f"mag={stats['mag_pair']['d_argmin']}/{stats['mag_pair']['n_raw_identical']} "
                      f"self={stats['self_off']['d_action_tick']}+{stats['self_on']['d_action_tick']}",
                      flush=True)
        for r in runners.values():
            zg.observe(r.agent)
        summaries = {k: r.summary() for k, r in runners.items()}
        pairs_out: Dict[str, Any] = {}
        for p, st in stats.items():
            st = {k: v for k, v in st.items() if not k.startswith("_")}
            pairs_out[p] = {
                **st,
                "argmin_divergence": (st["d_argmin"] / st["n_raw_identical"])
                if st["n_raw_identical"] else None,
                "raw_identical_fraction_of_cofresh": (st["n_raw_identical"] / st["n_cofresh"])
                if st["n_cofresh"] else None,
                "action_divergence_cofresh_raw_identical": (st["d_action_raw_identical"]
                                                            / st["n_raw_identical"])
                if st["n_raw_identical"] else None,
                "action_divergence_both_committed": (st["d_action_both_committed"]
                                                     / st["n_both_committed"])
                if st["n_both_committed"] else None,
            }
        flips = [summaries["AUTH_OFF_INTACT"]["head_flip_fraction"],
                 summaries["AUTH_ON_INTACT"]["head_flip_fraction"],
                 summaries["MAG_INTACT"]["head_flip_fraction"]]
        auths = [summaries["AUTH_ON_INTACT"]["authority_active_fraction"],
                 summaries["AUTH_ON_ABLATED"]["authority_active_fraction"]]
        row: Dict[str, Any] = {
            "arm_id": "yoked_group", "seed": seed, "runners": summaries, "pairs": pairs_out,
            "r1_self_yoke_max_divergent_ticks": float(max(stats["self_off"]["d_action_tick"],
                                                          stats["self_on"]["d_action_tick"],
                                                          stats["self_off"]["d_argmin"],
                                                          stats["self_on"]["d_argmin"])),
            "r2_min_head_flip_fraction": (min(flips) if all(f is not None for f in flips) else None),
            "r3_min_authority_active_fraction": (min(auths) if all(a is not None for a in auths)
                                                 else None),
            "r4_min_raw_identical_cofresh": float(min(pairs_out["on_pair"]["n_raw_identical"],
                                                      pairs_out["off_pair"]["n_raw_identical"],
                                                      pairs_out["mag_pair"]["n_raw_identical"])),
            "r5_authority_only_argmin_divergence": pairs_out["authority_only"]["argmin_divergence"],
            "summary_fallback_calls_max": max(s["summary_fallback_calls"] for s in summaries.values()),
        }
        row["r1_met"] = _r1_ok(row)
        row["r2_met"] = _r2_ok(row)
        row["r3_met"] = _r3_ok(row)
        row["r4_met"] = bool(row["r4_min_raw_identical_cofresh"] >= MIN_RAW_IDENTICAL_CO_FRESH)
        row["r5_met"] = _r5_ok(row)
        row["ready"] = bool(row["r1_met"] and row["r2_met"] and row["r3_met"]
                            and row["r4_met"] and row["r5_met"])
        on_d = pairs_out["on_pair"]["argmin_divergence"]
        off_d = pairs_out["off_pair"]["argmin_divergence"]
        mag_d = pairs_out["mag_pair"]["argmin_divergence"]
        row["c1_on_divergent"] = bool(on_d is not None and on_d >= DIV_FLOOR)
        row["c2_off_null"] = bool(off_d is not None and off_d <= NULL_CEILING)
        row["c3_interaction"] = bool(on_d is not None and off_d is not None
                                     and (on_d - off_d) >= INTERACTION_MARGIN)
        row["c4_magnitude_alone_divergent"] = bool(mag_d is not None and mag_d >= DIV_FLOOR)
        cell.stamp(row)
    print(f"verdict: {'PASS' if row['ready'] else 'FAIL'}", flush=True)
    return row


def _worst(rows, key, mode):
    vals = [(r[key], r["seed"]) for r in rows if r.get(key) is not None]
    if not vals:
        return None, None
    v = (min if mode == "min" else max)(vals)
    return v[0], f"seed{v[1]}"


def run_experiment(episodes: Dict[str, int], dry_run: bool) -> Dict[str, Any]:
    t0 = time.perf_counter()
    anchors = _assert_anchors()
    zg = ZGoalStreamAccumulator()
    rows = [_run_seed(s, episodes, zg) for s in SEEDS]
    ready_rows = [r for r in rows if r["ready"]]
    n_ready = len(ready_rows)
    enough = n_ready >= MIN_READY_SEEDS
    n_c1 = sum(1 for r in ready_rows if r["c1_on_divergent"])
    n_c2 = sum(1 for r in ready_rows if r["c2_off_null"])
    n_c3 = sum(1 for r in ready_rows if r["c3_interaction"])
    n_c4 = sum(1 for r in ready_rows if r["c4_magnitude_alone_divergent"])
    c1 = bool(enough and n_c1 >= SEED_MAJORITY)
    c2 = bool(enough and n_c2 >= SEED_MAJORITY)
    c3 = bool(enough and n_c3 >= SEED_MAJORITY)
    c4 = bool(enough and n_c4 >= SEED_MAJORITY)
    if not enough:
        label = "substrate_not_ready_requeue"
    elif c1 and c2:
        scales = [r["runners"]["AUTH_ON_INTACT"]["authority_scale_factor_median"]
                  for r in ready_rows
                  if r["runners"]["AUTH_ON_INTACT"]["authority_scale_factor_median"] is not None]
        underpowered = bool(scales and float(np.median(scales)) > MAG_GAIN)
        suffix = ("magnitude_alone_suffices" if c4 else
                  ("magnitude_test_underpowered" if underpowered else "magnitude_alone_insufficient"))
        label = "H_selection_authority_bounded_supported_" + suffix
    elif c2:
        label = "readout_inert_even_with_authority"
    elif c1:
        label = "readout_consequential_without_authority"
    else:
        label = "anomaly_off_divergent_on_null"
    outcome = "PASS" if (enough and c1 and c2) else "FAIL"
    if outcome == "PASS" and c4:
        reading = ("The readout never moves the argmin at its native magnitude (authority OFF), "
                   "moves it under gap-relative authority, AND moves it when only its own "
                   "magnitude is raised: the binding constraint is readout MAGNITUDE relative "
                   "to the E3 score range, not something authority-specific.")
    elif outcome == "PASS" and label.endswith("underpowered"):
        reading = ("Inert at native magnitude and consequential under authority; the "
                   "magnitude-only pair did not move the argmin, but authority's median scale "
                   "factor exceeded MAG_GAIN, so that null does not separate magnitude from "
                   "authority.")
    elif outcome == "PASS":
        reading = ("Inert at native magnitude, consequential under authority, but NOT when its "
                   "own magnitude alone is raised x MAG_GAIN: authority's effect is not reducible "
                   "to the readout's magnitude (it also rescales co-summed channels such as the "
                   "gated_policy bias).")
    else:
        reading = f"see label {label}"

    w1, w1c = _worst(rows, "r1_self_yoke_max_divergent_ticks", "max")
    w2, w2c = _worst(rows, "r2_min_head_flip_fraction", "min")
    w3, w3c = _worst(rows, "r3_min_authority_active_fraction", "min")
    w4, w4c = _worst(rows, "r4_min_raw_identical_cofresh", "min")
    w5, w5c = _worst(rows, "r5_authority_only_argmin_divergence", "min")
    preconditions = [
        {"name": "ready_seed_count", "kind": "readiness",
         "description": "seeds passing all five per-seed readiness gates",
         "measured": float(n_ready), "threshold": float(MIN_READY_SEEDS), "direction": "lower",
         "met": bool(enough)},
        {"name": "self_yoke_bit_identical", "kind": "readiness",
         "description": ("max divergent scored ticks (committed action or argmin) across both "
                         "self-yoke pairs; worst seed"),
         "control": "identical runner yoked against itself on a private RNG stream",
         "measured": w1, "threshold": 1e-9, "direction": "upper", "offending_cell": w1c,
         "met": bool(w1 is not None and w1 <= 1e-9)},
        {"name": "rule_readout_ablation_moves_head_argmax", "kind": "readiness",
         "description": ("min over intact runners (OFF, ON, MAG) of the fraction of fresh selects "
                         "on which the head's own argmax changes with rule_state zeroed; worst seed"),
         "control": "x1020._raw_ratio_and_flip on live candidate summaries",
         "measured": w2, "threshold": FLIP_FRACTION_FLOOR, "direction": "lower",
         "offending_cell": w2c, "met": bool(w2 is not None and w2 >= FLIP_FRACTION_FLOOR)},
        {"name": "authority_engages_on_auth_on_runners", "kind": "readiness",
         "description": "min fraction of fresh selects with modulatory_authority_active; worst seed",
         "measured": w3, "threshold": AUTHORITY_ACTIVE_FLOOR, "direction": "lower",
         "offending_cell": w3c, "met": bool(w3 is not None and w3 >= AUTHORITY_ACTIVE_FLOOR)},
        {"name": "raw_identical_cofresh_sample_size", "kind": "readiness",
         "description": "min raw-identical co-fresh ticks over on/off/mag pairs; worst seed",
         "measured": w4, "threshold": float(MIN_RAW_IDENTICAL_CO_FRESH), "direction": "lower",
         "offending_cell": w4c, "met": bool(w4 is not None and w4 >= MIN_RAW_IDENTICAL_CO_FRESH)},
        {"name": "authority_reaches_selection", "kind": "readiness",
         "description": ("argmin divergence between AUTH_OFF_INTACT and AUTH_ON_INTACT on "
                         "raw-identical co-fresh ticks: authority must change SOME selections, "
                         "or an ON-pair null is uninformative; worst seed"),
         "control": ("authoring probe committed-action proxy 12/71 on seed 611 (the argmin "
                     "statistic itself was not recorded by that probe)"),
         "measured": w5, "threshold": DIV_FLOOR, "direction": "lower", "offending_cell": w5c,
         "met": bool(w5 is not None and w5 >= DIV_FLOOR)},
    ]
    criteria = [
        {"name": "C1_on_pair_argmin_divergent", "load_bearing": True, "passed": c1,
         "measured": float(n_c1), "threshold": float(SEED_MAJORITY), "comparator": ">=",
         "per_cell_floor": DIV_FLOOR,
         "per_seed": {str(r["seed"]): r["pairs"]["on_pair"]["argmin_divergence"] for r in rows}},
        {"name": "C2_off_pair_argmin_null", "load_bearing": True, "passed": c2,
         "measured": float(n_c2), "threshold": float(SEED_MAJORITY), "comparator": ">=",
         "per_cell_ceiling": NULL_CEILING,
         "per_seed": {str(r["seed"]): r["pairs"]["off_pair"]["argmin_divergence"] for r in rows}},
        {"name": "C3_interaction", "load_bearing": False, "passed": c3,
         "measured": float(n_c3), "threshold": float(SEED_MAJORITY), "comparator": ">=",
         "per_cell_margin": INTERACTION_MARGIN},
        {"name": "C4_magnitude_alone_divergent", "load_bearing": False, "passed": c4,
         "measured": float(n_c4), "threshold": float(SEED_MAJORITY), "comparator": ">=",
         "per_cell_floor": DIV_FLOOR, "mag_gain": MAG_GAIN,
         "per_seed": {str(r["seed"]): r["pairs"]["mag_pair"]["argmin_divergence"] for r in rows}},
    ]
    combination_rule = ("PASS iff >= 3 readiness-green seeds AND C1 (authority-ON pair argmin "
                        "divergent) AND C2 (authority-OFF pair argmin null). DV counted on "
                        "co-fresh ticks whose raw E3 scores are identical across the pair. C3 "
                        "recorded; C4 (magnitude-only pair) suffixes the PASS label.")
    non_degen = {"C1_on_pair_argmin_divergent": bool(enough),
                 "C2_off_pair_argmin_null": bool(enough),
                 "C3_interaction": bool(enough), "C4_magnitude_alone_divergent": bool(enough)}
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
        "interpretation": {"label": label, "reading": reading, "criteria_aggregation": "all",
                           "preconditions": preconditions,
                           "criteria_non_degenerate": non_degen,
                           "combination_rule": combination_rule},
        "arm_results": rows,
        "diagnostics": {"anchor_reachability": anchors},
    }
    readout: Dict[str, Any] = {"n_ready_seeds": n_ready, "n_c1_seeds": n_c1,
                               "n_c2_seeds": n_c2, "n_c3_seeds": n_c3, "n_c4_seeds": n_c4,
                               "c1_on_pair_argmin_divergent": c1, "c2_off_pair_argmin_null": c2,
                               "c3_interaction": c3, "c4_magnitude_alone_divergent": c4}
    for r in rows:
        for p in ("on_pair", "off_pair", "mag_pair", "authority_only", "magnitude_only"):
            readout[f"{p}_argmin_div_seed{r['seed']}"] = r["pairs"][p]["argmin_divergence"]
        readout[f"ready_seed{r['seed']}"] = r["ready"]
    manifest["readout"] = flat_readout(readout)
    manifest["_zg"] = zg
    manifest["_t0"] = t0
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    episodes = ({"warmup": 2, "score": 3, "steps": 12} if args.dry_run
                else {"warmup": WARMUP_EPISODES, "score": SCORE_EPISODES,
                      "steps": STEPS_PER_EPISODE})
    manifest = run_experiment(episodes, args.dry_run)
    zg = manifest.pop("_zg")
    t0 = manifest.pop("_t0")
    out_path = write_flat_manifest(
        manifest, dry_run=args.dry_run,
        config={"episodes": episodes, **_config_slice(),
                "thresholds": {"flip_fraction_floor": FLIP_FRACTION_FLOOR,
                               "authority_active_floor": AUTHORITY_ACTIVE_FLOOR,
                               "min_raw_identical_cofresh": MIN_RAW_IDENTICAL_CO_FRESH,
                               "div_floor": DIV_FLOOR, "null_ceiling": NULL_CEILING,
                               "interaction_margin": INTERACTION_MARGIN}},
        seeds=SEEDS, script_path=Path(__file__), started_at=t0,
        z_goal_stream_stats=zg.stats(),
    )
    print(f"manifest: {out_path}", flush=True)
    print(f"label: {manifest['interpretation']['label']}", flush=True)
    print(f"outcome: {manifest['outcome']}", flush=True)
    return manifest, out_path, args.dry_run


if __name__ == "__main__":
    _manifest, _out_path, _dry = main()
    _o = str(_manifest["outcome"]).upper()
    emit_outcome(outcome=_o if _o in ("PASS", "FAIL") else "FAIL",
                 manifest_path=_out_path, dry_run=_dry)
