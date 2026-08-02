"""V3-EXQ-878: MECH-332 -- efference-copy comparator vs descending-modulation
double dissociation.

Proposal EVB-0268 / EXP-0482 (experiment_proposals.v1.json, priority medium,
dispatch_mode targeted_probe). claim_ids = [MECH-332] ONLY.

=== THE CLAIM AND ITS SUBSTRATE (re-verified fresh, not inherited) ===
MECH-332 asserts nociceptive attenuation on z_harm_s is produced by two
mechanistically INDEPENDENT pathways:
  Pathway 1 -- per-step efference-copy comparator (MECH-256/SD-029):
      residual = z_harm_s_observed - E2_harm_s(z_harm_s_{t-1}, a_actual)
  Pathway 2 -- commitment-gated PAG/RVM-analog descending precision suppression
      (SD-021, now subsumed by SD-032c): harm_s_gain = f(operating_mode, drive) < 1.0
      when committed.

Two modules OFTEN GUESSED as the substrate do NOT implement this claim -- read
and confirmed by inspection, not inherited from prior research:
  - `ree_core/comparator/tpj_comparator.py` (MECH-095) computes SELF/WORLD
    attribution on z_SELF via an efference-copy mismatch on the AGENCY axis
    ("did I cause this movement"). It has no z_harm_s input and no output that
    feeds a harm-attenuation gain. Wrong claim's substrate.
  - `ree_core/pag/freeze_gate.py` (MECH-279) computes a BEHAVIOURAL FREEZE
    commit/exit predicate from z_harm_a (the affective/C-fiber stream) plus a
    GABAergic exit threshold. It gates the ACTION SELECTOR into no-op moves; it
    does not attenuate z_harm_s and never reads z_harm_s at all. Wrong stream,
    wrong output.

The REAL substrate (confirmed by direct inspection of agent.py this session):
  - Pathway 1 = `ree_core/predictors/e2_harm_s.py` (`E2HarmSForward`, ARC-033).
    `agent.py::_update_scientist_attribution` (the "Self domain -- ARC-033
    E2HarmSForward" branch) computes exactly the claim's formula:
        pred_actual = self.e2_harm_s.forward(zh_prev, a_in)
        attribution = ||zh_obs - pred_actual||   (agent.py, "Self domain" block,
        immediately below the "self domain (ARC-033 E2HarmSForward): attribution
        = ||z_harm_s_obs - E2_harm_s(z_prev, a)||" comment)
    Constructed at `self.e2_harm_s = E2HarmSForward(harm_s_cfg)` when
    `config.latent.use_e2_harm_s_forward` is True (agent.py, agent __init__,
    "ARC-033: E2_harm_s sensory-discriminative harm forward model" block).
    NOTE (substrate-accuracy, load-bearing for the D3 design below): this
    attribution/residual is consumed ONLY by the MECH-276 ScientistAttributionBuffer
    (SD-003 causal_sig / MECH-275 sleep aggregation feedstock) -- it is NOT itself
    a multiplicative gain applied back onto z_harm_s in the sense() pipeline.
    Pathway 1 in the current substrate is a PREDICTION-ACCURACY signal, not a
    real-time attenuator.
  - Pathway 2 = `ree_core/cingulate/aic_analog.py` (`AICAnalog`, SD-032c). Its own
    docstring states it "subsumes SD-021 descending pain modulation". It computes
        harm_s_gain = 1.0 - base_attenuation * mode_weight * drive_protect
    (`AICAnalog.tick()`), and `agent.py`'s per-step harm-attenuation block (the
    "SD-021: descending pain modulation (commitment-gated sensory harm
    attenuation)" comment) DOES multiply z_harm by this gain in real time:
        if self.aic is not None:
            attn = float(self.aic.harm_s_gain)
            if attn < 1.0:
                new_latent.z_harm = new_latent.z_harm * attn
    gated on `harm_descending_mod_enabled` and `use_aic_analog`
    (`ree_core/utils/config.py`: `use_e2_harm_s_forward` line 273,
    `use_aic_analog` line 2851, both default False / backward-compatible-off).

=== claims.yaml / evidence state (re-checked this session) ===
MECH-332 still carries `status: candidate`, `v3_pending: true`,
`implementation_phase: v3`, decision `hold_pending_v3_substrate/applied` dated
2026-05-19T21:27:25Z. `claim_evidence.v1.json` still shows `genuine_exp_count: 0`,
`experimental_confidence: 0.0`, one literature-only entry (lit_conf 0.739,
evidence_quadrant plausible_unproven). `pending_review.md` carries no MECH-332 /
EVB-0268 / EXP-0482 entries. Nothing has changed since the task brief's
2026-05-19 dating -- the v3_pending gate PRE-DATES (or missed) the ARC-033/SD-032c
mapping confirmed above and is STALE: both real V3 substrates already exist and
are independently wireable via config flags. Flagged for governance to lift
`v3_pending`/re-route the `hold_pending_v3_substrate` decision as a companion
action to this queue entry -- not edited here.

=== GOV-REUSE-1 (existing-evidence / reanalysis-first check) ===
Decisive readouts: (a) `z_harm_s_ratio` (committed/uncommitted attenuation, the
Pathway-2 signature) and (b) `self_other_discrimination_ratio` (efference-copy
residual on agent_caused vs env_caused hazard transitions, the Pathway-1
signature), each measured with the OTHER pathway both present and absent.
`reanalysis_query.py query --readout z_harm_s_gain_attenuation_ratio --claim
MECH-332` and `--readout z_harm_s_attenuation --claim MECH-332` both scanned 0
flat manifests (the flat `evidence/experiments/*.json` glob the tool reads is
empty; recent runs live under `runs/<run_id>/manifest.json`, which the tool does
not scan -- treated as UNVERIFIABLE, not a match). A direct grep for any existing
script combining `use_aic_analog` and `use_e2_harm_s_forward` in the same driver
(`ree-v3/experiments/`) returns ZERO scripts -- every prior AIC experiment
(v3_exq_325c/d/e/f) runs with `use_e2_harm_s_forward` unset/False, and every prior
E2_harm_s experiment (v3_exq_264/329/353/431/433*/508/etc.) runs with
`use_aic_analog` unset/False. No prior run exercises both pathways in the same
cell, so the D1/D2/D3 dissociation this experiment tests is NOT recoverable
post-hoc from any recorded manifest on a compatible substrate. Proceeding to run.

=== SUBSTRATE READINESS (Step 2.5) ===
Both real modules exist, are independently wireable via the config flags cited
above, and require no new build -- this experiment is `complex (probe-gated)`,
specifically a `puzzle (known rules)`: the mechanisms are implemented and their
formulas are known; what is missing is the empirical fact of whether they behave
independently in situ. Re-derive brake (2.5b): zero prior `/failure-autopsy`
artifacts tag MECH-332 (fresh claim, first experiment), so the brake does not
apply.

=== DESIGN: 2x2 FACTORIAL, 4 NAMED ARMS x 3 SEEDS ===
Factor E (Pathway 1)   use_e2_harm_s_forward   ON / OFF
Factor A (Pathway 2)   use_aic_analog + harm_descending_mod_enabled   ON / OFF
(the two Pathway-2 flags move together -- "AIC OFF" here means the descending
attenuation path is entirely absent, not merely routed through the legacy
raw-beta_gate branch, so the isolation is clean)

  ARM_BOTH       E=ON  A=ON   -- both pathways co-active (the biological default)
  ARM_E2_ONLY    E=ON  A=OFF  -- Pathway 1 isolated (tests D2)
  ARM_AIC_ONLY   E=OFF A=ON   -- Pathway 2 isolated (tests D1)
  ARM_NEITHER    E=OFF A=OFF  -- control / non-degeneracy floor

Arena: SD-022 body-damage substrate (`limb_damage_enabled=True`, the home turf
where z_harm_s / z_harm_a are causally independent per EXQ-241b) + SD-029
balanced-event curriculum with EXQ-479's calibrated params (interval=10, prob=1.0,
adjacent_only=True, hazard_harm=0.02, size=8, num_hazards=2, num_resources=3,
proximity_harm_scale=0.1) so both agent_caused_hazard and env_caused_hazard
transitions occur reliably (per the V3-EXQ-433f/479 calibration history). E3
commitment substrate on (`HeartbeatConfig(beta_gate_bistable=True)`), needed for
Pathway 2's commitment gate. Identical arena/curriculum/schedule across all four
arms -- only the two pathway-flag pairs vary.

=== DV-SYMMETRY / D3 REINTERPRETATION (mandatory declaration) ===
The claim's own prose says "their attenuation effects are additive" (D3). Taken
literally this would mean measuring a single combined GAIN and checking it sums.
That is NOT what this substrate can measure, because -- per the substrate note
above -- Pathway 1 is not a gain on z_harm_s at all; it is a prediction-residual
signal consumed by a different subsystem (MECH-276/SD-003). A literal "summed
attenuation" DV does not exist for Pathway 1, so no arm's DV is invariant under a
symmetry that would make a fabricated combined-gain number meaningless (there is
no such number to begin with) -- the hazard this declaration guards against
(DV-SYMMETRY INVARIANCE, `_lib` corpus lint) is instead a NAMING hazard: reporting
D3 as if it were a measured super-additive gain would misrepresent an unmeasured
quantity as data. D3 is therefore operationalised as INDEPENDENCE / NON-INTERFERENCE:
does each pathway's own signature metric survive, undegraded, when the other
pathway is simultaneously active? (ARM_BOTH's z_harm_s_ratio vs ARM_AIC_ONLY's;
ARM_BOTH's self_other_discrimination_ratio vs ARM_E2_ONLY's.) This is the honest
reading of "their effects are additive" available from what the substrate
actually implements: not "the numbers sum", but "neither pathway crowds out or
overwrites the other's effect when both are wired in". Recorded verbatim in
custom_information.d3_operationalization_note. Every arm's own DVs ARE meaningful
under their own symmetry: z_harm_s_ratio is a ratio of two encoder-output norms
(not an argmax/rank statistic invariant under the AIC gain, which multiplies
z_harm directly -- see the agent.py block quoted above), and
self_other_discrimination_ratio is a ratio of two L2-residual means (not a
set-aggregate the a_actual/a_cf action choice would leave invariant, since
E2_harm_s is trained with the SD-013 interventional/contrastive loss specifically
to make its predictions ACTION-SENSITIVE).

=== PRE-REGISTERED CRITERIA (constants below; nothing derived post-hoc) ===
Per seed:
  D1 (Pathway 2 independent of Pathway 1 -- claim's D1 prediction)
     |z_harm_s_ratio[AIC_ONLY] - z_harm_s_ratio[BOTH]| <= D1_NEAR_EQUAL_TOL   AND
     z_harm_s_ratio[AIC_ONLY] <= z_harm_s_ratio[NEITHER] - D1_ATTENUATION_MARGIN AND
     z_harm_s_ratio[BOTH]     <= z_harm_s_ratio[NEITHER] - D1_ATTENUATION_MARGIN
  D2 (Pathway 1 independent of Pathway 2 -- claim's D2 prediction)
     |self_other[E2_ONLY] - self_other[BOTH]| <= D2_NEAR_EQUAL_TOL   AND
     self_other[E2_ONLY] >= D2_DISCRIM_FLOOR   AND   self_other[BOTH] >= D2_DISCRIM_FLOOR
  D3 (independence when combined -- see reinterpretation above)
     z_harm_s_ratio[BOTH] <= z_harm_s_ratio[AIC_ONLY] + D3_TOL   AND
     self_other[BOTH] >= self_other[E2_ONLY] - D3_TOL

  seed_pass = D1 and D2 and D3, GATED on non-degeneracy (below).
  PASS = seed_pass in >= 2/3 seeds  ->  evidence_direction "supports"
  FAIL with D1 and D2 BOTH failing in every seed -> "weakens" (the pathways
     are NOT dissociable on this substrate -- a decision-flipping negative)
  FAIL otherwise (partial) -> "mixed"
  FAIL from non-degeneracy (below floor) -> "non_contributory" (instrument
     issue: insufficient commitment ticks or hazard-event trials, NOT a verdict
     on the claim)

=== NON-DEGENERACY (else non_contributory, never a substrate verdict) ===
  n_committed_steps >= N_COMMITTED_FLOOR in ARM_AIC_ONLY and ARM_BOTH (Pathway 2
     needs the commitment substrate to have actually fired)
  n_agent_caused_trials, n_env_caused_trials >= N_EVENT_TRIALS_FLOOR in
     ARM_E2_ONLY and ARM_BOTH (Pathway 1's self/other discrimination needs both
     event types to have actually occurred)
Recorded as top-level `non_degenerate` + `degeneracy_reason` (Experimental
Recording Standard "Non-degeneracy scoring net -- applies to evidence too").

=== SAMPLE-SIZE INTEGRITY ===
`agent.e3._committed_trajectory` is polled once per genuine `agent.select_action`
call in the eval loop (never a `last_*` diagnostic re-read without an intervening
selection), so the committed/uncommitted tally is not pseudo-replicated. The
efference-copy residual pairing follows the V3-EXQ-433f convention exactly: the
PRE-transition (z_harm_s, action) pair is matched against the resulting
POST-transition z_harm_s, tagged by that step's own `info["transition_type"]" --
never a stale tag from a different transition.

=== ETHICS PREFLIGHT (descriptive, non-enforced -- see /queue-experiment Step 2.6) ===
involves_negative_valence: true (z_harm_s / z_harm_a drive, as in every V3 harm-
  stream experiment) -- involves_suffering_like_state: false (MECH-219 accumulator
  not engaged) -- involves_self_model: false -- involves_inescapability_or_
  helplessness: false (escapability unconstrained) -- involves_offline_replay_
  over_harm: false (no sleep phases used) -- involves_social_mind_or_language:
  false -- involves_human_data_or_clinical_context: false -- decision: allow.
Pre-ethical instrumentation per SENT-0; no V4-boundary concern.

=== SCOPE ===
Tests whether the two CURRENTLY-IMPLEMENTED candidate substrates for MECH-332
behave independently. A PASS does not establish that either module is the
UNIQUE correct implementation of its pathway, only that the two, as built, do not
confound each other. `custom_information.scope_note`.

No sleep phases are used (SLEEP DRIVER: not applicable). No head is trained on a
downstream loss from z_world/z_harm beyond the ARC-033 phased P0/P1 protocol
E2_harm_s already requires (P0 encoder warmup with no e2_harm_s training, P1
frozen-target MSE + SD-013 interventional loss) -- AICAnalog is pure arithmetic,
no gradient flow, so no phased-training discipline applies to it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig, HeartbeatConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from experiments._lib.baselines import (  # noqa: E402
    exq878_mech332_efference_aic_baseline as BASE,
)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402

# ------------------------------------------------------------------ #
# Identity                                                            #
# ------------------------------------------------------------------ #
EXPERIMENT_TYPE = "v3_exq_878_mech332_efference_aic_dissociation"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS: List[str] = ["MECH-332"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
BACKLOG_ID = "EVB-0268"

OUT_DIR = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"

# ------------------------------------------------------------------ #
# Pre-registered constants (NOT derived from the run's own statistics) #
# ------------------------------------------------------------------ #
SEEDS = [42, 7, 13]
ARMS: Dict[str, Dict[str, bool]] = {
    "ARM_BOTH": {
        "use_e2_harm_s_forward": True,
        "use_aic_analog": True,
        "harm_descending_mod_enabled": True,
    },
    "ARM_E2_ONLY": {
        "use_e2_harm_s_forward": True,
        "use_aic_analog": False,
        "harm_descending_mod_enabled": False,
    },
    "ARM_AIC_ONLY": {
        "use_e2_harm_s_forward": False,
        "use_aic_analog": True,
        "harm_descending_mod_enabled": True,
    },
    "ARM_NEITHER": {
        "use_e2_harm_s_forward": False,
        "use_aic_analog": False,
        "harm_descending_mod_enabled": False,
    },
}
EVENT_TYPES = ["agent_caused_hazard", "env_caused_hazard"]

D1_NEAR_EQUAL_TOL = 0.2
D1_ATTENUATION_MARGIN = 0.03
D2_NEAR_EQUAL_TOL = 0.6
D2_DISCRIM_FLOOR = 1.15
D3_TOL = 0.15
PASS_MIN_SEEDS = 2
PASS_FRACTION_REQUIRED = 2.0 / 3.0
N_COMMITTED_FLOOR = 8

# Dry-run overrides (smoke only -- never used for the pre-registered thresholds).
DRY_SEEDS = [SEEDS[0]]
DRY_P0_EPS = 2
DRY_P1_EPS = 2
DRY_EVAL_MAX_EPS = 3
DRY_STEPS_PER_EP = 25
DRY_TARGET_TRIALS = 2


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #
def _onehot(idx: int, n: int) -> torch.Tensor:
    v = torch.zeros(1, n)
    v[0, idx] = 1.0
    return v


def _find_adjacent_hazard_action(env: CausalGridWorldV2) -> Optional[int]:
    """Scripted agent_caused_hazard elicitation (V3-EXQ-433f/479 pattern): return
    a move action that steps the agent onto an adjacent hazard cell, or None."""
    ax, ay = env.agent_x, env.agent_y
    hazard_set = {(h[0], h[1]) for h in env.hazards}
    for action_idx, delta in env.ACTIONS.items():
        if action_idx == 4:  # NOOP
            continue
        dx, dy = delta
        if getattr(env, "toroidal", False):
            nx, ny = (ax + dx) % env.size, (ay + dy) % env.size
        else:
            nx, ny = ax + dx, ay + dy
            if not (0 <= nx < env.size and 0 <= ny < env.size):
                continue
        if (nx, ny) in hazard_set:
            return action_idx
    return None


def make_config(env: CausalGridWorldV2, flags: Dict[str, bool]) -> REEConfig:
    hb = HeartbeatConfig(beta_gate_bistable=True)
    kwargs = BASE.agent_kwargs(flags)
    return REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        heartbeat=hb,
        **kwargs,
    )


# ------------------------------------------------------------------ #
# Train phase -- P0 (encoder warmup) + P1 (E2_harm_s phased training)  #
# ------------------------------------------------------------------ #
def train_arm(
    agent: REEAgent,
    env: CausalGridWorldV2,
    gen: torch.Generator,
    p0_eps: int,
    p1_eps: int,
    steps_per_ep: int,
    use_e2: bool,
) -> None:
    action_dim = env.action_dim
    optimizer = torch.optim.Adam(agent.parameters(), lr=BASE.LR)
    harm_opt = None
    if use_e2 and agent.e2_harm_s is not None:
        harm_opt = torch.optim.Adam(
            agent.e2_harm_s.parameters(), lr=BASE.HARM_FWD_LR
        )

    agent.train()
    total_eps = p0_eps + p1_eps

    for ep in range(total_eps):
        train_e2_this_ep = use_e2 and harm_opt is not None and ep >= p0_eps
        _, obs_dict = env.reset()
        agent.reset()
        last_z_harm: Optional[torch.Tensor] = None
        last_action: Optional[torch.Tensor] = None

        for _ in range(steps_per_ep):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm = obs_dict.get("harm_obs")
            obs_harm_a = obs_dict.get("harm_obs_a")
            obs_harm_history = obs_dict.get("harm_history")

            latent = agent.sense(
                obs_body, obs_world, obs_harm=obs_harm,
                obs_harm_a=obs_harm_a, obs_harm_history=obs_harm_history,
            )
            agent.clock.advance()

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss

            # P1: E2_harm_s phased training on FROZEN (.detach()ed) z_harm_s
            # targets -- SD-013 interventional/contrastive loss applied on a
            # fraction of steps to force action-sensitive predictions (required
            # for the D2 self/other discrimination to be non-trivial).
            if (
                train_e2_this_ep
                and last_z_harm is not None
                and last_action is not None
                and latent.z_harm is not None
            ):
                z_pred = agent.e2_harm_s.forward(last_z_harm.detach(), last_action.detach())
                target = latent.z_harm.detach()
                h_loss = agent.e2_harm_s.compute_loss(z_pred, target)
                h_total = h_loss
                if torch.rand(1, generator=gen).item() < BASE.INTERVENTIONAL_FRACTION:
                    actual_idx = last_action.argmax(dim=-1)
                    cf_idx = (
                        actual_idx + 1
                        + torch.randint(0, action_dim - 1, actual_idx.shape, generator=gen)
                    ) % action_dim
                    a_cf = F.one_hot(cf_idx, num_classes=action_dim).float()
                    h_int = agent.e2_harm_s.compute_interventional_loss(
                        last_z_harm.detach(), last_action.detach(), a_cf
                    )
                    h_total = h_total + h_int
                harm_opt.zero_grad()
                h_total.backward()
                harm_opt.step()

            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            action_idx = int(torch.randint(0, action_dim, (1,), generator=gen).item())
            action_oh = _onehot(action_idx, action_dim)
            agent._last_action = action_oh
            if latent.z_harm is not None:
                last_z_harm = latent.z_harm.detach()
                last_action = action_oh.detach()
            _, _, done, _, obs_dict = env.step(action_oh)
            if done:
                break

        if (ep + 1) % 40 == 0 or ep == total_eps - 1:
            print(f"  [train] ep {ep + 1}/{total_eps}", flush=True)


# ------------------------------------------------------------------ #
# Eval phase -- commitment-gated attenuation + efference-copy residual #
# ------------------------------------------------------------------ #
def eval_arm(
    agent: REEAgent,
    env: CausalGridWorldV2,
    gen: torch.Generator,
    use_e2: bool,
    eval_max_eps: int,
    steps_per_ep: int,
    target_trials: int,
) -> Dict[str, Any]:
    action_dim = env.action_dim
    agent.eval()

    z_s_committed: List[float] = []
    z_s_uncommitted: List[float] = []
    aic_saliences: List[float] = []
    n_committed = 0
    n_uncommitted = 0

    trials_done = {et: 0 for et in EVENT_TYPES}
    residuals: Dict[str, List[float]] = {et: [] for et in EVENT_TYPES}
    # HOLD-WEIGHTED-E3-READOUT triage (validate_experiments.py advisory finding):
    # agent.select_action() returns the HELD action (and generate_trajectories the
    # cached candidates) on a non-e3_tick, so a per-step accumulation can weight a
    # single commitment decision by its hold duration. Neither DV here is the
    # "commitment quality" statistic the finding is about -- z_harm_s_ratio is a
    # continuous magnitude ratio and self_other_discrimination_ratio is gated on
    # actual environment TRANSITIONS (info["transition_type"], which fires on
    # physical events regardless of E3 cadence), not on repeated reads of one
    # cached decision -- but the counters below make the hold/fresh split
    # auditable rather than asserted.
    n_fresh_select_ticks = 0
    n_latched_ticks = 0

    for _ in range(eval_max_eps):
        if all(trials_done[et] >= target_trials for et in EVENT_TYPES):
            break
        _, obs_dict = env.reset()
        agent.reset()
        prev_z_harm: Optional[torch.Tensor] = None
        prev_action: Optional[torch.Tensor] = None
        prev_etype = "none"

        for _ in range(steps_per_ep):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm = obs_dict.get("harm_obs")
            obs_harm_a = obs_dict.get("harm_obs_a")
            obs_harm_history = obs_dict.get("harm_history")

            with torch.no_grad():
                latent = agent.sense(
                    obs_body, obs_world, obs_harm=obs_harm,
                    obs_harm_a=obs_harm_a, obs_harm_history=obs_harm_history,
                )
                ticks = agent.clock.advance()
                wdim = latent.z_world.shape[-1]
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick", False)
                    else torch.zeros(1, wdim)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)

                scripted_idx = None
                if trials_done["agent_caused_hazard"] < target_trials:
                    scripted_idx = _find_adjacent_hazard_action(env)
                if scripted_idx is not None:
                    action = _onehot(scripted_idx, action_dim)
                    is_fresh_select = True  # scripted, not a cached E3 hold
                else:
                    action = agent.select_action(candidates, ticks)
                    is_fresh_select = bool(ticks.get("e3_tick", False))
            if is_fresh_select:
                n_fresh_select_ticks += 1
            else:
                n_latched_ticks += 1

            if agent.aic is not None:
                aic_saliences.append(float(agent.aic.aic_salience))

            is_committed = agent.e3._committed_trajectory is not None
            z_harm_curr = latent.z_harm.detach() if latent.z_harm is not None else None
            z_s_post = float(z_harm_curr.norm().item()) if z_harm_curr is not None else 0.0
            if is_committed:
                z_s_committed.append(z_s_post)
                n_committed += 1
            else:
                z_s_uncommitted.append(z_s_post)
                n_uncommitted += 1

            # Pathway-1 residual: pair the PRE-transition (z_harm_s, action) with
            # the resulting POST-transition z_harm_s, tagged by the transition
            # type that JUST produced it (prev_etype, set at the end of the prior
            # iteration -- never a stale re-read of a latched diagnostic).
            if (
                use_e2 and agent.e2_harm_s is not None
                and prev_z_harm is not None and prev_action is not None
                and z_harm_curr is not None
                and prev_etype in trials_done
                and trials_done[prev_etype] < target_trials
            ):
                with torch.no_grad():
                    z_pred = agent.e2_harm_s.forward(prev_z_harm, prev_action)
                residual = float((z_harm_curr - z_pred).norm(dim=-1).mean().item())
                residuals[prev_etype].append(residual)
                trials_done[prev_etype] += 1

            _, _, done, info, obs_dict = env.step(action)
            etype = info.get("transition_type", "none")
            prev_z_harm = z_harm_curr
            prev_action = action.detach()
            prev_etype = etype
            if done:
                break

    def _mean(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    residual_agent = _mean(residuals["agent_caused_hazard"])
    residual_env = _mean(residuals["env_caused_hazard"])
    self_other_ratio = (
        residual_env / max(1e-6, residual_agent) if use_e2 else None
    )
    z_ratio = (
        _mean(z_s_committed) / max(1e-8, _mean(z_s_uncommitted))
        if z_s_committed and z_s_uncommitted else 1.0
    )

    n_ticks_total = n_fresh_select_ticks + n_latched_ticks
    fresh_select_yield = (
        float(n_fresh_select_ticks) / n_ticks_total if n_ticks_total > 0 else 0.0
    )

    return {
        "z_harm_s_ratio": z_ratio,
        "z_harm_s_mean_committed": _mean(z_s_committed),
        "z_harm_s_mean_uncommitted": _mean(z_s_uncommitted),
        "n_committed_steps": n_committed,
        "n_uncommitted_steps": n_uncommitted,
        "n_fresh_select_ticks": n_fresh_select_ticks,
        "n_latched_ticks": n_latched_ticks,
        "fresh_select_yield": fresh_select_yield,
        "aic_salience_mean": _mean(aic_saliences),
        "n_agent_caused_trials": trials_done["agent_caused_hazard"],
        "n_env_caused_trials": trials_done["env_caused_hazard"],
        "residual_agent_caused_mean": residual_agent if use_e2 else None,
        "residual_env_caused_mean": residual_env if use_e2 else None,
        "self_other_discrimination_ratio": self_other_ratio,
    }


# ------------------------------------------------------------------ #
# Cell driver + config-slice declaration (arm_fingerprint)             #
# ------------------------------------------------------------------ #
def _arm_config_slice(arm_id: str, seed: int, schedule: Dict[str, int]) -> Dict[str, Any]:
    if arm_id == "ARM_NEITHER":
        slice_ = BASE.off_path_config_slice(seed)
        slice_["schedule"] = schedule
        return slice_
    flags = ARMS[arm_id]
    return {
        "env_kwargs": BASE.env_kwargs(seed),
        "agent_kwargs": BASE.agent_kwargs(flags),
        "schedule": schedule,
        "lr": BASE.LR,
        "harm_fwd_lr": BASE.HARM_FWD_LR,
        "interventional_fraction": BASE.INTERVENTIONAL_FRACTION,
        "interventional_margin": BASE.INTERVENTIONAL_MARGIN,
    }


def _run_cell(
    arm_id: str, seed: int, p0_eps: int, p1_eps: int,
    eval_max_eps: int, steps_per_ep: int, target_trials: int,
    zg_accumulator: Optional[ZGoalStreamAccumulator] = None,
) -> Dict[str, Any]:
    flags = ARMS[arm_id]
    env = CausalGridWorldV2(**BASE.env_kwargs(seed))
    cfg = make_config(env, flags)
    agent = REEAgent(cfg)
    gen = torch.Generator()
    gen.manual_seed(seed)

    train_arm(agent, env, gen, p0_eps, p1_eps, steps_per_ep, flags["use_e2_harm_s_forward"])
    result = eval_arm(
        agent, env, gen, flags["use_e2_harm_s_forward"],
        eval_max_eps, steps_per_ep, target_trials,
    )
    row: Dict[str, Any] = {"arm_id": arm_id, "seed": seed, **flags, **result}
    if zg_accumulator is not None:
        # AFTER stepping -- observe() reads the counters at call time. Called
        # here (not by the caller holding a reference) so the agent -- nets,
        # optimiser state, e2_harm_s -- can be garbage-collected once this
        # function returns, rather than kept alive for all 12 cells.
        zg_accumulator.observe(agent)
    return row


# ------------------------------------------------------------------ #
# Analysis                                                            #
# ------------------------------------------------------------------ #
def _analyse(rows: List[Dict[str, Any]], target_trials: int) -> Dict[str, Any]:
    by_key = {(r["arm_id"], r["seed"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows})
    n_event_floor = max(1, target_trials // 2)

    per_seed = []
    for s in seeds:
        both = by_key[("ARM_BOTH", s)]
        e2_only = by_key[("ARM_E2_ONLY", s)]
        aic_only = by_key[("ARM_AIC_ONLY", s)]
        neither = by_key[("ARM_NEITHER", s)]

        ratio_both = both["z_harm_s_ratio"]
        ratio_aic_only = aic_only["z_harm_s_ratio"]
        ratio_neither = neither["z_harm_s_ratio"]

        d1_near_equal = abs(ratio_aic_only - ratio_both) <= D1_NEAR_EQUAL_TOL
        d1_attenuation_present = (
            ratio_aic_only <= ratio_neither - D1_ATTENUATION_MARGIN
            and ratio_both <= ratio_neither - D1_ATTENUATION_MARGIN
        )
        d1_pass = bool(d1_near_equal and d1_attenuation_present)

        so_both = both["self_other_discrimination_ratio"]
        so_e2_only = e2_only["self_other_discrimination_ratio"]
        d2_near_equal = bool(
            so_both is not None and so_e2_only is not None
            and abs(so_e2_only - so_both) <= D2_NEAR_EQUAL_TOL
        )
        d2_discrim_present = bool(
            so_both is not None and so_e2_only is not None
            and so_both >= D2_DISCRIM_FLOOR and so_e2_only >= D2_DISCRIM_FLOOR
        )
        d2_pass = bool(d2_near_equal and d2_discrim_present)

        d3_pathway2_preserved = ratio_both <= ratio_aic_only + D3_TOL
        d3_pathway1_preserved = bool(
            so_both is not None and so_e2_only is not None
            and so_both >= so_e2_only - D3_TOL
        )
        d3_pass = bool(d3_pathway2_preserved and d3_pathway1_preserved)

        n_committed_ok = (
            aic_only["n_committed_steps"] >= N_COMMITTED_FLOOR
            and both["n_committed_steps"] >= N_COMMITTED_FLOOR
        )
        n_events_ok = (
            e2_only["n_agent_caused_trials"] >= n_event_floor
            and e2_only["n_env_caused_trials"] >= n_event_floor
            and both["n_agent_caused_trials"] >= n_event_floor
            and both["n_env_caused_trials"] >= n_event_floor
        )
        seed_non_degenerate = bool(n_committed_ok and n_events_ok)
        seed_pass = bool(seed_non_degenerate and d1_pass and d2_pass and d3_pass)

        per_seed.append({
            "seed": s,
            "d1_pass": d1_pass, "d2_pass": d2_pass, "d3_pass": d3_pass,
            "seed_non_degenerate": seed_non_degenerate, "seed_pass": seed_pass,
            "ratio_both": ratio_both, "ratio_aic_only": ratio_aic_only,
            "ratio_neither": ratio_neither,
            "self_other_both": so_both, "self_other_e2_only": so_e2_only,
            "n_committed_aic_only": aic_only["n_committed_steps"],
            "n_committed_both": both["n_committed_steps"],
            "n_agent_caused_e2_only": e2_only["n_agent_caused_trials"],
            "n_env_caused_e2_only": e2_only["n_env_caused_trials"],
            "n_agent_caused_both": both["n_agent_caused_trials"],
            "n_env_caused_both": both["n_env_caused_trials"],
        })

    n_seeds = len(per_seed)
    seeds_passing = sum(1 for r in per_seed if r["seed_pass"])
    seeds_non_degenerate = sum(1 for r in per_seed if r["seed_non_degenerate"])
    d1_fail_seeds = sum(1 for r in per_seed if not r["d1_pass"])
    d2_fail_seeds = sum(1 for r in per_seed if not r["d2_pass"])
    d3_fail_seeds = sum(1 for r in per_seed if not r["d3_pass"])

    overall_non_degenerate = seeds_non_degenerate >= math.ceil(
        n_seeds * PASS_FRACTION_REQUIRED
    )
    overall_pass = seeds_passing >= PASS_MIN_SEEDS

    return {
        "per_seed": per_seed,
        "n_seeds": n_seeds,
        "seeds_passing": seeds_passing,
        "pass_min_seeds": PASS_MIN_SEEDS,
        "seeds_non_degenerate": seeds_non_degenerate,
        "overall_non_degenerate": overall_non_degenerate,
        "overall_pass": overall_pass,
        "d1_fail_seeds": d1_fail_seeds,
        "d2_fail_seeds": d2_fail_seeds,
        "d3_fail_seeds": d3_fail_seeds,
        "n_event_floor": n_event_floor,
    }


# ------------------------------------------------------------------ #
# Driver                                                              #
# ------------------------------------------------------------------ #
def run_experiment(dry_run: bool) -> Dict[str, Any]:
    seeds = DRY_SEEDS if dry_run else SEEDS
    p0_eps = DRY_P0_EPS if dry_run else BASE.P0_EPS
    p1_eps = DRY_P1_EPS if dry_run else BASE.P1_EPS
    eval_max_eps = DRY_EVAL_MAX_EPS if dry_run else BASE.EVAL_MAX_EPS
    steps_per_ep = DRY_STEPS_PER_EP if dry_run else BASE.STEPS_PER_EP
    target_trials = DRY_TARGET_TRIALS if dry_run else BASE.TARGET_TRIALS_PER_TYPE

    schedule = {
        "steps_per_ep": steps_per_ep, "p0_eps": p0_eps, "p1_eps": p1_eps,
        "eval_max_eps": eval_max_eps, "target_trials_per_type": target_trials,
    }

    rows: List[Dict[str, Any]] = []
    zg_accumulator = ZGoalStreamAccumulator()
    for arm_id in ARMS:
        for seed in seeds:
            print(f"Seed {seed} Condition {arm_id}", flush=True)
            slice_ = _arm_config_slice(arm_id, seed, schedule)
            with arm_cell(
                seed,
                config_slice=slice_,
                script_path=Path(__file__),
                config_slice_declared=True,
                # ARM_NEITHER is this lineage's OFF/baseline arm -- mint it
                # cross-driver reusable. The three treatment arms are specific
                # to this dissociation design; left at the driver-inclusive
                # default.
                include_driver_script_in_hash=(arm_id != "ARM_NEITHER"),
            ) as cell:
                row = _run_cell(
                    arm_id, seed, p0_eps, p1_eps, eval_max_eps,
                    steps_per_ep, target_trials, zg_accumulator=zg_accumulator,
                )
                cell.stamp(row)
            rows.append(row)
            engaged = row["n_committed_steps"] + row["n_uncommitted_steps"] > 0
            print(f"verdict: {'PASS' if engaged else 'FAIL'}", flush=True)

    analysis = _analyse(rows, target_trials)

    if not analysis["overall_non_degenerate"]:
        outcome = "FAIL"
        evidence_direction = "non_contributory"
        label = "substrate_not_ready_requeue"
    elif analysis["overall_pass"]:
        outcome = "PASS"
        evidence_direction = "supports"
        label = "mech332_pathways_dissociable"
    elif (
        analysis["d1_fail_seeds"] == analysis["n_seeds"]
        and analysis["d2_fail_seeds"] == analysis["n_seeds"]
    ):
        outcome = "FAIL"
        evidence_direction = "weakens"
        label = "mech332_pathways_not_dissociable"
    else:
        outcome = "FAIL"
        evidence_direction = "mixed"
        label = "mech332_partial_dissociation"

    manifest: Dict[str, Any] = {
        "schema_version": "v1",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "backlog_id": BACKLOG_ID,
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "non_degenerate": analysis["overall_non_degenerate"],
        "arm_results": rows,
        "analysis": analysis,
        "interpretation": {"label": label},
        "pre_registered_thresholds": {
            "D1_NEAR_EQUAL_TOL": D1_NEAR_EQUAL_TOL,
            "D1_ATTENUATION_MARGIN": D1_ATTENUATION_MARGIN,
            "D2_NEAR_EQUAL_TOL": D2_NEAR_EQUAL_TOL,
            "D2_DISCRIM_FLOOR": D2_DISCRIM_FLOOR,
            "D3_TOL": D3_TOL,
            "PASS_MIN_SEEDS": PASS_MIN_SEEDS,
            "N_COMMITTED_FLOOR": N_COMMITTED_FLOOR,
            "n_event_floor": analysis["n_event_floor"],
        },
        "custom_information": {
            "d3_operationalization_note": (
                "Pathway 1 (E2_harm_s) is a prediction-residual signal consumed "
                "by the MECH-276 ScientistAttributionBuffer, not a real-time "
                "multiplicative gain on z_harm_s -- unlike Pathway 2 (AICAnalog "
                "harm_s_gain), which does multiply z_harm in agent.py's sense() "
                "pipeline. So D3 ('attenuation effects are additive') is NOT "
                "tested as a summed gain -- no such combined number exists in "
                "this substrate. It is tested as independence/non-interference: "
                "does each pathway's own signature metric survive undegraded "
                "when the other pathway is co-active."
            ),
            "scope_note": (
                "Tests whether the two currently-implemented MECH-332 candidate "
                "modules (E2HarmSForward / ARC-033, AICAnalog / SD-032c) behave "
                "independently. Does not establish either module as the unique "
                "correct implementation of its pathway."
            ),
            "v3_pending_gate_stale_note": (
                "claims.yaml MECH-332 still carries v3_pending=true / "
                "hold_pending_v3_substrate dated 2026-05-19, predating (or "
                "missing) the ARC-033 (e2_harm_s.py) + SD-032c (aic_analog.py) "
                "substrate mapping confirmed by this experiment's own code "
                "review. Flagged for governance to lift as a companion action."
            ),
            "ethics_preflight": {
                "involves_negative_valence": True,
                "involves_suffering_like_state": False,
                "involves_self_model": False,
                "involves_inescapability_or_helplessness": False,
                "involves_offline_replay_over_harm": False,
                "involves_social_mind_or_language": False,
                "involves_human_data_or_clinical_context": False,
                "decision": "allow",
            },
        },
        # Consumed and popped in __main__ before writing -- not a manifest field.
        "_zg_stats_tmp": zg_accumulator.stats(),
    }
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    manifest = run_experiment(dry_run=args.dry_run)
    zg_stats = manifest.pop("_zg_stats_tmp", None)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    manifest["timestamp_utc"] = ts
    manifest["dry_run"] = bool(args.dry_run)

    seeds_used = DRY_SEEDS if args.dry_run else SEEDS
    full_config = {
        "arms": ARMS,
        "schedule": {
            "steps_per_ep": DRY_STEPS_PER_EP if args.dry_run else BASE.STEPS_PER_EP,
            "p0_eps": DRY_P0_EPS if args.dry_run else BASE.P0_EPS,
            "p1_eps": DRY_P1_EPS if args.dry_run else BASE.P1_EPS,
            "eval_max_eps": DRY_EVAL_MAX_EPS if args.dry_run else BASE.EVAL_MAX_EPS,
            "target_trials_per_type": (
                DRY_TARGET_TRIALS if args.dry_run else BASE.TARGET_TRIALS_PER_TYPE
            ),
        },
        "arena": BASE.ARENA,
        "agent_common_kwargs": BASE.COMMON_AGENT_KWARGS,
        "lr": BASE.LR, "harm_fwd_lr": BASE.HARM_FWD_LR,
        "interventional_fraction": BASE.INTERVENTIONAL_FRACTION,
        "interventional_margin": BASE.INTERVENTIONAL_MARGIN,
        "pre_registered_thresholds": manifest["pre_registered_thresholds"],
    }

    # AFTER arm_results is assembled, so substrate_hash HOISTS from the
    # per-cell fingerprints rather than being recomputed (and mismatching).
    stamp_recording_core(
        manifest,
        config=full_config,
        seeds=seeds_used,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=zg_stats,
    )

    out_path = write_flat_manifest(
        manifest, OUT_DIR, dry_run=args.dry_run, stamp=False,
    )

    print(json.dumps({
        "run_id": manifest["run_id"],
        "outcome": manifest["outcome"],
        "evidence_direction": manifest["evidence_direction"],
        "non_degenerate": manifest["non_degenerate"],
        "label": manifest["interpretation"]["label"],
        "seeds_passing": manifest["analysis"]["seeds_passing"],
        "n_seeds": manifest["analysis"]["n_seeds"],
        "manifest": str(out_path),
    }, indent=2), flush=True)

    _outcome_raw = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
