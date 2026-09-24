#!/opt/local/bin/python3
"""
V3-EXQ-1071 -- INV-063 four-arm intake ladder: the registered falsifier.

SLEEP DRIVER: manual-cycle-loop (run_sleep_cycle() called once per cycle in a
              dedicated N_CYCLES wake-sleep-test loop)

WHAT THIS IS
-------------
INV-063 `minimum_entropy_intake_sleep_dependency` asserts that a minimum level of
environmental entropy intake during waking is required to sustain offline sleep
function. This is that claim's registered falsifier -- P1-P5, C1, C2 and F1/F3 as
written in its `what_would_answer` -- and it is the claim's FIRST simulation evidence
of any kind (claim_evidence.v1.json carries five literature entries and zero runs).

It is queued only after a four-turn sequence of measured refusals, each of which
removed a specific way this run could have been vacuous. That history is load-bearing
for reading the design, so it is stated rather than assumed:

  V3-EXQ-1060  the E2.world_forward sleep trainer moves at all (leg B's DV was
               structurally 0.0 before it).
  V3-EXQ-1063  WHICH frozen-battery readout moves in the asserted direction, and at
               what base convergence. Found the 701b MSE readout going NEGATIVE on a
               converged base on 3/3 seeds, and the SD-056 InfoNCE readout pinned at
               its chance value ln(K) at the shipped tau.
  V3-EXQ-1069  whether P1's intake ladder grades MEL at all under V3-EXQ-798a's exact
               P0 and configuration. PASS, 2/3 seeds -- which is what licenses this run.
  tau ladder   on 798a's converged base, headroom and the asserted across-sleep
  (2026-09-20) direction TRADE OFF in tau with no overlap: readability needs
               tau <= 0.01, a positive delta needs tau >= 0.03. Staged at
               REE_assembly/evidence/planning/inv063_legb_tau_bind_20260920.md.

THE USER-RATIFIED AMENDMENTS, both pending governance application
------------------------------------------------------------------
1. LEG B'S READOUT. claims.yaml still says leg B is measured with "the V3-EXQ-701b/701c
   frozen-probe instrument", i.e. per-element MSE reconstruction error. This run uses
   the SD-056 InfoNCE frozen-battery objective instead, at a PINNED
   tau = 1e-3. USER-RATIFIED 2026-09-19T21:42:30Z; PENDING governance application of
   GFLAG-0364. Same posture V3-EXQ-1039a used for its A2 narrowing. Basis: on a
   converged 798a-config base the MSE readout degrades across sleep (V3-EXQ-1063,
   9/9 cells negative), while at tau = 1e-3 InfoNCE carries 18.54% headroom below
   ln(K) against the 0.62% the shipped tau = 0.1 gives.
2. THE REFUSAL ROUTE. USER-RATIFIED 2026-09-20 (option (c)). Because that same tau
   ladder shows the across-sleep delta NEGATIVE at every readable tau, this run
   pre-registers `leg_b_dv_sign_inverted`: if leg B's delta is negative at the pinned
   tau across arms, the run does NOT read C1 leg B and emits no F1/F2/F3 verdict. It
   still reads and reports everything else. The full tau ladder is recorded in EVERY
   cell -- re-evaluating a fixed distance matrix, microseconds -- so leg B can be
   re-read at another tau without re-running.

   That refusal route is why this design can be queued at all. Without it a negative
   leg B would route to F1, which INV-063 pre-registers as "the invariant is genuinely
   falsified rather than substrate-confounded" -- and on the tau evidence that reading
   would be wrong. A false F1 against a claim with zero prior simulation evidence is
   the expensive failure here, not the compute.

ARMS -- FOUR INTAKE + THE ONE P1 CONTROL
-----------------------------------------
`world_rule_shift_interval` in {0 = never, 60 = long, 25 = medium, 10 = short} at
`world_rule_shift_depth = 2` -- 798a's validated ladder, which is also INV-063's
"{0 / long / medium / short}". Plus ARM_NOISE: P1's own matched-PE
observation-noise control at 798a's sigma = 0.12, which 798a measured as MEL-matched
to its HIGH arm. It is DELIBERATELY UN-LEARNABLE -- additive Gaussian noise on the
exteroceptive channel has no structure to re-learn -- so if it reproduces the DV
pattern the ladder is grading NOISE, not learning load, which is the DV-symmetry
artifact SD-MEL-PRODUCER's design note names by that name.

Seeds 42 / 123 / 456 (798a's and 1069's set; seed 44 is deliberately absent --
recurring reef-config early-death instability, EXQ-539/540, V3-EXQ-538a).

THE DVs, AND THEIR POLARITY
----------------------------
C1 reads "non-increasing across intake-sorted arms". Intake-sorted means DESCENDING
(HIGH, MED, LOW, NONE): function falls as intake falls. Every DV below is oriented so
that MORE is BETTER, so all four must be non-increasing along HIGH -> NONE.

  LEG A (Types 1/2 contrastive replay), all three named in C1:
    A1  VALENCE_SURPRISE residue writes per waking period (agent._surprise_write_count)
    A2  realised surprise_weight at the replay call, min(1, _pe_ema * 5.0) -- and P4
        requires _pe_ema > 0 so it is not pinned at its 0.3 fallback
    A3  spread of the replay start-selection priority. The registered text offers
        "rising entropy of the start-selection histogram / falling SPREAD of replay
        priority"; the SPREAD form is used because it shares the others' polarity
        (falls as intake falls), and the entropy form is recorded alongside.
  LEG B:
    B1  across-sleep frozen-battery SD-056 InfoNCE delta at the pinned tau = 1e-3
        (pre-sleep minus post-sleep on the SAME battery, so the DV is what sleep
        ADDED, not how hard the waking period was).

C1 -- both legs monotone (MONO_TOL slack, 798a:379) on >= 2/3 seeds.
C2 -- THE KNEE, and the load-bearing leg. Per seed, the drop between the two LOWEST
     intake arms must exceed the drop between the two HIGHEST by
     max(2 x pooled cross-seed SD of the arm-to-arm delta, 20% of the highest arm's
     DV), on >= 2/3 seeds, in the SAME direction on both legs. Self-computing from
     this run's own data -- no threshold is imported and none is invented. Applied to
     EVERY C1 DV, which is the literal reading of "the C1 DVs" (plural) and is the
     STRICT one: it cannot be accused of weakening by selecting a favourable DV.

P1-P5, EVERY ONE ASSERTED FROM MEASURED OUTPUT
------------------------------------------------
P1  the ladder is REAL, not nominal: mean waking MEL monotone across the four intake
    arms with registered relative spread (mel[HIGH]-mel[NONE])/mel[NONE] >= 0.25.
    V3-EXQ-1069 met this on 2/3 seeds, NOT 3/3, so it is carried as a PER-SEED
    precondition: a seed that fails P1 is EXCLUDED from C1/C2 scoring rather than
    failing the run. Plus the reducibility control -- elevated PE must DECAY within a
    stationary window, asserted against the emitted `steps_since_world_rule_shift` --
    and ARM_NOISE must not reproduce the DV pattern.
P2  the offline budget is PINNED and SHOWN pinned: use_mel_consumer /
    use_entry_pressure / use_within_life_sleep_trigger all False, and
    `sws_n_writes` / `rem_n_rollouts` asserted from OUTPUT to have ZERO cross-arm
    variance. (INV-063's text names `cumulative_sws_writes` / `cumulative_rem_rollouts`;
    NEITHER EXISTS in ree_core -- they appear only in a mel_consumer docstring. The
    real merged-cycle keys are the two used here. Correction owed to /governance as
    GFLAG-0359.)
P3  sleep FIRES and the offline pathway is UNSILENCED: multi-episode driver so
    notify_episode_end is reachable, use_sleep_aggregation_cluster=True, and non-zero
    SWS writes AND non-zero REM rollouts asserted in EVERY arm including the floor arm.
P4  the MECH-205 instrument is LIVE: surprise_gated_replay=True AND valence_enabled,
    a NON-ZERO count of VALENCE_SURPRISE writes, _pe_ema > 0 at the replay call, and
    genuine PE VARIANCE across episodes.
P5  the ladder is NOT read through a broken slot instrument. No DV here routes through
    ContextMemory `slot_cosine_sim`; the Type-3 leg is EXCLUDED from this falsifier by
    the claim itself, and e2_fast.py carries zero ContextMemory references.

DEVIATION FROM 798a, the same one V3-EXQ-1069 made and for the same reason: P0 is
computed ONCE PER SEED and restored into a freshly built agent per arm via a detached
state_dict snapshot. 798a's `_train_p0_and_probe` takes no arm parameter, so per-cell
recomputation yields a bit-identical agent; this is the same computation performed
once, and it makes the five arms exactly seed-matched. CONSEQUENCE, declared: cells
share a P0 agent, so every cell is emitted reuse_ineligible.

DV-SYMMETRY / per-arm declaration (mandatory)
-----------------------------------------------
The four intake arms share one DV family: per-arm MEANS and COUNTS over a fixed
waking budget, plus a signed across-sleep difference on a FIXED battery. None is an
argmax/rank statistic (monotone-rescaling class cannot apply). A1 is a COUNT over a
fixed budget and A2/A3/B1 are means or differences -- set-aggregates -- so the
permutation class WOULD apply to a manipulation that permuted interchangeable units;
the manipulation is the RATE of action-map re-permutation, which changes the CONTENT
the model is wrong about rather than reordering units, so it is not such a
manipulation. A broadcast constant common to all arms cancels in C2's differences,
which is correct: a uniform shift in any DV is exactly what "no intake grading" means
and must not read as a knee.
  ARM_NOISE carries the same DV family and the OPPOSITE expectation: its manipulation
  (additive observation noise) is deliberately un-learnable, so a DV pattern there is
  evidence the ladder grades noise. It is scored as a CONTROL, never as a fifth rung.

WHAT THIS RUN CANNOT SETTLE
-----------------------------
- The Type-3 NREM schema-consolidation leg and the E2 motor-sequence leg are EXCLUDED
  by INV-063 itself (P5 and the action-learning competence floor). A result here says
  nothing about either.
- With the refusal route armed, a `leg_b_dv_sign_inverted` outcome leaves C1/C2
  unadjudicated. That is the point of the route, not a defect.
- P1 held on 2/3 seeds in V3-EXQ-1069 and is carried per-seed here; if it fails on
  enough seeds this run reports that and scores nothing, which is the honest outcome
  rather than a verdict.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_1071_inv063_four_arm_intake_ladder.py --dry-run
  /opt/local/bin/python3 experiments/v3_exq_1071_inv063_four_arm_intake_ladder.py
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell
from experiments.pack_writer import write_flat_manifest

EXPERIMENT_TYPE = "v3_exq_1071_inv063_four_arm_intake_ladder"
QUEUE_ID = "V3-EXQ-1071"
CLAIM_IDS: List[str] = ["INV-063"]
EXPERIMENT_PURPOSE = "evidence"

DEAD_Z_GOAL_STREAM_EXEMPT = (
    "inherited verbatim from V3-EXQ-798a's _make_agent for comparability with the P1 "
    "regime V3-EXQ-1069 validated; the dead stream was adjudicated non_contributory "
    "2026-07-27 and the knobs are arm-symmetric. Wiring it live would change behaviour "
    "relative to the runs this one must be comparable to."
)
SD056_ROLLOUT_CLAMP_EXEMPT = (
    "no imagination rollout is run: the InfoNCE readout is a no_grad evaluation on a "
    "fixed battery, and the only contrastive training is the sleep pass's 8 steps at "
    "lr 1e-3 scoped to the two world heads. Divergence is DETECTED (all e1/e2 params "
    "asserted finite after every cycle) rather than assumed absent. Enabling the clamp "
    "would break comparability with V3-EXQ-1060/1063, whose readouts this extends."
)
ANCHOR_REACHABILITY_EXEMPT = (
    "P1 and R2 are inherited verbatim from V3-EXQ-701c via 798a and MEASURED clearing "
    "on this exact instrument and base: 798a conv 1.0 / V3-EXQ-1069 conv 0.9313-0.9871 "
    "against a 0.10 floor, and 1069 met P1's own spread bar on 2/3 seeds. The P4/P2/P3 "
    "liveness anchors were measured non-zero in this session's feasibility probe under "
    "this exact config (221 surprise writes, _pe_ema 0.00622, sws_n_writes 5.0, "
    "rem_n_rollouts 10.0, updates_e2_world 8.0). No anchor here is narrower than the "
    "state it anchors to."
)

# --- 798a ENV_BASE / drift, verbatim (798a :410-428) ------------------------
ENV_BASE: Dict[str, Any] = dict(
    size=12, num_hazards=4, num_resources=5, hazard_harm=0.05,
    proximity_harm_scale=0.1, proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2, hazard_field_decay=0.5,
    resource_respawn_on_consume=True, toroidal=False,
    harm_history_len=10, use_proxy_fields=True,
)
STABLE_DRIFT = dict(env_drift_interval=999, env_drift_prob=0.0)

# --- schedule ---------------------------------------------------------------
SEEDS: Tuple[int, ...] = (42, 123, 456)
CONV_EPISODES = 60
STEPS_PER_EPISODE = 90
P0_STEPS = CONV_EPISODES * STEPS_PER_EPISODE          # 5400, 798a's exact budget
N_CYCLES = 3
WAKE_EPS_PER_CYCLE = 2
EPISODES_PER_RUN = CONV_EPISODES + N_CYCLES * WAKE_EPS_PER_CYCLE   # 66

# --- 798a E2 P0 training (798a :364-372) ------------------------------------
SD056_WEIGHT = 0.05
E2_LR = 1e-3
CONTRASTIVE_BATCH_K = 8
MIN_BUFFER_BEFORE_TRAIN = 16
MAX_GRAD_NORM = 1.0
TRANSITION_BUFFER_MAX = 256

# --- sleep stack ------------------------------------------------------------
CMC_STEPS, CMC_LR, CMC_BATCH = 8, 1e-3, 16
BATTERY_SIZE = 64
BATTERY_FLOOR = 32.0
PROBE_STEPS = 100

# --- the PINNED leg-B readout temperature (USER-RATIFIED, GFLAG-0364 pending)
TAU_PINNED = 1e-3
# Recorded in every cell so leg B is re-readable without a re-run (option (c)).
TAU_LADDER: Tuple[float, ...] = (10.0, 1.0, 0.1, 0.03, 0.01, 3e-3, 1e-3, 1e-4)

# --- pre-registered thresholds, NOT derived from this run's statistics -------
MIN_REL_MEL_SPREAD = 0.25     # P1, the V3-EXQ-701c relative criterion
MONO_TOL = 0.02               # C1/P1 monotonicity slack vs the floor arm (798a:379)
SEED_PASS_FRAC = 2.0 / 3.0
MIN_REL_CONV_DROP = 0.10      # R2, P0 converged (798a:376)
C2_SD_MULT = 2.0              # C2 margin, registered
C2_ABS_FRAC = 0.20            # C2 margin, registered
NOISE_SIGMA = 0.12            # 798a's measured MEL-match to its HIGH arm
EPS = 1e-12

ARM_NONE, ARM_LOW, ARM_MED, ARM_HIGH = (
    "ARM_0_NONE", "ARM_1_LOW", "ARM_2_MED", "ARM_3_HIGH")
ARM_NOISE = "ARM_4_NOISE_CONTROL"
# Declared INTAKE-DESCENDING: C1 requires the DVs non-increasing along this order.
INTAKE_DESC: Tuple[str, ...] = (ARM_HIGH, ARM_MED, ARM_LOW, ARM_NONE)
ARM_SPEC: Dict[str, Dict[str, Any]] = {
    ARM_NONE: {"interval": 0, "depth": 0, "sigma": 0.0, "rung": True},
    ARM_LOW: {"interval": 60, "depth": 2, "sigma": 0.0, "rung": True},
    ARM_MED: {"interval": 25, "depth": 2, "sigma": 0.0, "rung": True},
    ARM_HIGH: {"interval": 10, "depth": 2, "sigma": 0.0, "rung": True},
    ARM_NOISE: {"interval": 0, "depth": 0, "sigma": NOISE_SIGMA, "rung": False},
}
ARMS: Tuple[str, ...] = (ARM_NONE, ARM_LOW, ARM_MED, ARM_HIGH, ARM_NOISE)

# The C1 DVs, all oriented MORE-IS-BETTER so all are non-increasing HIGH -> NONE.
DV_KEYS: Tuple[str, ...] = ("A1_surprise_writes", "A2_surprise_weight",
                            "A3_replay_priority_spread", "B1_infonce_delta")
LEG_A_KEYS = DV_KEYS[:3]
LEG_B_KEYS = DV_KEYS[3:]

ETHICS_PREFLIGHT = {
    "involves_negative_valence": False, "involves_suffering_like_state": False,
    "involves_self_model": False, "involves_inescapability_or_helplessness": False,
    "involves_offline_replay_over_harm": False,
    "involves_social_mind_or_language": False,
    "involves_human_data_or_clinical_context": False, "decision": "allow",
}


# ---------------------------------------------------------------------------
def _fin(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _mean(xs) -> float:
    v = [float(x) for x in xs if _fin(x) is not None]
    return sum(v) / len(v) if v else float("nan")


def _sd(xs) -> float:
    v = [float(x) for x in xs if _fin(x) is not None]
    if len(v) < 2:
        return float("nan")
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def _make_env(seed: int, interval: int, depth: int) -> CausalGridWorldV2:
    kw = dict(ENV_BASE); kw.update(STABLE_DRIFT)
    kw.update(world_rule_shift_enabled=(interval > 0),
              world_rule_shift_interval=interval, world_rule_shift_depth=depth)
    return CausalGridWorldV2(seed=seed, **kw)


def _make_probe_env(seed: int) -> CausalGridWorldV2:
    kw = dict(ENV_BASE); kw.update(STABLE_DRIFT)
    return CausalGridWorldV2(seed=seed, **kw)


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    """798a's _make_agent VERBATIM, plus the sleep stack P2/P3/P4 require."""
    return REEAgent(REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32, world_dim=32, alpha_world=0.9, alpha_self=0.3,
        use_harm_stream=True, z_harm_dim=32, use_affective_harm_stream=True,
        z_harm_a_dim=16, harm_history_len=10,
        z_goal_enabled=True, goal_weight=0.5, drive_weight=2.0,
        e1_goal_conditioned=True,
        use_resource_proximity_head=True, resource_proximity_weight=0.5,
        benefit_eval_enabled=True, benefit_weight=1.0,
        e2_action_contrastive_enabled=True, e2_action_contrastive_weight=SD056_WEIGHT,
        # P3
        sws_enabled=True, rem_enabled=True, use_sleep_loop=True,
        sleep_loop_episodes_K=1_000_000, use_sleep_aggregation_cluster=True,
        use_cross_module_consolidation=True,
        cross_module_consolidation_schedule="interleaved",
        cross_module_consolidation_steps=CMC_STEPS,
        cross_module_consolidation_lr=CMC_LR,
        cross_module_consolidation_batch=CMC_BATCH,
        use_sleep_world_forward_consolidation=True,
        # P4
        surprise_gated_replay=True, pe_ema_alpha=0.02,
        # P2 -- the MEL CONSUMER stays ABSENT
        use_mel_consumer=False, use_entry_pressure=False,
        use_within_life_sleep_trigger=False))


REQUIRED_CONFIG: Dict[str, Any] = {
    "latent.alpha_world": 0.9, "latent.self_dim": 32, "latent.world_dim": 32,
}


def _assert_config(agent: REEAgent) -> Dict[str, Any]:
    """Dotted paths: from_dims routes these into NESTED sub-configs, so a flat
    getattr returns None (V3-EXQ-1069's authoring smoke caught that as a gate
    unmeetable by construction)."""
    out: Dict[str, Any] = {}
    for path, want in REQUIRED_CONFIG.items():
        node: Any = agent.config
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        out[path] = {"want": want,
                     "got": float(node) if isinstance(node, (int, float)) else node,
                     "ok": bool(isinstance(node, (int, float))
                                and float(node) == float(want))}
    return out


def _obs(d: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    v = d.get(key)
    if v is None:
        return None
    return v if torch.is_tensor(v) else torch.as_tensor(v, dtype=torch.float32)


def _apply_obs_noise(obs_dict: Dict[str, Any], sigma: float,
                     gen: Optional[torch.Generator]) -> Dict[str, Any]:
    """798a :565-577 verbatim. NEGATIVE CONTROL: additive Gaussian noise on the
    exteroceptive channel, deliberately UN-LEARNABLE -- there is no structure here to
    re-learn, so a model cannot reduce the PE it induces no matter how long it trains."""
    if sigma <= 0.0:
        return obs_dict
    out = dict(obs_dict)
    ws = _obs(obs_dict, "world_state")
    if ws is not None:
        out["world_state"] = ws + torch.randn(
            ws.shape, generator=gen, dtype=ws.dtype) * sigma
    return out


def _sense_latent(agent: REEAgent, obs_dict: Dict[str, Any]):
    body = obs_dict["body_state"].float()
    world = obs_dict["world_state"].float()
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    return agent.sense(obs_body=body, obs_world=world,
                       obs_harm=_obs(obs_dict, "harm_obs"),
                       obs_harm_a=_obs(obs_dict, "harm_obs_a"),
                       obs_harm_history=_obs(obs_dict, "harm_history"))


# ---------------------------------------------------------------------------
# leg A3 -- replay start-selection, recorded by a PURE shim
# ---------------------------------------------------------------------------
class _StartSelectionRecorder:
    """Wraps HippocampalModule._select_valence_weighted_start to record the priority
    vector it scores and the index it picks. PURE RECORDING: it calls the original and
    returns its result unchanged, adds no RNG draw, and mutates no agent state -- the
    method returns only the chosen tensor, so the priorities are not otherwise
    observable. A3 is its SPREAD (max - min), which the registered text names as the
    alternative form of 'the start-selection distribution collapses toward uniform'
    and which shares the other leg-A DVs' polarity (falls as intake falls)."""

    def __init__(self, agent: REEAgent):
        self.agent = agent
        self.spreads: List[float] = []
        self.chosen: List[int] = []
        # A2 as REGISTERED -- "the realised surprise_weight AT THE REPLAY CALL".
        # drive_state[VALENCE_SURPRISE] is exactly the value agent.py:10838 wrote,
        # so reading it here is the registered quantity rather than a _pe_ema proxy.
        self.weights: List[float] = []
        self.n_calls = 0
        self._orig = None

    def __enter__(self):
        hip = self.agent.hippocampal
        self._orig = hip._select_valence_weighted_start

        def shim(theta_buffer_recent, drive_state):
            self.n_calls += 1
            try:
                if drive_state is not None and drive_state.numel() > 3:
                    self.weights.append(float(drive_state[3].item()))  # VALENCE_SURPRISE
                rf = self.agent.residue_field
                if hasattr(rf, "get_valence_priority") and theta_buffer_recent is not None:
                    with torch.no_grad():
                        pri = [float(rf.get_valence_priority(theta_buffer_recent[t],
                                                             drive_state).sum().item())
                               for t in range(theta_buffer_recent.shape[0])]
                    if pri:
                        self.spreads.append(max(pri) - min(pri))
                        self.chosen.append(int(max(range(len(pri)),
                                                   key=lambda i: pri[i])))
            except (RuntimeError, ValueError, AttributeError, IndexError):
                pass
            return self._orig(theta_buffer_recent, drive_state)

        hip._select_valence_weighted_start = shim
        return self

    def __exit__(self, *exc):
        if self._orig is not None:
            self.agent.hippocampal._select_valence_weighted_start = self._orig
        return False

    def histogram_entropy(self) -> float:
        """The registered ALTERNATIVE form, recorded alongside A3."""
        if not self.chosen:
            return float("nan")
        counts: Dict[int, int] = {}
        for i in self.chosen:
            counts[i] = counts.get(i, 0) + 1
        n = float(len(self.chosen))
        return -sum((c / n) * math.log(c / n) for c in counts.values())


# ---------------------------------------------------------------------------
# frozen held-out battery + the two readouts
# ---------------------------------------------------------------------------
def _sample_probe_battery(agent: REEAgent, seed: int, size: int, steps: int):
    """798a :798-825 / 1069, transcribed. Held-out env, fixed action policy, PURE
    READ -- never calls _e1_tick, so it appends nothing to the trainer's buffers."""
    env = _make_probe_env(seed + 9973)
    _, obs_dict = env.reset()
    agent.reset(); agent.e1.reset_hidden_state()
    battery: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    prev: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    rng = np.random.default_rng(seed + 4242)
    for _t in range(steps):
        if len(battery) >= size:
            break
        z_now = _sense_latent(agent, obs_dict).z_world.detach().reshape(-1).clone()
        if prev is not None:
            battery.append((prev[0], prev[1], z_now))
        a_idx = int(rng.integers(0, env.action_dim))
        a_vec = torch.zeros(env.action_dim, dtype=torch.float32)
        a_vec[a_idx] = 1.0
        prev = (z_now, a_vec)
        _, _r, done, _i, obs_dict = env.step(a_idx)
        if done:
            _, obs_dict = env.reset()
            agent.reset(); agent.e1.reset_hidden_state()
            prev = None
    return battery


def _battery_readouts(agent: REEAgent, battery) -> Dict[str, float]:
    """InfoNCE at EVERY tau in the ladder, plus the 701b MSE for continuity with
    V3-EXQ-1063. One forward pass; the taus are re-normalisations of one fixed
    squared-distance matrix, which is why recording the whole ladder is free."""
    out: Dict[str, float] = {}
    if len(battery) < 2:
        return {f"infonce_tau_{t:g}": float("nan") for t in TAU_LADDER}
    z0 = torch.stack([b[0] for b in battery]).to(agent.device)
    a = torch.stack([b[1] for b in battery]).to(agent.device)
    z1 = torch.stack([b[2] for b in battery]).to(agent.device)
    with torch.no_grad():
        pred = agent.e2.world_forward(z0, a)
        out["mse"] = float((pred - z1).pow(2).mean().item())
        sq = (pred.unsqueeze(0) - z1.unsqueeze(1)).pow(2).sum(dim=-1)
        labels = torch.arange(len(battery), device=agent.device)
        for t in TAU_LADDER:
            out[f"infonce_tau_{t:g}"] = float(
                F.cross_entropy(-sq / float(t), labels).item())
        out["diag_sq_mean"] = float(sq.diagonal().mean().item())
        out["offdiag_sq_mean"] = float(
            sq[~torch.eye(len(battery), dtype=torch.bool, device=agent.device)]
            .mean().item())
    return out


# ---------------------------------------------------------------------------
# P0 -- 798a's, shared per seed
# ---------------------------------------------------------------------------
def _e2_train_step(agent, buffer: Deque, opt, rng) -> Optional[float]:
    """798a :604-626. RECON-ONLY -- the SD-056 auxiliary is a CONFIRMED P0
    destabiliser (V3-EXQ-701b ablation) and is not added to the P0 loss."""
    if len(buffer) < MIN_BUFFER_BEFORE_TRAIN:
        return None
    pool = list(buffer)
    batch = pool if len(pool) <= CONTRASTIVE_BATCH_K else rng.sample(
        pool, CONTRASTIVE_BATCH_K)
    z0 = torch.stack([t[0] for t in batch]).to(agent.device)
    ac = torch.stack([t[1] for t in batch]).to(agent.device)
    z1 = torch.stack([t[2] for t in batch]).to(agent.device)
    opt.zero_grad(set_to_none=True)
    loss = F.mse_loss(agent.e2.world_forward(z0, ac), z1)
    v = float(loss.detach().item())
    if not math.isfinite(v):
        return v
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.e2.parameters(), max_norm=MAX_GRAD_NORM)
    opt.step()
    return v


def _waking_step(agent, env, obs_dict, sigma, noise_gen, train, buffer, opt, rng,
                 pending, allow_replay: bool = False):
    """798a's _step_cycle, plus two additions this claim's leg A requires.

    (1) the observation-noise hook the P1 matched-PE control needs.
    (2) `allow_replay`: the MECH-092 quiescent-replay branch. THE AUTHORING SMOKE
        CAUGHT THAT LEG A IS OTHERWISE UNMEASURABLE BY CONSTRUCTION. `_do_replay` is
        called ONLY from agent.act() / act_with_split_obs() / act_with_log_prob()
        (agent.py:10750-10751, :10774-10775), gated on ticks["e3_quiescent"]. This
        driver hand-rolls act()'s body the way 798a does, and 798a omitted the replay
        branch because 798a had no leg-A DV -- so HippocampalModule.replay() was never
        called, _select_valence_weighted_start never fired, and A3 read nan in every
        cell of the first smoke. INV-063's leg A is defined AT the replay call ("the
        realised surprise_weight AT THE REPLAY CALL", "the replay start-selection
        distribution"), so the call has to happen. This restores the substrate's OWN
        behaviour through its OWN gate; it invents nothing.

        Scoped to the MEASUREMENT phase only. P0 stays byte-for-byte 798a's form, so
        the warmup this run shares with V3-EXQ-1069 is unchanged. The measurement phase
        does therefore differ from 1069's step form, which is why P1 is re-measured
        here per seed rather than inherited -- see the P1 precondition.
    """
    latent = _sense_latent(agent, _apply_obs_noise(obs_dict, sigma, noise_gen))
    if train and buffer is not None:
        pend = pending[0]
        if pend is not None:
            z1o = latent.z_world.detach().reshape(-1).clone()
            if torch.isfinite(pend[0]).all() and torch.isfinite(z1o).all():
                buffer.append((pend[0], pend[1], z1o))
            pending[0] = None
    ticks = agent.clock.advance()
    wdim = latent.z_world.shape[-1]
    e1_prior = (agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device))
    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
    action = agent.select_action(candidates, ticks)
    if action is None:
        idx = int(np.random.randint(0, env.action_dim))
        action = torch.zeros(1, env.action_dim, device=agent.device)
        action[0, idx] = 1.0
        agent._last_action = action
    if not torch.isfinite(action).all():
        return None, obs_dict, True, None
    if train and buffer is not None and torch.isfinite(latent.z_world).all():
        pending[0] = (latent.z_world.detach().reshape(-1).clone(),
                      action.detach().reshape(-1).clone())
        if opt is not None and rng is not None:
            _e2_train_step(agent, buffer, opt, rng)
    agent.record_executed_action(action)
    _, harm_signal, done, info, nxt = env.step(action)
    with torch.no_grad():
        metrics = agent.update_residue(harm_signal=float(harm_signal),
                                       world_delta=None, hypothesis_tag=False,
                                       owned=True)
    if allow_replay and ticks.get("e3_quiescent", False):
        # MECH-092, via the substrate's own gate. All replay content carries
        # hypothesis_tag=True and cannot produce residue (MECH-094), so this adds
        # no DV contamination -- it is what makes leg A's DVs exist at all.
        agent._do_replay(latent)
    pe = metrics.get("e3_prediction_error")
    pe = (float(pe.detach().item()) if torch.is_tensor(pe)
          else (float(pe) if pe is not None else None))
    ssl = int(getattr(env, "_steps_since_world_rule_shift", 0))
    return (pe if (pe is not None and math.isfinite(pe)) else None), nxt, bool(done), ssl


def train_p0(seed: int, p0_steps: int, probe_steps: int, probe_size: int
             ) -> Dict[str, Any]:
    env = _make_probe_env(seed)
    agent = _make_agent(env)
    cfg_check = _assert_config(agent)
    opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_LR)
    buffer: Deque = deque(maxlen=TRANSITION_BUFFER_MAX)
    rng = random.Random(seed)

    battery = _sample_probe_battery(agent, seed, probe_size, probe_steps)
    pre = _battery_readouts(agent, battery)

    _, obs_dict = env.reset(); agent.reset(); agent.e1.reset_hidden_state()
    pending: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None]
    used = 0
    while used < p0_steps:
        print(f"  [train] P0_SHARED seed={seed} P0 ep "
              f"{min(used // STEPS_PER_EPISODE, CONV_EPISODES - 1) + 1}/"
              f"{EPISODES_PER_RUN}", flush=True)
        _, obs_dict = env.reset(); agent.reset(); agent.e1.reset_hidden_state()
        pending[0] = None
        for _s in range(STEPS_PER_EPISODE):
            if used >= p0_steps:
                break
            _pe, obs_dict, done, _ssl = _waking_step(
                agent, env, obs_dict, 0.0, None, True, buffer, opt, rng, pending)
            used += 1
            if done:
                break
    post = _battery_readouts(agent, battery)
    conv = ((pre["mse"] - post["mse"]) / pre["mse"]) if pre["mse"] > EPS else 0.0
    state = {k: (v.detach().clone() if torch.is_tensor(v) else v)
             for k, v in agent.state_dict().items()}
    print(f"  [P0 seed={seed}] conv_rel_drop={conv:.4f} "
          f"battery MSE {pre['mse']:.6g} -> {post['mse']:.6g} "
          f"cfg_ok={all(v['ok'] for v in cfg_check.values())}", flush=True)
    return {"agent": agent, "p0_state": state, "battery": battery,
            "conv_rel_drop": float(conv), "config_check": cfg_check}


# ---------------------------------------------------------------------------
def run_cell(arm_id: str, seed: int, p0: Dict[str, Any], n_cycles: int,
             wake_eps: int, steps: int) -> Dict[str, Any]:
    spec = ARM_SPEC[arm_id]
    print(f"Seed {seed} Condition {arm_id}", flush=True)
    env = _make_env(seed, spec["interval"], spec["depth"])
    agent = _make_agent(env)
    agent.load_state_dict(p0["p0_state"])
    restore_ok = all(torch.equal(v, p0["p0_state"][k])
                     for k, v in agent.state_dict().items() if torch.is_tensor(v))
    battery = p0["battery"]
    noise_gen = torch.Generator().manual_seed(seed + 31337)

    _, obs_dict = env.reset(); agent.reset(); agent.e1.reset_hidden_state()
    pending: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None]
    pes: List[float] = []
    pe_by_ssl: Dict[int, List[float]] = {}
    ep_pe_means: List[float] = []
    cyc_writes: List[float] = []
    cyc_weights: List[float] = []
    cyc_spreads: List[float] = []
    cyc_entropies: List[float] = []
    cyc_replay_calls: List[float] = []
    cyc_infonce: List[Dict[str, float]] = []
    sws, rem, fired, finite_ok = [], [], 0, True
    prev_writes = int(getattr(agent, "_surprise_write_count", 0))
    ep_global = CONV_EPISODES

    for cyc in range(n_cycles):
        with _StartSelectionRecorder(agent) as rec:
            for _ep in range(wake_eps):
                print(f"  [train] {arm_id} seed={seed} P1_LADDER ep "
                      f"{ep_global + 1}/{EPISODES_PER_RUN} cycle {cyc + 1}/{n_cycles}",
                      flush=True)
                ep_global += 1
                _, obs_dict = env.reset(); agent.reset(); agent.e1.reset_hidden_state()
                pending[0] = None
                ep_pes: List[float] = []
                for _s in range(steps):
                    pe, obs_dict, done, ssl = _waking_step(
                        agent, env, obs_dict, spec["sigma"], noise_gen,
                        False, None, None, None, pending, allow_replay=True)
                    if pe is None and done and ssl is None:
                        finite_ok = False
                        break
                    if pe is not None:
                        pes.append(pe); ep_pes.append(pe)
                        if ssl is not None:
                            pe_by_ssl.setdefault(min(int(ssl), 60), []).append(pe)
                    if done:
                        break
                if ep_pes:
                    ep_pe_means.append(_mean(ep_pes))
                if not finite_ok:
                    break
            ema = float(getattr(agent, "_pe_ema", 0.0))
            # A2: prefer the value READ AT the replay call; fall back to the
            # _pe_ema-derived proxy only if no replay call happened this cycle, and
            # record which was used so the distinction is auditable.
            cyc_weights.append(_mean(rec.weights) if rec.weights
                               else (min(1.0, ema * 5.0) if ema > 0 else 0.3))
            cyc_replay_calls.append(float(rec.n_calls))
            cyc_spreads.append(_mean(rec.spreads) if rec.spreads else float("nan"))
            cyc_entropies.append(rec.histogram_entropy())
        if not finite_ok:
            break
        pre_r = _battery_readouts(agent, battery)
        sm = agent.sleep_loop.force_cycle(agent) or {}
        post_r = _battery_readouts(agent, battery)
        if "post_sleep_z_goal_retention" in sm:
            fired += 1
        sws.append(float(sm.get("sws_n_writes", 0.0)))
        rem.append(float(sm.get("rem_n_rollouts", 0.0)))
        cyc_infonce.append({k: pre_r[k] - post_r[k] for k in pre_r
                            if k.startswith("infonce_tau_")}
                           | {"mse_delta": pre_r["mse"] - post_r["mse"],
                              "infonce_pre_pinned": pre_r[f"infonce_tau_{TAU_PINNED:g}"]})
        w = int(getattr(agent, "_surprise_write_count", 0))
        cyc_writes.append(float(w - prev_writes)); prev_writes = w

    params_finite = all(bool(torch.isfinite(p).all().item())
                        for p in list(agent.e1.parameters()) + list(agent.e2.parameters()))
    mel = _mean(pes)
    pe_var = (_sd(pes) ** 2) if len(pes) > 1 else float("nan")
    b1 = _mean([c[f"infonce_tau_{TAU_PINNED:g}"] for c in cyc_infonce])
    dvs = {
        "A1_surprise_writes": _mean(cyc_writes),
        "A2_surprise_weight": _mean(cyc_weights),
        "A3_replay_priority_spread": _mean(cyc_spreads),
        "B1_infonce_delta": b1,
    }
    # P1 reducibility control: elevated PE must DECAY within a stationary window.
    early = _mean([v for k, vs in pe_by_ssl.items() if k <= 2 for v in vs])
    late = _mean([v for k, vs in pe_by_ssl.items() if k >= 8 for v in vs])
    decay = ((early - late) / early) if _fin(early) and early > EPS else float("nan")

    ok = bool(restore_ok and params_finite and fired == n_cycles and pes)
    print(f"  {arm_id} seed={seed} iv={spec['interval']} sig={spec['sigma']} "
          f"MEL={mel:.6g} pe_var={pe_var:.4g} cycles={fired}/{n_cycles} "
          f"A1={dvs['A1_surprise_writes']:.4g} A2={dvs['A2_surprise_weight']:.4g} "
          f"A3={dvs['A3_replay_priority_spread']:.4g} B1={dvs['B1_infonce_delta']:.6g} "
          f"sws={_mean(sws):.3g} rem={_mean(rem):.3g} decay={decay:.4g} "
          f"replays={sum(cyc_replay_calls):.0f} "
          f"restore={int(restore_ok)}", flush=True)
    print(f"verdict: {'PASS' if ok else 'FAIL'}", flush=True)

    return {
        "arm_id": arm_id, "seed": seed, "is_rung": bool(spec["rung"]),
        "world_rule_shift_interval": spec["interval"],
        "world_rule_shift_depth": spec["depth"], "obs_noise_sigma": spec["sigma"],
        "mel_mean_pe": mel, "n_meas_pe": len(pes), "pe_variance": pe_var,
        "pe_variance_across_episodes": (_sd(ep_pe_means) ** 2
                                        if len(ep_pe_means) > 1 else float("nan")),
        "dvs": dvs,
        "a3_start_selection_entropy": _mean(cyc_entropies),
        "replay_calls_per_cycle": cyc_replay_calls,
        "replay_calls_total": float(sum(cyc_replay_calls)),
        "surprise_writes_per_cycle": cyc_writes,
        "surprise_weight_per_cycle": cyc_weights,
        "infonce_delta_per_cycle_full_tau_ladder": cyc_infonce,
        "sws_n_writes_per_cycle": sws, "rem_n_rollouts_per_cycle": rem,
        "sws_n_writes_mean": _mean(sws), "rem_n_rollouts_mean": _mean(rem),
        "cycles_fired": float(fired),
        "pe_decay_within_stationary_window": decay,
        "conv_rel_drop": p0["conv_rel_drop"],
        "p0_state_restore_ok": bool(restore_ok),
        "params_finite": bool(params_finite),
        "cell_ok": ok,
    }


# ---------------------------------------------------------------------------
def _knee(vals_desc: List[float], sd_delta: float) -> Tuple[bool, float, float]:
    """C2, registered. vals_desc is the DV at HIGH, MED, LOW, NONE. The drop between
    the two LOWEST intake arms must exceed the drop between the two HIGHEST by
    max(2 x pooled cross-seed SD of the arm-to-arm delta, 20% of the highest arm's DV)."""
    hi_drop = vals_desc[0] - vals_desc[1]          # HIGH -> MED
    lo_drop = vals_desc[2] - vals_desc[3]          # LOW  -> NONE
    margin = max(C2_SD_MULT * sd_delta if _fin(sd_delta) else 0.0,
                 C2_ABS_FRAC * abs(vals_desc[0]) if _fin(vals_desc[0]) else 0.0)
    return bool(lo_drop - hi_drop > margin), float(lo_drop - hi_drop), float(margin)


def main(dry_run: bool = False) -> Tuple[str, Optional[str]]:
    seeds = list(SEEDS[:1]) if dry_run else list(SEEDS)
    p0_steps = 90 if dry_run else P0_STEPS
    n_cycles = 2 if dry_run else N_CYCLES
    wake_eps = 1 if dry_run else WAKE_EPS_PER_CYCLE
    # 35, not 30: n_cycles(2) x wake_eps(1) x steps must exceed ARM_1_LOW's interval
    # (60) with margin, or LOW's first world-rule shift never fires within the smoke
    # budget and LOW reads bit-identical to NONE (coverage gap, not a design issue).
    steps = 35 if dry_run else STEPS_PER_EPISODE
    probe_steps = 40 if dry_run else PROBE_STEPS
    probe_size = BATTERY_SIZE

    t0 = time.time()
    rows: Dict[Tuple[str, int], Dict[str, Any]] = {}
    arm_results: List[Dict[str, Any]] = []
    agents: List[REEAgent] = []
    p0_by_seed: Dict[int, Dict[str, Any]] = {}

    for seed in seeds:
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        p0_by_seed[seed] = train_p0(seed, p0_steps, probe_steps, probe_size)
        agents.append(p0_by_seed[seed]["agent"])
        for arm_id in ARMS:
            slice_ = {"experiment": EXPERIMENT_TYPE, "arm": arm_id,
                      "interval": ARM_SPEC[arm_id]["interval"],
                      "depth": ARM_SPEC[arm_id]["depth"],
                      "sigma": ARM_SPEC[arm_id]["sigma"],
                      "p0_steps": p0_steps, "n_cycles": n_cycles,
                      "wake_eps": wake_eps, "steps": steps,
                      "tau_pinned": TAU_PINNED, "battery": probe_size}
            with arm_cell(seed, config_slice=slice_, script_path=Path(__file__),
                          extra_ineligible_reasons=[
                              "shared_p0_agent_across_arms_within_seed"]) as cell:
                row = run_cell(arm_id, seed, p0_by_seed[seed], n_cycles, wake_eps,
                               steps)
                cell.stamp(row)
            rows[(arm_id, seed)] = row
            arm_results.append(row)

    elapsed = time.time() - t0
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    rungs = [a for a in INTAKE_DESC]

    # ---- P1..P5, per seed where they are per-seed ---------------------------
    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        mel_asc = [rows[(a, seed)]["mel_mean_pe"] for a in
                   (ARM_NONE, ARM_LOW, ARM_MED, ARM_HIGH)]
        fin = all(_fin(m) is not None for m in mel_asc)
        floor = mel_asc[0] if fin else float("nan")
        tol = MONO_TOL * abs(floor) if fin else 0.0
        p1_mono = bool(fin and all(mel_asc[i + 1] >= mel_asc[i] - tol
                                   for i in range(3)))
        p1_spread = (((mel_asc[-1] - floor) / floor)
                     if fin and abs(floor) > EPS else float("nan"))
        noise_mel = rows[(ARM_NOISE, seed)]["mel_mean_pe"]
        noise_reproduces = bool(_fin(noise_mel) is not None and noise_mel >= mel_asc[-1])
        pe_decay_per_arm = {a: rows[(a, seed)]["pe_decay_within_stationary_window"]
                            for a in ARMS}
        # P1's reducibility control (INV-063 what_would_answer, GFLAG-0446): elevated
        # PE must DECAY within a stationary window. Checked on the three rungs that
        # actually undergo a world-rule shift (ARM_LOW/MED/HIGH) -- ARM_NONE and
        # ARM_NOISE have world_rule_shift_enabled=False, so there is no "elevated PE"
        # event for either to decay from.
        p1_decay_ok = bool(all(_fin(pe_decay_per_arm[a]) is not None
                               and pe_decay_per_arm[a] > 0
                               for a in (ARM_LOW, ARM_MED, ARM_HIGH)))
        p1_met = bool(p1_mono and _fin(p1_spread) is not None
                      and p1_spread >= MIN_REL_MEL_SPREAD
                      and p1_decay_ok and not noise_reproduces)
        per_seed.append({
            "seed": seed, "mel_ascending_intake": mel_asc,
            "p1_monotone": p1_mono, "p1_relative_spread": p1_spread,
            "p1_decay_ok": p1_decay_ok,
            "p1_met": p1_met,
            "conv_rel_drop": rows[(ARM_NONE, seed)]["conv_rel_drop"],
            "noise_arm_mel": noise_mel,
            "noise_arm_at_least_as_elevated_as_high": noise_reproduces,
            "pe_decay_per_arm": pe_decay_per_arm,
        })
    p1_seeds = [p["seed"] for p in per_seed if p["p1_met"]]
    scored = p1_seeds
    # NOMINAL, from the total seed pool -- NOT len(scored). This is a vote-count floor
    # ("did we retain enough of the intended population to trust a verdict"), not a
    # per-DV threshold computed from the very population it gates; deriving it from
    # len(scored) instead makes `len(scored) >= need` a tautology (ceil(2/3*k) <= k for
    # every k), silently disabling the "too few scored seeds" degeneracy signal below.
    need = math.ceil(SEED_PASS_FRAC * len(seeds))

    all_rows = [rows[k] for k in rows]
    cells_ok = all(r["cell_ok"] for r in all_rows)
    cfg_ok = all(v["ok"] for p in p0_by_seed.values()
                 for v in p["config_check"].values())
    min_conv = min(p["conv_rel_drop"] for p in per_seed)
    sws_var = _sd([rows[(a, s)]["sws_n_writes_mean"] for a in ARMS for s in seeds])
    rem_var = _sd([rows[(a, s)]["rem_n_rollouts_mean"] for a in ARMS for s in seeds])
    min_sws = min(rows[(a, s)]["sws_n_writes_mean"] for a in ARMS for s in seeds)
    min_rem = min(rows[(a, s)]["rem_n_rollouts_mean"] for a in ARMS for s in seeds)
    min_writes = min(rows[(a, s)]["dvs"]["A1_surprise_writes"] for a in ARMS for s in seeds)
    max_weight = max(rows[(a, s)]["dvs"]["A2_surprise_weight"] for a in ARMS for s in seeds)
    min_pe_var = min(rows[(a, s)]["pe_variance_across_episodes"]
                     for a in ARMS for s in seeds)

    preconditions = [
        {"name": "P1_intake_ladder_graded_per_seed", "kind": "readiness",
         "description": ("mean waking MEL monotone with registered relative spread "
                         ">= 0.25; PER SEED -- V3-EXQ-1069 met it on 2/3, so a failing "
                         "seed is EXCLUDED from scoring rather than failing the run"),
         "measured": float(len(p1_seeds)), "threshold": 1.0, "direction": "lower",
         "comparator": ">=", "control": "798a's validated ladder at its own P0",
         "met": bool(len(p1_seeds) >= 1)},
        {"name": "P2_offline_budget_pinned_zero_cross_arm_variance", "kind": "readiness",
         "description": ("sws_n_writes / rem_n_rollouts cross-arm SD, with the MEL "
                         "consumer absent (INV-063 names cumulative_* keys that do not "
                         "exist -- GFLAG-0359)"),
         "measured": float(max(sws_var if _fin(sws_var) else 0.0,
                               rem_var if _fin(rem_var) else 0.0)),
         "threshold": 1e-9, "direction": "upper", "comparator": "<=",
         "control": "scheduler-pinned budget with use_mel_consumer=False",
         "met": bool((_fin(sws_var) or 0.0) <= 1e-9 and (_fin(rem_var) or 0.0) <= 1e-9)},
        {"name": "P3_sleep_fired_with_work_in_every_arm", "kind": "readiness",
         "description": "non-zero SWS writes AND REM rollouts in EVERY arm and seed",
         "measured": float(min(min_sws, min_rem)), "threshold": 0.0,
         "direction": "lower", "comparator": ">",
         "control": "use_sleep_aggregation_cluster=True, multi-episode driver",
         "met": bool(min_sws > 0.0 and min_rem > 0.0)},
        {"name": "P4a_surprise_writes_nonzero", "kind": "readiness",
         "description": "MECH-205 instrument live: non-zero VALENCE_SURPRISE writes",
         "measured": float(min_writes), "threshold": 0.0, "direction": "lower",
         "comparator": ">", "control": "surprise_gated_replay=True",
         "met": bool(min_writes > 0.0)},
        {"name": "P4b_surprise_weight_not_pinned_at_fallback", "kind": "readiness",
         "description": ("_pe_ema > 0 at the replay call, so surprise_weight is not "
                         "pinned at its 0.3 fallback (agent.py:10812-10814)"),
         "measured": float(max_weight), "threshold": 0.3, "direction": "upper",
         "comparator": "<", "control": "the fallback value itself is the bound",
         "met": bool(max_weight < 0.3)},
        {"name": "P4c_pe_variance_across_episodes", "kind": "readiness",
         "description": "genuine PE variance across episodes (a flat stream is unmeasurable)",
         "measured": float(min_pe_var), "threshold": 0.0, "direction": "lower",
         "comparator": ">", "control": "per-episode MEL means within a cell",
         "met": bool(_fin(min_pe_var) is not None and min_pe_var > 0.0)},
        {"name": "P4e_replay_calls_nonzero", "kind": "readiness",
         "description": ("MECH-092 quiescent replay actually fired, per cell. Leg A's "
                         "A2 and A3 are defined AT the replay call; the authoring "
                         "smoke measured A3 = nan in every cell before the quiescent "
                         "branch was restored, i.e. leg A was unmeasurable by "
                         "construction"),
         "measured": float(min(rows[(a, s)]["replay_calls_total"]
                               for a in ARMS for s in seeds)),
         "threshold": 0.0, "direction": "lower", "comparator": ">",
         "control": "agent._do_replay through the substrate's own e3_quiescent gate",
         "met": bool(min(rows[(a, s)]["replay_calls_total"]
                         for a in ARMS for s in seeds) > 0.0)},
        {"name": "R2_p0_converged", "kind": "readiness",
         "description": "798a R2, frozen-probe PE drop across P0",
         "measured": float(min_conv), "threshold": MIN_REL_CONV_DROP,
         "direction": "lower", "comparator": ">",
         "control": "recon-only P0 on the stable env", "met": bool(min_conv > MIN_REL_CONV_DROP)},
        {"name": "P4d_config_is_798a_and_cells_restored", "kind": "readiness",
         "description": "798a config asserted off the live agent; P0 weights restored per cell",
         "measured": float(1.0 if (cfg_ok and cells_ok) else 0.0), "threshold": 1.0,
         "direction": "lower", "comparator": ">=",
         "control": "dotted-path config equality + torch.equal per cell",
         "met": bool(cfg_ok and cells_ok)},
    ]
    gating_ok = all(p["met"] for p in preconditions)

    run_config = {
        "arms": {a: dict(ARM_SPEC[a]) for a in ARMS},
        "intake_descending": list(INTAKE_DESC), "seeds": list(seeds),
        "p0_steps": p0_steps, "n_cycles": n_cycles, "wake_eps_per_cycle": wake_eps,
        "steps_per_episode": steps, "episodes_per_run": EPISODES_PER_RUN,
        "battery_size": probe_size, "tau_pinned": TAU_PINNED,
        "tau_ladder": list(TAU_LADDER), "min_rel_mel_spread": MIN_REL_MEL_SPREAD,
        "mono_tol": MONO_TOL, "seed_pass_frac": SEED_PASS_FRAC,
        "c2_sd_mult": C2_SD_MULT, "c2_abs_frac": C2_ABS_FRAC,
        "noise_sigma": NOISE_SIGMA, "env_base": dict(ENV_BASE),
        "cmc_steps": CMC_STEPS, "cmc_lr": CMC_LR, "cmc_batch": CMC_BATCH,
        "p0_shared_per_seed": True,
    }
    base = {
        "schema_version": "v1", "run_id": run_id, "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID, "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "sleep_driver_pattern": ("manual-cycle-loop (run_sleep_cycle() called once per "
                                 "cycle in a dedicated N_CYCLES wake-sleep-test loop)"),
        "timestamp_utc": ts, "started_utc": datetime.utcnow().isoformat() + "Z",
        "arm_results": arm_results, "per_seed_p1": per_seed,
        "ethics_preflight": dict(ETHICS_PREFLIGHT),
        "ratified_amendments": {
            "leg_b_readout": (
                "claims.yaml specifies the V3-EXQ-701b/701c frozen-probe instrument "
                "(per-element MSE). This run uses the SD-056 InfoNCE frozen-battery "
                f"objective at a PINNED tau = {TAU_PINNED:g}. USER-RATIFIED "
                "2026-09-19T21:42:30Z, PENDING governance application of GFLAG-0364 "
                "(same posture V3-EXQ-1039a used for its A2 narrowing). Basis: the MSE "
                "readout degrades across sleep on a converged base (V3-EXQ-1063, 9/9 "
                "cells negative); at this tau InfoNCE carries 18.54% headroom below "
                "ln(K) against 0.62% at the shipped tau=0.1."),
            "leg_b_refusal_route": (
                "USER-RATIFIED 2026-09-20 (option (c)). The tau ladder on a converged "
                "798a base shows headroom and the asserted across-sleep direction "
                "trading off with NO overlap (readable needs tau <= 0.01, positive "
                "delta needs tau >= 0.03; staged at inv063_legb_tau_bind_20260920.md, "
                "GFLAG-0382). So if leg B's delta is negative at the pinned tau across "
                "arms this run emits label leg_b_dv_sign_inverted, does NOT read C1 "
                "leg B, and routes NO F1/F2/F3 verdict. Without that route a negative "
                "leg B would route to F1, which INV-063 pre-registers as a GENUINE "
                "falsification -- and on the tau evidence that reading would be wrong."),
        },
        "scope_note": (
            "INV-063's registered falsifier. The Type-3 NREM leg and the E2 "
            "motor-sequence leg are EXCLUDED by the claim itself (P5 and the "
            "action-learning competence floor); nothing here bears on either."),
    }

    def _write(m):
        if dry_run:
            print(f"[{EXPERIMENT_TYPE}] dry-run complete; not writing manifest.")
            return None
        p = write_flat_manifest(m, dry_run=False, config=m.get("config"),
                                seeds=list(seeds), script_path=Path(__file__),
                                agent=agents)
        print(f"Result written to: {p}")
        return str(p)

    if not gating_ok or not scored or len(scored) < need:
        unmet = [p["name"] for p in preconditions if not p["met"]]
        if not scored:
            unmet.append("P1_met_on_zero_seeds_nothing_scorable")
        elif len(scored) < need:
            # Too few seeds survived P1 to satisfy SEED_PASS_FRAC of the NOMINAL seed
            # pool: falling through to C1/C2 here would compare n_monotone/n_knee
            # (capped at len(scored)) against `need`, which is unsatisfiable by
            # construction and would emit a false F1_flat/"genuinely falsified" label
            # -- this is "scores nothing", not a finding in either direction.
            unmet.append(f"P1_met_on_too_few_seeds ({len(scored)} of {len(seeds)} "
                        f"scored, need {need})")
        reason = f"preconditions unmet: {unmet}"
        print(f"\n[{EXPERIMENT_TYPE}] -> substrate_not_ready_requeue: {reason}", flush=True)
        m = dict(base); m.update({
            "outcome": "FAIL", "result": "FAIL",
            "evidence_direction": "non_contributory",
            "evidence_direction_note": (
                "Non-degeneracy preconditions unmet; NO C1/C2 reading and NO F1/F2/F3 "
                "verdict is emitted. This is not an INV-063 finding in either "
                "direction. " + reason),
            "non_degenerate": False, "degeneracy_reason": "substrate_not_ready: " + reason,
            "interpretation": {"label": "substrate_not_ready_requeue",
                               "preconditions": preconditions,
                               "criteria_non_degenerate": {}},
            "readout": {"substrate_ready": 0, "overall_pass": 0,
                        "p1_seeds_met": float(len(p1_seeds))},
            "config": run_config, "elapsed_seconds": elapsed})
        return "FAIL", _write(m)

    # ---- the RATIFIED refusal route, checked BEFORE C1 leg B ---------------
    # ANY negative arm fires the route, not ALL: a negative delta at the pinned tau
    # is the sign-inversion artifact GFLAG-0382 measured (readable needs tau <= 0.01,
    # a positive delta needs tau >= 0.03, no overlap), so even one contaminated arm
    # makes C1's cross-arm monotonicity/knee read on leg B unreliable. Requiring ALL
    # four arms negative let a single non-negative arm through to F1_flat/weakens
    # (the false falsification the route exists to prevent) and let a marginally
    # positive HIGH arm with the other three negative reach F2/confirmed while sleep
    # degraded the readout in 3 of 4 arms.
    legb_by_arm = {a: _mean([rows[(a, s)]["dvs"]["B1_infonce_delta"] for s in scored])
                   for a in rungs}
    legb_negative_arms = [a for a, v in legb_by_arm.items()
                          if _fin(v) is not None and v < 0]
    legb_any_negative = bool(legb_negative_arms)

    # ---- C1 / C2 ------------------------------------------------------------
    def dv_desc(key, seed):
        return [rows[(a, seed)]["dvs"][key] for a in INTAKE_DESC]

    c1_by_key: Dict[str, Dict[str, Any]] = {}
    for key in DV_KEYS:
        seeds_mono = []
        for s in scored:
            v = dv_desc(key, s)
            fin = all(_fin(x) is not None for x in v)
            tol = MONO_TOL * abs(v[-1]) if fin else 0.0
            seeds_mono.append(bool(fin and all(v[i] >= v[i + 1] - tol
                                               for i in range(3))))
        c1_by_key[key] = {"per_seed_monotone": seeds_mono,
                          "n_monotone": int(sum(seeds_mono)),
                          "per_seed_values_intake_desc": [dv_desc(key, s) for s in scored]}

    c2_by_key: Dict[str, Dict[str, Any]] = {}
    for key in DV_KEYS:
        deltas = [[dv_desc(key, s)[i] - dv_desc(key, s)[i + 1] for i in range(3)]
                  for s in scored]
        pooled = _mean([_sd([d[i] for d in deltas]) for i in range(3)])
        res, diffs, margins = [], [], []
        for s in scored:
            ok, diff, marg = _knee(dv_desc(key, s), pooled)
            res.append(ok); diffs.append(diff); margins.append(marg)
        c2_by_key[key] = {"per_seed_knee": res, "n_knee": int(sum(res)),
                          "per_seed_excess": diffs, "per_seed_margin": margins,
                          "pooled_cross_seed_sd_of_arm_to_arm_delta": pooled}

    legA_c1 = all(c1_by_key[k]["n_monotone"] >= need for k in LEG_A_KEYS)
    legB_c1 = (None if legb_any_negative
               else all(c1_by_key[k]["n_monotone"] >= need for k in LEG_B_KEYS))
    legA_c2 = all(c2_by_key[k]["n_knee"] >= need for k in LEG_A_KEYS)
    legB_c2 = (None if legb_any_negative
               else all(c2_by_key[k]["n_knee"] >= need for k in LEG_B_KEYS))

    if legb_any_negative:
        label = "leg_b_dv_sign_inverted"
        outcome, direction = "FAIL", "non_contributory"
        note = (
            f"RATIFIED REFUSAL ROUTE FIRED. Leg B's across-sleep InfoNCE delta at the "
            f"pinned tau = {TAU_PINNED:g} is NEGATIVE in {len(legb_negative_arms)} of "
            f"{len(rungs)} intake arms ({sorted(legb_negative_arms)}: "
            f"{ {a: round(v, 6) for a, v in legb_by_arm.items()} }), i.e. sleep makes "
            f"the readout WORSE in at least one arm before any intake comparison. C1 "
            f"leg B is therefore NOT read and NO F1/F2/F3 verdict is emitted -- a "
            f"negative leg B would otherwise route to F1, which INV-063 pre-registers "
            f"as a GENUINE falsification, and on the tau evidence (GFLAG-0382) that "
            f"reading would be wrong. Everything else IS reported: leg A's C1 held on "
            f"{ {k: c1_by_key[k]['n_monotone'] for k in LEG_A_KEYS} } of "
            f"{len(scored)} scored seeds and its C2 on "
            f"{ {k: c2_by_key[k]['n_knee'] for k in LEG_A_KEYS} }. The full tau ladder "
            f"is recorded per cell, so leg B can be re-read at another tau without "
            f"re-running this experiment.")
    elif legA_c1 and legB_c1 and legA_c2 and legB_c2:
        label = "inv063_confirmed_starvation_with_floor"
        outcome, direction = "PASS", "supports"
        note = ("C1 and C2 both hold on both legs: offline function degrades "
                "monotonically as intake falls at a pinned offline budget, and the "
                "degradation is DISPROPORTIONATE at the low end. First simulation "
                "evidence of any kind for INV-063.")
    elif legA_c1 and legB_c1:
        label = "F2_smooth_no_knee"
        outcome, direction = "FAIL", "mixed"
        note = ("F2, the pre-registered likeliest outcome: C1 holds (monotone "
                "degradation) but C2 fails -- proportional across the ladder with no "
                "low-end collapse. Pre-registered consequence: the word 'minimum' is "
                "not earned; the threshold content retires into INV-050 and this "
                "invariant's residual narrows to the FUNCTION-SIDE and "
                "function-SPECIFICITY readings.")
    else:
        label = "F1_flat"
        outcome, direction = "FAIL", "weakens"
        note = ("F1: with P1-P5 met, the C1 function DVs do NOT degrade monotonically "
                "with falling intake. Offline function is insensitive to intake at "
                "fixed budget -- the starvation mechanism this invariant asserts is "
                "genuinely falsified rather than substrate-confounded.")

    # Over ALL measured seeds, not just `scored`: a seed with the noise arm
    # reproducing the pattern is now EXCLUDED from `scored` by p1_met (the fix
    # above), so restricting this aggregate to `scored` would make it vacuously
    # False by construction on every run that reaches this point -- the seeds it
    # would have flagged are never in `scored` to begin with.
    noise_reproduces = all(
        p["noise_arm_at_least_as_elevated_as_high"] for p in per_seed)
    crit = [
        {"name": "C1_both_legs_monotone", "load_bearing": True,
         "passed": bool(legA_c1 and (legB_c1 is True)),
         "measured": float(min(c1_by_key[k]["n_monotone"] for k in DV_KEYS)),
         "threshold": float(need), "comparator": ">=", "per_dv": c1_by_key},
        {"name": "C2_knee_both_legs", "load_bearing": True,
         "passed": bool(legA_c2 and (legB_c2 is True)),
         "measured": float(min(c2_by_key[k]["n_knee"] for k in DV_KEYS)),
         "threshold": float(need), "comparator": ">=", "per_dv": c2_by_key},
    ]
    nd = {
        "C1_both_legs_monotone": bool(not legb_any_negative and len(scored) >= need),
        "C2_knee_both_legs": bool(not legb_any_negative and len(scored) >= need),
    }

    print(f"\n[{EXPERIMENT_TYPE}] P1 scored seeds: {scored} of {seeds}", flush=True)
    for k in DV_KEYS:
        print(f"  {k}: C1 {c1_by_key[k]['n_monotone']}/{len(scored)} monotone, "
              f"C2 {c2_by_key[k]['n_knee']}/{len(scored)} knee (need {need})", flush=True)
    print(f"  -> {label} ({outcome}); elapsed={elapsed:.1f}s", flush=True)

    flat = {
        "overall_pass": int(outcome == "PASS"),
        "p1_seeds_met": float(len(p1_seeds)), "n_seeds_scored": float(len(scored)),
        "seeds_required": float(need),
        "leg_b_sign_inverted": int(legb_any_negative),
        "min_conv_rel_drop": float(min_conv),
        "min_sws_n_writes": float(min_sws), "min_rem_n_rollouts": float(min_rem),
        "budget_cross_arm_sd": float(max(sws_var if _fin(sws_var) else 0.0,
                                         rem_var if _fin(rem_var) else 0.0)),
        "min_surprise_writes": float(min_writes),
        "max_surprise_weight": float(max_weight),
        "noise_control_at_least_as_elevated": int(noise_reproduces),
    }
    for k in DV_KEYS:
        flat[f"c1_monotone_seeds_{k.lower()}"] = float(c1_by_key[k]["n_monotone"])
        flat[f"c2_knee_seeds_{k.lower()}"] = float(c2_by_key[k]["n_knee"])
    for a in rungs:
        v = _fin(legb_by_arm[a])
        if v is not None:
            flat[f"legb_infonce_delta_{a.lower()}"] = v

    m = dict(base); m.update({
        "outcome": outcome, "result": outcome, "evidence_direction": direction,
        "evidence_direction_note": note,
        "non_degenerate": bool(all(nd.values())),
        "degeneracy_reason": (None if all(nd.values()) else
                              "leg B refused at the ratified sign-inversion route, or "
                              "fewer than the required seeds met P1"),
        "interpretation": {
            "label": label, "criteria": crit,
            "combination_rule": (
                "P1 is PER SEED: a seed failing it is EXCLUDED from scoring, not fatal "
                "(V3-EXQ-1069 met P1 on 2/3). P2-P5 and R2 GATE the whole run. The "
                "RATIFIED refusal route is checked BEFORE C1 leg B: leg B negative at "
                "the pinned tau in every intake arm -> leg_b_dv_sign_inverted, no "
                "C1-legB reading, no F1/F2/F3. Otherwise PASS iff C1 AND C2 on BOTH "
                "legs; C1-only -> F2; neither -> F1. C1/C2 are applied to EVERY C1 DV, "
                "the literal reading of 'the C1 DVs' and the strict one."),
            "criteria_non_degenerate": nd, "preconditions": preconditions},
        "readout": flat,
        "leg_b_tau_ladder_note": (
            "infonce_delta_per_cycle_full_tau_ladder in every arm_results row carries "
            f"the across-sleep delta at every tau in {list(TAU_LADDER)}, plus the 701b "
            "MSE delta and the battery's diagonal/off-diagonal squared distances. Those "
            "are re-normalisations of one fixed distance matrix, so recording the whole "
            "ladder costs microseconds -- and it is what lets leg B be re-read at a "
            "different tau without re-running an 8-hour experiment."),
        "config": run_config, "elapsed_seconds": elapsed})
    return outcome, _write(m)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    _o, _p = main(dry_run=args.dry_run)
    emit_outcome(outcome=(str(_o).upper() if str(_o).upper() in ("PASS", "FAIL") else "FAIL"),
                 manifest_path=_p, dry_run=args.dry_run)
    sys.exit(0)
