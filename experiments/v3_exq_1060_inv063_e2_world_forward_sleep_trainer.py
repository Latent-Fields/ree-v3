#!/opt/local/bin/python3
"""
V3-EXQ-1060 -- E2.world_forward SLEEP-TRAINER liveness validation (INV-063 leg B).

SLEEP DRIVER: manual-cycle-loop (run_sleep_cycle() called once per cycle in a
              dedicated N_CYCLES wake-sleep-test loop)

WHY THIS RUN (the gap it closes)
---------------------------------
V3-EXQ-1026 PASSED and proved the SleepLoopManager call site fires MECH-423's R3
CrossModuleConsolidator through a real sleep cycle. Its confirmed autopsy
(failure_autopsy_V3-EXQ-1026_2026-09-14, Step 7c red-team F1, independently
re-verified) then established what that PASS did NOT show, verbatim:

  "That is INACCURATE -- the measured update is confined to E2's self-forward
   head (predict_next_self); E2.world_forward moved by exactly 0."

substrate_queue.json entry `e2-world-forward-sleep-trainer` records the gap and
pre-registers this run's criterion in its one open failure_record, verbatim:

  metric: "C3 max|delta_e2| = 0.00769 (min over 3 seeds) is entirely within E2's
           self-forward parameters; E2.world_forward parameter delta == 0.0
           exactly on every seed and arm (Step 7c red-team per-named-parameter
           probe, independently re-verified)."
  target: "a sleep-integrated cycle must move E2.world_forward parameters
           (delta > 0 on the ON arm, == 0 on the OFF arm) so INV-063 leg B's
           frozen held-out world-forward prediction-error DV becomes
           non-degenerate."

The build that closes it is ree-v3 4610133 (2026-09-17), lever
`use_sleep_world_forward_consolidation` (REEConfig, default False, explicit in
from_dims), pinned by 13 contract tests W1-W8 in
tests/contracts/test_e2_world_forward_sleep_trainer.py. This run is the first
exercise of that lever in a landed experiment manifest.

WHY LEG B IS E2, NOT E1 (resolved before authoring -- do not re-derive)
-----------------------------------------------------------------------
claims.yaml INV-063 labels leg B "Leg B (E1 world-model updating)", while this
entry, the 1026 driver and the 1026 autopsy all name E2.world_forward. Those are
different modules, so the chain was checked against the tree before this script
was written:

  1. E1DeepPredictor has NO world_forward head at all. Every occurrence of
     "world_forward" in ree_core/predictors/e1_deep.py (lines 622, 1380, 1516,
     1520) is a COMMENT referring to E2's -- e.g. "SD-056 already constrains
     E2's one-step `world_forward`".
  2. The instrument INV-063's own what_would_answer NAMES BY NAME -- "the
     V3-EXQ-701b/701c frozen-probe instrument" -- is an E2 instrument BY
     CONSTRUCTION: v3_exq_701b.py:547 and v3_exq_701c.py:554 both compute
     `agent.e2.world_forward(z0, a)`, and 701b's docstring states "P0 trains
     ONLY agent.e2 ... conv_rel_drop on the fixed battery reflects ONLY
     world_forward convergence".

So "(E1 world-model updating)" is a PROSE function-label inherited from INV-063's
four-function description taxonomy; the OPERATIONAL DV named in the same sentence
is an agent.e2.world_forward measurement. The substrate chain is correct and this
build does unblock leg B. A claims.yaml label correction is owed to /governance
(raised as a governance flag alongside this queue entry); it is not a code change
and does not affect this run.

TWO ARMS (seed-matched; the ONLY difference is the lever)
---------------------------------------------------------
Both arms run the FULL MECH-423 consolidation pass (consolidator present,
CMC_STEPS=8, interleaved), exactly as V3-EXQ-1026's ON arm did. The only
difference is `use_sleep_world_forward_consolidation`:

  ARM_WORLD_FORWARD_ON  -- lever True. A third module "e2_world" joins the
                           interleaved schedule, scoped to e2.world_transition +
                           e2.world_action_encoder (phase_manager.py:710-718).
  ARM_WORLD_FORWARD_OFF -- lever False. Bit-identical by STRUCTURAL ABSENCE --
                           no key, no closure, no RNG draw
                           (phase_manager.py:700-708 comment).

WHY THE OFF ARM KEEPS CONSOLIDATION ON, rather than disabling it as 1026's OFF
arm did: that is what makes C4 a sharp attribution control. With the pass
running and the lever off, the world heads ARE in the "e2" optimiser's parameter
list and still take zero gradient -- which is precisely the recorded defect. A
consolidation-disabled arm would only show that nothing happens when nothing
runs.

Both arms run in THIS run and are NOT compared against 1026's landed manifest:
adding a third module to the interleaved schedule consumes global-RNG draws
BETWEEN the e1 and e2 replay draws and changes cross_module_replay_share
denominators (contract W2c measures exactly this divergence), so the two are not
tick-comparable across runs.

CRITERIA -- LIVENESS ONLY, TRANSCRIBED FROM THE RATIFIED ENTRY
--------------------------------------------------------------
C1/C2 carry V3-EXQ-1026's forward, extended to the new "e2_world" module name;
C3/C4 are the entry's pre-registered liveness pair, re-pointed from
agent.e2.parameters() to the two world heads.

THE EFFICACY LEG IS RECORDED, NOT GATED (user decision, 2026-09-19)
-------------------------------------------------------------------
Across-sleep held-out world-forward prediction error IS measured -- pre-sleep
minus post-sleep on a FROZEN held-out battery, the V3-EXQ-701b/701c frozen-probe
form INV-063's what_would_answer names -- and carries NO pass/fail criterion.
That is a deliberate, user-made decision, not an omission: no threshold for this
quantity is pre-registered anywhere in the corpus (not in the substrate_queue
entry, whose failure_record pre-registers liveness only; not in INV-063's
what_would_answer, whose C1/C2 thresholds belong to the four-arm intake ladder, a
different design; not in the 1026 driver), and the one precedent value --
v3_exq_701b.py:194 MIN_REL_CONV_DROP = 0.10 -- gates P0 convergence over
CONV_EPISODES = 60 of dedicated Adam training, NOT 8 consolidation steps at
lr 1e-3 inside one sleep cycle. Inventing a threshold across that regime gap is
the "unreachable by construction" failure the V3-EXQ-981 autopsy recorded three
times in one run, and INV-063's own text already warns against reusing this
constant family ("the absolute ABS_MEL_FLOOR=1e-4 of 701c is structurally
unreachable on a converged base and must NOT be reused").

BOTH metrics are recorded on the same frozen battery, because the trainer's
objective and INV-063's DV are DIFFERENT quantities and the gap is the point:
compute_e2_world_loss minimises the SD-056 InfoNCE CONTRASTIVE loss, while the
701b frozen-probe DV is per-element MSE RECONSTRUCTION error. InfoNCE is
insensitive to a global scale/shift of the prediction, so it can improve while
frozen-battery MSE does not move. Recording both makes that mismatch measurable
rather than a confound, and is what would let a LATER run set a defensible
threshold. This run's PASS/FAIL turns on C1-C4 alone.

WHAT A PASS DOES AND DOES NOT MEAN
-----------------------------------
EXPERIMENT_PURPOSE = "diagnostic". A PASS makes INV-063 leg B's frozen held-out
world-forward prediction-error DV NON-DEGENERATE. It does NOT adjudicate INV-063:
the four-arm intake-ladder falsifier is a separate and much heavier design and
already has an open chip (chip-proposal-exp-0736-paced). claim_ids=["MECH-423"]
is for context/traceability only; diagnostic runs are excluded from governance
confidence/conflict scoring.

SCOPE -- THE DV IS NOT WIDENED PAST THE TWO WORLD HEADS
-------------------------------------------------------
The parameter-delta DV covers e2.world_transition + e2.world_action_encoder
only. V3-EXQ-1026 scoped its C3 to agent.e2.parameters() deliberately (its
"WHY THE E2 LEG (not E1)" section): ree_core/predictors/e2_fast.py has ZERO
references to ContextMemory / context_memory / e1_deep (verified again at
authoring time: grep count 0), so the E2 reading is clean of the OPEN
`corrupting` substrate_queue entry
`contextmemory-write-path-addressing-degeneracy`
(substrate_paths: ree_core/predictors/e1_deep.py::ContextMemory.write).
The consolidation pass DOES step "e1" in both arms, and that path reads
ContextMemory -- but it is common to both arms and cannot move a world-head
parameter: the OFF arm has no world loss closure at all, and the ON arm's world
loss is computed from _world_experience_buffer / _action_experience_buffer with
no ContextMemory involvement. Widening the DV onto the E1 leg would import that
confound and trip the corrupting-overlap rule; it is deliberately not done.

PARAMETER SCOPING -- THE GUARANTEE IS ONE-DIRECTIONAL, NOT DISJOINTNESS
-----------------------------------------------------------------------
phase_manager.py:691 sets _cmc_params["e2"] = list(agent.e2.parameters()), which
is a SUPERSET of _cmc_params["e2_world"]. So the two parameter LISTS are NOT
disjoint and asserting that they are would fail by construction. The build's
actual guarantee is one-directional: the "e2_world" set contains no parameter
outside the two world heads, so it cannot double-step the z_self head "e2" owns.
Measured at authoring time: |world| = 6, |all e2| = 18, |non-world e2| = 12, and
world INTERSECT non-world = 0. That is the assertion this script makes (P4).
Double-stepping is additionally absent in practice because
cross_module_consolidation.py:178 `opt.zero_grad()` is per-optimiser and
set_to_none, and the "e2" step's backward (compute_e2_loss -> predict_next_self
only) leaves the world heads' grads None, so Adam skips them.

NUMERICAL STABILITY: as in V3-EXQ-1026, consolidate() builds LOCAL per-module
Adam optimizers scoped only to the passed parameter lists, so the 680b/680c
shared-encoder (LatentStack) divergence does not transfer. This script asserts
every e1/e2 parameter is finite after each cycle and FAILs cleanly rather than
crashing past a manifest write.

DV-SYMMETRY / per-arm declaration (mandatory):
  ARM_WORLD_FORWARD_ON and ARM_WORLD_FORWARD_OFF share one DV -- max|delta| over
  a FIXED, explicitly-named parameter set, a genuine per-run measurement. It is
  not an argmax/rank statistic, not a monotone rescaling of anything, and not a
  set-aggregate invariant under permutation of interchangeable units. The
  manipulation is a BINARY CODE-PATH GATE (whether a third loss closure and
  optimiser group exist at all), not a value transform of the DV, so none of the
  three DV-symmetry-invariance classes applies to either arm. The efficacy
  readouts are ungated telemetry and route no verdict, so they carry no
  DV-symmetry obligation.

RED-TEAM (Step 4.5, fable 5.1, one pass): CONTESTED -- 4 findings, all disposed.

  F1 (primary) FIXED. The driver never called any method that sets
     REEAgent._last_action, so _e1_action_one_hot() (agent.py:11148-11175)
     appended a ZERO vector on every _e1_tick and the world trainer trained on an
     all-zero action buffer. Because world_action_encoder is nn.Linear(a_dim,
     a_dim) (e2_fast.py:137), a zero input gives dL/dW = 0 EXACTLY -- so the
     action-conditioning WEIGHT never moved even on the ON arm, and C3's pooled
     max hid that behind world_transition. Verified independently against source
     and by direct measurement: nonzero_fraction 0.0, world_action_encoder.weight
     max|delta| EXACTLY 0.0. FIX: record_executed_action(action) in BOTH arms
     (the condition that method's own docstring sets for an ablation), which
     measured nonzero_fraction 0.967 and weight delta 0.0077. Guarded against
     regression by the action_buffer_non_vacuous precondition and by C6.

  F2 FIXED (reporting). The OFF arm's exactly-zero efficacy readout is a
     structural identity, not a measured null, and the ungated numbers could
     mislead a later reader. The efficacy block now carries
     off_arm_zero_is_structural + off_arm_note saying so explicitly, and an
     action_distribution_note recording the battery-vs-replay action distribution.

  F3 ACKNOWLEDGED, and it is why C6 exists. Given the readiness gate, Adam moves
     any parameter with a non-zero gradient by ~lr, so C3 alone ("something in the
     pooled set moved") carries little information beyond "the optimiser was
     wired" -- which contract tests W3/W4 already pin. That is inherent to a
     LIVENESS criterion, which is what the substrate_queue entry ratified, and is
     precisely why the efficacy telemetry was added. C6 restores discriminating
     power within the ratified target: it genuinely failed under the pre-fix
     driver and fails again on any regression of the action-buffer wiring.

  F4 FIXED. non_degenerate was set unconditionally True on the criteria path even
     when an entry of criteria_non_degenerate was False; it is now derived from
     that dict, with a degeneracy_reason naming the offending criteria. The
     non-finite route additionally names numerical DIVERGENCE explicitly rather
     than labelling it as merely-unmet readiness.

  CLEAN, checked and cited by the reviewer: the lever reaches the DV; the "e2"
  optimiser cannot contaminate C4 (per-optimiser set_to_none zero_grad precedes
  its backward, and compute_e2_loss reaches only predict_next_self); the pre-cycle
  loss probe cannot leak an update (buffers detached at write, no learnable
  temperature, grads cleared in a finally, run in BOTH arms so RNG-matched); the
  battery capture does not touch the trainer's buffers (only _e1_tick appends).

Red-team verdict is also recorded in the queue entry note.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_1060_inv063_e2_world_forward_sleep_trainer.py --dry-run
  /opt/local/bin/python3 experiments/v3_exq_1060_inv063_e2_world_forward_sleep_trainer.py
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell
from experiments._metrics import p0_readiness_gate, P0NotReady
from experiments.pack_writer import write_flat_manifest

EXPERIMENT_TYPE = "v3_exq_1060_inv063_e2_world_forward_sleep_trainer"
QUEUE_ID = "V3-EXQ-1060"
CLAIM_IDS: List[str] = ["MECH-423"]
EXPERIMENT_PURPOSE = "diagnostic"

ARM_ON = "ARM_WORLD_FORWARD_ON"
ARM_OFF = "ARM_WORLD_FORWARD_OFF"
ARMS = (ARM_ON, ARM_OFF)
SEEDS = (42, 123, 456)

# Substrate + rollout constants carried unchanged from V3-EXQ-1026 (:140-151).
GRID_SIZE = 5
N_HAZARDS = 1
N_RESOURCES = 1
SELF_DIM = 16
WORLD_DIM = 16

WAKING_EPISODES = 3          # the [train] denominator (M in "ep N/M")
STEPS_PER_EPISODE = 60       # -> ~180 transitions/cell

CMC_STEPS = 8
CMC_LR = 1e-3
CMC_BATCH = 16

# The world-domain trainer reads _world_experience_buffer / _action_experience_buffer
# (appended in REEAgent._e1_tick), NOT _e2_transition_buffer. It needs
# n_pairs = min(len(wbuf), len(abuf)) - 1 >= 2 to form InfoNCE negatives at all,
# and n_pairs >= CMC_BATCH to draw a full batch. Reachable by construction, not a
# hand-tuned degeneracy definition: heartbeat.e1_steps_per_tick == 1 in this
# config (measured at authoring time), so the buffers grow 1:1 with env steps and
# WAKING_EPISODES(3) x STEPS_PER_EPISODE(60) ~= 180 entries clears the floor >10x.
WORLD_BUFFER_FLOOR = float(CMC_BATCH + 1)
ANCHOR_REACHABILITY_EXEMPT = (
    "world_experience_buffer floor (17) is >10x cleared by the fixed waking "
    "rollout (~180 entries/cell at e1_steps_per_tick=1); not a hand-tuned "
    "degeneracy definition."
)

# Frozen held-out probe battery (701b:178 PROBE_BATTERY_SIZE = 64).
BATTERY_SIZE = 64
EPS = 1e-12

WORLD_HEAD_MODULES = ("world_transition", "world_action_encoder")

# The SD-056 rollout clamp is deliberately NOT enabled, for two reasons.
# (1) SCOPE: enabling it would change E2's imagination-rollout behaviour relative
#     to V3-EXQ-1026's configuration, which this run re-points rather than
#     redesigns -- the chip's instruction is to keep 1026's constants absent a
#     stated reason, and an unratified substrate-behaviour change is exactly what
#     would make the two non-comparable.
# (2) REACH: the lint's named failure mode is an unbounded imagination rollout
#     diverging to 1e16-1e18 and saturating an f_variance_share-style readout,
#     annihilating additive E3 score terms. This run reads NEITHER an imagination
#     rollout nor any E3 score term: its DV is a parameter delta over two named
#     modules, and its telemetry is a frozen-battery one-step prediction error.
#     The contrastive term is trained for EIGHT steps at lr 1e-3 inside a single
#     sleep cycle -- not the sustained training the divergence was measured under
#     (V3-EXQ-569e, V3-EXQ-936).
# Divergence is nonetheless GUARDED rather than assumed away: every e1/e2/world
# parameter is asserted finite after the cycle and the run FAILs cleanly (never
# crashes past a manifest write) if any is not.
SD056_ROLLOUT_CLAMP_EXEMPT = (
    "DV is a parameter delta + a frozen one-step battery error; no imagination "
    "rollout or E3 score term is read. 8 consolidation steps at lr 1e-3 is not "
    "the sustained contrastive training the divergence was measured under, and "
    "post-cycle finiteness is asserted with a clean FAIL. Enabling the clamp "
    "would also be an unratified change away from the V3-EXQ-1026 baseline this "
    "run re-points."
)


def _to_batched(x, device) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    else:
        x = x.to(device)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return x


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_hazards=N_HAZARDS,
        num_resources=N_RESOURCES,
        use_proxy_fields=True,
    )


def _make_agent(env: CausalGridWorldV2, world_forward_on: bool) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        sws_enabled=True,
        rem_enabled=True,
        use_sleep_loop=True,
        # A huge K keeps notify_episode_end's automatic K-episode cadence from
        # ever firing during the waking rollout, so the ONE deliberate
        # force_cycle() call is the only sleep cycle this cell fires.
        sleep_loop_episodes_K=1_000_000,
        use_sleep_aggregation_cluster=True,
        # IDENTICAL IN BOTH ARMS -- see the docstring. Only the lever below differs.
        use_cross_module_consolidation=True,
        cross_module_consolidation_schedule="interleaved",
        cross_module_consolidation_steps=CMC_STEPS,
        cross_module_consolidation_lr=CMC_LR,
        cross_module_consolidation_batch=CMC_BATCH,
        use_sleep_world_forward_consolidation=world_forward_on,
    )
    return REEAgent(cfg)


def _world_head_params(agent: REEAgent) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    for name in WORLD_HEAD_MODULES:
        mod = getattr(agent.e2, name, None)
        if mod is not None:
            out.extend(mod.parameters())
    return out


def _non_world_e2_params(agent: REEAgent) -> List[torch.Tensor]:
    return [
        p for n, p in agent.e2.named_parameters()
        if not any(n.startswith(m) for m in WORLD_HEAD_MODULES)
    ]


def _snapshot(params: List[torch.Tensor]) -> List[torch.Tensor]:
    return [p.detach().clone() for p in params]


def _max_abs_delta(before: List[torch.Tensor], after: List[torch.Tensor]) -> float:
    worst = 0.0
    for b, a in zip(before, after):
        if b.numel() == 0:
            continue
        worst = max(worst, float((a - b).abs().max().item()))
    return worst


def _all_finite(params: List[torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(p).all().item()) for p in params)


def _sense_z_world(agent: REEAgent, obs_dict: Dict) -> torch.Tensor:
    device = agent.device
    obs_harm = obs_dict.get("harm_obs", None)
    latent = agent.sense(
        _to_batched(obs_dict["body_state"], device),
        _to_batched(obs_dict["world_state"], device),
        obs_harm=_to_batched(obs_harm, device) if obs_harm is not None else None,
    )
    return latent.z_world.detach().reshape(1, -1).clone()


def _sample_probe_battery(
    agent: REEAgent, seed: int, n_transitions: int,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """FIXED held-out probe battery of one-step (z0, action, z1) transitions,
    captured with the agent's OWN (frozen) encoder so z0/z1 live in the latent
    space world_forward predicts. 701b's _sample_probe_battery form (:489-531),
    re-pointed from P0-convergence to across-sleep.

    HELD OUT: a distinct env instance (seed offset) and a FIXED action policy
    independent of training, so the battery is deterministic given
    (agent-init, seed) and its states are not the trainer's replay content.

    PURE READ: senses and steps the env only. It never calls _e1_tick, so it
    appends nothing to the replay buffers the trainer draws from, and it never
    trains anything.
    """
    env = _make_env(seed + 9973)
    _, obs_dict = env.reset()
    act_rng = random.Random(seed + 9973)
    battery: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    prev: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    guard = 0
    max_guard = max(n_transitions, 1) * 8
    with torch.no_grad():
        while len(battery) < n_transitions and guard < max_guard:
            guard += 1
            z_now = _sense_z_world(agent, obs_dict)
            if not bool(torch.isfinite(z_now).all().item()):
                break
            if prev is not None:
                z0, a = prev
                battery.append((z0, a, z_now))
            idx = act_rng.randrange(env.action_dim)
            action = torch.zeros(1, env.action_dim, device=agent.device)
            action[0, idx] = 1.0
            _, _, done, _, obs_dict = env.step(action)
            prev = (z_now, action)
            if done:
                _, obs_dict = env.reset()
                prev = None
    return battery


def _battery_tensors(
    battery: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if len(battery) < 2:
        return None
    z0 = torch.cat([b[0] for b in battery], dim=0).to(device)
    acts = torch.cat([b[1] for b in battery], dim=0).to(device)
    z1 = torch.cat([b[2] for b in battery], dim=0).to(device)
    return z0, acts, z1


def _battery_readouts(
    agent: REEAgent, tensors: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Dict[str, float]:
    """Both efficacy metrics on the SAME frozen battery. UNGATED telemetry.

    mse    -- per-element reconstruction error, 701b:534-552 _frozen_probe_pe.
              This is the quantity INV-063's what_would_answer names.
    infonce -- the SD-056 objective compute_e2_world_loss actually minimises,
              called exactly as the trainer's call site does (min_batch_classes=1).
    """
    out = {"mse": float("nan"), "infonce": float("nan")}
    if tensors is None:
        return out
    z0, acts, z1 = tensors
    with torch.no_grad():
        pred = agent.e2.world_forward(z0, acts)
        out["mse"] = float((pred - z1).pow(2).mean().item())
        try:
            loss = agent.e2.world_forward_contrastive_loss(
                z_world_0=z0, actions=acts, z_world_1_targets=z1,
                min_batch_classes=1, simulation_mode=False,
            )
            out["infonce"] = float(loss.detach().item()) if torch.is_tensor(loss) else float("nan")
        except (RuntimeError, ValueError):
            out["infonce"] = float("nan")
    return out


def _rel_improvement(pre: float, post: float) -> float:
    """(pre - post) / pre. NaN when pre is NaN or ~0 (no scale to normalise by)."""
    if pre != pre or post != post or abs(pre) < EPS:
        return float("nan")
    return (pre - post) / pre


def run_cell(arm: str, seed: int, waking_episodes: int, steps: int,
             battery_size: int) -> Dict:
    """One (arm, seed) cell: real waking rollout -> frozen battery -> ONE real
    sleep-integrated cycle -> world-head delta + across-sleep battery readouts."""
    print(f"Seed {seed} Condition {arm}", flush=True)
    on = arm == ARM_ON

    env = _make_env(seed)
    agent = _make_agent(env, on)
    device = agent.device
    assert agent.sleep_loop is not None, "use_sleep_loop=True must build sleep_loop"

    # P4 -- parameter scoping, asserted from the live module tree, not from config.
    world_params = _world_head_params(agent)
    nonworld_ids = {id(p) for p in _non_world_e2_params(agent)}
    world_ids = {id(p) for p in world_params}
    all_e2_ids = {id(p) for p in agent.e2.parameters()}
    scoping_overlap = len(world_ids & nonworld_ids)
    world_subset_of_e2 = world_ids <= all_e2_ids

    rng = torch.Generator(device="cpu").manual_seed(seed)

    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()

    # ---------------- waking rollout: populate the REAL replay buffers -------
    for ep in range(waking_episodes):
        print(f"  [train] {arm} seed={seed} ep {ep+1}/{waking_episodes}", flush=True)
        for _step in range(steps):
            obs_body = _to_batched(obs_dict["body_state"], device)
            obs_world = _to_batched(obs_dict["world_state"], device)
            obs_harm = obs_dict.get("harm_obs", None)
            if obs_harm is not None:
                obs_harm = _to_batched(obs_harm, device)

            prev_latent = agent._current_latent
            prev_z_self = (
                prev_latent.z_self.detach().clone() if prev_latent is not None else None
            )

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            ticks = agent.clock.advance()
            if ticks.get("e1_tick", False):
                # This is what appends to _world_experience_buffer /
                # _action_experience_buffer (agent.py:5995-5999) -- the two
                # buffers compute_e2_world_loss draws its triples from.
                agent._e1_tick(latent)

            action_idx = int(torch.randint(0, env.action_dim, (1,), generator=rng).item())
            action = torch.zeros(1, env.action_dim, device=device)
            action[0, action_idx] = 1.0
            if prev_z_self is not None:
                agent.record_transition(prev_z_self, action, latent.z_self.detach())
            # RED-TEAM Finding 1 (CONTESTED -> FIXED). Without this, _last_action
            # stays None, so _e1_action_one_hot() (agent.py:11148-11175) appends a
            # ZERO vector on every _e1_tick and the world trainer trains on an
            # all-zero action buffer. world_action_encoder is nn.Linear(a_dim,
            # a_dim) (e2_fast.py:137), so a zero input gives dL/dW = delta (x) x = 0
            # EXACTLY -- its weight matrix, the parameter that makes world_forward
            # action-CONDITIONAL at all, would never move even on the ON arm, and
            # C3's pooled max would hide that behind world_transition's movement.
            # Measured at authoring time: nonzero_fraction 0.0 -> 0.967 and
            # world_action_encoder.weight max|delta| 0.0 -> 0.0077 with this call.
            # Called in BOTH arms, which is the condition record_executed_action's
            # own docstring (agent.py:11195-11199) sets for an ablation, so the
            # arms still differ by the lever alone. Placed AFTER record_transition
            # and BEFORE env.step so the NEXT _e1_tick appends the action that led
            # INTO the state it is appending -- the +1 alignment
            # compute_e2_world_loss documents (agent.py:12500-12506).
            agent.record_executed_action(action)

            _, harm_signal, done, info, obs_dict = env.step(action)
            agent.update_residue(float(harm_signal) if float(harm_signal) < 0 else 0.0)
            if done:
                _, obs_dict = env.reset()
                agent.e1.reset_hidden_state()

    n_world_buffer = len(agent._world_experience_buffer)
    n_action_buffer = len(agent._action_experience_buffer)
    n_pairs = min(n_world_buffer, n_action_buffer) - 1

    # ---------------- frozen held-out battery, captured PRE-sleep ------------
    # Both arms capture it at the identical point with the identical procedure,
    # so the arms stay RNG-matched up to the cycle itself.
    battery = _sample_probe_battery(agent, seed, battery_size)
    battery_tensors = _battery_tensors(battery, device)
    n_battery = len(battery)

    # ---------------- P2/P3 -- the trainer's OWN loss and gradient, from output
    # Run in BOTH arms so the global-RNG state entering force_cycle is matched;
    # the OFF arm simply never uses the closure. Grads are cleared after, and
    # cross_module_consolidation.py:178 zero_grad()s before its own backward, so
    # this probe cannot leak an update into either arm.
    world_loss_probe = float("nan")
    world_grad_probe = float("nan")
    try:
        _loss = agent.compute_e2_world_loss(batch_size=CMC_BATCH)
        world_loss_probe = float(_loss.detach().item())
        for p in world_params:
            p.grad = None
        if _loss.requires_grad and world_loss_probe != 0.0:
            _loss.backward()
            world_grad_probe = max(
                (float(p.grad.abs().max().item()) if p.grad is not None else 0.0)
                for p in world_params
            ) if world_params else 0.0
        else:
            # The exactly-zero graph-anchored sentinel: no usable replay content.
            world_grad_probe = 0.0
    except (RuntimeError, ValueError):
        world_loss_probe = 0.0
        world_grad_probe = 0.0
    finally:
        for p in world_params:
            p.grad = None

    # The substrate's OWN non-vacuity detector for this exact failure mode
    # (agent.py e1_action_buffer_stats: "nonzero_fraction near 0 means the buffer
    # is full of zero actions and an action-conditioned arm is conditioning on
    # nothing"). Asserted from output, per cell, in both arms.
    try:
        action_nonzero_fraction = float(
            agent.e1_action_buffer_stats().get("nonzero_fraction", 0.0))
    except (AttributeError, RuntimeError, ValueError):
        action_nonzero_fraction = float("nan")

    ns = f"{arm}:{seed}"
    try:
        preconditions = p0_readiness_gate([
            {"name": f"{ns}:world_replay_pairs_populated",
             "measured": float(n_pairs), "threshold": WORLD_BUFFER_FLOOR,
             "direction": "lower",
             "control": "real waking rollout via _e1_tick, not synthetic buffers"},
            # THE SENTINEL GUARD. compute_e2_world_loss returns an exactly-zero
            # graph-anchored value when n_pairs < 2, and consolidate()'s contract
            # says such a step does NOT count as touching the module. Without
            # this, a world-head delta of 0 with the lever ON would be a
            # statement about replay-buffer length, not about the trainer.
            {"name": f"{ns}:world_loss_non_sentinel",
             "measured": world_loss_probe, "threshold": 0.0,
             "direction": "lower", "comparator": ">",
             "control": "probed on the same real buffers the sleep cycle draws from"},
            {"name": f"{ns}:world_grad_nonzero",
             "measured": world_grad_probe, "threshold": 0.0,
             "direction": "lower", "comparator": ">",
             "control": "backward() from the realised world loss onto the two world heads"},
            {"name": f"{ns}:world_param_scoping_one_directional",
             "measured": float(scoping_overlap), "threshold": 0.0,
             "direction": "upper",
             "control": "the e2_world set must contain NO non-world e2 parameter "
                        "(the build's actual guarantee; the e2 list is a SUPERSET, "
                        "so literal disjointness is false by construction)"},
            # RED-TEAM Finding 1 guard: a zero-action replay buffer makes the
            # world trainer's action-conditioning channel structurally untrainable.
            # Floor 0.5, not >0: one zero entry per episode start is CORRECT
            # ("the honest encoding of no action led to this state"), so the
            # healthy value is ~(n-1)/n, measured 0.967 at authoring time, while a
            # driver that never calls record_executed_action reads exactly 0.0.
            {"name": f"{ns}:action_buffer_non_vacuous",
             "measured": action_nonzero_fraction, "threshold": 0.5,
             "direction": "lower",
             "control": "agent.e1_action_buffer_stats(), the substrate's own "
                        "non-vacuity detector for this failure mode"},
            {"name": f"{ns}:frozen_battery_populated",
             "measured": float(n_battery), "threshold": 2.0, "direction": "lower",
             "control": "held-out env (seed+9973) + fixed action policy, pure read"},
        ])
    except P0NotReady as e:
        preconditions = e.preconditions

    # ---------------- pre-sleep efficacy readouts ---------------------------
    pre = _battery_readouts(agent, battery_tensors)

    # ---------------- fire ONE real sleep-integrated cycle ------------------
    world_before = _snapshot(world_params)
    e2_before = _snapshot(list(agent.e2.parameters()))
    e1_before = _snapshot(list(agent.e1.parameters()))
    metrics = agent.sleep_loop.force_cycle(agent) or {}
    world_after = _snapshot(world_params)
    e2_after = _snapshot(list(agent.e2.parameters()))
    e1_after = _snapshot(list(agent.e1.parameters()))

    # ---------------- post-sleep efficacy readouts (SAME battery tensors) ----
    post = _battery_readouts(agent, battery_tensors)

    # Prove the cycle actually RAN rather than short-circuiting -- a key
    # phase_manager._run_cycle merges unconditionally at the end of every
    # completed cycle, independent of any internal gate. Without this, the OFF
    # arm's bit-identical world-head delta would be meaningless.
    sleep_cycle_fired = "post_sleep_z_goal_retention" in metrics
    preconditions = list(preconditions) + [{
        "name": f"{ns}:sleep_cycle_fired", "measured": float(sleep_cycle_fired),
        "threshold": 1.0, "direction": "lower",
        "control": "post_sleep_z_goal_retention is merged unconditionally at the "
                   "end of _run_cycle",
        "met": sleep_cycle_fired,
    }]

    # Per-module deltas. C3's pooled max is the ratified form, but pooling is
    # exactly what let a structurally-frozen action-conditioning weight hide
    # behind world_transition's movement (red-team Finding 1), so every world
    # head is reported separately and the action channel gets its own criterion.
    per_module_delta = {}
    for _mod_name in WORLD_HEAD_MODULES:
        _mod = getattr(agent.e2, _mod_name, None)
        if _mod is None:
            continue
        for _pn, _p in _mod.named_parameters():
            _key = f"{_mod_name}.{_pn}"
            _idx = [id(q) for q in world_params].index(id(_p))
            per_module_delta[_key] = _max_abs_delta([world_before[_idx]], [world_after[_idx]])

    world_finite = _all_finite(world_params)
    e2_finite = _all_finite(list(agent.e2.parameters()))
    e1_finite = _all_finite(list(agent.e1.parameters()))
    world_delta = _max_abs_delta(world_before, world_after) if world_finite else float("nan")
    e2_delta = _max_abs_delta(e2_before, e2_after) if e2_finite else float("nan")
    e1_delta = _max_abs_delta(e1_before, e1_after) if e1_finite else float("nan")

    cmc_keys = sorted(k for k in metrics if k.startswith("cross_module_consolidation_"))
    updates_e2_world = float(metrics.get("cross_module_consolidation_updates_e2_world", 0.0))
    updates_e2 = float(metrics.get("cross_module_consolidation_updates_e2", 0.0))
    updates_e1 = float(metrics.get("cross_module_consolidation_updates_e1", 0.0))
    replay_share = float(metrics.get("cross_module_consolidation_cross_module_replay_share", float("nan")))
    # The e2_world KEY must be PRESENT on ON and ABSENT on OFF -- read from the
    # merged metrics, not from the config that set the lever.
    e2_world_key_present = "cross_module_consolidation_updates_e2_world" in metrics

    print(
        f"  {arm} seed={seed} n_pairs={n_pairs} battery={n_battery} "
        f"world_loss={world_loss_probe:.6g} world_grad={world_grad_probe:.6g} "
        f"act_nonzero={action_nonzero_fraction:.3g} "
        f"fired={sleep_cycle_fired} e2_world_key={e2_world_key_present} "
        f"updates_e2_world={updates_e2_world:.0f} world_delta={world_delta:.6g} "
        f"wenc_w_delta={per_module_delta.get('world_action_encoder.weight', float('nan')):.6g} "
        f"e2_delta={e2_delta:.6g} e1_delta={e1_delta:.6g}",
        flush=True,
    )
    print(
        f"  {arm} seed={seed} [efficacy, UNGATED] mse {pre['mse']:.6g} -> {post['mse']:.6g} "
        f"(rel {_rel_improvement(pre['mse'], post['mse']):.6g})  "
        f"infonce {pre['infonce']:.6g} -> {post['infonce']:.6g} "
        f"(rel {_rel_improvement(pre['infonce'], post['infonce']):.6g})",
        flush=True,
    )

    ready = bool(
        n_pairs >= WORLD_BUFFER_FLOOR
        and world_loss_probe > 0.0
        and world_grad_probe > 0.0
        and scoping_overlap == 0
        and n_battery >= 2
        and sleep_cycle_fired
        and action_nonzero_fraction >= 0.5
    )
    cell_ok = bool(ready and world_finite and e2_finite and e1_finite)
    print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)

    return {
        "arm": arm,
        "seed": seed,
        "n_world_buffer": n_world_buffer,
        "n_action_buffer": n_action_buffer,
        "n_world_replay_pairs": n_pairs,
        "n_frozen_battery": n_battery,
        "world_loss_probe": world_loss_probe,
        "world_grad_probe": world_grad_probe,
        "world_param_count": len(world_params),
        "non_world_e2_param_count": len(nonworld_ids),
        "world_param_scoping_overlap": scoping_overlap,
        "world_subset_of_e2_params": bool(world_subset_of_e2),
        "sleep_cycle_fired": sleep_cycle_fired,
        "ready": ready,
        "cmc_metric_keys": cmc_keys,
        "cmc_metrics_merged": bool(cmc_keys),
        "e2_world_key_present": bool(e2_world_key_present),
        "updates_e2_world": updates_e2_world,
        "updates_e2": updates_e2,
        "updates_e1": updates_e1,
        "cross_module_replay_share": replay_share,
        "world_head_max_abs_delta": world_delta,
        "world_head_per_param_max_abs_delta": per_module_delta,
        "world_action_encoder_weight_delta": float(
            per_module_delta.get("world_action_encoder.weight", float("nan"))),
        "world_transition_delta": max(
            [v for k, v in per_module_delta.items() if k.startswith("world_transition")]
            or [float("nan")]),
        "action_buffer_nonzero_fraction": action_nonzero_fraction,
        "e2_max_abs_delta": e2_delta,
        "e1_max_abs_delta": e1_delta,
        "world_finite": world_finite,
        "e2_finite": e2_finite,
        "e1_finite": e1_finite,
        # --- ungated efficacy telemetry (both metrics, same frozen battery) ---
        "frozen_battery_mse_pre": pre["mse"],
        "frozen_battery_mse_post": post["mse"],
        "frozen_battery_mse_rel_improvement": _rel_improvement(pre["mse"], post["mse"]),
        "frozen_battery_infonce_pre": pre["infonce"],
        "frozen_battery_infonce_post": post["infonce"],
        "frozen_battery_infonce_rel_improvement": _rel_improvement(pre["infonce"], post["infonce"]),
        "readiness_preconditions": preconditions,
        "agent": agent,
    }


def _finite_or_none(x: float) -> Optional[float]:
    """Flat readout hygiene: drop non-finite rather than emit NaN (a NaN IS
    numeric to the indexer and pollutes a delta; an absent key correctly reads
    as unmeasured)."""
    return float(x) if x == x and abs(x) != float("inf") else None


def _mean(xs: List[float]) -> float:
    vals = [x for x in xs if x == x]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def main(dry_run: bool = False):
    """Returns (outcome, manifest_path). manifest_path is None on dry-run."""
    waking_episodes = 1 if dry_run else WAKING_EPISODES
    # The dry-run must still clear WORLD_BUFFER_FLOOR (17 pairs) so the smoke
    # exercises the real criteria path, not just the FORK path. At
    # e1_steps_per_tick=1, 30 steps -> ~30 entries -> 29 pairs.
    steps = 30 if dry_run else STEPS_PER_EPISODE
    seeds = (SEEDS[0],) if dry_run else SEEDS
    battery = 16 if dry_run else BATTERY_SIZE

    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run}) arms={ARMS} seeds={seeds} "
          f"waking_episodes={waking_episodes} steps={steps} cmc_steps={CMC_STEPS} "
          f"battery={battery}", flush=True)
    t0 = time.time()

    rows: Dict[tuple, Dict] = {}
    arm_results: List[Dict] = []
    agents_seen: List[REEAgent] = []
    for arm in ARMS:
        for seed in seeds:
            cfg_slice = {
                "experiment": EXPERIMENT_TYPE,
                "arm": arm,
                "waking_episodes": waking_episodes,
                "steps_per_episode": steps,
                "cmc_steps": CMC_STEPS,
                "cmc_lr": CMC_LR,
                "cmc_batch": CMC_BATCH,
                "grid_size": GRID_SIZE,
                "num_hazards": N_HAZARDS,
                "num_resources": N_RESOURCES,
                "self_dim": SELF_DIM,
                "world_dim": WORLD_DIM,
                "battery_size": battery,
                "use_sleep_world_forward_consolidation": arm == ARM_ON,
            }
            with arm_cell(seed, config_slice=cfg_slice, script_path=Path(__file__),
                          extra_ineligible_reasons=[
                              "diagnostic_substrate_validation_no_reuse"]) as cell:
                row = run_cell(arm, seed, waking_episodes, steps, battery)
                agent_obj = row.pop("agent")
                cell.stamp(row)
            agents_seen.append(agent_obj)
            rows[(arm, seed)] = row
            arm_results.append(row)

    elapsed = time.time() - t0

    on_rows = [rows[(ARM_ON, s)] for s in seeds]
    off_rows = [rows[(ARM_OFF, s)] for s in seeds]

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    unready_cells = [
        (r["arm"], r["seed"]) for r in on_rows + off_rows
        if not r["ready"] or not r["world_finite"] or not r["e2_finite"] or not r["e1_finite"]
    ]

    # --- ungated efficacy aggregate (telemetry only; routes no verdict) ------
    efficacy = {}
    for label, rws in (("on", on_rows), ("off", off_rows)):
        efficacy[label] = {
            "mse_pre_mean": _mean([r["frozen_battery_mse_pre"] for r in rws]),
            "mse_post_mean": _mean([r["frozen_battery_mse_post"] for r in rws]),
            "mse_rel_improvement_mean": _mean(
                [r["frozen_battery_mse_rel_improvement"] for r in rws]),
            "mse_rel_improvement_per_seed": [
                r["frozen_battery_mse_rel_improvement"] for r in rws],
            "infonce_pre_mean": _mean([r["frozen_battery_infonce_pre"] for r in rws]),
            "infonce_post_mean": _mean([r["frozen_battery_infonce_post"] for r in rws]),
            "infonce_rel_improvement_mean": _mean(
                [r["frozen_battery_infonce_rel_improvement"] for r in rws]),
            "infonce_rel_improvement_per_seed": [
                r["frozen_battery_infonce_rel_improvement"] for r in rws],
            "seeds": [r["seed"] for r in rws],
        }
    efficacy["gated"] = False
    efficacy["off_arm_zero_is_structural"] = True
    efficacy["off_arm_note"] = (
        "The OFF arm's rel_improvement is EXACTLY 0 by STRUCTURAL IDENTITY, not by "
        "measurement: world_forward depends only on the two world heads "
        "(e2_fast.py:201-221), the OFF arm applies no gradient to them, and the "
        "SAME captured battery tensors are re-evaluated -- so pre and post are "
        "bitwise identical. Do NOT read it as a measured null effect or as a "
        "control against which the ON arm's change is a difference-in-differences."
    )
    efficacy["action_distribution_note"] = (
        "The battery's actions are real one-hots while the replay triples the "
        "trainer draws carry one zero-action entry per episode start "
        "(_e1_action_one_hot's honest encoding of 'no action led to this state'); "
        "measured action_buffer_nonzero_fraction is reported per cell."
    )
    efficacy["note"] = (
        "UNGATED TELEMETRY -- routes no verdict. No threshold for across-sleep "
        "held-out world-forward prediction error is pre-registered anywhere in "
        "the corpus, and the one precedent (v3_exq_701b MIN_REL_CONV_DROP=0.10) "
        "gates 60 episodes of dedicated Adam training, not 8 consolidation steps "
        "at lr 1e-3. Both metrics are recorded because the trainer minimises "
        "SD-056 InfoNCE while INV-063's DV is per-element MSE reconstruction "
        "error, and InfoNCE is insensitive to a global scale/shift of the "
        "prediction -- so it can improve while MSE does not. This run's "
        "PASS/FAIL turns on C1-C4 (liveness) alone."
    )

    base_manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "sleep_driver_pattern": (
            "manual-cycle-loop (run_sleep_cycle() called once per cycle in a "
            "dedicated N_CYCLES wake-sleep-test loop)"
        ),
        "timestamp_utc": ts,
        "started_utc": datetime.utcnow().isoformat() + "Z",
        "arm_results": arm_results,
        "inv063_leg_b_dv_identity": (
            "E2, not E1. E1DeepPredictor has no world_forward head (every hit in "
            "e1_deep.py is a comment about E2's), and the V3-EXQ-701b/701c "
            "frozen-probe instrument INV-063's what_would_answer names computes "
            "agent.e2.world_forward (701b:547, 701c:554). claims.yaml's "
            "'Leg B (E1 world-model updating)' label is prose inherited from the "
            "four-function description taxonomy; a label correction is owed to "
            "/governance and does not affect this run."
        ),
        "frozen_battery_efficacy": efficacy,
    }

    def _write_manifest(manifest: Dict):
        if dry_run:
            print(f"[{EXPERIMENT_TYPE}] dry-run complete; not writing manifest.")
            return None
        out_path = write_flat_manifest(
            manifest,
            dry_run=False,
            config=manifest.get("config"),
            seeds=list(seeds),
            script_path=Path(__file__),
            agent=agents_seen,
        )
        print(f"Result written to: {out_path}")
        return str(out_path)

    run_config = {
        "arms": list(ARMS),
        "seeds": list(seeds),
        "waking_episodes": waking_episodes,
        "steps_per_episode": steps,
        "grid_size": GRID_SIZE,
        "num_hazards": N_HAZARDS,
        "num_resources": N_RESOURCES,
        "self_dim": SELF_DIM,
        "world_dim": WORLD_DIM,
        "cmc_steps": CMC_STEPS,
        "cmc_lr": CMC_LR,
        "cmc_batch": CMC_BATCH,
        "cmc_schedule": "interleaved",
        "world_buffer_floor": WORLD_BUFFER_FLOOR,
        "battery_size": battery,
        "lever": "use_sleep_world_forward_consolidation",
    }

    if unready_cells:
        reason = f"substrate not ready in cell(s): {unready_cells}"
        print(f"\n[{EXPERIMENT_TYPE}] -> substrate_not_ready_requeue: {reason}")
        manifest = dict(base_manifest)
        manifest.update({
            "outcome": "FAIL",
            "result": "FAIL",
            "evidence_direction": "inconclusive",
            "evidence_direction_note": (
                "Diagnostic substrate-readiness precondition unmet or a non-finite "
                "parameter appeared after a sleep-integrated cycle; not a MECH-423 "
                "or INV-063 finding. " + reason
            ),
            "non_degenerate": False,
            "degeneracy_reason": (
                "substrate_not_ready: " + reason + (
                    " NOTE: at least one cell carried a NON-FINITE parameter after "
                    "the cycle -- that is numerical DIVERGENCE, not merely an unmet "
                    "readiness precondition, and should be read as such."
                    if any(not (r["world_finite"] and r["e2_finite"] and r["e1_finite"])
                           for r in on_rows + off_rows) else "")),
            "interpretation": {
                "label": "substrate_not_ready_requeue",
                "preconditions": [
                    p for r in (on_rows + off_rows) for p in r["readiness_preconditions"]
                ],
                "criteria_non_degenerate": {},
            },
            "readout": {"substrate_ready": 0, "overall_pass": 0},
            "config": run_config,
            "elapsed_seconds": elapsed,
        })
        return "FAIL", _write_manifest(manifest)

    # ---- load-bearing criteria --------------------------------------------
    min_updates_e2_world = min(r["updates_e2_world"] for r in on_rows)
    min_world_delta_on = min(r["world_head_max_abs_delta"] for r in on_rows)
    max_world_delta_off = max(r["world_head_max_abs_delta"] for r in off_rows)
    all_merged = all(r["cmc_metrics_merged"] for r in on_rows + off_rows)
    on_key_all = all(r["e2_world_key_present"] for r in on_rows)
    off_key_none = not any(r["e2_world_key_present"] for r in off_rows)

    c1_metrics_merged = bool(all_merged)
    c2_e2_world_touched = bool(min_updates_e2_world >= 1.0)
    c3_world_delta_positive_on = bool(min_world_delta_on > 0.0)
    c4_world_delta_zero_off = bool(max_world_delta_off == 0.0)
    c5_e2_world_key_arm_asymmetric = bool(on_key_all and off_key_none)
    # C6 -- the ACTION-CONDITIONING channel specifically. This is not an addition
    # to the ratified target but a faithful reading of it: the substrate_queue
    # entry says "move E2.world_forward parameters", and world_action_encoder.weight
    # IS one. C3's pooled max was the under-specification -- red-team Finding 1
    # measured that weight frozen at EXACTLY 0.0 while C3 still passed on
    # world_transition alone. C6 is also what makes C3 non-tautological: given the
    # readiness gate, Adam moves any parameter with a non-zero gradient by ~lr, so
    # a pooled "something moved" carries little information, whereas this channel
    # genuinely failed under the pre-fix driver and would fail again on any
    # regression of the action-buffer wiring.
    min_wenc_weight_delta_on = min(
        r["world_action_encoder_weight_delta"] for r in on_rows)
    max_wenc_weight_delta_off = max(
        r["world_action_encoder_weight_delta"] for r in off_rows)
    c6_action_channel_moves_on = bool(min_wenc_weight_delta_on > 0.0)

    criteria = [
        {"name": "C1_cross_module_consolidation_metrics_merged", "load_bearing": True,
         "passed": c1_metrics_merged, "measured": float(all_merged), "threshold": 1.0,
         "comparator": ">="},
        {"name": "C2_e2_world_touched_under_interleaved_schedule", "load_bearing": True,
         "passed": c2_e2_world_touched, "measured": min_updates_e2_world,
         "threshold": 1.0, "comparator": ">="},
        {"name": "C3_world_head_delta_positive_on", "load_bearing": True,
         "passed": c3_world_delta_positive_on, "measured": min_world_delta_on,
         "threshold": 0.0, "comparator": ">"},
        {"name": "C4_off_arm_world_head_bit_identical_negative_control",
         "load_bearing": True, "passed": c4_world_delta_zero_off,
         "measured": max_world_delta_off, "threshold": 0.0, "comparator": "<="},
        {"name": "C5_e2_world_metric_key_present_on_absent_off", "load_bearing": True,
         "passed": c5_e2_world_key_arm_asymmetric,
         "measured": float(on_key_all and off_key_none), "threshold": 1.0,
         "comparator": ">="},
        {"name": "C6_action_conditioning_weight_moves_on", "load_bearing": True,
         "passed": c6_action_channel_moves_on, "measured": min_wenc_weight_delta_on,
         "threshold": 0.0, "comparator": ">"},
    ]
    criteria_non_degenerate = {
        "C1_cross_module_consolidation_metrics_merged": c1_metrics_merged,
        "C2_e2_world_touched_under_interleaved_schedule": c2_e2_world_touched,
        # Non-degenerate BECAUSE the sentinel guard passed: a non-zero realised
        # world loss AND a non-zero gradient were measured on every ON cell, so a
        # world-head delta of 0 could not have been a buffer-length artifact.
        "C3_world_head_delta_positive_on": bool(
            c3_world_delta_positive_on
            and all(r["world_loss_probe"] > 0.0 and r["world_grad_probe"] > 0.0
                    for r in on_rows)),
        # The negative control's value is non-degenerate exactly because it
        # measures a genuine mechanistic absence (no loss closure -> no gradient)
        # while the SAME consolidation pass runs in both arms -- not a
        # coincidental tie between two live computations, and not the trivial
        # "nothing ran" case (1026's OFF arm disabled consolidation entirely).
        "C4_off_arm_world_head_bit_identical_negative_control": bool(
            all(r["sleep_cycle_fired"] and r["updates_e2"] >= 1.0 for r in off_rows)),
        "C5_e2_world_metric_key_present_on_absent_off": c5_e2_world_key_arm_asymmetric,
        # Non-degenerate BECAUSE the action buffer was measured non-vacuous on
        # every ON cell: with an all-zero action buffer this channel is
        # structurally untrainable and a pass would be impossible, so a pass here
        # is a real measurement rather than an artifact of the wiring.
        "C6_action_conditioning_weight_moves_on": bool(
            c6_action_channel_moves_on
            and all(r["action_buffer_nonzero_fraction"] >= 0.5 for r in on_rows)),
    }

    all_pass = all(c["passed"] for c in criteria)
    outcome = "PASS" if all_pass else "FAIL"
    evidence_direction = "supports" if all_pass else "inconclusive"
    label = (
        "e2_world_forward_sleep_trainer_live" if all_pass
        else "e2_world_forward_sleep_trainer_not_confirmed"
    )
    note = (
        f"The 2026-09-17 lever use_sleep_world_forward_consolidation (ree-v3 4610133) "
        f"moves E2's world-forward heads through a REAL sleep cycle: min "
        f"max|delta| over world_transition + world_action_encoder across "
        f"{len(on_rows)} seeds = {min_world_delta_on:.6g} (was EXACTLY 0.0 in "
        f"V3-EXQ-1026); min updates_e2_world = {min_updates_e2_world:.0f}. "
        f"Negative control confirms attribution: OFF-arm max|delta| = "
        f"{max_world_delta_off:.6g} with the SAME consolidation pass running "
        f"(updates_e2 >= 1 on every OFF cell), so the movement is the lever's, "
        f"not another offline writer's. Diagnostic substrate-readiness result: it "
        f"makes INV-063 leg B's frozen held-out world-forward prediction-error DV "
        f"NON-DEGENERATE; it does NOT adjudicate INV-063 (the four-arm intake "
        f"ladder is a separate design). Across-sleep frozen-battery efficacy is "
        f"recorded UNGATED in frozen_battery_efficacy -- see its note."
    ) if all_pass else (
        f"One or more load-bearing criteria failed: "
        f"{[c['name'] for c in criteria if not c['passed']]}. See criteria[] for "
        f"measured/threshold detail."
    )

    print(f"\n[{EXPERIMENT_TYPE}] verdict:")
    for c in criteria:
        print(f"  {c['name']}: passed={c['passed']} measured={c['measured']:.6g} "
              f"threshold={c['threshold']}")
    print(f"  -> {label} ({outcome}); elapsed={elapsed:.1f}s")

    flat = {
        "c1_metrics_merged": int(c1_metrics_merged),
        "c2_e2_world_touched": int(c2_e2_world_touched),
        "c3_world_delta_positive_on": int(c3_world_delta_positive_on),
        "c4_off_arm_zero_delta": int(c4_world_delta_zero_off),
        "c5_e2_world_key_arm_asymmetric": int(c5_e2_world_key_arm_asymmetric),
        "min_world_head_max_abs_delta_on": min_world_delta_on,
        "max_world_head_max_abs_delta_off": max_world_delta_off,
        "min_updates_e2_world_on": min_updates_e2_world,
        "c6_action_conditioning_weight_moves_on": int(c6_action_channel_moves_on),
        "min_action_encoder_weight_delta_on": min_wenc_weight_delta_on,
        "max_action_encoder_weight_delta_off": max_wenc_weight_delta_off,
        "min_action_buffer_nonzero_fraction_on": min(
            r["action_buffer_nonzero_fraction"] for r in on_rows),
        "overall_pass": int(all_pass),
    }
    for k, v in (
        ("efficacy_mse_rel_improvement_on", efficacy["on"]["mse_rel_improvement_mean"]),
        ("efficacy_mse_rel_improvement_off", efficacy["off"]["mse_rel_improvement_mean"]),
        ("efficacy_infonce_rel_improvement_on", efficacy["on"]["infonce_rel_improvement_mean"]),
        ("efficacy_infonce_rel_improvement_off", efficacy["off"]["infonce_rel_improvement_mean"]),
        ("efficacy_mse_pre_on", efficacy["on"]["mse_pre_mean"]),
        ("efficacy_mse_post_on", efficacy["on"]["mse_post_mean"]),
        ("efficacy_infonce_pre_on", efficacy["on"]["infonce_pre_mean"]),
        ("efficacy_infonce_post_on", efficacy["on"]["infonce_post_mean"]),
    ):
        fv = _finite_or_none(v)
        if fv is not None:
            flat[k] = fv

    manifest = dict(base_manifest)
    manifest.update({
        "outcome": outcome,
        "result": outcome,
        "evidence_direction": evidence_direction,
        "evidence_direction_note": note,
        "non_degenerate": bool(all(criteria_non_degenerate.values())),
        "degeneracy_reason": (
            None if all(criteria_non_degenerate.values())
            else "degenerate criteria: " + ", ".join(
                k for k, v in criteria_non_degenerate.items() if not v)),
        "interpretation": {
            "label": label,
            "criteria": criteria,
            "combination_rule": (
                "PASS iff ALL of C1..C6 pass (plain AND, no OR/any() branching). "
                "The frozen-battery efficacy readouts are UNGATED and route no "
                "verdict."
            ),
            "criteria_non_degenerate": criteria_non_degenerate,
            "preconditions": [
                p for r in (on_rows + off_rows) for p in r["readiness_preconditions"]
            ],
        },
        "readout": flat,
        "e2_world_forward_sleep_trainer": {
            "min_world_head_max_abs_delta_on": min_world_delta_on,
            "max_world_head_max_abs_delta_off": max_world_delta_off,
            "min_updates_e2_world_on": min_updates_e2_world,
            "world_head_modules": list(WORLD_HEAD_MODULES),
            "lever": "use_sleep_world_forward_consolidation",
            "build_commit": "4610133",
            "cmc_steps": CMC_STEPS,
            "cmc_lr": CMC_LR,
            "cmc_batch": CMC_BATCH,
            "prior_recorded_value": (
                "V3-EXQ-1026: E2.world_forward parameter delta == 0.0 exactly on "
                "every seed and arm (substrate_queue failure_record, confirmed "
                "autopsy Step 7c red-team probe)."
            ),
        },
        "config": run_config,
        "elapsed_seconds": elapsed,
    })
    return outcome, _write_manifest(manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Smoke run, no manifest.")
    args = parser.parse_args()
    _outcome, _manifest_path = main(dry_run=args.dry_run)
    _outcome_clean = str(_outcome).upper() if str(_outcome).upper() in ("PASS", "FAIL") else "FAIL"
    emit_outcome(outcome=_outcome_clean, manifest_path=_manifest_path, dry_run=args.dry_run)
    sys.exit(0)
