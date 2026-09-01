"""V3-EXQ-967: why were V3-EXQ-966's two arms behaviourally identical? (confirmer)

A minimal INSTRUMENT-LOCALISING diagnostic. It tests no claim -- `claim_ids` is
deliberately empty -- and its whole job is to separate two live explanations that
the V3-EXQ-966 autopsy left open, cheaply, BEFORE any substrate work is specified
and BEFORE any 966a redesign is queued.

THE FACT TO EXPLAIN. Confirmed autopsy failure_autopsy_966-436g-951-959-822d-cluster
_2026-08-30 (REE_assembly 40dd17331e), governance-ratified 2026-08-30 (54dbe477be),
found V3-EXQ-966's `block_contacts` BIT-IDENTICAL between ARM_FIXED_VALUE and
ARM_SHUFFLED_VALUE on 4/4 seeds x 4/4 blocks -- including blocks 3 and 4, where the
two arms assign the HIGH amplitude to DIFFERENT resource types. Because `frac_high`
is (contacts of the block's current high type)/(total), that bit-identity forces
frac_high_SHUFFLED = 1 - frac_high_FIXED exactly, so `fixed_post + shuffled_post`
summed to precisely 1.000 on every seed and the equivalence test read a pure
RELABELLING as evidence. Both per-claim directions were withdrawn to
non_contributory.

WHY THE MECHANISM IS STILL OPEN. An earlier revision of that autopsy explained the
inertness by asserting the amplitude has no observable route to the policy, since
`agent_health` is ceiling-clamped at NUM_HAZARDS=0 and `agent_energy` is
floor-clamped. THE AUTOPSY'S OWN ADVERSARIAL PASS REFUTED THAT AND IT WAS WITHDRAWN:
energy resets to 1.0 each episode and decays 0.01/step
(causal_grid_world.py:2691), so it spends most of a 120-step episode strictly
between the clamps, where a contact's bump lands UNCLAMPED and amplitude-scaled --
`agent_energy = min(1.0, agent_energy + contact_benefit*0.5)` at :2176 with
`contact_benefit = resource_benefit * amp` at :2133, i.e. +0.15 (HIGH, amp 1.0) vs
+0.0225 (LOW, amp 0.15) -- into body_state[3] (:3627), an OBSERVED channel. The
substrate proposal was withdrawn and the mechanism is UNRESOLVED. Two explanations
remain, with opposite implications:

  H-inert-in-fact       something prevents the amplitude difference from reaching
                        behaviour despite the open energy channel. The run measured
                        nothing and the fix is ENVIRONMENTAL.
  H-insensitive-policy  the amplitude IS applied, energy genuinely differs, and E3
                        selection is nonetheless unmoved by it. That is a
                        substantive and more troubling finding about the selection
                        pathway's sensitivity to body state.

The user ratified at the autopsy's Step 8 gate (2026-08-30T15:41:56Z, user present)
that "the cheap confirmer runs FIRST -- one seed, blocks 3-4, both arms, logging
amplitudes, agent_energy and per-step actions -- before any substrate work is
specified or 966a is queued." This is that confirmer. It does NOT redesign 966 and
it does NOT specify a substrate build.

HOW IT DISCRIMINATES. One seed, both arms, 966's exact env/agent config and block
schedule, with per-step logging restricted to the two blocks where the arms differ
(0-indexed 2 and 3 = "blocks 3-4"). Blocks 0-1 are still RUN -- they are identical
in both arms and are what puts the agent in the block-3 state -- but not logged.
Five booleans are then read off the matched-step logs, in a fixed chain:

  D1 amplitudes_differ       the live env amplitude tuple differs between arms
  D2 contact_benefit_differs the benefit actually applied at contacts differs
  D3 energy_differs          the agent_energy trajectories differ
  D4 actions_differ          the per-step committed action indices differ
  D5 contacts_differ         block_contacts differ (i.e. 966 does NOT reproduce)

  not D1                  -> manipulation_not_applied            (instrument bug upstream)
  D1, not D2              -> amplitude_not_reaching_consumption
  D1 D2, not D3           -> H_inert_in_fact_energy_route_blocked
  D1 D2 D3, not D4        -> H_insensitive_policy                 (the troubling one)
  D1 D2 D3 D4, not D5     -> behaviour_diverges_contacts_coincide
  D1 D2 D3 D4 D5          -> phenomenon_not_reproduced            (966's bit-identity absent)

Reproducing 966's bit-identity is itself a control: if `block_contacts` are NOT
identical here, the phenomenon did not reproduce and every route label above is
about a different run, which the last branch says explicitly rather than papering
over.

wanting_weight note: V3-EXQ-966 passes `wanting_weight` through
`REEConfig.from_dims(...)`, and a separate open chip
(chip-20260830-exq966-wantingweight-fromdims-dropsite) asserts that is a silent
drop site. Read against source, `from_dims` DOES assign it
(`config.hippocampal.wanting_weight = wanting_weight`, config.py:7926) and it IS a
declared parameter (:6717) -- so that premise looks doubtful. This driver does not
adjudicate that chip. It instead sets the field by from_dims AND asserts the landed
value at runtime as a recorded precondition, so this confirmer is correct whichever
way that chip resolves and never silently inherits a drop.

red-team (fable, predecessor session c1-queue-refill-20260901): CONTESTED
(1 BLOCKING) -> all findings applied. (1) BLOCKING: D2 compared distinct-benefit
SETS and MEANS, which TIE under exact reproduction of 966's bit-identity (both
arms see both tags; equal counts make the means equal), so the run would have
confidently emitted `amplitude_not_reaching_consumption` and made the whole
H-inert vs H-insensitive adjudication unreachable -- D2 is now a MATCHED-STEP
comparison, with the aggregates kept as recorded diagnostics. (2) HIGH: 966
trains e1+latent_stack for 12 P0 warmup episodes (Adam 3e-4, prediction +
resource-encoder loss); this driver rolled the warmup without any optimizer, so
its logged-block agent was untrained-encoder rather than 966's agent -- 966's P0
block is now replicated verbatim. (3) D1 and the applied benefit are now read
from the LIVE env (`env.resource_type_benefit_amplitudes`) rather than from
driver constants, with an independent energy-delta cross-check. (4) energy
divergence onset and extent are recorded, and C2 is non-degenerate only when the
post-divergence window clears MIN_POST_DIVERGENCE_STEPS -- so H_insensitive_policy
cannot fire off a near-zero exposure window. (5) the per-cell verdict print now
keys to MIN_LOGGED_CONTACTS, and the :2131 cite is corrected to :2133.
One further gap, found in this session's own Step 3.5 review rather than by the
red-team: D3 routes on agent_energy, but nothing asserted that a contact can MOVE
agent_energy in this config -- so a flat D3=False could have been an instrument
artifact (wrong attribute, wrong tick, bump not landing) wearing the
H_inert_in_fact substrate label. A `contact_bump_reaches_agent_energy` readiness
precondition now reconstructs contact_benefit from the observable channel alone
and routes a dead channel to substrate_not_ready_requeue instead. That control is
a LIVENESS FLOOR, not a calibration: it does not subtract the action/movement
energy cost at :2707, so its value is not a contact_benefit estimate.

SMOKE OBSERVATION worth carrying into the read (--dry-run, seed 42, 15-step
episodes, 3 logged contacts per arm): the amplitude-derived applied benefits
correctly differ between arms (0.3 vs 0.045 -- D2 True under the matched-step
form, and False under the withdrawn aggregate form, which is the BLOCKING finding
reproducing itself in miniature), while the reconstructed energy bump is the SAME
0.04 in both arms and matches neither amplitude. If that survives the full run it
is direct support for H_inert_in_fact_energy_route_blocked. It is NOT a result --
the dry run is far too short and the reconstruction is uncalibrated -- but it is
the specific number to look at first.

EXPERIMENT_PURPOSE = "diagnostic" -- excluded from governance confidence scoring.
"""

import sys
import math
import time
import argparse
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments._harness import StepHarness  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402

EXPERIMENT_PURPOSE = "diagnostic"
EXPERIMENT_TYPE = "v3_exq_967_mech144_shuffle_inertness_confirmer"

# Each readiness predicate measures exactly the quantity that must hold for its
# discriminator to be readable, rather than a narrower hand-written proxy for it
# (the V3-EXQ-778d failure mode this check guards). `logged_contacts_sufficient` IS
# the sample the consumption route is read from; `energy_spends_time_unclamped` IS
# the autopsy's own contested premise, measured rather than assumed; and
# `wanting_weight_landed` is a config identity. Reachability is not asserted by
# hand-wave: at the shipped 120-step episode, energy decays 0.01/step from a 1.0
# reset (causal_grid_world.py:2691), so it is strictly below the ceiling from
# ~step 15 onward and the 0.50 floor is cleared by construction; a --dry-run at 15
# steps legitimately does NOT clear it, and correctly routes not-ready.
ANCHOR_REACHABILITY_EXEMPT = (
    "predicates are the sample floor, the autopsy's own premise measured directly, "
    "and a config-landing identity -- none is a narrower proxy for the routed "
    "discriminator; energy-unclamped reachability follows from the 0.01/step decay "
    "over a 120-step episode"
)

# --- V3-EXQ-966's config, reproduced exactly (the phenomenon is config-specific) ---
SEED = 42
ARMS = ["ARM_FIXED_VALUE", "ARM_SHUFFLED_VALUE"]

GRID_SIZE = 9
NUM_HAZARDS = 0
NUM_RESOURCES = 6
RESOURCE_BENEFIT = 0.3
HIGH_AMP = 1.0
LOW_AMP = 0.15                      # contact_benefit 0.30 (HIGH) vs 0.045 (LOW)

WARMUP_EPISODES = 12
N_BLOCKS = 4
EPISODES_PER_BLOCK = 10
STEPS_PER_EP = 120

WANTING_WEIGHT = 5.0
P0_LR = 3e-4                        # 966's P0 warmup optimiser LR, replicated
WORLD_DIM = 32
SELF_DIM = 32
ALPHA_WORLD = 0.9

BLOCK_HIGH_IDX = {
    "ARM_FIXED_VALUE":    [0, 0, 0, 0],
    "ARM_SHUFFLED_VALUE": [0, 0, 1, 1],   # single swap between block 2 and 3
}
# 0-indexed blocks where the arms DIFFER -- "blocks 3-4" in the chip's 1-indexed
# wording. Only these are logged per step; 0-1 are run but identical in both arms.
LOGGED_BLOCKS = (2, 3)

# --- Pre-registered thresholds (constants; never derived from this run) ---
MIN_LOGGED_CONTACTS = 4        # below this nothing about consumption is measurable
MIN_UNCLAMPED_FRAC = 0.50      # the autopsy's own claim: energy sits between the
                               # clamps for most of a 120-step episode. If it does
                               # not, D3 cannot speak to the energy route.
ENERGY_TOL = 1e-9              # float tolerance for "energy trajectories differ"
MIN_POST_DIVERGENCE_STEPS = 20 # matched steps that must REMAIN after energy first
                               # diverges. Below this, 'the policy did not respond'
                               # is a statement about an exposure window too short
                               # to respond IN, not about the policy.

DRY_WARMUP_EPISODES = 2
DRY_EPISODES_PER_BLOCK = 2
DRY_STEPS = 15

_ZG = ZGoalStreamAccumulator()


# --------------------------------------------------------------------------- #
# Construction (966-faithful)                                                  #
# --------------------------------------------------------------------------- #

def _amps_for(high_idx: int) -> Tuple[float, float]:
    amps = [LOW_AMP, LOW_AMP]
    amps[high_idx] = HIGH_AMP
    return tuple(amps)


def _make_env(seed: int, high_idx: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES,
        resource_benefit=RESOURCE_BENEFIT,
        resource_respawn_on_consume=True,
        multi_resource_heterogeneity_enabled=True,
        n_resource_types=2,
        resource_type_names=("type_a", "type_b"),
        resource_type_drive_axes=("hunger", "thirst"),
        resource_type_benefit_curves=("sigmoidal_saturating", "sigmoidal_saturating"),
        resource_type_distribution=(1.0, 1.0),
        resource_type_benefit_amplitudes=_amps_for(high_idx),
        per_axis_drive_enabled=False,
        max_episode_steps=STEPS_PER_EP,
    )


def _set_high_idx(env: CausalGridWorldV2, high_idx: int) -> None:
    """Live-mutate the amplitude tuple, exactly as V3-EXQ-966 does.

    `resource_type_benefit_amplitudes` is a plain attribute (no property, no
    __slots__) and `_consume_resource_at` reads it fresh at contact time
    (causal_grid_world.py:2131), so no env rebuild is needed.
    """
    env.resource_type_benefit_amplitudes = _amps_for(high_idx)


def _make_agent(seed: int, env: CausalGridWorldV2) -> REEAgent:
    torch.manual_seed(seed)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=ALPHA_WORLD,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        z_goal_enabled=True,
        wanting_weight=WANTING_WEIGHT,
        use_incentive_token_bank=True,
        incentive_use_per_axis_drive=False,
    )
    config.latent.use_resource_encoder = True
    return REEAgent(config)


def _config_slice(arm: str) -> Dict[str, Any]:
    return {
        "env_kwargs": {
            "size": GRID_SIZE,
            "num_hazards": NUM_HAZARDS,
            "num_resources": NUM_RESOURCES,
            "resource_benefit": RESOURCE_BENEFIT,
            "high_amp": HIGH_AMP,
            "low_amp": LOW_AMP,
            "n_resource_types": 2,
            "per_axis_drive_enabled": False,
            "resource_respawn_on_consume": True,
            "max_episode_steps": STEPS_PER_EP,
        },
        "schedule": {
            "warmup_episodes": WARMUP_EPISODES,
            "n_blocks": N_BLOCKS,
            "episodes_per_block": EPISODES_PER_BLOCK,
            "steps_per_episode": STEPS_PER_EP,
            "logged_blocks": list(LOGGED_BLOCKS),
        },
        "agent": {
            "self_dim": SELF_DIM,
            "world_dim": WORLD_DIM,
            "alpha_world": ALPHA_WORLD,
            "wanting_weight": WANTING_WEIGHT,
            "use_resource_proximity_head": True,
            "resource_proximity_weight": 0.5,
            "z_goal_enabled": True,
            "use_incentive_token_bank": True,
            "incentive_use_per_axis_drive": False,
            "use_resource_encoder": True,
        },
        "readout_constants": {
            "min_logged_contacts": MIN_LOGGED_CONTACTS,
            "min_unclamped_frac": MIN_UNCLAMPED_FRAC,
            "energy_tol": ENERGY_TOL,
            "min_post_divergence_steps": MIN_POST_DIVERGENCE_STEPS,
            "p0_lr": P0_LR,
        },
        "block_high_idx": list(BLOCK_HIGH_IDX[arm]),
        "arm": arm,
    }


# --------------------------------------------------------------------------- #
# Per-cell run                                                                 #
# --------------------------------------------------------------------------- #

def _run_episode(harness: StepHarness, env, steps: int,
                 log: Optional[List[Dict[str, Any]]],
                 block: int, amps: Tuple[float, float],
                 contacts: Dict[int, int], step_counter: List[int]) -> None:
    """One episode. When `log` is not None, records the per-step trace."""
    _flat, obs_dict = env.reset()
    for _ in range(steps):
        energy_pre = float(getattr(env, "agent_energy", float("nan")))
        # Read the amplitude tuple off the LIVE env, not from the driver's own
        # constants -- the ratified spec asks for the env's value, and a driver
        # constant cannot detect a manipulation that failed to land.
        env_amps = [float(x) for x in
                    (getattr(env, "resource_type_benefit_amplitudes", ()) or ())]
        result = harness.step(obs_dict)
        obs_dict = result.next_obs_dict

        tag = int(result.info.get("sd049_consumed_type_tag_this_tick", 0) or 0)
        if tag in contacts:
            contacts[tag] += 1

        if log is not None:
            act = result.action
            try:
                action_idx = int(act.argmax().item()) if act is not None else -1
            except Exception:
                action_idx = -1
            body = obs_dict.get("body_state")
            try:
                body3 = float(np.asarray(body).reshape(-1)[3])
            except Exception:
                body3 = float("nan")
            # Benefit actually applied this tick. Mirrors causal_grid_world.py:2131
            # (contact_benefit = resource_benefit * amp). Safe here because the
            # transient-benefit-patch override at :2138 requires transient patches,
            # which this config does not enable.
            # Prefer the LIVE env amplitude; fall back to the driver constant only
            # if the env exposes no tuple (it always does on this config).
            if tag in (1, 2):
                amp_live = env_amps[tag - 1] if len(env_amps) >= tag else amps[tag - 1]
                applied = RESOURCE_BENEFIT * amp_live
            else:
                applied = 0.0
            energy_post = float(getattr(env, "agent_energy", float("nan")))
            # Independent cross-check of `applied`, derived from the observable
            # channel rather than from any amplitude reading: the env applies
            # `agent_energy += contact_benefit * 0.5` (causal_grid_world.py:2176)
            # in the same tick it applies the 0.01 decay (:2691). Only meaningful
            # at an UNCLAMPED contact step; NaN otherwise, never silently 0.0.
            if (tag in (1, 2) and math.isfinite(energy_pre)
                    and math.isfinite(energy_post) and energy_post < 1.0 - 1e-12):
                implied = (energy_post - energy_pre + float(
                    getattr(env, "energy_decay", 0.01))) * 2.0
            else:
                implied = float("nan")
            log.append({
                "step": int(step_counter[0]),
                "block": int(block),
                "amps": [float(a) for a in amps],
                "env_amps": env_amps,
                "energy_pre": energy_pre,
                "energy_post": energy_post,
                "energy_delta": (energy_post - energy_pre
                                 if math.isfinite(energy_pre) and math.isfinite(energy_post)
                                 else float("nan")),
                "implied_contact_benefit": implied,
                "body3": body3,
                "action_idx": action_idx,
                "consumed_tag": tag,
                "contact_benefit_applied": float(applied),
            })
        step_counter[0] += 1

        if result.done:
            _flat, obs_dict = env.reset()


def _run_p0_warmup(harness: StepHarness, env, agent: REEAgent, episodes: int,
                   steps: int, on_episode) -> int:
    """V3-EXQ-966's P0 representational warmup, replicated verbatim.

    966 trains e1 + latent_stack for WARMUP_EPISODES on a prediction +
    resource-encoder loss (v3_exq_966...py P0 block, Adam @ P0_LR). Rolling the
    same episodes WITHOUT that optimiser -- which this driver did until the
    red-team caught it -- produces an untrained-encoder agent, i.e. a DIFFERENT
    agent from the one whose behaviour is being explained. That matters most for
    the H_insensitive_policy branch, which would otherwise read an untrained
    encoder's indifference as a finding about the selection pathway.

    Returns the number of optimiser steps that actually ran, so the manifest can
    show the warmup trained rather than silently swallowing every loss (966's
    own `except Exception: pass` is preserved for fidelity, but here it is
    COUNTED as well as caught).
    """
    p0_opt = torch.optim.Adam(
        list(agent.e1.parameters()) + list(agent.latent_stack.parameters()),
        lr=P0_LR,
    )
    opt_steps = 0
    _flat, obs_dict = env.reset()
    for _ in range(episodes):
        for _ in range(steps):
            result = harness.step(obs_dict)
            rfv = obs_dict.get("resource_field_view")
            target = float(rfv.max().item()) if rfv is not None else 0.0
            try:
                loss = agent.compute_prediction_loss()
                loss = loss + agent.compute_resource_encoder_loss(target, result.latent)
                if loss.requires_grad:
                    p0_opt.zero_grad()
                    loss.backward()
                    p0_opt.step()
                    opt_steps += 1
            except Exception:
                pass  # 966 fidelity: P0 is best-effort representational shaping.
            obs_dict = result.next_obs_dict
            if result.done:
                _flat, obs_dict = env.reset()
        _flat, obs_dict = env.reset()
        on_episode()
    return opt_steps


def run_cell(arm: str, seed: int, dry_run: bool = False) -> Dict[str, Any]:
    print(f"Seed {seed} Condition {arm}", flush=True)
    warmup = DRY_WARMUP_EPISODES if dry_run else WARMUP_EPISODES
    eps_per_block = DRY_EPISODES_PER_BLOCK if dry_run else EPISODES_PER_BLOCK
    steps = DRY_STEPS if dry_run else STEPS_PER_EP
    total_eps = warmup + N_BLOCKS * eps_per_block

    schedule = BLOCK_HIGH_IDX[arm]

    with arm_cell(
        seed,
        config_slice=_config_slice(arm),
        script_path=Path(__file__),
        config_slice_declared=True,
    ) as cell:
        env = _make_env(seed, schedule[0])
        agent = _make_agent(seed, env)
        harness = StepHarness(agent, env, train_mode=True, seed=seed)

        # Config-landing assert. from_dims silently swallows unknown kwargs, and a
        # separate open chip alleges wanting_weight is dropped on this exact path.
        wanting_live = float(
            getattr(getattr(agent, "hippocampal", None), "config", None).wanting_weight
            if getattr(getattr(agent, "hippocampal", None), "config", None) is not None
            else float("nan")
        )

        ep_done = [0]
        step_counter = [0]

        def _tick_warmup() -> None:
            ep_done[0] += 1
            if ep_done[0] % 5 == 0:
                print(f"  [train] confirmer seed={seed} arm={arm} "
                      f"ep {ep_done[0]}/{total_eps} (warmup)", flush=True)

        # -- P0 warmup: 966's TRAINING warmup, identical in both arms --
        p0_opt_steps = _run_p0_warmup(harness, env, agent, warmup, steps, _tick_warmup)

        # -- P1 blocks. 0-1 run but unlogged; 2-3 logged per step. --
        per_step_log: List[Dict[str, Any]] = []
        block_contacts: List[Dict[str, int]] = []
        for b in range(N_BLOCKS):
            high_idx = schedule[b]
            _set_high_idx(env, high_idx)
            amps = _amps_for(high_idx)
            block_total = {1: 0, 2: 0}
            log_target = per_step_log if b in LOGGED_BLOCKS else None
            for _ in range(eps_per_block):
                _run_episode(harness, env, steps, log_target, b, amps,
                             block_total, step_counter)
                ep_done[0] += 1
                if ep_done[0] % 5 == 0:
                    print(f"  [train] confirmer seed={seed} arm={arm} "
                          f"ep {ep_done[0]}/{total_eps} (block {b})", flush=True)
            block_contacts.append({"type_1": block_total[1], "type_2": block_total[2]})

        _ZG.observe(agent)

        logged_contacts = sum(1 for r in per_step_log if r["consumed_tag"] in (1, 2))
        energies = [r["energy_pre"] for r in per_step_log
                    if isinstance(r["energy_pre"], float) and math.isfinite(r["energy_pre"])]
        unclamped = [e for e in energies if e < 1.0 - 1e-12]
        benefits = [r["contact_benefit_applied"] for r in per_step_log
                    if r["consumed_tag"] in (1, 2)]
        # POSITIVE CONTROL for the channel D3 routes on. D3 asks whether the
        # agent_energy trajectories DIFFER between arms; a flat False is only a
        # finding about the energy route if a contact can move agent_energy at
        # all in this config. If it cannot -- because the bump does not land, or
        # because this driver reads the wrong attribute or the wrong tick -- then
        # D3 is False by construction and `H_inert_in_fact_energy_route_blocked`
        # would be an INSTRUMENT artifact wearing a substrate verdict's label.
        # `implied_contact_benefit` reconstructs contact_benefit from the
        # observable channel alone (energy_post - energy_pre + decay, x2), and is
        # NaN at clamped or non-contact steps, so this measures only steps where
        # a bump WOULD have been visible.
        bumps = [r["implied_contact_benefit"] for r in per_step_log
                 if r["consumed_tag"] in (1, 2)
                 and math.isfinite(r.get("implied_contact_benefit", float("nan")))]

        row: Dict[str, Any] = {
            "arm": arm,
            "seed": int(seed),
            "p0_opt_steps": int(p0_opt_steps),
            "p0_warmup_trained": bool(p0_opt_steps > 0),
            "wanting_weight_live": wanting_live,
            "wanting_weight_landed": bool(
                math.isfinite(wanting_live) and abs(wanting_live - WANTING_WEIGHT) < 1e-9
            ),
            "block_contacts": block_contacts,
            "block_high_idx": list(schedule),
            "n_logged_steps": len(per_step_log),
            "n_logged_contacts": int(logged_contacts),
            "energy_unclamped_frac": (
                float(len(unclamped)) / float(len(energies)) if energies else 0.0
            ),
            "energy_mean": float(statistics.fmean(energies)) if energies else 0.0,
            "energy_min": float(min(energies)) if energies else 0.0,
            "contact_benefit_mean": float(statistics.fmean(benefits)) if benefits else 0.0,
            "observed_contact_bump_max": float(max(bumps)) if bumps else 0.0,
            "n_bump_observable_contacts": int(len(bumps)),
            "env_energy_decay": float(getattr(env, "energy_decay", float("nan"))),
            "contact_benefit_distinct": sorted({round(b, 6) for b in benefits}),
            "per_step_log": per_step_log,
        }
        cell.stamp(row)

    print(
        f"verdict: "
        f"{'PASS' if row['n_logged_contacts'] >= MIN_LOGGED_CONTACTS else 'FAIL'}",
        flush=True,
    )
    return row


# --------------------------------------------------------------------------- #
# Discrimination                                                               #
# --------------------------------------------------------------------------- #

def _compare(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Read the five discriminators off the two arms' matched-step logs."""
    la, lb = a["per_step_log"], b["per_step_log"]
    n = min(len(la), len(lb))

    # D1 is read from the LIVE env amplitude tuple recorded per step, not from
    # the driver's own block schedule -- a driver constant cannot detect a
    # manipulation that failed to land on the env.
    def _amps_at(row: Dict[str, Any]) -> List[float]:
        return row.get("env_amps") or row.get("amps") or []

    d1 = any(_amps_at(la[i]) != _amps_at(lb[i]) for i in range(n))

    # D2 is a MATCHED-STEP comparison. It must NOT be a comparison of distinct
    # benefit SETS or of MEANS: under exact reproduction of 966's bit-identity
    # both arms visit the same resources at the same steps, so each arm's set of
    # applied benefits is {LOW, HIGH} and, with equal per-type counts, the means
    # are equal too. The aggregate form therefore reads False in precisely the
    # scenario this confirmer exists to explain, emits
    # `amplitude_not_reaching_consumption`, and makes D3/D4 -- the entire
    # H-inert vs H-insensitive adjudication -- unreachable. (Red-team BLOCKING
    # finding, 2026-09-01.) The aggregates are kept below as DIAGNOSTICS only.
    d2 = any(
        abs(la[i]["contact_benefit_applied"] - lb[i]["contact_benefit_applied"]) > 1e-9
        for i in range(n)
    )
    d2_aggregate_sets_differ = (
        sorted(a["contact_benefit_distinct"]) != sorted(b["contact_benefit_distinct"])
    )
    d2_aggregate_mean_differs = (
        abs(a["contact_benefit_mean"] - b["contact_benefit_mean"]) > 1e-9
    )
    n_benefit_diff_steps = sum(
        1 for i in range(n)
        if abs(la[i]["contact_benefit_applied"] - lb[i]["contact_benefit_applied"]) > 1e-9
    )
    # Independent of any amplitude reading: does the OBSERVABLE energy channel
    # imply different contact benefits? NaN-skipping, so a clamped or
    # non-contact step contributes nothing rather than a spurious 0.0.
    _implied = [
        abs(la[i]["implied_contact_benefit"] - lb[i]["implied_contact_benefit"])
        for i in range(n)
        if math.isfinite(la[i].get("implied_contact_benefit", float("nan")))
        and math.isfinite(lb[i].get("implied_contact_benefit", float("nan")))
    ]
    max_implied_benefit_diff = float(max(_implied)) if _implied else 0.0
    n_implied_comparable_steps = len(_implied)

    # Compare energy_POST, not energy_pre. energy_pre is read BEFORE harness.step,
    # so a contact's amplitude-scaled bump lands only in the FOLLOWING step's
    # energy_pre -- and is missed entirely when the contact is the last logged step.
    # The dry run hit exactly that: applied benefits differed (0.045 vs 0.3) while
    # every energy_pre matched, because the single contact fell at the end. Reading
    # energy_post captures the bump on the tick that produced it. Both series are
    # recorded; the pre-based figure is kept alongside purely for reference.
    def _maxdiff(key: str) -> float:
        vals = [
            abs(la[i][key] - lb[i][key])
            for i in range(n)
            if math.isfinite(la[i][key]) and math.isfinite(lb[i][key])
        ]
        return float(max(vals)) if vals else 0.0

    max_energy_diff = _maxdiff("energy_post")
    # Onset and extent of the energy divergence. Without these, an
    # H_insensitive_policy verdict can be returned on an exposure window far too
    # short for the policy to have responded IN -- "no response" would then be a
    # statement about the window, not about the policy. (Red-team finding 4.)
    first_energy_div_idx = -1
    n_energy_diff_steps = 0
    for i in range(n):
        ea, eb = la[i]["energy_post"], lb[i]["energy_post"]
        if not (math.isfinite(ea) and math.isfinite(eb)):
            continue
        if abs(ea - eb) > ENERGY_TOL:
            n_energy_diff_steps += 1
            if first_energy_div_idx < 0:
                first_energy_div_idx = i
    first_energy_divergence_step = (
        int(la[first_energy_div_idx]["step"]) if first_energy_div_idx >= 0 else -1
    )
    steps_after_energy_divergence = (
        int(n - first_energy_div_idx) if first_energy_div_idx >= 0 else 0
    )
    max_energy_diff_pre = _maxdiff("energy_pre")
    max_body3_diff = _maxdiff("body3")
    d3 = max_energy_diff > ENERGY_TOL

    first_div = -1
    n_action_diff = 0
    for i in range(n):
        if la[i]["action_idx"] != lb[i]["action_idx"]:
            n_action_diff += 1
            if first_div < 0:
                first_div = int(la[i]["step"])
    d4 = n_action_diff > 0

    d5 = a["block_contacts"] != b["block_contacts"]

    return {
        "D1_amplitudes_differ": bool(d1),
        "D2_contact_benefit_differs": bool(d2),
        "D3_energy_differs": bool(d3),
        "D4_actions_differ": bool(d4),
        "D5_contacts_differ": bool(d5),
        "d2_aggregate_sets_differ": bool(d2_aggregate_sets_differ),
        "d2_aggregate_mean_differs": bool(d2_aggregate_mean_differs),
        "n_benefit_diff_steps": int(n_benefit_diff_steps),
        "max_implied_benefit_diff": max_implied_benefit_diff,
        "n_implied_comparable_steps": int(n_implied_comparable_steps),
        "first_energy_divergence_step": first_energy_divergence_step,
        "n_energy_diff_steps": int(n_energy_diff_steps),
        "steps_after_energy_divergence": int(steps_after_energy_divergence),
        "max_energy_diff": max_energy_diff,
        "max_energy_diff_pre": max_energy_diff_pre,
        "max_body3_diff": max_body3_diff,
        "n_action_diff": int(n_action_diff),
        "first_action_divergence_step": int(first_div),
        "n_matched_steps": int(n),
        "action_diff_frac": (float(n_action_diff) / float(n)) if n else 0.0,
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    rows = [run_cell(arm, SEED, dry_run=dry_run) for arm in ARMS]
    a, b = rows[0], rows[1]
    d = _compare(a, b)

    worst_contacts = min(int(r["n_logged_contacts"]) for r in rows)
    worst_unclamped = min(float(r["energy_unclamped_frac"]) for r in rows)
    worst_cell = min(rows, key=lambda r: float(r["energy_unclamped_frac"]))["arm"]
    landed = 1.0 if all(bool(r["wanting_weight_landed"]) for r in rows) else 0.0

    worst_p0 = min(int(r["p0_opt_steps"]) for r in rows)
    worst_bump = min(float(r["observed_contact_bump_max"]) for r in rows)
    worst_bump_cell = min(
        rows, key=lambda r: float(r["observed_contact_bump_max"]))["arm"]

    preconditions = [
        {
            "name": "p0_warmup_actually_trained",
            "description": (
                "optimiser steps taken during the P0 representational warmup. "
                "966 trains e1+latent_stack here; a driver that merely ROLLS the "
                "warmup episodes measures an untrained-encoder agent, which is "
                "not the agent whose behaviour is under explanation -- and would "
                "let an untrained encoder's indifference be read as "
                "H_insensitive_policy"
            ),
            "kind": "readiness",
            "measured": float(worst_p0),
            "threshold": 1.0,
            "direction": "lower",
            "control": "worst arm; 966's P0 block, replicated (Adam @ P0_LR)",
            "met": bool(worst_p0 >= 1),
        },
        {
            "name": "contact_bump_reaches_agent_energy",
            "description": (
                "largest contact benefit reconstructible from the OBSERVABLE "
                "energy channel alone at an unclamped contact step. This is the "
                "positive control for the very statistic D3 routes on: if a "
                "contact cannot move agent_energy here, D3 is False by "
                "construction and the H_inert label would be an instrument "
                "artifact rather than a finding about the energy route"
            ),
            "kind": "readiness",
            "measured": float(worst_bump),
            "threshold": 1e-6,
            "direction": "lower",
            "control": (
                "worst arm; the env applies agent_energy += contact_benefit*0.5 "
                "at causal_grid_world.py:2176, so a real contact must show a "
                "positive reconstructed bump at an unclamped step. USED ONLY AS A "
                "LIVENESS FLOOR: the reconstruction subtracts the per-step decay "
                "(:2691) but NOT any action/movement energy cost (:2707), so its "
                "VALUE is not a calibrated contact_benefit and must not be read as "
                "one -- only its being above zero is asserted here. Read the "
                "per-step energy_delta series (contact vs non-contact steps) if the "
                "magnitude itself matters"
            ),
            "offending_cell": worst_bump_cell,
            "met": bool(worst_bump > 1e-6),
        },
        {
            "name": "logged_contacts_sufficient",
            "description": (
                "resource contacts inside the logged (arm-differing) blocks -- "
                "nothing about the consumption route is measurable without them"
            ),
            "kind": "readiness",
            "measured": float(worst_contacts),
            "threshold": float(MIN_LOGGED_CONTACTS),
            "direction": "lower",
            "control": "worst arm",
            "met": bool(worst_contacts >= MIN_LOGGED_CONTACTS),
        },
        {
            "name": "energy_spends_time_unclamped",
            "description": (
                "fraction of logged steps with agent_energy strictly below the 1.0 "
                "ceiling. This is the autopsy's OWN contested premise made "
                "measurable: if energy is pinned at the clamp, the amplitude bump "
                "is absorbed and D3 cannot speak to the energy route at all"
            ),
            "kind": "readiness",
            "measured": float(worst_unclamped),
            "threshold": float(MIN_UNCLAMPED_FRAC),
            "direction": "lower",
            "control": "worst arm; autopsy asserts energy sits between the clamps "
                       "for most of a 120-step episode",
            "offending_cell": worst_cell,
            "met": bool(worst_unclamped >= MIN_UNCLAMPED_FRAC),
        },
        {
            "name": "wanting_weight_landed",
            "description": (
                "config.hippocampal.wanting_weight equals the requested value at "
                "runtime -- guards the alleged from_dims drop site without "
                "depending on how that separate chip resolves"
            ),
            "kind": "readiness",
            "measured": float(landed),
            "threshold": 1.0,
            "direction": "lower",
            "control": "both arms; 1.0 = landed in every cell",
            "met": bool(landed >= 1.0),
        },
    ]
    readiness_ok = all(bool(p["met"]) for p in preconditions)

    if not readiness_ok:
        label = "substrate_not_ready_requeue"
    elif not d["D1_amplitudes_differ"]:
        label = "manipulation_not_applied"
    elif not d["D2_contact_benefit_differs"]:
        label = "amplitude_not_reaching_consumption"
    elif not d["D3_energy_differs"]:
        label = "H_inert_in_fact_energy_route_blocked"
    elif not d["D4_actions_differ"]:
        label = "H_insensitive_policy"
    elif not d["D5_contacts_differ"]:
        label = "behaviour_diverges_contacts_coincide"
    else:
        label = "phenomenon_not_reproduced"

    note_map = {
        "substrate_not_ready_requeue": (
            "A readiness precondition is unmet, so the discriminators cannot be read. "
            "Re-queue at adequate contact count / unclamped-energy fraction. No route "
            "is claimed."
        ),
        "manipulation_not_applied": (
            "The live amplitude tuple did NOT differ between arms in the logged "
            "blocks. The inertness is an upstream instrument bug -- the manipulation "
            "never existed -- and neither H-inert-in-fact nor H-insensitive-policy is "
            "implicated. Fix the driver, not the substrate."
        ),
        "amplitude_not_reaching_consumption": (
            "Amplitudes differ but the benefit actually applied at contacts does not. "
            "The amplitude is not reaching _consume_resource_at's scaling. "
            "ENVIRONMENTAL fix; H-insensitive-policy is NOT implicated."
        ),
        "H_inert_in_fact_energy_route_blocked": (
            "Amplitudes and applied contact benefits differ, but the agent_energy "
            "trajectories do NOT. Something absorbs the difference before it reaches "
            "the observable channel, so V3-EXQ-966 measured nothing and the fix is "
            "ENVIRONMENTAL. This SUPPORTS H-inert-in-fact and refutes the autopsy's "
            "withdrawn-then-reinstated claim that the energy route is open."
        ),
        "H_insensitive_policy": (
            "The amplitude is applied, the applied benefit differs, and agent_energy "
            "genuinely differs between arms -- yet the committed action sequence is "
            "identical at every matched step. E3 selection is unmoved by a real body-"
            "state difference. This is the SUBSTANTIVE and more troubling branch: it "
            "is a finding about the selection pathway's sensitivity to body state, "
            "not an environment bug, and it is what a substrate specification should "
            "be written against. Read it together with "
            "discriminators.steps_after_energy_divergence: if that window is "
            "shorter than MIN_POST_DIVERGENCE_STEPS the criterion "
            "C2_policy_responds is marked non-degenerate=false and this label is "
            "NOT adjudicable -- the run saw too little post-divergence exposure "
            "for the policy to have responded in."
        ),
        "behaviour_diverges_contacts_coincide": (
            "Actions DO diverge between arms, yet block_contacts still coincide. The "
            "bit-identity is then a property of the contact-counting DV (saturation / "
            "coarse quantisation), not of the policy. The DV, not the substrate, is "
            "what needs replacing in any 966 successor."
        ),
        "phenomenon_not_reproduced": (
            "block_contacts DIFFER between arms here, so V3-EXQ-966's bit-identity did "
            "not reproduce at this seed/config. No route is localised; the difference "
            "between this run and 966 must be explained before any 966a is designed."
        ),
    }

    criteria = [
        {
            "name": "C1_energy_route_open",
            "load_bearing": True,
            "passed": bool(d["D3_energy_differs"]),
            "detail": (
                f"max |energy_A - energy_B| over {d['n_matched_steps']} matched steps "
                f"= {d['max_energy_diff']:.6g} vs tol {ENERGY_TOL}"
            ),
        },
        {
            "name": "C2_policy_responds",
            "load_bearing": False,
            "passed": bool(d["D4_actions_differ"]),
            "detail": (
                f"{d['n_action_diff']}/{d['n_matched_steps']} matched steps differ "
                f"(frac {d['action_diff_frac']:.4f}); first divergence at step "
                f"{d['first_action_divergence_step']}"
            ),
        },
        {
            "name": "C3_phenomenon_reproduced",
            "load_bearing": False,
            "passed": bool(not d["D5_contacts_differ"]),
            "detail": (
                f"block_contacts A={a['block_contacts']} B={b['block_contacts']}"
            ),
        },
    ]

    criteria_non_degenerate = {
        "C1_energy_route_open": bool(readiness_ok and d["n_matched_steps"] > 0),
        # Asking whether the policy responds is only meaningful once there is a
        # body-state difference for it to respond TO -- AND enough matched steps
        # remain after that difference first appears for a response to be
        # observable at all. A divergence in the final few steps cannot support
        # "the policy did not respond". (Red-team finding 4.)
        "C2_policy_responds": bool(
            readiness_ok
            and d["D3_energy_differs"]
            and d["steps_after_energy_divergence"] >= MIN_POST_DIVERGENCE_STEPS
        ),
        "C3_phenomenon_reproduced": bool(readiness_ok and d["n_matched_steps"] > 0),
    }

    if dry_run:
        for pc in preconditions:
            print(f"  [smoke] precondition {pc['name']}: measured={pc['measured']} "
                  f"threshold={pc['threshold']} met={pc['met']}", flush=True)
        for r in rows:
            print(f"  [smoke] arm={r['arm']} logged_steps={r['n_logged_steps']} "
                  f"contacts={r['n_logged_contacts']} "
                  f"unclamped_frac={r['energy_unclamped_frac']:.4f} "
                  f"energy_mean={r['energy_mean']:.6f} energy_min={r['energy_min']:.6f} "
                  f"benefits={r['contact_benefit_distinct']} "
                  f"bump_max={r['observed_contact_bump_max']:.6g} "
                  f"bump_obs={r['n_bump_observable_contacts']} "
                  f"block_contacts={r['block_contacts']}", flush=True)

    outcome = "FAIL" if not readiness_ok else "PASS"
    outcome_note = (
        f"{label}: D1_amp={d['D1_amplitudes_differ']} D2_benefit={d['D2_contact_benefit_differs']} "
        f"D3_energy={d['D3_energy_differs']} D4_actions={d['D4_actions_differ']} "
        f"D5_contacts={d['D5_contacts_differ']}; max_energy_diff={d['max_energy_diff']:.6g}; "
        f"action divergence {d['n_action_diff']}/{d['n_matched_steps']} "
        f"(first at step {d['first_action_divergence_step']}); "
        f"benefits applied A={a['contact_benefit_distinct']} B={b['contact_benefit_distinct']}; "
        f"energy unclamped frac A={a['energy_unclamped_frac']:.4f} "
        f"B={b['energy_unclamped_frac']:.4f}. {note_map[label]}"
    )

    return {
        "outcome": outcome,
        "outcome_note": outcome_note,
        "arm_results": rows,
        "discriminators": d,
        "criteria": criteria,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "combination_rule": (
                "A FIXED CHAIN, not a conjunction: the first discriminator in the "
                "order D1->D2->D3->D4->D5 that reads False determines the route, and "
                "later discriminators are not consulted once the chain stops. C1 is "
                "load-bearing because D3 is the autopsy's contested premise."
            ),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-967: V3-EXQ-966 shuffle-inertness confirmer"
    )
    parser.add_argument("--dry-run", action="store_true", help="Short run for smoke testing")
    args = parser.parse_args()

    t0 = time.perf_counter()
    result = run_experiment(dry_run=args.dry_run)

    timestamp = __import__("datetime").datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    out_dir = (
        Path(__file__).parent.parent.parent / "REE_assembly" / "evidence" / "experiments"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    full_config = {
        "seed": SEED,
        "seeds": [SEED],
        "arms": ARMS,
        "block_high_idx": BLOCK_HIGH_IDX,
        "logged_blocks": list(LOGGED_BLOCKS),
        "warmup_episodes": WARMUP_EPISODES,
        "n_blocks": N_BLOCKS,
        "episodes_per_block": EPISODES_PER_BLOCK,
        "steps_per_episode": STEPS_PER_EP,
        "grid_size": GRID_SIZE,
        "num_hazards": NUM_HAZARDS,
        "num_resources": NUM_RESOURCES,
        "resource_benefit": RESOURCE_BENEFIT,
        "high_amp": HIGH_AMP,
        "low_amp": LOW_AMP,
        "wanting_weight": WANTING_WEIGHT,
        "min_logged_contacts": MIN_LOGGED_CONTACTS,
        "min_unclamped_frac": MIN_UNCLAMPED_FRAC,
        "energy_tol": ENERGY_TOL,
        "min_post_divergence_steps": MIN_POST_DIVERGENCE_STEPS,
        "p0_lr": P0_LR,
        "arm_config_slices": {a: _config_slice(a) for a in ARMS},
    }

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        # Deliberately EMPTY: this run localises an instrument, it tests no claim.
        "claim_ids": [],
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": "diagnostic",
        "outcome": result["outcome"],
        "outcome_note": result["outcome_note"],
        "timestamp_utc": timestamp,
        "seeds": [SEED],
        "arm_results": result["arm_results"],
        "discriminators": result["discriminators"],
        "criteria": result["criteria"],
        "interpretation": result["interpretation"],
        "custom_information": {
            "confirms": "V3-EXQ-966 arm bit-identity (block_contacts)",
            "predecessor_autopsy": "failure_autopsy_966-436g-951-959-822d-cluster_2026-08-30",
            "ratified_ordering": (
                "user, 2026-08-30T15:41:56Z: the cheap confirmer runs FIRST, before "
                "any substrate work is specified or 966a is queued"
            ),
            "hypotheses": {
                "H_inert_in_fact": "amplitude never reaches behaviour; fix is environmental",
                "H_insensitive_policy": "amplitude reaches body state; E3 selection unmoved",
            },
            "does_not_do": [
                "does not redesign V3-EXQ-966 (no 966a here)",
                "does not specify a substrate build",
                "does not adjudicate chip-20260830-exq966-wantingweight-fromdims-dropsite",
            ],
        },
    }

    stamp_recording_core(
        manifest, config=full_config, seeds=[SEED],
        script_path=Path(__file__), started_at=t0,
    )

    out_path = write_flat_manifest(
        manifest, out_dir, dry_run=bool(args.dry_run),
        config=full_config, seeds=[SEED],
        script_path=Path(__file__), started_at=t0,
        z_goal_stream_stats=_ZG.stats(), json_default=str,
    )
    print(f"Manifest written: {out_path}")

    d = result["discriminators"]
    print(
        f"SUMMARY: D1_amp={d['D1_amplitudes_differ']} D2_benefit={d['D2_contact_benefit_differs']} "
        f"D3_energy={d['D3_energy_differs']} D4_actions={d['D4_actions_differ']} "
        f"D5_contacts={d['D5_contacts_differ']} "
        f"max_energy_diff={d['max_energy_diff']:.6g} "
        f"action_diff={d['n_action_diff']}/{d['n_matched_steps']} "
        f"first_div={d['first_action_divergence_step']}",
        flush=True,
    )
    print(f"LABEL: {result['interpretation']['label']}", flush=True)

    if args.dry_run:
        print("DRY RUN complete.")

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=bool(args.dry_run),
    )
