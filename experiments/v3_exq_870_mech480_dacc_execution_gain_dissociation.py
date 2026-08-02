#!/opt/local/bin/python3
"""V3-EXQ-870 -- MECH-480 dACC-analog execution-gain dissociation (EXP-0413).

SLEEP DRIVER: not applicable (no sleep loop in this experiment).

Source paper (found via /lit-pull-style targeted search 2026-08-02; the raw
thought intake at docs/thoughts/2026-07-30_acc_lofc_strategy_authority_vs_
execution_gain.md cited it only as "a recent Nature Communications paper"):
  Asaoka N., Pagano D., Hayashi Y. "Dissociable roles of prefrontal plasticity
  in decision making strategy and execution of habitual behavior." Nature
  Communications (2026). DOI: 10.1038/s41467-026-75706-1 (Kyoto University;
  bioRxiv preprint 10.1101/2025.05.19.654900). The paper dissociates an
  ACC-to-retrosplenial-cortex circuit (determines WHETHER a behaviour becomes
  a habit -- strategy authority) from a lateral-OFC-to-central-striatum
  circuit (determines HOW STRONGLY the habit is executed -- execution gain /
  vigor), showing the two are causally separable via independent circuit
  manipulations. This experiment translates that dissociation onto REE's
  existing SD-032b dACC-analog substrate.

Claim: MECH-480 (docs/claims/claims.yaml). SD-032b's dACC-analog output may
conflate two separable channels: strategy authority (does accumulated
action-outcome evidence retain control over the policy -- already covered by
SD-032b/MECH-258) and a LOFC-analog execution-gain channel (how strongly a
persisting policy is expressed, independent of whether it retains decision
authority). Related claims (context only, NOT scored -- see claim_ids below):
SD-032b, MECH-258, MECH-260.

Design (EXP-0413, REE_assembly/evidence/planning/manual_proposals.v1.json):
Two-arm ablation on SD-032b's existing dacc_weight config flag -- NO new
dACC/OFC substrate. Arm OFF (dacc_weight=0, use_dacc=False): baseline. Arm ON
(dacc_weight=3.0, use_dacc=True): dACC-analog active (config values carried
from v3_exq_450's calibration -- the magnitude found necessary there to shift
E3's argmax distribution in a small-seed budget).

CONTINGENCY-SWITCH MECHANISM (the "smallest addition" the proposal
sanctions -- no ree_core edits). CausalGridWorldV2's resource-value gradient
is uniform across all num_resources cells and resource identity is not
independently valued, so a reward-contingency change is implemented via
DIRECT geometric relocation of the resource cluster mid-episode -- the same
"mutate env attributes directly at a phase boundary" convention already used
in-tree (e.g. v3_exq_655's env.reef_bipartite_layout toggle). Each P2 episode:
resources start confined to the LEFT quarter of the grid (x in
[0, size//4)); at env-step SWITCH_STEP they are geometrically mirrored to the
RIGHT quarter (x in [size - size//4, size)) via _relocate_resources() +
env._compute_proximity_fields() (the env's own field-rebuild method -- reused,
not reimplemented, so the decay-field math cannot drift out of sync with the
env's own semantics). Hazards are left at the env's own placement (2 mild
hazards; not part of the manipulation).

DVs (both read off the existing, unmodified E3 selector -- no new substrate):
  (i) outcome-sensitivity: post_switch_toward_new_frac -- fraction of ticks in
      [SWITCH_STEP, STEPS_PER_EP) whose REALIZED agent_x displacement is
      positive (toward the new, right-mirrored resource region). Measured
      every tick (the agent moves every tick regardless of E3 cadence, using
      its currently-committed/held action), so this is NOT subject to the
      E3-tick latch hazard below.
  (ii) execution strength: logit margin of the committed action --
      second-lowest minus lowest of agent.e3.last_scores (scores are COSTS,
      lower=preferred, so this is the two-candidate cost gap the softmax
      commits over; equivalent to a logit margin under probs = softmax(-scores
      / temperature)). CRITICAL: last_scores is written ONLY inside
      E3.select(), which itself only runs on a genuine E3 tick
      (heartbeat.e3_steps_per_tick, default cadence 10 env-steps) --
      agent.py returns the held/stepped action on every other tick without
      ever reaching e3.select(). Per the Sample-size-integrity convention
      (queue-experiment skill Step 3.5), last_scores is explicitly CLEARED
      before every select_action() call and the read is skipped (counted in
      n_latched_ticks) whenever it is still None afterward -- so only genuine
      selections are averaged into execution_strength, never a stale
      re-reported value from several ticks back.

NON-DEGENERACY (EXP-0413 acceptance check 3, MANDATORY precondition before any
PASS/FAIL verdict is trusted): execution-strength (logit margin) must show
non-zero cross-seed variance in Arm OFF -- computed post-hoc via
_metrics.check_degeneracy over the 3 per-seed Arm-OFF mean margins (not a
pre-compute P0 gate: the quantity needs live per-seed data from the arm
itself, so it is checked once both arms have run, and a degenerate reading
routes the WHOLE run to substrate_not_ready_requeue, never a weakens/FAIL).

PRE-REGISTERED ACCEPTANCE (thresholds are constants below, not derived from
this run's own statistics):
  NON-DEGENERACY (checked first): std across the 3 Arm-OFF per-seed mean
    execution-strength values > MARGIN_VARIANCE_FLOOR. Unmet ->
    substrate_not_ready_requeue (non_contributory), NEVER a verdict.
  C1_outcome_sensitivity_shift: (ON.post_switch_toward_new_frac -
    OFF.post_switch_toward_new_frac) >= OUTCOME_SENSITIVITY_MARGIN on
    >= MIN_PASS_SEEDS of SEEDS (paired by seed).
  C2_execution_strength_equivalence: abs(ON.execution_strength -
    OFF.execution_strength) <= EXECUTION_STRENGTH_EQUIVALENCE_MARGIN on
    >= MIN_PASS_SEEDS of SEEDS (paired by seed).
  PASS / supports (dissociation): C1 AND C2 on a majority of seeds --
    outcome-sensitivity is dACC-dependent while execution strength is not.
  FAIL / weakens (no dissociation): NOT (C1 AND C2) -- either both DVs move
    together (C1 holds but C2 does not: execution strength is also
    dACC-dependent) or C1 itself does not hold (the ablation produces no
    reliable outcome-sensitivity effect either) -- either way, this claim's
    proposed dissociation is not supported and folds back into SD-032b's
    existing single-channel account.

claim_ids: ["MECH-480"] -- only claim actually exercised by this design.
  SD-032b/MECH-258/MECH-260 are the substrate ablated and the prior lineage
  this claim reframes, but a PASS or FAIL here is a statement about whether
  dACC's CURRENT channel is doing double duty, not new evidence for/against
  those claims' own (already-covered) content -- so they are cited as
  related_claims in the manifest, not scored.
experiment_purpose: "evidence" (retests an existing config flag's downstream
  behavioural dissociation; not a substrate-readiness probe).

See REE_assembly/docs/architecture/sd_032_cingulate_integration_substrate.md
See ree-v3/CLAUDE.md "SD-032b / MECH-258 / MECH-260 / ARC-058 ..." section.
"""

import argparse
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiments._metrics import check_degeneracy
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

EXPERIMENT_TYPE = "v3_exq_870_mech480_dacc_execution_gain_dissociation"
QUEUE_ID = "V3-EXQ-870"
CLAIM_IDS: List[str] = ["MECH-480"]
RELATED_CLAIMS: List[str] = ["SD-032b", "MECH-258", "MECH-260"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 7, 13]
CONDITIONS = ["OFF", "ON"]

# -- Env geometry (the contingency-switch region split) --
SIZE = 10
NUM_HAZARDS = 2
NUM_RESOURCES = 4
LEFT_REGION_X_MAX = SIZE // 4          # x in [0, LEFT_REGION_X_MAX)
RIGHT_REGION_X_MIN = SIZE - LEFT_REGION_X_MAX  # x in [RIGHT_REGION_X_MIN, SIZE)

# -- Episode timing --
# No separate E1/E2-world-forward encoder-warmup phase: the direct precedent
# for this exact SD-032b dACC-ablation lineage (v3_exq_450) runs its ENTIRE
# pre-eval budget as plain env-stepping + E2_harm_a training only, with no
# external E1/E2-world-forward optimizer -- matched here rather than adding
# an unvalidated extra phase.
STEPS_PER_EP = 80
SWITCH_STEP = 40                       # contingency switch fires at this env-step
# post-switch window is [SWITCH_STEP, STEPS_PER_EP) -- spans to episode end
P1_EPS = 90                            # E2_harm_a training (ON arm only; readiness for dACC)
P2_EPS = 20                            # eval episodes (contingency switch + DV measurement)
TRAIN_EPS = P1_EPS                     # progress denominator M

SELF_DIM = 16
WORLD_DIM = 32
HARM_DIM = 8
HARM_A_DIM = 8
HARM_HISTORY_LEN = 10

# -- dACC config (ON arm; magnitudes carried from v3_exq_450's calibration --
#    the gain found there necessary to shift E3's committed-selection
#    distribution in a small-seed budget). --
DACC_WEIGHT = 3.0
DACC_SUPPRESSION_WEIGHT = 1.5
DACC_INTERACTION_WEIGHT = 0.8
DACC_FORAGING_WEIGHT = 0.5
DACC_SUPPRESSION_MEMORY = 8
DACC_PRECISION_SCALE = 500.0
DACC_EFFORT_COST = 0.1

LR_E2_HARM_A = 5e-4

# -- Pre-registered thresholds (constants, NOT derived from this run's data) --
MARGIN_VARIANCE_FLOOR = 1e-4            # Arm-OFF cross-seed execution-strength std floor
OUTCOME_SENSITIVITY_MARGIN = 0.15       # ON - OFF post_switch_toward_new_frac delta
EXECUTION_STRENGTH_EQUIVALENCE_MARGIN = 0.05  # |ON - OFF| execution-strength band
MIN_PASS_SEEDS = 2                      # of 3


# --------------------------------------------------------------------------- #
# env / agent construction
# --------------------------------------------------------------------------- #
def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=SIZE,
        num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES,
        hazard_harm=0.04,
        proximity_harm_scale=0.1,
        proximity_benefit_scale=0.12,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        resource_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        harm_history_len=HARM_HISTORY_LEN,
    )


def _relocate_resources(env: CausalGridWorldV2, region: str) -> None:
    """Geometrically relocate every active resource into ``region``
    ("left" or "right") and rebuild the proximity fields via the env's own
    field-rebuild method (reused, not reimplemented -- see module docstring).
    Hazards are untouched. y-coordinates are preserved; only x is remapped,
    so the two regions are cleanly separated on the axis the outcome-
    sensitivity DV reads (sign of realized agent_x displacement).
    """
    new_resources: List[List[int]] = []
    for rx, ry in env.resources:
        if region == "left":
            nx = int(rx) % max(1, LEFT_REGION_X_MAX)
        else:
            nx = RIGHT_REGION_X_MIN + (int(rx) % max(1, LEFT_REGION_X_MAX))
        new_resources.append([nx, int(ry)])
    env.resources = new_resources
    env._compute_proximity_fields()


def _make_agent(env: CausalGridWorldV2, condition: str) -> REEAgent:
    use_dacc = condition == "ON"
    use_e2_harm_a = condition == "ON"

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
        use_harm_stream=True,
        harm_dim=HARM_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        use_affective_harm_stream=use_e2_harm_a,
        z_harm_a_dim=HARM_A_DIM,
        use_e2_harm_a=use_e2_harm_a,
        use_shared_harm_trunk=False,
        e2_harm_a_lr=LR_E2_HARM_A,
        use_dacc=use_dacc,
        dacc_weight=DACC_WEIGHT if use_dacc else 0.0,
        dacc_interaction_weight=DACC_INTERACTION_WEIGHT,
        dacc_foraging_weight=DACC_FORAGING_WEIGHT,
        dacc_suppression_weight=DACC_SUPPRESSION_WEIGHT,
        dacc_suppression_memory=DACC_SUPPRESSION_MEMORY,
        dacc_precision_scale=DACC_PRECISION_SCALE,
        dacc_effort_cost=DACC_EFFORT_COST,
        dacc_drive_coupling=0.0,
    )
    return REEAgent(config)


def _obs_tensors(obs_dict):
    body = obs_dict["body_state"].float().unsqueeze(0)
    world = obs_dict["world_state"].float().unsqueeze(0)
    harm = obs_dict["harm_obs"].float().unsqueeze(0) if "harm_obs" in obs_dict else None
    harm_a = (
        obs_dict["harm_obs_a"].float().unsqueeze(0)
        if "harm_obs_a" in obs_dict else None
    )
    harm_hist = (
        obs_dict["harm_history"].float().unsqueeze(0)
        if "harm_history" in obs_dict else None
    )
    return body, world, harm, harm_a, harm_hist


def _preflight() -> None:
    env = _make_env(seed=1)
    off = _make_agent(env, "OFF")
    on = _make_agent(env, "ON")
    assert off.dacc is None
    assert on.dacc is not None
    assert abs(float(on.config.dacc_weight) - DACC_WEIGHT) < 1e-9
    assert off.e2_harm_a is None
    assert on.e2_harm_a is not None
    assert hasattr(on.e3, "last_scores")
    assert hasattr(on.e3, "select")
    # relocation sanity: resources placed left of LEFT_REGION_X_MAX, then
    # mirrored right of RIGHT_REGION_X_MIN, with no dependence on env source
    # beyond the public `resources` list + `_compute_proximity_fields`.
    # env.resources is populated by reset(), not __init__ -- must reset first.
    env.reset()
    _relocate_resources(env, "left")
    assert all(0 <= r[0] < LEFT_REGION_X_MAX for r in env.resources)
    assert float(env.resource_field.max()) > 0.0
    _relocate_resources(env, "right")
    assert all(RIGHT_REGION_X_MIN <= r[0] < SIZE for r in env.resources)
    assert float(env.resource_field.max()) > 0.0
    del env, off, on
    print(
        "Preflight PASS: dACC OFF/ON config, harm_a stream gated on dacc, "
        "last_scores/select present, resource relocation confirmed in both "
        "regions with non-empty proximity field",
        flush=True,
    )



# --------------------------------------------------------------------------- #
# P1: E2_harm_a training on .detach()ed targets (ON arm only; a no-op pass-
# through for OFF, which has no e2_harm_a -- still steps the env so both
# arms see the same total episode budget).
# --------------------------------------------------------------------------- #
def _p1_train(agent: REEAgent, env: CausalGridWorldV2, episodes: int, steps: int,
               arm: str, has_e2_harm_a: bool, zg: ZGoalStreamAccumulator,
               total_eps: int) -> None:
    device = agent.device
    optim_e2_a = (
        torch.optim.Adam(agent.e2_harm_a.parameters(), lr=LR_E2_HARM_A)
        if has_e2_harm_a else None
    )
    for ep in range(episodes):
        _, obs_dict = env.reset()
        agent.reset()
        _relocate_resources(env, "left")
        prev_z_harm_a = None
        for _ in range(steps):
            body, world, harm, harm_a, harm_hist = _obs_tensors(obs_dict)
            latent = agent.sense(
                obs_body=body, obs_world=world, obs_harm=harm,
                obs_harm_a=harm_a, obs_harm_history=harm_hist,
            )
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, WORLD_DIM, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            agent.update_z_goal(
                benefit_exposure=max(0.0, float(obs_dict.get("benefit_exposure", 0.0))),
                drive_level=REEAgent.compute_drive_level(body),
            )
            action = agent.select_action(candidates, ticks, temperature=1.0)
            if action is None:
                action = torch.zeros(1, env.action_dim, device=device)
                action[0, random.randrange(env.action_dim)] = 1.0
                agent._last_action = action
            if (
                has_e2_harm_a and prev_z_harm_a is not None
                and latent.z_harm_a is not None
            ):
                z_prev_det = prev_z_harm_a.detach()
                z_next_det = latent.z_harm_a.detach()
                a_prev_det = action.detach()
                z_pred = agent.e2_harm_a(z_prev_det, a_prev_det)
                loss = agent.e2_harm_a.compute_loss(z_pred, z_next_det)
                optim_e2_a.zero_grad()
                loss.backward()
                optim_e2_a.step()
            prev_z_harm_a = (
                latent.z_harm_a.detach().clone() if latent.z_harm_a is not None else None
            )
            _, _harm_signal, done, _info, obs_dict = env.step(action)
            zg.observe(agent)
            if done:
                break
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  [train] {arm} ep {ep + 1}/{total_eps} phase=P1", flush=True)


# --------------------------------------------------------------------------- #
# P2: eval -- contingency switch mid-episode; outcome-sensitivity + execution
# strength measured from the unmodified E3 selector.
# --------------------------------------------------------------------------- #
def _p2_eval(agent: REEAgent, env: CausalGridWorldV2, episodes: int, steps: int,
             switch_step: int, zg: ZGoalStreamAccumulator) -> Dict:
    device = agent.device
    toward_new_hits = 0
    toward_new_total = 0
    toward_old_hits = 0
    toward_old_total = 0
    margins: List[float] = []
    n_latched = 0
    n_genuine_selections = 0

    for _ep in range(episodes):
        _, obs_dict = env.reset()
        agent.reset()
        _relocate_resources(env, "left")
        switched = False
        for step in range(steps):
            body, world, harm, harm_a, harm_hist = _obs_tensors(obs_dict)
            latent = agent.sense(
                obs_body=body, obs_world=world, obs_harm=harm,
                obs_harm_a=harm_a, obs_harm_history=harm_hist,
            )
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, WORLD_DIM, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            agent.update_z_goal(
                benefit_exposure=max(0.0, float(obs_dict.get("benefit_exposure", 0.0))),
                drive_level=REEAgent.compute_drive_level(body),
            )

            # Sample-size integrity (queue-experiment skill Step 3.5): clear the
            # latch immediately before the call so a tick where select() did
            # NOT run (off the E3 cadence) is never mistaken for a fresh read.
            agent.e3.last_scores = None
            x_before = env.agent_x
            action = agent.select_action(candidates, ticks, temperature=1.0)
            if action is None:
                action = torch.zeros(1, env.action_dim, device=device)
                action[0, random.randrange(env.action_dim)] = 1.0
                agent._last_action = action
            scores = agent.e3.last_scores
            if scores is None:
                n_latched += 1
            elif scores.numel() >= 2:
                n_genuine_selections += 1
                sorted_scores, _ = torch.sort(scores.detach().flatten())
                margin = float((sorted_scores[1] - sorted_scores[0]).item())
                if step >= switch_step:
                    margins.append(margin)

            if step == switch_step and not switched:
                _relocate_resources(env, "right")
                switched = True

            _, _harm_signal, done, _info, obs_dict = env.step(action)
            zg.observe(agent)
            x_after = env.agent_x
            dx = x_after - x_before

            if step < switch_step:
                toward_old_total += 1
                if dx < 0:
                    toward_old_hits += 1
            else:
                toward_new_total += 1
                if dx > 0:
                    toward_new_hits += 1

            if done:
                break

    post_switch_toward_new_frac = (
        toward_new_hits / toward_new_total if toward_new_total > 0 else 0.0
    )
    pre_switch_toward_old_frac = (
        toward_old_hits / toward_old_total if toward_old_total > 0 else 0.0
    )
    execution_strength = (
        float(sum(margins) / len(margins)) if margins else float("nan")
    )
    return {
        "post_switch_toward_new_frac": float(post_switch_toward_new_frac),
        "pre_switch_toward_old_frac": float(pre_switch_toward_old_frac),
        "execution_strength": execution_strength,
        "n_margin_samples": len(margins),
        "n_latched_ticks": n_latched,
        "n_genuine_selections": n_genuine_selections,
    }


def _run_cell(seed: int, condition: str, zg: ZGoalStreamAccumulator,
              steps_per_ep: int, p1_eps: int, p2_eps: int,
              switch_step: int, verbose: bool = True) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    env = _make_env(seed)
    agent = _make_agent(env, condition)
    has_e2_harm_a = condition == "ON" and agent.e2_harm_a is not None

    _p1_train(agent, env, p1_eps, steps_per_ep, condition, has_e2_harm_a, zg,
              p1_eps)
    eval_stats = _p2_eval(agent, env, p2_eps, steps_per_ep, switch_step, zg)

    row = {
        "seed": seed,
        "condition": condition,
        **eval_stats,
    }

    if verbose:
        es_str = (
            f"{eval_stats['execution_strength']:.5f}"
            if not math.isnan(eval_stats["execution_strength"]) else "n/a"
        )
        print(
            f"  [seed={seed} {condition}] "
            f"post_switch_toward_new={eval_stats['post_switch_toward_new_frac']:.3f} "
            f"pre_switch_toward_old={eval_stats['pre_switch_toward_old_frac']:.3f} "
            f"execution_strength={es_str} "
            f"n_latched={eval_stats['n_latched_ticks']} "
            f"n_genuine={eval_stats['n_genuine_selections']}",
            flush=True,
        )
    return row


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    _run_started = datetime.now(timezone.utc)

    _preflight()

    if args.dry_run:
        print(
            "Smoke: seed=42, tiny P1=4/P2=2, steps=20, switch=10, "
            "both conditions",
            flush=True,
        )
        seeds = [42]
        steps_per_ep, p1_eps, p2_eps, switch_step = 20, 4, 2, 10
    else:
        seeds = SEEDS
        steps_per_ep, p1_eps, p2_eps, switch_step = (
            STEPS_PER_EP, P1_EPS, P2_EPS, SWITCH_STEP,
        )

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    script_dir = Path(__file__).resolve().parents[1]
    out_dir = (
        Path(args.output_dir) if args.output_dir
        else script_dir.parent / "REE_assembly" / "evidence" / "experiments"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    zg = ZGoalStreamAccumulator()
    all_rows: List[Dict] = []
    arm_results: List[Dict] = []
    for seed in seeds:
        print(f"Seed {seed}")
        for cond in CONDITIONS:
            print(f"Seed {seed} Condition {cond}")
            config_slice = {
                "env_kwargs": {
                    "size": SIZE, "num_hazards": NUM_HAZARDS,
                    "num_resources": NUM_RESOURCES,
                },
                "condition": cond,
                "dacc_weight": DACC_WEIGHT if cond == "ON" else 0.0,
                "steps_per_ep": steps_per_ep,
                "switch_step": switch_step,
                "p1_eps": p1_eps, "p2_eps": p2_eps,
            }
            with arm_cell(
                seed, config_slice=config_slice, script_path=Path(__file__),
                config_slice_declared=True,
            ) as cell:
                row = _run_cell(
                    seed=seed, condition=cond, zg=zg,
                    steps_per_ep=steps_per_ep, p1_eps=p1_eps,
                    p2_eps=p2_eps, switch_step=switch_step,
                )
                cell.stamp(row)
            if args.dry_run:
                assert row["n_genuine_selections"] > 0, (
                    "no genuine E3 selection observed in the post-switch window -- "
                    "the decisive DV (execution-strength logit margin) is "
                    "structurally unmeasured; check heartbeat.e3_steps_per_tick "
                    "vs switch_step/steps"
                )
            all_rows.append(row)
            arm_results.append(row)
            verdict = "PASS" if row["n_genuine_selections"] > 0 else "FAIL"
            print(f"verdict: {verdict}")

    def by_cond(c):
        return [r for r in all_rows if r["condition"] == c]

    off_rows = {r["seed"]: r for r in by_cond("OFF")}
    on_rows = {r["seed"]: r for r in by_cond("ON")}

    # -- Non-degeneracy (checked FIRST; an unmet floor overrides any verdict).
    # check_degeneracy's `values` spread test is max-min against `eps` -- pass
    # MARGIN_VARIANCE_FLOOR as eps so "degenerate" means "spread <= floor",
    # the exact EXP-0413 acceptance check 3 semantics (a <2-sample collection
    # spreads to 0 and correctly reads degenerate -- the conservative default). --
    off_margins_per_seed = [
        off_rows[s]["execution_strength"] for s in seeds
        if not math.isnan(off_rows[s]["execution_strength"])
    ]
    degeneracy = check_degeneracy(
        {"arm_off_execution_strength_cross_seed": off_margins_per_seed},
        eps=MARGIN_VARIANCE_FLOOR,
    )
    non_degenerate = degeneracy["non_degenerate"]
    off_margin_range = (
        float(max(off_margins_per_seed) - min(off_margins_per_seed))
        if len(off_margins_per_seed) >= 2 else 0.0
    )

    c1_seed_hits = 0
    c2_seed_hits = 0
    per_seed_deltas = {}
    for s in seeds:
        off_r, on_r = off_rows.get(s), on_rows.get(s)
        if off_r is None or on_r is None:
            continue
        outcome_delta = (
            on_r["post_switch_toward_new_frac"] - off_r["post_switch_toward_new_frac"]
        )
        es_off, es_on = off_r["execution_strength"], on_r["execution_strength"]
        es_delta = (
            abs(es_on - es_off)
            if not (math.isnan(es_off) or math.isnan(es_on)) else float("inf")
        )
        per_seed_deltas[s] = {
            "outcome_sensitivity_delta": outcome_delta,
            "execution_strength_abs_delta": es_delta,
        }
        if outcome_delta >= OUTCOME_SENSITIVITY_MARGIN:
            c1_seed_hits += 1
        if es_delta <= EXECUTION_STRENGTH_EQUIVALENCE_MARGIN:
            c2_seed_hits += 1

    c1 = c1_seed_hits >= MIN_PASS_SEEDS
    c2 = c2_seed_hits >= MIN_PASS_SEEDS

    if not non_degenerate:
        outcome = "FAIL"
        evidence_direction = "non_contributory"
        interpretation_label = "substrate_not_ready_requeue"
    elif c1 and c2:
        outcome = "PASS"
        evidence_direction = "supports"
        interpretation_label = "dissociation_supported"
    else:
        outcome = "FAIL"
        evidence_direction = "weakens"
        interpretation_label = (
            "both_move_together_folds_into_sd032b" if c1
            else "no_reliable_outcome_sensitivity_effect"
        )

    summary = {
        "non_degeneracy": {
            "arm_off_execution_strength_range_across_seeds": off_margin_range,
            "threshold": MARGIN_VARIANCE_FLOOR,
            "pass": non_degenerate,
            "desc": "Arm-OFF execution-strength (logit margin) must show "
                    "non-zero cross-seed variance before any verdict is trusted.",
        },
        "c1_outcome_sensitivity_shift": {
            "seed_hits": c1_seed_hits,
            "threshold_margin": OUTCOME_SENSITIVITY_MARGIN,
            "pass": c1,
            "desc": "(ON - OFF) post_switch_toward_new_frac >= margin on "
                    f">= {MIN_PASS_SEEDS}/{len(seeds)} seeds.",
        },
        "c2_execution_strength_equivalence": {
            "seed_hits": c2_seed_hits,
            "threshold_margin": EXECUTION_STRENGTH_EQUIVALENCE_MARGIN,
            "pass": c2,
            "desc": "|ON - OFF| execution_strength <= margin on "
                    f">= {MIN_PASS_SEEDS}/{len(seeds)} seeds.",
        },
        "per_seed_deltas": per_seed_deltas,
    }

    print(f"\nOutcome: {outcome} ({interpretation_label})")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "related_claims": RELATED_CLAIMS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "interpretation": {
            "label": interpretation_label,
            "preconditions": [{
                "name": "arm_off_execution_strength_cross_seed_variance",
                "measured": off_margin_range,
                "threshold": MARGIN_VARIANCE_FLOOR,
                "direction": "lower",
                "control": "Arm-OFF (dacc_weight=0) per-seed mean logit margin",
                "met": non_degenerate,
            }],
            "criteria_non_degenerate": {
                "C1_outcome_sensitivity_shift": non_degenerate,
                "C2_execution_strength_equivalence": non_degenerate,
            },
        },
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy.get("degeneracy_reason", ""),
        "pass_criteria_summary": summary,
        "per_seed_results": all_rows,
        "arm_results": arm_results,
        "config": {
            "conditions": CONDITIONS,
            "seeds": seeds,
            "size": SIZE,
            "num_hazards": NUM_HAZARDS,
            "num_resources": NUM_RESOURCES,
            "steps_per_ep": steps_per_ep,
            "switch_step": switch_step,
            "p1_eps": p1_eps,
            "p2_eps": p2_eps,
            "dacc_weight": DACC_WEIGHT,
            "dacc_suppression_weight": DACC_SUPPRESSION_WEIGHT,
            "dacc_interaction_weight": DACC_INTERACTION_WEIGHT,
            "dacc_foraging_weight": DACC_FORAGING_WEIGHT,
            "outcome_sensitivity_margin": OUTCOME_SENSITIVITY_MARGIN,
            "execution_strength_equivalence_margin": EXECUTION_STRENGTH_EQUIVALENCE_MARGIN,
            "margin_variance_floor": MARGIN_VARIANCE_FLOOR,
        },
    }

    out_file = write_flat_manifest(
        output,
        out_dir,
        dry_run=args.dry_run,
        config=output.get("config"),
        seeds=seeds,
        script_path=Path(__file__),
        elapsed_seconds=(datetime.now(timezone.utc) - _run_started).total_seconds(),
        z_goal_stream_stats=zg.stats(),
    )
    if args.dry_run:
        print("Smoke test PASSED")
    print(f"Output written to: {out_file}")

    return out_file, outcome


if __name__ == "__main__":
    _out_file, _outcome = main()
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_file,
        dry_run="--dry-run" in sys.argv,
    )
