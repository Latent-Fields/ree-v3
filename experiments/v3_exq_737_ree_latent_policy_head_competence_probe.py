"""V3-EXQ-737 -- Trainable policy head on the REE latent (competence-floor H1 DIAGNOSTIC).

The DIRECT test of the "prediction-rich, action-poor" localization
(failure_autopsy_V3-EXQ-724_2026-07-09): REE trains a substantial world model but learns
ACTION only through a thin bias head over frozen prediction-trained representations
(bias-head-only REINFORCE). 724's H1 hypothesis is "a real trainable policy head would
recover competence." 734 tests H1 only INDIRECTLY (a non-REE PPO on the RAW observation);
neither 734 nor 735 ever bolts a real actor onto REE's OWN representation. This experiment
does exactly that.

MECHANISM. For each (env-rung, seed) it warms the all-ON REE stack with the IDENTICAL 724-A0
recipe (P0=200 world-model warmup -> P1=90 two-head REINFORCE, SD-056 e2 encoder FROZEN in P1;
imported verbatim from V3-EXQ-734), then compares four policies on the shared capability_eval
yardstick:
  * ree_bias_head   -- the warmed all-ON agent under its own bias-head selection
                       (capability_eval.REEForwardPolicy). The incompetence CONTROL (reproduces
                       the 724/719a ~0.1-0.5 res/ep deficit on the same env/seed).
  * ppo_ree_latent  -- a real PPO actor-critic trained on the agent's FROZEN z_world latent
                       (agent.sense(obs).z_world.detach()). THE H1 TREATMENT: does a proper
                       trainable policy head on REE's prediction-rich representation forage?
  * ppo_raw_obs     -- the same PPO actor trained on the RAW observation vector (734's
                       vanilla-PPO learner). REFERENCE: isolates whether the REE ENCODER
                       discards foraging signal that the raw obs retains.
  * greedy_oracle / random_walk -- ceiling (floor-achievability readiness) / floor anchors.

Two env rungs: D0 (the 724 baseline env where the deficit was localized) and D3 (hazard-free,
the fair "is the representation usable" test with the hazard confound removed) -- both imported
from 734's DIFFICULTY_RUNGS so they are byte-identical to the sibling sweep.

PURPOSE: diagnostic (claim_ids=[]); promotes/demotes nothing; brake-EXEMPT (a competence
localization probe, not a conversion/de-commit falsifier; claim_ids=[] zeroes the re-derive
brake). It converts the campaign's central build decision from INFERENCE to MEASUREMENT: if a
trainable actor on the REE latent recovers competence, the fix is a policy-learning substrate
on the existing representation (H1); if only the raw-obs actor recovers, the encoder is lossy
(route the observation encoding, P-C / V3-EXQ-739); if neither recovers on a hazard-free
oracle-achievable env, the deeper issue is policy-learning adequacy (contrast the
V3-EXQ-738 local_view_greedy anchor, which DID forage the same env).

PRE-REGISTERED SELF-ROUTE (HYPOTHESIS, not a verdict -- adjudicate before any governance use):
  * READINESS fails (either rung's oracle below the floor, OR ree_bias_head already clears the
    floor at D0 so the premise is not reproduced) -> `substrate_not_ready_requeue`.
  * H1 CONFIRMED: readiness holds AND ppo_ree_latent clears the floor (strict-majority of
    seeds) at the hazard-free D3 rung -> `policy_head_on_latent_recovers_competence`. REE's
    latent IS a usable action-learning substrate; the bias-head action-learning mechanism was
    the deficit -> build a trainable policy head.
  * ENCODER LOSSY: readiness holds, ppo_ree_latent stays sub-floor at D3 but ppo_raw_obs
    clears -> `latent_lossy_raw_obs_recovers`. The frozen encoder discards foraging signal the
    raw obs keeps -> route the observation encoding (P-C), not (only) a policy head.
  * DEEPER / LEARNER-ADEQUACY: readiness holds but NEITHER ppo arm clears the floor at the
    hazard-free D3 -> `policy_learning_insufficient_or_deeper`. Even a real actor on either
    representation cannot forage an oracle-achievable env -> PPO under-powered or a deeper
    obstruction (weigh against V3-EXQ-738's local_view_greedy, which cleared the same env).

UNTRAINED-WORLD-ENCODER GUARD (wired 2026-07-19; DETECTION ONLY).
`x734._train_all_on_agent` trains e2, the lPFC bias head and the OFC devaluation head; NONE of
those optimizer groups covers ANY of the 61 `latent_stack` parameters, so
`split_encoder.world_encoder` receives no gradient and z_world is a FROZEN RANDOM PROJECTION at
initialisation (measured: 0/61 latent_stack changed, 0/4 world_encoder changed, max|delta| =
0.000e+00). Diagnosis: REE_assembly/evidence/planning/
zworld_bc_install_failure_V3-EXQ-780_2026-07-19.md. The guard is wired at THIS script's call
site, NOT inside x734's shared function -- three drivers with different premises call it.

POLICY FOR THIS SCRIPT: NON-STRICT (record loudly, do NOT refuse). Justification: this probe
asks whether a policy HEAD trained on top of z_world can reach competence -- a question about
the READOUT. Its arms are a latent PPO actor on z_world plus a raw-obs PPO control. A frozen
random projection does not make the run uninterpretable; it RE-LABELS the question from "is the
learned z_world sufficient" to "is a frozen random projection sufficient", which is still
informative -- and the ppo_raw_obs control is entirely unaffected, so the discriminating
comparison survives intact. Refusing the arm would destroy a usable result; scoring it silently
as if z_world were learned would license a wrong claim. So: record, do not raise.

WHERE THE GUARD ENTRY LIVES, AND WHY NOT IN `preconditions[]`. The REE_assembly indexer reads
`interpretation.preconditions` FLAT and returns whole-run `precondition_unmet` on the first
unmet entry. Since this run is deliberately NOT refused, putting an unmet guard entry in that
adjudicating list would flag the entire run as precondition-unmet and bury it -- the opposite
of the intent. The guard entry is therefore carried under the non-adjudicating
`interpretation.recorded_preconditions` (plus `diagnostics.zworld_encoder_guard`), with
`interpretation.preconditions_scope_note` stating that it is recorded rather than gating and
why. It still carries honest `measured`/`threshold`/`met` so any recompute agrees with the
guard's own verdict. `substrate_not_ready_requeue` is NOT set by the guard and
`non_degenerate` is NOT falsified by it: the run is interpretable.

This module is ASCII-only in all runtime strings.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiments._lib.capability_eval import (  # noqa: E402
    COMPETENCE_RESOURCE_FLOOR,
    OraclePolicy,
    Policy,
    RandomPolicy,
    REEForwardPolicy,
    evaluate_seed,
)
from experiments._lib.zworld_encoder_guard import (  # noqa: E402
    assert_world_encoder_trained,
    latent_stack_snapshot,
    latent_stack_weight_delta,
    zworld_precondition,
)
import experiments.v3_exq_724_competence_localization_diagnostic as x724  # noqa: E402
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_737_ree_latent_policy_head_competence_probe"
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# Multi-arm run (writes per_rung x arm cells) but the learner arms TRAIN per (rung, seed) --
# nothing is a pure function of a shared baseline config that a later run could reuse (each
# arm re-warms the REE stack / re-trains a PPO actor from scratch). No baseline to bank.
ARM_FINGERPRINT_EXEMPT = "per-cell REE warmup + PPO training from scratch; no reusable baseline arm to bank"

SEEDS: List[int] = [42, 43, 44]
# Reuse the exact 724-A0 recipe + PPO budget from 734 so arms are comparable across siblings.
ZWORLD_P0_EPISODES = x734.ZWORLD_P0_EPISODES     # 60; SD-070 z_world encoder warmup (P0a).
                                                 # Sourced from x734 so this driver cannot
                                                 # drift from the function it imports.
P0_WARMUP_EPISODES = x734.P0_WARMUP_EPISODES     # 200
P1_REINFORCE_EPISODES = x734.P1_REINFORCE_EPISODES  # 90
P1_PPO_EPISODES = x734.P1_PPO_EPISODES           # 1000
EVAL_EPISODES = x734.EVAL_EPISODES               # 20
STEPS_PER_EPISODE = x734.STEPS_PER_EPISODE       # 200
PPO_ROLLOUT_EPISODES = x734.PPO_ROLLOUT_EPISODES  # 8

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_PPO = 6
DRY_RUN_EVAL = 2
DRY_RUN_STEPS = 20
DRY_RUN_ROLLOUT = 3

# Two rungs: D0 (724 baseline, deficit localized here) + D3 (hazard-free, fair usability test).
RUNGS = [x734.DIFFICULTY_RUNGS[0], x734.DIFFICULTY_RUNGS[-1]]
D0_RUNG_ID = RUNGS[0]["rung_id"]
D3_RUNG_ID = RUNGS[-1]["rung_id"]

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Latent state extraction: z_world of the (frozen) warmed all-ON agent.
# ---------------------------------------------------------------------------
def _agent_zworld(agent, obs_dict: Dict[str, Any]) -> torch.Tensor:
    body = obs_dict["body_state"].float()
    world = obs_dict["world_state"].float()
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    with torch.no_grad():
        latent = agent.sense(
            obs_body=body,
            obs_world=world,
            obs_harm=x724._obs_harm(obs_dict),
            obs_harm_a=x724._obs_harm_a(obs_dict),
            obs_harm_history=x724._obs_harm_history(obs_dict),
        )
    return latent.z_world.detach().reshape(1, -1).to(DEVICE)


# ---------------------------------------------------------------------------
# Generic PPO trainer -- mirrors x734._train_ppo but with a pluggable state_fn +
# per-episode on_reset hook (so it trains on z_world OR raw obs). Reuses 734's
# validated _ppo_update / _compute_gae / _RunningStd / _novelty_bonus verbatim.
# ---------------------------------------------------------------------------
def _train_ppo_generic(
    env: CausalGridWorldV2,
    policy: "x734.PPOPolicyNet",
    optimiser: torch.optim.Optimizer,
    state_fn,
    on_reset,
    n_episodes: int,
    rollout_episodes: int,
    steps_per_episode: int,
    arm_label: str,
    rung_id: str,
    seed: int,
    total_denominator: int,
) -> None:
    reward_std = x734._RunningStd()
    novelty_counter: Dict[Tuple[int, int], int] = {}
    episodes_done = 0

    while episodes_done < n_episodes:
        batch_states: List[torch.Tensor] = []
        batch_actions: List[int] = []
        batch_old_logp: List[float] = []
        batch_returns: List[float] = []
        batch_advantages: List[float] = []

        eps_this_batch = min(rollout_episodes, n_episodes - episodes_done)
        for _b in range(eps_this_batch):
            _, obs_dict = env.reset()
            if on_reset is not None:
                on_reset()
            ep_states: List[torch.Tensor] = []
            ep_actions: List[int] = []
            ep_logp: List[float] = []
            ep_values: List[float] = []
            ep_rewards: List[float] = []
            terminal = False
            bootstrap_value = 0.0
            state = state_fn(obs_dict)
            for _step in range(steps_per_episode):
                with torch.no_grad():
                    logits, value = policy(state)
                    dist = torch.distributions.Categorical(logits=logits.reshape(1, -1))
                    a = dist.sample()
                    logp = dist.log_prob(a)
                a_idx = int(a.item())
                _, harm_signal, done, info, obs_dict = env.step(a_idx)
                ttype = str(info.get("transition_type", "none"))
                pos = (int(env.agent_x), int(env.agent_y))
                shaped = (
                    float(harm_signal)
                    + (x734.FORAGE_BONUS if ttype == "resource" else 0.0)
                    + x734._novelty_bonus(novelty_counter, pos)
                )
                reward_std.update(shaped)
                ep_states.append(state.reshape(-1).detach())
                ep_actions.append(a_idx)
                ep_logp.append(float(logp.item()))
                ep_values.append(float(value.item()))
                ep_rewards.append(shaped)
                if done:
                    terminal = True
                    break
                state = state_fn(obs_dict)
            if not terminal:
                with torch.no_grad():
                    _, bv = policy(state_fn(obs_dict))
                bootstrap_value = float(bv.item())
            scale = reward_std.std + x734.REWARD_STD_EPS
            scaled_rewards = [r / scale for r in ep_rewards]
            advs, rets = x734._compute_gae(scaled_rewards, ep_values, bootstrap_value, terminal)
            batch_states.extend(ep_states)
            batch_actions.extend(ep_actions)
            batch_old_logp.extend(ep_logp)
            batch_returns.extend(rets)
            batch_advantages.extend(advs)
            episodes_done += 1
            if episodes_done % 200 == 0 or episodes_done == n_episodes:
                print(
                    f"  [train] {arm_label} rung={rung_id} seed={seed} phase=P1 "
                    f"ep {episodes_done}/{total_denominator}", flush=True,
                )

        if not batch_states:
            continue
        states_t = torch.stack(batch_states).to(DEVICE)
        actions_t = torch.tensor(batch_actions, dtype=torch.long, device=DEVICE)
        old_logp_t = torch.tensor(batch_old_logp, dtype=torch.float32, device=DEVICE)
        returns_t = torch.tensor(batch_returns, dtype=torch.float32, device=DEVICE)
        adv_t = torch.tensor(batch_advantages, dtype=torch.float32, device=DEVICE)
        x734._ppo_update(policy, optimiser, states_t, actions_t, old_logp_t, returns_t, adv_t, DEVICE)


class LatentPPOEvalPolicy(Policy):
    """Greedy (argmax) eval of a PPO actor trained on the agent's frozen z_world latent."""

    name = "ppo_ree_latent"

    def __init__(self, policy_net, agent) -> None:
        self.policy_net = policy_net
        self.agent = agent

    def reset(self, env: Any) -> None:
        self.agent.reset()

    def act(self, env: Any, obs_dict: Dict[str, Any]) -> int:
        state = _agent_zworld(self.agent, obs_dict)
        with torch.no_grad():
            logits, _v = self.policy_net(state)
        if not torch.isfinite(logits).all():
            return int(np.random.randint(0, int(env.action_dim)))
        return int(torch.argmax(logits.reshape(-1), dim=-1).item())


def _summ(foragings: List[float]) -> Dict[str, Any]:
    n = len(foragings)
    n_supra = int(sum(1 for f in foragings if f >= COMPETENCE_RESOURCE_FLOOR))
    return {
        "foraging_competence_mean": round(float(sum(foragings) / n), 6) if n else 0.0,
        "foraging_competence_per_seed": [round(f, 6) for f in foragings],
        "n_seeds": n,
        "n_seeds_supra_floor": n_supra,
        "majority_supra_floor": bool(n_supra >= (n + 1) // 2) if n else False,
    }


def _run_cell(
    rung: Dict[str, Any],
    seed: int,
    p0: int,
    p1: int,
    ppo_eps: int,
    eval_eps: int,
    steps: int,
    rollout: int,
    zworld_p0: int = 0,
    dry_run: bool = False,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """One (rung, seed) cell: warm the all-ON REE stack, then eval/train the four arms.

    Returns (results, guard_report). `results` keeps its original meaning EXACTLY -- an
    arm -> foraging_competence scalar map. `guard_report` is the untrained-world-encoder audit
    record for this cell (see the module docstring); it is threaded out rather than folded into
    `results` so nothing about `results[arm]` changes.
    """
    env_kwargs = x734._env_kwargs_for_rung(rung)
    rid = rung["rung_id"]
    total_denom = p0 + p1

    # --- warm the all-ON REE stack (724-A0 recipe) on THIS rung's env -----------
    torch.manual_seed(seed)
    np.random.seed(seed)
    warm_env = x734._make_env(seed, env_kwargs)
    agent = x734._make_all_on_agent(warm_env)
    print(f"Seed {seed} Condition {rid}:warmup_all_on", flush=True)
    before = latent_stack_snapshot(agent)
    # SD-070 P0a encoder warmup ON. Without it the guard below measures 0 of 4 world_encoder
    # tensors changed and this probe's latent policy head reads a frozen random projection --
    # the V3-EXQ-737a finding this driver produced. Dedicated env: the P0a rollout consumes
    # env RNG, so reusing warm_env would shift the layout sequence P0b/P1 then see.
    x734._train_all_on_agent(
        agent, warm_env, seed=seed, p0_episodes=p0, p1_episodes=p1,
        steps_per_episode=steps, rung_id=rid, total_denominator=total_denom,
        zworld_p0_episodes=zworld_p0,
        zworld_p0_env=(x734._make_env(seed, env_kwargs) if zworld_p0 > 0 else None),
        zworld_p0_dry_run=dry_run,
    )
    guard_report = latent_stack_weight_delta(agent, before)
    guard_report["p0_episodes"] = int(p0)
    guard_report["guard_checked"] = bool(p0 > 0 and before)
    # NON-STRICT by design: prints the unmissable GUARD-WARNING and returns, never raises.
    # See the module docstring for why a frozen projection still yields an interpretable
    # readout-side result here.
    assert_world_encoder_trained(
        agent, before, p0=p0, strict=False,
        context=f"V3-EXQ-737a rung={rid} seed={seed}",
        escape_hint=(
            "this arm is recorded, not refused: see the module docstring for why a frozen "
            "projection still yields an interpretable readout-side result"
        ),
    )
    guard_report["rung_id"] = str(rid)
    guard_report["seed"] = int(seed)

    # z_world dim (probe) for the latent PPO actor.
    _, probe_obs = x734._make_env(seed, env_kwargs).reset()
    z_dim = int(_agent_zworld(agent, probe_obs).shape[-1])

    results: Dict[str, float] = {}

    # --- arm 1: ree_bias_head (incompetence control) ----------------------------
    print(f"Seed {seed} Condition {rid}:ree_bias_head", flush=True)
    eval_env = x734._make_env(seed, env_kwargs)
    row = evaluate_seed(REEForwardPolicy(agent, name="ree_bias_head"), eval_env, eval_eps, steps)
    results["ree_bias_head"] = float(row["foraging_competence"])
    print(f"verdict: {'PASS' if row['competence_supra_floor'] else 'FAIL'}", flush=True)

    # --- arm 2: ppo_ree_latent (H1 treatment) -----------------------------------
    print(f"Seed {seed} Condition {rid}:ppo_ree_latent", flush=True)
    torch.manual_seed(seed + 1000)
    latent_net = x734.PPOPolicyNet(in_dim=z_dim, action_dim=int(warm_env.action_dim)).to(DEVICE)
    latent_opt = torch.optim.Adam(latent_net.parameters(), lr=x734.PPO_LR)
    train_env = x734._make_env(seed, env_kwargs)
    _train_ppo_generic(
        train_env, latent_net, latent_opt,
        state_fn=lambda od: _agent_zworld(agent, od),
        on_reset=agent.reset,
        n_episodes=ppo_eps, rollout_episodes=rollout, steps_per_episode=steps,
        arm_label="ppo_ree_latent", rung_id=rid, seed=seed, total_denominator=ppo_eps,
    )
    eval_env = x734._make_env(seed, env_kwargs)
    row = evaluate_seed(LatentPPOEvalPolicy(latent_net, agent), eval_env, eval_eps, steps)
    results["ppo_ree_latent"] = float(row["foraging_competence"])
    print(f"verdict: {'PASS' if row['competence_supra_floor'] else 'FAIL'}", flush=True)

    # --- arm 3: ppo_raw_obs (encoder-loss reference) ----------------------------
    print(f"Seed {seed} Condition {rid}:ppo_raw_obs", flush=True)
    torch.manual_seed(seed + 2000)
    _, probe2 = x734._make_env(seed, env_kwargs).reset()
    obs_keys = x734._raw_obs_keys_present(probe2)
    raw_dim = int(x734._raw_obs_vector(probe2, obs_keys, DEVICE).shape[-1])
    raw_net = x734.PPOPolicyNet(in_dim=raw_dim, action_dim=int(warm_env.action_dim)).to(DEVICE)
    raw_opt = torch.optim.Adam(raw_net.parameters(), lr=x734.PPO_LR)
    train_env = x734._make_env(seed, env_kwargs)
    _train_ppo_generic(
        train_env, raw_net, raw_opt,
        state_fn=lambda od: x734._raw_obs_vector(od, obs_keys, DEVICE),
        on_reset=None,
        n_episodes=ppo_eps, rollout_episodes=rollout, steps_per_episode=steps,
        arm_label="ppo_raw_obs", rung_id=rid, seed=seed, total_denominator=ppo_eps,
    )
    eval_env = x734._make_env(seed, env_kwargs)
    row = evaluate_seed(x734.PPOEvalPolicy(raw_net, obs_keys, DEVICE), eval_env, eval_eps, steps)
    results["ppo_raw_obs"] = float(row["foraging_competence"])
    print(f"verdict: {'PASS' if row['competence_supra_floor'] else 'FAIL'}", flush=True)

    # --- anchors: greedy_oracle (ceiling) + random_walk (floor) ------------------
    print(f"Seed {seed} Condition {rid}:greedy_oracle", flush=True)
    eval_env = x734._make_env(seed, env_kwargs)
    row = evaluate_seed(OraclePolicy(), eval_env, eval_eps, steps)
    results["greedy_oracle"] = float(row["foraging_competence"])
    print(f"verdict: {'PASS' if row['competence_supra_floor'] else 'FAIL'}", flush=True)

    print(f"Seed {seed} Condition {rid}:random_walk", flush=True)
    eval_env = x734._make_env(seed, env_kwargs)
    row = evaluate_seed(RandomPolicy(seed), eval_env, eval_eps, steps)
    results["random_walk"] = float(row["foraging_competence"])
    print(f"verdict: {'PASS' if row['competence_supra_floor'] else 'FAIL'}", flush=True)

    return results, guard_report


ARM_ORDER = ["ree_bias_head", "ppo_ree_latent", "ppo_raw_obs", "greedy_oracle", "random_walk"]


def run_experiment(
    seeds: List[int],
    p0: int,
    p1: int,
    ppo_eps: int,
    eval_eps: int,
    steps: int,
    rollout: int,
    zworld_p0: int = 0,
    dry_run: bool = False,
) -> Dict[str, Any]:
    # per_rung[rid][arm] = list of per-seed foraging values
    per_seed_forage: Dict[str, Dict[str, List[float]]] = {r["rung_id"]: {a: [] for a in ARM_ORDER} for r in RUNGS}
    guard_reports: List[Dict[str, Any]] = []
    for rung in RUNGS:
        rid = rung["rung_id"]
        for seed in seeds:
            cell, guard_report = _run_cell(
                rung, seed, p0, p1, ppo_eps, eval_eps, steps, rollout,
                zworld_p0=zworld_p0, dry_run=dry_run,
            )
            guard_reports.append(guard_report)
            for arm in ARM_ORDER:
                per_seed_forage[rid][arm].append(cell[arm])

    checked = [g for g in guard_reports if g.get("guard_checked")]
    any_trained = bool(any(g.get("zworld_encoder_trained") for g in checked))
    all_frozen = bool(checked) and not any(g.get("zworld_encoder_trained") for g in checked)
    zworld_guard = {
        "policy": "record_not_refuse",
        "policy_reason": (
            "This probe asks whether a policy HEAD on top of z_world can reach competence -- a "
            "question about the readout. A frozen random projection re-labels the question from "
            "'is the learned z_world sufficient' to 'is a frozen random projection sufficient', "
            "which is still informative, and the ppo_raw_obs control is unaffected and remains "
            "the discriminating comparison. Refusing would destroy a usable result; silently "
            "scoring it as a learned representation would license a wrong claim."
        ),
        "detection_only": True,
        "guard_site": (
            "this driver's x734._train_all_on_agent call site in _run_cell, NOT inside x734's "
            "shared function (three drivers with different premises call it)"
        ),
        "n_cells": len(guard_reports),
        "n_cells_checked": len(checked),
        "any_trained": any_trained,
        "all_frozen": all_frozen,
        "per_cell": guard_reports,
    }

    per_rung: Dict[str, Dict[str, Any]] = {}
    for rid in per_seed_forage:
        per_rung[rid] = {arm: _summ(per_seed_forage[rid][arm]) for arm in ARM_ORDER}

    def _mean(rid: str, arm: str) -> float:
        return float(per_rung[rid][arm]["foraging_competence_mean"])

    def _maj(rid: str, arm: str) -> bool:
        return bool(per_rung[rid][arm]["majority_supra_floor"])

    # ---- readiness -----------------------------------------------------------
    d0_oracle_ok = _mean(D0_RUNG_ID, "greedy_oracle") >= COMPETENCE_RESOURCE_FLOOR
    d3_oracle_ok = _mean(D3_RUNG_ID, "greedy_oracle") >= COMPETENCE_RESOURCE_FLOOR
    bias_head_reproduces = not _maj(D0_RUNG_ID, "ree_bias_head")  # premise: incompetent at D0
    readiness_met = bool(d0_oracle_ok and d3_oracle_ok and bias_head_reproduces)

    # ---- load-bearing: ppo_ree_latent clears floor at the hazard-free D3 -------
    latent_recovers_d3 = _maj(D3_RUNG_ID, "ppo_ree_latent")
    raw_recovers_d3 = _maj(D3_RUNG_ID, "ppo_raw_obs")

    if not readiness_met:
        outcome, label = "FAIL", "substrate_not_ready_requeue"
    elif latent_recovers_d3:
        outcome, label = "PASS", "policy_head_on_latent_recovers_competence"
    elif raw_recovers_d3:
        outcome, label = "FAIL", "latent_lossy_raw_obs_recovers"
    else:
        outcome, label = "FAIL", "policy_learning_insufficient_or_deeper"

    if all_frozen:
        zworld_arm_reading = (
            "READ THE ppo_ree_latent ARM AS 'PPO ON A FROZEN RANDOM PROJECTION', NOT AS 'PPO ON "
            "A LEARNED REE REPRESENTATION'. The untrained-world-encoder guard measured 0 changed "
            "split_encoder.world_encoder tensors in every checked cell, so z_world was never "
            "prediction-trained: the P0/P1 warmup has no optimizer group covering latent_stack. "
            "Any result on this arm speaks to what a frozen random projection of the observation "
            "supports, not to what REE's learned world representation supports. The ppo_raw_obs "
            "control is unaffected and remains the discriminating comparison."
        )
    elif any_trained:
        zworld_arm_reading = (
            "The untrained-world-encoder guard found at least one cell whose world encoder "
            "moved; check diagnostics.zworld_encoder_guard.per_cell before reading the "
            "ppo_ree_latent arm as a learned-representation result in any given cell."
        )
    else:
        zworld_arm_reading = (
            "The untrained-world-encoder guard did not run (no warmup episodes), so the "
            "ppo_ree_latent arm carries no evidence that z_world was prediction-trained."
        )

    interpretation = {
        "label": label,
        "label_qualifier": (
            "zworld_arm_ran_on_frozen_random_projection" if all_frozen else "zworld_encoder_state_see_guard"
        ),
        "zworld_arm_reading": zworld_arm_reading,
        "preconditions_scope_note": (
            "The zworld_world_encoder_trained guard entries are carried under "
            "'recorded_preconditions' (and diagnostics.zworld_encoder_guard), NOT under the "
            "adjudicating flat 'preconditions' list. For this probe the guard is RECORDED, not "
            "GATING: the question is readout-side, a frozen random projection re-labels the "
            "z_world arm rather than voiding the run, and the ppo_raw_obs control is unaffected. "
            "The REE_assembly indexer returns whole-run precondition_unmet on the first unmet "
            "flat entry, so an unmet guard entry there would bury an interpretable run -- the "
            "opposite of the intent. The recorded entries still carry honest "
            "measured/threshold/met so any recompute agrees with the guard's own verdict."
        ),
        "recorded_preconditions": [
            zworld_precondition(
                g,
                arm="ppo_ree_latent",
                context=f"V3-EXQ-737a rung={g.get('rung_id', '')} seed={g.get('seed', '')}",
            )
            for g in guard_reports
        ],
        "preconditions": [
            {
                "name": "d0_oracle_clears_floor",
                "description": "D0 (724 baseline) env must be floor-achievable with global info.",
                "control": "greedy_oracle on D0 vs the 1.0 floor",
                "measured": round(_mean(D0_RUNG_ID, "greedy_oracle"), 6),
                "threshold": float(COMPETENCE_RESOURCE_FLOOR),
                "met": bool(d0_oracle_ok),
            },
            {
                "name": "d3_oracle_clears_floor",
                "description": "Hazard-free D3 env must be floor-achievable with global info.",
                "control": "greedy_oracle on D3 vs the 1.0 floor",
                "measured": round(_mean(D3_RUNG_ID, "greedy_oracle"), 6),
                "threshold": float(COMPETENCE_RESOURCE_FLOOR),
                "met": bool(d3_oracle_ok),
            },
            {
                "name": "bias_head_reproduces_incompetence_at_d0",
                "description": (
                    "The warmed all-ON bias-head agent must forage BELOW the floor at D0 "
                    "(reproduce the 724/719a deficit) for the H1 premise to hold; if it already "
                    "clears the floor the deficit is not present and no H1 read is licensed."
                ),
                "control": "ree_bias_head D0 mean resources/ep vs floor (strict majority of seeds)",
                "measured": round(_mean(D0_RUNG_ID, "ree_bias_head"), 6),
                "threshold": float(COMPETENCE_RESOURCE_FLOOR),
                "direction": "upper",  # premise MET when measured < threshold (stays below floor)
                "met": bool(bias_head_reproduces),
            },
        ],
        "criteria_non_degenerate": {
            "ppo_ree_latent_beats_random_floor_at_d3": bool(_mean(D3_RUNG_ID, "ppo_ree_latent") > _mean(D3_RUNG_ID, "random_walk")),
            "random_floor_below_competence_floor_at_d3": bool(_mean(D3_RUNG_ID, "random_walk") < COMPETENCE_RESOURCE_FLOOR),
            "bias_head_below_oracle_at_d3": bool(_mean(D3_RUNG_ID, "ree_bias_head") < _mean(D3_RUNG_ID, "greedy_oracle")),
        },
        "criteria": [
            {
                "name": "C_ppo_ree_latent_clears_floor_at_D3",
                "load_bearing": True,
                "passed": bool(latent_recovers_d3),
            }
        ],
    }

    return {
        "outcome": outcome,
        "interpretation": interpretation,
        "diagnostics": {"zworld_encoder_guard": zworld_guard},
        "per_rung": per_rung,
        "readiness": {
            "readiness_met": readiness_met,
            "d0_oracle_clears_floor": d0_oracle_ok,
            "d3_oracle_clears_floor": d3_oracle_ok,
            "bias_head_reproduces_incompetence_at_d0": bias_head_reproduces,
        },
        "headline": {
            "d3_ppo_ree_latent_forage": round(_mean(D3_RUNG_ID, "ppo_ree_latent"), 6),
            "d3_ppo_ree_latent_clears_majority": latent_recovers_d3,
            "d3_ppo_raw_obs_forage": round(_mean(D3_RUNG_ID, "ppo_raw_obs"), 6),
            "d3_ppo_raw_obs_clears_majority": raw_recovers_d3,
            "d3_ree_bias_head_forage": round(_mean(D3_RUNG_ID, "ree_bias_head"), 6),
            "d0_ppo_ree_latent_forage": round(_mean(D0_RUNG_ID, "ppo_ree_latent"), 6),
            "d0_ree_bias_head_forage": round(_mean(D0_RUNG_ID, "ree_bias_head"), 6),
            "d3_greedy_oracle_forage": round(_mean(D3_RUNG_ID, "greedy_oracle"), 6),
        },
    }


def _build_manifest(result: Dict[str, Any], timestamp_utc: str, dry_run: bool) -> Dict[str, Any]:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    return {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": [],
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "brake_exempt": True,
        "brake_exempt_reason": "competence localization probe; claim_ids=[]; not a conversion/de-commit falsifier",
        "timestamp_utc": timestamp_utc,
        "dry_run": bool(dry_run),
        "outcome": result["outcome"],
        "interpretation": result["interpretation"],
        "interpretation_label": result["interpretation"]["label"],
        "diagnostics": result["diagnostics"],
        "readiness": result["readiness"],
        "headline": result["headline"],
        "per_rung": result["per_rung"],
        "config": {
            "seeds": SEEDS if not dry_run else DRY_RUN_SEEDS,
            "rungs": [r["rung_id"] for r in RUNGS],
            "arms": ARM_ORDER,
            "zworld_p0_episodes": (
                ZWORLD_P0_EPISODES if not dry_run else x734.DRY_RUN_ZWORLD_P0
            ),
            "p0_warmup_episodes": P0_WARMUP_EPISODES if not dry_run else DRY_RUN_P0,
            "p1_reinforce_episodes": P1_REINFORCE_EPISODES if not dry_run else DRY_RUN_P1,
            "p1_ppo_episodes": P1_PPO_EPISODES if not dry_run else DRY_RUN_PPO,
            "eval_episodes": EVAL_EPISODES if not dry_run else DRY_RUN_EVAL,
            "steps_per_episode": STEPS_PER_EPISODE if not dry_run else DRY_RUN_STEPS,
            "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
        },
        "load_bearing_dv": (
            "D3 ppo_ree_latent mean resources/ep (PPO actor on the frozen REE z_world) vs the "
            "1.0 competence floor, strict majority of seeds"
        ),
        "notes": (
            "Direct H1 test of the 724 'prediction-rich, action-poor' localization: bolts a real "
            "trainable PPO actor onto REE's OWN frozen z_world latent (ppo_ree_latent) and asks "
            "whether it recovers foraging where the bias-head stack (ree_bias_head) cannot. "
            "ppo_raw_obs isolates encoder-loss. PASS (policy_head_on_latent_recovers_competence) "
            "= H1 confirmed, build a policy-learning substrate on the existing representation. "
            "Brake-exempt; PROMOTES/DEMOTES NOTHING. Route result to /failure-autopsy before any "
            "governance use. Sibling of 738 (local-view anchor, already showed raw obs is "
            "forageable) and 739 (encoder probe, reserve). "
            "UNTRAINED-WORLD-ENCODER GUARD: wired at this driver's x734._train_all_on_agent call "
            "site, DETECTION ONLY, policy RECORD-not-refuse. See "
            "diagnostics.zworld_encoder_guard and interpretation.zworld_arm_reading before "
            "reading the ppo_ree_latent arm -- if the encoder never moved, that arm is PPO on a "
            "FROZEN RANDOM PROJECTION, not on a learned REE representation."
        ),
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description=(
            "V3-EXQ-737 trainable policy head on the REE latent DIAGNOSTIC "
            "(does a real actor on z_world recover competence; H1; claim_ids=[])"
        )
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    _run_started = datetime.now(timezone.utc)

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        p0, p1, ppo = DRY_RUN_P0, DRY_RUN_P1, DRY_RUN_PPO
        eval_eps, steps, rollout = DRY_RUN_EVAL, DRY_RUN_STEPS, DRY_RUN_ROLLOUT
        zworld_p0 = x734.DRY_RUN_ZWORLD_P0
    else:
        seeds = list(SEEDS)
        p0, p1, ppo = P0_WARMUP_EPISODES, P1_REINFORCE_EPISODES, P1_PPO_EPISODES
        eval_eps, steps, rollout = EVAL_EPISODES, STEPS_PER_EPISODE, PPO_ROLLOUT_EPISODES
        zworld_p0 = ZWORLD_P0_EPISODES

    result = run_experiment(
        seeds=seeds, p0=p0, p1=p1, ppo_eps=ppo, eval_eps=eval_eps, steps=steps,
        rollout=rollout, zworld_p0=zworld_p0, dry_run=bool(args.dry_run),
    )

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run))

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=bool(args.dry_run),
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
        elapsed_seconds=(datetime.now(timezone.utc) - _run_started).total_seconds(),
    )

    print(f"manifest: {out_path}", flush=True)
    if not args.dry_run:
        print(f"Result written to: {out_path}", flush=True)
    hl = result["headline"]
    rd = result["readiness"]
    print(
        f"outcome: {result['outcome']} label={result['interpretation']['label']} "
        f"readiness_met={rd['readiness_met']} "
        f"d3_ppo_latent/ep={hl['d3_ppo_ree_latent_forage']} (clears={hl['d3_ppo_ree_latent_clears_majority']}) "
        f"d3_ppo_raw/ep={hl['d3_ppo_raw_obs_forage']} (clears={hl['d3_ppo_raw_obs_clears_majority']}) "
        f"d3_bias_head/ep={hl['d3_ree_bias_head_forage']} d3_oracle/ep={hl['d3_greedy_oracle_forage']}",
        flush=True,
    )
    for rung in RUNGS:
        rid = rung["rung_id"]
        pr = result["per_rung"][rid]
        print(
            f"  RUNG {rid}: bias_head/ep={pr['ree_bias_head']['foraging_competence_mean']} "
            f"ppo_latent/ep={pr['ppo_ree_latent']['foraging_competence_mean']} "
            f"(supra {pr['ppo_ree_latent']['n_seeds_supra_floor']}/{pr['ppo_ree_latent']['n_seeds']}) "
            f"ppo_raw/ep={pr['ppo_raw_obs']['foraging_competence_mean']} "
            f"(supra {pr['ppo_raw_obs']['n_seeds_supra_floor']}/{pr['ppo_raw_obs']['n_seeds']}) "
            f"oracle/ep={pr['greedy_oracle']['foraging_competence_mean']} "
            f"random/ep={pr['random_walk']['foraging_competence_mean']}",
            flush=True,
        )

    if args.dry_run:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass

    outcome_norm = result["outcome"].upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    manifest_for_sentinel = str(out_path) if not args.dry_run else None
    return outcome_emit, manifest_for_sentinel, bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _manifest_path, _dry_run = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path, dry_run=_dry_run)
