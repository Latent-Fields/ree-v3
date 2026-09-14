"""V3-EXQ-1021: Survive, recover, resume -- a non-evidential V3 showcase.

User-authorized 2026-09-11 exception to queue-experiment Step 2.5c: keep the
mode-governance, E1 rollout, and ContextMemory limitations visible. No claim tags.
SLEEP DRIVER: not applicable (sleep disabled; waking evaluation only).
Red-team: pending independent design review before queueing.

P0: SD-070 world encoder + affective-history auxiliary; P1: frozen-encoder
world/self prediction and harm/benefit heads; P2: no optimizer, no guidance.
Select goal weight using development seeds ONLY. Five held-out trained agents,
12 maps each. Compare REE, identical trained clone without event resets,
memoryless local-field policy, and random. A privileged control calibrates only.
Reset removal changes timing AND frequency; this is the total reset contribution.

DV symmetry: permuting episode order preserves mean safe-recovery success;
changing reset execution or policy changes ordered actions and physical outcomes,
not an invariant relabeling, broadcast offset or monotone score rescaling.
All attempted episodes count; death/non-recovery has success=0 and latency=100.
Never infer competence from signal variance, or sentience from affect telemetry.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import random
import sys
import time
import types
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments._harness import StepHarness, StepHooks
from experiments._lib.arm_fingerprint import arm_cell, reset_all_rng
from experiments._lib.manifest_core import stamp_recording_core
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiments._lib.zworld_p0_warmup import run_zworld_p0
from experiments.pack_writer import write_flat_manifest
from experiment_protocol import emit_outcome

EXPERIMENT_TYPE = "v3_exq_1021_survive_recover_resume_showcase"
EXPERIMENT_PURPOSE = "baseline"
QUEUE_ID = "V3-EXQ-1021"
CLAIM_IDS = []
DEV_SEEDS = [42, 43, 44]
SEEDS = [107, 211, 307, 401, 503]
ARMS = ["ree", "no_reset", "reactive", "random"]
GOAL_WEIGHTS = [0.5, 1.0]
REACTIVE_WEIGHTS = [1.0, 3.0]
EPISODES = 12
STEPS = 300
TRAIN_BUDGET = {"world": 60, "affect": 20, "heads": 60, "steps": 200}
ENV = dict(size=10, num_hazards=2, num_resources=8, use_proxy_fields=True,
           reef_enabled=True, n_reef_patches=2, reef_patch_radius=1,
           toroidal=False, contamination_spread=0.0, hazard_harm=0.04,
           contaminated_harm=0.04, resource_benefit=0.3, energy_decay=0.004,
           resource_respawn_on_consume=True, harm_history_len=10,
           env_drift_interval=1, env_drift_prob=0.05, hazard_food_attraction=0.0,
           proximity_harm_scale=0.001, proximity_benefit_scale=0.0,
           max_episode_steps=STEPS)
LIMITATIONS = ["mode-governance-engagement", "SD-e1-rollout-consistency-training",
               "contextmemory-write-path-addressing-degeneracy", "SD-018"]


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256("/".join(map(str, parts)).encode()).digest()[:4], "little")


def clean_obs(obs, contact=0.0):
    # Do not give any policy an explicit episode clock / phase indicator.
    obs["body_state"] = obs["body_state"].clone()
    obs["body_state"][9] = 0.0
    # Actual consumed benefit, not a proximity reward or stale EMA, seeds goals.
    obs["benefit_exposure"] = float(contact)
    return obs


class TimedReef(CausalGridWorldV2):
    """Existing ecology with paired RNG tapes and an exogenous threat schedule.

    Separate step/drift/respawn streams prevent consumption from shifting hazard
    randomness. Occupancy can still change realized paths, as it should.
    """
    def __init__(self, seed, steps=STEPS, pressure=0.5):
        self.tape_seed = int(seed)
        self.limit = int(steps)
        self.pressure = float(pressure)
        self.tape_episode = -1
        super().__init__(seed=seed, **dict(ENV, max_episode_steps=steps))

    def reset(self):
        self.tape_episode += 1
        self._rng = np.random.default_rng(seed_for(self.tape_seed, self.tape_episode, "layout"))
        self.tapes = {kind: [seed_for(self.tape_seed, self.tape_episode, kind, t)
                            for t in range(self.limit + 1)]
                      for kind in ("step", "drift", "respawn")}
        flat, obs = super().reset()
        return flat, clean_obs(obs)

    def _drift_hazards(self):
        saved = self._rng
        self._rng = np.random.default_rng(self.tapes["drift"][min(self.steps, self.limit)])
        try:
            return super()._drift_hazards()
        finally:
            self._rng = saved

    def _respawn_resource(self):
        saved = self._rng
        self._rng = np.random.default_rng(self.tapes["respawn"][min(self.steps, self.limit)])
        try:
            return super()._respawn_resource()
        finally:
            self._rng = saved

    def step(self, action):
        third = self.limit // 3
        self.env_drift_prob = self.pressure if third <= self.steps < 2 * third else 0.05
        self._rng = np.random.default_rng(self.tapes["step"][min(self.steps, self.limit)])
        flat, reward, done, info, obs = super().step(action)
        contact = reward if info["transition_type"] == "resource" else 0.0
        return flat, reward, done, info, clean_obs(obs, contact)


class LocalPolicy:
    """Memoryless; reads only the same local fields and local entity view as REE."""
    def __init__(self, seed, harm_weight=1.0, uniform=False):
        self.rng = np.random.default_rng(seed)
        self.harm_weight = harm_weight
        self.uniform = uniform

    def reset(self, env):
        pass

    def act(self, env, obs):
        local = obs["world_state"][:175].reshape(5, 5, 7)
        actions = list(range(env.action_dim))
        legal = [a for a in actions if local[2 + env.ACTIONS[a][0],
                                            2 + env.ACTIONS[a][1], 1] < 0.5]
        if self.uniform:
            return int(self.rng.choice(legal))
        harm = obs["hazard_field_view"].reshape(5, 5)
        food = obs["resource_field_view"].reshape(5, 5)
        scores = []
        for a in legal:
            dx, dy = env.ACTIONS[a]
            x, y = 2 + dx, 2 + dy
            # Tiny stay penalty, fixed before calibration, avoids an arbitrary
            # stay preference when the gradient is locally flat.
            scores.append(float(food[x, y] - self.harm_weight * harm[x, y]
                                + local[x, y, 2] - local[x, y, 3]) - 0.001 * (a == 4))
        best = np.flatnonzero(np.isclose(scores, max(scores), rtol=0, atol=1e-8))
        return legal[int(self.rng.choice(best))]


def privileged_action(env):
    """Full-map BFS resource planner; calibration only, never a peer benchmark."""
    origin = (env.agent_x, env.agent_y)
    targets = {tuple(r) for r in env.resources}
    queue = deque([(origin, None)])
    seen = {origin}
    while queue:
        (x, y), first = queue.popleft()
        if (x, y) in targets and first is not None:
            return first
        for a, (dx, dy) in list(env.ACTIONS.items())[:4]:
            n = (x + dx, y + dy)
            if n in seen or not (0 < n[0] < env.size - 1 and 0 < n[1] < env.size - 1):
                continue
            if env.grid[n] in (env.ENTITY_TYPES["hazard"], env.ENTITY_TYPES["contaminated"]):
                continue
            seen.add(n)
            queue.append((n, a if first is None else first))
    return 4


def make_agent(env, obs):
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim, self_dim=32, world_dim=32, alpha_world=0.9,
        use_harm_stream=True, use_affective_harm_stream=True,
        harm_obs_a_dim=int(obs["harm_obs_a"].numel()), harm_history_len=10,
        use_resource_proximity_head=True, benefit_eval_enabled=True,
        benefit_weight=1.0, z_goal_enabled=True, drive_weight=2.0, goal_weight=0.5,
        use_suffering_derivative_comparator=True,
        e2_action_contrastive_enabled=True)
    cfg.heartbeat.beta_gate_bistable = True
    # No optional tonic-vigor, blocked-agency, contextual-safety or sleep gates:
    # these add their own known unresolved calibration assumptions.
    return REEAgent(cfg)


def full_clone(agent):
    """Clone all Python-side state as well as weights, detaching tensor graphs.

    Some runtime helpers retain a Python module as RNG provider, which cannot be
    pickled; share module singletons only. Mutable tensors/arrays/state are copied.
    """
    seen, memo = set(), {}
    def walk(obj):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, types.ModuleType):
            memo[id(obj)] = obj
            return
        if isinstance(obj, torch.Tensor):
            value = obj.detach().clone()
            if isinstance(obj, torch.nn.Parameter):
                value = torch.nn.Parameter(value, requires_grad=False)
            memo[id(obj)] = value
            return
        if isinstance(obj, (type, types.FunctionType, types.MethodType, str, int, float, bool, type(None))):
            return
        if isinstance(obj, dict):
            values = obj.values()
        elif isinstance(obj, (list, tuple, set, deque)):
            values = obj
        elif hasattr(obj, "__dict__"):
            values = vars(obj).values()
        else:
            return
        for value in values:
            walk(value)
    walk(agent)
    result = copy.deepcopy(agent, memo)
    assert result.goal_state is not agent.goal_state
    assert result.residue_field is not agent.residue_field
    assert result.clock is not agent.clock
    for p, q in zip(agent.parameters(), result.parameters()):
        assert torch.equal(p, q) and p.data_ptr() != q.data_ptr()
    return result


def sense(agent, obs):
    return agent.sense(obs["body_state"], obs["world_state"],
                       obs_harm=obs["harm_obs"], obs_harm_a=obs["harm_obs_a"],
                       obs_harm_history=obs["harm_history"])


def train(seed, budget, pressure, dry_run):
    reset_all_rng(seed)
    env = TimedReef(seed_for(seed, "train"), budget["steps"], pressure)
    _, obs = env.reset()
    agent = make_agent(env, obs)
    total = budget["world"] + budget["affect"] + budget["heads"]
    diag = {"budget": budget, "label_balance": {}, "losses": {}}
    diag["world"] = run_zworld_p0(
        agent, TimedReef(seed_for(seed, "world"), budget["steps"], pressure),
        seed, budget["world"], budget["steps"],
        LocalPolicy(seed_for(seed, "world_policy"), uniform=True),
        label="showcase P0", dry_run=dry_run)
    done = budget["world"]
    print(f"[train] world seed={seed} ep {done}/{total}", flush=True)
    # P0 affective auxiliary before fitting any downstream prediction heads.
    affect_params = [p for name, p in agent.latent_stack.named_parameters()
                     if "affective_harm_encoder" in name]
    if not affect_params:
        raise RuntimeError("affective encoder parameter path is absent")
    opt = torch.optim.Adam(affect_params, lr=1e-3)
    losses, labels = [], []
    for ep in range(budget["affect"]):
        _, obs = env.reset()
        agent.reset()
        policy = LocalPolicy(seed_for(seed, "affect", ep), uniform=True)
        for _ in range(budget["steps"]):
            latent = sense(agent, obs)
            target = float(obs["accumulated_harm"])
            loss = agent.compute_harm_accum_loss(target, latent)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(affect_params, 1.0)
            opt.step()
            losses.append(float(loss.detach()))
            labels.append(target)
            _, _, dead, _, obs = env.step(policy.act(env, obs))
            if dead:
                break
        done += 1
        print(f"[train] affect seed={seed} ep {done}/{total}", flush=True)
    diag["losses"]["affect_mean"] = float(np.mean(losses))
    diag["label_balance"]["affect_positive_fraction"] = float(np.mean(np.asarray(labels) > 0.01))
    for p in agent.latent_stack.parameters():
        p.requires_grad_(False)
    encoder_before = {k: v.clone() for k, v in agent.latent_stack.state_dict().items()}
    head_params = list(agent.e1.parameters()) + list(agent.e2.parameters())
    head_params += list(agent.e3.harm_eval_head.parameters()) + list(agent.e3.benefit_eval_head.parameters())
    opt = torch.optim.Adam(head_params, lr=1e-3)
    buffer = deque(maxlen=2048)
    rng = random.Random(seed_for(seed, "minibatch"))
    losses, harm_labels, food_labels = [], [], []
    for ep in range(budget["heads"]):
        _, obs = env.reset()
        agent.reset()
        policy = LocalPolicy(seed_for(seed, "heads", ep), uniform=ep % 2 == 0)
        previous = None
        for t in range(budget["steps"]):
            with torch.no_grad():
                latent = sense(agent, obs)
                agent.update_z_goal(benefit_exposure=obs["benefit_exposure"],
                                    drive_level=agent.compute_drive_level(obs["body_state"]))
            harm = float(obs["hazard_field_view"][12])
            food = float(obs["resource_field_view"][12])
            harm_labels.append(harm)
            food_labels.append(food)
            if previous is not None:
                zw, zs, action = previous
                buffer.append((zw, zs, action, latent.z_world.detach().clone(),
                               latent.z_self.detach().clone()))
            # Only detached latents reach downstream heads.
            loss = F.mse_loss(agent.e3.harm_eval_head(latent.z_world.detach()),
                              torch.full((1, 1), harm))
            loss = loss + F.mse_loss(agent.e3.benefit_eval_head(latent.z_world.detach()),
                                    torch.full((1, 1), food))
            if len(buffer) >= 8:
                batch = rng.sample(list(buffer), min(32, len(buffer)))
                zw, zs, ac, nw, ns = [torch.cat([row[k] for row in batch]) for k in range(5)]
                loss = loss + F.mse_loss(agent.e2.world_forward(zw, ac), nw)
                loss = loss + F.mse_loss(agent.e2.predict_next_self(zs, ac), ns)
            loss = loss + agent.compute_prediction_loss()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, 1.0)
            opt.step()
            losses.append(float(loss.detach()))
            index = policy.act(env, obs)
            action = F.one_hot(torch.tensor([index]), env.action_dim).float()
            previous = (latent.z_world.detach().clone(), latent.z_self.detach().clone(), action)
            agent._last_action = action
            _, reward, dead, _, obs = env.step(action)
            with torch.no_grad():
                agent.update_residue(harm_signal=float(reward), hypothesis_tag=False, owned=True)
            if dead:
                break
        done += 1
        print(f"[train] heads seed={seed} ep {done}/{total}", flush=True)
    assert all(torch.equal(v, agent.latent_stack.state_dict()[k]) for k, v in encoder_before.items())
    diag["encoder_frozen_in_p1"] = True
    diag["losses"]["heads_mean"] = float(np.mean(losses))
    diag["label_balance"].update(harm_above_half=float(np.mean(np.asarray(harm_labels) > 0.5)),
                                 resource_above_half=float(np.mean(np.asarray(food_labels) > 0.5)))
    diag["goal_active_after_training"] = bool(agent.goal_state.is_active())
    for p in agent.parameters():
        p.requires_grad_(False)
    agent.eval()
    agent.reset()
    return agent, diag


def episode_metrics(frames, steps):
    third = steps // 3
    recovery = [f for f in frames if f["t"] >= 2 * third]
    alive = len(frames) == steps and frames[-1]["health"] > 0
    contacts = sum(f["contact"] for f in recovery)
    damage = sum(f["damage"] for f in recovery)
    first = next((f["t"] - 2 * third + 1 for f in recovery if f["contact"]), third)
    return dict(safe_recovery_success=int(alive and contacts >= 1 and damage <= 0.1),
                alive=int(alive), recovery_contacts=contacts, recovery_damage=damage,
                recovery_latency=first if alive else third, steps_realized=len(frames),
                steps_intended=steps, resources_total=sum(f["contact"] for f in frames),
                total_damage=sum(f["damage"] for f in frames),
                threat_damage=sum(f["damage"] for f in frames if third <= f["t"] < 2 * third))


def rollout(snapshot, arm, seed, map_id, steps, pressure, goal_weight, reactive_weight, zg=None):
    episode_seed = seed_for(seed, "eval_map", map_id)
    policy_seed = seed_for(seed, "eval_policy", map_id)
    env = TimedReef(episode_seed, steps, pressure)
    _, obs = env.reset()
    policy = LocalPolicy(policy_seed, reactive_weight, uniform=arm == "random")
    agent = full_clone(snapshot) if snapshot is not None and arm in ("ree", "no_reset") else None
    counters = {"reset_requests": 0, "resets_executed": 0, "fresh_e3_ticks": 0}
    signals = {}
    harness = None
    if agent is not None:
        agent.config.goal.goal_weight = goal_weight
        agent.config.e3.goal_weight = goal_weight
        agent.goal_state.config.goal_weight = goal_weight
        agent.e3.config.goal_weight = goal_weight
        original = agent.clock.phase_reset
        def reset_policy(*args, **kwargs):
            counters["reset_requests"] += 1
            if arm == "ree":
                counters["resets_executed"] += 1
                return original(*args, **kwargs)
        agent.clock.phase_reset = reset_policy
        def on_sense(**kw):
            latent = kw["latent"]
            signals.clear()
            signals.update(drive=float(kw["drive_level"]),
                           goal=float(agent.goal_state.goal_norm()),
                           harm=float(latent.z_harm_a.norm()),
                           relief=int(bool(agent._relief_completion_event)))
        def on_action(**kw):
            counters["fresh_e3_ticks"] += int(kw["ticks"]["e3_tick"])
            signals["e3_tick"] = int(kw["ticks"]["e3_tick"])
        harness = StepHarness(agent, env, train_mode=False,
                              hooks=StepHooks(on_sense=on_sense, on_action=on_action), seed=policy_seed)
    frames = []
    start_grid = env.grid.tolist()
    harm_prev = 0.0
    for t in range(steps):
        innovation = seed_for(policy_seed, "tick", t)
        if harness is not None:
            torch.manual_seed(innovation)
            random.seed(innovation)
            np.random.seed(innovation)
            harness._rng.seed(innovation)
            result = harness.step(obs)
            index = int(result.action.argmax())
            obs, dead, info = result.next_obs_dict, result.done, result.info
        else:
            policy.rng = np.random.default_rng(innovation)
            index = privileged_action(env) if arm == "privileged" else policy.act(env, obs)
            _, _, dead, info, obs = env.step(index)
        damage = float(env.total_harm) - harm_prev
        harm_prev = float(env.total_harm)
        frames.append(dict(t=t, x=env.agent_x, y=env.agent_y, action=index,
                           health=float(env.agent_health), energy=float(env.agent_energy),
                           damage=damage, contact=int(info["transition_type"] == "resource"),
                           event=info["transition_type"], hazards=copy.deepcopy(env.hazards),
                           resources=copy.deepcopy(env.resources), signals=dict(signals),
                           reset_requests=counters["reset_requests"],
                           resets_executed=counters["resets_executed"]))
        if dead:
            break
    if zg is not None and agent is not None:
        zg.observe(agent)
    row = dict(seed=seed, map_id=map_id, arm=arm, **episode_metrics(frames, steps), **counters)
    trace = dict(seed=seed, map_id=map_id, arm=arm, size=env.size, steps=steps,
                 initial_grid=start_grid, reef=[list(v) for v in sorted(env._reef_cells)],
                 tapes=env.tapes, frames=frames, metrics=row)
    return row, trace


def mean_score(rows):
    return float(np.mean([r["safe_recovery_success"] for r in rows]))


def criteria(rows, seeds, steps):
    means = {arm: [mean_score([r for r in rows if r["seed"] == s and r["arm"] == arm])
                   for s in seeds] for arm in ARMS}
    ree = np.array(means["ree"])
    rng = np.random.default_rng(seed_for("bootstrap", QUEUE_ID))
    draws = rng.integers(0, len(seeds), (10000, len(seeds)))
    out = []
    values = {"safe_recovery_success": float(ree.mean())}
    def gate(name, value, threshold, comparator=">="):
        passed = value >= threshold if comparator == ">=" else value > threshold
        out.append(dict(name=name, measured=float(value), threshold=float(threshold),
                        comparator=comparator, passed=bool(passed), load_bearing=True))
    gate("C1_absolute_success", ree.mean(), 0.6)
    for arm, floor, key in [("random", 0.2, "C2"), ("no_reset", 0.1, "C3")]:
        delta = ree - np.array(means[arm])
        low, high = np.quantile(delta[draws].mean(axis=1), [0.025, 0.975])
        values.update({f"delta_{arm}": float(delta.mean()), f"delta_{arm}_ci_low": float(low),
                       f"delta_{arm}_ci_high": float(high)})
        gate(f"{key}_effect", delta.mean(), floor)
        gate(f"{key}_interval", low, 0., ">")
    for arm in ARMS:
        values[f"success_{arm}"] = float(np.mean(means[arm]))
        arm_rows = [r for r in rows if r["arm"] == arm]
        for key in ["alive", "recovery_contacts", "recovery_latency", "total_damage", "resources_total"]:
            values[f"{arm}_{key}"] = float(np.mean([r[key] for r in arm_rows]))
    return out, values, means


def run(dry_run=False, output_dir=None):
    started = time.perf_counter()
    torch.set_num_threads(1)
    dev = DEV_SEEDS[:1] if dry_run else DEV_SEEDS
    heldout = [42] if dry_run else SEEDS
    steps = 90 if dry_run else STEPS
    budget = dict(world=2, affect=2, heads=2, steps=40) if dry_run else dict(TRAIN_BUDGET)
    n_maps = 2 if dry_run else EPISODES
    cal_maps = 1 if dry_run else 3
    # Environment-only calibration cannot select on held-out REE outcomes.
    calibration = []
    candidates = []
    for pressure in [0.25, 0.5]:
        for weight in REACTIVE_WEIGHTS:
            rows = []
            for s in dev:
                for m in range(cal_maps):
                    for arm in ["privileged", "reactive", "random"]:
                        row, _ = rollout(None, arm, s, m, steps, pressure, 0.5, weight)
                        rows.append(row)
            scores = {a: mean_score([r for r in rows if r["arm"] == a])
                      for a in ["privileged", "reactive", "random"]}
            calibration.append(dict(pressure=pressure, reactive_weight=weight, scores=scores, rows=rows))
            candidates.append((scores["privileged"] >= .8 and scores["reactive"] >= .6,
                               scores["reactive"] - scores["random"], pressure, -weight))
    selected = max(range(len(candidates)), key=lambda i: candidates[i])
    setting = calibration[selected]
    pressure, reactive_weight = setting["pressure"], setting["reactive_weight"]
    print(f"[calibration] pressure={pressure} reactive_weight={reactive_weight} scores={setting['scores']}", flush=True)
    trained_dev, dev_results, training = {}, [], {}
    for s in dev:
        print(f"Seed {s} Condition development_selection", flush=True)
        agent, diag = train(s, budget, pressure, dry_run)
        trained_dev[s] = agent
        training[f"development_{s}"] = diag
        for w in GOAL_WEIGHTS:
            for m in range(cal_maps, 2 * cal_maps):
                r, _ = rollout(agent, "ree", s, m, steps, pressure, w, reactive_weight)
                dev_results.append(dict(r, goal_weight=w))
        print(f"verdict: {'PASS' if diag['goal_active_after_training'] else 'FAIL'} development seed={s}", flush=True)
    goal_weight = max(GOAL_WEIGHTS, key=lambda w: (
        mean_score([r for r in dev_results if r["goal_weight"] == w]),
        np.mean([r["resources_total"] for r in dev_results if r["goal_weight"] == w]), -w))
    print(f"[selection] goal_weight={goal_weight}; held-out outcomes have not been read", flush=True)
    rows, traces, arm_results = [], [], []
    zg = ZGoalStreamAccumulator()
    agents_config = {}
    smoke_checks = {}
    for s in heldout:
        print(f"Seed {s} Condition heldout_bundle", flush=True)
        if dry_run:
            agent = trained_dev[s]
        else:
            agent, diag = train(s, budget, pressure, dry_run)
            training[f"heldout_{s}"] = diag
        agents_config[str(s)] = dataclasses.asdict(agent.config)
        # Equivalence and non-aliasing: no-intervention clones must reproduce
        # exact actions, physical outcomes and event counters on the same tape.
        if dry_run:
            a, ta = rollout(agent, "ree", s, 999, steps, pressure, goal_weight, reactive_weight)
            b, tb = rollout(agent, "ree", s, 999, steps, pressure, goal_weight, reactive_weight)
            assert ta == tb
            smoke_checks["identical_clone_replay"] = True
        for arm in ARMS:
            config_slice = dict(agent=agents_config[str(s)], budget=budget, env=ENV,
                                pressure=pressure, goal_weight=goal_weight, reactive_weight=reactive_weight,
                                arm=arm, steps=steps, maps=n_maps, map_offset=100)
            with arm_cell(s, config_slice=config_slice, script_path=Path(__file__)) as cell:
                arm_rows = []
                for m in range(100, 100 + n_maps):
                    r, trace = rollout(agent, arm, s, m, steps, pressure, goal_weight, reactive_weight, zg)
                    rows.append(r)
                    arm_rows.append(r)
                    traces.append(trace)
                row = dict(arm=arm, seed=s, episodes=arm_rows, safe_recovery_success=mean_score(arm_rows))
                cell.stamp(row)
                arm_results.append(row)
        print(f"verdict: {'PASS' if mean_score([r for r in rows if r['seed']==s and r['arm']=='ree']) >= .6 else 'FAIL'} heldout seed={s}", flush=True)
    tests, readout, per_seed = criteria(rows, heldout, steps)
    prereqs = [dict(name="attainability_primary_endpoint", kind="readiness", measured=setting["scores"]["privileged"],
                    threshold=.8, met=setting["scores"]["privileged"] >= .8,
                    control="full-map BFS on development seeds; privileged, does not prove learnability"),
               dict(name="local_observation_positive_control", kind="readiness", measured=setting["scores"]["reactive"],
                    threshold=.6, met=setting["scores"]["reactive"] >= .6,
                    control="memoryless policy with the same local fields on development seeds")]
    reset_requests = sum(r["reset_requests"] for r in rows if r["arm"] == "ree")
    readout["reset_requests_ree"] = reset_requests
    resets_live = reset_requests > 0 and all(r["resets_executed"] == 0 for r in rows if r["arm"] == "no_reset")
    prereqs.append(dict(name="reset_intervention_engaged", kind="readiness", measured=reset_requests,
                        threshold=1, met=resets_live, control="real agent phase_reset calls, not injected events"))
    ready = all(p["met"] for p in prereqs)
    passed = ready and all(t["passed"] for t in tests) and not dry_run
    outcome = "PASS" if passed else "FAIL"
    if dry_run:
        # The smoke checks the instrument, not the held-out scientific gates.
        for m in range(100, 100 + n_maps):
            paired = [t for t in traces if t["map_id"] == m]
            assert all(t["initial_grid"] == paired[0]["initial_grid"] and t["tapes"] == paired[0]["tapes"] for t in paired)
        assert all(r["resets_executed"] == 0 for r in rows if r["arm"] == "no_reset")
        assert reset_requests > 0, "No actual event reset was observed in smoke"
        assert setting["scores"]["privileged"] > 0, "Primary endpoint never engages on its positive control"
        dead = [dict(t=0, health=0., contact=0, damage=1.)]
        assert episode_metrics(dead, steps)["safe_recovery_success"] == 0
        assert episode_metrics(dead, steps)["recovery_latency"] == steps // 3
        smoke_checks.update(paired_tapes=True, death_denominator=True, real_reset_requests=reset_requests,
                            positive_control_primary_endpoint=setting["scores"]["privileged"])
        print(f"[smoke] instrument checks PASS: {smoke_checks}", flush=True)
    run_id = f"{EXPERIMENT_TYPE}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_v3"
    output = Path(output_dir) if output_dir else ROOT.parent / "REE_assembly/evidence/experiments"
    if dry_run and output_dir is None:
        output = ROOT / ".showcase_smoke"
    output.mkdir(parents=True, exist_ok=True)
    bundle_name = f"{run_id}_episode_log.json"
    replay_name = f"{run_id}_replay.html"
    ordered = sorted([r for r in rows if r["arm"] == "ree"],
                     key=lambda r: (r["safe_recovery_success"], r["resources_total"], r["seed"], r["map_id"]))
    representative = ordered[len(ordered) // 2]
    bundle = dict(title="Survive, recover, resume", dry_run=dry_run, outcome=outcome,
                  selected=dict(seed=representative["seed"], map_id=representative["map_id"]),
                  selection_rule="upper median by success, resource intake, seed, map; all episodes available",
                  steps=steps, arms=ARMS, traces=traces, readout=readout, criteria=tests,
                  per_seed=per_seed, preconditions=prereqs, limitations=LIMITATIONS)
    (output / bundle_name).write_text(json.dumps(bundle, allow_nan=False), encoding="utf-8")
    template = (Path(__file__).parent / "showcase_replay.html").read_text(encoding="utf-8")
    payload = json.dumps(bundle, allow_nan=False).replace("<", "\\u003c")
    (output / replay_name).write_text(template.replace("/*__SHOWCASE_DATA__*/null", payload), encoding="utf-8")
    manifest = dict(run_id=run_id, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    experiment_type=EXPERIMENT_TYPE, experiment_purpose=EXPERIMENT_PURPOSE,
                    queue_id=QUEUE_ID, claim_ids=[], evidence_direction="non_contributory", outcome=outcome,
                    architecture_epoch="ree_hybrid_guardrails_v1", dry_run=dry_run,
                    arm_results=arm_results, readout=readout, per_seed=per_seed, criteria=tests,
                    combination_rule="ALL C1, C2 effect and interval, C3 effect and interval AND readiness; smoke never PASS",
                    interpretation=dict(label="smoke_only" if dry_run else ("showcase_success" if passed else
                                        "showcase_criteria_not_met" if ready else "substrate_not_ready_requeue"),
                                        preconditions=prereqs,
                                        criteria_non_degenerate={t["name"]: bool(ready and len(heldout) >= 5) for t in tests}),
                    known_limitations=LIMITATIONS, user_exception="2026-09-11 user: proceed; non-evidential showcase",
                    training=training, calibration=calibration, development_selection=dev_results,
                    selection=dict(goal_weight=goal_weight, pressure=pressure, reactive_weight=reactive_weight),
                    smoke_checks=smoke_checks, output_files=[bundle_name, replay_name])
    config = dict(agents=agents_config, environment=ENV, budget=budget, steps=steps, n_maps=n_maps,
                  development_seeds=dev, heldout_seeds=heldout, selection=manifest["selection"])
    stamp_recording_core(manifest, script_path=Path(__file__), started_at=started,
                         config=config, seeds=heldout, z_goal_stream_stats=zg.to_block())
    path = write_flat_manifest(manifest, output, dry_run=dry_run, script_path=Path(__file__),
                               z_goal_stream_stats=zg.to_block())
    print(f"[output] manifest={path} replay={output / replay_name}", flush=True)
    return manifest, path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result, manifest_path = run(args.dry_run, args.output_dir)
    emit_outcome(outcome=result["outcome"], manifest_path=manifest_path,
                 queue_id=QUEUE_ID, dry_run=args.dry_run)
