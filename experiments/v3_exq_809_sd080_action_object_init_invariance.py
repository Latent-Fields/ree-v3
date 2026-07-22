"""V3-EXQ-809 -- SD-080 prior probe: is action-object CONTENT load-bearing for EXQ-003's PASS?

PURPOSE (diagnostic / re-pricing probe, NOT governance evidence for SD-004).

BACKGROUND. The 2026-07-22 scoping spike (evidence/planning/action_object_invariance_spike_
2026-07-22.md) measured that E2.action_object_head receives ZERO gradient from every REE
training path, so the action-object space O is a frozen random projection fixed at init:
99.5% of action-object variance is explained by the action label alone, the 5-action
pairwise-distance matrix varies <1% across 120 world states, within-action-pair consequence
Spearman is ~0, and the head's parameters are bit-identical after 40 warmup episodes
(delta L2 exactly 0.0). That is the finding registered as candidate claim SD-080
(e2.action_object_consequence_grounding; not yet in claims.yaml at authoring time -- see
CLAIM TAGGING below).

Section 4.2 of that spike proposes a 3-arm falsifier (frozen head / consequence-grounded
auxiliary loss / shuffled-target parameter control). Its FINAL paragraph proposes this
cheaper PRIOR PROBE first, and that is what this script is:

    Re-run the EXQ-003 TERRAIN-vs-RANDOM contrast with the action_object_head
    RE-INITIALISED at several different random seeds. If EXQ-003's 6x survival
    result is invariant to O's CONTENT, that is near-free evidence that O's
    structure is not load-bearing for SD-004's PASS -- and it materially
    re-prices (possibly cancels) the 3-arm experiment before any training
    objective is written.

WHY THE MANIPULATION IS LEGITIMATE AND CHEAP. The head is never trained (spike M0), so
re-initialising it is a pure content intervention with no optimisation-dynamics side
effect. It is not a no-op on the planner: HippocampalModule runs num_cem_iterations=3
(config.py:1684) and the legacy elite refit at module.py:1400-1409 reads back
traj.get_action_object_sequence() -- the E2-computed action objects from THIS head -- to
refit ao_mean/ao_std. So the head's content feeds the CEM search distribution on
iterations 2 and 3, and hence the candidates E3 finally chooses from.

DESIGN. Grid of (seed x ao_init) cells replicating EXQ-003's setup
(CausalGridWorld defaults, self_dim=world_dim=32, Adam lr=1e-3, TERRAIN =
hippocampal.propose_trajectories, RANDOM = e2.generate_candidates_random). ao_init=0 is
one more random draw, NOT a privileged "canonical" arm -- every init, including 0, is
re-drawn inside a saved/restored global RNG state so the downstream training RNG stream
is bit-identical across inits and the init effect is not confounded with stream drift.

NOTE ON alpha_world. This deliberately keeps EXQ-003's config, which does NOT set
alpha_world=0.9 (default 0.3). Raising it would be a better substrate but a WORSE
replication: the question is what produced EXQ-003's PASS, so EXQ-003's config is the
one that must be re-priced.

TWO LEVELS OF READOUT.
  D1 MECHANISM (cheap, matched-state): on one trained agent per seed, sweep all inits over
     a FIXED bank of (z_world, z_self) states with the sampling RNG reset identically
     before every propose_trajectories call, and record each init's empirical distribution
     over the proposed candidates' FIRST-ACTION classes. DV = mean pairwise total-variation
     distance across init pairs. This isolates the head's influence on the planner with
     everything else held bit-identical.
  D2 BEHAVIOURAL: EXQ-003's own DVs -- terrain/random harm_rate and mean_survival, and the
     terrain advantage survival_ratio = terrain_mean_survival / random_mean_survival.
     DV = spread of that advantage ACROSS INITS relative to its spread ACROSS SEEDS.
     Init-effect ratio well below 1 means O's content does not move the EXQ-003 result
     relative to ordinary seed noise.

BOTH DIRECTIONS ARE DECLARED (carried in substance from spike Sec 4.2).
  * ao_content_not_load_bearing_for_exq003_pass -- D1 low AND D2 invariant. O's semantic
    grounding is NOT what produced EXQ-003's 6x survival; SD-004's stated efficiency
    rationale ("semantically grounded in world-effects", "planning horizon extension")
    needs reframing, and the actual mechanism is terrain/residue navigation. The 3-arm
    falsifier is re-priced downward -- its behavioural read is predicted null in advance,
    so it would be run (if at all) for the substrate-side DVs and a correctness finding,
    not for a capability win. SD-080 would stay a correctness-not-capability claim.
  * ao_content_load_bearing_3arm_warranted -- D1 high AND D2 varies with init comparably
    to or more than with seed. O's content IS load-bearing; a frozen random O is then
    plausibly capping planning quality, and the 3-arm falsifier is worth its cost with a
    real prior of a behavioural effect.
  * ao_content_reaches_proposals_but_not_behaviour -- D1 high, D2 invariant. The head's
    content demonstrably moves the CEM search distribution yet does not move harm or
    survival. The 3-arm remains warranted but with an effect-size prior near zero on the
    behavioural DVs, and the substrate-side gate (spike Sec 4.2 primary) becomes the
    load-bearing read.
  A NULL ON D2 IS A REAL RESULT, NOT A FAILURE. It is the second bullet of spike Sec 4.2's
  "Both directions declared" block, and it is the outcome that most changes what gets built.

GOV-REUSE-1 (Step 2.4). Decisive readout: the spread of EXQ-003's terrain-advantage DVs
ACROSS RE-INITIALISATIONS of action_object_head. Checked the SD-004 / action-object
evidence surface (evidence/experiments/, EXQ-003's own manifests, and the spike's raw
arrays): no recorded manifest varies the ao head's initialisation -- every run holds one
draw fixed, and the spike measured O's geometry but never re-ran the behavioural contrast.
The readout is neither recorded nor derivable post-hoc. Not recoverable -> run.

DV-SYMMETRY (Step 3 mandatory declaration, per arm). The manipulation on EVERY arm is a
re-draw of action_object_head's weights.
  * D1 arms (matched-state proposal distribution). DV = the empirical distribution over
    the 8 proposed candidates' first-action classes, which is a set-aggregate (symmetric
    under permutation of candidates) of argmax-derived labels (invariant under a uniform
    additive shift or any monotone rescaling of the decoder's per-class outputs). A weight
    re-draw is NOT such a transform: it changes ao_mean and ao_std PER-COORDINATE and
    non-monotonically at the elite refit, so different classes' decoded outputs move by
    different amounts and the argmax can change. Not invariant.
  * D2 arms (harm_rate, mean_survival, survival_ratio). DV = per-episode aggregates,
    symmetric under permutation of episodes and of seeds. The manipulation is applied
    WITHIN a seed with the downstream RNG stream held bit-identical, so it is orthogonal
    to the seed permutation the DV is symmetric under. Not invariant.
  Neither delta is an arithmetic identity fixed before the run; both are empirical. The
  possibility that the head is annihilated on the way to the DV is exactly what D1
  MEASURES, and D1 is reported as a readout, not used as a gate (see PRECONDITIONS).

CLAIM TAGGING. claim_ids = ["SD-004"] only. This probe tests whether SD-004's stated
action-object-grounding rationale is load-bearing for its own EXQ-003 PASS -- that is the
claim the implementation here actually exercises. It does NOT test SD-080's factual
content (that O is ungrounded); the spike already measured that directly, and SD-080 is
not yet registered in claims.yaml, so tagging it would dangle. SD-080 is carried in
custom_information.related_claims and in the queue-entry note.
"""
from __future__ import annotations

import argparse
import math
import random as _random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorld  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_809_sd080_action_object_init_invariance"
EXPERIMENT_PURPOSE = "diagnostic"
ANCHOR_REACHABILITY_EXEMPT = (
    "Each precondition's predicate IS the degeneracy definition, not a narrower "
    "hand-written proxy for it. (1) ao_head_reinit_divergence asserts exactly "
    "'the re-draws differ', which independent draws satisfy by construction "
    "(smoke: 9.94 against a 0.10 floor). (2) cem_elite_refit_consumed_action_objects "
    "asserts the CEM returned candidates carrying action-object sequences, which "
    "propose_trajectories sets by construction via compute_action_objects=True "
    "(module.py:1310; smoke: 1.00 against a 0.90 floor). (3) "
    "d2_cross_seed_spread_nondegenerate IS the denominator of the load-bearing C2 "
    "ratio -- the identical statistic, not a control for it, so there is no separate "
    "reference whose reproducibility could fail independently. (4) "
    "residue_field_shaped_by_warmup_harm is EXQ-003's own C3 verbatim, and this run "
    "replicates EXQ-003's config, under which it held."
)

CLAIM_IDS = ["SD-004"]
RELATED_CLAIMS = ["SD-080"]

# ---------------------------------------------------------------- schedule --
SEEDS = [0, 1, 2]
AO_INITS = [0, 1, 2, 3]          # 4 independent re-draws of action_object_head
AO_INIT_SEED_BASE = 90900        # dedicated stream, disjoint from env/agent seeds
WARMUP_EPISODES = 40
EVAL_EPISODES = 15               # per condition (TERRAIN, RANDOM)
STEPS_PER_EPISODE = 200          # EXQ-003's value; the survival DV needs the range
SELF_DIM = 32
WORLD_DIM = 32
LR = 1e-3
NUM_CANDIDATES_EVAL = 8
E3_TEMPERATURE = 1.5

# D1 mechanism probe (runs once per seed, on that seed's ao_init=0 trained agent)
MECH_BANK_SIZE = 40              # matched (z_world, z_self) states
MECH_BANK_EVERY = 5              # sample a bank state every N steps of the pre-pass
MECH_RNG_SEED = 777000           # reset before EVERY propose_trajectories call

# ------------------------------------------- pre-registered thresholds ------
# D1: proposals are "invariant to O's content" below this mean pairwise TV distance.
TV_INVARIANT_CEIL = 0.05
# D2: the terrain advantage is "invariant to O's content" when its across-init spread is
# at most this fraction of its ordinary across-seed spread.
INIT_EFFECT_RATIO_CEIL = 0.50
# READINESS floors (see interpretation.preconditions):
AO_PARAM_L2_FLOOR = 0.10         # the re-draws must genuinely differ
CEM_AO_SEQ_FRAC_FLOOR = 0.90     # the refit path that consumes the head must have run
SEED_SD_FLOOR = 0.02             # the D2 denominator must be non-degenerate

ACTION_DIM = 5                   # CausalGridWorld.ACTIONS


# --------------------------------------------------------------- utilities --
def _reinit_action_object_head(agent: REEAgent, ao_init: int) -> None:
    """Re-draw action_object_head from a dedicated RNG stream.

    The global torch RNG state is saved and restored around the re-draw, so the
    stream every downstream consumer sees is BIT-IDENTICAL across ao_init values.
    Without that, the init effect would be confounded with RNG-stream drift.
    Applied to EVERY init including 0, so no arm is privileged.
    """
    state = torch.get_rng_state()
    try:
        torch.manual_seed(AO_INIT_SEED_BASE + ao_init)
        for layer in agent.e2.action_object_head:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
    finally:
        torch.set_rng_state(state)


def _ao_head_param_vector(agent: REEAgent) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in agent.e2.action_object_head.parameters()])


def _mean_pairwise_l2(vectors: List[torch.Tensor]) -> float:
    if len(vectors) < 2:
        return 0.0
    dists = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            dists.append(float(torch.norm(vectors[i] - vectors[j])))
    return float(statistics.fmean(dists))


def _mean_pairwise_tv(dists: List[List[float]]) -> float:
    """Mean pairwise total-variation distance between discrete distributions."""
    if len(dists) < 2:
        return 0.0
    out = []
    for i in range(len(dists)):
        for j in range(i + 1, len(dists)):
            out.append(0.5 * sum(abs(a - b) for a, b in zip(dists[i], dists[j])))
    return float(statistics.fmean(out))


def _sd(values: List[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    if len(vals) < 2:
        return 0.0
    return float(statistics.stdev(vals))


# ------------------------------------------------------------ EXQ-003 core --
def _train_episodes(agent, env, optimizer, num_episodes, steps_per_episode, label):
    """EXQ-003 warmup: E1 + E2 (self + world) updates; residue accumulates on harm."""
    agent.train()
    buf = []
    harm_events = 0
    total_harm = 0.0

    for ep in range(num_episodes):
        _flat, obs_dict = env.reset()
        agent.reset()
        z_self_prev = z_world_prev = action_prev = None

        for _step in range(steps_per_episode):
            latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
            ticks = agent.clock.advance()
            if ticks["e1_tick"]:
                agent._e1_tick(latent)

            if z_world_prev is not None and action_prev is not None:
                buf.append((z_world_prev, action_prev, latent.z_world.detach()))
                if len(buf) > 500:
                    buf = buf[-500:]
            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            with torch.no_grad():
                e1_prior = torch.zeros(1, agent.config.latent.world_dim, device=agent.device)
                cands = agent.hippocampal.propose_trajectories(
                    z_world=latent.z_world.detach(),
                    z_self=latent.z_self.detach(),
                    num_candidates=4,
                    e1_prior=e1_prior,
                )
                action = agent.e3.select(cands, temperature=E3_TEMPERATURE).selected_action
                agent._last_action = action

            z_self_prev = latent.z_self.detach()
            z_world_prev = latent.z_world.detach()
            action_prev = action.detach()

            _flat, harm_signal, done, info, obs_dict = env.step(action)

            if harm_signal < 0:
                harm_events += 1
                total_harm += abs(harm_signal)
                agent.update_residue(
                    harm_signal=harm_signal, hypothesis_tag=False,
                    owned=(info["transition_type"] == "agent_caused_hazard"),
                )

            e1_loss = agent.compute_prediction_loss()
            e2_self_loss = agent.compute_e2_loss()
            e2_world_loss = e1_loss.new_zeros(())
            if len(buf) >= 4:
                n = min(16, len(buf))
                idxs = torch.randperm(len(buf))[:n].tolist()
                zw_t, acts, zw_t1 = zip(*[buf[i] for i in idxs])
                e2_world_loss = F.mse_loss(
                    agent.e2.world_forward(torch.cat(zw_t, 0), torch.cat(acts, 0)),
                    torch.cat(zw_t1, 0),
                )

            loss = e1_loss + e2_self_loss + e2_world_loss
            if loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if done:
                break

        if (ep + 1) % 10 == 0 or ep == num_episodes - 1:
            print(f"  [train] {label} ep {ep + 1}/{num_episodes}  harm={total_harm:.2f}",
                  flush=True)

    return {"harm_events": harm_events, "total_harm": total_harm}


def _eval_condition(agent, env, num_episodes, steps_per_episode, use_terrain, seed):
    """EXQ-003 eval: TERRAIN (SD-004 CEM) vs RANDOM (random action shooting)."""
    agent.eval()
    torch.manual_seed(seed + (2000 if use_terrain else 3000))
    _random.seed(seed + (2000 if use_terrain else 3000))

    harm_events = 0
    resource_events = 0
    survival_steps: List[int] = []
    residue_scores: List[float] = []
    fatal_errors = 0
    label = "TERRAIN" if use_terrain else "RANDOM"

    try:
        for _ep in range(num_episodes):
            _flat, obs_dict = env.reset()
            agent.reset()
            for _step in range(steps_per_episode):
                with torch.no_grad():
                    latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
                    agent.clock.advance()
                    z_world = latent.z_world.detach()
                    z_self = latent.z_self.detach()
                    e1_prior = torch.zeros(1, agent.config.latent.world_dim,
                                           device=agent.device)

                    if use_terrain:
                        cands = agent.hippocampal.propose_trajectories(
                            z_world=z_world, z_self=z_self,
                            num_candidates=NUM_CANDIDATES_EVAL, e1_prior=e1_prior,
                        )
                        res = agent.e3.select(cands, temperature=E3_TEMPERATURE)
                        action = res.selected_action
                        sel = cands[res.selected_index if hasattr(res, "selected_index") else 0]
                    else:
                        cands = agent.e2.generate_candidates_random(
                            initial_z_self=z_self, initial_z_world=z_world,
                            num_candidates=NUM_CANDIDATES_EVAL,
                            horizon=agent.config.hippocampal.horizon,
                            compute_action_objects=False,
                        )
                        sel = cands[_random.randrange(len(cands))]
                        action = sel.actions[:, 0, :]

                    wseq = sel.get_world_state_sequence()
                    if wseq is not None and not torch.isnan(wseq).any():
                        raw = agent.residue_field.evaluate_trajectory(wseq).sum().item()
                        if not math.isnan(raw):
                            residue_scores.append(raw)

                _flat, harm_signal, done, info, obs_dict = env.step(action)
                if harm_signal < 0:
                    harm_events += 1
                    agent.update_residue(
                        harm_signal=harm_signal, hypothesis_tag=False,
                        owned=(info["transition_type"] == "agent_caused_hazard"),
                    )
                elif harm_signal > 0:
                    resource_events += 1
                if done:
                    break
            survival_steps.append(env.steps)
    except Exception:
        import traceback
        fatal_errors += 1
        print(f"  FATAL in {label}:\n{traceback.format_exc()}", flush=True)

    total_steps = sum(survival_steps)
    return {
        "condition": label,
        "harm_events": harm_events,
        "resource_events": resource_events,
        "harm_rate": harm_events / max(1, total_steps),
        "mean_trajectory_residue_score": (
            float(statistics.fmean(residue_scores)) if residue_scores else 0.0),
        "mean_survival_steps": float(sum(survival_steps) / max(1, len(survival_steps))),
        "fatal_errors": fatal_errors,
    }


# ------------------------------------------------ D1 matched-state probe ----
def _collect_state_bank(agent, env, seed) -> List[Dict[str, torch.Tensor]]:
    """Fixed bank of (z_world, z_self) states from a deterministic scripted pre-pass."""
    bank: List[Dict[str, torch.Tensor]] = []
    torch.manual_seed(seed + 5000)
    _flat, obs_dict = env.reset()
    agent.reset()
    step = 0
    while len(bank) < MECH_BANK_SIZE and step < MECH_BANK_SIZE * MECH_BANK_EVERY * 2:
        with torch.no_grad():
            latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
            agent.clock.advance()
            if step % MECH_BANK_EVERY == 0:
                bank.append({"z_world": latent.z_world.detach().clone(),
                             "z_self": latent.z_self.detach().clone()})
            a = torch.zeros(1, ACTION_DIM)
            a[0, step % ACTION_DIM] = 1.0            # fixed cyclic scripted action
        _flat, _h, done, _info, obs_dict = env.step(a)
        step += 1
        if done:
            _flat, obs_dict = env.reset()
            agent.reset()
    return bank


def _proposal_distribution(agent, bank) -> Dict[str, Any]:
    """First-action-class distribution over proposed candidates, RNG matched per state."""
    counts = [0.0] * ACTION_DIM
    n_calls = 0
    n_calls_with_ao_seq = 0
    for idx, st in enumerate(bank):
        torch.manual_seed(MECH_RNG_SEED + idx)      # identical noise draw across inits
        with torch.no_grad():
            cands = agent.hippocampal.propose_trajectories(
                z_world=st["z_world"], z_self=st["z_self"],
                num_candidates=NUM_CANDIDATES_EVAL,
                e1_prior=torch.zeros(1, agent.config.latent.world_dim, device=agent.device),
            )
        n_calls += 1
        if any(c.get_action_object_sequence() is not None for c in cands):
            n_calls_with_ao_seq += 1
        for c in cands:
            counts[int(c.actions[:, 0, :].argmax(dim=-1).item())] += 1.0
    total = sum(counts) or 1.0
    return {
        "first_action_dist": [c / total for c in counts],
        "n_calls": n_calls,
        "ao_seq_frac": n_calls_with_ao_seq / max(1, n_calls),
    }


def _ao_embedding_matrix(agent, bank) -> torch.Tensor:
    """[n_states * ACTION_DIM, ao_dim] action-object embeddings under the current head."""
    rows = []
    with torch.no_grad():
        for st in bank:
            for a in range(ACTION_DIM):
                oh = torch.zeros(1, ACTION_DIM)
                oh[0, a] = 1.0
                rows.append(agent.e2.action_object(st["z_world"], oh).reshape(-1))
    return torch.stack(rows)


def _run_mechanism_probe(agent, bank, seed) -> Dict[str, Any]:
    """Sweep every ao_init on ONE trained agent with everything else held identical."""
    saved = {k: v.detach().clone()
             for k, v in agent.e2.action_object_head.state_dict().items()}
    per_init = []
    try:
        for ao_init in AO_INITS:
            _reinit_action_object_head(agent, ao_init)
            pd = _proposal_distribution(agent, bank)
            per_init.append({
                "ao_init": ao_init,
                "first_action_dist": pd["first_action_dist"],
                "ao_seq_frac": pd["ao_seq_frac"],
                "ao_param_vec": _ao_head_param_vector(agent),
                "ao_embed": _ao_embedding_matrix(agent, bank),
            })
            print(f"  [mech] seed={seed} ao_init={ao_init} "
                  f"first_action_dist="
                  f"{[round(x, 3) for x in pd['first_action_dist']]} "
                  f"ao_seq_frac={pd['ao_seq_frac']:.2f}", flush=True)
    finally:
        agent.e2.action_object_head.load_state_dict(saved)

    dists = [p["first_action_dist"] for p in per_init]
    embeds = [p["ao_embed"].reshape(-1) for p in per_init]
    return {
        "seed": seed,
        "n_bank_states": len(bank),
        "per_init_first_action_dist": dists,
        "mean_pairwise_tv_first_action": _mean_pairwise_tv(dists),
        "mean_pairwise_ao_param_l2": _mean_pairwise_l2([p["ao_param_vec"] for p in per_init]),
        "mean_pairwise_ao_embed_l2": _mean_pairwise_l2(embeds),
        "min_ao_seq_frac": min(p["ao_seq_frac"] for p in per_init),
    }


# ------------------------------------------------------------------- cell ---
def _config_slice(ao_init: int) -> Dict[str, Any]:
    return {
        "env": {"class": "CausalGridWorld", "defaults": True},
        "schedule": {"warmup_episodes": WARMUP_EPISODES,
                     "eval_episodes": EVAL_EPISODES,
                     "steps_per_episode": STEPS_PER_EPISODE},
        "self_dim": SELF_DIM, "world_dim": WORLD_DIM, "lr": LR,
        "num_candidates_eval": NUM_CANDIDATES_EVAL,
        "e3_temperature": E3_TEMPERATURE,
        "ao_init": ao_init, "ao_init_seed_base": AO_INIT_SEED_BASE,
    }


def _run_cell(seed: int, ao_init: int, run_mechanism: bool) -> Dict[str, Any]:
    print(f"Seed {seed} Condition AO_INIT_{ao_init}", flush=True)
    with arm_cell(seed, config_slice=_config_slice(ao_init),
                  script_path=Path(__file__)) as cell:
        env = CausalGridWorld(seed=seed)
        config = REEConfig.from_dims(
            body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
            action_dim=env.action_dim, self_dim=SELF_DIM, world_dim=WORLD_DIM,
        )
        agent = REEAgent(config)
        _reinit_action_object_head(agent, ao_init)
        ao_params_before = _ao_head_param_vector(agent).clone()
        optimizer = optim.Adam(agent.parameters(), lr=LR)

        train = _train_episodes(agent, env, optimizer, WARMUP_EPISODES,
                                STEPS_PER_EPISODE, f"seed={seed} ao_init={ao_init}")

        # Spike M0 replication: the head must still be bit-identical after warmup.
        ao_delta = float(torch.norm(_ao_head_param_vector(agent) - ao_params_before))

        r_terrain = _eval_condition(agent, env, EVAL_EPISODES, STEPS_PER_EPISODE,
                                    use_terrain=True, seed=seed)
        r_random = _eval_condition(agent, env, EVAL_EPISODES, STEPS_PER_EPISODE,
                                   use_terrain=False, seed=seed)

        mech: Optional[Dict[str, Any]] = None
        if run_mechanism:
            bank = _collect_state_bank(agent, env, seed)
            mech = _run_mechanism_probe(agent, bank, seed)

        rs, rr = r_terrain["mean_survival_steps"], r_random["mean_survival_steps"]
        row = {
            "arm": f"AO_INIT_{ao_init}",
            "seed": seed,
            "ao_init": ao_init,
            "ao_head_param_delta_after_warmup": ao_delta,
            "warmup_harm_events": train["harm_events"],
            "terrain_harm_rate": r_terrain["harm_rate"],
            "random_harm_rate": r_random["harm_rate"],
            "terrain_mean_survival": rs,
            "random_mean_survival": rr,
            "terrain_mean_trajectory_residue": r_terrain["mean_trajectory_residue_score"],
            "random_mean_trajectory_residue": r_random["mean_trajectory_residue_score"],
            "terrain_resource_events": r_terrain["resource_events"],
            "random_resource_events": r_random["resource_events"],
            "survival_ratio": (rs / rr) if rr > 0 else float("nan"),
            "harm_rate_delta": r_random["harm_rate"] - r_terrain["harm_rate"],
            "fatal_errors": r_terrain["fatal_errors"] + r_random["fatal_errors"],
            "mechanism": mech,
        }
        cell.stamp(row)

    # EXQ-003's own C1 (terrain reduces harm) as this cell's verdict line.
    passed = (row["terrain_harm_rate"] < row["random_harm_rate"]
              and row["fatal_errors"] == 0)
    print(f"  [cell] seed={seed} ao_init={ao_init} "
          f"terrain_harm={row['terrain_harm_rate']:.4f} "
          f"random_harm={row['random_harm_rate']:.4f} "
          f"survival_ratio={row['survival_ratio']:.3f} "
          f"ao_delta={ao_delta:.6f}", flush=True)
    print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)
    return row


# -------------------------------------------------------------- analysis ----
def run_experiment() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        for ao_init in AO_INITS:
            rows.append(_run_cell(seed, ao_init, run_mechanism=(ao_init == AO_INITS[0])))

    mechs = [r["mechanism"] for r in rows if r["mechanism"] is not None]

    # ---- D1 mechanism ----
    tv_values = [m["mean_pairwise_tv_first_action"] for m in mechs]
    tv_mean = float(statistics.fmean(tv_values)) if tv_values else 0.0
    ao_param_l2 = min((m["mean_pairwise_ao_param_l2"] for m in mechs), default=0.0)
    ao_param_l2_cell = min(mechs, key=lambda m: m["mean_pairwise_ao_param_l2"])["seed"] \
        if mechs else "none"
    ao_seq_frac = min((m["min_ao_seq_frac"] for m in mechs), default=0.0)
    ao_seq_frac_cell = min(mechs, key=lambda m: m["min_ao_seq_frac"])["seed"] \
        if mechs else "none"
    mechanism_invariant = tv_mean <= TV_INVARIANT_CEIL

    # ---- D2 behavioural: across-init spread vs across-seed spread ----
    def _ratio_of_spreads(key: str):
        per_seed_sd = [_sd([r[key] for r in rows if r["seed"] == s]) for s in SEEDS]
        per_init_sd = [_sd([r[key] for r in rows if r["ao_init"] == a]) for a in AO_INITS]
        init_spread = float(statistics.fmean(per_seed_sd))
        seed_spread = float(statistics.fmean(per_init_sd))
        ratio = (init_spread / seed_spread) if seed_spread > 0 else float("inf")
        return init_spread, seed_spread, ratio

    sr_init_sd, sr_seed_sd, sr_ratio = _ratio_of_spreads("survival_ratio")
    hd_init_sd, hd_seed_sd, hd_ratio = _ratio_of_spreads("harm_rate_delta")
    behaviour_invariant = (sr_ratio <= INIT_EFFECT_RATIO_CEIL)

    # ---- readiness preconditions ----
    p_param = ao_param_l2 >= AO_PARAM_L2_FLOOR
    p_aoseq = ao_seq_frac >= CEM_AO_SEQ_FRAC_FLOOR
    p_denom = sr_seed_sd >= SEED_SD_FLOOR
    worst_warmup_row = min(rows, key=lambda r: r["warmup_harm_events"])
    p_harm = worst_warmup_row["warmup_harm_events"] > 0
    ready = p_param and p_aoseq and p_denom and p_harm
    fatal = sum(r["fatal_errors"] for r in rows)

    if not ready:
        label = "substrate_not_ready_requeue"
    elif mechanism_invariant and behaviour_invariant:
        label = "ao_content_not_load_bearing_for_exq003_pass"
    elif not mechanism_invariant and behaviour_invariant:
        label = "ao_content_reaches_proposals_but_not_behaviour"
    else:
        label = "ao_content_load_bearing_3arm_warranted"

    outcome = "PASS" if (ready and fatal == 0) else "FAIL"

    metrics = {
        "d1_mean_pairwise_tv_first_action": tv_mean,
        "d1_tv_per_seed": tv_values,
        "d1_min_mean_pairwise_ao_param_l2": ao_param_l2,
        "d1_mean_pairwise_ao_embed_l2": (
            float(statistics.fmean(m["mean_pairwise_ao_embed_l2"] for m in mechs))
            if mechs else 0.0),
        "d1_min_ao_seq_frac": ao_seq_frac,
        "d1_mechanism_invariant": mechanism_invariant,
        "d2_survival_ratio_init_sd": sr_init_sd,
        "d2_survival_ratio_seed_sd": sr_seed_sd,
        "d2_survival_ratio_init_effect_ratio": sr_ratio,
        "d2_harm_delta_init_sd": hd_init_sd,
        "d2_harm_delta_seed_sd": hd_seed_sd,
        "d2_harm_delta_init_effect_ratio": hd_ratio,
        "d2_behaviour_invariant": behaviour_invariant,
        "mean_survival_ratio": float(statistics.fmean(
            r["survival_ratio"] for r in rows if not math.isnan(r["survival_ratio"]))),
        "mean_terrain_harm_rate": float(statistics.fmean(r["terrain_harm_rate"] for r in rows)),
        "mean_random_harm_rate": float(statistics.fmean(r["random_harm_rate"] for r in rows)),
        "max_ao_head_param_delta_after_warmup": max(
            r["ao_head_param_delta_after_warmup"] for r in rows),
        "min_warmup_harm_events": worst_warmup_row["warmup_harm_events"],
        "fatal_error_count": float(fatal),
    }

    return {
        "outcome": outcome,
        "metrics": metrics,
        "per_seed_rows": rows,
        "arm_results": rows,
        "mechanism_probe": mechs,
        "interpretation": {
            "label": label,
            "preconditions": [
                {"name": "ao_head_reinit_divergence",
                 "description": ("The re-draws of action_object_head must genuinely "
                                 "differ. Below this floor the manipulation was never "
                                 "applied and any null is about nothing."),
                 "measured": ao_param_l2, "threshold": AO_PARAM_L2_FLOOR,
                 "direction": "lower",
                 "control": ("worst seed's mean pairwise L2 between the 4 re-drawn "
                             "parameter vectors, measured directly on the head"),
                 "offending_cell": f"seed{ao_param_l2_cell}",
                 "met": p_param},
                {"name": "cem_elite_refit_consumed_action_objects",
                 "description": ("The CEM elite refit (module.py:1400-1409) is the only "
                                 "path by which the head reaches behaviour. This is the "
                                 "fraction of matched-state propose_trajectories calls "
                                 "returning candidates that carry action-object "
                                 "sequences. Below the floor the head was never read."),
                 "measured": ao_seq_frac, "threshold": CEM_AO_SEQ_FRAC_FLOOR,
                 "direction": "lower",
                 "control": ("worst seed in the D1 matched-state bank, where the CEM "
                             "path is exercised by construction"),
                 "offending_cell": f"seed{ao_seq_frac_cell}",
                 "met": p_aoseq},
                {"name": "d2_cross_seed_spread_nondegenerate",
                 "description": ("SAME STATISTIC the load-bearing C2 routes on: C2 is a "
                                 "ratio of the across-init spread of survival_ratio to "
                                 "its across-seed spread. If that DENOMINATOR is ~0 the "
                                 "ratio is meaningless and an apparent invariance would "
                                 "be an artefact of a pinned DV, not a finding."),
                 "measured": sr_seed_sd, "threshold": SEED_SD_FLOOR,
                 "direction": "lower",
                 "control": ("ordinary seed-to-seed variation of the terrain advantage, "
                             "measured within each ao_init and averaged"),
                 "met": p_denom},
                {"name": "residue_field_shaped_by_warmup_harm",
                 "description": ("EXQ-003's own C3. TERRAIN is terrain-guided CEM over "
                                 "the residue field; with zero warmup harm the terrain "
                                 "is unshaped and the contrast is not EXQ-003's."),
                 "measured": float(worst_warmup_row["warmup_harm_events"]),
                 "threshold": 1.0, "direction": "lower",
                 "control": ("worst cell across all seeds x inits"),
                 "offending_cell": (f"seed{worst_warmup_row['seed']}/"
                                    f"ao_init{worst_warmup_row['ao_init']}"),
                 "met": p_harm},
            ],
            "criteria": [
                {"name": "C1_mechanism_proposal_invariance",
                 "load_bearing": False, "passed": mechanism_invariant},
                {"name": "C2_behavioural_init_effect_below_seed_noise",
                 "load_bearing": True, "passed": behaviour_invariant},
            ],
            "criteria_non_degenerate": {
                # C1 discriminates only with >=2 inits and a non-empty matched-state bank.
                "C1_mechanism_proposal_invariance": bool(
                    len(AO_INITS) >= 2 and mechs
                    and all(m["n_bank_states"] >= 5 for m in mechs)),
                # C2 discriminates only with >=3 inits AND >=2 seeds AND a live denominator.
                "C2_behavioural_init_effect_below_seed_noise": bool(
                    len(AO_INITS) >= 3 and len(SEEDS) >= 2 and sr_seed_sd > 0.0),
            },
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    global SEEDS, AO_INITS, WARMUP_EPISODES, EVAL_EPISODES, STEPS_PER_EPISODE
    global MECH_BANK_SIZE
    if args.dry_run:
        SEEDS = [0]
        AO_INITS = [0, 1]
        WARMUP_EPISODES = 2
        EVAL_EPISODES = 2
        STEPS_PER_EPISODE = 12
        MECH_BANK_SIZE = 4

    result = run_experiment()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    full_config = {
        "seeds": SEEDS, "ao_inits": AO_INITS,
        "ao_init_seed_base": AO_INIT_SEED_BASE,
        "warmup_episodes": WARMUP_EPISODES, "eval_episodes": EVAL_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "self_dim": SELF_DIM, "world_dim": WORLD_DIM, "lr": LR,
        "num_candidates_eval": NUM_CANDIDATES_EVAL,
        "e3_temperature": E3_TEMPERATURE,
        "mech_bank_size": MECH_BANK_SIZE, "mech_bank_every": MECH_BANK_EVERY,
        "mech_rng_seed": MECH_RNG_SEED,
        "tv_invariant_ceil": TV_INVARIANT_CEIL,
        "init_effect_ratio_ceil": INIT_EFFECT_RATIO_CEIL,
        "ao_param_l2_floor": AO_PARAM_L2_FLOOR,
        "cem_ao_seq_frac_floor": CEM_AO_SEQ_FRAC_FLOOR,
        "seed_sd_floor": SEED_SD_FLOOR,
        "env": "CausalGridWorld (EXQ-003 defaults; alpha_world NOT raised, see docstring)",
        "arm_config_slices": {f"AO_INIT_{a}": _config_slice(a) for a in AO_INITS},
    }
    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "unknown",
        "outcome": result["outcome"],
        "timestamp_utc": ts,
        "metrics": result["metrics"],
        "per_seed_rows": result["per_seed_rows"],
        "arm_results": result["arm_results"],
        "mechanism_probe": result["mechanism_probe"],
        "interpretation": result["interpretation"],
        "custom_information": {
            "related_claims": RELATED_CLAIMS,
            "spike_reference": ("REE_assembly/evidence/planning/"
                                "action_object_invariance_spike_2026-07-22.md Sec 4.2"),
            "replicates": "v3_exq_003_sd004_action_objects (TERRAIN vs RANDOM)",
            "prior_probe_for": ("SD-080 3-arm falsifier (frozen / consequence-grounded / "
                                "shuffled-target control) -- this probe re-prices it"),
        },
    }
    out_path = write_flat_manifest(
        manifest,
        Path(__file__).resolve().parents[2] / "REE_assembly" / "evidence" / "experiments",
        dry_run=args.dry_run, config=full_config, seeds=SEEDS,
        script_path=Path(__file__), started_at=t0,
    )
    m = result["metrics"]
    print(f"outcome: {result['outcome']}", flush=True)
    print(f"label: {result['interpretation']['label']}", flush=True)
    print(f"D1 tv={m['d1_mean_pairwise_tv_first_action']:.4f} "
          f"(ceil {TV_INVARIANT_CEIL}) ao_param_l2={m['d1_min_mean_pairwise_ao_param_l2']:.4f} "
          f"ao_seq_frac={m['d1_min_ao_seq_frac']:.2f}", flush=True)
    print(f"D2 init_sd={m['d2_survival_ratio_init_sd']:.4f} "
          f"seed_sd={m['d2_survival_ratio_seed_sd']:.4f} "
          f"ratio={m['d2_survival_ratio_init_effect_ratio']:.3f} "
          f"(ceil {INIT_EFFECT_RATIO_CEIL})", flush=True)
    print(f"mean_survival_ratio={m['mean_survival_ratio']:.3f} "
          f"max_ao_delta={m['max_ao_head_param_delta_after_warmup']:.6f}", flush=True)
    print(f"wrote: {out_path}", flush=True)
    return result, out_path, args.dry_run


if __name__ == "__main__":
    _result, _out_path, _dry_run = main()
    _outcome_raw = str(_result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(_out_path),
        dry_run=_dry_run,
    )
