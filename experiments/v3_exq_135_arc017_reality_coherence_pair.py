#!/opt/local/bin/python3
"""
V3-EXQ-135 -- ARC-017 Reality-Coherence Lane Discriminative Pair

Claims: ARC-017
Proposal: EXP-0031 / EVB-0023

ARC-017 asserts: "Minimal stream tags with typed exteroception and explicit
reality-coherence lane." The REALITY_COHERENCE stream is defined in the
architecture as: "provenance/authority/identity consistency signal generated
from hippocampal trace structure and trusted control-store checks."

This experiment tests the REALITY_COHERENCE lane specifically -- a distinct
facet of ARC-017 not covered by V3-EXQ-129 (which tested stream-type routing:
world_obs vs harm_obs separation).

The REALITY_COHERENCE property states that the world encoder must distinguish
genuine sensory inputs from internally-generated (replayed, imagined, or
simulated) inputs. If this distinction is architecturally necessary (ARC-017),
then training the world encoder with a provenance constraint -- anchoring it
to REAL observations only and excluding replay-contaminated updates -- should
produce better harm-discrimination representations than an encoder that treats
replay and real observations as equivalent.

V3-accessible proxy design
---------------------------
CausalGridWorldV2 produces real sensory observations at each step. The agent
maintains a replay buffer of past (z_world, harm_label) tuples. In the ABLATED
condition, we train the z_world encoder also on internally replayed/reconstructed
inputs (mixing real and internally generated), violating the reality-coherence
constraint. In the ON condition, the world encoder is trained only on real
sensory inputs (reality-coherent), while replay is used only for the harm_eval
head (which operates on z_world representations, not raw obs).

REALITY_COHERENCE_ON (provenance-separated training):
  - z_world encoder: trains only on REAL env observations (online, step-by-step)
  - harm_eval_head: trains on buffered (z_world, harm_label) from real steps
  - No replay signal enters the world encoder pathway
  - The world latent is grounded in real sensory experience

REALITY_COHERENCE_ABLATED (provenance-collapsed training):
  - z_world encoder: trains on BOTH real observations AND internally reconstructed
    observations (noise-perturbed replay of past world_obs vectors that simulate
    "imagined" sensory inputs)
  - harm_eval_head: same buffered training as ON
  - The encoder treats imagined world states as equivalent to real ones
  - The reality-coherence constraint is violated

The noise-perturbed reconstruction simulates the internally-generated world
observations that arise when the agent "imagines" or "simulates" future states.
We apply Gaussian perturbation to past world_obs samples and run them through
the encoder as additional training signal, injecting imagined-world contamination
into the encoder's training distribution.

Measurement: harm_eval_head quality (gap between harm-positive and harm-negative
evaluations during a held-out eval phase that uses ONLY real env observations in
both conditions). Both conditions are evaluated identically -- only training differs.

If ARC-017's reality-coherence lane is architecturally necessary:
  - REALITY_COHERENCE_ON harm_eval gap > REALITY_COHERENCE_ABLATED gap
  - The ON condition's z_world is better anchored to real harm-relevant features
  - The ABLATED condition's z_world is diluted by imagined/reconstructed inputs
    that have no genuine nociceptive correlates

Pre-registered acceptance criteria:
  C1: gap_on >= 0.04  (REALITY_COHERENCE_ON gap above floor -- both seeds)
  C2: per-seed delta (ON - ABLATED) >= 0.02  (reality grounding adds >=2pp -- both seeds)
  C3: gap_ablated >= 0.0  (ablation still learns something -- data quality)
  C4: n_harm_eval_min >= 20  (sufficient harm events in eval -- both seeds)
  C5: n_replay_contamination_min >= 50  (ablation received enough imagined inputs
      to have meaningful contamination effect -- manipulation check, both seeds)

Decision scoring:
  PASS (all 5): retain_ree -- reality-coherence lane is architecturally necessary
  C1+C2+C4:     hybridize  -- coherence effect present but magnitude uncertain
  C2 fails:     retire_ree_claim -- no detectable advantage of provenance separation

Seeds: [42, 123] (matched -- same env per seed across conditions)
Env:   CausalGridWorldV2, size=6, 4 hazards, nav_bias=0.40
Warmup: 400 eps x 200 steps
Eval:  50 eps x 200 steps (no training, fixed weights)

Complementarity note (EXQ-129 vs EXQ-135):
  EXQ-129 tested stream-type routing: world_obs and harm_obs merged (ablated) vs
    routed to separate encoders (ON). Stream-type dimension of ARC-017.
  EXQ-135 tests provenance: world encoder trained on real+imagined inputs (ablated)
    vs real only (ON). Reality-coherence dimension of ARC-017.
  Together they provide two-axis coverage of ARC-017's full claim.
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_135_arc017_reality_coherence_pair"
CLAIM_IDS = ["ARC-017"]

# Pre-registered thresholds
THRESH_C1 = 0.04   # gap_on >= 0.04 (REALITY_COHERENCE_ON above floor)
THRESH_C2 = 0.02   # per-seed delta (ON - ABLATED) >= 0.02
THRESH_C3 = 0.0    # gap_ablated >= 0.0 (ablation learns something)
THRESH_C4 = 20     # n_harm_eval_min >= 20 (data quality)
THRESH_C5 = 50     # n_replay_contamination_min >= 50 (manipulation check)

# Ablated-condition parameters
REPLAY_NOISE_STD = 0.10   # noise level for imagined/reconstructed obs
REPLAY_BATCH_PER_STEP = 4  # how many imagined inputs injected per real step (ablated)


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _hazard_approach_action(env: CausalGridWorldV2, n_actions: int) -> int:
    """Return action index that moves toward the nearest hazard gradient."""
    obs_dict = env._get_observation_dict()
    world_state = obs_dict.get("world_state", None)
    if world_state is None or not env.use_proxy_fields:
        return random.randint(0, n_actions - 1)
    # world_state[225:250] = hazard_field_view (5x5 flattened, proxy channel)
    field_view = world_state[225:250].numpy().reshape(5, 5)
    # Agent at center (2,2); actions: 0=up(-1,0), 1=down(+1,0), 2=left(0,-1), 3=right(0,+1)
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    vals = []
    for dr, dc in deltas:
        r, c = 2 + dr, 2 + dc
        if 0 <= r < 5 and 0 <= c < 5:
            vals.append(float(field_view[r, c]))
        else:
            vals.append(-1.0)
    return int(np.argmax(vals))


def _run_single(
    seed: int,
    reality_coherence_on: bool,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    lr: float,
    alpha_world: float,
    alpha_self: float,
    harm_scale: float,
    proximity_harm_scale: float,
    nav_bias: float,
    dry_run: bool,
) -> Dict:
    """Run one (seed, condition) cell; return harm eval gap metrics.

    REALITY_COHERENCE_ON: world encoder trains only on real env observations.
    REALITY_COHERENCE_ABLATED: world encoder also trains on noise-perturbed
      replay of past world_obs (imagined/internally-reconstructed inputs).
    """
    torch.manual_seed(seed)
    random.seed(seed)

    cond_label = "REALITY_COHERENCE_ON" if reality_coherence_on else "REALITY_COHERENCE_ABLATED"

    env = CausalGridWorldV2(
        seed=seed,
        size=6,
        num_hazards=4,
        num_resources=3,
        hazard_harm=harm_scale,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=proximity_harm_scale,
        proximity_benefit_scale=proximity_harm_scale * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        use_proxy_fields=True,
    )
    n_actions = env.action_dim
    world_obs_dim = env.world_obs_dim

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=0,
    )

    agent = REEAgent(config)

    # Harm eval buffer for harm_eval_head training (used in BOTH conditions)
    buf_zw: List[torch.Tensor] = []
    buf_labels: List[float] = []
    MAX_BUF = 2000

    # Raw world obs buffer -- used in ABLATED condition for replay contamination
    buf_world_obs_raw: List[torch.Tensor] = []
    MAX_OBS_BUF = 500

    # Optimizer: standard params (excluding harm_eval_head)
    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n
    ]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())

    optimizer = optim.Adam(standard_params, lr=lr)
    harm_eval_optimizer = optim.Adam(harm_eval_params, lr=1e-4)

    counts: Dict[str, int] = {
        "hazard_approach": 0,
        "env_caused_hazard": 0,
        "agent_caused_hazard": 0,
        "none": 0,
    }
    n_replay_contamination = 0  # C5: manipulation check (ablated condition only)

    if dry_run:
        actual_warmup = min(3, warmup_episodes)
        actual_eval = min(2, eval_episodes)
    else:
        actual_warmup = warmup_episodes
        actual_eval = eval_episodes

    # --- TRAIN ---
    agent.train()

    for ep in range(actual_warmup):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world_raw = obs_dict["world_state"]

            # Real observation forward pass
            latent = agent.sense(obs_body, obs_world_raw)
            agent.clock.advance()
            z_world_curr = latent.z_world.detach()

            # Standard E1 + E2 losses from REAL observation
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            real_loss = e1_loss + e2_loss

            if reality_coherence_on:
                # ON: Only train on real observations
                total_loss = real_loss
            else:
                # ABLATED: Also inject noise-perturbed (imagined) world_obs into
                # the encoder training. We perturb buffered real obs to simulate
                # internally-generated / imagined world states and train the encoder
                # on these contaminating inputs.
                imagined_loss = torch.tensor(0.0, requires_grad=False)
                if len(buf_world_obs_raw) >= REPLAY_BATCH_PER_STEP:
                    # Sample from past real obs and add noise to simulate imagination
                    idxs = random.sample(
                        range(len(buf_world_obs_raw)),
                        min(REPLAY_BATCH_PER_STEP, len(buf_world_obs_raw))
                    )
                    for idx in idxs:
                        obs_imag = buf_world_obs_raw[idx].clone()
                        # Add Gaussian noise: imagined inputs have perturbed sensory content
                        noise = torch.randn_like(obs_imag) * REPLAY_NOISE_STD
                        obs_imag_noisy = obs_imag + noise
                        # Forward pass through encoder with imagined input
                        _ = agent.sense(obs_body, obs_imag_noisy)
                        # Compute E1 prediction loss -- this contaminates the encoder
                        # with imagined-world prediction errors as if they were real
                        e1_imag = agent.compute_prediction_loss()
                        imagined_loss = imagined_loss + e1_imag
                        n_replay_contamination += 1
                    # Restore real observation state
                    _ = agent.sense(obs_body, obs_world_raw)

                total_loss = real_loss + imagined_loss * 0.3

            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            # Action selection
            if random.random() < nav_bias:
                action_idx = _hazard_approach_action(env, n_actions)
            else:
                action_idx = random.randint(0, n_actions - 1)
            action = _action_to_onehot(action_idx, n_actions, agent.device)
            agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")
            if ttype in counts:
                counts[ttype] += 1

            harm_sig_float = float(harm_signal)
            is_harm = harm_sig_float < 0

            # Buffer for harm_eval training (REAL observations only -- both conditions)
            buf_zw.append(z_world_curr)
            buf_labels.append(1.0 if is_harm else 0.0)
            if len(buf_zw) > MAX_BUF:
                buf_zw = buf_zw[-MAX_BUF:]
                buf_labels = buf_labels[-MAX_BUF:]

            # Buffer raw obs for ablated replay contamination
            if not reality_coherence_on:
                buf_world_obs_raw.append(obs_world_raw.clone())
                if len(buf_world_obs_raw) > MAX_OBS_BUF:
                    buf_world_obs_raw = buf_world_obs_raw[-MAX_OBS_BUF:]

            # Harm eval head online update (both conditions: trains on real-experience buffer)
            n_harm_buf = sum(1 for lbl in buf_labels if lbl > 0.5)
            n_safe_buf = sum(1 for lbl in buf_labels if lbl <= 0.5)
            if n_harm_buf >= 4 and n_safe_buf >= 4:
                harm_idxs = [i for i, lbl in enumerate(buf_labels) if lbl > 0.5]
                safe_idxs = [i for i, lbl in enumerate(buf_labels) if lbl <= 0.5]
                k = min(8, min(len(harm_idxs), len(safe_idxs)))
                sel_h = random.sample(harm_idxs, k)
                sel_s = random.sample(safe_idxs, k)
                sel = sel_h + sel_s
                zw_b = torch.cat([buf_zw[i] for i in sel], dim=0)
                labels_b = torch.tensor(
                    [buf_labels[i] for i in sel],
                    dtype=torch.float32,
                ).unsqueeze(1)
                pred = agent.e3.harm_eval(zw_b)
                loss_he = F.mse_loss(pred, labels_b)
                if loss_he.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    loss_he.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5
                    )
                    harm_eval_optimizer.step()

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == actual_warmup - 1:
            print(
                f"  [train] seed={seed} cond={cond_label}"
                f" ep {ep+1}/{actual_warmup}"
                f" approach={counts['hazard_approach']}"
                f" contact={counts['env_caused_hazard']+counts['agent_caused_hazard']}"
                f" n_replay_contam={n_replay_contamination}",
                flush=True,
            )

    # --- EVAL ---
    # Both conditions evaluated identically using REAL env observations only
    agent.eval()

    eval_scores_harm: List[float] = []
    eval_scores_safe: List[float] = []
    n_fatal = 0

    for _ in range(actual_eval):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world_raw = obs_dict["world_state"]

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world_raw)
                agent.clock.advance()
                z_world_curr = latent.z_world

            action_idx = random.randint(0, n_actions - 1)
            action = _action_to_onehot(action_idx, n_actions, agent.device)
            agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action)
            is_harm = float(harm_signal) < 0

            try:
                with torch.no_grad():
                    score = float(agent.e3.harm_eval(z_world_curr).item())
                if is_harm:
                    eval_scores_harm.append(score)
                else:
                    eval_scores_safe.append(score)
            except Exception:
                n_fatal += 1

            if done:
                break

    n_harm_eval = len(eval_scores_harm)
    n_safe_eval = len(eval_scores_safe)

    mean_harm = float(sum(eval_scores_harm) / max(1, n_harm_eval))
    mean_safe = float(sum(eval_scores_safe) / max(1, n_safe_eval))
    gap = mean_harm - mean_safe

    print(
        f"  [eval] seed={seed} cond={cond_label}"
        f" gap={gap:.4f} mean_harm={mean_harm:.4f} mean_safe={mean_safe:.4f}"
        f" n_harm={n_harm_eval} n_safe={n_safe_eval}"
        f" n_replay_contam={n_replay_contamination}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "reality_coherence_on": reality_coherence_on,
        "harm_eval_gap": float(gap),
        "mean_harm_score": float(mean_harm),
        "mean_safe_score": float(mean_safe),
        "n_harm_eval": int(n_harm_eval),
        "n_safe_eval": int(n_safe_eval),
        "n_fatal": int(n_fatal),
        "n_replay_contamination": int(n_replay_contamination),
        "train_approach": int(counts["hazard_approach"]),
        "train_contact": int(counts["env_caused_hazard"] + counts["agent_caused_hazard"]),
    }


def run(
    seeds: Tuple = (42, 123),
    warmup_episodes: int = 400,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_harm_scale: float = 0.05,
    nav_bias: float = 0.40,
    dry_run: bool = False,
    **kwargs,
) -> dict:
    """Discriminative pair: REALITY_COHERENCE_ON (world encoder trains on real obs only)
    vs REALITY_COHERENCE_ABLATED (world encoder also trains on noise-perturbed replayed
    inputs, violating reality-coherence provenance constraint).
    Tests ARC-017: explicit reality-coherence lane is architecturally necessary.
    """
    results_on:      List[Dict] = []
    results_ablated: List[Dict] = []

    for seed in seeds:
        for rc_on in [True, False]:
            label = "REALITY_COHERENCE_ON" if rc_on else "REALITY_COHERENCE_ABLATED"
            print(
                f"\n[V3-EXQ-135] {label} seed={seed}"
                f" warmup={warmup_episodes} eval={eval_episodes}"
                f" nav_bias={nav_bias}",
                flush=True,
            )
            r = _run_single(
                seed=seed,
                reality_coherence_on=rc_on,
                warmup_episodes=warmup_episodes,
                eval_episodes=eval_episodes,
                steps_per_episode=steps_per_episode,
                self_dim=self_dim,
                world_dim=world_dim,
                lr=lr,
                alpha_world=alpha_world,
                alpha_self=alpha_self,
                harm_scale=harm_scale,
                proximity_harm_scale=proximity_harm_scale,
                nav_bias=nav_bias,
                dry_run=dry_run,
            )
            if rc_on:
                results_on.append(r)
            else:
                results_ablated.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    gap_on      = _avg(results_on,      "harm_eval_gap")
    gap_ablated = _avg(results_ablated, "harm_eval_gap")
    delta_gap   = gap_on - gap_ablated

    n_harm_min = min(r["n_harm_eval"] for r in results_on + results_ablated)
    n_replay_contam_min = min(
        r["n_replay_contamination"] for r in results_ablated
    ) if results_ablated else 0

    # Pre-registered PASS criteria
    # C1: REALITY_COHERENCE_ON gap >= THRESH_C1 in ALL seeds
    c1_pass = all(r["harm_eval_gap"] >= THRESH_C1 for r in results_on)
    # C2: per-seed delta (ON - ABLATED) >= THRESH_C2 in ALL seeds
    per_seed_deltas: List[float] = []
    for r_on in results_on:
        matching = [r for r in results_ablated if r["seed"] == r_on["seed"]]
        if matching:
            per_seed_deltas.append(r_on["harm_eval_gap"] - matching[0]["harm_eval_gap"])
    c2_pass = len(per_seed_deltas) > 0 and all(d >= THRESH_C2 for d in per_seed_deltas)
    # C3: REALITY_COHERENCE_ABLATED gap >= 0 (ablation still learns)
    c3_pass = gap_ablated >= THRESH_C3
    # C4: sufficient harm events in eval
    c4_pass = n_harm_min >= THRESH_C4
    # C5: manipulation check -- ablated condition received enough imagined inputs
    c5_pass = n_replay_contam_min >= THRESH_C5

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])
    status = "PASS" if all_pass else "FAIL"

    if all_pass:
        decision = "retain_ree"
    elif c1_pass and c4_pass:
        decision = "hybridize"
    else:
        decision = "retire_ree_claim"

    print(f"\n[V3-EXQ-135] Results:", flush=True)
    print(
        f"  gap_on={gap_on:.4f}"
        f" gap_ablated={gap_ablated:.4f}"
        f" delta_gap={delta_gap:+.4f}",
        flush=True,
    )
    print(
        f"  n_harm_min={n_harm_min}"
        f"  n_replay_contam_min={n_replay_contam_min}"
        f"  per_seed_deltas={[round(d, 4) for d in per_seed_deltas]}"
        f"  decision={decision}  status={status} ({criteria_met}/5)",
        flush=True,
    )

    failure_notes: List[str] = []
    if not c1_pass:
        failing = [r["seed"] for r in results_on if r["harm_eval_gap"] < THRESH_C1]
        failure_notes.append(
            f"C1 FAIL: REALITY_COHERENCE_ON gap below {THRESH_C1} in seeds {failing}"
            " -- reality-grounded encoder does not produce discriminative z_world"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: per-seed deltas {[round(d, 4) for d in per_seed_deltas]}"
            f" < {THRESH_C2}"
            " -- reality-coherence constraint does not add >=2pp over replay-contaminated encoder"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: gap_ablated={gap_ablated:.4f} < {THRESH_C3}"
            " -- ablation (replay-contaminated) fails entirely; confound check"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: n_harm_min={n_harm_min} < {THRESH_C4}"
            " -- insufficient harm events in eval phase"
        )
    if not c5_pass:
        failure_notes.append(
            f"C5 FAIL: n_replay_contam_min={n_replay_contam_min} < {THRESH_C5}"
            " -- manipulation check: ablated encoder did not receive enough imagined inputs"
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    if all_pass:
        interpretation = (
            "ARC-017 REALITY_COHERENCE SUPPORTED: grounding the world encoder"
            " exclusively to real sensory observations (ON) produces higher"
            f" harm_eval_head quality (gap_on={gap_on:.4f}) vs encoder"
            f" contaminated with noise-perturbed replay inputs (gap_ablated={gap_ablated:.4f},"
            f" delta={delta_gap:+.4f} across {len(seeds)} seeds)."
            " When imagined/reconstructed world states are injected into the encoder"
            " training distribution (ablated), the encoder's z_world latent becomes"
            " less anchored to genuine sensory harm-relevant features, degrading"
            " downstream harm discrimination. The explicit reality-coherence lane"
            " (provenance-separated training) is architecturally necessary, consistent"
            " with ARC-017's requirement for a dedicated reality-coherence stream."
        )
    elif c1_pass and c4_pass:
        interpretation = (
            f"Partial support: REALITY_COHERENCE_ON achieves gap={gap_on:.4f} (C1 PASS)"
            f" but per-seed delta={per_seed_deltas} (C2 FAIL, threshold={THRESH_C2})."
            " Reality-grounded training produces adequate harm discrimination but the"
            " advantage over replay-contaminated encoder is below the pre-registered"
            " threshold at this training scale. Possible: noise level {REPLAY_NOISE_STD}"
            " is too low to produce strong contamination, or warmup scale is insufficient."
        )
    else:
        interpretation = (
            f"ARC-017 reality-coherence lane NOT supported at V3 proxy level:"
            f" gap_on={gap_on:.4f} (C1 {'PASS' if c1_pass else 'FAIL'}),"
            f" delta={delta_gap:+.4f} (C2 {'PASS' if c2_pass else 'FAIL'})."
            " Grounding the encoder to real observations only does not produce a"
            " detectable improvement in harm_eval discrimination vs replay-contaminated"
            " encoder at this training scale. Note: this is a V3 proxy -- the full"
            " ARC-017 reality-coherence mechanism requires hippocampal trace provenance"
            " tracking which is only partially approximated by this noise-perturbation proxy."
        )

    per_on_rows = "\n".join(
        f"  seed={r['seed']}: gap={r['harm_eval_gap']:.4f}"
        f" n_harm={r['n_harm_eval']}"
        for r in results_on
    )
    per_ablated_rows = "\n".join(
        f"  seed={r['seed']}: gap={r['harm_eval_gap']:.4f}"
        f" n_harm={r['n_harm_eval']}"
        f" n_replay_contam={r['n_replay_contamination']}"
        for r in results_ablated
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-135 -- ARC-017 Reality-Coherence Lane Discriminative Pair\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** ARC-017\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**Conditions:** REALITY_COHERENCE_ON vs REALITY_COHERENCE_ABLATED\n"
        f"**Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps\n"
        f"**Env:** CausalGridWorldV2 size=6, 4 hazards, nav_bias={nav_bias}\n\n"
        f"## Design\n\n"
        f"REALITY_COHERENCE_ON: world encoder trains only on REAL env observations;\n"
        f"the reality-coherence provenance constraint is preserved.\n"
        f"REALITY_COHERENCE_ABLATED: world encoder also trains on noise-perturbed\n"
        f"replay of past world_obs (imagined inputs, noise_std={REPLAY_NOISE_STD});\n"
        f"provenance constraint is violated (real and imagined treated equivalently).\n"
        f"Evaluation uses ONLY real env observations in both conditions.\n\n"
        f"Complementary to V3-EXQ-129 (stream-type routing facet of ARC-017).\n"
        f"This experiment tests the reality-coherence facet of ARC-017.\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: gap_on >= {THRESH_C1} in all seeds  (REALITY_COHERENCE_ON above floor)\n"
        f"C2: per-seed delta (ON - ABLATED) >= {THRESH_C2} in all seeds  "
        f"(reality grounding adds >=2pp)\n"
        f"C3: gap_ablated >= {THRESH_C3}  (ablation learns something; data quality)\n"
        f"C4: n_harm_eval_min >= {THRESH_C4}  (sufficient harm events in eval)\n"
        f"C5: n_replay_contam_min >= {THRESH_C5}  "
        f"(manipulation check: ablation received imagined inputs)\n\n"
        f"## Results\n\n"
        f"| Condition | gap (avg) | mean_harm | mean_safe |\n"
        f"|-----------|-----------|-----------|----------|\n"
        f"| REALITY_COHERENCE_ON      | {gap_on:.4f}"
        f" | {_avg(results_on,      'mean_harm_score'):.4f}"
        f" | {_avg(results_on,      'mean_safe_score'):.4f} |\n"
        f"| REALITY_COHERENCE_ABLATED | {gap_ablated:.4f}"
        f" | {_avg(results_ablated, 'mean_harm_score'):.4f}"
        f" | {_avg(results_ablated, 'mean_safe_score'):.4f} |\n\n"
        f"**delta_gap (ON - ABLATED): {delta_gap:+.4f}**\n"
        f"**n_replay_contamination_min (ablated): {n_replay_contam_min}**\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: gap_on >= {THRESH_C1} (all seeds) | {'PASS' if c1_pass else 'FAIL'}"
        f" | {gap_on:.4f} |\n"
        f"| C2: per-seed delta >= {THRESH_C2} (all seeds) | {'PASS' if c2_pass else 'FAIL'}"
        f" | {[round(d, 4) for d in per_seed_deltas]} |\n"
        f"| C3: gap_ablated >= {THRESH_C3} | {'PASS' if c3_pass else 'FAIL'}"
        f" | {gap_ablated:.4f} |\n"
        f"| C4: n_harm_min >= {THRESH_C4} | {'PASS' if c4_pass else 'FAIL'}"
        f" | {n_harm_min} |\n"
        f"| C5: n_replay_contam_min >= {THRESH_C5} | {'PASS' if c5_pass else 'FAIL'}"
        f" | {n_replay_contam_min} |\n\n"
        f"Criteria met: {criteria_met}/5 -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n\n"
        f"## Per-Seed\n\n"
        f"REALITY_COHERENCE_ON:\n{per_on_rows}\n\n"
        f"REALITY_COHERENCE_ABLATED:\n{per_ablated_rows}\n"
        f"{failure_section}\n"
    )

    metrics = {
        "gap_reality_on":               float(gap_on),
        "gap_ablated":                  float(gap_ablated),
        "delta_gap":                    float(delta_gap),
        "n_harm_eval_min":              float(n_harm_min),
        "n_replay_contamination_min":   float(n_replay_contam_min),
        "n_seeds":                      float(len(seeds)),
        "nav_bias":                     float(nav_bias),
        "alpha_world":                  float(alpha_world),
        "replay_noise_std":             float(REPLAY_NOISE_STD),
        "replay_batch_per_step":        float(REPLAY_BATCH_PER_STEP),
        "per_seed_delta_min":           float(min(per_seed_deltas)) if per_seed_deltas else 0.0,
        "per_seed_delta_max":           float(max(per_seed_deltas)) if per_seed_deltas else 0.0,
        "crit1_pass":                   1.0 if c1_pass else 0.0,
        "crit2_pass":                   1.0 if c2_pass else 0.0,
        "crit3_pass":                   1.0 if c3_pass else 0.0,
        "crit4_pass":                   1.0 if c4_pass else 0.0,
        "crit5_pass":                   1.0 if c5_pass else 0.0,
        "criteria_met":                 float(criteria_met),
    }

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if criteria_met >= 3 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": sum(r["n_fatal"] for r in results_on + results_ablated),
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",            type=int, nargs="+", default=[42, 123])
    parser.add_argument("--warmup",           type=int,   default=400)
    parser.add_argument("--eval-eps",         type=int,   default=50)
    parser.add_argument("--steps",            type=int,   default=200)
    parser.add_argument("--alpha-world",      type=float, default=0.9)
    parser.add_argument("--alpha-self",       type=float, default=0.3)
    parser.add_argument("--harm-scale",       type=float, default=0.02)
    parser.add_argument("--proximity-scale",  type=float, default=0.05)
    parser.add_argument("--nav-bias",         type=float, default=0.40)
    parser.add_argument("--dry-run",          action="store_true",
                        help="Run 3 warmup + 2 eval episodes per cell to check for errors.")
    args = parser.parse_args()

    result = run(
        seeds=tuple(args.seeds),
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        harm_scale=args.harm_scale,
        proximity_harm_scale=args.proximity_scale,
        nav_bias=args.nav_bias,
        dry_run=args.dry_run,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["registered_thresholds"] = {
        "C1_gap_reality_on":          THRESH_C1,
        "C2_per_seed_delta":          THRESH_C2,
        "C3_gap_ablated":             THRESH_C3,
        "C4_n_harm_eval_min":         THRESH_C4,
        "C5_n_replay_contam_min":     THRESH_C5,
    }
    result["seeds"] = list(args.seeds)
    result["conditions"] = ["REALITY_COHERENCE_ON", "REALITY_COHERENCE_ABLATED"]
    result["dispatch_mode"] = "discriminative_pair"
    result["backlog_id"] = "EVB-0023"
    result["evidence_class"] = "discriminative_pair"
    result["claim_ids_tested"] = CLAIM_IDS

    if args.dry_run:
        print("\n[dry-run] Skipping file output.", flush=True)
        sys.exit(0)

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
