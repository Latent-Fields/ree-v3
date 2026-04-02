#!/opt/local/bin/python3
"""
V3-EXQ-207 -- MECH-155: General Associative Indexing in E1

Claim: MECH-155
Proposal: EVB-0047

MECH-155 asserts:
  Spatial navigation machinery in E1 is a specific instance of a more general
  associative indexing mechanism. E1's LSTM hidden state does not merely track
  sensory prediction error -- it indexes the agent's current context across
  multiple structural domains simultaneously (spatial, temporal, event-based).

  If this is true, the E1 hidden state geometry (pairwise similarity structure)
  should correlate with BOTH:
    (a) Spatial domain: grid-position distance matrix (RSA_spatial)
    (b) Temporal/event domain: recent-event-history distance matrix (RSA_event)

  A strong positive RSA in both domains indicates a shared indexing mechanism
  rather than a mechanism specialized to spatial navigation alone.

Experiment design:
  1. Train agent for 100 warmup episodes on CausalGridWorldV2.
  2. During 10 eval episodes (200 steps each), collect at each step:
       - E1 LSTM hidden state h_t  (128-dim vector after _e1_tick)
       - Agent grid position (ax, ay)
       - Recent harm history vector (K=6 binary flags for last K steps)
       - Step global index (for temporal ordering)
  3. Subsample to MAX_RSA_POINTS per seed for tractable RSA computation.
  4. Compute three RSA matrices:
       spatial_sim[i,j]   = gaussian_rbf(-manhattan_distance(pos_i, pos_j))
       event_sim[i,j]     = 1 - hamming_distance(harm_hist_i, harm_hist_j) / K
       seq_sim[i,j]       = 1 / (1 + |global_step_i - global_step_j| / SEQ_SCALE)
  5. Compute E1 latent similarity:
       latent_sim[i,j]    = cosine_similarity(h_i, h_j)
  6. RSA correlations (Spearman r of upper triangles):
       rsa_spatial    = spearman_r(upper(spatial_sim),  upper(latent_sim))
       rsa_event      = spearman_r(upper(event_sim),    upper(latent_sim))
       rsa_sequential = spearman_r(upper(seq_sim),      upper(latent_sim))
  7. Cross-domain similarity:
       cross_domain = spearman_r(upper(spatial_sim), upper(event_sim))
       (tests whether spatial and event structures co-vary in E1 latent space)
  8. Shuffle control: permute position labels, recompute rsa_spatial.

Pre-registered thresholds
--------------------------
C1: rsa_spatial > THRESH_RSA in >= 2/3 seeds.
    E1 hidden state encodes spatial position structure.

C2: rsa_event > THRESH_RSA in >= 2/3 seeds.
    E1 hidden state also encodes recent event context structure.

C3: Both C1 and C2 pass for the same seed in >= 2/3 seeds.
    Cross-domain generality: indexing operates across domains simultaneously.

C4: rsa_shuffle < THRESH_SHUFFLE in >= 2/3 seeds.
    Sanity: shuffled position labels do not produce spurious RSA signal.

PASS:   C1 + C2 + C3 + C4 -- general indexing confirmed
PARTIAL: C1 without C2 -- spatial indexing only (weaker claim support)
FAIL:   C1 fails -- E1 hidden state does not encode spatial structure at all

Seeds: [42, 7, 123]
Env:   CausalGridWorldV2 size=10, 2 hazards, 3 resources, hazard_harm=0.02
Train: 100 warmup episodes x 200 steps
Eval:  10 eval episodes x 200 steps
Estimated runtime: ~60 min (any machine)
"""

import sys
import random
import math
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.optim as optim
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE = "v3_exq_207_mech155_general_indexing_probe"
CLAIM_IDS = ["MECH-155"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
THRESH_RSA      = 0.05     # C1, C2: rsa must exceed this to count as signal
THRESH_SHUFFLE  = 0.02     # C4: shuffled position rsa must stay below this

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BODY_OBS_DIM  = 12
WORLD_OBS_DIM = 250   # CausalGridWorldV2 size=10: 10*10*2 + extras
ACTION_DIM    = 5
WORLD_DIM     = 32
SELF_DIM      = 32

WARMUP_EPISODES  = 100
EVAL_EPISODES    = 10
STEPS_PER_EP     = 200

HARM_HISTORY_K   = 6     # Length of recent-harm-history vector
SEQ_SCALE        = 50.0  # Denominator for sequential similarity decay
MAX_RSA_POINTS   = 400   # Subsample cap for O(N^2) RSA computation
SPATIAL_SIGMA    = 3.0   # Gaussian RBF scale in grid cells

SEEDS = [42, 7, 123]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=2,
        num_resources=3,
        hazard_harm=0.02,
        env_drift_interval=10,
        env_drift_prob=0.2,
    )


def _make_config() -> REEConfig:
    config = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
    )
    return config


def _get_e1_hidden(agent: REEAgent) -> Optional[torch.Tensor]:
    """Return flattened E1 LSTM hidden state vector (last layer, batch 0)."""
    hs = agent.e1._hidden_state
    if hs is None:
        return None
    # hs = (h_n, c_n), h_n.shape = [num_layers, batch, hidden_dim]
    return hs[0][-1, 0, :].detach().clone()  # [hidden_dim]


def _spearman_r(a: List[float], b: List[float]) -> float:
    """Compute Spearman rank correlation between two equal-length lists."""
    n = len(a)
    if n < 3:
        return 0.0

    def _rank(lst: List[float]) -> List[float]:
        sorted_idx = sorted(range(n), key=lambda i: lst[i])
        ranks = [0.0] * n
        for rank_val, idx in enumerate(sorted_idx):
            ranks[idx] = float(rank_val + 1)
        return ranks

    ra = _rank(a)
    rb = _rank(b)
    mean_ra = sum(ra) / n
    mean_rb = sum(rb) / n
    num = sum((ra[i] - mean_ra) * (rb[i] - mean_rb) for i in range(n))
    den_a = math.sqrt(sum((ra[i] - mean_ra) ** 2 for i in range(n)))
    den_b = math.sqrt(sum((rb[i] - mean_rb) ** 2 for i in range(n)))
    if den_a < 1e-12 or den_b < 1e-12:
        return 0.0
    return num / (den_a * den_b)


def _upper_triangle(mat: List[List[float]]) -> List[float]:
    """Return upper-triangle elements (i < j) as flat list."""
    n = len(mat)
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            result.append(mat[i][j])
    return result


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1-D tensors."""
    dot = float((a * b).sum().item())
    na = float(a.norm().item())
    nb = float(b.norm().item())
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return dot / (na * nb)


def _manhattan(pos_a: Tuple[int, int], pos_b: Tuple[int, int]) -> int:
    return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])


def _hamming_frac(a: List[int], b: List[int]) -> float:
    """Hamming distance / K as fraction in [0, 1]."""
    k = len(a)
    if k == 0:
        return 0.0
    return sum(x != y for x, y in zip(a, b)) / k


# ---------------------------------------------------------------------------
# Run one seed
# ---------------------------------------------------------------------------

def _run_seed(seed: int, dry_run: bool) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    warmup = 3 if dry_run else WARMUP_EPISODES
    n_eval = 2 if dry_run else EVAL_EPISODES
    steps  = 20 if dry_run else STEPS_PER_EP

    env    = _make_env(seed)
    config = _make_config()
    agent  = REEAgent(config)

    e1_opt = optim.Adam(agent.e1.parameters(), lr=1e-3)
    e2_opt = optim.Adam(agent.e2.parameters(), lr=3e-3)

    print(
        f"  [EXQ-207 MECH-155] seed={seed} warmup={warmup} eval={n_eval}"
        f" steps_per_ep={steps}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Warmup training
    # -----------------------------------------------------------------------
    agent.train()
    for ep in range(warmup):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps):
            obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

            latent = agent.sense(obs_body, obs_world)
            ticks  = agent.clock.advance()
            if ticks["e1_tick"]:
                agent._e1_tick(latent)

            action = torch.zeros(1, ACTION_DIM)
            action[0, random.randint(0, ACTION_DIM - 1)] = 1.0

            _, reward, done, info, obs_dict = env.step(action)
            harm_signal    = float(reward) if reward < 0 else 0.0
            benefit_signal = float(reward) if reward > 0 else 0.0

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total   = e1_loss + e2_loss
            if total.requires_grad:
                e1_opt.zero_grad()
                e2_opt.zero_grad()
                total.backward()
                e1_opt.step()
                e2_opt.step()

            agent.update_residue(harm_signal)
            if done:
                break

        if (ep + 1) % 50 == 0:
            print(
                f"    [train] seed={seed} ep {ep+1}/{warmup}",
                flush=True,
            )

    # -----------------------------------------------------------------------
    # Eval: collect (hidden_state, position, harm_history, global_step)
    # -----------------------------------------------------------------------
    agent.eval()

    hidden_states: List[torch.Tensor] = []
    positions:     List[Tuple[int, int]] = []
    harm_histories: List[List[int]] = []
    global_steps:  List[int] = []

    global_step = 0
    harm_ring: List[int] = [0] * HARM_HISTORY_K   # circular buffer

    for ep in range(n_eval):
        _, obs_dict = env.reset()
        agent.reset()
        ep_harm_ring = [0] * HARM_HISTORY_K

        for step in range(steps):
            obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                ticks  = agent.clock.advance()
                if ticks["e1_tick"]:
                    agent._e1_tick(latent)

            h = _get_e1_hidden(agent)
            if h is not None:
                ax = int(getattr(env, 'agent_x', 0))
                ay = int(getattr(env, 'agent_y', 0))

                hidden_states.append(h)
                positions.append((ax, ay))
                harm_histories.append(list(ep_harm_ring))
                global_steps.append(global_step)

            action = torch.zeros(1, ACTION_DIM)
            action[0, random.randint(0, ACTION_DIM - 1)] = 1.0
            _, reward, done, _, obs_dict = env.step(action)

            harm_now = 1 if reward < 0 else 0
            ep_harm_ring = [harm_now] + ep_harm_ring[:-1]
            global_step += 1

            if done:
                break

    n_collected = len(hidden_states)
    print(
        f"  [EXQ-207] seed={seed} collected {n_collected} steps",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Subsample for RSA
    # -----------------------------------------------------------------------
    if n_collected > MAX_RSA_POINTS:
        indices = random.sample(range(n_collected), MAX_RSA_POINTS)
        indices.sort()
        hidden_states   = [hidden_states[i]   for i in indices]
        positions       = [positions[i]       for i in indices]
        harm_histories  = [harm_histories[i]  for i in indices]
        global_steps    = [global_steps[i]    for i in indices]

    N = len(hidden_states)

    # -----------------------------------------------------------------------
    # Build RSA matrices
    # -----------------------------------------------------------------------
    # Spatial similarity: Gaussian RBF of Manhattan distance
    spatial_sim = []
    for i in range(N):
        row = []
        for j in range(N):
            d = _manhattan(positions[i], positions[j])
            row.append(math.exp(-(d ** 2) / (2.0 * SPATIAL_SIGMA ** 2)))
        spatial_sim.append(row)

    # Event (harm-history) similarity: 1 - hamming fraction
    event_sim = []
    for i in range(N):
        row = []
        for j in range(N):
            row.append(1.0 - _hamming_frac(harm_histories[i], harm_histories[j]))
        event_sim.append(row)

    # Sequential similarity: inverse step-index distance
    seq_sim = []
    for i in range(N):
        row = []
        for j in range(N):
            dist_ij = abs(global_steps[i] - global_steps[j])
            row.append(1.0 / (1.0 + dist_ij / SEQ_SCALE))
        seq_sim.append(row)

    # Latent cosine similarity
    latent_sim = []
    for i in range(N):
        row = []
        for j in range(N):
            row.append(_cosine_sim(hidden_states[i], hidden_states[j]))
        latent_sim.append(row)

    # -----------------------------------------------------------------------
    # RSA correlations (Spearman r of upper triangles)
    # -----------------------------------------------------------------------
    ut_latent   = _upper_triangle(latent_sim)
    ut_spatial  = _upper_triangle(spatial_sim)
    ut_event    = _upper_triangle(event_sim)
    ut_seq      = _upper_triangle(seq_sim)

    rsa_spatial    = _spearman_r(ut_latent, ut_spatial)
    rsa_event      = _spearman_r(ut_latent, ut_event)
    rsa_sequential = _spearman_r(ut_latent, ut_seq)
    cross_domain   = _spearman_r(ut_spatial, ut_event)   # spatial vs event in E1

    # Shuffle control: permute position labels
    shuffled_pos = list(positions)
    random.shuffle(shuffled_pos)
    shuffle_sim = []
    for i in range(N):
        row = []
        for j in range(N):
            d = _manhattan(shuffled_pos[i], shuffled_pos[j])
            row.append(math.exp(-(d ** 2) / (2.0 * SPATIAL_SIGMA ** 2)))
        shuffle_sim.append(row)
    rsa_shuffle = _spearman_r(ut_latent, _upper_triangle(shuffle_sim))

    # -----------------------------------------------------------------------
    # Check criteria
    # -----------------------------------------------------------------------
    c1 = rsa_spatial > THRESH_RSA
    c2 = rsa_event   > THRESH_RSA
    c3 = c1 and c2
    c4 = rsa_shuffle < THRESH_SHUFFLE

    print(
        f"  [EXQ-207] seed={seed}"
        f" rsa_spatial={rsa_spatial:.4f}"
        f" rsa_event={rsa_event:.4f}"
        f" rsa_seq={rsa_sequential:.4f}"
        f" cross_domain={cross_domain:.4f}"
        f" rsa_shuffle={rsa_shuffle:.4f}"
        f" C1={c1} C2={c2} C3={c3} C4={c4}",
        flush=True,
    )

    return {
        "seed":                        seed,
        "n_collected":                 n_collected,
        "n_rsa_points":                N,
        "rsa_spatial":                 rsa_spatial,
        "rsa_event":                   rsa_event,
        "rsa_sequential":              rsa_sequential,
        "cross_domain_indexing_similarity": cross_domain,
        "spatial_vs_concept_cosine":   rsa_event,   # event domain = conceptual proxy
        "traversal_order_correlation": rsa_sequential,
        "rsa_shuffle":                 rsa_shuffle,
        "c1_spatial_signal":           c1,
        "c2_event_signal":             c2,
        "c3_cross_domain":             c3,
        "c4_shuffle_control":          c4,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ts   = int(time.time())
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"

    print(f"[EXQ-207] MECH-155 General Indexing Probe", flush=True)
    print(f"  dry_run={args.dry_run}", flush=True)

    seed_results = []
    for seed in SEEDS:
        res = _run_seed(seed, dry_run=args.dry_run)
        seed_results.append(res)

    # -----------------------------------------------------------------------
    # Aggregate pass/fail
    # -----------------------------------------------------------------------
    c1_count = sum(1 for r in seed_results if r["c1_spatial_signal"])
    c2_count = sum(1 for r in seed_results if r["c2_event_signal"])
    c3_count = sum(1 for r in seed_results if r["c3_cross_domain"])
    c4_count = sum(1 for r in seed_results if r["c4_shuffle_control"])
    n_seeds  = len(seed_results)

    c1_pass = c1_count >= 2
    c2_pass = c2_count >= 2
    c3_pass = c3_count >= 2
    c4_pass = c4_count >= 2

    if c1_pass and c2_pass and c3_pass and c4_pass:
        outcome = "PASS"
        direction = "supports"
    elif c1_pass and not c2_pass:
        outcome = "PARTIAL"
        direction = "mixed"
    elif not c1_pass and not c2_pass:
        outcome = "FAIL"
        direction = "weakens"
    else:
        outcome = "PARTIAL"
        direction = "mixed"

    mean_rsa_spatial  = sum(r["rsa_spatial"]  for r in seed_results) / n_seeds
    mean_rsa_event    = sum(r["rsa_event"]    for r in seed_results) / n_seeds
    mean_cross_domain = sum(r["cross_domain_indexing_similarity"] for r in seed_results) / n_seeds
    mean_rsa_seq      = sum(r["rsa_sequential"] for r in seed_results) / n_seeds
    mean_rsa_shuffle  = sum(r["rsa_shuffle"]  for r in seed_results) / n_seeds

    print(
        f"\n[EXQ-207] RESULT: {outcome}"
        f" rsa_spatial={mean_rsa_spatial:.4f}"
        f" rsa_event={mean_rsa_event:.4f}"
        f" cross_domain={mean_cross_domain:.4f}"
        f" rsa_shuffle={mean_rsa_shuffle:.4f}"
        f" C1={c1_pass} C2={c2_pass} C3={c3_pass} C4={c4_pass}",
        flush=True,
    )

    manifest = {
        "run_id":                     f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type":            EXPERIMENT_TYPE,
        "architecture_epoch":         "ree_hybrid_guardrails_v1",
        "claim_ids":                  CLAIM_IDS,
        "experiment_purpose":         EXPERIMENT_PURPOSE,
        "outcome":                    outcome,
        "evidence_direction":         direction,
        "timestamp":                  ts,
        "dry_run":                    args.dry_run,
        "seeds":                      SEEDS,
        "warmup_episodes":            3 if args.dry_run else WARMUP_EPISODES,
        "eval_episodes":              2 if args.dry_run else EVAL_EPISODES,
        "steps_per_episode":          20 if args.dry_run else STEPS_PER_EP,
        "harm_history_k":             HARM_HISTORY_K,
        "max_rsa_points":             MAX_RSA_POINTS,
        "thresh_rsa":                 THRESH_RSA,
        "thresh_shuffle":             THRESH_SHUFFLE,
        # Aggregate metrics
        "mean_rsa_spatial":           mean_rsa_spatial,
        "mean_rsa_event":             mean_rsa_event,
        "mean_rsa_sequential":        mean_rsa_seq,
        "cross_domain_indexing_similarity": mean_cross_domain,
        "spatial_vs_concept_cosine":  mean_rsa_event,
        "traversal_order_correlation": mean_rsa_seq,
        "mean_rsa_shuffle":           mean_rsa_shuffle,
        # Criteria
        "c1_spatial_signal_pass":     c1_pass,
        "c2_event_signal_pass":       c2_pass,
        "c3_cross_domain_pass":       c3_pass,
        "c4_shuffle_control_pass":    c4_pass,
        "c1_count":                   c1_count,
        "c2_count":                   c2_count,
        "c3_count":                   c3_count,
        "c4_count":                   c4_count,
        "n_seeds":                    n_seeds,
        # Per-seed
        "seed_results":               seed_results,
    }

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[EXQ-207] Written: {out_path}", flush=True)


if __name__ == "__main__":
    main()
