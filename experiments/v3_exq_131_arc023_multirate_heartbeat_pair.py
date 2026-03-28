#!/opt/local/bin/python3
"""
V3-EXQ-131 -- ARC-023 Multi-Rate Heartbeat Discriminative Pair

Claims: ARC-023
Proposal: EXP-0033 / EVB-0025

ARC-023 asserts: "Three BG-like loops operate at characteristic thalamic heartbeat rates."
  - E1 loop: continuous/frame-rate (updates every step)
  - E2 loop: motor-command rate (updates every step, but conceptually slower than E1)
  - E3 loop: deliberation rate -- slowest, updates at a coarser timescale

The functional restatement (required for ANN proxy):
  Three loops operate at distinct update rates. An update-rate management mechanism
  prevents loop drift under variable processing latency. The functional requirement is
  rate-separated asynchronous updates regardless of substrate; the biological substrate
  (thalamic pacemaking) is not required.

V3-accessible proxy design
---------------------------
CausalGridWorldV2 provides a synchronous environment. We implement rate separation
via time-multiplexing: E3 updates only every E3_HEARTBEAT_K steps; E1/E2 update
every step. This is the SD-006 phase 1 proxy (time-multiplexed, not async).

The claim predicts that running E3 at a deliberately slower rate (deliberation-timescale)
should produce BETTER harm evaluation quality than running E3 at the same rate as E1/E2
(same-rate synchronous updates). The reason: slower E3 integration allows the harm
evaluation head to aggregate evidence across a full action chunk before updating, rather
than reacting to every instantaneous signal change.

MULTIRATE_ON (rate-separated: ARC-023 architecture):
  - E1 (prediction error) updates every step
  - E2 (motor-sensory) updates every step
  - E3 (harm evaluation, deliberation) updates only every E3_HEARTBEAT_K steps
  - Harm eval head sees E3 output accumulated over K steps
  - E3 deliberates over an action chunk, not per-step noise

MULTIRATE_ABLATED (same-rate: ablation):
  - E1, E2, E3 all update every step
  - No rate separation -- E3 processes every individual step identically to E1/E2
  - E3 deliberation degrades to instantaneous per-step processing

Measurement: harm_eval_head discrimination quality (gap between harm-positive and
harm-negative evaluations at eval). The prediction: rate-separated E3 reduces
integration noise, improving the signal-to-noise ratio of the harm evaluation.

Secondary diagnostic: harm_eval_variance -- the variance of harm_eval scores at
harm events. Slower E3 should produce lower variance (less noise, more stable signal).

If ARC-023's rate-separation principle is functionally necessary:
  - MULTIRATE_ON harm_eval gap > MULTIRATE_ABLATED harm_eval gap
  - The delta should be consistently positive across seeds
  - MULTIRATE_ON harm_eval_variance <= MULTIRATE_ABLATED harm_eval_variance

Pre-registered acceptance criteria:
  C1: gap_multirate_on >= 0.04    (MULTIRATE_ON harm eval above floor -- both seeds)
  C2: per-seed delta (ON - ABLATED) >= 0.02   (rate separation adds >=2pp -- both seeds)
  C3: gap_ablated >= 0.0          (ablation still learns something -- data quality)
  C4: n_harm_eval_min >= 20       (sufficient harm events in eval -- both seeds)
  C5: no fatal errors in either condition

Decision scoring:
  PASS (all 5): retain_ree -- rate separation improves E3 deliberation quality
  C1+C2+C4:    hybridize  -- rate separation effect present but magnitude uncertain
  C2 fails:    retire_ree_claim -- no detectable advantage of rate separation at V3 proxy scale

Note on ARC-023 status: claim has evidence_quality_note that SD-006 phase 2 async
implementation is the full blocker. This experiment tests the functional claim (rate
separation benefits E3 quality) using the phase 1 time-multiplexed proxy. A PASS here
is partial support pending SD-006 phase 2; a FAIL here is informative about whether
the functional claim holds at all in the synchronous proxy regime.

Seeds: [42, 123] (matched -- same env per seed across conditions)
Env:   CausalGridWorldV2, size=6, 4 hazards, nav_bias=0.40
Warmup: 400 eps x 200 steps
Eval:  50 eps x 200 steps (no training, fixed weights)
E3 heartbeat rate: E3_HEARTBEAT_K=5 (E3 updates every 5 steps in MULTIRATE_ON)
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


EXPERIMENT_TYPE = "v3_exq_131_arc023_multirate_heartbeat_pair"
CLAIM_IDS = ["ARC-023"]

# Pre-registered thresholds
THRESH_C1 = 0.04   # gap_multirate_on >= 0.04 (MULTIRATE_ON harm eval above floor)
THRESH_C2 = 0.02   # per-seed delta (ON - ABLATED) >= 0.02
THRESH_C3 = 0.0    # gap_ablated >= 0.0 (ablation still learns something)
THRESH_C4 = 20     # n_harm_eval_min >= 20 (data quality)

# E3 heartbeat rate: E3 updates every E3_HEARTBEAT_K steps in MULTIRATE_ON
# Value of 5 chosen to represent deliberation-rate separation (E3 ~ 1/5 of E1/E2 rate)
E3_HEARTBEAT_K = 5

# Dimension constants
HARM_OBS_DIM = 51  # SD-010 nociceptive stream: hazard_field_view[25] + resource_field_view[25] + harm_exposure[1]


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _hazard_approach_action(env: CausalGridWorldV2, n_actions: int) -> int:
    """Return action index that moves toward the nearest hazard gradient.
    Falls back to random if proxy fields are unavailable."""
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
    multirate_on: bool,
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
    e3_heartbeat_k: int,
    dry_run: bool,
) -> Dict:
    """Run one (seed, condition) cell; return harm eval gap metrics.

    MULTIRATE_ON:      E3 updates only every e3_heartbeat_k steps (rate-separated).
    MULTIRATE_ABLATED: E3 updates every step (same rate as E1/E2).
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    cond_label = "MULTIRATE_ON" if multirate_on else "MULTIRATE_ABLATED"

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
    body_obs_dim = env.body_obs_dim

    config = REEConfig.from_dims(
        body_obs_dim=body_obs_dim,
        world_obs_dim=world_obs_dim,
        action_dim=n_actions,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=0,
    )
    agent = REEAgent(config)

    MAX_BUF = 2000

    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n
    ]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())

    optimizer = optim.Adam(standard_params, lr=lr)
    harm_eval_optimizer = optim.Adam(harm_eval_params, lr=1e-4)

    # Harm eval buffer: accumulates z_world latents and harm labels
    buf_zw: List[torch.Tensor] = []
    buf_labels: List[float] = []

    # In MULTIRATE_ON, E3 accumulates z_world over K steps, then updates once.
    # We track this as a rolling window that gets flushed at heartbeat boundary.
    e3_step_buffer_zw: List[torch.Tensor] = []
    e3_step_buffer_labels: List[float] = []

    counts: Dict[str, int] = {
        "hazard_approach": 0,
        "env_caused_hazard": 0,
        "agent_caused_hazard": 0,
        "none": 0,
    }

    if dry_run:
        actual_warmup = min(3, warmup_episodes)
        actual_eval = min(2, eval_episodes)
    else:
        actual_warmup = warmup_episodes
        actual_eval = eval_episodes

    # --- TRAIN ---
    agent.train()

    step_count = 0

    for ep in range(actual_warmup):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_curr = latent.z_world.detach()

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

            step_count += 1
            harm_sig_float = float(harm_signal)
            is_harm = harm_sig_float < 0

            # --- E1 + E2 update (every step, both conditions) ---
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            # --- E3 / harm_eval update (rate-controlled) ---
            if multirate_on:
                # Accumulate z_world and labels into step buffer
                # E3 harm_eval training only fires at the end of each K-step chunk
                e3_step_buffer_zw.append(z_world_curr)
                e3_step_buffer_labels.append(1.0 if is_harm else 0.0)

                if len(e3_step_buffer_zw) >= e3_heartbeat_k:
                    # Heartbeat tick: flush step buffer into main training buffer
                    # Use averaged z_world over the chunk as the E3 "heartbeat sample"
                    chunk_zw = torch.stack(e3_step_buffer_zw, dim=0).mean(dim=0, keepdim=True)
                    chunk_label = 1.0 if any(l > 0.5 for l in e3_step_buffer_labels) else 0.0
                    buf_zw.append(chunk_zw)
                    buf_labels.append(chunk_label)
                    e3_step_buffer_zw = []
                    e3_step_buffer_labels = []
                    if len(buf_zw) > MAX_BUF:
                        buf_zw = buf_zw[-MAX_BUF:]
                        buf_labels = buf_labels[-MAX_BUF:]

                    # Harm eval training at heartbeat boundary
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
                        # Re-encode z_world through harm_eval_head
                        pred = agent.e3.harm_eval(zw_b)
                        loss_he = F.mse_loss(pred, labels_b)
                        if loss_he.requires_grad:
                            harm_eval_optimizer.zero_grad()
                            loss_he.backward()
                            torch.nn.utils.clip_grad_norm_(
                                agent.e3.harm_eval_head.parameters(), 0.5
                            )
                            harm_eval_optimizer.step()
            else:
                # ABLATED: E3 harm_eval updates every step (same rate as E1/E2)
                buf_zw.append(z_world_curr)
                buf_labels.append(1.0 if is_harm else 0.0)
                if len(buf_zw) > MAX_BUF:
                    buf_zw = buf_zw[-MAX_BUF:]
                    buf_labels = buf_labels[-MAX_BUF:]

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
                f" contact={counts['env_caused_hazard']+counts['agent_caused_hazard']}",
                flush=True,
            )

    # --- EVAL ---
    agent.eval()

    eval_scores_harm: List[float] = []
    eval_scores_safe: List[float] = []
    n_fatal = 0

    # Track per-step z_world for variance diagnostic
    e3_chunk_zw_eval: List[torch.Tensor] = []
    eval_step_count = 0

    for _ in range(actual_eval):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                z_world_curr = latent.z_world

            action_idx = random.randint(0, n_actions - 1)
            action = _action_to_onehot(action_idx, n_actions, agent.device)
            agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action)
            is_harm = float(harm_signal) < 0
            eval_step_count += 1

            if multirate_on:
                # In eval, E3 evaluates using accumulated z_world over K steps
                e3_chunk_zw_eval.append(z_world_curr.detach())
                if len(e3_chunk_zw_eval) >= e3_heartbeat_k:
                    # Heartbeat evaluation: average z_world over chunk
                    chunk_zw = torch.stack(e3_chunk_zw_eval, dim=0).mean(dim=0, keepdim=True)
                    e3_chunk_zw_eval = []
                    try:
                        with torch.no_grad():
                            score = float(agent.e3.harm_eval(chunk_zw).item())
                        # Assign this score to the is_harm label of the last step in chunk
                        if is_harm:
                            eval_scores_harm.append(score)
                        else:
                            eval_scores_safe.append(score)
                    except Exception:
                        n_fatal += 1
            else:
                # ABLATED: eval per step
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

    # Variance diagnostic: lower variance in MULTIRATE_ON indicates more stable signal
    var_harm = float(np.var(eval_scores_harm)) if n_harm_eval > 1 else 0.0
    var_safe = float(np.var(eval_scores_safe)) if n_safe_eval > 1 else 0.0
    var_harm_eval = (var_harm + var_safe) / 2.0

    print(
        f"  [eval] seed={seed} cond={cond_label}"
        f" gap={gap:.4f} mean_harm={mean_harm:.4f} mean_safe={mean_safe:.4f}"
        f" n_harm={n_harm_eval} n_safe={n_safe_eval}"
        f" var_harm_eval={var_harm_eval:.6f}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "multirate_on": multirate_on,
        "harm_eval_gap": float(gap),
        "mean_harm_score": float(mean_harm),
        "mean_safe_score": float(mean_safe),
        "n_harm_eval": int(n_harm_eval),
        "n_safe_eval": int(n_safe_eval),
        "n_fatal": int(n_fatal),
        "var_harm_eval": float(var_harm_eval),
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
    e3_heartbeat_k: int = E3_HEARTBEAT_K,
    dry_run: bool = False,
    **kwargs,
) -> dict:
    """Discriminative pair: MULTIRATE_ON (E3 updates every E3_HEARTBEAT_K steps --
    ARC-023 rate-separated architecture) vs MULTIRATE_ABLATED (E3 updates every step
    -- same rate as E1/E2, ablation of rate separation).
    Tests ARC-023: distinct update rates for the three BG-like loops improve E3
    deliberation quality (harm evaluation signal-to-noise).
    """
    results_on:      List[Dict] = []
    results_ablated: List[Dict] = []

    # Run cells in matched-seed order: for each seed, run both conditions
    for seed in seeds:
        for multirate in [True, False]:
            label = "MULTIRATE_ON" if multirate else "MULTIRATE_ABLATED"
            print(
                f"\n[V3-EXQ-131] {label} seed={seed}"
                f" warmup={warmup_episodes} eval={eval_episodes}"
                f" e3_heartbeat_k={e3_heartbeat_k}"
                f" nav_bias={nav_bias}",
                flush=True,
            )
            r = _run_single(
                seed=seed,
                multirate_on=multirate,
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
                e3_heartbeat_k=e3_heartbeat_k,
                dry_run=dry_run,
            )
            if multirate:
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

    var_on      = _avg(results_on,      "var_harm_eval")
    var_ablated = _avg(results_ablated, "var_harm_eval")

    # Pre-registered PASS criteria
    # C1: MULTIRATE_ON gap >= THRESH_C1 in ALL seeds
    c1_pass = all(r["harm_eval_gap"] >= THRESH_C1 for r in results_on)
    # C2: per-seed delta (ON - ABLATED) >= THRESH_C2 in ALL seeds
    per_seed_deltas: List[float] = []
    for r_on in results_on:
        matching = [r for r in results_ablated if r["seed"] == r_on["seed"]]
        if matching:
            per_seed_deltas.append(r_on["harm_eval_gap"] - matching[0]["harm_eval_gap"])
    c2_pass = len(per_seed_deltas) > 0 and all(d >= THRESH_C2 for d in per_seed_deltas)
    # C3: MULTIRATE_ABLATED gap >= 0 (ablation still learns something)
    c3_pass = gap_ablated >= THRESH_C3
    # C4: sufficient harm events in eval
    c4_pass = n_harm_min >= THRESH_C4
    # C5: no fatal errors
    c5_pass = all(r["n_fatal"] == 0 for r in results_on + results_ablated)

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])
    status = "PASS" if all_pass else "FAIL"

    if all_pass:
        decision = "retain_ree"
    elif c1_pass and c4_pass:
        decision = "hybridize"
    else:
        decision = "retire_ree_claim"

    print(f"\n[V3-EXQ-131] Results:", flush=True)
    print(
        f"  gap_on={gap_on:.4f}"
        f" gap_ablated={gap_ablated:.4f}"
        f" delta_gap={delta_gap:+.4f}",
        flush=True,
    )
    print(
        f"  var_on={var_on:.6f}"
        f" var_ablated={var_ablated:.6f}"
        f" (lower var_on=more stable E3 heartbeat signal)",
        flush=True,
    )
    print(
        f"  n_harm_min={n_harm_min}"
        f"  per_seed_deltas={[round(d, 4) for d in per_seed_deltas]}"
        f"  decision={decision}  status={status} ({criteria_met}/5)",
        flush=True,
    )

    failure_notes: List[str] = []
    if not c1_pass:
        failing = [r["seed"] for r in results_on if r["harm_eval_gap"] < THRESH_C1]
        failure_notes.append(
            f"C1 FAIL: MULTIRATE_ON gap below {THRESH_C1} in seeds {failing}"
            " -- rate-separated E3 does not produce a discriminative harm signal"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: per-seed deltas {[round(d, 4) for d in per_seed_deltas]}"
            f" < {THRESH_C2}"
            " -- rate separation does not add >=2pp over same-rate E3"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: gap_ablated={gap_ablated:.4f} < {THRESH_C3}"
            " -- same-rate ablation fails entirely; confound check"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: n_harm_min={n_harm_min} < {THRESH_C4}"
            " -- insufficient harm events in eval phase"
        )
    if not c5_pass:
        fatal_counts = {r["condition"]: r["n_fatal"] for r in results_on + results_ablated}
        failure_notes.append(f"C5 FAIL: fatal errors detected -- {fatal_counts}")

    for note in failure_notes:
        print(f"  {note}", flush=True)

    if all_pass:
        interpretation = (
            "ARC-023 SUPPORTED (partial -- V3 proxy): rate-separated E3 (heartbeat_k="
            f"{e3_heartbeat_k}, MULTIRATE_ON) produces higher harm_eval quality"
            f" (gap_on={gap_on:.4f} vs gap_ablated={gap_ablated:.4f},"
            f" delta={delta_gap:+.4f} across {len(seeds)} seeds)."
            " E3 integrating z_world evidence over a K-step deliberation window"
            " reduces per-step noise, improving harm/safe discrimination. This is"
            " consistent with ARC-023's functional claim: rate-separated loops"
            " with E3 at a slower deliberation rate produce better evaluation quality."
            " Note: this is a synchronous time-multiplexed proxy (SD-006 phase 1)."
            " Full async implementation (SD-006 phase 2) required for final validation."
        )
    elif c1_pass and c4_pass:
        interpretation = (
            f"Partial support: MULTIRATE_ON achieves gap={gap_on:.4f} (C1 PASS)"
            f" but per-seed delta={per_seed_deltas} (C2 FAIL,"
            f" threshold={THRESH_C2}). Rate-separated E3 produces adequate harm"
            " discrimination but the advantage over same-rate E3 is below the"
            " pre-registered threshold at this scale. Possible: training scale is"
            " insufficient to reveal the integration benefit; or heartbeat_k="
            f"{e3_heartbeat_k} is not the optimal rate ratio at world_dim=32."
        )
    else:
        interpretation = (
            f"ARC-023 NOT supported at V3 proxy level: gap_on={gap_on:.4f}"
            f" (C1 {'PASS' if c1_pass else 'FAIL'}),"
            f" delta={delta_gap:+.4f} (C2 {'PASS' if c2_pass else 'FAIL'})."
            " Rate-separated E3 does not produce a detectable improvement in"
            " harm_eval discrimination at this training scale. The synchronous"
            " time-multiplexed proxy may not capture the full benefit of async"
            " rate separation (SD-006 phase 2 required). Note: the chunk-averaging"
            " approach used here (mean z_world over K steps) is a conservative"
            " proxy -- the benefit of heartbeat synchronization may require the"
            " full async E3 implementation to manifest clearly."
        )

    per_on_rows = "\n".join(
        f"  seed={r['seed']}: gap={r['harm_eval_gap']:.4f}"
        f" n_harm={r['n_harm_eval']}"
        f" var_harm_eval={r['var_harm_eval']:.6f}"
        for r in results_on
    )
    per_ablated_rows = "\n".join(
        f"  seed={r['seed']}: gap={r['harm_eval_gap']:.4f}"
        f" n_harm={r['n_harm_eval']}"
        f" var_harm_eval={r['var_harm_eval']:.6f}"
        for r in results_ablated
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-131 -- ARC-023 Multi-Rate Heartbeat Discriminative Pair\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** ARC-023\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**Conditions:** MULTIRATE_ON vs MULTIRATE_ABLATED\n"
        f"**Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps\n"
        f"**E3 heartbeat rate:** every {e3_heartbeat_k} steps (MULTIRATE_ON only)\n"
        f"**Env:** CausalGridWorldV2 size=6, 4 hazards, nav_bias={nav_bias}\n\n"
        f"## Design\n\n"
        f"ARC-023 asserts three BG-like loops operate at characteristic thalamic heartbeat rates.\n"
        f"Functional restatement: E3 (deliberation loop) operates at a slower rate than E1/E2.\n\n"
        f"MULTIRATE_ON: E3 harm_eval trains/evaluates on averaged z_world over {e3_heartbeat_k}-step chunks.\n"
        f"E1 and E2 update every step as usual.\n"
        f"E3 heartbeat fires every {e3_heartbeat_k} steps (time-multiplexed SD-006 phase 1 proxy).\n\n"
        f"MULTIRATE_ABLATED: E3 harm_eval trains/evaluates per step.\n"
        f"All loops update at the same rate -- no rate separation.\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: gap_on >= {THRESH_C1} in all seeds  (MULTIRATE_ON harm eval above floor)\n"
        f"C2: per-seed delta (ON - ABLATED) >= {THRESH_C2} in all seeds"
        f"  (rate separation adds >=2pp)\n"
        f"C3: gap_ablated >= {THRESH_C3}  (ablation learns something; data quality)\n"
        f"C4: n_harm_eval_min >= {THRESH_C4}  (sufficient harm events in eval)\n"
        f"C5: no fatal errors\n\n"
        f"## Results\n\n"
        f"| Condition | gap (avg) | mean_harm | mean_safe | var_harm_eval |\n"
        f"|-----------|-----------|-----------|-----------|---------------|\n"
        f"| MULTIRATE_ON      | {gap_on:.4f}"
        f" | {_avg(results_on,      'mean_harm_score'):.4f}"
        f" | {_avg(results_on,      'mean_safe_score'):.4f}"
        f" | {var_on:.6f} |\n"
        f"| MULTIRATE_ABLATED | {gap_ablated:.4f}"
        f" | {_avg(results_ablated, 'mean_harm_score'):.4f}"
        f" | {_avg(results_ablated, 'mean_safe_score'):.4f}"
        f" | {var_ablated:.6f} |\n\n"
        f"**delta_gap (ON - ABLATED): {delta_gap:+.4f}**\n\n"
        f"Diagnostic: var_harm_eval -- lower variance in MULTIRATE_ON indicates more stable\n"
        f"E3 heartbeat integration (less per-step noise).\n"
        f"  MULTIRATE_ON: {var_on:.6f}  |  MULTIRATE_ABLATED: {var_ablated:.6f}\n\n"
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
        f"| C5: no fatal errors | {'PASS' if c5_pass else 'FAIL'} | -- |\n\n"
        f"Criteria met: {criteria_met}/5 -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n\n"
        f"## Per-Seed\n\n"
        f"MULTIRATE_ON:\n{per_on_rows}\n\n"
        f"MULTIRATE_ABLATED:\n{per_ablated_rows}\n"
        f"{failure_section}\n"
    )

    metrics = {
        "gap_multirate_on":      float(gap_on),
        "gap_ablated":           float(gap_ablated),
        "delta_gap":             float(delta_gap),
        "n_harm_eval_min":       float(n_harm_min),
        "n_seeds":               float(len(seeds)),
        "nav_bias":              float(nav_bias),
        "e3_heartbeat_k":        float(e3_heartbeat_k),
        "alpha_world":           float(alpha_world),
        "per_seed_delta_min":    float(min(per_seed_deltas)) if per_seed_deltas else 0.0,
        "per_seed_delta_max":    float(max(per_seed_deltas)) if per_seed_deltas else 0.0,
        "var_harm_eval_on":      float(var_on),
        "var_harm_eval_ablated": float(var_ablated),
        "var_delta":             float(var_on - var_ablated),
        "crit1_pass":            1.0 if c1_pass else 0.0,
        "crit2_pass":            1.0 if c2_pass else 0.0,
        "crit3_pass":            1.0 if c3_pass else 0.0,
        "crit4_pass":            1.0 if c4_pass else 0.0,
        "crit5_pass":            1.0 if c5_pass else 0.0,
        "criteria_met":          float(criteria_met),
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
    parser.add_argument("--seeds",              type=int, nargs="+", default=[42, 123])
    parser.add_argument("--warmup",             type=int,   default=400)
    parser.add_argument("--eval-eps",           type=int,   default=50)
    parser.add_argument("--steps",              type=int,   default=200)
    parser.add_argument("--alpha-world",        type=float, default=0.9)
    parser.add_argument("--alpha-self",         type=float, default=0.3)
    parser.add_argument("--harm-scale",         type=float, default=0.02)
    parser.add_argument("--proximity-scale",    type=float, default=0.05)
    parser.add_argument("--nav-bias",           type=float, default=0.40)
    parser.add_argument("--e3-heartbeat-k",     type=int,   default=E3_HEARTBEAT_K,
                        help="E3 heartbeat rate: E3 updates every K steps (MULTIRATE_ON only).")
    parser.add_argument("--dry-run",            action="store_true",
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
        e3_heartbeat_k=args.e3_heartbeat_k,
        dry_run=args.dry_run,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["registered_thresholds"] = {
        "C1_gap_multirate_on":  THRESH_C1,
        "C2_per_seed_delta":    THRESH_C2,
        "C3_gap_ablated":       THRESH_C3,
        "C4_n_harm_eval_min":   THRESH_C4,
    }
    result["seeds"] = list(args.seeds)
    result["conditions"] = ["MULTIRATE_ON", "MULTIRATE_ABLATED"]
    result["dispatch_mode"] = "discriminative_pair"
    result["backlog_id"] = "EVB-0025"
    result["evidence_class"] = "discriminative_pair"
    result["claim_ids_tested"] = CLAIM_IDS
    result["e3_heartbeat_k"] = args.e3_heartbeat_k

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
