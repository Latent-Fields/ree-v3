#!/opt/local/bin/python3
"""
V3-EXQ-140 -- MECH-094 Hypothesis Tag Write Gate Discriminative Pair

Claims: MECH-094
Proposal: EXP-0070 / EVB-0060

MECH-094 asserts:
  "Hypothesis tag is a categorical write gate separating simulation from
  committed residue updates; tag loss is the PTSD mechanism."

Specifically:
  - During simulation/replay (hypothesis_tag=True), the post-commit error
    channel is suppressed -- harm signals from replayed content do NOT write to
    the residue accumulation buffer (phi(z) write pathway).
  - Only real-interaction harm events (hypothesis_tag=False) train the harm_eval
    head (post-commit channel open).
  - Tag loss pathology: when the tag is absent, replayed harm accumulates in
    phi(z) as if it were real -- the mechanism of MECH-076 (map deformation)
    and the PTSD flashback model (re-experiencing as real harm trace).

Functional restatement (from claims.yaml):
  The tag is categorical, not a precision attenuation parameter: it routes
  simulation signals away from the phi(z) write pathway entirely. Attenuation
  of phi(z) readout during imagination is a secondary effect; route separation
  is primary. The qualitatively different affective texture of imagination vs.
  real experience is the phenomenological signature of this channel suppression.

V3-accessible proxy design
--------------------------
We simulate the hypothesis_tag gate by controlling which harm experiences
are allowed to train the harm_eval head (the V3 proxy for phi(z) write).

The experiment runs N_REPLAY_PER_EP "replay" (simulated) episodes per real
episode. In replay episodes the agent encounters hazards in a perturbed (noisy)
version of the environment -- these are structurally similar to real harm events
but are "internally generated" content (the agent is replaying rather than
acting in the real world).

TAG_GATE_ON (MECH-094 architecture):
  - Real episodes: harm_eval buffer updated normally (hypothesis_tag=False).
  - Replay episodes: harm buffer NOT updated -- hypothesis_tag=True gates all
    harm signals away from the phi(z) write pathway.
  - Result: harm_eval head is trained ONLY on real harm events.
  - Prediction: harm_eval score is high on real harm steps, low on safe steps
    (good discrimination). Replay harm does not inflate the baseline.

TAG_GATE_ABLATED (tag loss / PTSD mechanism):
  - Real and replay episodes: harm_eval buffer updated for BOTH.
  - hypothesis_tag absent: replay harm signals accumulate alongside real ones.
  - Result: harm_eval head is trained on real + replayed harm equally.
  - Prediction: harm_eval score is also high on replay harm steps (spurious
    elevation). Contamination gap (replay harm eval score vs safe score) is
    high -- harm_eval cannot distinguish real from replayed harm.

Key discriminative metrics
--------------------------
The two conditions share identical training data volume; they differ only in
WHICH harm events are allowed to write to the buffer. This isolates the tagging
gate from any quantity effect.

Primary metric:
  contamination_gap = mean harm_eval score on replay harm steps
                    - mean harm_eval score on safe steps
  TAG_GATE_ON:      contamination_gap should be LOW (replay harm not accumulated;
                    harm_eval does not spuriously elevate for replayed content)
  TAG_GATE_ABLATED: contamination_gap should be HIGH (replay harm accumulated;
                    harm_eval treats replayed harm as real -- PTSD contamination)

Secondary metric:
  real_harm_gap = mean harm_eval score on real harm steps
                - mean harm_eval score on safe steps
  Both conditions: real_harm_gap should be HIGH (real harm always accumulates).
  This confirms the tag gate does not suppress real harm signals -- only replayed.

Discriminative prediction:
  delta_contamination = contamination_gap_ABLATED - contamination_gap_ON > 0
  (ablated contaminates more than the tagged condition)

  The real_harm_gap should remain similar across conditions (real harm always
  trains both conditions), confirming the manipulation specifically affects
  replay routing rather than overall harm learning.

Testable criterion structure:
  C1: contamination_gap_ON <= THRESH_C1        (tag gate suppresses replay harm --
                                               contamination low when tag active)
  C2: contamination_gap_ABLATED >= THRESH_C2   (tag loss produces contamination --
                                               replay harm inflates eval baseline)
  C3: delta_contamination >= THRESH_C3 both seeds (discriminative gap exists)
  C4: real_harm_gap_ON >= THRESH_C4 both seeds  (real harm still learned with tag ON)
  C5: n_real_harm_min >= THRESH_N_HARM and n_replay_harm_min >= THRESH_N_HARM
      (sufficient harm events in both channels for valid measurement)

Interpretation:
  C1+C2+C3+C4+C5 PASS: MECH-094 SUPPORTED. Tag gate routes simulation harm
    away from phi(z) write; ablation (tag loss) produces PTSD-like contamination.
  C1+C3 fail, C2 high: tag loss does contaminate but tag gate does not suppress;
    ambiguous -- confound check required.
  C3 fail: no discriminative difference; tag has no effect at V3 proxy scale.
  C4 fail: real harm not learned; training confound -- increase warmup or nav_bias.
  C5 fail: data quality; insufficient harm events.

Seeds: [42, 123] (matched -- same env per seed across conditions)
Env:   CausalGridWorldV2, size=6, 4 hazards, nav_bias=0.45
       N_REPLAY_PER_EP=2 replay episodes per real episode
Warmup: 400 real episodes + 800 replay episodes (interleaved)
Eval:  50 real + 100 replay eval episodes (no training, fixed weights)
Estimated runtime: ~90 min any machine
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_140_mech094_hypothesis_tag_gate_pair"
CLAIM_IDS = ["MECH-094"]

# Pre-registered thresholds
# C1: contamination_gap_ON (replay harm - safe) should be LOW -- tag suppresses replay
THRESH_C1 = 0.02   # contamination_gap_ON <= 0.02 both seeds (near-zero contamination)
# C2: contamination_gap_ABLATED (replay harm - safe) should be HIGH -- tag loss contaminates
THRESH_C2 = 0.04   # contamination_gap_ABLATED >= 0.04 both seeds
# C3: discriminative gap = ABLATED contamination - ON contamination
THRESH_C3 = 0.03   # delta_contamination >= 0.03 both seeds (>=3pp separation)
# C4: real_harm_gap_ON should remain high (real harm still learned)
THRESH_C4 = 0.03   # real_harm_gap_ON >= 0.03 both seeds
# C5: data quality
THRESH_N_HARM = 15  # minimum harm events per channel per seed

# Replay parameters
N_REPLAY_PER_EP = 2   # replay episodes per real episode
REPLAY_NOISE_STD = 0.10  # noise injected into world obs to mark replay content


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
    tag_gate_on: bool,
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
    n_replay_per_ep: int,
    replay_noise_std: float,
    dry_run: bool,
) -> Dict:
    """Run one (seed, condition) cell; return contamination and real-harm gaps.

    TAG_GATE_ON:      replay harm events are tagged (hypothesis_tag=True) and
                      do NOT write to the harm_eval training buffer.
    TAG_GATE_ABLATED: tag absent; replay harm events write to buffer alongside
                      real harm events.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    cond_label = "TAG_GATE_ON" if tag_gate_on else "TAG_GATE_ABLATED"

    # Real environment
    env_real = CausalGridWorldV2(
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
    # Replay environment -- same layout as real, but different seed offset so
    # hazard positions are slightly shifted (simulating internally generated content)
    env_replay = CausalGridWorldV2(
        seed=seed + 10000,
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

    n_actions = env_real.action_dim
    world_obs_dim = env_real.world_obs_dim
    body_obs_dim = env_real.body_obs_dim

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

    # Separate optimizers: standard (E1+E2+E3 non-harm-eval) and harm_eval
    e1_params = list(agent.e1.parameters())
    e2_params = list(agent.e2.parameters())
    e3_params = [
        p for n, p in agent.e3.named_parameters()
        if "harm_eval_head" not in n
    ]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())

    optimizer = optim.Adam(e1_params + e2_params + e3_params, lr=lr)
    harm_eval_optimizer = optim.Adam(harm_eval_params, lr=1e-4)

    # Harm eval buffer: stores (z_world, label) pairs for training
    # Only REAL harm events write to this buffer in TAG_GATE_ON condition
    # Both real and replay harm events write in TAG_GATE_ABLATED condition
    buf_zw: List[torch.Tensor] = []
    buf_labels: List[float] = []

    # Tracking counters
    n_real_harm_train = 0
    n_replay_harm_train = 0
    n_real_harm_buf_writes = 0
    n_replay_harm_buf_writes = 0

    if dry_run:
        actual_warmup = min(3, warmup_episodes)
        actual_eval = min(2, eval_episodes)
    else:
        actual_warmup = warmup_episodes
        actual_eval = eval_episodes

    def _run_episode(env: CausalGridWorldV2, is_replay: bool, train: bool) -> Dict:
        """Run one episode. Returns harm events and z_world snapshots."""
        nonlocal buf_zw, buf_labels
        nonlocal n_real_harm_train, n_replay_harm_train
        nonlocal n_real_harm_buf_writes, n_replay_harm_buf_writes

        _, obs_dict = env.reset()
        agent.reset()

        ep_harm_events = []   # list of (z_world, is_harm)
        ep_real_harm = 0
        ep_replay_harm = 0

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            # Inject noise into world obs for replay to mark as internally generated
            if is_replay and replay_noise_std > 0:
                noise = torch.randn_like(obs_world) * replay_noise_std
                obs_world = obs_world + noise

            with (torch.no_grad() if not train else torch.enable_grad()):
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
            is_harm = float(harm_signal) < 0

            ep_harm_events.append((z_world_curr, is_harm))

            if is_harm:
                if is_replay:
                    ep_replay_harm += 1
                else:
                    ep_real_harm += 1

            # Update harm_eval buffer based on hypothesis_tag gate
            # hypothesis_tag=True (replay) -> gate controls write
            # hypothesis_tag=False (real)   -> always writes
            should_write_to_buf = True
            if is_replay and tag_gate_on:
                should_write_to_buf = False  # tag gate blocks replay from phi(z) write

            if should_write_to_buf:
                buf_zw.append(z_world_curr)
                buf_labels.append(1.0 if is_harm else 0.0)
                if len(buf_zw) > MAX_BUF:
                    buf_zw = buf_zw[-MAX_BUF:]
                    buf_labels = buf_labels[-MAX_BUF:]

                if is_harm:
                    if is_replay:
                        n_replay_harm_buf_writes += 1
                    else:
                        n_real_harm_buf_writes += 1
            elif is_harm and is_replay:
                pass  # tag gate blocked write (counted separately)

            if train:
                # E1 prediction loss update
                try:
                    e1_loss = agent.compute_prediction_loss()
                    if e1_loss.requires_grad:
                        optimizer.zero_grad()
                        e1_loss.backward()
                        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                        optimizer.step()
                except Exception:
                    pass

                # Harm eval training from buffer
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

            if is_replay:
                n_replay_harm_train += ep_replay_harm
            else:
                n_real_harm_train += ep_real_harm

            if done:
                break

        return {
            "harm_events": ep_harm_events,
            "n_real_harm": ep_real_harm,
            "n_replay_harm": ep_replay_harm,
        }

    # --- TRAIN ---
    agent.train()

    for ep in range(actual_warmup):
        # One real episode
        _run_episode(env_real, is_replay=False, train=True)

        # N replay episodes (interleaved)
        for _ in range(n_replay_per_ep):
            _run_episode(env_replay, is_replay=True, train=True)

        if (ep + 1) % 100 == 0 or ep == actual_warmup - 1:
            print(
                f"  [train] seed={seed} cond={cond_label}"
                f" ep {ep+1}/{actual_warmup}"
                f" real_harm_buf={n_real_harm_buf_writes}"
                f" replay_harm_buf={n_replay_harm_buf_writes}",
                flush=True,
            )

    # --- EVAL ---
    agent.eval()

    # Collect harm eval scores on real harm, real safe, replay harm, replay safe
    real_harm_scores: List[float] = []
    real_safe_scores: List[float] = []
    replay_harm_scores: List[float] = []
    replay_safe_scores: List[float] = []
    n_fatal = 0

    def _eval_episode(env: CausalGridWorldV2, is_replay: bool) -> None:
        nonlocal n_fatal
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            if is_replay and replay_noise_std > 0:
                noise = torch.randn_like(obs_world) * replay_noise_std
                obs_world = obs_world + noise

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                z_world_curr = latent.z_world

            action_idx = random.randint(0, n_actions - 1)
            action = _action_to_onehot(action_idx, n_actions, agent.device)
            agent._last_action = action

            _, harm_signal, done, _, obs_dict = env.step(action)
            is_harm = float(harm_signal) < 0

            try:
                with torch.no_grad():
                    score = float(agent.e3.harm_eval(z_world_curr).item())
                if is_replay:
                    if is_harm:
                        replay_harm_scores.append(score)
                    else:
                        replay_safe_scores.append(score)
                else:
                    if is_harm:
                        real_harm_scores.append(score)
                    else:
                        real_safe_scores.append(score)
            except Exception:
                n_fatal += 1

            if done:
                break

    for ep in range(actual_eval):
        _eval_episode(env_real, is_replay=False)
        for _ in range(n_replay_per_ep):
            _eval_episode(env_replay, is_replay=True)

    n_real_harm_eval = len(real_harm_scores)
    n_replay_harm_eval = len(replay_harm_scores)
    n_real_safe_eval = len(real_safe_scores)
    n_replay_safe_eval = len(replay_safe_scores)

    mean_real_harm   = float(sum(real_harm_scores)   / max(1, n_real_harm_eval))
    mean_real_safe   = float(sum(real_safe_scores)   / max(1, n_real_safe_eval))
    mean_replay_harm = float(sum(replay_harm_scores) / max(1, n_replay_harm_eval))
    mean_replay_safe = float(sum(replay_safe_scores) / max(1, n_replay_safe_eval))

    # Core gaps
    real_harm_gap         = mean_real_harm   - mean_real_safe   # should be high in both conditions
    contamination_gap     = mean_replay_harm - mean_replay_safe  # should be LOW if tag ON, HIGH if ablated

    print(
        f"  [eval] seed={seed} cond={cond_label}"
        f" real_harm_gap={real_harm_gap:.4f}"
        f" contamination_gap={contamination_gap:.4f}"
        f" n_real_harm={n_real_harm_eval} n_replay_harm={n_replay_harm_eval}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "tag_gate_on": tag_gate_on,
        "real_harm_gap":         float(real_harm_gap),
        "contamination_gap":     float(contamination_gap),
        "mean_real_harm_score":  float(mean_real_harm),
        "mean_real_safe_score":  float(mean_real_safe),
        "mean_replay_harm_score": float(mean_replay_harm),
        "mean_replay_safe_score": float(mean_replay_safe),
        "n_real_harm_eval":      int(n_real_harm_eval),
        "n_replay_harm_eval":    int(n_replay_harm_eval),
        "n_real_safe_eval":      int(n_real_safe_eval),
        "n_replay_safe_eval":    int(n_replay_safe_eval),
        "n_real_harm_buf_writes":   int(n_real_harm_buf_writes),
        "n_replay_harm_buf_writes": int(n_replay_harm_buf_writes),
        "n_fatal": int(n_fatal),
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
    nav_bias: float = 0.45,
    n_replay_per_ep: int = N_REPLAY_PER_EP,
    replay_noise_std: float = REPLAY_NOISE_STD,
    dry_run: bool = False,
    **kwargs,
) -> dict:
    """Discriminative pair: TAG_GATE_ON (hypothesis tag blocks replay harm from
    phi(z) write -- MECH-094 architecture) vs TAG_GATE_ABLATED (tag absent; replay
    harm accumulates alongside real harm -- PTSD contamination mechanism).
    Tests MECH-094: hypothesis tag gates simulation harm away from the residue
    write pathway; tag loss produces spurious harm-eval contamination from replayed
    content.
    """
    results_on:      List[Dict] = []
    results_ablated: List[Dict] = []

    for seed in seeds:
        for gate_on in [True, False]:
            label = "TAG_GATE_ON" if gate_on else "TAG_GATE_ABLATED"
            print(
                f"\n[V3-EXQ-140] {label} seed={seed}"
                f" warmup={warmup_episodes} eval={eval_episodes}"
                f" nav_bias={nav_bias} n_replay_per_ep={n_replay_per_ep}"
                f" replay_noise_std={replay_noise_std}",
                flush=True,
            )
            r = _run_single(
                seed=seed,
                tag_gate_on=gate_on,
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
                n_replay_per_ep=n_replay_per_ep,
                replay_noise_std=replay_noise_std,
                dry_run=dry_run,
            )
            if gate_on:
                results_on.append(r)
            else:
                results_ablated.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    contam_gap_on      = _avg(results_on,      "contamination_gap")
    contam_gap_ablated = _avg(results_ablated, "contamination_gap")
    real_harm_gap_on   = _avg(results_on,      "real_harm_gap")
    real_harm_gap_abl  = _avg(results_ablated, "real_harm_gap")

    # Per-seed delta_contamination (ABLATED - ON): should be >= THRESH_C3 both seeds
    per_seed_delta_contam: List[float] = []
    for r_on in results_on:
        matching = [r for r in results_ablated if r["seed"] == r_on["seed"]]
        if matching:
            per_seed_delta_contam.append(
                matching[0]["contamination_gap"] - r_on["contamination_gap"]
            )

    n_real_harm_min   = min(r["n_real_harm_eval"]   for r in results_on + results_ablated)
    n_replay_harm_min = min(r["n_replay_harm_eval"]  for r in results_on + results_ablated)

    # Pre-registered PASS criteria
    # C1: contamination_gap_ON <= THRESH_C1 in ALL seeds (tag gate suppresses replay)
    c1_pass = all(r["contamination_gap"] <= THRESH_C1 for r in results_on)
    # C2: contamination_gap_ABLATED >= THRESH_C2 in ALL seeds (tag loss contaminates)
    c2_pass = all(r["contamination_gap"] >= THRESH_C2 for r in results_ablated)
    # C3: per-seed delta_contamination (ABLATED - ON) >= THRESH_C3 in ALL seeds
    c3_pass = len(per_seed_delta_contam) > 0 and all(
        d >= THRESH_C3 for d in per_seed_delta_contam
    )
    # C4: real_harm_gap_ON >= THRESH_C4 in ALL seeds (real harm still learned with tag ON)
    c4_pass = all(r["real_harm_gap"] >= THRESH_C4 for r in results_on)
    # C5: data quality -- enough harm events in both channels
    c5_pass = n_real_harm_min >= THRESH_N_HARM and n_replay_harm_min >= THRESH_N_HARM

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])
    status = "PASS" if all_pass else "FAIL"

    if all_pass:
        decision = "retain_ree"
    elif c3_pass and c4_pass:
        decision = "hybridize"   # directional but thresholds marginal
    elif not c3_pass:
        decision = "retire_ree_claim"
    else:
        decision = "hybridize"

    print(f"\n[V3-EXQ-140] Results:", flush=True)
    print(
        f"  contam_gap_ON={contam_gap_on:.4f}"
        f" contam_gap_ABLATED={contam_gap_ablated:.4f}"
        f" real_harm_gap_ON={real_harm_gap_on:.4f}"
        f" real_harm_gap_ABLATED={real_harm_gap_abl:.4f}",
        flush=True,
    )
    print(
        f"  per_seed_delta_contam={[round(d, 4) for d in per_seed_delta_contam]}"
        f" n_real_harm_min={n_real_harm_min} n_replay_harm_min={n_replay_harm_min}"
        f" decision={decision}  status={status} ({criteria_met}/5)",
        flush=True,
    )

    failure_notes: List[str] = []
    if not c1_pass:
        failing = [r["seed"] for r in results_on if r["contamination_gap"] > THRESH_C1]
        failure_notes.append(
            f"C1 FAIL: TAG_GATE_ON contamination_gap > {THRESH_C1} in seeds {failing}"
            " -- tag gate does not suppress replay harm from harm_eval"
        )
    if not c2_pass:
        failing = [r["seed"] for r in results_ablated if r["contamination_gap"] < THRESH_C2]
        failure_notes.append(
            f"C2 FAIL: TAG_GATE_ABLATED contamination_gap < {THRESH_C2} in seeds {failing}"
            " -- tag loss does not produce measurable contamination; check replay harm volume"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: per-seed delta_contamination {[round(d, 4) for d in per_seed_delta_contam]}"
            f" < {THRESH_C3}"
            " -- no discriminative contamination difference; tag gate has no effect at V3 proxy scale"
        )
    if not c4_pass:
        failing = [r["seed"] for r in results_on if r["real_harm_gap"] < THRESH_C4]
        failure_notes.append(
            f"C4 FAIL: real_harm_gap_ON < {THRESH_C4} in seeds {failing}"
            " -- real harm not learned even with tag ON; increase nav_bias or warmup"
        )
    if not c5_pass:
        failure_notes.append(
            f"C5 FAIL: n_real_harm_min={n_real_harm_min} or"
            f" n_replay_harm_min={n_replay_harm_min} < {THRESH_N_HARM}"
            " -- insufficient harm events for valid measurement; increase nav_bias or eval episodes"
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    if all_pass:
        interpretation = (
            "MECH-094 SUPPORTED: hypothesis tag write gate successfully separates"
            f" simulation from committed residue updates."
            f" TAG_GATE_ON contamination_gap={contam_gap_on:.4f} (low: replay harm"
            f" not accumulated in phi(z)) vs TAG_GATE_ABLATED contamination_gap="
            f"{contam_gap_ablated:.4f} (high: replay harm inflates harm_eval baseline"
            f" -- PTSD-like contamination mechanism)."
            f" Real harm gap preserved in TAG_GATE_ON: real_harm_gap_ON={real_harm_gap_on:.4f}"
            f" (real harm still trains harm_eval; gate only blocks replay routing)."
            f" delta_contamination={[round(d,4) for d in per_seed_delta_contam]}"
            f" across {len(seeds)} seeds. Supports MECH-094: categorical tag routes"
            f" simulation harm away from phi(z) write; tag loss produces spurious"
            f" contamination matching PTSD flashback mechanism."
        )
    elif c3_pass and c4_pass:
        interpretation = (
            f"Partial support: discriminative contamination gap"
            f" {[round(d,4) for d in per_seed_delta_contam]} (C3 PASS),"
            f" real_harm_gap preserved (C4 PASS), but marginal threshold crossings"
            f" on C1 or C2. Tag gate directionally correct but threshold margins narrow."
            f" Consider increasing replay volume or adjusting noise_std."
        )
    else:
        interpretation = (
            f"MECH-094 NOT supported at V3 proxy level:"
            f" contam_gap_ON={contam_gap_on:.4f} (C1 {'PASS' if c1_pass else 'FAIL'}),"
            f" contam_gap_ABLATED={contam_gap_ablated:.4f} (C2 {'PASS' if c2_pass else 'FAIL'}),"
            f" delta_contamination={[round(d,4) for d in per_seed_delta_contam]}"
            f" (C3 {'PASS' if c3_pass else 'FAIL'})."
            f" The V3 proxy (noisy replay in offset-seed env) may not generate"
            f" sufficiently distinct replay harm signal to test the write-gate"
            f" mechanism. Consider: (a) sharper replay/real contrast"
            f" (higher replay_noise_std or structurally different replay env);"
            f" (b) more episodes; (c) checking whether harm_eval baseline is"
            f" dominated by training data volume rather than routing source."
        )

    per_on_rows = "\n".join(
        f"  seed={r['seed']}: contam_gap={r['contamination_gap']:.4f}"
        f" real_harm_gap={r['real_harm_gap']:.4f}"
        f" n_real_harm={r['n_real_harm_eval']}"
        f" n_replay_harm={r['n_replay_harm_eval']}"
        for r in results_on
    )
    per_ablated_rows = "\n".join(
        f"  seed={r['seed']}: contam_gap={r['contamination_gap']:.4f}"
        f" real_harm_gap={r['real_harm_gap']:.4f}"
        f" n_real_harm={r['n_real_harm_eval']}"
        f" n_replay_harm={r['n_replay_harm_eval']}"
        for r in results_ablated
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-140 -- MECH-094 Hypothesis Tag Write Gate Discriminative Pair\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-094\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**Conditions:** TAG_GATE_ON vs TAG_GATE_ABLATED\n"
        f"**Warmup:** {warmup_episodes} real eps + {warmup_episodes * n_replay_per_ep}"
        f" replay eps  **Eval:** {eval_episodes} real + {eval_episodes * n_replay_per_ep}"
        f" replay eps\n"
        f"**Env:** CausalGridWorldV2 size=6, 4 hazards, nav_bias={nav_bias}\n"
        f"**Replay:** n_replay_per_ep={n_replay_per_ep}, noise_std={replay_noise_std}"
        f" (offset-seed env)\n\n"
        f"## Design\n\n"
        f"MECH-094 asserts the hypothesis tag is a categorical write gate separating"
        f" simulation from committed residue updates. The experiment controls whether"
        f" replay harm events are allowed to train the harm_eval head (proxy for phi(z)"
        f" write pathway).\n\n"
        f"TAG_GATE_ON: replay harm is tagged (hypothesis_tag=True);"
        f" only real harm trains harm_eval buffer.\n\n"
        f"TAG_GATE_ABLATED: tag absent; replay harm accumulates alongside real harm"
        f" (PTSD contamination mechanism: replayed content writes as if real).\n\n"
        f"Key metric: contamination_gap = harm_eval_score(replay harm)"
        f" - harm_eval_score(replay safe).\n"
        f"Low contamination_gap: tag suppresses replay routing (MECH-094 ON).\n"
        f"High contamination_gap: tag loss allows replay routing (ablation = PTSD model).\n\n"
        f"Secondary: real_harm_gap = harm_eval_score(real harm) - harm_eval_score(real safe)."
        f" Should be high in both conditions (real harm always trains both).\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: contamination_gap_ON <= {THRESH_C1} both seeds"
        f"  (tag gate suppresses replay; near-zero contamination)\n"
        f"C2: contamination_gap_ABLATED >= {THRESH_C2} both seeds"
        f"  (tag loss produces contamination)\n"
        f"C3: per-seed delta_contamination (ABLATED-ON) >= {THRESH_C3} both seeds"
        f"  (discriminative gap)\n"
        f"C4: real_harm_gap_ON >= {THRESH_C4} both seeds"
        f"  (real harm still learned with tag ON)\n"
        f"C5: n_real_harm_min >= {THRESH_N_HARM} and n_replay_harm_min >= {THRESH_N_HARM}"
        f"  (data quality)\n\n"
        f"## Results\n\n"
        f"| Condition | contam_gap | real_harm_gap | n_real_harm | n_replay_harm |\n"
        f"|-----------|------------|---------------|-------------|---------------|\n"
        f"| TAG_GATE_ON      | {contam_gap_on:.4f}"
        f" | {real_harm_gap_on:.4f}"
        f" | {n_real_harm_min}"
        f" | {n_replay_harm_min} |\n"
        f"| TAG_GATE_ABLATED | {contam_gap_ablated:.4f}"
        f" | {real_harm_gap_abl:.4f}"
        f" | {n_real_harm_min}"
        f" | {n_replay_harm_min} |\n\n"
        f"**delta_contamination per seed: {[round(d,4) for d in per_seed_delta_contam]}**\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: contam_gap_ON <= {THRESH_C1} (all seeds)"
        f" | {'PASS' if c1_pass else 'FAIL'}"
        f" | {contam_gap_on:.4f} |\n"
        f"| C2: contam_gap_ABLATED >= {THRESH_C2} (all seeds)"
        f" | {'PASS' if c2_pass else 'FAIL'}"
        f" | {contam_gap_ablated:.4f} |\n"
        f"| C3: per-seed delta >= {THRESH_C3} (all seeds)"
        f" | {'PASS' if c3_pass else 'FAIL'}"
        f" | {[round(d,4) for d in per_seed_delta_contam]} |\n"
        f"| C4: real_harm_gap_ON >= {THRESH_C4} (all seeds)"
        f" | {'PASS' if c4_pass else 'FAIL'}"
        f" | {real_harm_gap_on:.4f} |\n"
        f"| C5: n_harm_min >= {THRESH_N_HARM}"
        f" | {'PASS' if c5_pass else 'FAIL'}"
        f" | real={n_real_harm_min} replay={n_replay_harm_min} |\n\n"
        f"Criteria met: {criteria_met}/5 -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n\n"
        f"## Per-Seed\n\n"
        f"TAG_GATE_ON:\n{per_on_rows}\n\n"
        f"TAG_GATE_ABLATED:\n{per_ablated_rows}\n"
        f"{failure_section}\n"
    )

    metrics = {
        "contamination_gap_on":       float(contam_gap_on),
        "contamination_gap_ablated":  float(contam_gap_ablated),
        "real_harm_gap_on":           float(real_harm_gap_on),
        "real_harm_gap_ablated":      float(real_harm_gap_abl),
        "delta_contamination_min":    float(min(per_seed_delta_contam)) if per_seed_delta_contam else 0.0,
        "delta_contamination_max":    float(max(per_seed_delta_contam)) if per_seed_delta_contam else 0.0,
        "n_real_harm_eval_min":       float(n_real_harm_min),
        "n_replay_harm_eval_min":     float(n_replay_harm_min),
        "n_seeds":                    float(len(seeds)),
        "nav_bias":                   float(nav_bias),
        "alpha_world":                float(alpha_world),
        "n_replay_per_ep":            float(n_replay_per_ep),
        "replay_noise_std":           float(replay_noise_std),
        "crit1_pass":                 1.0 if c1_pass else 0.0,
        "crit2_pass":                 1.0 if c2_pass else 0.0,
        "crit3_pass":                 1.0 if c3_pass else 0.0,
        "crit4_pass":                 1.0 if c4_pass else 0.0,
        "crit5_pass":                 1.0 if c5_pass else 0.0,
        "criteria_met":               float(criteria_met),
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
    parser.add_argument("--seeds",           type=int, nargs="+", default=[42, 123])
    parser.add_argument("--warmup",          type=int,   default=400)
    parser.add_argument("--eval-eps",        type=int,   default=50)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--alpha-self",      type=float, default=0.3)
    parser.add_argument("--harm-scale",      type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.05)
    parser.add_argument("--nav-bias",        type=float, default=0.45)
    parser.add_argument("--n-replay-per-ep", type=int,   default=N_REPLAY_PER_EP)
    parser.add_argument("--replay-noise-std", type=float, default=REPLAY_NOISE_STD)
    parser.add_argument("--dry-run",         action="store_true",
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
        n_replay_per_ep=args.n_replay_per_ep,
        replay_noise_std=args.replay_noise_std,
        dry_run=args.dry_run,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["registered_thresholds"] = {
        "C1_contamination_gap_on":        THRESH_C1,
        "C2_contamination_gap_ablated":   THRESH_C2,
        "C3_per_seed_delta_contamination": THRESH_C3,
        "C4_real_harm_gap_on":            THRESH_C4,
        "C5_n_harm_min":                  THRESH_N_HARM,
    }
    result["seeds"] = list(args.seeds)
    result["conditions"] = ["TAG_GATE_ON", "TAG_GATE_ABLATED"]
    result["dispatch_mode"] = "discriminative_pair"
    result["backlog_id"] = "EVB-0060"
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
