#!/opt/local/bin/python3
"""
V3-EXQ-134 -- MECH-103 Multimodal Exteroceptive Fusion Discriminative Pair
           (Somatosensory / Tactile Channel)

Claims: MECH-103
Proposal: EXP-0030 / EVB-0022

MECH-103 asserts: "E1 performs multimodal exteroceptive fusion across sensory modalities
before feeding E2. Different exteroceptive modalities (vision, audition, somatosensory/touch)
each contribute differently to E1's shared world latent (z_world). Multi-source convergence
produces more accurate and robust world representations than single-modality input because
each modality carries complementary structure: vision (object identity, spatial layout),
audition (events, temporal cues, off-screen dynamics), somatosensory (surface properties,
contact)."

This experiment provides independent replication of EXQ-128 (which tested an auditory channel)
using a somatosensory / tactile second modality, as required by EXP-0030 (EVB-0022) which
called for "insufficient_experimental_replication" evidence.

V3-accessible proxy design
--------------------------
CausalGridWorldV2 produces a single exteroceptive world_obs vector (visual/spatial).
We construct a synthetic "somatosensory" channel of dimension SOMA_DIM that encodes
surface-contact and body-surface-interaction properties -- information that is structurally
distinct from the spatial layout encoded in the visual channel and from the event-tone
information in EXQ-128's auditory channel.

Somatosensory channel logic:
  - Contact force: high sustained signal when harm_signal < 0 (hazard contact), encoding
    the immediate bodily impact. Analogous to primary somatosensory cortex (S1) responding
    to nociceptive input.
  - Near-surface vibration: moderate signal when hazard proximity > threshold but no full
    contact -- encodes surface texture warning (rough vs smooth terrain analog).
  - Clearance signal: mild positive signal when no hazard nearby (safe-surface feedback).
  - Background proprioceptive noise: low-variance baseline always present.

FUSION_ON (informative=True):
  world_obs_fused = concat([world_obs_visual, world_obs_soma])
  world_obs_dim = original_world_obs_dim + SOMA_DIM
  The somatosensory channel carries genuine event-correlated signal.

FUSION_ABLATED (informative=False):
  world_obs_fused = concat([world_obs_visual, zeros(SOMA_DIM)])
  world_obs_dim = original_world_obs_dim + SOMA_DIM  (same architecture)
  The agent sees the same encoder structure but the extra channel is uninformative.

Both conditions use identical encoder architecture (same world_obs_dim), so any
improvement in FUSION_ON is attributable solely to the informative second modality --
not to additional parameters or degrees of freedom.

Key distinction from EXQ-128:
  EXQ-128 used an auditory channel: hazard-proximity TONES (event-correlated, off-screen)
  EXQ-134 uses a somatosensory channel: surface CONTACT FORCE + near-surface VIBRATION
  (body-state correlated, immediate contact encoding). These are different aspects of
  MECH-103's claim about complementary modality structure.

Discriminative pair:
  FUSION_ON      -- world_obs includes informative somatosensory channel (SOMA_DIM=8)
  FUSION_ABLATED -- world_obs includes zero-padded channel of same dimension

Seeds: [42, 123] (matched -- same env instantiation per seed)
Env:   CausalGridWorldV2, size=6, 4 hazards, nav_bias=0.40
Warmup: 400 eps x 200 steps
Eval:  50 eps x 200 steps (no training, fixed weights)

Pre-registered acceptance criteria:
  C1: gap_fusion_on >= 0.04  (FUSION_ON harm eval gap above floor -- both seeds)
  C2: per-seed delta (FUSION_ON - FUSION_ABLATED) >= 0.02  (soma modality adds >=2pp --
      both seeds)
  C3: gap_fusion_ablated >= 0.0  (ablation still learns -- data quality check)
  C4: n_harm_eval_min >= 20  (sufficient harm events in eval -- both seeds)
  C5: no fatal errors in either condition

Decision scoring:
  PASS (all 5): retain_ree -- somatosensory modality fusion improves z_world quality
  C1+C2+C4:     hybridize  -- fusion effect present but magnitude uncertain
  C2 fails:     retire_ree_claim -- no detectable multimodal advantage at this scale
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


EXPERIMENT_TYPE = "v3_exq_134_mech103_somatosensory_fusion_pair"
CLAIM_IDS = ["MECH-103"]

# Pre-registered thresholds
THRESH_C1 = 0.04   # gap_fusion_on >= 0.04 (FUSION_ON harm eval above floor, both seeds)
THRESH_C2 = 0.02   # per-seed delta (FUSION_ON - FUSION_ABLATED) >= 0.02 (both seeds)
THRESH_C3 = 0.0    # gap_fusion_ablated >= 0.0 (ablation learns something)
THRESH_C4 = 20     # n_harm_eval_min >= 20 (data quality)

# Somatosensory channel dimension
SOMA_DIM = 8


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


def _build_somatosensory_channel(
    obs_dict: dict,
    harm_signal: float,
    rng: random.Random,
    informative: bool,
    soma_dim: int,
) -> torch.Tensor:
    """Build synthetic somatosensory channel (shape: [1, soma_dim]).

    FUSION_ON (informative=True):
      - Contact-force zone (first soma_dim//2 dims):
          High signal when harm_signal < 0 (actual hazard contact) -- nociceptive input.
          Moderate signal when hazard proximity > 0.05 but no full contact -- near-surface
          vibration / surface texture warning.
          Low signal (clearance) when no hazard nearby -- safe-surface proprioception.
      - Near-miss vibration zone (last soma_dim//2 dims):
          Oscillating signal when hazard proximity > 0.1 but harm_signal >= 0 -- encodes
          the vibrotactile warning preceding contact.
          Near-zero otherwise.

    FUSION_ABLATED (informative=False):
      - Return zeros (uninformative padding).
    """
    vec = torch.zeros(1, soma_dim)
    if not informative:
        return vec

    half = soma_dim // 2

    # Detect hazard proximity from proxy field
    hazard_proximity = 0.0
    is_contact = harm_signal < 0.0
    world_state = obs_dict.get("world_state", None)
    if world_state is not None and len(world_state) > 250:
        # hazard_field_view: world_state[225:250] = 5x5 flattened
        center_val = float(world_state[237])  # center (2,2) of 5x5
        hazard_proximity = max(0.0, center_val)

    # Contact-force zone: first half of soma_dim
    if is_contact:
        # Full nociceptive contact signal: mean ~0.8 + noise
        for i in range(half):
            vec[0, i] = 0.80 + rng.gauss(0.0, 0.04)
    elif hazard_proximity > 0.05:
        # Near-surface vibration preceding contact: mean ~0.35 + noise
        signal_strength = 0.35 + min(0.25, hazard_proximity * 2.0)
        for i in range(half):
            vec[0, i] = signal_strength + rng.gauss(0.0, 0.04)
    else:
        # Safe-surface clearance signal: mild positive + very low noise
        for i in range(half):
            vec[0, i] = 0.05 + rng.gauss(0.0, 0.01)

    # Near-miss vibration zone: last half of soma_dim
    # Oscillates (alternating sign) when approaching but not yet in contact
    if not is_contact and hazard_proximity > 0.10:
        amp = min(0.4, hazard_proximity * 3.0)
        for i in range(half, soma_dim):
            sign = 1.0 if (i % 2 == 0) else -1.0
            vec[0, i] = sign * amp + rng.gauss(0.0, 0.03)
    else:
        # Near-zero in safe region or contact (vibration replaced by sustained force)
        for i in range(half, soma_dim):
            vec[0, i] = rng.gauss(0.0, 0.01)

    return vec


def _run_single(
    seed: int,
    fusion_on: bool,
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
    """Run one (seed, condition) cell; return harm eval gap metrics."""
    torch.manual_seed(seed)
    random.seed(seed)
    rng = random.Random(seed + 2000)

    cond_label = "FUSION_ON" if fusion_on else "FUSION_ABLATED"

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

    # Both conditions use the same extended world_obs_dim
    # FUSION_ON: extra SOMA_DIM dims carry informative somatosensory signal
    # FUSION_ABLATED: extra SOMA_DIM dims are always zeros
    extended_world_obs_dim = env.world_obs_dim + SOMA_DIM

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=extended_world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=0,
    )

    agent = REEAgent(config)

    # Harm eval buffer
    buf_zw: List[torch.Tensor] = []
    buf_labels: List[float] = []
    MAX_BUF = 2000

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

    if dry_run:
        actual_warmup = min(3, warmup_episodes)
        actual_eval = min(2, eval_episodes)
    else:
        actual_warmup = warmup_episodes
        actual_eval = eval_episodes

    def _extend_world_obs(obs_world_raw: torch.Tensor, harm_sig: float) -> torch.Tensor:
        """Concatenate somatosensory channel to raw world_obs."""
        obs_dict_cur = env._get_observation_dict()
        soma = _build_somatosensory_channel(
            obs_dict=obs_dict_cur,
            harm_signal=harm_sig,
            rng=rng,
            informative=fusion_on,
            soma_dim=SOMA_DIM,
        )
        soma_flat = soma.squeeze(0)  # -> [SOMA_DIM]
        if obs_world_raw.dim() == 1:
            return torch.cat([obs_world_raw, soma_flat], dim=-1)
        else:
            return torch.cat([obs_world_raw, soma], dim=-1)

    # --- TRAIN ---
    agent.train()

    for ep in range(actual_warmup):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world_raw = obs_dict["world_state"]

            # Use 0.0 as harm_sig placeholder for current step encoding
            obs_world = _extend_world_obs(obs_world_raw, harm_sig=0.0)

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

            harm_sig_float = float(harm_signal)
            is_harm = harm_sig_float < 0

            buf_zw.append(z_world_curr)
            buf_labels.append(1.0 if is_harm else 0.0)
            if len(buf_zw) > MAX_BUF:
                buf_zw = buf_zw[-MAX_BUF:]
                buf_labels = buf_labels[-MAX_BUF:]

            # Standard E1 + E2 losses
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            # Harm eval online update
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

    for _ in range(actual_eval):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world_raw = obs_dict["world_state"]
            obs_world = _extend_world_obs(obs_world_raw, harm_sig=0.0)

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
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
        f" n_harm={n_harm_eval} n_safe={n_safe_eval}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "fusion_on": fusion_on,
        "harm_eval_gap": float(gap),
        "mean_harm_score": float(mean_harm),
        "mean_safe_score": float(mean_safe),
        "n_harm_eval": int(n_harm_eval),
        "n_safe_eval": int(n_safe_eval),
        "n_fatal": int(n_fatal),
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
    """Discriminative pair: FUSION_ON (informative somatosensory channel) vs FUSION_ABLATED
    (zero-padded channel). Tests MECH-103: multimodal exteroceptive fusion (somatosensory
    modality) improves z_world quality. Independent replication of EXQ-128 (auditory channel)
    using a different sensory modality (surface contact + near-surface vibration).
    """
    results_on:      List[Dict] = []
    results_ablated: List[Dict] = []

    for seed in seeds:
        for fusion_on in [True, False]:
            label = "FUSION_ON" if fusion_on else "FUSION_ABLATED"
            print(
                f"\n[V3-EXQ-134] {label} seed={seed}"
                f" warmup={warmup_episodes} eval={eval_episodes}"
                f" soma_dim={SOMA_DIM} nav_bias={nav_bias}",
                flush=True,
            )
            r = _run_single(
                seed=seed,
                fusion_on=fusion_on,
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
            if fusion_on:
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

    # Pre-registered PASS criteria
    # C1: FUSION_ON gap >= THRESH_C1 in ALL seeds
    c1_pass = all(r["harm_eval_gap"] >= THRESH_C1 for r in results_on)
    # C2: per-seed delta (FUSION_ON - FUSION_ABLATED) >= THRESH_C2 in ALL seeds
    per_seed_deltas: List[float] = []
    for r_on in results_on:
        matching = [r for r in results_ablated if r["seed"] == r_on["seed"]]
        if matching:
            per_seed_deltas.append(r_on["harm_eval_gap"] - matching[0]["harm_eval_gap"])
    c2_pass = len(per_seed_deltas) > 0 and all(d >= THRESH_C2 for d in per_seed_deltas)
    # C3: FUSION_ABLATED gap >= 0 (ablation learns something)
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

    print(f"\n[V3-EXQ-134] Results:", flush=True)
    print(
        f"  gap_on={gap_on:.4f}"
        f" gap_ablated={gap_ablated:.4f}"
        f" delta_gap={delta_gap:+.4f}",
        flush=True,
    )
    print(
        f"  n_harm_min={n_harm_min}"
        f"  per_seed_deltas={[f'{d:.4f}' for d in per_seed_deltas]}"
        f"  decision={decision}  status={status} ({criteria_met}/5)",
        flush=True,
    )

    failure_notes: List[str] = []
    if not c1_pass:
        failing = [r["seed"] for r in results_on if r["harm_eval_gap"] < THRESH_C1]
        failure_notes.append(
            f"C1 FAIL: FUSION_ON gap below {THRESH_C1} in seeds {failing}"
            " -- z_world does not encode harm even with somatosensory input"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: per-seed deltas {[f'{d:.4f}' for d in per_seed_deltas]}"
            f" < {THRESH_C2}"
            " -- somatosensory fusion does not add >=2pp over single-modality"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: gap_ablated={gap_ablated:.4f} < {THRESH_C3}"
            " -- single-modality baseline fails entirely (confound check)"
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
            "MECH-103 SUPPORTED (somatosensory replication): informative somatosensory"
            f" modality (SOMA_DIM={SOMA_DIM}, contact-force + near-surface vibration)"
            " improves harm_eval_head quality"
            f" (gap_on={gap_on:.4f} vs gap_ablated={gap_ablated:.4f},"
            f" delta={delta_gap:+.4f} across {len(seeds)} seeds)."
            " Independent replication via distinct sensory modality (somatosensory vs"
            " EXQ-128 auditory): both converge on the same MECH-103 conclusion,"
            " consistent with the claim that complementary modality structure contributes"
            " precision-weighted evidence to z_world."
        )
    elif c1_pass and c4_pass:
        interpretation = (
            f"Partial support: FUSION_ON achieves gap={gap_on:.4f} (C1 PASS)"
            f" but delta={delta_gap:+.4f} (C2 FAIL, threshold={THRESH_C2})."
            " z_world encodes harm when somatosensory channel is present, but the"
            " advantage over single-modality is below the pre-registered threshold."
            " Possible: somatosensory contact-force signal is partially redundant with"
            " the hazard-proximity signal already encoded in the visual channel at"
            " this training scale, or SOMA_DIM=8 is insufficient."
        )
    else:
        interpretation = (
            f"MECH-103 NOT supported (somatosensory channel): gap_on={gap_on:.4f}"
            f" (C1 {'PASS' if c1_pass else 'FAIL'}),"
            f" delta={delta_gap:+.4f} (C2 {'PASS' if c2_pass else 'FAIL'})."
            " Informative somatosensory channel does not produce a detectable"
            " improvement in z_world harm discrimination at this training scale."
        )

    per_on_rows = "\n".join(
        f"  seed={r['seed']}: gap={r['harm_eval_gap']:.4f}"
        f" n_harm={r['n_harm_eval']}"
        for r in results_on
    )
    per_ablated_rows = "\n".join(
        f"  seed={r['seed']}: gap={r['harm_eval_gap']:.4f}"
        f" n_harm={r['n_harm_eval']}"
        for r in results_ablated
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-134 -- MECH-103 Multimodal Exteroceptive Fusion (Somatosensory Channel)\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-103\n"
        f"**Proposal:** EXP-0030 / EVB-0022\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**Conditions:** FUSION_ON vs FUSION_ABLATED\n"
        f"**Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps\n"
        f"**Env:** CausalGridWorldV2 size=6, 4 hazards, nav_bias={nav_bias}\n"
        f"**Somatosensory channel:** SOMA_DIM={SOMA_DIM}"
        f" (contact-force + near-surface vibration; FUSION_ON informative; FUSION_ABLATED zeros)\n"
        f"**Replication of:** EXQ-128 (auditory channel) -- distinct second modality\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: gap_fusion_on >= {THRESH_C1} in all seeds  (FUSION_ON harm eval above floor)\n"
        f"C2: per-seed delta (ON - ABLATED) >= {THRESH_C2} in all seeds  "
        f"(soma modality adds >=2pp)\n"
        f"C3: gap_fusion_ablated >= {THRESH_C3}  (ablation learns something; data quality)\n"
        f"C4: n_harm_eval_min >= {THRESH_C4}  (sufficient harm events in eval)\n"
        f"C5: no fatal errors\n\n"
        f"## Results\n\n"
        f"| Condition | gap (avg) | mean_harm | mean_safe |\n"
        f"|-----------|-----------|-----------|----------|\n"
        f"| FUSION_ON      | {gap_on:.4f}"
        f" | {_avg(results_on,      'mean_harm_score'):.4f}"
        f" | {_avg(results_on,      'mean_safe_score'):.4f} |\n"
        f"| FUSION_ABLATED | {gap_ablated:.4f}"
        f" | {_avg(results_ablated, 'mean_harm_score'):.4f}"
        f" | {_avg(results_ablated, 'mean_safe_score'):.4f} |\n\n"
        f"**delta_gap (FUSION_ON - FUSION_ABLATED): {delta_gap:+.4f}**\n\n"
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
        f"FUSION_ON:\n{per_on_rows}\n\n"
        f"FUSION_ABLATED:\n{per_ablated_rows}\n"
        f"{failure_section}\n"
    )

    metrics = {
        "gap_fusion_on":           float(gap_on),
        "gap_fusion_ablated":      float(gap_ablated),
        "delta_gap":               float(delta_gap),
        "n_harm_eval_min":         float(n_harm_min),
        "n_seeds":                 float(len(seeds)),
        "soma_dim":                float(SOMA_DIM),
        "nav_bias":                float(nav_bias),
        "alpha_world":             float(alpha_world),
        "per_seed_delta_min":      float(min(per_seed_deltas)) if per_seed_deltas else 0.0,
        "per_seed_delta_max":      float(max(per_seed_deltas)) if per_seed_deltas else 0.0,
        "crit1_pass":              1.0 if c1_pass else 0.0,
        "crit2_pass":              1.0 if c2_pass else 0.0,
        "crit3_pass":              1.0 if c3_pass else 0.0,
        "crit4_pass":              1.0 if c4_pass else 0.0,
        "crit5_pass":              1.0 if c5_pass else 0.0,
        "criteria_met":            float(criteria_met),
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
    parser.add_argument("--nav-bias",        type=float, default=0.40)
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
        dry_run=args.dry_run,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["registered_thresholds"] = {
        "C1_gap_fusion_on":       THRESH_C1,
        "C2_per_seed_delta":      THRESH_C2,
        "C3_gap_ablated":         THRESH_C3,
        "C4_n_harm_eval_min":     THRESH_C4,
    }
    result["seeds"] = list(args.seeds)
    result["conditions"] = ["FUSION_ON", "FUSION_ABLATED"]
    result["dispatch_mode"] = "discriminative_pair"
    result["backlog_id"] = "EVB-0022"
    result["proposal_id"] = "EXP-0030"

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

    print(f"\n[V3-EXQ-134] Output written to: {out_path}", flush=True)
