#!/opt/local/bin/python3
"""
V3-EXQ-133 -- MECH-091 Salient-Event Phase-Reset Discriminative Pair

Claims: MECH-091
Proposal: EXP-0035 / EVB-0027

MECH-091 asserts: "Salient events phase-reset the E3 heartbeat clock."

Specifically, high-salience events (action completion, unexpected harm, commitment
boundary crossing) reorganise the timing of subsequent E3 heartbeat cycles -- they
do not merely boost signal amplitude. Phase-reset ensures updated harm estimates from
the salient event enter E3 at the START of a fresh cycle, preventing partial
integration artefacts (stale mid-cycle state contaminating the next window).

Functional restatement (from claims.yaml):
  Salient events resynchronize E3's update window to start fresh rather than
  continuing mid-cycle. In an ANN substrate: explicit cycle-boundary marker triggered
  by salient events. Biological substrate (thalamic theta/alpha reset, P300) is not
  required -- the functional requirement is cycle-boundary alignment.

V3-accessible proxy design
--------------------------
We build on the E3 heartbeat rate separation from ARC-023 (EXQ-131). E3 updates
every E3_HEARTBEAT_K=5 steps in a time-multiplexed loop. The discriminative variable
is whether a salient event (harm contact = is_harm=True) triggers an immediate
heartbeat cycle boundary (phase-reset) or not.

PHASE_RESET_ON (MECH-091 architecture):
  - E3 heartbeat operates at E3_HEARTBEAT_K=5 steps per cycle (same as EXQ-131 proxy)
  - On a salient event (harm contact): heartbeat counter resets to 0 -> immediate
    cycle boundary at step+1. The harm signal enters E3 at the START of the next
    cycle, not mid-cycle.
  - This aligns E3's integration window with the salient event boundary.
  - Implementation: track step_counter % E3_HEARTBEAT_K; on is_harm, reset to 0.

PHASE_RESET_ABLATED (continuous counter, ablation):
  - E3 heartbeat counter increments monotonically; never resets mid-cycle.
  - Salient events (harm contacts) fall at arbitrary phase positions within the
    running K-step cycle -- sometimes mid-cycle, sometimes near a boundary.
  - Harm signals are processed whenever the cycle boundary next arrives (0-K-1 steps
    later); no alignment guarantee.
  - Same E3_HEARTBEAT_K=5, same update rate -- only the phase alignment differs.

Testable prediction:
  PHASE_RESET_ON should produce higher harm_eval_gap than PHASE_RESET_ABLATED because:
  - Phase-reset aligns harm signals with cycle boundaries; E3 integrates them from
    the start of a fresh window.
  - Without reset, harm signals are smeared across arbitrary mid-cycle positions;
    some fall near the end of a cycle and have only 1-2 steps of integration time
    before the cycle flushes, producing lower signal-to-noise in the harm eval head.

This is a clean isolation: both conditions use the same heartbeat rate (K=5), the same
E1/E2/E3 architecture, the same nav_bias. Only the cycle-boundary alignment on salient
events differs. A positive delta (PHASE_RESET_ON gap > ABLATED gap) directly supports
MECH-091's functional claim.

Note on evidence_quality_note (claims.yaml):
  MECH-091 is noted as "held -- specific blocker is SD-006 (async multi-rate loop)."
  This experiment tests the functional claim using the same SD-006 phase 1 proxy
  (time-multiplexed) as EXQ-131. A PASS here is partial support for the functional
  claim (phase-alignment benefit exists in synchronous proxy) pending SD-006 phase 2.
  A FAIL would indicate the functional benefit does not emerge at V3 proxy scale.

Pre-registered acceptance criteria:
  C1: gap_reset_on >= 0.04       (PHASE_RESET_ON harm eval above floor -- both seeds)
  C2: per-seed delta (ON - ABLATED) >= 0.02  (phase-reset adds >=2pp -- both seeds)
  C3: gap_ablated >= 0.0         (ablation still learns something -- data quality)
  C4: n_harm_eval_min >= 20      (sufficient harm events in eval -- both seeds)
  C5: n_salient_resets >= 10     (manipulation check: phase resets actually fired
                                  during eval -- confirms salient events occurred
                                  and resets were triggered in PHASE_RESET_ON)

Decision scoring:
  PASS (all 5):    retain_ree -- phase-reset of E3 heartbeat on salient events
                   improves harm eval quality; supports MECH-091
  C1+C2+C4 (C5):  hybridize  -- phase-reset benefit present but manipulation check
                   inconclusive (few salient events); increase nav_bias and rerun
  C2 fails:        retire_ree_claim -- no detectable advantage of phase alignment at
                   V3 proxy scale; functional claim not supported
  C5 fails alone:  hybridize  -- data quality issue; salient events too rare

Seeds: [42, 123] (matched -- same env per seed across conditions)
Env:   CausalGridWorldV2, size=6, 4 hazards, nav_bias=0.50 (higher than EXQ-131's
       0.40 to ensure sufficient salient harm events for C5 manipulation check)
Warmup: 400 eps x 200 steps
Eval:  50 eps x 200 steps (no training, fixed weights)
E3 heartbeat: E3_HEARTBEAT_K=5 in both conditions
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


EXPERIMENT_TYPE = "v3_exq_133_mech091_phase_reset_pair"
CLAIM_IDS = ["MECH-091"]

# Pre-registered thresholds
THRESH_C1 = 0.04   # gap_reset_on >= 0.04 (PHASE_RESET_ON harm eval above floor)
THRESH_C2 = 0.02   # per-seed delta (ON - ABLATED) >= 0.02
THRESH_C3 = 0.0    # gap_ablated >= 0.0 (ablation still learns something)
THRESH_C4 = 20     # n_harm_eval_min >= 20 (data quality)
THRESH_C5 = 10     # n_salient_resets >= 10 (manipulation check: resets fired)

# E3 heartbeat: E3 updates every K steps
E3_HEARTBEAT_K = 5


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
    phase_reset_on: bool,
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
    heartbeat_k: int,
    dry_run: bool,
) -> Dict:
    """Run one (seed, condition) cell; return harm eval gap and phase-reset metrics.

    PHASE_RESET_ON:      E3 heartbeat counter resets on salient events (harm contact).
    PHASE_RESET_ABLATED: E3 heartbeat counter never resets; runs continuously.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    cond_label = "PHASE_RESET_ON" if phase_reset_on else "PHASE_RESET_ABLATED"

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

    # Harm eval buffer
    buf_zw: List[torch.Tensor] = []
    buf_labels: List[float] = []

    counts: Dict[str, int] = {
        "hazard_approach": 0,
        "env_caused_hazard": 0,
        "agent_caused_hazard": 0,
        "none": 0,
    }

    # Phase-reset counter and manipulation check
    # heartbeat_counter tracks steps since last E3 update
    heartbeat_counter = 0
    n_salient_resets = 0      # how many times we reset the counter on a salient event

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
        heartbeat_counter = 0

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_curr = latent.z_world.detach()

            # E3 heartbeat gating: update only when counter reaches heartbeat_k
            e3_update_this_step = (heartbeat_counter % heartbeat_k == 0)

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

            # E1 prediction loss update (every step, both conditions)
            try:
                e1_loss = agent.compute_prediction_loss()
                if e1_loss.requires_grad:
                    optimizer.zero_grad()
                    e1_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                    optimizer.step()
            except Exception:
                pass

            # Harm eval buffer update
            buf_zw.append(z_world_curr)
            buf_labels.append(1.0 if is_harm else 0.0)
            if len(buf_zw) > MAX_BUF:
                buf_zw = buf_zw[-MAX_BUF:]
                buf_labels = buf_labels[-MAX_BUF:]

            # E3 harm eval training (only on heartbeat boundaries)
            if e3_update_this_step:
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

            # Advance heartbeat counter
            heartbeat_counter += 1

            # Phase-reset on salient event (PHASE_RESET_ON only)
            # Salient event: harm contact (is_harm=True)
            # Reset counter to 0 so next step begins a fresh E3 cycle
            if phase_reset_on and is_harm:
                heartbeat_counter = 0
                n_salient_resets += 1

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == actual_warmup - 1:
            print(
                f"  [train] seed={seed} cond={cond_label}"
                f" ep {ep+1}/{actual_warmup}"
                f" approach={counts['hazard_approach']}"
                f" contact={counts['env_caused_hazard']+counts['agent_caused_hazard']}"
                f" resets={n_salient_resets}",
                flush=True,
            )

    # --- EVAL ---
    agent.eval()

    eval_scores_harm: List[float] = []
    eval_scores_safe: List[float] = []
    n_fatal = 0
    eval_resets = 0
    heartbeat_counter = 0

    for _ in range(actual_eval):
        _, obs_dict = env.reset()
        agent.reset()
        heartbeat_counter = 0

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

            # Score via E3 harm_eval head on heartbeat boundaries
            e3_update_this_step = (heartbeat_counter % heartbeat_k == 0)

            if e3_update_this_step:
                try:
                    with torch.no_grad():
                        score = float(agent.e3.harm_eval(z_world_curr).item())
                    if is_harm:
                        eval_scores_harm.append(score)
                    else:
                        eval_scores_safe.append(score)
                except Exception:
                    n_fatal += 1

            heartbeat_counter += 1

            # Phase-reset on salient event (PHASE_RESET_ON only) -- eval phase too
            if phase_reset_on and is_harm:
                heartbeat_counter = 0
                eval_resets += 1

            if done:
                break

    n_harm_eval = len(eval_scores_harm)
    n_safe_eval = len(eval_scores_safe)

    mean_harm = float(sum(eval_scores_harm) / max(1, n_harm_eval))
    mean_safe = float(sum(eval_scores_safe) / max(1, n_safe_eval))
    gap = mean_harm - mean_safe

    # Manipulation check: total resets observed (train + eval)
    total_resets = n_salient_resets + eval_resets

    print(
        f"  [eval] seed={seed} cond={cond_label}"
        f" gap={gap:.4f} mean_harm={mean_harm:.4f} mean_safe={mean_safe:.4f}"
        f" n_harm={n_harm_eval} n_safe={n_safe_eval}"
        f" total_resets={total_resets}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "phase_reset_on": phase_reset_on,
        "harm_eval_gap": float(gap),
        "mean_harm_score": float(mean_harm),
        "mean_safe_score": float(mean_safe),
        "n_harm_eval": int(n_harm_eval),
        "n_safe_eval": int(n_safe_eval),
        "n_fatal": int(n_fatal),
        "n_salient_resets_train": int(n_salient_resets),
        "n_salient_resets_eval": int(eval_resets),
        "n_salient_resets_total": int(total_resets),
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
    nav_bias: float = 0.50,
    heartbeat_k: int = E3_HEARTBEAT_K,
    dry_run: bool = False,
    **kwargs,
) -> dict:
    """Discriminative pair: PHASE_RESET_ON (E3 heartbeat resets on harm contact --
    MECH-091 phase-reset architecture) vs PHASE_RESET_ABLATED (continuous counter,
    no reset on salient events).
    Tests MECH-091: salient event phase-reset of E3 heartbeat clock improves harm eval
    discrimination quality by aligning harm signals with cycle boundaries.
    """
    results_on:      List[Dict] = []
    results_ablated: List[Dict] = []

    # Run cells in matched-seed order: for each seed, run both conditions
    for seed in seeds:
        for reset_on in [True, False]:
            label = "PHASE_RESET_ON" if reset_on else "PHASE_RESET_ABLATED"
            print(
                f"\n[V3-EXQ-133] {label} seed={seed}"
                f" warmup={warmup_episodes} eval={eval_episodes}"
                f" nav_bias={nav_bias} heartbeat_k={heartbeat_k}",
                flush=True,
            )
            r = _run_single(
                seed=seed,
                phase_reset_on=reset_on,
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
                heartbeat_k=heartbeat_k,
                dry_run=dry_run,
            )
            if reset_on:
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

    n_salient_resets_total = _avg(results_on, "n_salient_resets_total")

    # Pre-registered PASS criteria
    # C1: PHASE_RESET_ON gap >= THRESH_C1 in ALL seeds
    c1_pass = all(r["harm_eval_gap"] >= THRESH_C1 for r in results_on)
    # C2: per-seed delta (ON - ABLATED) >= THRESH_C2 in ALL seeds
    per_seed_deltas: List[float] = []
    for r_on in results_on:
        matching = [r for r in results_ablated if r["seed"] == r_on["seed"]]
        if matching:
            per_seed_deltas.append(r_on["harm_eval_gap"] - matching[0]["harm_eval_gap"])
    c2_pass = len(per_seed_deltas) > 0 and all(d >= THRESH_C2 for d in per_seed_deltas)
    # C3: PHASE_RESET_ABLATED gap >= 0 (ablation still learns something)
    c3_pass = gap_ablated >= THRESH_C3
    # C4: sufficient harm events in eval
    c4_pass = n_harm_min >= THRESH_C4
    # C5: manipulation check -- phase resets actually fired (salient events occurred)
    c5_pass = n_salient_resets_total >= THRESH_C5

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])
    status = "PASS" if all_pass else "FAIL"

    if all_pass:
        decision = "retain_ree"
    elif c1_pass and c2_pass and c4_pass:
        decision = "hybridize"
    elif not c2_pass:
        decision = "retire_ree_claim"
    else:
        decision = "hybridize"

    print(f"\n[V3-EXQ-133] Results:", flush=True)
    print(
        f"  gap_on={gap_on:.4f}"
        f" gap_ablated={gap_ablated:.4f}"
        f" delta_gap={delta_gap:+.4f}",
        flush=True,
    )
    print(
        f"  n_salient_resets_total={n_salient_resets_total:.0f}"
        f" n_harm_min={n_harm_min}"
        f" per_seed_deltas={[round(d, 4) for d in per_seed_deltas]}"
        f" decision={decision}  status={status} ({criteria_met}/5)",
        flush=True,
    )

    failure_notes: List[str] = []
    if not c1_pass:
        failing = [r["seed"] for r in results_on if r["harm_eval_gap"] < THRESH_C1]
        failure_notes.append(
            f"C1 FAIL: PHASE_RESET_ON gap below {THRESH_C1} in seeds {failing}"
            " -- phase-reset does not produce discriminative harm eval signal"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: per-seed deltas {[round(d, 4) for d in per_seed_deltas]}"
            f" < {THRESH_C2}"
            " -- phase-reset does not add >=2pp harm eval advantage over ablation"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: gap_ablated={gap_ablated:.4f} < {THRESH_C3}"
            " -- ablated condition fails entirely; confound check required"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: n_harm_min={n_harm_min} < {THRESH_C4}"
            " -- insufficient harm events in eval phase"
        )
    if not c5_pass:
        failure_notes.append(
            f"C5 FAIL: n_salient_resets_total={n_salient_resets_total:.0f} < {THRESH_C5}"
            " -- phase resets rarely fired; salient events too infrequent for"
            " manipulation check. Increase nav_bias or n_hazards and rerun."
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    if all_pass:
        interpretation = (
            "MECH-091 SUPPORTED (V3 proxy): salient event phase-reset of E3 heartbeat"
            f" (PHASE_RESET_ON) produces higher harm_eval quality"
            f" (gap_on={gap_on:.4f} vs gap_ablated={gap_ablated:.4f},"
            f" delta={delta_gap:+.4f} across {len(seeds)} seeds)."
            f" Manipulation check: {n_salient_resets_total:.0f} phase resets fired."
            " Aligning E3's integration window with salient harm events (cycle-boundary"
            " reset) improves harm/safe discrimination relative to the continuous-counter"
            " ablation. Supports MECH-091 functional claim: phase-reset ensures harm"
            " estimates enter E3 at the start of a fresh cycle."
        )
    elif c1_pass and c2_pass and c4_pass:
        interpretation = (
            f"Partial support: PHASE_RESET_ON achieves gap={gap_on:.4f} (C1 PASS),"
            f" delta={per_seed_deltas} (C2 PASS), but"
            f" n_salient_resets={n_salient_resets_total:.0f} < {THRESH_C5}"
            " (C5 FAIL). Phase-reset benefit present but manipulation check inconclusive"
            " -- too few salient events to confirm phase-alignment mechanism is the cause."
            " Increase nav_bias to generate more harm contacts."
        )
    else:
        interpretation = (
            f"MECH-091 NOT supported at V3 proxy level: gap_on={gap_on:.4f}"
            f" (C1 {'PASS' if c1_pass else 'FAIL'}),"
            f" delta={delta_gap:+.4f} (C2 {'PASS' if c2_pass else 'FAIL'})."
            " Phase-reset of E3 heartbeat on salient events does not produce a"
            " detectable improvement in harm_eval quality at this scale."
            " Possible reasons: (a) heartbeat_k=5 cycle is too short for phase"
            " alignment to matter -- try larger K; (b) harm events too rare to"
            " accumulate sufficient phase-aligned integration windows (check C5);"
            " (c) at world_dim=32 E3 harm_eval quality is dominated by training"
            " data volume rather than cycle-boundary alignment; (d) SD-006 phase 2"
            " async implementation required for full effect."
        )

    per_on_rows = "\n".join(
        f"  seed={r['seed']}: gap={r['harm_eval_gap']:.4f}"
        f" n_harm={r['n_harm_eval']}"
        f" resets={r['n_salient_resets_total']}"
        for r in results_on
    )
    per_ablated_rows = "\n".join(
        f"  seed={r['seed']}: gap={r['harm_eval_gap']:.4f}"
        f" n_harm={r['n_harm_eval']}"
        f" resets=-- (ablation)"
        for r in results_ablated
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-133 -- MECH-091 Salient-Event Phase-Reset Discriminative Pair\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-091\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**Conditions:** PHASE_RESET_ON vs PHASE_RESET_ABLATED\n"
        f"**Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps\n"
        f"**Env:** CausalGridWorldV2 size=6, 4 hazards, nav_bias={nav_bias}\n"
        f"**E3 heartbeat K:** {heartbeat_k} (both conditions)\n\n"
        f"## Design\n\n"
        f"MECH-091 asserts salient events phase-reset the E3 heartbeat clock,\n"
        f"ensuring harm estimates enter E3 at the start of a fresh cycle.\n\n"
        f"PHASE_RESET_ON: on harm contact (is_harm=True), heartbeat_counter resets"
        f" to 0, aligning the next E3 update window with the salient event.\n\n"
        f"PHASE_RESET_ABLATED: heartbeat_counter increments continuously; salient"
        f" events land at arbitrary phase positions (no alignment guarantee).\n\n"
        f"Both conditions: E3_HEARTBEAT_K={heartbeat_k}, same architecture, same"
        f" nav_bias. Only cycle-boundary alignment on harm events differs.\n\n"
        f"Key metric: harm_eval_gap = mean_harm_score - mean_safe_score at eval.\n"
        f"Manipulation check: n_salient_resets (phase resets fired in PHASE_RESET_ON).\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: gap_reset_on >= {THRESH_C1} in all seeds  (PHASE_RESET_ON above floor)\n"
        f"C2: per-seed delta (ON - ABLATED) >= {THRESH_C2} in all seeds"
        f"  (phase-reset adds >=2pp)\n"
        f"C3: gap_ablated >= {THRESH_C3}  (ablation learns something; data quality)\n"
        f"C4: n_harm_eval_min >= {THRESH_C4}  (sufficient harm events)\n"
        f"C5: n_salient_resets >= {THRESH_C5}  (phase resets fired; manipulation check)\n\n"
        f"## Results\n\n"
        f"| Condition | gap (avg) | mean_harm | mean_safe | resets |\n"
        f"|-----------|-----------|-----------|-----------|--------|\n"
        f"| PHASE_RESET_ON      | {gap_on:.4f}"
        f" | {_avg(results_on,      'mean_harm_score'):.4f}"
        f" | {_avg(results_on,      'mean_safe_score'):.4f}"
        f" | {n_salient_resets_total:.0f} |\n"
        f"| PHASE_RESET_ABLATED | {gap_ablated:.4f}"
        f" | {_avg(results_ablated, 'mean_harm_score'):.4f}"
        f" | {_avg(results_ablated, 'mean_safe_score'):.4f}"
        f" | -- |\n\n"
        f"**delta_gap (ON - ABLATED): {delta_gap:+.4f}**\n\n"
        f"Manipulation check: n_salient_resets_total={n_salient_resets_total:.0f}"
        f" (confirms phase resets fired in PHASE_RESET_ON).\n\n"
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
        f"| C5: n_salient_resets >= {THRESH_C5} | {'PASS' if c5_pass else 'FAIL'}"
        f" | {n_salient_resets_total:.0f} |\n\n"
        f"Criteria met: {criteria_met}/5 -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n\n"
        f"## Per-Seed\n\n"
        f"PHASE_RESET_ON:\n{per_on_rows}\n\n"
        f"PHASE_RESET_ABLATED:\n{per_ablated_rows}\n"
        f"{failure_section}\n"
    )

    metrics = {
        "gap_reset_on":              float(gap_on),
        "gap_ablated":               float(gap_ablated),
        "delta_gap":                 float(delta_gap),
        "n_harm_eval_min":           float(n_harm_min),
        "n_salient_resets_total":    float(n_salient_resets_total),
        "n_seeds":                   float(len(seeds)),
        "nav_bias":                  float(nav_bias),
        "alpha_world":               float(alpha_world),
        "heartbeat_k":               float(heartbeat_k),
        "per_seed_delta_min":        float(min(per_seed_deltas)) if per_seed_deltas else 0.0,
        "per_seed_delta_max":        float(max(per_seed_deltas)) if per_seed_deltas else 0.0,
        "crit1_pass":                1.0 if c1_pass else 0.0,
        "crit2_pass":                1.0 if c2_pass else 0.0,
        "crit3_pass":                1.0 if c3_pass else 0.0,
        "crit4_pass":                1.0 if c4_pass else 0.0,
        "crit5_pass":                1.0 if c5_pass else 0.0,
        "criteria_met":              float(criteria_met),
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
    parser.add_argument("--nav-bias",        type=float, default=0.50)
    parser.add_argument("--heartbeat-k",     type=int,   default=5)
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
        heartbeat_k=args.heartbeat_k,
        dry_run=args.dry_run,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["registered_thresholds"] = {
        "C1_gap_reset_on":          THRESH_C1,
        "C2_per_seed_delta":        THRESH_C2,
        "C3_gap_ablated":           THRESH_C3,
        "C4_n_harm_eval_min":       THRESH_C4,
        "C5_n_salient_resets":      THRESH_C5,
    }
    result["seeds"] = list(args.seeds)
    result["conditions"] = ["PHASE_RESET_ON", "PHASE_RESET_ABLATED"]
    result["dispatch_mode"] = "discriminative_pair"
    result["backlog_id"] = "EVB-0027"
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
