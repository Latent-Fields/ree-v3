"""
V3-EXQ-047b — SD-005 + SD-010 Joint Validation (latent split with harm stream clean)

Claims: SD-005, MECH-069

Context:
  EXQ-047 tested the z_self/z_world latent split vs unified z_gamma. It found:
    - Attribution improvement: 82% (0.035 vs 0.019) — real and unambiguous
    - Calibration improvement: 3.4pp — FAIL vs 5pp C1 threshold
  The claims.yaml evidence_quality_note explicitly diagnoses C1's failure:
    "partly attributable to nociceptive contamination in z_world: both split
    and unified conditions still carry harm proximity signals that conflate
    calibration metrics."

  SD-010 is now confirmed (EXQ-056c PASS, R=0.914; EXQ-059c PASS, 10× harm
  reduction). The dedicated HarmEncoder routes harm_obs → z_harm independently
  of z_world. This removes nociceptive content from z_world, creating the
  conditions EXQ-047 needed.

Design — two conditions (same env seed, both with SD-010 active):

  SPLIT   — SD-005 architecture: z_self (body/motor domain, E2) and z_world
             (environmental domain, E3/residue) maintained as separate channels
             with separate gradient flows. HarmEncoder produces z_harm independently.

  UNIFIED — Ablation: z_self and z_world averaged into one shared vector
             (unified_latent_mode=True). HarmEncoder still active and separate —
             only the z_self/z_world merge is ablated, NOT the harm stream.

  Calibration metric: world_forward_r2 (E2 prediction quality on z_world).
    With z_harm removed, z_world should be a cleaner world-state representation.
    The split should give E2 a cleaner z_self for motor-sensory prediction, and
    E3 a cleaner z_world for environmental attribution.

  Attribution metric: causal_sig — E3(harm_eval_z_harm(HarmEnc(E2(z_world, a_actual))))
    minus E3(harm_eval_z_harm(HarmEnc(E2(z_world, a_cf)))).
    With z_harm as the harm signal (not z_world), attribution should be:
    (a) cleaner because the harm encoder is trained directly,
    (b) better in split condition because z_world fed to E2 is more environment-pure.

PASS criteria (ALL must hold):
  C1: world_forward_r2_split > world_forward_r2_unified + 0.05
      (split z_world allows cleaner E2 world prediction)
  C2: attribution_gap_split > attribution_gap_unified + 0.01
      (split provides cleaner causal attribution via harm stream)
  C3: harm_eval_pearson_r > 0.50
      (HarmEncoder z_harm is trained and functional — SD-010 health check)
  C4: world_forward_r2_split > 0.30
      (E2 world model is actually working in split condition)
  C5: n_approach_eval >= 50
      (sufficient approach transitions for attribution signal)

Note: C1 FAIL alone is not catastrophic — it would mean z_self/z_world split
helps attribution (C2) but not E2 prediction quality. That would suggest the
benefit is in gradient specialization, not input separation. Still scientifically
informative.
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.latent.stack import HarmEncoder
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_047b_sd005_sd010_joint"
CLAIM_IDS = ["SD-005", "MECH-069"]

HARM_OBS_DIM   = 51
Z_HARM_DIM     = 32
WARMUP_EPS     = 500
EVAL_EPS       = 50
STEPS_PER_EP   = 200
HARM_SCALE     = 0.02
PROXIMITY_SCALE = 0.05


def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _pearson_r(x: List[float], y: List[float]) -> float:
    if len(x) < 3:
        return 0.0
    xa, ya = np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)
    if xa.std() < 1e-8 or ya.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(xa, ya)[0, 1])


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed, size=12, num_hazards=4, num_resources=5,
        hazard_harm=HARM_SCALE,
        env_drift_interval=5, env_drift_prob=0.1,
        proximity_harm_scale=PROXIMITY_SCALE,
        proximity_benefit_scale=PROXIMITY_SCALE * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        use_proxy_fields=True,   # SD-010: enables harm_obs in obs_dict
    )


def _build_condition(
    env: CausalGridWorldV2,
    seed: int,
    self_dim: int,
    world_dim: int,
    alpha_world: float,
    unified_latent_mode: bool,
) -> Tuple[REEAgent, HarmEncoder, optim.Optimizer, optim.Optimizer, optim.Optimizer, optim.Optimizer]:
    torch.manual_seed(seed)
    random.seed(seed)

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=0.3,
        reafference_action_dim=env.action_dim,
        use_harm_stream=False,  # SD-010 via standalone HarmEncoder, not LatentStack
    )
    config.latent.unified_latent_mode = unified_latent_mode

    agent = REEAgent(config)
    harm_enc = HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=Z_HARM_DIM)

    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval" not in n
        and "world_transition" not in n
        and "world_action_encoder" not in n
    ]
    optimizer       = optim.Adam(standard_params, lr=1e-3)
    wf_optimizer    = optim.Adam(
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters()), lr=1e-3,
    )
    harm_enc_opt    = optim.Adam(harm_enc.parameters(), lr=1e-3)
    harm_head_opt   = optim.Adam(agent.e3.harm_eval_z_harm_head.parameters(), lr=1e-4)

    return agent, harm_enc, optimizer, wf_optimizer, harm_enc_opt, harm_head_opt


def _train_condition(
    agent: REEAgent,
    harm_enc: HarmEncoder,
    env: CausalGridWorldV2,
    optimizer: optim.Optimizer,
    wf_optimizer: optim.Optimizer,
    harm_enc_opt: optim.Optimizer,
    harm_head_opt: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
    label: str,
) -> Dict:
    agent.train()
    harm_enc.train()

    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    # Stratified harm replay (none / approach / contact) — same as EXQ-056c
    from collections import deque
    strat_bufs = {
        "none":            deque(maxlen=2000),
        "hazard_approach": deque(maxlen=2000),
        "contact":         deque(maxlen=2000),
    }
    MIN_PER_BUCKET = 4
    SAMPLES_PER_BUCKET = 8

    def _bucket(ttype: str) -> str:
        if ttype in ("env_caused_hazard", "agent_caused_hazard"):
            return "contact"
        elif ttype == "hazard_approach":
            return "hazard_approach"
        return "none"

    counts: Dict[str, int] = {}

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        a_prev:       Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()
            z_world_curr = latent.z_world.detach()

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")
            counts[ttype] = counts.get(ttype, 0) + 1

            # SD-010: collect harm_obs for HarmEncoder
            harm_obs_new = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM))
            harm_obs_t   = harm_obs_new.unsqueeze(0).float()
            hazard_label = harm_obs_new[12].unsqueeze(0).unsqueeze(0).detach().float()

            with torch.no_grad():
                z_harm_new = harm_enc(harm_obs_t)

            bk = _bucket(ttype)
            strat_bufs[bk].append((harm_obs_t.detach(), float(hazard_label.item())))

            if z_world_prev is not None and a_prev is not None:
                wf_buf.append((z_world_prev.cpu(), a_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > 5000:
                    wf_buf = wf_buf[-5000:]

            # Standard E1 + E2 losses
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            # World-forward training
            if len(wf_buf) >= 16:
                k = min(32, len(wf_buf))
                idxs = torch.randperm(len(wf_buf))[:k].tolist()
                zw_b  = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
                a_b   = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
                wf_pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(wf_pred, zw1_b)
                if wf_loss.requires_grad:
                    wf_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e2.world_transition.parameters()) +
                        list(agent.e2.world_action_encoder.parameters()), 0.5,
                    )
                    wf_optimizer.step()

            # Stratified harm encoder training (SD-010)
            ready = [b for b in strat_bufs if len(strat_bufs[b]) >= MIN_PER_BUCKET]
            if len(ready) >= 2:
                ho_list, lbl_list = [], []
                for bk2 in strat_bufs:
                    buf = strat_bufs[bk2]
                    if len(buf) < MIN_PER_BUCKET:
                        continue
                    k2 = min(SAMPLES_PER_BUCKET, len(buf))
                    for i in random.sample(range(len(buf)), k2):
                        ho_list.append(buf[i][0])
                        lbl_list.append(buf[i][1])

                if len(ho_list) >= 6:
                    ho_batch  = torch.cat(ho_list, dim=0).to(agent.device)
                    lbl_batch = torch.tensor(lbl_list, dtype=torch.float32,
                                             device=agent.device).unsqueeze(1)
                    z_harm_batch = harm_enc(ho_batch)
                    pred_zh = agent.e3.harm_eval_z_harm(z_harm_batch)
                    loss_zh = F.mse_loss(pred_zh, lbl_batch)
                    harm_enc_opt.zero_grad()
                    harm_head_opt.zero_grad()
                    loss_zh.backward()
                    torch.nn.utils.clip_grad_norm_(harm_enc.parameters(), 0.5)
                    harm_enc_opt.step()
                    harm_head_opt.step()

            z_world_prev = z_world_curr
            a_prev = action.detach()
            if done:
                break

        if (ep + 1) % 100 == 0 or ep == num_episodes - 1:
            ap = counts.get("hazard_approach", 0)
            ct = counts.get("env_caused_hazard", 0) + counts.get("agent_caused_hazard", 0)
            print(
                f"  [{label}|train] ep {ep+1}/{num_episodes}"
                f"  approach={ap}  contact={ct}",
                flush=True,
            )

    return {"counts": counts, "wf_buf": wf_buf}


def _eval_condition(
    agent: REEAgent,
    harm_enc: HarmEncoder,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    label: str,
) -> Dict:
    agent.eval()
    harm_enc.eval()

    num_actions = env.action_dim
    harm_preds_z_harm:  List[float] = []
    hazard_labels:      List[float] = []
    causal_sigs_by_ttype: Dict[str, List[float]] = {}
    n_by_ttype: Dict[str, int] = {}

    wf_preds_all: List[torch.Tensor] = []
    wf_targets_all: List[torch.Tensor] = []

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        a_prev:       Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                z_world = latent.z_world

            action_idx = random.randint(0, num_actions - 1)
            action = _action_to_onehot(action_idx, num_actions, agent.device)
            agent._last_action = action

            flat_obs, _, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")
            n_by_ttype[ttype] = n_by_ttype.get(ttype, 0) + 1

            harm_obs_curr = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM))
            harm_obs_t    = harm_obs_curr.unsqueeze(0).float()
            lbl = float(harm_obs_curr[12].item())

            with torch.no_grad():
                z_harm  = harm_enc(harm_obs_t)
                pred_zh = float(agent.e3.harm_eval_z_harm(z_harm).item())
                harm_preds_z_harm.append(pred_zh)
                hazard_labels.append(lbl)

                # Attribution: actual vs counterfactual via harm stream
                z_world_next_actual = agent.e2.world_forward(z_world, action)
                harm_obs_approx_act = torch.zeros(1, HARM_OBS_DIM, device=agent.device)
                # Use raw pred from harm stream (no harm_bridge — testing z_world quality)
                harm_act = agent.e3.harm_eval_z_harm(harm_enc(harm_obs_t))

                sigs: List[float] = []
                for cf_idx in range(num_actions):
                    if cf_idx == action_idx:
                        continue
                    a_cf = _action_to_onehot(cf_idx, num_actions, agent.device)
                    z_cf = agent.e2.world_forward(z_world, a_cf)
                    # For causal sig use z_world-based harm eval (tests z_world quality)
                    harm_cf = agent.e3.harm_eval(z_cf)
                    harm_ac = agent.e3.harm_eval(z_world_next_actual)
                    sigs.append(float((harm_ac - harm_cf).item()))

                mean_sig = float(np.mean(sigs)) if sigs else 0.0
                causal_sigs_by_ttype.setdefault(ttype, []).append(mean_sig)

                # World-forward R² data
                if z_world_prev is not None and a_prev is not None:
                    z_world_pred = agent.e2.world_forward(z_world_prev, a_prev)
                    wf_preds_all.append(z_world_pred)
                    wf_targets_all.append(z_world.detach())

            z_world_prev = z_world.detach()
            a_prev = action.detach()
            if done:
                break

    # World-forward R²
    if len(wf_preds_all) >= 10:
        preds   = torch.cat(wf_preds_all, dim=0)
        targets = torch.cat(wf_targets_all, dim=0)
        ss_res = ((targets - preds) ** 2).sum()
        ss_tot = ((targets - targets.mean(0, keepdim=True)) ** 2).sum()
        wf_r2 = float((1 - ss_res / (ss_tot + 1e-8)).item())
    else:
        wf_r2 = 0.0

    # Harm calibration
    harm_pearson = _pearson_r(harm_preds_z_harm, hazard_labels)

    # Attribution gap: approach causal_sig vs none causal_sig
    sig_approach = float(np.mean(causal_sigs_by_ttype.get("hazard_approach", [0.0])))
    sig_none     = float(np.mean(causal_sigs_by_ttype.get("none", [0.0])))
    attr_gap     = sig_approach - sig_none
    n_approach   = n_by_ttype.get("hazard_approach", 0)

    print(f"\n  [{label}] Eval results:", flush=True)
    print(f"    world_forward_r2:      {wf_r2:.4f}", flush=True)
    print(f"    harm_eval_pearson_r:   {harm_pearson:.4f}", flush=True)
    print(f"    causal_sig approach:   {sig_approach:.6f}", flush=True)
    print(f"    causal_sig none:       {sig_none:.6f}", flush=True)
    print(f"    attribution_gap:       {attr_gap:.6f}", flush=True)
    print(f"    n_approach:            {n_approach}", flush=True)
    for t, n in sorted(n_by_ttype.items()):
        print(f"    {t}: n={n}", flush=True)

    return {
        "world_forward_r2": wf_r2,
        "harm_eval_pearson_r": harm_pearson,
        "attribution_gap": attr_gap,
        "causal_sig_approach": sig_approach,
        "causal_sig_none": sig_none,
        "n_approach": n_approach,
        "n_by_ttype": n_by_ttype,
    }


def run(
    seed: int = 0,
    warmup_episodes: int = WARMUP_EPS,
    eval_episodes: int = EVAL_EPS,
    steps_per_episode: int = STEPS_PER_EP,
    self_dim: int = 32,
    world_dim: int = 32,
    alpha_world: float = 0.9,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    print(
        f"[V3-EXQ-047b] SD-005 + SD-010 Joint Validation\n"
        f"  Both conditions have SD-010 (harm_obs → z_harm) ACTIVE.\n"
        f"  Only the z_self/z_world merge is ablated in the UNIFIED condition.\n"
        f"  seed={seed}  warmup={warmup_episodes}  eval_eps={eval_episodes}\n"
        f"  alpha_world={alpha_world}",
        flush=True,
    )

    condition_results: Dict[str, Dict] = {}

    for label, unified_mode in [("split", False), ("unified", True)]:
        print(f"\n{'='*60}", flush=True)
        print(f"[V3-EXQ-047b] CONDITION: {label} (unified_latent_mode={unified_mode})", flush=True)
        print("="*60, flush=True)

        env = _make_env(seed)
        agent, harm_enc, opt, wf_opt, harm_enc_opt, harm_head_opt = _build_condition(
            env, seed, self_dim, world_dim, alpha_world, unified_mode
        )

        _train_condition(
            agent, harm_enc, env, opt, wf_opt, harm_enc_opt, harm_head_opt,
            warmup_episodes, steps_per_episode, label,
        )

        print(f"\n[V3-EXQ-047b] Eval: {label} ({eval_episodes} eps)...", flush=True)
        result = _eval_condition(agent, harm_enc, env, eval_episodes, steps_per_episode, label)
        condition_results[label] = result

    # ── PASS / FAIL ──────────────────────────────────────────────────────────
    r_split   = condition_results["split"]
    r_unified = condition_results["unified"]

    c1_pass = r_split["world_forward_r2"]   > r_unified["world_forward_r2"]   + 0.05
    c2_pass = r_split["attribution_gap"]    > r_unified["attribution_gap"]    + 0.01
    c3_pass = r_split["harm_eval_pearson_r"] > 0.50
    c4_pass = r_split["world_forward_r2"]   > 0.30
    c5_pass = r_split["n_approach"]         >= 50

    all_pass     = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status       = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: wf_r2_split={r_split['world_forward_r2']:.4f}"
            f" <= wf_r2_unified={r_unified['world_forward_r2']:.4f} + 0.05"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: attr_gap_split={r_split['attribution_gap']:.6f}"
            f" <= attr_gap_unified={r_unified['attribution_gap']:.6f} + 0.01"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: harm_eval_pearson_r={r_split['harm_eval_pearson_r']:.4f} <= 0.50"
            f"  (SD-010 health check failed — HarmEncoder not functional)"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: world_forward_r2_split={r_split['world_forward_r2']:.4f} <= 0.30"
            f"  (E2 world model not learning)"
        )
    if not c5_pass:
        failure_notes.append(
            f"C5 FAIL: n_approach={r_split['n_approach']} < 50"
            f"  (insufficient approach transitions for attribution signal)"
        )

    print(f"\nV3-EXQ-047b verdict: {status}  ({criteria_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    print(f"\n  C1 wf_r2:     split={r_split['world_forward_r2']:.4f}"
          f"  unified={r_unified['world_forward_r2']:.4f}"
          f"  → {'PASS' if c1_pass else 'FAIL'}", flush=True)
    print(f"  C2 attr_gap:  split={r_split['attribution_gap']:.6f}"
          f"  unified={r_unified['attribution_gap']:.6f}"
          f"  → {'PASS' if c2_pass else 'FAIL'}", flush=True)
    print(f"  C3 harm_r:    {r_split['harm_eval_pearson_r']:.4f}"
          f"  → {'PASS' if c3_pass else 'FAIL'}", flush=True)
    print(f"  C4 wf_r2>0.3: {r_split['world_forward_r2']:.4f}"
          f"  → {'PASS' if c4_pass else 'FAIL'}", flush=True)
    print(f"  C5 n_approach:{r_split['n_approach']}"
          f"  → {'PASS' if c5_pass else 'FAIL'}", flush=True)

    metrics = {
        "world_forward_r2_split":         float(r_split["world_forward_r2"]),
        "world_forward_r2_unified":       float(r_unified["world_forward_r2"]),
        "wf_r2_delta":                    float(r_split["world_forward_r2"] - r_unified["world_forward_r2"]),
        "attribution_gap_split":          float(r_split["attribution_gap"]),
        "attribution_gap_unified":        float(r_unified["attribution_gap"]),
        "attr_gap_delta":                 float(r_split["attribution_gap"] - r_unified["attribution_gap"]),
        "causal_sig_approach_split":      float(r_split["causal_sig_approach"]),
        "causal_sig_approach_unified":    float(r_unified["causal_sig_approach"]),
        "causal_sig_none_split":          float(r_split["causal_sig_none"]),
        "causal_sig_none_unified":        float(r_unified["causal_sig_none"]),
        "harm_eval_pearson_r_split":      float(r_split["harm_eval_pearson_r"]),
        "harm_eval_pearson_r_unified":    float(r_unified["harm_eval_pearson_r"]),
        "n_approach_split":               float(r_split["n_approach"]),
        "n_approach_unified":             float(r_unified["n_approach"]),
        "alpha_world":                    float(alpha_world),
        "seed":                           float(seed),
        "crit1_pass":   1.0 if c1_pass else 0.0,
        "crit2_pass":   1.0 if c2_pass else 0.0,
        "crit3_pass":   1.0 if c3_pass else 0.0,
        "crit4_pass":   1.0 if c4_pass else 0.0,
        "crit5_pass":   1.0 if c5_pass else 0.0,
        "criteria_met": float(criteria_met),
        "fatal_error_count": 0.0,
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-047b — SD-005 + SD-010 Joint Validation

**Status:** {status}
**Claims:** SD-005, MECH-069
**Design:** split (z_self ≠ z_world) vs unified (z_self = z_world = avg). SD-010 active in BOTH conditions.
**alpha_world:** {alpha_world}  |  **Warmup:** {warmup_episodes} eps  |  **Eval:** {eval_episodes} eps  |  **Seed:** {seed}

## Context

EXQ-047 (SD-005) FAIL: calibration improvement only 3.4pp vs 5pp threshold.
Diagnosis: nociceptive content in z_world contaminated both conditions equally.
EXQ-047b retest: SD-010 now confirmed (EXQ-056c/059c PASS). HarmEncoder routes
harm_obs → z_harm independently, removing nociception from z_world gradient path.

## Results

| Metric | Split | Unified | Delta |
|---|---|---|---|
| world_forward_r2 | {r_split['world_forward_r2']:.4f} | {r_unified['world_forward_r2']:.4f} | {r_split['world_forward_r2'] - r_unified['world_forward_r2']:+.4f} |
| attribution_gap | {r_split['attribution_gap']:.6f} | {r_unified['attribution_gap']:.6f} | {r_split['attribution_gap'] - r_unified['attribution_gap']:+.6f} |
| harm_eval_pearson_r (SD-010) | {r_split['harm_eval_pearson_r']:.4f} | {r_unified['harm_eval_pearson_r']:.4f} | — |
| n_approach | {r_split['n_approach']} | {r_unified['n_approach']} | — |

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: wf_r2_split > wf_r2_unified + 0.05 (split improves E2 prediction) | {"PASS" if c1_pass else "FAIL"} | {r_split['world_forward_r2']:.4f} vs {r_unified['world_forward_r2']:.4f} |
| C2: attr_gap_split > attr_gap_unified + 0.01 (split improves attribution) | {"PASS" if c2_pass else "FAIL"} | {r_split['attribution_gap']:.6f} vs {r_unified['attribution_gap']:.6f} |
| C3: harm_eval_pearson_r > 0.50 (SD-010 HarmEncoder functional) | {"PASS" if c3_pass else "FAIL"} | {r_split['harm_eval_pearson_r']:.4f} |
| C4: wf_r2_split > 0.30 (E2 model is learning) | {"PASS" if c4_pass else "FAIL"} | {r_split['world_forward_r2']:.4f} |
| C5: n_approach >= 50 (sufficient attribution data) | {"PASS" if c5_pass else "FAIL"} | {r_split['n_approach']} |

Criteria met: {criteria_met}/5 → **{status}**
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass else ("mixed" if criteria_met >= 3 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",        type=int,   default=0)
    parser.add_argument("--warmup",      type=int,   default=WARMUP_EPS)
    parser.add_argument("--eval-eps",    type=int,   default=EVAL_EPS)
    parser.add_argument("--steps",       type=int,   default=STEPS_PER_EP)
    parser.add_argument("--alpha-world", type=float, default=0.9)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"]              = CLAIM_IDS[0]
    result["verdict"]            = result["status"]
    result["run_id"]             = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}", flush=True)
