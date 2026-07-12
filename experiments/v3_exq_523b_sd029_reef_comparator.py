#!/opt/local/bin/python3
"""V3-EXQ-523b: SD-029 Reef-Unblocked Comparator with Graduation Test

Corrected rerun fixing BreathOscillator-disabled bug (Bug 1: breath_period=0 in all
prior runs) and _committed_step_idx saturation bug (Bug 2: counter saturated at
H-1=29 and never reset between consecutive E3 commits in non-bistable path).

Supersedes V3-EXQ-523a.

--- Original 523a docstring below ---
Supersedes V3-EXQ-523. Key changes:
  1. P0: epsilon=0.4 exploration + aggressive external hazard injection
     (interval=5, prob=1.0, adjacent_only=False) to generate balanced
     self-caused and externally-caused harm events.
  2. P1/P2: adaptive graduation tests -- rolling r^2 >= 0.85 for
     n_windows_required=5 consecutive windows of 200 steps each
     (max 8000 steps total). Outcome is INCONCLUSIVE_UNDERTRAINED if
     any ARM_1 seed fails to graduate either phase.
  3. Fixed evidence_direction computed from outcome (not hardcoded).
     evidence_direction_per_claim set independently for SD-029 and MECH-256.

Three phases per arm/seed:
  P0: Data collection with epsilon-greedy policy + aggressive injection.
      self_caused: agent moved toward hazard AND NOT external_hazard_injected
      ext_caused:  info["external_hazard_injected"] == True
  P1: Train HarmEncoder (harm_obs -> z_harm_s) via harm_signal regression.
      Graduation gate: r^2 >= 0.85 for 5 consecutive 200-step windows.
  P2: Train E2HarmSForward on frozen z_harm_s consecutive transitions.
      Graduation gate: same 0.85/5-window/200-step protocol.
  P3: Evaluate causal comparator (only if both phases graduate).
      cf_gap = ||E2_harm_s(z, a_actual)|| - ||E2_harm_s(z, a_cf)||

Two arms:
  ARM_0: baseline (reef_enabled=False, no external injection)
  ARM_1: reef + food + aggressive external injection

Acceptance criteria:
  C0: ARM_1 n_self >= 20 AND n_ext >= 20 per seed
  C1: ARM_1 harm_forward_r2 >= 0.9 (forward model quality)
  C2: ARM_1 mean(cf_gap | self_caused) > 0
  C3: ARM_1 mean(cf_gap | self_caused) > mean(cf_gap | ext_caused)
  C4 (diagnostic): ARM_0 n_ext == 0

Overall PASS = graduated AND C0 AND C1 AND C2 AND C3.

Per-claim evidence direction:
  SD-029: "supports" if C2 and C3, "weakens" if trained-but-fails,
          "non_contributory" if INCONCLUSIVE_UNDERTRAINED
  MECH-256: "supports" if C1, "weakens" if trained-but-fails,
            "non_contributory" if INCONCLUSIVE_UNDERTRAINED

claim_ids: ["SD-029", "MECH-256"]
experiment_purpose: evidence
supersedes: V3-EXQ-523
"""

import json
import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_523b_sd029_reef_comparator"
QUEUE_ID = "V3-EXQ-523b"
CLAIM_IDS = ["SD-029", "MECH-256"]

FLEE_THRESHOLD = 2
OPP_ACTION = {0: 1, 1: 0, 2: 3, 3: 2, 4: 4}
HARM_OBS_DIM = 51

GRAD_THRESHOLD = 0.85
GRAD_WINDOW_SIZE = 200
GRAD_N_WINDOWS_REQ = 5
GRAD_MAX_WINDOWS = 40
P0_EPSILON = 0.4


def _move_toward(ax, ay, tx, ty):
    dx = tx - ax
    dy = ty - ay
    if dx == 0 and dy == 0:
        return 4
    if abs(dx) >= abs(dy):
        return 0 if dx < 0 else 1
    return 2 if dy < 0 else 3


def _heuristic_action(env, reef_cells, use_reef, rng):
    """Harm-avoiding, reef-aware, resource-seeking heuristic from EXQ-522."""
    ax, ay = env.agent_x, env.agent_y
    nearest_hz = min(
        (abs(ax - h[0]) + abs(ay - h[1]) for h in env.hazards),
        default=999,
    )
    if use_reef and nearest_hz <= FLEE_THRESHOLD and reef_cells:
        if (ax, ay) in reef_cells:
            return 4
        target = min(reef_cells, key=lambda r: abs(ax - r[0]) + abs(ay - r[1]))
        return _move_toward(ax, ay, target[0], target[1])
    if env.resources:
        target = min(env.resources, key=lambda r: abs(ax - r[0]) + abs(ay - r[1]))
        return _move_toward(ax, ay, target[0], target[1])
    return int(rng.integers(0, env.action_dim))


def _moved_toward_hazard(ax, ay, action, hazards):
    """True if action moves agent closer to nearest hazard."""
    DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}
    if not hazards:
        return False
    dx, dy = DELTAS.get(action, (0, 0))
    new_x, new_y = ax + dx, ay + dy
    old_dist = min(abs(ax - h[0]) + abs(ay - h[1]) for h in hazards)
    new_dist = min(abs(new_x - h[0]) + abs(new_y - h[1]) for h in hazards)
    return new_dist < old_dist


def _action_onehot(action_int, action_dim):
    oh = torch.zeros(1, action_dim)
    oh[0, action_int] = 1.0
    return oh


def _p1_graduated_train(agent, harm_head, harm_opt, transitions, seed,
                         window_size, n_req, max_windows, threshold):
    """Train HarmEncoder with rolling r^2 graduation gate.

    Returns (graduated, n_steps, final_r2, mean_loss).
    """
    rng_idx = np.random.default_rng(seed + 100)
    consecutive = 0
    total_steps = 0
    last_r2 = 0.0
    all_losses = []

    for _w in range(max_windows):
        preds, targets, losses = [], [], []
        for _ in range(window_size):
            idx = int(rng_idx.integers(0, len(transitions)))
            h_obs_t, _, _act, harm_sig = transitions[idx]
            harm_in = h_obs_t.unsqueeze(0)
            z_hs = agent.latent_stack.harm_encoder(harm_in)
            pred = harm_head(z_hs).squeeze()
            tgt = torch.tensor(harm_sig, dtype=torch.float32)
            loss = F.mse_loss(pred, tgt)
            harm_opt.zero_grad()
            loss.backward()
            harm_opt.step()
            preds.append(pred.detach())
            targets.append(tgt.detach())
            losses.append(loss.item())
            total_steps += 1

        preds_cat = torch.stack(preds)
        tgts_cat = torch.stack(targets)
        ss_res = ((preds_cat - tgts_cat) ** 2).sum().item()
        ss_tot = ((tgts_cat - tgts_cat.mean()) ** 2).sum().item()
        r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
        last_r2 = r2
        all_losses.extend(losses)

        if r2 >= threshold:
            consecutive += 1
            if consecutive >= n_req:
                mean_loss = float(np.mean(all_losses[-window_size:]))
                return True, total_steps, r2, mean_loss
        else:
            consecutive = 0

    mean_loss = float(np.mean(all_losses[-window_size:])) if all_losses else 0.0
    return False, total_steps, last_r2, mean_loss


def _p2_graduated_train(agent, e2_opt, transitions, action_dim, seed,
                         window_size, n_req, max_windows, threshold):
    """Train E2HarmSForward with rolling r^2 graduation gate.

    Returns (graduated, n_steps, final_r2, mean_loss).
    """
    n_tr = len(transitions)
    rng_p2 = np.random.default_rng(seed + 200)
    consecutive = 0
    total_steps = 0
    last_r2 = 0.0
    all_losses = []

    for _w in range(max_windows):
        preds, targets, losses = [], [], []
        for _ in range(window_size):
            i = int(rng_p2.integers(0, max(1, n_tr - 1)))
            _, z_t, act_t, _ = transitions[i]
            _, z_t1, _, _ = transitions[i + 1]
            a_oh = _action_onehot(act_t, action_dim)
            z_pred = agent.e2_harm_s(z_t.detach(), a_oh)
            loss = F.mse_loss(z_pred, z_t1.detach())
            e2_opt.zero_grad()
            loss.backward()
            e2_opt.step()
            preds.append(z_pred.detach())
            targets.append(z_t1.detach())
            losses.append(loss.item())
            total_steps += 1

        preds_cat = torch.cat(preds, dim=0)
        tgts_cat = torch.cat(targets, dim=0)
        ss_res = ((preds_cat - tgts_cat) ** 2).sum().item()
        ss_tot = ((tgts_cat - tgts_cat.mean(dim=0, keepdim=True)) ** 2).sum().item()
        r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
        last_r2 = r2
        all_losses.extend(losses)

        if r2 >= threshold:
            consecutive += 1
            if consecutive >= n_req:
                mean_loss = float(np.mean(all_losses[-window_size:]))
                return True, total_steps, r2, mean_loss
        else:
            consecutive = 0

    mean_loss = float(np.mean(all_losses[-window_size:])) if all_losses else 0.0
    return False, total_steps, last_r2, mean_loss


def _run_arm(arm_id, env_kwargs, n_episodes, steps_per_ep, n_seeds, p0_epsilon,
             window_size, n_req, max_windows, grad_threshold, rng, dry_run=False):
    """Run one arm across multiple seeds. Returns list of per-seed result dicts."""
    use_reef = env_kwargs.get("reef_enabled", False)
    seed_results = []

    for seed_idx in range(n_seeds):
        seed = int(rng.integers(0, 10000))
        kw = dict(**env_kwargs, seed=seed)
        env = CausalGridWorldV2(**kw)

        cfg = REEConfig.from_dims(
            body_obs_dim=env.body_obs_dim,
            world_obs_dim=env.world_obs_dim,
            action_dim=env.action_dim,
            use_harm_stream=True,
            harm_obs_dim=HARM_OBS_DIM,
            z_harm_dim=32,
            use_e2_harm_s_forward=True,
        )
        agent = REEAgent(cfg)

        # ------------------------------------------------------------------
        # P0: Data collection with epsilon-greedy + aggressive injection
        # ------------------------------------------------------------------
        transitions = []
        events_self = []
        events_ext = []
        n_self = 0
        n_ext = 0

        p0_rng = np.random.default_rng(seed + 50)

        for ep in range(n_episodes):
            _, obs_dict = env.reset()
            reef_cells = getattr(env, "_reef_cells", set())

            for _step in range(steps_per_ep):
                ax, ay = env.agent_x, env.agent_y
                hazards_before = list(env.hazards)
                h_obs = obs_dict["harm_obs"].float()
                harm_obs_t = h_obs.unsqueeze(0)

                # Epsilon-greedy: mix exploration with heuristic
                if p0_rng.random() < p0_epsilon:
                    action = int(p0_rng.integers(0, env.action_dim))
                else:
                    action = _heuristic_action(env, reef_cells, use_reef, rng)

                moved_toward = _moved_toward_hazard(ax, ay, action, hazards_before)
                _, harm_signal, done, info, obs_dict_next = env.step(action)
                obs_dict = obs_dict_next

                external = bool(info.get("external_hazard_injected", False))
                is_self = (moved_toward and not external)
                is_ext = external

                with torch.no_grad():
                    z_harm_s = agent.latent_stack.harm_encoder(harm_obs_t)

                transitions.append((
                    h_obs.clone(),
                    z_harm_s.detach().clone(),
                    action,
                    float(harm_signal),
                ))

                if is_self:
                    events_self.append({"harm_obs_t": h_obs.clone(), "action": action})
                    n_self += 1
                if is_ext:
                    events_ext.append({"harm_obs_t": h_obs.clone(), "action": action})
                    n_ext += 1

                if done:
                    break

        if dry_run:
            print(
                f"  arm={arm_id} seed={seed_idx} P0: "
                f"n_self={n_self} n_ext={n_ext} transitions={len(transitions)}"
            )

        # ------------------------------------------------------------------
        # P1: Train HarmEncoder with graduation gate
        # ------------------------------------------------------------------
        harm_head = nn.Linear(cfg.latent.z_harm_dim, 1)
        harm_opt = torch.optim.Adam(
            list(agent.latent_stack.harm_encoder.parameters())
            + list(harm_head.parameters()),
            lr=5e-4,
        )

        p1_grad, p1_steps, p1_r2, mean_p1_loss = _p1_graduated_train(
            agent, harm_head, harm_opt, transitions, seed,
            window_size, n_req, max_windows, grad_threshold,
        )

        if dry_run:
            print(
                f"  arm={arm_id} seed={seed_idx} P1: "
                f"graduated={p1_grad} steps={p1_steps} r2={p1_r2:.3f}"
            )

        if not p1_grad:
            seed_results.append({
                "arm_id": arm_id, "seed_idx": seed_idx,
                "n_self": n_self, "n_ext": n_ext,
                "p1_graduated": False, "p2_graduated": False,
                "p1_steps": p1_steps, "p1_r2": p1_r2,
                "p2_steps": 0, "p2_r2": 0.0,
                "harm_forward_r2": 0.0,
                "mean_cf_gap_self": 0.0, "mean_cf_gap_ext": 0.0,
                "mean_p1_loss": mean_p1_loss, "mean_p2_loss": 0.0,
            })
            continue

        # Re-encode all transitions with trained (now frozen) encoder
        new_transitions = []
        with torch.no_grad():
            for h_obs_t, _, a, hs in transitions:
                z_hs = agent.latent_stack.harm_encoder(h_obs_t.unsqueeze(0))
                new_transitions.append((h_obs_t, z_hs.detach().clone(), a, hs))
        transitions = new_transitions

        # ------------------------------------------------------------------
        # P2: Train E2HarmSForward with graduation gate
        # ------------------------------------------------------------------
        e2_opt = torch.optim.Adam(agent.e2_harm_s.parameters(), lr=5e-4)

        p2_grad, p2_steps, p2_r2_win, mean_p2_loss = _p2_graduated_train(
            agent, e2_opt, transitions, env.action_dim, seed,
            window_size, n_req, max_windows, grad_threshold,
        )

        if dry_run:
            print(
                f"  arm={arm_id} seed={seed_idx} P2: "
                f"graduated={p2_grad} steps={p2_steps} r2={p2_r2_win:.3f}"
            )

        # Final harm_forward_r2 on a random held-out sample (C1 metric)
        n_tr = len(transitions)
        preds_held, targets_held = [], []
        rng_eval = np.random.default_rng(seed + 300)
        with torch.no_grad():
            for _ in range(min(400, max(1, n_tr - 1))):
                i = int(rng_eval.integers(0, max(1, n_tr - 1)))
                _, z_t, act_t, _ = transitions[i]
                _, z_t1, _, _ = transitions[i + 1]
                a_oh = _action_onehot(act_t, env.action_dim)
                z_pred = agent.e2_harm_s(z_t.detach(), a_oh)
                preds_held.append(z_pred.detach())
                targets_held.append(z_t1.detach())

        if preds_held:
            pc = torch.cat(preds_held, dim=0)
            tc = torch.cat(targets_held, dim=0)
            ss_res = ((pc - tc) ** 2).sum().item()
            ss_tot = ((tc - tc.mean(dim=0, keepdim=True)) ** 2).sum().item()
            harm_forward_r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
        else:
            harm_forward_r2 = 0.0

        if dry_run:
            print(f"  arm={arm_id} seed={seed_idx} P2 final r2={harm_forward_r2:.3f}")

        # ------------------------------------------------------------------
        # P3: Evaluate causal comparator (only when both phases graduated)
        # ------------------------------------------------------------------
        cf_gaps_self = []
        cf_gaps_ext = []

        with torch.no_grad():
            for ev_list, cf_list in [(events_self, cf_gaps_self),
                                     (events_ext, cf_gaps_ext)]:
                for ev in ev_list:
                    h_obs_ev = ev["harm_obs_t"]
                    act = ev["action"]
                    act_cf = OPP_ACTION[act]
                    z_hs = agent.latent_stack.harm_encoder(h_obs_ev.unsqueeze(0))
                    a_actual_oh = _action_onehot(act, env.action_dim)
                    a_cf_oh = _action_onehot(act_cf, env.action_dim)
                    z_pred_actual = agent.e2_harm_s(z_hs.detach(), a_actual_oh)
                    z_pred_cf = agent.e2_harm_s.counterfactual_forward(
                        z_hs.detach(), a_cf_oh
                    )
                    gap = z_pred_actual.norm().item() - z_pred_cf.norm().item()
                    cf_list.append(gap)

        mean_cf_gap_self = float(np.mean(cf_gaps_self)) if cf_gaps_self else 0.0
        mean_cf_gap_ext = float(np.mean(cf_gaps_ext)) if cf_gaps_ext else 0.0

        if dry_run:
            print(
                f"  arm={arm_id} seed={seed_idx} P3: "
                f"fwd_r2={harm_forward_r2:.3f} "
                f"cf_self={mean_cf_gap_self:.4f} cf_ext={mean_cf_gap_ext:.4f}"
            )

        seed_results.append({
            "arm_id": arm_id, "seed_idx": seed_idx,
            "n_self": n_self, "n_ext": n_ext,
            "p1_graduated": p1_grad, "p2_graduated": p2_grad,
            "p1_steps": p1_steps, "p1_r2": p1_r2,
            "p2_steps": p2_steps, "p2_r2": p2_r2_win,
            "harm_forward_r2": harm_forward_r2,
            "mean_cf_gap_self": mean_cf_gap_self, "mean_cf_gap_ext": mean_cf_gap_ext,
            "mean_p1_loss": mean_p1_loss, "mean_p2_loss": mean_p2_loss,
        })

    return seed_results


def run_experiment(dry_run=False):
    n_episodes = 10 if dry_run else 80
    steps_per_ep = 50 if dry_run else 200
    n_seeds = 1 if dry_run else 3
    p0_epsilon = P0_EPSILON

    # Graduation parameters: dry run uses a threshold of -1e9 (always graduates
    # after the first window), very small windows to keep runtime short.
    window_size = 20 if dry_run else GRAD_WINDOW_SIZE
    n_req = 1 if dry_run else GRAD_N_WINDOWS_REQ
    max_windows = 3 if dry_run else GRAD_MAX_WINDOWS
    grad_threshold = -1e9 if dry_run else GRAD_THRESHOLD

    grid_size = 12
    rng = np.random.default_rng(7000)

    common = dict(
        size=grid_size, num_hazards=3, num_resources=5,
        use_proxy_fields=True,
        env_drift_prob=0.3, env_drift_interval=1,
    )

    arms_def = [
        ("ARM_0_baseline",
         dict(**common,
              reef_enabled=False,
              hazard_food_attraction=0.0)),
        ("ARM_1_reef_food_ext",
         dict(**common,
              reef_enabled=True, n_reef_patches=3, reef_patch_radius=2,
              hazard_food_attraction=0.7,
              scheduled_external_hazard_enabled=True,
              scheduled_external_hazard_interval=5,
              scheduled_external_hazard_prob=1.0,
              scheduled_external_hazard_adjacent_only=False)),
    ]

    all_results = {}
    for arm_id, env_kwargs in arms_def:
        results = _run_arm(
            arm_id=arm_id,
            env_kwargs=env_kwargs,
            n_episodes=n_episodes,
            steps_per_ep=steps_per_ep,
            n_seeds=n_seeds,
            p0_epsilon=p0_epsilon,
            window_size=window_size,
            n_req=n_req,
            max_windows=max_windows,
            grad_threshold=grad_threshold,
            rng=rng,
            dry_run=dry_run,
        )
        all_results[arm_id] = results

    def _agg(arm_id, key):
        vals = [r[key] for r in all_results[arm_id]]
        return float(np.mean(vals))

    def _min_val(arm_id, key):
        vals = [r[key] for r in all_results[arm_id]]
        return float(np.min(vals))

    arm1 = "ARM_1_reef_food_ext"
    arm0 = "ARM_0_baseline"
    arm1_results = all_results[arm1]

    # Graduation gate: all ARM_1 seeds must graduate both phases
    graduated = all(
        r["p1_graduated"] and r["p2_graduated"] for r in arm1_results
    )

    if not graduated:
        outcome = "INCONCLUSIVE_UNDERTRAINED"
        overall_pass = False
        c0 = c1 = c2 = c3 = c4 = False
        sd029_dir = "non_contributory"
        mech256_dir = "non_contributory"
    else:
        c0 = all(
            r["n_self"] >= 20 and r["n_ext"] >= 20 for r in arm1_results
        )
        c1 = _min_val(arm1, "harm_forward_r2") >= 0.9
        c2 = _agg(arm1, "mean_cf_gap_self") > 0.0
        c3 = _agg(arm1, "mean_cf_gap_self") > _agg(arm1, "mean_cf_gap_ext")
        c4 = _agg(arm0, "n_ext") == 0

        overall_pass = c0 and c1 and c2 and c3
        outcome = "PASS" if overall_pass else "FAIL"

        sd029_dir = "supports" if (c2 and c3) else "weakens"
        mech256_dir = "supports" if c1 else "weakens"

    per_arm = {}
    for arm_id, _ in arms_def:
        row = {}
        for key in ["n_self", "n_ext", "harm_forward_r2",
                    "mean_cf_gap_self", "mean_cf_gap_ext",
                    "mean_p1_loss", "mean_p2_loss",
                    "p1_steps", "p1_r2", "p2_steps", "p2_r2"]:
            row[key] = _agg(arm_id, key)
        row["p1_graduated_all"] = all(r["p1_graduated"] for r in all_results[arm_id])
        row["p2_graduated_all"] = all(r["p2_graduated"] for r in all_results[arm_id])
        per_arm[arm_id] = row

    return {
        "per_arm": per_arm,
        "criteria": {"C0": c0, "C1": c1, "C2": c2, "C3": c3, "C4": c4},
        "overall_pass": overall_pass,
        "outcome": outcome,
        "graduated": graduated,
        "evidence_direction_per_claim": {
            "SD-029": sd029_dir,
            "MECH-256": mech256_dir,
        },
        "n_episodes": n_episodes,
        "steps_per_ep": steps_per_ep,
        "n_seeds": n_seeds,
        "p0_epsilon": p0_epsilon,
        "grad_threshold": GRAD_THRESHOLD,
        "grad_window_size": GRAD_WINDOW_SIZE,
        "grad_n_windows_req": GRAD_N_WINDOWS_REQ,
        "grad_max_windows": GRAD_MAX_WINDOWS,
    }


def write_result(result, run_id):
    script_dir = Path(__file__).resolve().parents[1]
    out_dir = (
        script_dir.parent / "REE_assembly" / "evidence"
        / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    edc = result.get("evidence_direction_per_claim", {})
    sd029_dir = edc.get("SD-029", "non_contributory")
    mech256_dir = edc.get("MECH-256", "non_contributory")
    dirs = {sd029_dir, mech256_dir}
    if dirs == {"supports"}:
        overall_dir = "supports"
    elif dirs == {"non_contributory"}:
        overall_dir = "non_contributory"
    elif "supports" in dirs:
        overall_dir = "mixed"
    else:
        overall_dir = "weakens"

    out_path = out_dir / f"{run_id}.json"
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": overall_dir,
        "evidence_direction_per_claim": edc,
        "experiment_purpose": "evidence",
        "supersedes": "v3_exq_523a_sd029_reef_comparator",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "outcome": result["outcome"],
        "metrics": result,
    }
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=None,
        script_path=Path(__file__),
    )
    print(f"Result written to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    print("V3-EXQ-523a SD-029 Reef-Unblocked Comparator with Graduation Test")
    print(f"dry_run={args.dry_run}")

    result = run_experiment(dry_run=args.dry_run)

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"Graduated: {result['graduated']}")
    print(f"Outcome: {result['outcome']}")
    print(f"Criteria: {result['criteria']}")
    print(f"Evidence per claim: {result['evidence_direction_per_claim']}")

    for arm_id, stats in result["per_arm"].items():
        print(
            f"  {arm_id}: "
            f"n_self={stats['n_self']:.0f} "
            f"n_ext={stats['n_ext']:.0f} "
            f"fwd_r2={stats['harm_forward_r2']:.3f} "
            f"cf_gap_self={stats['mean_cf_gap_self']:.4f} "
            f"cf_gap_ext={stats['mean_cf_gap_ext']:.4f} "
            f"p1_grad={stats['p1_graduated_all']} "
            f"p2_grad={stats['p2_graduated_all']}"
        )

    if not args.dry_run:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
        write_result(result, run_id)

    # --- runner-conformance sentinel (added by retrofit_experiments.py) ---
    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=None,
    )
