#!/opt/local/bin/python3
"""
V3-EXQ-630 -- ARC-029 / MECH-090: Across-tick nav_competence gate, ECOLOGICAL
              suppress-on-degrade / admit-on-recover (successor to V3-EXQ-063a)

Claims: ARC-029, MECH-090

WHAT THIS COMPLETES. V3-EXQ-063a re-ran the ARC-029 committed-mode design with the
MECH-090 R-c commit-entry gate active but exercised ONLY the WITHIN-TICK
decisiveness axis (per-candidate score margin). It deliberately left the ACROSS-TICK
nav_competence axis (use_mech090_readiness_conjunction) OFF, because that axis needs
a per-tick readiness OUTCOME signal and -- at the time -- no code path produced one
(committed_mode_curriculum computes nav_competence but never pushes it via
notify_outcome; grep-verified zero callers repo-wide). So the readiness EMA sat
fail-open (pinned at the initial 1.0) and the across-tick gate added no signal.

The Phase-2 env-source follow-on landed this session (ree-v3 fa026a0,
ree-v3/CLAUDE.md "MECH-090 R-c continuation Phase-2 follow-on"): CausalGridWorldV2
gains mech090_readiness_outcome_enabled emitting
info["mech090_readiness_outcome"] = clip(1 - mean(limb_damage), 0, 1) (a Cisek &
Kalaska 2010 affordance-preparation / Roesch-Calu-Schoenbaum 2007 dopaminergic
motor-program-readiness scalar), and REEAgent.sense(mech090_readiness_outcome=...)
forwards it into commit_readiness.update(). This experiment is the first ecological
run that exercises the across-tick axis: it drives readiness DOWN (SD-022 scheduled
limb-damage injection) and UP (healing) and measures whether the nav_competence gate
SUPPRESSES commitment when readiness degrades (readiness < floor) and ADMITS it when
readiness recovers (readiness >= floor).

CLAIM TAGGING (re-evaluated from scratch per the claim_ids accuracy rule; NOT
inherited from 063a's single ARC-029 tag):
  MECH-090 -- this run directly exercises and measures the R-c across-tick
    nav_competence gate (the suppress/admit dynamic IS the load-bearing
    measurement). evidence_direction_per_claim governs.
  ARC-029  -- committed-mode is the context; whether the gate admits committed
    mode when motor-ready and suppresses it when not is a substantively different
    ARC-029 verdict than 063's ungated PASS (see the C3-FAIL routing below).

SUBSTRATE READINESS (verified): MECH-090 R-c continuation IMPLEMENTED 2026-05-29 +
the env-source Phase-2 follow-on landed this session; SD-022 limb damage + scheduled
injection IMPLEMENTED; commit_readiness consumer + AND-composition at both elevate
sites wired. All present in ree-v3/CLAUDE.md.

DESIGN -- 3-arm eval on ONE trained-vanilla agent per seed (weights shared via
state_dict so the arm comparison isolates the eval-time gate, not the weights):

  Train: vanilla agent (no R-c gates -> commits cleanly) on a standard env with
  limb_damage_enabled=True (so body_obs_dim matches the eval env) until
  running_variance collapses below commit_threshold (committed-capable). No
  readiness emission, no scheduled injection during training.

  Eval env (degrade/recover): limb_damage_enabled=True +
  mech090_readiness_outcome_enabled=True + SD-022 scheduled all-limb injection
  (periodic pulses) + healing. Readiness oscillates: post-injection DEGRADED
  windows (readiness < floor) and recovered READY windows (readiness >= floor).
  The env-emitted outcome is forwarded into the NEXT sense() call.

  Arms (fresh agent per arm, trained weights loaded; gate config set at
  construction -- no runtime flag toggling):
    ARM_0_OFF_BASELINE       use_mech090_readiness_conjunction=False,
                             use_commit_readiness_gate=False; readiness NOT
                             forwarded. Control: is commitment modulated by the
                             readiness regime when the gate is OFF? It must NOT be
                             (rules out a damage->variance env confound).
    ARM_2_GATED_NAV_COMP_ON  use_mech090_readiness_conjunction=True,
                             use_commit_readiness_gate=False (nav axis ALONE);
                             readiness forwarded. The 592b ARM_2 reading,
                             ecological. Suppress-on-degrade / admit-on-recover.
    ARM_3_GATED_BOTH_ON      both gates True; readiness forwarded. The 592b ARM_3
                             reading. Composed conjunction at least as strict as
                             either single axis.

  "Committed step" = agent.beta_gate.is_elevated after select_action (whether the
  gate actually elevated this tick). Each step is bucketed by the readiness reading
  BEFORE select_action: READY (readiness >= floor) vs DEGRADED (readiness < floor).
  committed_rate_<bucket> = committed_steps_in_bucket / total_steps_in_bucket.

  mech090_readiness_floor = 0.3 (the 2026-05-29 default). NOT re-tuned here.

PASS criteria (C1-C4 required; C5 informative):
  C1 GATE FIRES: in the GATED arms, the nav gate blocked at least once
     (commit_readiness n_blocks_emitted >= 1 aggregated), i.e. degraded windows
     occurred AND the agent was committed-capable so the gate was consulted.
  C2 SUPPRESS-ON-DEGRADE: ARM_2 committed_rate_ready - committed_rate_degraded
     >= SUPPRESS_MARGIN (0.15). Commitment is suppressed in degraded windows.
  C3 ADMIT-ON-RECOVER: ARM_2 committed_rate_ready >= ADMIT_FLOOR (0.10). The gate
     does NOT permanently lock out commitment; it admits when readiness recovers.
  C4 OFF-BASELINE NULL: ARM_0 abs(committed_rate_ready - committed_rate_degraded)
     < SUPPRESS_MARGIN (0.15). Commitment is NOT readiness-conditioned when the
     gate is OFF -> the C2 effect is the GATE, not a damage->commitment confound.
  C5 COMPOSED AT LEAST AS STRICT: ARM_3 committed_rate_degraded
     <= ARM_2 committed_rate_degraded + 0.05.

INTERPRETATION GRID (for /governance + /failure-autopsy when 630 completes):
  - PASS (C1-C4) -> MECH-090 R-c across-tick nav_competence axis validated
    ecologically; ARC-029 committed-mode is entered when motor-ready and suppressed
    when not. Both claims: supports on the current substrate.
  - C2 holds but C3 FAILS (committed_rate_ready also ~0 -> the gate permanently
    suppresses commitment, never admitting even when readiness recovers): this is
    the LOAD-BEARING branch and is NOT noise. It means the R-c gate suppresses
    commitment in this env = a substantively different ARC-029 verdict than the
    original 063 PASS. Route to /failure-autopsy, NOT /diagnose-errors: the question
    is whether mech090_readiness_floor=0.3 is mis-calibrated for this env's readiness
    distribution, or whether committed-mode is genuinely no longer entered under R-c
    gating. DO NOT silently re-tune commit_readiness_floor -- that is a substrate
    decision. (Per REE_assembly/evidence/planning/epoch_stale_evidence_review_2026-06-02.md
    REVALIDATION STATUS section, C3-FAIL routing.)
  - C1 FAIL (n_blocks == 0 despite the degrade curriculum) -> wiring/instrumentation:
    readiness never dropped below floor, the agent was never committed-capable, or
    the outcome was not forwarded. Route to /diagnose-errors (check env emission,
    forwarding, and post-train running_variance < commit_threshold).
  - C4 FAIL (OFF baseline ALSO shows readiness-conditioned commitment) -> confound:
    limb damage independently suppresses commitment (e.g. via movement-failure
    perturbing running_variance), so the C2 effect cannot be attributed to the gate.
    Route to /diagnose-errors.
  - C5 FAIL -> composition anomaly (both-on less strict than nav-alone). /diagnose-errors.
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_630_arc029_acrosstick_nav_competence_ecological"
CLAIM_IDS = ["ARC-029", "MECH-090"]
EXPERIMENT_PURPOSE = "evidence"

MECH090_READINESS_FLOOR = 0.3      # the 2026-05-29 default; NOT re-tuned here
COMMIT_READINESS_EMA_ALPHA = 0.2   # ~5-tick half-life so the EMA tracks the
                                   # injection/heal curriculum within an episode

# Pre-registered acceptance thresholds (constants, not derived post-hoc).
SUPPRESS_MARGIN = 0.15   # C2 / C4
ADMIT_FLOOR = 0.10       # C3
COMPOSE_EPS = 0.05       # C5

# Standard training env. limb_damage_enabled=True so body_obs_dim (17) matches the
# eval env; no readiness emission, no scheduled injection during training.
TRAIN_ENV_KWARGS = dict(
    size=12, num_hazards=4, num_resources=5,
    hazard_harm=0.02,
    env_drift_interval=5, env_drift_prob=0.1,
    proximity_harm_scale=0.05,
    proximity_benefit_scale=0.03,
    proximity_approach_threshold=0.15,
    hazard_field_decay=0.5,
    limb_damage_enabled=True,
    damage_increment=0.08, heal_rate=0.01, failure_prob_scale=0.2,
)

# Degrade/recover eval env: readiness emission ON + SD-022 scheduled all-limb
# injection pulses (interval 60, prob 1.0, magnitude 0.8) + healing (0.02/step).
# Post-injection: mean damage ~0.8 -> readiness ~0.2 (DEGRADED, < floor 0.3).
# Over ~60 heal steps damage decays ~0.8*0.98^60 ~= 0.24 -> readiness ~0.76
# (READY, >= floor) before the next pulse. Produces clear degrade + recover windows.
EVAL_ENV_KWARGS = dict(
    size=12, num_hazards=4, num_resources=5,
    hazard_harm=0.02,
    env_drift_interval=50, env_drift_prob=0.0,   # stable layout: isolate the readiness axis
    proximity_harm_scale=0.05,
    proximity_benefit_scale=0.03,
    proximity_approach_threshold=0.15,
    hazard_field_decay=0.5,
    limb_damage_enabled=True,
    damage_increment=0.08, heal_rate=0.02, failure_prob_scale=0.2,
    mech090_readiness_outcome_enabled=True,
    scheduled_limb_damage_enabled=True,
    scheduled_limb_damage_interval=60,
    scheduled_limb_damage_prob=1.0,
    scheduled_limb_damage_magnitude=0.8,
    scheduled_limb_damage_limb_selection="all",
)

# Arm gate configs. forward_readiness controls whether the env-emitted outcome is
# pushed into sense() (it is harmless but pointless to forward when no gate reads it).
ARMS = [
    ("ARM_0_OFF_BASELINE",
     dict(use_mech090_readiness_conjunction=False, use_commit_readiness_gate=False),
     False),
    ("ARM_2_GATED_NAV_COMP_ON",
     dict(use_mech090_readiness_conjunction=True, use_commit_readiness_gate=False),
     True),
    ("ARM_3_GATED_BOTH_ON",
     dict(use_mech090_readiness_conjunction=True, use_commit_readiness_gate=True),
     True),
]


def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _safe_rate(num: int, den: int) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def _build_config(env: CausalGridWorldV2, self_dim: int, world_dim: int,
                  alpha_world: float, gate_flags: Dict) -> REEConfig:
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=0.3,
        reafference_action_dim=env.action_dim,
        limb_damage_enabled=True,
        # across-tick nav_competence axis (REEConfig-level flags)
        use_mech090_readiness_conjunction=bool(gate_flags["use_mech090_readiness_conjunction"]),
        mech090_readiness_floor=MECH090_READINESS_FLOOR,
        commit_readiness_ema_alpha=COMMIT_READINESS_EMA_ALPHA,
    )
    # within-tick decisiveness axis (HeartbeatConfig-level; NOT in from_dims,
    # matches the beta_gate_bistable precedent -- set after the dataclass build).
    config.heartbeat.use_commit_readiness_gate = bool(gate_flags["use_commit_readiness_gate"])
    config.heartbeat.commit_readiness_floor = 0.05
    return config


def _train_agent(agent: REEAgent, env: CausalGridWorldV2, num_episodes: int,
                 steps_per_episode: int, world_dim: int) -> Dict:
    """Train until running_variance collapses to the committed state. Vanilla
    (no R-c gates consulted): readiness is NOT forwarded so the agent commits
    cleanly; the eval arms apply the gate."""
    agent.train()
    optimizer = optim.Adam(list(agent.e1.parameters()), lr=1e-3)
    wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters()), lr=1e-3,
    )
    harm_eval_optimizer = optim.Adam(
        list(agent.e3.harm_eval_head.parameters()), lr=1e-4,
    )

    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)  # readiness NOT forwarded

            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            z_world_curr = latent.z_world.detach()

            action = agent.select_action(candidates, ticks, temperature=1.0)
            if action is None:
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                )
                agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > 2000:
                    wf_buf = wf_buf[-2000:]

            theta_z = agent.theta_buffer.summary()
            (harm_buf_pos if harm_signal < 0 else harm_buf_neg).append(theta_z.detach())
            if len(harm_buf_pos) > 1000: harm_buf_pos = harm_buf_pos[-1000:]
            if len(harm_buf_neg) > 1000: harm_buf_neg = harm_buf_neg[-1000:]

            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                optimizer.zero_grad()
                e1_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                optimizer.step()

            if len(wf_buf) >= 16:
                k = min(32, len(wf_buf))
                idxs = torch.randperm(len(wf_buf))[:k].tolist()
                zw_b = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
                a_b = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
                wf_pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(wf_pred, zw1_b)
                if wf_loss.requires_grad:
                    wf_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e2.world_transition.parameters()) +
                        list(agent.e2.world_action_encoder.parameters()), 1.0,
                    )
                    wf_optimizer.step()
                with torch.no_grad():
                    agent.e3.update_running_variance((wf_pred.detach() - zw1_b).detach())

            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_p = min(16, len(harm_buf_pos))
                k_n = min(16, len(harm_buf_neg))
                pi = torch.randperm(len(harm_buf_pos))[:k_p].tolist()
                ni = torch.randperm(len(harm_buf_neg))[:k_n].tolist()
                zw_b = torch.cat(
                    [harm_buf_pos[i] for i in pi] + [harm_buf_neg[i] for i in ni], dim=0
                )
                target = torch.cat([
                    torch.ones(k_p, 1, device=agent.device),
                    torch.zeros(k_n, 1, device=agent.device),
                ], dim=0)
                pred = agent.e3.harm_eval(zw_b)
                harm_loss = F.mse_loss(pred, target)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e3.harm_eval_head.parameters(), 0.5)
                    harm_eval_optimizer.step()

            z_world_prev = z_world_curr
            action_prev = action.detach()
            if done:
                break

        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            print(
                f"  [train] ep {ep+1}/{num_episodes}"
                f"  running_var={agent.e3._running_variance:.6f}",
                flush=True,
            )

    return {"final_running_variance": float(agent.e3._running_variance)}


def _eval_arm(agent: REEAgent, env: CausalGridWorldV2, num_episodes: int,
              steps_per_episode: int, world_dim: int, forward_readiness: bool,
              train_variance: float, label: str) -> Dict:
    """Run the degrade/recover eval, bucketing each step by the readiness reading
    (before select_action) into READY (>= floor) vs DEGRADED (< floor) and
    recording whether beta elevated (committed) that tick."""
    agent.eval()
    floor = MECH090_READINESS_FLOOR

    committed_ready = total_ready = 0
    committed_degraded = total_degraded = 0
    n_blocks = 0
    n_elevation_blocked = 0
    fatal = 0
    readiness_min = 1.0
    readiness_max = 0.0
    n_committed_capable = 0  # steps where running_variance < commit_threshold

    for _ in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        agent.e3._running_variance = train_variance  # restore committed-capable state
        outcome: Optional[float] = None

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            try:
                with torch.no_grad():
                    latent = agent.sense(
                        obs_body, obs_world,
                        mech090_readiness_outcome=(outcome if forward_readiness else None),
                    )
                    # readiness reading BEFORE select_action determines the bucket
                    if agent.commit_readiness is not None:
                        rdy = float(agent.commit_readiness.get_readiness())
                    else:
                        rdy = 1.0
                    readiness_min = min(readiness_min, rdy)
                    readiness_max = max(readiness_max, rdy)

                    ticks = agent.clock.advance()
                    e1_prior = (
                        agent._e1_tick(latent) if ticks.get("e1_tick", False)
                        else torch.zeros(1, world_dim, device=agent.device)
                    )
                    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                    action = agent.select_action(candidates, ticks, temperature=1.0)

                if action is None:
                    action = _action_to_onehot(
                        random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                    )
                    agent._last_action = action

                committed_step = bool(getattr(agent.beta_gate, "is_elevated", False))
                committed_capable = (
                    agent.e3._running_variance < agent.e3.commit_threshold
                )
                if committed_capable:
                    n_committed_capable += 1

                if rdy >= floor:
                    total_ready += 1
                    if committed_step:
                        committed_ready += 1
                else:
                    total_degraded += 1
                    if committed_step:
                        committed_degraded += 1

                flat_obs, harm_signal, done, info, obs_dict = env.step(action)
                outcome = info.get("mech090_readiness_outcome")

            except Exception:
                fatal += 1
                flat_obs, obs_dict = env.reset()
                outcome = None
                done = True

            if done:
                break

        # accumulate per-episode diagnostic counters before the next reset() zeroes them
        if agent.commit_readiness is not None:
            n_blocks += int(agent.commit_readiness.get_state().get("n_blocks_emitted", 0))
        bg_state = agent.beta_gate.get_state() if hasattr(agent.beta_gate, "get_state") else {}
        n_elevation_blocked += int(bg_state.get("mech090_n_elevation_blocked", 0))

    rate_ready = _safe_rate(committed_ready, total_ready)
    rate_degraded = _safe_rate(committed_degraded, total_degraded)
    print(
        f"\n  [{label}]  committed_rate_ready={rate_ready:.4f} (n={total_ready})"
        f"  committed_rate_degraded={rate_degraded:.4f} (n={total_degraded})"
        f"  n_blocks={n_blocks}  n_elev_blocked={n_elevation_blocked}"
        f"  readiness[min={readiness_min:.3f},max={readiness_max:.3f}]"
        f"  committed_capable={n_committed_capable}  fatal={fatal}",
        flush=True,
    )
    return {
        "committed_rate_ready": rate_ready,
        "committed_rate_degraded": rate_degraded,
        "total_ready": total_ready,
        "total_degraded": total_degraded,
        "committed_ready": committed_ready,
        "committed_degraded": committed_degraded,
        "n_blocks_emitted": n_blocks,
        "n_elevation_blocked": n_elevation_blocked,
        "readiness_min": readiness_min,
        "readiness_max": readiness_max,
        "n_committed_capable": n_committed_capable,
        "fatal_errors": fatal,
    }


def run(seeds: List[int] = None, warmup_episodes: int = 300, eval_episodes: int = 40,
        steps_per_episode: int = 200, alpha_world: float = 0.9,
        self_dim: int = 32, world_dim: int = 32, **kwargs) -> dict:
    if seeds is None:
        seeds = [0, 1]

    print(
        f"[V3-EXQ-630] ARC-029 / MECH-090 across-tick nav_competence (ECOLOGICAL)\n"
        f"  3 arms: OFF_BASELINE / GATED_NAV_COMP_ON / GATED_BOTH_ON\n"
        f"  Eval env: limb_damage + readiness emission + scheduled all-limb injection\n"
        f"  floor={MECH090_READINESS_FLOOR} ema_alpha={COMMIT_READINESS_EMA_ALPHA}\n"
        f"  seeds={seeds} warmup={warmup_episodes} eval={eval_episodes} steps={steps_per_episode}",
        flush=True,
    )

    per_seed: List[Dict] = []
    for seed in seeds:
        print(f"\n{'='*60}\nSeed {seed} Condition base\n{'='*60}", flush=True)
        torch.manual_seed(seed)
        random.seed(seed)

        # Train one vanilla agent (no R-c gates) -> committed-capable.
        train_env = CausalGridWorldV2(seed=seed, **TRAIN_ENV_KWARGS)
        train_cfg = _build_config(
            train_env, self_dim, world_dim, alpha_world,
            {"use_mech090_readiness_conjunction": False, "use_commit_readiness_gate": False},
        )
        trainer = REEAgent(train_cfg)
        train_out = _train_agent(trainer, train_env, warmup_episodes, steps_per_episode, world_dim)
        train_variance = train_out["final_running_variance"]
        committed_capable = train_variance < trainer.e3.commit_threshold
        print(
            f"\n  Post-train running_variance={train_variance:.6f}"
            f"  commit_threshold={trainer.e3.commit_threshold:.4f}"
            f"  committed_capable={committed_capable}",
            flush=True,
        )
        trained_state = trainer.state_dict()

        seed_arm_results: Dict[str, Dict] = {}
        for arm_label, gate_flags, forward_readiness in ARMS:
            # fresh agent per arm with the arm's gate config; shared trained weights.
            eval_env = CausalGridWorldV2(seed=seed + 500, **EVAL_ENV_KWARGS)
            arm_cfg = _build_config(eval_env, self_dim, world_dim, alpha_world, gate_flags)
            arm_agent = REEAgent(arm_cfg)
            arm_agent.load_state_dict(trained_state)
            seed_arm_results[arm_label] = _eval_arm(
                arm_agent, eval_env, eval_episodes, steps_per_episode, world_dim,
                forward_readiness=forward_readiness, train_variance=train_variance,
                label=f"seed{seed}_{arm_label}",
            )

        per_seed.append({
            "seed": seed,
            "train_variance": train_variance,
            "committed_capable": bool(committed_capable),
            "arms": seed_arm_results,
        })

        # per-seed verdict line (seeds x conditions = seeds x 1 verdict lines)
        a0 = seed_arm_results["ARM_0_OFF_BASELINE"]
        a2 = seed_arm_results["ARM_2_GATED_NAV_COMP_ON"]
        seed_c2 = (a2["committed_rate_ready"] - a2["committed_rate_degraded"]) >= SUPPRESS_MARGIN
        seed_c4 = abs(a0["committed_rate_ready"] - a0["committed_rate_degraded"]) < SUPPRESS_MARGIN
        print(f"verdict: {'PASS' if (seed_c2 and seed_c4) else 'FAIL'}", flush=True)

    # ---- aggregate across seeds ----
    def _avg(arm: str, key: str) -> float:
        vals = [s["arms"][arm][key] for s in per_seed]
        return float(sum(vals) / len(vals))

    def _sum(arm: str, key: str) -> int:
        return int(sum(s["arms"][arm][key] for s in per_seed))

    arm2_rate_ready = _avg("ARM_2_GATED_NAV_COMP_ON", "committed_rate_ready")
    arm2_rate_degraded = _avg("ARM_2_GATED_NAV_COMP_ON", "committed_rate_degraded")
    arm0_rate_ready = _avg("ARM_0_OFF_BASELINE", "committed_rate_ready")
    arm0_rate_degraded = _avg("ARM_0_OFF_BASELINE", "committed_rate_degraded")
    arm3_rate_degraded = _avg("ARM_3_GATED_BOTH_ON", "committed_rate_degraded")

    blocks_nav = _sum("ARM_2_GATED_NAV_COMP_ON", "n_blocks_emitted")
    blocks_both = _sum("ARM_3_GATED_BOTH_ON", "n_blocks_emitted")
    total_fatal = sum(
        s["arms"][a]["fatal_errors"] for s in per_seed for a in s["arms"]
    )

    suppress_delta = arm2_rate_ready - arm2_rate_degraded
    baseline_delta = abs(arm0_rate_ready - arm0_rate_degraded)

    c1_pass = (blocks_nav + blocks_both) >= 1
    c2_pass = suppress_delta >= SUPPRESS_MARGIN
    c3_pass = arm2_rate_ready >= ADMIT_FLOOR
    c4_pass = baseline_delta < SUPPRESS_MARGIN
    c5_pass = arm3_rate_degraded <= (arm2_rate_degraded + COMPOSE_EPS)

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass
    status = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    notes = []
    if not c1_pass:
        notes.append(f"C1 FAIL: nav gate never blocked (blocks_nav={blocks_nav} blocks_both={blocks_both}); "
                     f"degraded windows may not have formed or agent not committed-capable -> /diagnose-errors")
    if not c2_pass:
        notes.append(f"C2 FAIL: ARM_2 suppress_delta={suppress_delta:.4f} < {SUPPRESS_MARGIN} "
                     f"(ready={arm2_rate_ready:.4f} degraded={arm2_rate_degraded:.4f}); gate did not suppress on degrade")
    if not c3_pass:
        notes.append(f"C3 FAIL: ARM_2 committed_rate_ready={arm2_rate_ready:.4f} < {ADMIT_FLOOR}: gate permanently "
                     f"suppresses commitment -> substantively different ARC-029 verdict, route /failure-autopsy "
                     f"(do NOT silently re-tune commit_readiness_floor)")
    if not c4_pass:
        notes.append(f"C4 FAIL: ARM_0 baseline_delta={baseline_delta:.4f} >= {SUPPRESS_MARGIN}: commitment is "
                     f"readiness-conditioned even with the gate OFF -> damage->commitment confound -> /diagnose-errors")
    if not c5_pass:
        notes.append(f"C5 FAIL: ARM_3 degraded={arm3_rate_degraded:.4f} > ARM_2 degraded={arm2_rate_degraded:.4f}+{COMPOSE_EPS}: "
                     f"composed conjunction less strict than nav-alone -> /diagnose-errors")

    print(f"\nV3-EXQ-630 final: {status} ({criteria_met}/5 incl C5)", flush=True)
    for n in notes:
        print(f"  {n}", flush=True)

    metrics = {
        "arm2_committed_rate_ready": arm2_rate_ready,
        "arm2_committed_rate_degraded": arm2_rate_degraded,
        "arm2_suppress_delta": suppress_delta,
        "arm0_committed_rate_ready": arm0_rate_ready,
        "arm0_committed_rate_degraded": arm0_rate_degraded,
        "arm0_baseline_delta": baseline_delta,
        "arm3_committed_rate_degraded": arm3_rate_degraded,
        "n_blocks_nav_arm": float(blocks_nav),
        "n_blocks_both_arm": float(blocks_both),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "crit5_pass": 1.0 if c5_pass else 0.0,
        "criteria_met": float(criteria_met),
        "fatal_error_count": float(total_fatal),
    }

    direction = "supports" if all_pass else ("mixed" if criteria_met >= 3 else "weakens")
    # Per-claim: both ARC-029 (committed-mode admitted-when-ready/suppressed-when-not)
    # and MECH-090 (the R-c across-tick gate dynamic) share the run verdict here --
    # the suppress/admit dynamic IS the joint measurement. C3-FAIL is the branch
    # that splits them (routes to /failure-autopsy per the interpretation grid).
    evidence_direction_per_claim = {"ARC-029": direction, "MECH-090": direction}

    failure_section = ""
    if notes:
        failure_section = "\n## Failure / Routing Notes\n\n" + "\n".join(f"- {n}" for n in notes)

    summary_markdown = f"""# V3-EXQ-630 -- ARC-029 / MECH-090: Across-tick nav_competence (ecological)

**Status:** {status}  ({criteria_met}/5 incl C5)
**Claims:** ARC-029, MECH-090
**Design:** 3-arm eval (OFF / NAV_COMP_ON / BOTH_ON) on shared trained weights;
degrade/recover env (SD-022 scheduled all-limb injection + env-emitted readiness).

## Results (averaged across seeds)

| Arm | committed_rate_ready | committed_rate_degraded |
|---|---|---|
| ARM_0 OFF baseline | {arm0_rate_ready:.4f} | {arm0_rate_degraded:.4f} |
| ARM_2 NAV_COMP_ON | {arm2_rate_ready:.4f} | {arm2_rate_degraded:.4f} |
| ARM_3 BOTH_ON | (ready) | {arm3_rate_degraded:.4f} |

| Metric | Value |
|---|---|
| ARM_2 suppress_delta (ready - degraded) | {suppress_delta:.4f} |
| ARM_0 baseline_delta (abs) | {baseline_delta:.4f} |
| nav-gate blocks (NAV / BOTH) | {blocks_nav} / {blocks_both} |

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1 gate fires (>=1 nav block) | {"PASS" if c1_pass else "FAIL"} | nav={blocks_nav} both={blocks_both} |
| C2 suppress-on-degrade (>= {SUPPRESS_MARGIN}) | {"PASS" if c2_pass else "FAIL"} | {suppress_delta:.4f} |
| C3 admit-on-recover (ready >= {ADMIT_FLOOR}) | {"PASS" if c3_pass else "FAIL"} | {arm2_rate_ready:.4f} |
| C4 OFF-baseline null (< {SUPPRESS_MARGIN}) | {"PASS" if c4_pass else "FAIL"} | {baseline_delta:.4f} |
| C5 composed at least as strict | {"PASS" if c5_pass else "FAIL"} | both_deg={arm3_rate_degraded:.4f} nav_deg={arm2_rate_degraded:.4f} |

PASS = C1 AND C2 AND C3 AND C4 -> **{status}**
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": direction,
        "evidence_direction_per_claim": evidence_direction_per_claim,
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "per_seed": per_seed,
        "fatal_error_count": float(total_fatal),
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--eval-eps", type=int, default=40)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--alpha-world", type=float, default=0.9)
    parser.add_argument("--dry-run", action="store_true",
                        help="tiny smoke run (1 seed, 2 warmup, 1 eval ep, 30 steps)")
    args = parser.parse_args()

    if args.dry_run:
        result = run(seeds=[0], warmup_episodes=2, eval_episodes=1, steps_per_episode=30,
                     alpha_world=args.alpha_world)
    else:
        result = run(seeds=args.seeds, warmup_episodes=args.warmup,
                     eval_episodes=args.eval_eps, steps_per_episode=args.steps,
                     alpha_world=args.alpha_world)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["timestamp_utc"] = ts
    result["outcome"] = result["status"]   # canonical outcome field (PASS/FAIL)
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments"
    )
    out_path = write_flat_manifest(
        result,
        out_dir,
        dry_run=args.dry_run,
        config=result.get("config"),
        seeds=None,
        script_path=Path(__file__),
    )

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.5f}", flush=True)

    _outcome_raw = str(result.get("status", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
    )
