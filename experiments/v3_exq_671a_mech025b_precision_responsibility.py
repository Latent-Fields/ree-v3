"""
V3-EXQ-671a -- MECH-025b: Precision-Responsibility Attribution Linkage
(CORRECTED -- residue-accumulation positive-control gate; supersedes V3-EXQ-671)

Claims: MECH-025b

Motivation (2026-06-11, corrected 2026-08-02):
  MECH-025b: "High-precision action mode carries responsibility attribution:
  the precision level at which an action was committed determines the degree
  of ethical accountability assigned to its consequences."

  Decomposed from MECH-025 (2026-04-02). MECH-025 established that doing mode
  produces distinct internal signatures (EXQ-050 PASS). MECH-025b extends this:
  not just binary committed/uncommitted, but DEGREE of precision should modulate
  responsibility weight.

WHY THIS RETEST (root-caused, not a re-run of the same code):
  V3-EXQ-671 FAILed DEGENERATE (failure_autopsy_batch9_2026-06-12, applied to
  claims.yaml MECH-025b evidence_quality_note): C1 (precision<->residue
  correlation) and C2 (high/low-precision residue ratio) were both exactly 0.0
  across 778 nominally-"committed" harm-events. The note attributed this to
  "ResidueField.total_residue never accumulates under committed harm" and
  recommended a positive-control gate on that premise.

  Code-level root-cause this run fixes (verified by reading ree_core/residue/
  field.py, ree_core/agent.py and V3-EXQ-671's own eval loop before writing
  this script -- NOT assumed from the autopsy prose):

  ResidueField.accumulate() / REEAgent.update_residue() are NOT broken --
  300+ other experiment scripts (and StepHarness, experiments/_harness.py,
  landed 2026-05-08 specifically to prevent this bug class -- "Skipping
  [update_residue] leaves precision pinned and the residue field empty (root
  cause of EXQ-530 / EXQ-536 incidents too)") call them successfully every
  waking tick. V3-EXQ-671's own `_eval_precision_responsibility()` NEVER once
  calls `agent.update_residue(...)` -- it reads `env.step()`'s harm_signal and
  `agent.residue_field.total_residue` but never feeds the former into the
  latter's only accumulation path. total_residue is therefore mathematically
  guaranteed to stay 0.0 for the whole eval loop, regardless of precision or
  harm dynamics -- an instrument defect, not a substrate ceiling. (The same
  defect class independently recurred at V3-EXQ-466d, autopsied 2026-06-24,
  ~2 weeks after 671 and >1 month after StepHarness existed to prevent it --
  see WORKSPACE_STATE.md 2026-06-24T22:11Z.)

  Two further script-side defects compounded the degenerate read and are also
  fixed here (same "corrected implementation, same question" scope -- CLAUDE.md
  EXQ lettering convention):

  (a) `is_committed = agent._committed_candidates is not None` does NOT measure
      BetaGate/E3 commitment. `_committed_candidates` (agent.py:5116) is a
      multi-rate-clock CANDIDATE CACHE set every time `generate_trajectories()`
      runs an e3_tick and reused between ticks (agent.py:5178-5192) -- it is
      non-None almost immediately after warmup and stays non-None essentially
      always. The actual commitment signal is `SelectionResult.committed`
      (e3_selector.py:130, "whether commitment threshold was met"), returned by
      the very `agent.e3.select(...)` call 671 already makes but never reads
      the field of. V3-EXQ-818 (ARC-016's own eval-derived-threshold retest,
      PASS 2026-07-25) uses exactly this field (`res.committed`) as its
      commit-rate DV -- this script follows that precedent rather than
      `_committed_candidates`.
  (b) The eval-sample filter `abs(harm_signal) > 1e-6` pooled BENEFIT ticks
      (harm_signal > 0) in with harm ticks. `update_residue` only ever calls
      `ResidueField.accumulate()` on `harm_signal < 0` (agent.py:8977); benefit
      ticks route to a disjoint `accumulate_benefit()` terrain and can never
      produce a residue delta. Pooling them dilutes the correlation/ratio
      samples with entries whose residue_delta is 0 by construction independent
      of the fix in (root-cause) above. Filter tightened to `harm_signal < 0`.

  A fourth correction, `owned=`, is new relative to 671 (which always defaulted
  `owned=True` implicitly by never passing the harm through `update_residue` at
  all): `agent.update_residue(..., owned=...)` is sourced from
  `info["transition_type"] == "agent_caused_hazard"`
  (ree_core/environment/causal_grid_world.py; the same ground-truth signal
  SD-003's own evidence_quality_note calls "agent_caused_hazard event counts").
  `ResidueField.accumulate()` is a no-op when `owned=False` (agent.py:8983) by
  design (MECH-025b's own claim text: responsibility requires the agent "could
  have done otherwise" -- ambient/environment-caused harm the agent did not
  cause should not accrue residue). Sample collection additionally requires
  `owned=True`, matching what can physically produce a nonzero residue delta.

  Experimental design (methodologically UNCHANGED from 671 -- same env, same
  train loop, same 6 PASS criteria, same thresholds):
    - Train agent with functional E2 (world_forward) + E3 (harm_eval + precision)
    - During eval, track precision at each committed step (E3.current_precision)
    - Measure residue accumulation per step (ResidueField changes)
    - Compare residue-per-harm in high-precision vs low-precision regimes

  Key metrics:
    precision_residue_correlation: Pearson r(precision, residue_accumulated)
    high_precision_residue_ratio: mean residue/harm when precision > median
                                   vs mean residue/harm when precision <= median

  PASS criteria (ALL must hold):
    C1: precision_residue_correlation > 0.15  (positive correlation exists)
    C2: high_precision_residue_ratio > 1.1    (high-precision steps accumulate
                                                proportionally more residue)
    C3: committed_step_count >= 20             (sufficient samples)
    C4: world_forward_r2 > 0.05                (E2 attribution functional)
    C5: harm_pred_std > 0.01                   (E3 not collapsed)
    C6: No fatal errors

POSITIVE-CONTROL GATE (new in 671a, this IS the autopsy's requested retest):
  Before C1/C2 are read as evidence, a pre-registered P0 readiness check
  (experiments._metrics.p0_readiness_gate / P0NotReady) asserts that
  ResidueField.total_residue actually moved under committed, agent-owned harm
  during THIS run: sum(residue_delta_samples) must clear a small positive
  floor. If unmet (e.g. some other, still-undiscovered path also pins residue
  at 0), the run self-routes FAIL / non_contributory /
  substrate_not_ready_requeue instead of reporting a misleading C1/C2 read --
  exactly the non-vacuity discipline degeneracy_selfreport_lint requires for
  a claim-pressing discriminative criterion (see failure_autopsy_batch9_2026-
  06-12 and validate_experiments.py's own docstring naming V3-EXQ-671 as a
  motivating case). A second, independent net --
  experiments._metrics.check_degeneracy on the residue_delta_samples spread --
  catches the case where residue moves but is pinned/constant across the
  comparison cells C1 actually reads.
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._metrics import (  # noqa: E402
    check_degeneracy,
    p0_readiness_gate,
    P0NotReady,
)


EXPERIMENT_TYPE = "v3_exq_671a_mech025b_precision_responsibility"
CLAIM_IDS = ["MECH-025b"]
SUPERSEDES = "v3_exq_671_mech025b_precision_responsibility_20260611T215625Z_v3"
RESIDUE_FLOOR = 1e-6  # P0: sum(residue_delta_samples) must clear this to be non-vacuous


def _action_to_onehot(action_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, action_idx] = 1.0
    return v


def _mean_safe(lst: List[float]) -> float:
    return float(sum(lst) / len(lst)) if lst else 0.0


def _pearson_correlation(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient between two lists."""
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return 0.0
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denom_x = sum((x[i] - mean_x) ** 2 for i in range(n))
    denom_y = sum((y[i] - mean_y) ** 2 for i in range(n))
    if denom_x < 1e-9 or denom_y < 1e-9:
        return 0.0
    return numerator / ((denom_x * denom_y) ** 0.5)


def _train(
    agent: REEAgent,
    env: CausalGridWorldV2,
    optimizer: optim.Optimizer,
    wf_optimizer: optim.Optimizer,
    harm_eval_optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
) -> Dict:
    """Standard full-pipeline training to get functional E3 + E2.world_forward.

    UNCHANGED from V3-EXQ-671: C4 (world_forward_r2) and C5 (harm_pred_std)
    were NOT degenerate in 671 (0.9324 / 0.1948 respectively) -- the defect
    was isolated to the eval loop's residue wiring, not training.
    """
    agent.train()
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    total_harm = 0
    e3_tick_total = 0

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        z_self_prev: Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            theta_z = agent.theta_buffer.summary()
            z_world_curr = latent.z_world.detach()

            if ticks.get("e3_tick", False) and candidates:
                e3_tick_total += 1
                result = agent.e3.select(candidates, temperature=1.0)
                action = result.selected_action.detach()
                agent._last_action = action
            else:
                action = agent._last_action
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

            if harm_signal < 0:
                total_harm += 1
                harm_buf_pos.append(theta_z.detach())
                if len(harm_buf_pos) > 1000:
                    harm_buf_pos = harm_buf_pos[-1000:]
            else:
                harm_buf_neg.append(theta_z.detach())
                if len(harm_buf_neg) > 1000:
                    harm_buf_neg = harm_buf_neg[-1000:]

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
                wf_loss = F.mse_loss(agent.e2.world_forward(zw_b, a_b), zw1_b)
                if wf_loss.requires_grad:
                    wf_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e2.world_transition.parameters())
                        + list(agent.e2.world_action_encoder.parameters()),
                        1.0,
                    )
                    wf_optimizer.step()

            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_p = min(16, len(harm_buf_pos))
                k_n = min(16, len(harm_buf_neg))
                pi = torch.randperm(len(harm_buf_pos))[:k_p].tolist()
                ni = torch.randperm(len(harm_buf_neg))[:k_n].tolist()
                zw_b = torch.cat(
                    [harm_buf_pos[i] for i in pi] + [harm_buf_neg[i] for i in ni], dim=0
                )
                target = torch.cat(
                    [
                        torch.ones(k_p, 1, device=agent.device),
                        torch.zeros(k_n, 1, device=agent.device),
                    ],
                    dim=0,
                )
                pred = agent.e3.harm_eval(zw_b)
                harm_loss = F.mse_loss(pred, target)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e3.harm_eval_head.parameters(), 0.5)
                    harm_eval_optimizer.step()

            z_world_prev = z_world_curr
            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            if done:
                break

        if (ep + 1) % 100 == 0 or ep == num_episodes - 1:
            print(
                f"  [train] ep {ep+1}/{num_episodes}  harm={total_harm}"
                f"  e3_ticks={e3_tick_total}",
                flush=True,
            )

    return {"total_harm": total_harm, "wf_buf": wf_buf, "e3_tick_total": e3_tick_total}


def _compute_world_forward_r2(agent: REEAgent, wf_buf: List, n_test: int = 200) -> float:
    if len(wf_buf) < n_test:
        return 0.0
    idxs = list(range(len(wf_buf) - n_test, len(wf_buf)))
    with torch.no_grad():
        zw = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
        a = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
        zw1 = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
        pred = agent.e2.world_forward(zw, a)
        ss_res = ((zw1 - pred) ** 2).sum()
        ss_tot = ((zw1 - zw1.mean(dim=0, keepdim=True)) ** 2).sum()
    return float((1 - ss_res / (ss_tot + 1e-8)).item())


def _eval_precision_responsibility(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
) -> Dict:
    """
    Probe precision-responsibility linkage by tracking:
      - precision at each committed step (E3.current_precision)
      - residue accumulated per step (ResidueField.total_residue delta)
      - harm magnitude per step

    Tests MECH-025b: high-precision actions should accumulate proportionally
    more residue (responsibility weight) than low-precision actions.

    CORRECTED vs V3-EXQ-671 (see module docstring for the root-cause trace):
      (1) calls agent.update_residue(...) after every env.step() -- the call
          671 never made, which is why total_residue never moved.
      (2) is_committed is sourced from SelectionResult.committed (the real
          BetaGate/E3 commitment-threshold outcome), not from
          agent._committed_candidates (a candidate CACHE flag, non-None almost
          always regardless of commitment).
      (3) owned= is sourced from info["transition_type"]=="agent_caused_hazard"
          -- only agent-caused harm can carry responsibility residue.
      (4) the eval-sample filter is harm_signal < 0 (genuine harm) AND owned,
          not abs(harm_signal) > 1e-6 (which also pooled in benefit ticks that
          can never produce a residue delta).
    """
    agent.eval()
    precision_samples: List[float] = []
    residue_delta_samples: List[float] = []
    harm_magnitude_samples: List[float] = []
    all_harm_preds: List[float] = []
    fatal = 0
    committed_step_count = 0
    total_residue_start = float(agent.residue_field.total_residue.item())

    for _ in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        held_committed = False
        residue_prev = float(agent.residue_field.total_residue.item())

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                if z_self_prev is not None and action_prev is not None:
                    agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent)
                    if ticks.get("e1_tick", False)
                    else torch.zeros(1, world_dim, device=agent.device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            try:
                if ticks.get("e3_tick", False) and candidates:
                    with torch.no_grad():
                        result = agent.e3.select(candidates, temperature=1.0)
                        action = result.selected_action.detach()
                        agent._last_action = action
                        # (2) real commitment signal, not the candidate cache.
                        held_committed = bool(result.committed)
                else:
                    action = agent._last_action
                    if action is None:
                        action = _action_to_onehot(
                            random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                        )
                        agent._last_action = action

                is_committed = held_committed

                # Track precision at committed steps
                if is_committed:
                    with torch.no_grad():
                        current_precision = agent.e3.current_precision
                        h_pred = float(agent.e3.harm_eval(latent.z_world).item())
                        all_harm_preds.append(h_pred)

                # Execute action
                flat_obs, harm_signal, done, info, obs_dict = env.step(action)

                # (3) agent-caused vs environment-caused: only agent-caused harm
                # is "owned" and can accrue responsibility residue.
                owned = info.get("transition_type") == "agent_caused_hazard"

                # (1) THE FIX: feed the harm signal into the canonical residue
                # accumulation path. Without this call total_residue can never
                # move -- exactly the V3-EXQ-671 degenerate-read root cause.
                agent.update_residue(
                    harm_signal=float(harm_signal),
                    world_delta=None,
                    hypothesis_tag=False,
                    owned=owned,
                )
                residue_curr = float(agent.residue_field.total_residue.item())
                residue_delta = residue_curr - residue_prev
                residue_prev = residue_curr

                # (4) Collect samples only where residue CAN move: committed,
                # agent-owned, genuine harm (harm_signal < 0).
                if is_committed and owned and harm_signal < -1e-6:
                    committed_step_count += 1
                    precision_samples.append(current_precision)
                    residue_delta_samples.append(residue_delta)
                    harm_magnitude_samples.append(abs(harm_signal))

            except Exception:
                fatal += 1
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                )
                agent._last_action = action
                flat_obs, harm_signal, done, info, obs_dict = env.step(action)
                residue_prev = float(agent.residue_field.total_residue.item())

            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            if done:
                break

    total_residue_end = float(agent.residue_field.total_residue.item())
    total_residue_accumulated = total_residue_end - total_residue_start

    # Compute correlation and ratio
    precision_residue_correlation = _pearson_correlation(precision_samples, residue_delta_samples)

    # High vs low precision comparison
    if len(precision_samples) >= 4:
        median_precision = sorted(precision_samples)[len(precision_samples) // 2]
        high_prec_residue = []
        low_prec_residue = []
        for i in range(len(precision_samples)):
            if precision_samples[i] > median_precision:
                high_prec_residue.append(residue_delta_samples[i] / max(harm_magnitude_samples[i], 1e-6))
            else:
                low_prec_residue.append(residue_delta_samples[i] / max(harm_magnitude_samples[i], 1e-6))

        mean_high = _mean_safe(high_prec_residue)
        mean_low = _mean_safe(low_prec_residue)
        high_precision_residue_ratio = mean_high / max(mean_low, 1e-6)
    else:
        high_precision_residue_ratio = 0.0

    harm_pred_std = (
        float(torch.tensor(all_harm_preds).std().item()) if len(all_harm_preds) > 1 else 0.0
    )

    print(
        f"  precision_residue_correlation={precision_residue_correlation:.4f}"
        f"  high_precision_residue_ratio={high_precision_residue_ratio:.4f}"
        f"  committed_steps={committed_step_count}"
        f"  total_residue_accumulated={total_residue_accumulated:.6f}",
        flush=True,
    )

    return {
        "precision_residue_correlation": precision_residue_correlation,
        "high_precision_residue_ratio": high_precision_residue_ratio,
        "committed_step_count": committed_step_count,
        "harm_pred_std": harm_pred_std,
        "fatal_errors": fatal,
        "n_samples": len(precision_samples),
        "residue_delta_samples": residue_delta_samples,
        "total_residue_accumulated": total_residue_accumulated,
    }


def run(
    seed: int = 0,
    warmup_episodes: int = 500,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_scale: float = 0.05,
    lr: float = 1e-3,
    self_dim: int = 32,
    world_dim: int = 32,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorldV2(
        seed=seed,
        size=12,
        num_hazards=4,
        num_resources=5,
        hazard_harm=harm_scale,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=proximity_scale,
        proximity_benefit_scale=proximity_scale * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=env.action_dim,
    )
    agent = REEAgent(config)

    optimizer = optim.Adam(list(agent.e1.parameters()), lr=lr)
    wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters()),
        lr=1e-3,
    )
    harm_eval_optimizer = optim.Adam(
        list(agent.e3.harm_eval_head.parameters()),
        lr=1e-4,
    )

    print(
        f"[V3-EXQ-671a] MECH-025b: Precision-Responsibility Attribution "
        f"(residue-accumulation positive-control gate)\n"
        f"  warmup={warmup_episodes}  eval={eval_episodes}  alpha_world={alpha_world}",
        flush=True,
    )

    train_out = _train(
        agent,
        env,
        optimizer,
        wf_optimizer,
        harm_eval_optimizer,
        warmup_episodes,
        steps_per_episode,
        world_dim,
    )
    world_forward_r2 = _compute_world_forward_r2(agent, train_out["wf_buf"])
    print(f"  world_forward_r2: {world_forward_r2:.4f}", flush=True)

    print(f"\n[V3-EXQ-671a] Eval -- probing precision-responsibility linkage...", flush=True)
    eval_out = _eval_precision_responsibility(
        agent, env, eval_episodes, steps_per_episode, world_dim
    )

    # --- POSITIVE-CONTROL GATE (new in 671a) --------------------------------
    # P0: residue must actually have moved under committed, agent-owned harm
    # in THIS run before C1/C2 are read as evidence at all.
    readiness_checks = [
        {
            "name": "residue_accumulates_under_committed_harm",
            "measured": sum(eval_out["residue_delta_samples"]),
            "threshold": RESIDUE_FLOOR,
            "direction": "lower",
            "control": (
                "sum of ResidueField.total_residue deltas across committed, "
                "agent-owned harm-events during eval -- must be > 0 for the "
                "C1/C2 precision<->residue readout to be non-vacuous "
                "(failure_autopsy_batch9_2026-06-12 root cause)."
            ),
        },
    ]
    ready = True
    preconditions = []
    try:
        preconditions = p0_readiness_gate(readiness_checks)
    except P0NotReady as e:
        preconditions = e.preconditions
        ready = False

    # Secondary net: even if residue moved, the C1 discriminative metric
    # (the per-event residue deltas C1's correlation is computed over) must
    # have usable spread, not be pinned at a single value.
    degeneracy = check_degeneracy(
        {"C1_residue_delta_samples": eval_out["residue_delta_samples"]}
    )
    non_degenerate = degeneracy["non_degenerate"]

    # PASS / FAIL (criteria unchanged from V3-EXQ-671)
    c1_pass = eval_out["precision_residue_correlation"] > 0.15
    c2_pass = eval_out["high_precision_residue_ratio"] > 1.1
    c3_pass = eval_out["committed_step_count"] >= 20
    c4_pass = world_forward_r2 > 0.05
    c5_pass = eval_out["harm_pred_std"] > 0.01
    c6_pass = eval_out["fatal_errors"] == 0

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass and c6_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass, c6_pass])

    if not ready:
        status = "FAIL"
        evidence_direction = "non_contributory"
        label = "substrate_not_ready_requeue"
    elif not non_degenerate:
        status = "FAIL"
        evidence_direction = "non_contributory"
        label = "c1_residue_delta_degenerate_vacuous_test"
    elif all_pass:
        status = "PASS"
        evidence_direction = "supports"
        label = "precision_modulates_residue_responsibility_weight"
    else:
        status = "FAIL"
        evidence_direction = "mixed" if criteria_met >= 4 else "weakens"
        label = "precision_does_not_modulate_residue_responsibility_weight"

    failure_notes = []
    if not ready:
        failure_notes.append(
            "P0 FAIL: residue_accumulates_under_committed_harm unmet -- "
            f"sum(residue_delta_samples)={sum(eval_out['residue_delta_samples']):.6g} "
            f"<= floor={RESIDUE_FLOOR:g} (substrate_not_ready_requeue)"
        )
    if ready and not non_degenerate:
        failure_notes.append(
            f"DEGENERATE: {degeneracy['degeneracy_reason']}"
        )
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: precision_residue_correlation="
            f"{eval_out['precision_residue_correlation']:.4f} <= 0.15"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: high_precision_residue_ratio="
            f"{eval_out['high_precision_residue_ratio']:.4f} <= 1.1"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: committed_step_count={eval_out['committed_step_count']} < 20"
        )
    if not c4_pass:
        failure_notes.append(f"C4 FAIL: world_forward_r2={world_forward_r2:.4f} <= 0.05")
    if not c5_pass:
        failure_notes.append(
            f"C5 FAIL: harm_pred_std={eval_out['harm_pred_std']:.4f} <= 0.01"
        )
    if not c6_pass:
        failure_notes.append(f"C6 FAIL: fatal_errors={eval_out['fatal_errors']}")

    print(f"\nV3-EXQ-671a verdict: {status}  label={label}  ({criteria_met}/6)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "precision_residue_correlation": float(eval_out["precision_residue_correlation"]),
        "high_precision_residue_ratio": float(eval_out["high_precision_residue_ratio"]),
        "committed_step_count": float(eval_out["committed_step_count"]),
        "harm_pred_std": float(eval_out["harm_pred_std"]),
        "world_forward_r2": float(world_forward_r2),
        "e3_tick_total": float(train_out["e3_tick_total"]),
        "total_harm_train": float(train_out["total_harm"]),
        "fatal_error_count": float(eval_out["fatal_errors"]),
        "n_samples": float(eval_out["n_samples"]),
        "total_residue_accumulated": float(eval_out["total_residue_accumulated"]),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "crit5_pass": 1.0 if c5_pass else 0.0,
        "crit6_pass": 1.0 if c6_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-671a -- MECH-025b: Precision-Responsibility Attribution (corrected)

**Status:** {status}
**Label:** {label}
**Claim:** MECH-025b -- high-precision action mode carries responsibility attribution
**Prerequisite:** MECH-025 (doing mode produces internal signature)
**Supersedes:** V3-EXQ-671 (degenerate: residue never accumulated -- instrument defect)
**alpha_world:** {alpha_world}
**Warmup:** {warmup_episodes} eps  |  Eval: {eval_episodes} eps
**Seed:** {seed}

## Motivation

MECH-025b tests the philosophical bridge: does precision level modulate
responsibility weight? Actions committed at higher precision should accumulate
proportionally more residue (ethical accountability) than low-precision actions,
because high-precision implies the agent had finer discrimination capacity.

## Positive-Control Gate

| Check | Measured | Floor | Met |
|---|---|---|---|
| residue_accumulates_under_committed_harm | {sum(eval_out['residue_delta_samples']):.6g} | {RESIDUE_FLOOR:g} | {ready} |

Total residue accumulated during eval (lifetime delta): {eval_out['total_residue_accumulated']:.6f}
C1 non-degeneracy (residue_delta_samples spread): {non_degenerate} ({degeneracy['degeneracy_reason'] or 'ok'})

## Results

| Metric | Value |
|--------|-------|
| Precision-Residue Correlation | {eval_out['precision_residue_correlation']:.4f} |
| High/Low Precision Residue Ratio | {eval_out['high_precision_residue_ratio']:.4f} |
| Committed Steps Sampled | {eval_out['committed_step_count']} |
| World Forward R2 | {world_forward_r2:.4f} |
| Harm Pred Std | {eval_out['harm_pred_std']:.4f} |

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: precision_residue_correlation > 0.15 | {"PASS" if c1_pass else "FAIL"} | {eval_out['precision_residue_correlation']:.4f} |
| C2: high_precision_residue_ratio > 1.1 | {"PASS" if c2_pass else "FAIL"} | {eval_out['high_precision_residue_ratio']:.4f} |
| C3: committed_step_count >= 20 | {"PASS" if c3_pass else "FAIL"} | {eval_out['committed_step_count']} |
| C4: world_forward_r2 > 0.05 | {"PASS" if c4_pass else "FAIL"} | {world_forward_r2:.4f} |
| C5: harm_pred_std > 0.01 | {"PASS" if c5_pass else "FAIL"} | {eval_out['harm_pred_std']:.4f} |
| C6: No fatal errors | {"PASS" if c6_pass else "FAIL"} | {eval_out['fatal_errors']} |

Criteria met: {criteria_met}/6 -> **{status}** (label: {label})
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "supersedes": SUPERSEDES,
        "evidence_direction": evidence_direction,
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": eval_out["fatal_errors"],
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": {"C1": non_degenerate},
            "criteria": [
                {"name": "C1_precision_residue_correlation", "load_bearing": True, "passed": bool(c1_pass)},
                {"name": "C2_high_precision_residue_ratio", "load_bearing": True, "passed": bool(c2_pass)},
                {"name": "C3_committed_step_count", "load_bearing": False, "passed": bool(c3_pass)},
                {"name": "C4_world_forward_r2", "load_bearing": False, "passed": bool(c4_pass)},
                {"name": "C5_harm_pred_std", "load_bearing": False, "passed": bool(c5_pass)},
                {"name": "C6_no_fatal_errors", "load_bearing": False, "passed": bool(c6_pass)},
            ],
        },
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy["degeneracy_reason"],
    }


if __name__ == "__main__":
    import argparse
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--eval-eps", type=int, default=50)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--alpha-world", type=float, default=0.9)
    parser.add_argument("--harm-scale", type=float, default=0.02)
    parser.add_argument("--dry-run", action="store_true", help="Quick validation run")
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] Quick validation mode", flush=True)
        args.warmup = 5
        args.eval_eps = 2
        args.steps = 50

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        harm_scale=args.harm_scale,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    out_path = None
    if not args.dry_run:
        out_path = write_flat_manifest(
            result,
            out_dir=None,
            dry_run=False,
            config=result.get("config"),
            seeds=None,
            script_path=Path(__file__),
        )

        print(f"\nResult written to: {out_path}", flush=True)

    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}", flush=True)

    if not args.dry_run:
        _outcome_raw = str(result.get("status", "FAIL")).upper()
        emit_outcome(
            outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
            manifest_path=out_path,
        )
