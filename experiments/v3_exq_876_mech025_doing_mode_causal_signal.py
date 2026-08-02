"""
V3-EXQ-876 -- MECH-025: Action-Doing Mode Causal Signal
(CORRECTED commitment instrument -- direct V3 confirmation of MECH-025 itself)

Claims: MECH-025

Motivation (2026-08-02):
  MECH-025: "Action mode prioritizes short-horizon, high-precision control via
  context-dependent precision modulation." Operationalized (V3-EXQ-050 lineage)
  as: committed ("doing") steps should show a distinct, LARGER causal signature
  than uncommitted ("exploring") steps --
      causal_sig = E3(E2.world_forward(z_world, a_actual))
                 - E3(E2.world_forward(z_world, a_cf))
      doing_mode_delta = mean|causal_sig|_committed - mean|causal_sig|_uncommitted

  MECH-025 currently carries `evidence: None` in claims.yaml (verified fresh
  2026-08-02) -- no V3 confirmation has ever landed, only three invalidated V3
  attempts, none of which reached a fair read of the causal-signature contrast:

    V3-EXQ-050  (2026-03-19): committed_step_count=7694, uncommitted=0.
    V3-EXQ-050b (2026-03-20): committed_step_count=7859, uncommitted=8.
    V3-EXQ-199  (2026-04-02): committed_step_count=0 (both seeds 42, 123),
                              uncommitted=4929/... claims.yaml's own note:
                              "BreathOscillator creates uncommitted windows but
                              agent never enters committed state during
                              evaluation -- cannot measure doing-mode causal
                              signal without committed steps."

  WHY THIS RETEST (root-caused by reading ree_core source, not assumed from
  claims.yaml prose -- same discipline the 2026-08-02 MECH-025b retest
  (V3-EXQ-671a, failure_autopsy_V3-EXQ-671a_2026-08-02.md) used on the SAME
  substrate and got n=29 genuine committed harm-events where the prior
  MECH-025b attempt (671) had committed_step_count effectively 0 too):

  All three prior MECH-025 scripts read commitment from a field that is
  either a stale CACHE or gets TORN DOWN before the eval loop reads it --
  never the actual per-tick commitment outcome:

  (a) V3-EXQ-050 / 050b: `is_committed = agent._committed_candidates is not
      None` (agent.py:2646/5116). `_committed_candidates` is a multi-rate-
      clock CANDIDATE CACHE set every `generate_trajectories()` e3_tick and
      simply reused between ticks (agent.py:5178-5192) -- non-None almost
      immediately after warmup and stays non-None essentially always. This is
      why 050/050b read >99.8% of steps as "committed": the cache, not the
      real commit decision. V3-EXQ-671's identical bug (autopsied
      failure_autopsy_batch9_2026-06-12, fixed in 671a 2026-08-02) is exactly
      this class, on the same field.

  (b) V3-EXQ-199: `is_committed = agent.e3._committed_trajectory is not
      None`. e3_selector.py:338-350 documents that `_committed_trajectory` is
      "set ONLY under the F-driven `if committed:` path and torn down every
      tick" by the post-action update that runs before the NEXT tick's read
      -- so a script checking it is reading a value that has already been
      reset to None by the time it looks. This is consistent with 199's
      committed_step_count=0 in both seeds: the flag is essentially never
      observably non-None from outside the tick that sets it.

  The correct, non-torn-down, non-cached field is `SelectionResult.committed`
  (e3_selector.py:114-132: "committed: whether commitment threshold was
  met"), returned DIRECTLY by the same `agent.e3.select(...)` call every one
  of these scripts already makes but never read the field of. This is the
  exact fix V3-EXQ-818 (ARC-016 eval-derived-threshold retest, PASS
  2026-07-25) and V3-EXQ-671a (MECH-025b, this same 2026-08-02 session
  batch) both use as their commit-rate DV.

  A SECOND, independently-diagnosed defect compounds (a)/(b): none of
  050/050b/199's eval loops call `agent.update_residue(...)` after
  `env.step()`. Per experiments/_harness.py's StepHarness invariant 3 and its
  own incident history (EXQ-530/536): `update_residue` is the CANONICAL path
  that drives `e3.post_action_update`, which updates `_running_variance` --
  and `committed = running_variance < commit_threshold` (e3_selector.py:15)
  is computed FROM that running variance. Skipping update_residue during eval
  leaves running_variance frozen at whatever value training left it at, so
  the commit decision itself is non-responsive for the whole eval run
  regardless of which field you read it from -- independently sufficient to
  explain why 050b's THRESHOLD CALIBRATION fix (2x mean training variance)
  still produced 7859/8 (99.9% committed): a frozen post-training variance
  sits below almost any calibrated multiple of itself once training has
  converged (this is literally V3-EXQ-199's own stated reason for abandoning
  050b's approach -- "after sufficient training, variance converges below any
  fixed multiple of the mean, yielding permanent commitment").

  Both defects are INSTRUMENT/MEASUREMENT defects (wrong field read; missing
  canonical per-tick call), not evidence of a substrate ceiling -- mirroring
  exactly the classification V3-EXQ-671's identical cache-field bug received
  (failure_autopsy_batch9_2026-06-12: "instrument defect, not substrate
  ceiling"). Re-derive-brake check (2026-08-02, /queue-experiment Step 2.5b):
  zero `failure_autopsy_*.json` artifacts tag MECH-025 at all (only
  failure_autopsy_V3-EXQ-671a_2026-08-02 targets MECH-025b) -- the formal
  re-derive brake data structure has nothing to count for MECH-025, so it
  does not fire regardless of how 050/050b/199's un-autopsied FAILs would be
  categorized.

  THREE FIXES applied in this retest (all root-caused above, applied
  together -- omitting any one reproduces a documented prior failure mode):
    1. Read commitment from `result.committed` (SelectionResult), never
       `_committed_candidates` or `_committed_trajectory`.
    2. Call `agent.update_residue(harm_signal=..., world_delta=None,
       hypothesis_tag=False, owned=True)` after every `env.step()` in eval,
       so `running_variance` (and hence the commit decision) is genuinely
       live for the whole eval run, not frozen at its post-training value.
    3. BreathOscillator (MECH-108, config.heartbeat.breath_period=50,
       breath_sweep_amplitude=0.30, breath_sweep_duration=10 -- same explicit
       values V3-EXQ-199 used, now passed directly as REEConfig.from_dims()
       kwargs rather than a post-construction patch) to guarantee periodic
       threshold-reduction windows, so genuine uncommitted samples exist even
       after running_variance converges low post-training (the mechanism
       199 targeted; its failure was defect (b) above, not the oscillator).
       `sweep_threshold_reduction` is passed straight into `agent.e3.select`,
       the same argument select_action itself forwards
       (agent.py:6084: `sweep_reduction = self.clock.sweep_amplitude if
       self.clock.sweep_active else 0.0`), so this changes the SAME decision
       that produces `result.committed` -- not a parallel, disconnected knob.

  Multi-seed (5 seeds) per re-derive-brake / queue-experiment Step 2.5b
  guidance for a claim regaining a fresh test after repeated invalid prior
  attempts, and per the sibling MECH-025b retest's own recommendation
  ("recording a direct V3 confirmation of MECH-025's own signature").

POSITIVE-CONTROL GATE (the exact quantity that broke this lineage 3 times):
  Before C1 (doing_mode_delta) is read as evidence, per-seed P0 preconditions
  assert BOTH regimes were genuinely sampled this run:
    committed_regime_sampled:   committed_step_count   >= 20
    uncommitted_regime_sampled: uncommitted_step_count >= 20
  Unmet -> that seed self-routes FAIL / non_contributory /
  substrate_not_ready_requeue instead of reporting a misleading C1 read.
  A secondary check_degeneracy() net additionally requires both the
  committed-step and uncommitted-step causal_sig sample lists to have real
  spread (not pinned at a single value).

Env: CausalGridWorldV2(size=6, num_hazards=4, num_resources=5, nav_bias=0.45)
  -- same tuned env as V3-EXQ-199 (denser hazard exposure per step than
  050/050b's size=12 default).
use_event_classifier=True (SD-009) -- same as V3-EXQ-199; the world model's
  event-responsiveness is directly load-bearing for causal_sig quality.

PASS criteria (ALL must hold, per seed):
  C1: doing_mode_delta > 0.002       (committed has higher |causal sig|)
  C2: committed_step_count >= 20     (doing mode genuinely entered)
  C3: uncommitted_step_count >= 20   (exploring mode genuinely entered)
  C4: world_forward_r2 > 0.05        (E2 world model functional)
  C5: harm_pred_std > 0.01           (E3 not collapsed)
  C6: No fatal errors

Overall verdict: PASS iff ALL seeds PASS (matching V3-EXQ-199's multi-seed
convention); evidence_direction from the aggregate criteria-met fraction.
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


EXPERIMENT_TYPE = "v3_exq_876_mech025_doing_mode_causal_signal"
CLAIM_IDS = ["MECH-025"]
# Same design (causal-signature contrast + BreathOscillator) as V3-EXQ-199,
# corrected for the commitment-field-read + missing-update_residue defects
# documented in the module docstring. Recorded for governance lineage even
# though this run was assigned a new queue number rather than a lettered
# "V3-EXQ-199a" (queue_id was pre-allocated by the orchestrating session).
SUPERSEDES = "v3_exq_199_mech025_doing_mode_breath"
COMMITTED_FLOOR = 20    # P0: committed_step_count must clear this
UNCOMMITTED_FLOOR = 20  # P0: uncommitted_step_count must clear this


def _action_to_onehot(action_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, action_idx] = 1.0
    return v


def _mean_safe(lst: List[float]) -> float:
    return float(sum(lst) / len(lst)) if lst else 0.0


def _train(
    agent: REEAgent,
    env: CausalGridWorldV2,
    optimizer: optim.Optimizer,
    wf_optimizer: optim.Optimizer,
    harm_eval_optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    nav_bias: float,
) -> Dict:
    """Standard full-pipeline training to get a functional E3 + E2.world_forward.

    UNCHANGED shape from V3-EXQ-050/199/671a: training does not call
    update_residue (residue/running_variance liveness is only required in
    EVAL, where the commit contrast is actually read; C4/C5 do not depend on
    it and all three prior scripts reached world_forward_r2 > 0.9 without it).
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

            # nav_bias (V3-EXQ-199 pattern): with probability nav_bias, override
            # action to move toward the nearest hazard -- increases harm
            # exposure for the E3.harm_eval training signal.
            if random.random() < nav_bias:
                agent_pos = getattr(env, "agent_pos", None)
                if agent_pos is not None and hasattr(env, "hazard_positions") and env.hazard_positions:
                    ax, ay = agent_pos
                    nearest = min(env.hazard_positions, key=lambda h: abs(h[0] - ax) + abs(h[1] - ay))
                    dx, dy = nearest[0] - ax, nearest[1] - ay
                    if abs(dx) >= abs(dy):
                        nav_act = 1 if dx > 0 else 0  # down / up
                    else:
                        nav_act = 3 if dy > 0 else 2  # right / left
                    action = _action_to_onehot(nav_act, env.action_dim, agent.device)
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


def _eval_doing_mode(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
) -> Dict:
    """
    Probe action-doing mode by comparing |causal signature| during committed
    vs uncommitted steps.

    CORRECTED vs V3-EXQ-050/050b/199 (see module docstring for the full
    root-cause trace):
      (1) is_committed is sourced from SelectionResult.committed (the real
          BetaGate/E3 commitment-threshold outcome returned directly by
          agent.e3.select(...)), never a candidate cache or a torn-down flag.
      (2) agent.update_residue(...) is called after every env.step() so
          running_variance -- what the commit decision is actually computed
          from -- is genuinely live during eval, not frozen at its
          post-training value.
      (3) sweep_threshold_reduction (BreathOscillator, MECH-108) is passed
          into the SAME agent.e3.select(...) call that produces
          result.committed, guaranteeing periodic genuine uncommitted
          windows even once running_variance converges low.
    """
    agent.eval()
    causal_sigs_committed: List[float] = []
    causal_sigs_uncommitted: List[float] = []
    precision_committed: List[float] = []
    precision_uncommitted: List[float] = []
    all_harm_preds: List[float] = []
    fatal = 0
    sweep_step_count = 0

    for _ in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        held_committed: bool = False
        held_precision: float = 0.0

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
                # (3) BreathOscillator sweep reduction -- same computation
                # select_action itself performs (agent.py:6084), passed here
                # directly into the e3.select() call whose return value we
                # read result.committed from.
                sweep_reduction = (
                    agent.clock.sweep_amplitude if agent.clock.sweep_active else 0.0
                )
                if agent.clock.sweep_active:
                    sweep_step_count += 1

                if ticks.get("e3_tick", False) and candidates:
                    with torch.no_grad():
                        result = agent.e3.select(
                            candidates,
                            temperature=1.0,
                            sweep_threshold_reduction=sweep_reduction,
                        )
                        action = result.selected_action.detach()
                        agent._last_action = action
                        # (1) the real commitment signal, never a cache or a
                        # torn-down flag.
                        held_committed = bool(result.committed)
                        held_precision = float(result.precision)
                else:
                    action = agent._last_action
                    if action is None:
                        action = _action_to_onehot(
                            random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                        )
                        agent._last_action = action
                    # Held state between e3 ticks: commitment/precision carry
                    # forward from the last e3_tick's read (there is no
                    # fresher signal available between ticks; held_committed
                    # defaults to False until the first e3_tick of the
                    # episode fires).

                is_committed = held_committed
                current_precision = held_precision

                # Compute causal signature via SD-003 counterfactual.
                with torch.no_grad():
                    z_world = latent.z_world  # [1, world_dim]
                    actual_idx = action.argmax(dim=-1).item()
                    cf_idx = (random.randint(0, env.action_dim - 2) + 1 + actual_idx) % env.action_dim
                    a_cf = _action_to_onehot(int(cf_idx), env.action_dim, agent.device)

                    z_actual = agent.e2.world_forward(z_world, action)
                    z_cf = agent.e2.world_forward(z_world, a_cf)
                    h_actual = float(agent.e3.harm_eval(z_actual).item())
                    h_cf = float(agent.e3.harm_eval(z_cf).item())
                    causal_sig = h_actual - h_cf
                    all_harm_preds.append(h_actual)

                # Execute action.
                flat_obs, harm_signal, done, info, obs_dict = env.step(action)

                # (2) THE FIX: feed the harm signal into the canonical
                # post-action path every tick, so running_variance (and
                # hence the commit decision) is genuinely live for the whole
                # eval run. Ownership does not matter for this claim (we are
                # not testing responsibility attribution), so owned=True
                # unconditionally -- matching StepHarness's canonical default.
                agent.update_residue(
                    harm_signal=float(harm_signal),
                    world_delta=None,
                    hypothesis_tag=False,
                    owned=True,
                )

                if is_committed:
                    causal_sigs_committed.append(abs(causal_sig))
                    precision_committed.append(current_precision)
                else:
                    causal_sigs_uncommitted.append(abs(causal_sig))
                    precision_uncommitted.append(current_precision)

            except Exception:
                fatal += 1
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                )
                agent._last_action = action
                flat_obs, harm_signal, done, info, obs_dict = env.step(action)
                try:
                    agent.update_residue(
                        harm_signal=float(harm_signal),
                        world_delta=None,
                        hypothesis_tag=False,
                        owned=True,
                    )
                except Exception:
                    pass

            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            if done:
                break

    mean_committed = _mean_safe(causal_sigs_committed)
    mean_uncommitted = _mean_safe(causal_sigs_uncommitted)
    doing_mode_delta = mean_committed - mean_uncommitted
    harm_pred_std = (
        float(torch.tensor(all_harm_preds).std().item()) if len(all_harm_preds) > 1 else 0.0
    )

    print(
        f"  |causal_sig| committed={mean_committed:.4f}  uncommitted={mean_uncommitted:.4f}"
        f"  doing_mode_delta={doing_mode_delta:+.4f}"
        f"  n_committed={len(causal_sigs_committed)}  n_uncommitted={len(causal_sigs_uncommitted)}"
        f"  sweep_steps={sweep_step_count}",
        flush=True,
    )

    return {
        "mean_abs_causal_sig_committed": mean_committed,
        "mean_abs_causal_sig_uncommitted": mean_uncommitted,
        "doing_mode_delta": doing_mode_delta,
        "committed_step_count": len(causal_sigs_committed),
        "uncommitted_step_count": len(causal_sigs_uncommitted),
        "harm_pred_std": harm_pred_std,
        "fatal_errors": fatal,
        "sweep_step_count": sweep_step_count,
        "precision_committed_mean": _mean_safe(precision_committed),
        "precision_uncommitted_mean": _mean_safe(precision_uncommitted),
        "causal_sigs_committed": causal_sigs_committed,
        "causal_sigs_uncommitted": causal_sigs_uncommitted,
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
    nav_bias: float = 0.45,
    lr: float = 1e-3,
    self_dim: int = 32,
    world_dim: int = 32,
    breath_period: int = 50,
    breath_sweep_amplitude: float = 0.30,
    breath_sweep_duration: int = 10,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorldV2(
        seed=seed,
        size=6,
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
        use_event_classifier=True,  # SD-009 (V3-EXQ-199 precedent)
        breath_period=breath_period,
        breath_sweep_amplitude=breath_sweep_amplitude,
        breath_sweep_duration=breath_sweep_duration,
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
        f"[V3-EXQ-876] MECH-025: Action-Doing Mode Causal Signal (seed={seed})\n"
        f"  warmup={warmup_episodes}  eval={eval_episodes}  alpha_world={alpha_world}\n"
        f"  breath_period={breath_period}  sweep_amplitude={breath_sweep_amplitude}"
        f"  sweep_duration={breath_sweep_duration}\n"
        f"  nav_bias={nav_bias}  size=6  n_hazards=4",
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
        nav_bias,
    )
    world_forward_r2 = _compute_world_forward_r2(agent, train_out["wf_buf"])
    print(f"  world_forward_r2: {world_forward_r2:.4f}", flush=True)

    print(f"\n[V3-EXQ-876] Eval -- probing action-doing mode causal signal...", flush=True)
    eval_out = _eval_doing_mode(agent, env, eval_episodes, steps_per_episode, world_dim)

    # --- POSITIVE-CONTROL GATE ------------------------------------------
    # P0: both regimes must be genuinely sampled this run before C1 is read
    # as evidence -- the exact quantity that broke this lineage 3 times.
    readiness_checks = [
        {
            "name": "committed_regime_sampled",
            "measured": eval_out["committed_step_count"],
            "threshold": COMMITTED_FLOOR,
            "direction": "lower",
            "control": (
                "committed_step_count from SelectionResult.committed, must "
                f">= {COMMITTED_FLOOR} for the doing-mode contrast to be "
                "non-vacuous (V3-EXQ-050/050b root cause: cache field read "
                "as always-committed)."
            ),
        },
        {
            "name": "uncommitted_regime_sampled",
            "measured": eval_out["uncommitted_step_count"],
            "threshold": UNCOMMITTED_FLOOR,
            "direction": "lower",
            "control": (
                "uncommitted_step_count from SelectionResult.committed, must "
                f">= {UNCOMMITTED_FLOOR} for the doing-mode contrast to be "
                "non-vacuous (V3-EXQ-199 root cause: torn-down field read as "
                "never-committed)."
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

    degeneracy = check_degeneracy(
        {
            "C1_causal_sig_committed": eval_out["causal_sigs_committed"],
            "C1_causal_sig_uncommitted": eval_out["causal_sigs_uncommitted"],
        }
    )
    non_degenerate = degeneracy["non_degenerate"]

    # PASS / FAIL
    c1_pass = eval_out["doing_mode_delta"] > 0.002
    c2_pass = eval_out["committed_step_count"] >= COMMITTED_FLOOR
    c3_pass = eval_out["uncommitted_step_count"] >= UNCOMMITTED_FLOOR
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
        label = "c1_causal_sig_degenerate_vacuous_test"
    elif all_pass:
        status = "PASS"
        evidence_direction = "supports"
        label = "doing_mode_produces_distinct_causal_signature"
    else:
        status = "FAIL"
        evidence_direction = "mixed" if criteria_met >= 4 else "weakens"
        label = "doing_mode_does_not_produce_distinct_causal_signature"

    failure_notes = []
    if not ready:
        failure_notes.append(
            "P0 FAIL: readiness precondition unmet -- "
            f"committed={eval_out['committed_step_count']} (floor {COMMITTED_FLOOR}), "
            f"uncommitted={eval_out['uncommitted_step_count']} (floor {UNCOMMITTED_FLOOR}) "
            "(substrate_not_ready_requeue)"
        )
    if ready and not non_degenerate:
        failure_notes.append(f"DEGENERATE: {degeneracy['degeneracy_reason']}")
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: doing_mode_delta={eval_out['doing_mode_delta']:.4f} <= 0.002"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: committed_step_count={eval_out['committed_step_count']} < {COMMITTED_FLOOR}"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: uncommitted_step_count={eval_out['uncommitted_step_count']} < {UNCOMMITTED_FLOOR}"
        )
    if not c4_pass:
        failure_notes.append(f"C4 FAIL: world_forward_r2={world_forward_r2:.4f} <= 0.05")
    if not c5_pass:
        failure_notes.append(f"C5 FAIL: harm_pred_std={eval_out['harm_pred_std']:.4f} <= 0.01")
    if not c6_pass:
        failure_notes.append(f"C6 FAIL: fatal_errors={eval_out['fatal_errors']}")

    print(f"\nV3-EXQ-876 seed={seed} verdict: {status}  label={label}  ({criteria_met}/6)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "mean_abs_causal_sig_committed": float(eval_out["mean_abs_causal_sig_committed"]),
        "mean_abs_causal_sig_uncommitted": float(eval_out["mean_abs_causal_sig_uncommitted"]),
        "doing_mode_delta": float(eval_out["doing_mode_delta"]),
        "committed_step_count": float(eval_out["committed_step_count"]),
        "uncommitted_step_count": float(eval_out["uncommitted_step_count"]),
        "harm_pred_std": float(eval_out["harm_pred_std"]),
        "world_forward_r2": float(world_forward_r2),
        "e3_tick_total": float(train_out["e3_tick_total"]),
        "total_harm_train": float(train_out["total_harm"]),
        "fatal_error_count": float(eval_out["fatal_errors"]),
        "sweep_step_count": float(eval_out["sweep_step_count"]),
        "precision_committed_mean": float(eval_out["precision_committed_mean"]),
        "precision_uncommitted_mean": float(eval_out["precision_uncommitted_mean"]),
        "breath_period": float(breath_period),
        "breath_sweep_amplitude": float(breath_sweep_amplitude),
        "breath_sweep_duration": float(breath_sweep_duration),
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

    summary_markdown = f"""# V3-EXQ-876 (seed={seed}) -- MECH-025: Action-Doing Mode Causal Signal

**Status:** {status}
**Label:** {label}
**Claim:** MECH-025 -- action-doing mode produces distinct internal (causal) signature
**Supersedes (lineage, not formal `supersedes` -- different instrument, same claim):**
  V3-EXQ-050, V3-EXQ-050b, V3-EXQ-199 (all invalidated: committed_step_count
  degenerate at 0 or ~100%, instrument defect not substrate ceiling)
**alpha_world:** {alpha_world}
**Warmup:** {warmup_episodes} eps  |  Eval: {eval_episodes} eps
**Seed:** {seed}

## Positive-Control Gate

| Check | Measured | Floor | Met |
|---|---|---|---|
| committed_regime_sampled | {eval_out['committed_step_count']} | {COMMITTED_FLOOR} | {eval_out['committed_step_count'] >= COMMITTED_FLOOR} |
| uncommitted_regime_sampled | {eval_out['uncommitted_step_count']} | {UNCOMMITTED_FLOOR} | {eval_out['uncommitted_step_count'] >= UNCOMMITTED_FLOOR} |

C1 non-degeneracy (causal_sig sample spread, both regimes): {non_degenerate} ({degeneracy['degeneracy_reason'] or 'ok'})

## Results

| Metric | Value |
|--------|-------|
| |causal_sig| Committed (mean) | {eval_out['mean_abs_causal_sig_committed']:.4f} |
| |causal_sig| Uncommitted (mean) | {eval_out['mean_abs_causal_sig_uncommitted']:.4f} |
| doing_mode_delta | {eval_out['doing_mode_delta']:+.4f} |
| Committed steps sampled | {eval_out['committed_step_count']} |
| Uncommitted steps sampled | {eval_out['uncommitted_step_count']} |
| Sweep steps (BreathOscillator) | {eval_out['sweep_step_count']} |
| precision (committed mean) | {eval_out['precision_committed_mean']:.4f} |
| precision (uncommitted mean) | {eval_out['precision_uncommitted_mean']:.4f} |
| World Forward R2 | {world_forward_r2:.4f} |
| Harm Pred Std | {eval_out['harm_pred_std']:.4f} |

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: doing_mode_delta > 0.002 | {"PASS" if c1_pass else "FAIL"} | {eval_out['doing_mode_delta']:+.4f} |
| C2: committed_step_count >= {COMMITTED_FLOOR} | {"PASS" if c2_pass else "FAIL"} | {eval_out['committed_step_count']} |
| C3: uncommitted_step_count >= {UNCOMMITTED_FLOOR} | {"PASS" if c3_pass else "FAIL"} | {eval_out['uncommitted_step_count']} |
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
        "evidence_direction": evidence_direction,
        "experiment_type": EXPERIMENT_TYPE,
        "supersedes": SUPERSEDES,
        "fatal_error_count": eval_out["fatal_errors"],
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": {"C1": non_degenerate},
            "criteria": [
                {"name": "C1_doing_mode_delta", "load_bearing": True, "passed": bool(c1_pass)},
                {"name": "C2_committed_step_count", "load_bearing": False, "passed": bool(c2_pass)},
                {"name": "C3_uncommitted_step_count", "load_bearing": False, "passed": bool(c3_pass)},
                {"name": "C4_world_forward_r2", "load_bearing": False, "passed": bool(c4_pass)},
                {"name": "C5_harm_pred_std", "load_bearing": False, "passed": bool(c5_pass)},
                {"name": "C6_no_fatal_errors", "load_bearing": False, "passed": bool(c6_pass)},
            ],
        },
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy["degeneracy_reason"],
    }


def _run_multi_seed(seeds, **run_kwargs) -> dict:
    """Run across multiple seeds, aggregate metrics, produce combined result."""
    per_seed = {}
    all_metrics_keys = None

    for seed in seeds:
        print(f"\n{'='*60}", flush=True)
        print(f"  Seed {seed}", flush=True)
        print(f"{'='*60}", flush=True)
        result = run(seed=seed, **run_kwargs)
        per_seed[seed] = result
        if all_metrics_keys is None:
            all_metrics_keys = list(result["metrics"].keys())

    agg_metrics = {}
    for key in all_metrics_keys:
        vals = [per_seed[s]["metrics"][key] for s in seeds]
        agg_metrics[key] = _mean_safe(vals)

    all_pass = all(per_seed[s]["status"] == "PASS" for s in seeds)
    overall_status = "PASS" if all_pass else "FAIL"
    total_criteria = sum(int(per_seed[s]["metrics"]["criteria_met"]) for s in seeds)
    max_criteria = 6 * len(seeds)

    if all_pass:
        evidence_direction = "supports"
    elif total_criteria >= 4 * len(seeds):
        evidence_direction = "mixed"
    else:
        evidence_direction = "weakens"

    # If EVERY seed self-routed non_contributory (P0 unmet or degenerate),
    # the aggregate should read as non_contributory too, not "weakens" --
    # that would misreport an unready substrate as claim pressure.
    labels = {per_seed[s]["interpretation"]["label"] for s in seeds}
    if labels <= {"substrate_not_ready_requeue", "c1_causal_sig_degenerate_vacuous_test"}:
        evidence_direction = "non_contributory"
        overall_status = "FAIL"

    seed_lines = []
    for s in seeds:
        m = per_seed[s]["metrics"]
        seed_lines.append(
            f"| {s} | {per_seed[s]['status']} | {int(m['criteria_met'])}/6 |"
            f" {m['doing_mode_delta']:+.4f} |"
            f" {int(m['committed_step_count'])} |"
            f" {int(m['uncommitted_step_count'])} |"
            f" {m['world_forward_r2']:.4f} |"
            f" {m['harm_pred_std']:.4f} |"
        )
    seed_table = "\n".join(seed_lines)

    failure_notes = []
    for s in seeds:
        if per_seed[s]["status"] != "PASS":
            for line in per_seed[s]["summary_markdown"].split("\n"):
                if line.startswith("- C") or line.startswith("- P0") or line.startswith("- DEGENERATE"):
                    if "FAIL" in line or "P0 FAIL" in line or "DEGENERATE" in line:
                        failure_notes.append(f"seed {s}: {line.strip('- ')}")

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    bp = run_kwargs.get("breath_period", 50)
    bsa = run_kwargs.get("breath_sweep_amplitude", 0.30)
    bsd = run_kwargs.get("breath_sweep_duration", 10)

    summary_markdown = f"""# V3-EXQ-876 -- MECH-025: Action-Doing Mode Causal Signal

**Overall Status:** {overall_status}  ({total_criteria}/{max_criteria} criteria across {len(seeds)} seeds)
**Claim:** MECH-025 -- action-doing mode produces distinct internal (causal) signature
**Prior invalidated attempts:** V3-EXQ-050, V3-EXQ-050b, V3-EXQ-199 (instrument
  defects: wrong commitment field read as a cache or a torn-down flag; missing
  update_residue call left running_variance frozen during eval)
**Fix:** SelectionResult.committed (not cache/torn-down field) +
  agent.update_residue() every eval tick + BreathOscillator
**Seeds:** {seeds}
**BreathOscillator:** period={bp}, amplitude={bsa}, duration={bsd}

## Per-Seed Results

| Seed | Status | Criteria | doing_mode_delta | n_committed | n_uncommitted | wf_r2 | harm_std |
|------|--------|----------|-------------------|-------------|-----------------|-------|----------|
{seed_table}

## Aggregate Metrics

| Metric | Value |
|--------|-------|
| doing_mode_delta (mean) | {agg_metrics['doing_mode_delta']:+.4f} |
| committed_step_count (mean) | {agg_metrics['committed_step_count']:.0f} |
| uncommitted_step_count (mean) | {agg_metrics['uncommitted_step_count']:.0f} |
| world_forward_r2 (mean) | {agg_metrics['world_forward_r2']:.4f} |
| harm_pred_std (mean) | {agg_metrics['harm_pred_std']:.4f} |
| sweep_step_count (mean) | {agg_metrics['sweep_step_count']:.0f} |
{failure_section}
"""

    return {
        "status": overall_status,
        "metrics": agg_metrics,
        "per_seed_metrics": {str(s): per_seed[s]["metrics"] for s in seeds},
        "per_seed_status": {str(s): per_seed[s]["status"] for s in seeds},
        "per_seed_interpretation": {str(s): per_seed[s]["interpretation"] for s in seeds},
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": evidence_direction,
        "experiment_type": EXPERIMENT_TYPE,
        "supersedes": SUPERSEDES,
        "fatal_error_count": int(agg_metrics.get("fatal_error_count", 0)),
    }


if __name__ == "__main__":
    import argparse
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="42,123,7,99,256")
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--eval-eps", type=int, default=50)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--alpha-world", type=float, default=0.9)
    parser.add_argument("--harm-scale", type=float, default=0.02)
    parser.add_argument("--nav-bias", type=float, default=0.45)
    parser.add_argument("--breath-period", type=int, default=50)
    parser.add_argument("--breath-amplitude", type=float, default=0.30)
    parser.add_argument("--breath-duration", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="Quick validation run")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    run_kwargs = dict(
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        harm_scale=args.harm_scale,
        nav_bias=args.nav_bias,
        breath_period=args.breath_period,
        breath_sweep_amplitude=args.breath_amplitude,
        breath_sweep_duration=args.breath_duration,
    )

    if args.dry_run:
        run_kwargs["warmup_episodes"] = 5
        run_kwargs["eval_episodes"] = 3
        run_kwargs["steps_per_episode"] = 30
        seeds = [42]
        print("[DRY RUN] Minimal config for smoke test", flush=True)

    result = _run_multi_seed(seeds, **run_kwargs)

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
            config=None,
            seeds=seeds,
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
    else:
        # Even in --dry-run, emit so the runner-conformance sentinel machinery
        # (and any harness checking for it) sees a clean exit signal, matching
        # V3-EXQ-199's dry-run convention.
        _outcome_raw = str(result.get("status", "FAIL")).upper()
        emit_outcome(
            outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
            manifest_path=None,
            dry_run=True,
        )
