"""V3-EXQ-1078: INV-069 coherence leg under decision Option C -- z_self maintenance
(arm a, transient perturbation) + frozen maintenance restricted to the UNSETTLED
post-boundary regime (arm b), both read on the what_would_answer's own DV
per_stream_vs['z_self'].

RED-TEAM (fable): CONTESTED -> fixed (labelling/recording only; nothing measured changed):
F1 untrained GRU => A1/A2 criteria_non_degenerate False + label drops 'maintenance';
F2 freeze identity voids arm (b) only; F3 noise band = detrended settled sd; B.passed
honours the unsettled-regime precondition; F4 (manipulation->DV paths) CLEAR.

=============================================================================
WHY THIS DESIGN, AND WHAT IT CANNOT DO (read before the result)
=============================================================================
INV-069 asserts the self is a dynamically sustained PROCESS, not a stored state.
Its what_would_answer (claims.yaml) names two manipulations on one trained
substrate per seed, read on per_stream_vs['z_self'] (ree_core/hippocampal/
module.py update_per_stream_vs: err = ||z_curr - z_prev|| / (||z_curr|| + eps);
score = clip(1 - err, 0, 1); vs = EMA(score, tau=0.1)):
  (a) TRANSIENT PERTURBATION of the recurrent hidden state, process left running;
  (b) FROZEN MAINTENANCE -- stored z_self untouched, process stopped.
CONFIRMING requires both, sign-consistent across >=3 seeds.

The behavioural leg is not measurable in V3 (GFLAG-0414: DR-10 is inert, the
GatedPolicy substitute is rank-invariant). GFLAG-0428 then proved that arm (b)
is SIGN-LOCKED on this DV: freezing z_self makes z_curr == z_prev, so every
frozen tick scores 1.0 (the instrument maximum) and the frozen vs can only ever
be >= the live vs. Stronger than GFLAG-0428 states it: the frozen trace is fully
determined in closed form by the vs value at freeze onset,
    vs_frozen(n) = 1 - (1 - tau)^n * (1 - vs_onset),
so the arm (b) contrast frozen-minus-live is an arithmetic function of the LIVE
trace alone. It is not a measurement of the freeze.

The human chose Option C (recommendation ledger rec-20260923-061462f2) knowing
this: arm (b) is run at the one operating point where the LIVE control has not
saturated -- ticks 2..9 after an episode boundary, where the recurrence is still
settling from its reset hidden state and live ||dz_self|| is large -- which
changes the SIZE of the gap, never its DIRECTION.

PRE-REGISTERED ROUTING (fixed here, before any run):
  * INV-069 evidence_direction is "unknown" in EVERY branch. No outcome of this
    design can confirm or falsify INV-069: CONFIRMING needs arm (b), and arm (b)
    cannot discriminate on this DV; arm (a) non-recovery is, per the
    what_would_answer itself, "not refuted but UNTESTABLE here" (routes to the
    ARC-053 temporal coherence loop), not a weakening. Hence
    EXPERIMENT_PURPOSE = "diagnostic" (excluded from confidence scoring).
  * ARM (b) is scoped OUT of scoring (queue-experiment Step 3 disposition (b),
    DV-symmetry class: the manipulation fixes its own DV). Its gap is recorded
    as a descriptive magnitude against the settled-regime noise band, and the
    closed form above is checked as an INSTRUMENT IDENTITY: a frozen trace that
    departs from it (or a single tick with frozen < live) means the freeze
    leaked, and routes instrument_fault -- never evidence.
    Interpretable arm (b) readings: (i) identity holds + gap above the noise
    band in the unsettled window -> "the live process is actively changing
    z_self after a boundary, and this DV reads that change as LOWER coherence
    than a stored vector" (a statement about the instrument/regime, fed back to
    GFLAG-0428's amendment); (ii) identity holds + gap within noise -> even the
    unsettled regime gives this DV no headroom; (iii) identity fails -> freeze
    hook defect. None of the three is a claim verdict.
  * ARM (a) is the only arm that can yield an interpretable substrate finding,
    and only when BOTH of its load-bearing criteria hold on all 3 seeds:
      A1 the per_stream_vs gap (control minus perturbed) returns to within
         RECOVERY_BAND by tick K_REC, AND
      A2 the perturbed z_self STATE converges back onto the control trajectory
         (relative state divergence at K_REC <= STATE_RESTORE_RATIO x its value
         at the burst).
    A2 is required because A1 alone is NOT diagnostic: vs is a consecutive-
    difference readout, so a perturbation that left a PERMANENT offset in a
    stored (unmaintained) vector would also let vs recover once the step
    changes returned to normal -- vs recovery is guaranteed by the EMA for any
    one-tick event. Only state convergence distinguishes "the process restored
    the self" from "the jump happened once and stuck". A1 without A2 routes
    undetermined (EMA artifact), never as a maintenance signature.
    A1+A2 on all seeds = "the recurrence restores z_self after a burst" -- a
    NECESSARY, not sufficient, condition of INV-069, reported as a substrate fact
    with evidence_direction "unknown" for the claim. Red-team F1: with an
    UNTRAINED GRUCell (gru_param_max_delta == 0, the expected case) its per-tick
    contraction c is fixed by init; for any c in [0.20, 0.926] gate + A1 + A2
    pass by construction, so A1/A2 are then recorded criteria_non_degenerate
    False and the label names init contraction, not maintenance. LATENT REVERSE
    ERROR, stated: a recurrence restoring >80% of the burst in ONE tick (c < 0.20)
    makes the peak dip fall under DIP_FLOOR and reads substrate_not_ready_requeue
    -- the dip gate excludes the strongest form of the restoring effect.
    A2's horizon is deliberately NOT moved (that would change what is measured).
  * NOISE BAND (red-team F3): the settled window (ticks 15..29) is still an EMA
    ramp, so the raw sd of vs there is trend; the arm-(b) gap is scaled by the
    sd of the residual after a linear detrend (raw sd also recorded). T_PERT=30
    is therefore "late", not strictly settled -- harmless for A1/A2, which are
    twin-matched at the same tick.

CAVEATS stated rather than papered over:
  * No loss in the V3 training path reaches the LatentStack at all (smoke
    2026-09-23: 0 of 53 latent_stack tensors changed after P0; GRU max parameter
    delta 0.0) -- E1/E2 train on detached latents. The process under test is a
    RANDOM-INIT self encoder + RANDOM-INIT GRU plus the TRAINED E1 generative
    anchor (coupling 0.15). A contractive random GRU restores perturbations
    generically; an A2 pass is therefore a property of this recurrence as built,
    not of a learned self-maintenance. Recorded per seed as gru_param_max_delta
    and latent_stack_tensors_changed.
  * V_s is an identity-prediction proxy, not INV-067's precision-weighted
    correspondence (the what_would_answer's own CAVEAT).
  * The pre-sampling E3 ranking re-convergence half of arm (a)'s CONFIRMING
    condition is behavioural and is not measured (GFLAG-0414). Twin action
    divergence is recorded descriptively only.

ENVIRONMENT: CausalGridWorldV2 size 10, 2 hazards (hazard_harm 0.1),
contaminated_harm 0.0. The default contamination harm killed the agent at tick
~7 of every episode in the pilot (revisited cells contaminate), leaving no
settled regime in which to place the arm (a) burst; harm is not the subject.

TWIN MECHANICS: each contrast is a pair of deep copies of (agent, env, harness)
taken at the split tick, run with the torch / numpy / python RNG states
restored to the split snapshot, so the unmanipulated twin is bit-identical to
the main line (a control-vs-control replay is recorded as a determinism check).
=============================================================================
"""

import argparse
import copy
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments._harness import StepHarness
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiments.pack_writer import write_flat_manifest
from experiment_protocol import emit_outcome

EXPERIMENT_PURPOSE = "diagnostic"
QUEUE_ID = "V3-EXQ-1078"
EXPERIMENT_TYPE = "v3_exq_1078_inv069_zself_coherence_unsettled"
CLAIM_IDS = ["INV-069"]

# ---- pre-registered constants (fixed before any run) ----
SEEDS = [42, 43, 45]          # 44 avoided (recurring per-seed instability)
N_TRAIN_EPISODES = 20
TRAIN_STEPS = 100
LR = 1e-3
N_EVAL_EVENTS = 4             # eval episodes per seed; each carries one arm-a and one arm-b event
MIN_EVENTS_PER_SEED = 3
EPISODES_PER_RUN = N_TRAIN_EPISODES + N_EVAL_EVENTS

TAU = 0.1                     # per_stream_vs_tau (substrate default, asserted at build)
# arm (b): unsettled post-boundary regime
T_ON = 2                      # freeze onset tick after reset (tick 0 initialises the vs cache)
W_B = 8                       # frozen window (pilot: live err decays 0.16 -> 0.003 over 8 ticks)
# arm (a): settled regime
T_PERT = 30                   # burst tick after reset
K_REC = 30                    # recovery horizon (0.9^30 = 0.04 of an EMA-only residue)
PERTURB_FRAC = 0.5            # burst norm = 0.5 x ||z_self|| (GFLAG-0414 pilot magnitude)
SETTLED_LO, SETTLED_HI = 15, 29   # settled window on the main line (inclusive ticks)

DIP_FLOOR = 0.015             # arm (a) readiness: peak vs gap must clear this (pilot 0.031-0.034)
RECOVERY_BAND = 0.005         # A1: |vs gap| at K_REC must be within the DV's own noise band
STATE_RESTORE_RATIO = 0.10    # A2: state divergence at K_REC <= 10% of its value at the burst
CEILING_UPPER = 1.0 - 1e-6    # control vs must sit STRICTLY inside (0,1) -- the WWA's own wording;
                              # headroom itself is certified by arm_a_dip_measurable, not by this bound
UNSETTLED_FACTOR = 5.0        # arm (b) regime: window live err >= 5x settled live err
NOISE_MULT = 3.0              # arm (b) descriptive: gap vs 3x settled sd
FREEZE_IDENTITY_TOL = 1e-6
RECURRENCE_DEPART_FLOOR = 1e-3

BASE_ENV = dict(size=10, num_hazards=2, num_resources=5,
                hazard_harm=0.1, contaminated_harm=0.0)
CFG_KW = dict(self_dim=32, world_dim=32, alpha_world=0.9,
              use_self_recurrence=True, self_recurrence_e1_coupling=0.15,
              use_per_stream_vs=True)

_ZG = ZGoalStreamAccumulator()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _f(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _detrended_sd(xs: List[float]) -> Optional[float]:
    """sd of the residual after a least-squares linear fit (F3: the settled window is
    an EMA ramp, so raw sd is trend, not noise)."""
    xs = [x for x in xs if x is not None]
    if len(xs) < 3:
        return None
    t = np.arange(len(xs), dtype=np.float64)
    coef = np.polyfit(t, np.asarray(xs, dtype=np.float64), 1)
    return float(np.std(np.asarray(xs) - np.polyval(coef, t)))


def _mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


# ---------------------------------------------------------------- twin plumbing

def _dcopy(obj: Any) -> Any:
    """deepcopy that shares the hippocampal module-level RNG handle (a module
    object, not picklable); the global RNG states are restored per twin instead."""
    memo: Dict[int, Any] = {}
    for ag in (obj, getattr(obj, "agent", None)):
        hip = getattr(ag, "hippocampal", None)
        rng = getattr(hip, "_rng", None)
        if rng is not None:
            memo[id(rng)] = rng
    return copy.deepcopy(obj, memo)


def _rng_snapshot():
    return (torch.get_rng_state(), np.random.get_state(), random.getstate())


def _rng_restore(snap) -> None:
    torch.set_rng_state(snap[0])
    np.random.set_state(snap[1])
    random.setstate(snap[2])


def _fork(agent, env, harness, obs):
    a = _dcopy(agent)
    e = copy.deepcopy(env)
    h = _dcopy(harness)
    h.agent = a
    h.env = e
    return a, e, h, copy.deepcopy(obs)


def _run_twin(agent, harness, obs, n: int, rng_snap, hook=None) -> List[Dict[str, Any]]:
    _rng_restore(rng_snap)
    rows: List[Dict[str, Any]] = []
    prev = agent._current_latent.z_self.detach().clone()
    for k in range(n):
        if hook is not None:
            hook(agent, k)
        r = harness.step(obs)
        obs = r.next_obs_dict
        z = r.latent.z_self.detach().clone()
        dz = float((z - prev).norm().item())
        rows.append({
            "vs": _f(agent.hippocampal.per_stream_vs.get("z_self")),
            "dz": dz,
            "err": dz / (float(z.norm().item()) + 1e-6),
            "z": z,
            "action": int(r.action.argmax().item()),
            "done": bool(r.done),
        })
        prev = z
        if r.done:
            break
    return rows


def _freeze(agent) -> None:
    """Arm (b): stop the process, keep the stored state. The SelfRecurrenceCell
    returns its hidden input unchanged and the E1-anchor blend is removed
    (coupling 0.0), so encode() emits z_self == previous stateful z_self."""
    sr = agent.latent_stack.self_recurrence
    sr.forward = lambda z_instant, z_prev: z_prev.detach().clone()
    agent.latent_stack.config.self_recurrence_e1_coupling = 0.0


# ---------------------------------------------------------------- one seed

def _make_env(seed: int) -> CausalGridWorldV2:
    kw = dict(BASE_ENV)
    kw["seed"] = seed
    return CausalGridWorldV2(**kw)


def run_seed(seed: int, dry_run: bool = False) -> Dict[str, Any]:
    n_train = 2 if dry_run else N_TRAIN_EPISODES
    train_steps = 30 if dry_run else TRAIN_STEPS
    n_events = 1 if dry_run else N_EVAL_EVENTS
    total_eps = n_train + n_events
    env = _make_env(seed)
    _, obs = env.reset()
    cfg_slice = dict(CFG_KW, body_obs_dim=env.body_obs_dim,
                     world_obs_dim=env.world_obs_dim, action_dim=env.action_dim)

    with arm_cell(seed, config_slice=dict(cfg_slice, env=BASE_ENV),
                  script_path=Path(__file__)) as cell:
        config = REEConfig.from_dims(**cfg_slice)
        agent = REEAgent(config)
        # Build-time assertion that the knobs landed (from_dims swallows unknown kwargs).
        assert config.latent.use_self_recurrence is True
        assert agent.latent_stack.self_recurrence is not None
        assert bool(agent.hippocampal.config.use_per_stream_vs) is True
        assert "z_self" in tuple(agent.hippocampal.config.per_stream_vs_streams)
        assert abs(float(agent.hippocampal.config.per_stream_vs_tau) - TAU) < 1e-12

        gru0 = {k: v.detach().clone()
                for k, v in agent.latent_stack.self_recurrence.state_dict().items()}
        ls0 = {k: v.detach().clone() for k, v in agent.latent_stack.state_dict().items()}

        # ---- P0: substrate warmup (E1 + E2 world-model losses; no readout head) ----
        opt = torch.optim.Adam(agent.parameters(), lr=LR)
        harness = StepHarness(agent, env, train_mode=True, seed=seed)
        agent.train()
        loss_first, loss_last = [], []
        for ep in range(n_train):
            _, obs = env.reset()
            agent.reset()
            harness.reset()
            ep_losses = []
            for _t in range(train_steps):
                r = harness.step(obs)
                obs = r.next_obs_dict
                loss = agent.compute_prediction_loss() + agent.compute_e2_loss()
                if loss.requires_grad:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                ep_losses.append(float(loss.detach().item()))
                if r.done:
                    break
            if ep == 0:
                loss_first = ep_losses
            loss_last = ep_losses
            print(f"  [train] INV069 seed={seed} ep {ep + 1}/{total_eps} "
                  f"len={len(ep_losses)} loss={_mean(ep_losses) or 0.0:.5f}", flush=True)

        gru_delta = max(float((agent.latent_stack.self_recurrence.state_dict()[k] - gru0[k])
                              .abs().max().item()) for k in gru0)
        ls_changed = sum(1 for k, v in agent.latent_stack.state_dict().items()
                         if k in ls0 and v.dtype.is_floating_point
                         and float((v - ls0[k]).abs().max().item()) > 0.0)
        agent.eval()

        # ---- P2: eval events ----
        events: List[Dict[str, Any]] = []
        n_skipped = 0
        for ev in range(n_events):
            eval_env = _make_env(seed * 1000 + 7 + ev)
            _, obs = eval_env.reset()
            agent.reset()
            main_h = StepHarness(agent, eval_env, train_mode=False, seed=seed * 1000 + ev)
            main_h.reset()
            main_vs: List[Optional[float]] = []
            main_err: List[Optional[float]] = []
            departures: List[float] = []
            anchor_hits = 0
            prev_z = None
            died = False

            def _main_step(o):
                nonlocal prev_z, anchor_hits
                r = main_h.step(o)
                z = r.latent.z_self.detach().clone()
                main_vs.append(_f(agent.hippocampal.per_stream_vs.get("z_self")))
                main_err.append(None if prev_z is None else
                                float((z - prev_z).norm().item()) / (float(z.norm().item()) + 1e-6))
                diag = getattr(r.latent, "self_recurrence_diag", None) or {}
                if diag.get("active"):
                    departures.append(float(diag.get("state_departure", 0.0)))
                    anchor_hits += 1 if diag.get("anchor_present") else 0
                prev_z = z
                return r

            for _t in range(T_ON):
                r = _main_step(obs)
                obs = r.next_obs_dict
                if r.done:
                    died = True
                    break
            if died:
                n_skipped += 1
                continue

            # ---- arm (b): frozen vs live twins in the unsettled window ----
            snap_b = _rng_snapshot()
            vs_onset = main_vs[-1]
            a_l, _e_l, h_l, o_l = _fork(agent, eval_env, main_h, obs)
            live = _run_twin(a_l, h_l, o_l, W_B, snap_b)
            a_f, _e_f, h_f, o_f = _fork(agent, eval_env, main_h, obs)
            _freeze(a_f)
            frz = _run_twin(a_f, h_f, o_f, W_B, snap_b)
            nb = min(len(live), len(frz))
            gap_b = [frz[k]["vs"] - live[k]["vs"] for k in range(nb)]
            closed = [1.0 - (1.0 - TAU) ** (k + 1) * (1.0 - vs_onset) for k in range(nb)]
            ident_dev = max(abs(frz[k]["vs"] - closed[k]) for k in range(nb)) if nb else None
            arm_b = {
                "n_ticks": nb,
                "vs_onset": vs_onset,
                "live_vs": [live[k]["vs"] for k in range(nb)],
                "frozen_vs": [frz[k]["vs"] for k in range(nb)],
                "live_err": [live[k]["err"] for k in range(nb)],
                "gap": gap_b,
                "gap_max": max(gap_b) if gap_b else None,
                "gap_mean": _mean(gap_b),
                "n_ticks_frozen_below_live": sum(1 for g in gap_b if g < -1e-9),
                "frozen_dz_max": max((frz[k]["dz"] for k in range(nb)), default=None),
                "closed_form_max_dev": ident_dev,
                "live_err_window_mean": _mean([live[k]["err"] for k in range(nb)]),
                "live_vs_min": min((live[k]["vs"] for k in range(nb)), default=None),
                "action_divergence_frac": (sum(1 for k in range(nb)
                                               if live[k]["action"] != frz[k]["action"]) / nb)
                if nb else None,
            }
            del a_l, h_l, a_f, h_f

            # continue the main line from the arm-b split (RNG restored => main == live twin)
            _rng_restore(snap_b)
            for _t in range(T_ON, T_PERT):
                r = _main_step(obs)
                obs = r.next_obs_dict
                if r.done:
                    died = True
                    break
            if died:
                n_skipped += 1
                continue

            settled_vs = [v for v in main_vs[SETTLED_LO:SETTLED_HI + 1] if v is not None]
            settled_err = [e for e in main_err[SETTLED_LO:SETTLED_HI + 1] if e is not None]

            # ---- arm (a): perturbation vs matched control twins ----
            snap_a = _rng_snapshot()
            g = torch.Generator().manual_seed(seed * 7919 + ev)

            def _burst(ag, k, _g=g):
                if k == 0:
                    z = ag._current_latent.z_self
                    noise = torch.randn(z.shape, generator=_g)
                    noise = noise / (noise.norm() + 1e-12) * PERTURB_FRAC * z.norm()
                    # replace (not in place): the vs cache shares storage with z
                    ag._current_latent.z_self = (z + noise.to(z.device)).detach()

            a_c, _e_c, h_c, o_c = _fork(agent, eval_env, main_h, obs)
            ctrl = _run_twin(a_c, h_c, o_c, K_REC + 1, snap_a)
            a_c2, _e_c2, h_c2, o_c2 = _fork(agent, eval_env, main_h, obs)
            ctrl2 = _run_twin(a_c2, h_c2, o_c2, K_REC + 1, snap_a)
            a_p, _e_p, h_p, o_p = _fork(agent, eval_env, main_h, obs)
            pert = _run_twin(a_p, h_p, o_p, K_REC + 1, snap_a, hook=_burst)
            del a_c, h_c, a_c2, h_c2, a_p, h_p
            na = min(len(ctrl), len(pert))
            if na < K_REC + 1:
                n_skipped += 1
                continue
            gap_a = [ctrl[k]["vs"] - pert[k]["vs"] for k in range(na)]
            sdiv = [float((pert[k]["z"] - ctrl[k]["z"]).norm().item())
                    / (float(ctrl[k]["z"].norm().item()) + 1e-6) for k in range(na)]
            arm_a = {
                "n_ticks": na,
                "ctrl_vs": [ctrl[k]["vs"] for k in range(na)],
                "pert_vs": [pert[k]["vs"] for k in range(na)],
                "gap": gap_a,
                "dip": max(gap_a),
                "gap_at_K": abs(gap_a[K_REC]),
                "state_div": sdiv,
                "state_div_at_burst": sdiv[0],
                "state_div_at_K": sdiv[K_REC],
                "state_div_ratio": sdiv[K_REC] / (sdiv[0] + 1e-12),
                "ctrl_vs_mean": _mean([ctrl[k]["vs"] for k in range(na)]),
                "ctrl_replay_max_abs_diff": max(abs(ctrl[k]["vs"] - ctrl2[k]["vs"])
                                                for k in range(min(na, len(ctrl2)))),
                "action_divergence_frac": sum(1 for k in range(na)
                                              if ctrl[k]["action"] != pert[k]["action"]) / na,
            }
            events.append({
                "event": ev,
                "arm_a": arm_a,
                "arm_b": arm_b,
                "settled_vs_mean": _mean(settled_vs),
                "settled_vs_sd": float(np.std(settled_vs)) if settled_vs else None,
                # F3 (red-team): ticks 15..29 are still an EMA ramp (monotone), so the raw sd
                # measures the TREND, not noise. The noise band is the sd of the residual
                # after a linear detrend; both are recorded.
                "settled_vs_sd_detrended": _detrended_sd(settled_vs),
                "settled_err_mean": _mean(settled_err),
                "main_trace_vs": main_vs,
                "state_departure_mean": _mean(departures),
                "anchor_present_frac": anchor_hits / max(1, len(departures)),
            })
            print(f"  [train] INV069 seed={seed} ep {n_train + ev + 1}/{total_eps} "
                  f"eval event: dip={arm_a['dip']:.4f} gapK={arm_a['gap_at_K']:.4f} "
                  f"sdiv_ratio={arm_a['state_div_ratio']:.4f} "
                  f"b_gap_max={arm_b['gap_max'] if arm_b['gap_max'] is not None else float('nan'):.4f}",
                  flush=True)

        _ZG.observe(agent)

        def ev_mean(fn):
            return _mean([fn(e) for e in events])

        row = {
            "arm": "INV069_OPTC",
            "seed": seed,
            "n_events": len(events),
            "n_events_skipped": n_skipped,
            "gru_param_max_delta": gru_delta,
            "latent_stack_tensors_changed": ls_changed,
            "latent_stack_tensors_total": len(ls0),
            "train_loss_first_ep_mean": _mean(loss_first),
            "train_loss_last_ep_mean": _mean(loss_last),
            # arm (a)
            "a_dip_mean": ev_mean(lambda e: e["arm_a"]["dip"]),
            "a_gap_at_K_mean": ev_mean(lambda e: e["arm_a"]["gap_at_K"]),
            "a_state_div_ratio_mean": ev_mean(lambda e: e["arm_a"]["state_div_ratio"]),
            "a_state_div_at_burst_mean": ev_mean(lambda e: e["arm_a"]["state_div_at_burst"]),
            "a_ctrl_vs_mean": ev_mean(lambda e: e["arm_a"]["ctrl_vs_mean"]),
            "a_ctrl_replay_max_abs_diff": max((e["arm_a"]["ctrl_replay_max_abs_diff"]
                                               for e in events), default=None),
            "a_action_divergence_frac": ev_mean(lambda e: e["arm_a"]["action_divergence_frac"]),
            # arm (b)
            "b_gap_max_mean": ev_mean(lambda e: e["arm_b"]["gap_max"]),
            "b_gap_mean_mean": ev_mean(lambda e: e["arm_b"]["gap_mean"]),
            "b_ticks_frozen_below_live": sum(e["arm_b"]["n_ticks_frozen_below_live"] for e in events),
            "b_frozen_dz_max": max((e["arm_b"]["frozen_dz_max"] for e in events
                                    if e["arm_b"]["frozen_dz_max"] is not None), default=None),
            "b_closed_form_max_dev": max((e["arm_b"]["closed_form_max_dev"] for e in events
                                         if e["arm_b"]["closed_form_max_dev"] is not None),
                                        default=None),
            "b_live_err_window_mean": ev_mean(lambda e: e["arm_b"]["live_err_window_mean"]),
            "b_live_vs_min": min((e["arm_b"]["live_vs_min"] for e in events
                                  if e["arm_b"]["live_vs_min"] is not None), default=None),
            "b_action_divergence_frac": ev_mean(lambda e: e["arm_b"]["action_divergence_frac"]),
            # regime / noise
            "settled_vs_mean": ev_mean(lambda e: e["settled_vs_mean"]),
            "settled_vs_sd": ev_mean(lambda e: e["settled_vs_sd"]),
            "settled_vs_sd_detrended": ev_mean(lambda e: e["settled_vs_sd_detrended"]),
            "settled_err_mean": ev_mean(lambda e: e["settled_err_mean"]),
            "state_departure_mean": ev_mean(lambda e: e["state_departure_mean"]),
            "anchor_present_frac": ev_mean(lambda e: e["anchor_present_frac"]),
            "events": events,
        }
        se = row["settled_err_mean"]
        row["b_unsettled_err_ratio"] = (row["b_live_err_window_mean"] / se) if se else None
        sd = row["settled_vs_sd"]
        row["b_gap_max_over_settled_raw_sd"] = (row["b_gap_max_mean"] / sd) if sd else None
        sdd = row["settled_vs_sd_detrended"]
        row["b_gap_max_over_settled_detrended_sd"] = (row["b_gap_max_mean"] / sdd) if sdd else None
        cell.stamp(row)
    return row


# ---------------------------------------------------------------- analysis

def _worst(rows, key, fn):
    vals = [(r[key], r["seed"]) for r in rows if r.get(key) is not None]
    if not vals:
        return None, None
    return fn(vals, key=lambda x: x[0])


def analyse(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    pre = []

    def add(name, desc, measured, threshold, met, direction="lower", offending=None, control=None):
        entry = {"name": name, "description": desc, "measured": _f(measured),
                 "threshold": threshold, "direction": direction, "met": bool(met)}
        if offending is not None:
            entry["offending_cell"] = offending
        if control is not None:
            entry["control"] = control
        pre.append(entry)
        return bool(met)

    dep, dep_s = _worst(rows, "state_departure_mean", min)
    p_rec = add("arm_a::recurrence_live", "DR-13 stateful z_self departs from the instantaneous encode",
                dep, RECURRENCE_DEPART_FLOOR, dep is not None and dep >= RECURRENCE_DEPART_FLOOR,
                offending=dep_s)
    fdz, fdz_s = _worst(rows, "b_frozen_dz_max", max)
    cfd, cfd_s = _worst(rows, "b_closed_form_max_dev", max)
    below = sum(int(r.get("b_ticks_frozen_below_live") or 0) for r in rows)
    ident_ok = (fdz is not None and fdz == 0.0 and cfd is not None
                and cfd <= FREEZE_IDENTITY_TOL and below == 0)
    p_frz = add("arm_b::freeze_engaged_closed_form_identity",
                "frozen ||dz_self|| exactly 0 and frozen vs == 1-(1-tau)^n(1-vs_onset) within tol, "
                "no tick with frozen < live (instrument identity, not evidence)",
                cfd, FREEZE_IDENTITY_TOL, ident_ok, direction="upper", offending=cfd_s)
    dip, dip_s = _worst(rows, "a_dip_mean", min)
    p_dip = add("arm_a::dip_measurable",
                "peak control-minus-perturbed vs gap clears the floor (the same gap statistic A1 reads at K)",
                dip, DIP_FLOOR, dip is not None and dip >= DIP_FLOOR, offending=dip_s,
                control="perturbed vs unperturbed twin of the same trained agent")
    cvs, cvs_s = _worst(rows, "a_ctrl_vs_mean", max)
    p_ceil = add("arm_a::control_off_ceiling", "control per_stream_vs['z_self'] strictly under 1.0",
                 cvs, CEILING_UPPER, cvs is not None and cvs <= CEILING_UPPER,
                 direction="upper", offending=cvs_s)
    nev, nev_s = _worst(rows, "n_events", min)
    p_nev = add("arm_a::events_completed", "eval events surviving to T_PERT+K_REC per seed",
                nev, MIN_EVENTS_PER_SEED, nev is not None and nev >= MIN_EVENTS_PER_SEED,
                offending=nev_s)
    rep, rep_s = _worst(rows, "a_ctrl_replay_max_abs_diff", max)
    p_det = add("twin_determinism", "control twin replay reproduces itself exactly",
                rep, 1e-9, rep is not None and rep <= 1e-9, direction="upper", offending=rep_s)
    ur, ur_s = _worst(rows, "b_unsettled_err_ratio", min)
    p_uns = add("arm_b::unsettled_regime_present",
                "live err in the post-boundary window >= 5x settled live err (arm b scope only)",
                ur, UNSETTLED_FACTOR, ur is not None and ur >= UNSETTLED_FACTOR, offending=ur_s)

    # F2 (red-team): the freeze identity is a property of _freeze(), which arm (a) never
    # invokes -- it voids arm (b) only, never the arm-(a) gate or label.
    gate_a = p_rec and p_dip and p_ceil and p_nev and p_det
    gru_max = max((float(r.get("gru_param_max_delta") or 0.0) for r in rows), default=0.0)
    gru_trained = gru_max > 0.0

    a1 = [r["a_gap_at_K_mean"] is not None and r["a_gap_at_K_mean"] <= RECOVERY_BAND for r in rows]
    a2 = [r["a_state_div_ratio_mean"] is not None
          and r["a_state_div_ratio_mean"] <= STATE_RESTORE_RATIO for r in rows]
    a1_all, a2_all = all(a1) and bool(rows), all(a2) and bool(rows)
    b_gap_ratio, _ = _worst(rows, "b_gap_max_over_settled_detrended_sd", min)
    b_above_noise = b_gap_ratio is not None and b_gap_ratio >= NOISE_MULT

    if not (p_rec and p_det):
        label = "instrument_fault_requeue"
    elif not (p_dip and p_ceil and p_nev):
        label = "substrate_not_ready_requeue"
    elif a1_all and a2_all:
        # F1 (red-team): with an UNTRAINED GRUCell the pass is init contraction, not a
        # learned maintenance process -- the label says only what was observed.
        label = ("arm_a_state_restores_after_burst__inv069_undetermined" if gru_trained
                 else "arm_a_state_restores_after_burst_untrained_gru_init_contraction"
                      "__inv069_undetermined")
    elif a1_all and not a2_all:
        label = "arm_a_vs_recovery_without_state_restoration_ema_artifact_undetermined"
    elif (not any(a1)) and (not any(a2)):
        label = "arm_a_no_recovery_untestable_routes_arc053"
    else:
        label = "arm_a_seed_inconsistent_undetermined"
    if p_frz and p_det:
        if not p_uns:
            b_label = "arm_b_unsettled_regime_not_reached"
        elif b_above_noise:
            b_label = "arm_b_sign_locked_gap_above_noise_non_contributory"
        else:
            b_label = "arm_b_sign_locked_gap_within_noise_non_contributory"
    else:
        b_label = "arm_b_instrument_fault"

    return {
        "preconditions": pre,
        "gate_a_green": gate_a,
        "gru_trained": gru_trained,
        "gru_param_max_delta_max": gru_max,
        "freeze_identity_ok": p_frz,
        "a1_per_seed": a1, "a2_per_seed": a2,
        "a1_all": a1_all, "a2_all": a2_all,
        "b_gap_over_sd_worst": b_gap_ratio,
        "b_above_noise": b_above_noise,
        "b_unsettled_ok": p_uns,
        "label": label,
        "b_label": b_label,
        "worst_gap_at_K": _worst(rows, "a_gap_at_K_mean", max)[0],
        "worst_state_div_ratio": _worst(rows, "a_state_div_ratio_mean", max)[0],
        "worst_dip": dip,
    }


def run(dry_run: bool = False) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    seeds = SEEDS[:1] if dry_run else SEEDS
    for seed in seeds:
        print(f"Seed {seed} Condition INV069_OPTC", flush=True)
        row = run_seed(seed, dry_run=dry_run)
        rows.append(row)
        ok = (row["n_events"] > 0 and row["a_gap_at_K_mean"] is not None
              and row["a_gap_at_K_mean"] <= RECOVERY_BAND
              and row["a_state_div_ratio_mean"] <= STATE_RESTORE_RATIO)
        print(f"verdict: {'PASS' if ok else 'FAIL'}", flush=True)

    s = analyse(rows)
    outcome = "PASS" if (s["gate_a_green"] and s["a1_all"] and s["a2_all"]) else "FAIL"

    criteria = [
        {"name": "A1_vs_gap_recovers_within_band_at_K", "load_bearing": True,
         "passed": bool(s["a1_all"]), "measured": _f(s["worst_gap_at_K"]),
         "threshold": RECOVERY_BAND, "direction": "upper",
         "note": "worst seed mean |control - perturbed vs| at tick K_REC; all 3 seeds required"},
        {"name": "A2_state_restored_onto_control_trajectory", "load_bearing": True,
         "passed": bool(s["a2_all"]), "measured": _f(s["worst_state_div_ratio"]),
         "threshold": STATE_RESTORE_RATIO, "direction": "upper",
         "note": "worst seed mean state-divergence ratio (K_REC / burst); all 3 seeds required"},
        {"name": "B_gap_above_settled_noise_band", "load_bearing": False,
         "passed": bool(s["b_above_noise"] and s["b_unsettled_ok"] and s["freeze_identity_ok"]),
         "measured": _f(s["b_gap_over_sd_worst"]),
         "threshold": NOISE_MULT, "direction": "lower",
         "note": "DESCRIPTIVE ONLY: worst-seed frozen-minus-live gap / DETRENDED settled vs sd; "
                 "passed requires the unsettled regime and the freeze identity too. Sign-locked "
                 "by construction and scoped out of scoring; magnitude feeds GFLAG-0428"},
    ]
    # F1 (red-team): an untrained GRUCell's per-tick contraction c sits in [0.20, 0.926]
    # by init, where gate + A1 + A2 pass by construction -> not a discriminating reading.
    nd_a = bool(s["gate_a_green"] and s["gru_trained"])
    crit_nd = {
        "A1_vs_gap_recovers_within_band_at_K": nd_a,
        "A2_state_restored_onto_control_trajectory": nd_a,
        # arm (b) cannot discriminate: the manipulation fixes its own DV
        "B_gap_above_settled_noise_band": False,
    }

    readout: Dict[str, float] = {}
    for key, value in (
        ("worst_dip", s["worst_dip"]),
        ("worst_gap_at_K", s["worst_gap_at_K"]),
        ("worst_state_div_ratio", s["worst_state_div_ratio"]),
        ("b_gap_over_settled_detrended_sd_worst", s["b_gap_over_sd_worst"]),
        ("gru_trained", 1 if s["gru_trained"] else 0),
        ("a1_all", 1 if s["a1_all"] else 0),
        ("a2_all", 1 if s["a2_all"] else 0),
        ("gate_a_green", 1 if s["gate_a_green"] else 0),
        ("b_unsettled_regime_ok", 1 if s["b_unsettled_ok"] else 0),
        ("gru_param_max_delta_max", max((r["gru_param_max_delta"] for r in rows), default=None)),
        ("a_action_divergence_frac_mean", _mean([r["a_action_divergence_frac"] for r in rows])),
        ("b_action_divergence_frac_mean", _mean([r["b_action_divergence_frac"] for r in rows])),
        ("settled_vs_sd_mean", _mean([r["settled_vs_sd"] for r in rows])),
        ("settled_vs_sd_detrended_mean", _mean([r["settled_vs_sd_detrended"] for r in rows])),
    ):
        v = _f(value)
        if v is not None:
            readout[key] = v
    for r in rows:
        for k in ("a_dip_mean", "a_gap_at_K_mean", "a_state_div_ratio_mean",
                  "b_gap_max_mean", "b_unsettled_err_ratio"):
            v = _f(r.get(k))
            if v is not None:
                readout[f"{k}_seed{r['seed']}"] = v

    label_full = f"{s['label']}|{s['b_label']}"
    return {
        "run_id": f"{EXPERIMENT_TYPE}_{_utc_stamp()}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        # Pre-registered: no outcome of this design moves INV-069 (see docstring).
        "evidence_direction": "unknown",
        "evidence_direction_per_claim": {"INV-069": "unknown"},
        "evidence_direction_note": (
            "Pre-registered unknown in every branch: CONFIRMING needs arm (b), which is "
            "sign-locked on per_stream_vs (GFLAG-0428); arm (a) non-recovery is UNTESTABLE per "
            "the what_would_answer, not a weakening. Decision rec-20260923-061462f2 (Option C)."),
        "outcome": outcome,
        "timestamp_utc": _utc_stamp(),
        "decision_ref": "rec-20260923-061462f2",
        "governance_flags": ["GFLAG-0414", "GFLAG-0428"],
        "arm_results": rows,
        "per_seed_a_gap_at_K": [r["a_gap_at_K_mean"] for r in rows],
        "per_seed_a_state_div_ratio": [r["a_state_div_ratio_mean"] for r in rows],
        "per_seed_b_gap_max": [r["b_gap_max_mean"] for r in rows],
        "criteria": criteria,
        "criteria_non_degenerate": crit_nd,
        "combination_rule": (
            "outcome PASS iff the arm-(a) precondition gate is green (recurrence live, freeze "
            "identity holds, twin determinism, dip >= floor, control off ceiling, >=3 events per "
            "seed) AND A1 AND A2 hold on ALL seeds. A1 without A2 routes undetermined (vs "
            "recovery is guaranteed by the EMA for any one-tick event). Arm (b) never gates and "
            "never scores. INV-069 evidence_direction is unknown in every branch."),
        "interpretation": {
            "label": label_full,
            "arm_a_label": s["label"],
            "arm_b_label": s["b_label"],
            # F2: the indexer reads this list flat and arm-blind, so a failed arm-(b)
            # instrument check must not vacate the arm-(a) reading -- failed arm_b::
            # preconditions are recorded separately (same pattern as precondition_gate's
            # adjudication_preconditions).
            "preconditions": [p for p in s["preconditions"]
                              if not (p["name"].startswith("arm_b::") and not p["met"])],
            "arm_b_failed_preconditions": [p for p in s["preconditions"]
                                           if p["name"].startswith("arm_b::") and not p["met"]],
            "preconditions_all": s["preconditions"],
            "criteria_non_degenerate": crit_nd,
            "scoped_out_arms": {"arm_b_frozen_maintenance":
                                "DV fixed by the manipulation (closed form), disposition (b)"},
        },
        "readout": readout,
        "summary": {k: v for k, v in s.items() if k != "preconditions"},
        "ethics_preflight": {
            "involves_negative_valence": False, "involves_suffering_like_state": False,
            "involves_self_model": False, "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False, "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False, "decision": "allow",
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _t_start = time.perf_counter()
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    manifest = run(dry_run=args.dry_run)

    out_path = write_flat_manifest(
        manifest,
        None,
        dry_run=args.dry_run,
        config={"env": BASE_ENV, "agent": CFG_KW, "n_train_episodes": N_TRAIN_EPISODES,
                "train_steps": TRAIN_STEPS, "lr": LR, "n_eval_events": N_EVAL_EVENTS,
                "t_on": T_ON, "w_b": W_B, "t_pert": T_PERT, "k_rec": K_REC,
                "perturb_frac": PERTURB_FRAC, "dip_floor": DIP_FLOOR,
                "recovery_band": RECOVERY_BAND, "state_restore_ratio": STATE_RESTORE_RATIO,
                "unsettled_factor": UNSETTLED_FACTOR, "noise_mult": NOISE_MULT},
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=_t_start,
        z_goal_stream_stats=_ZG.stats(),
    )
    print(f"\nResult written to: {out_path}", flush=True)
    print(f"label: {manifest['interpretation']['label']}", flush=True)

    _outcome = str(manifest.get("outcome", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
