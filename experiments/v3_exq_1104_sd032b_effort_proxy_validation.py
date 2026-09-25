"""V3-EXQ-1104 -- SD-032b candidate effort proxy validation (substrate readiness, diagnostic).

Validates substrate_queue entry `sd032b-candidate-effort-proxy` (ree-v3 4cce9b8):
REEConfig.dacc_candidate_effort_source = "harm_a_forward" replaces the constant
rollout-horizon effort proxy with a per-candidate E2_harm_a rollout cost.

WHY. With candidate_effort = c.actions.shape[1] (identical for every candidate),
control_required * effort is a uniform shift and the Croxson harm_interaction term is
identically zero, so nothing pe-dependent -- including MECH-268's f_sat -- can move the
E3 argmin. Measured before the build: effort spread > 0 on 0/120 ticks; E3 argmin
changed across the 7 reachable f_sat values on 0/120 ticks
(REE_assembly docs/architecture/mech_268_dacc_saturation_form.md, Consumer A).
This run asks whether the new proxy clears that failure record.

DESIGN -- one trained agent per seed; every contrast is a WITHIN-TICK COUNTERFACTUAL.
  P0 (5 eps)  : run the agent (ON config), no training.
  P1 (30 eps) : train E2_harm_a online on (z_harm_a_t, a_t) -> z_harm_a_{t+1}, detached
                targets (phased; E2_harm_a is the only trained head).
  P2 (60 eps) : evaluation. At every FRESH dACC tick (E3 ticks, plus step 0 of each
                episode, where the dACC fires without an E3 tick) a PRE-CALL snapshot of the
                agent is deep-copied (stdlib `random` shared via the memo; torch / numpy /
                random RNG state saved and restored around every copy) and the copy's
                select_action is replayed with:
                  - f_sat FORCED to each of the 7 reachable values (window 8, grace 2,
                    strength 0.3: 1/(1+0.3*k), k = 0..6), by patching the copy's
                    dacc._saturation_factor;
                  - dacc_effort_cost in {0.1 (default), 1.0, 10.0};
                  - effort source "harm_a_forward" (ON) and "horizon" (OFF control, cost
                    10.0 only -- argmin-invariant BY ARITHMETIC, so it is the instrument's
                    negative control, never a scored arm);
                  - POSITIVE CONTROL: effort injected on the real argmin candidate `a` only,
                    sized from a measured unit-effort score shift so its added cost exceeds
                    the gap to the runner-up at f=1 (1.67x gap) but not at the smallest f
                    (0.6x gap) -- an arithmetically guaranteed flip if the dACC bias reaches
                    the E3 scores linearly in control_required.
                The copy's argmin(e3.last_scores) is read; an unpatched copy must reproduce
                the real call's argmin (counterfactual fidelity precondition). Counterfactuals
                also run on step 0 of each episode, where the dACC fires without an E3 tick.
                REACH: per tick, (1 - f_min) * pe * cost * gain * effort_spread vs the real
                top-2 gap; C2 is non-degenerate only if reach > gap on some ticks.
  Behaviour config for all phases: dacc_candidate_effort_source="harm_a_forward",
  dacc_effort_cost=1.0, dacc_weight=0.5 (NOT 1.0: at 1.0 the dACC payoff self-feedback
  makes e3.last_scores an undamped integrator -- see DACC_WEIGHT), use_dacc, affective harm stream, salience
  coordinator, lateral PFC, dacc_saturation_enabled, use_e2_harm_a.

CRITERIA (pre-registered; outcome PASS iff C1 AND C2 -- the failure_record target).
  C1 (load-bearing): effort spread > SPREAD_EPS on >= C1_TICK_FRAC of fresh ticks,
     in >= SEEDS_REQUIRED seeds.
  C2 (load-bearing): fraction of fresh ticks whose ON argmin differs across the 7 forced
     f_sat values >= C2_FLIP_FRAC at the best of the 3 pre-registered cost levels, in
     >= SEEDS_REQUIRED seeds. Per-cost fractions are recorded.
  C2_default (label only): the same at the default cost 0.1.
  C3 (label only): Spearman rho between the SELECTED candidate's centred effort and the
     realised change in ||z_harm_a|| until the next fresh tick > C3_RHO_MIN (n >= C3_N_MIN
     pairs), in >= SEEDS_REQUIRED seeds. SD-PP-B9 (open) measured E2_harm_a below the
     persistence predictor, so liveness (C1/C2) is NOT validity; C3 asks the second
     question and E2_harm_a skill-vs-persistence is recorded beside it.

DV SYMMETRY (Step 3.5). ON arm: the manipulation (f_sat x cost scaling of
control_required) multiplies a PER-CANDIDATE effort vector, so it is not a uniform additive
constant and argmin is not invariant under it. OFF/horizon control: the effort vector is
constant, so the manipulation IS a broadcast scalar and argmin is invariant by arithmetic --
scoped out of scoring (disposition b) and used only as the negative-control precondition.

WHAT A NULL DOES NOT MEAN. C2 failing with reach > gap on some ticks is a calibration finding
(label scale_inert); C2 failing with reach never > gap is DEGENERATE (label c2_unreachable) --
neither is evidence against SD-032b. PE is small mid-episode once E2_harm_a is trained and
large only at episode start (no prediction yet), so flips and reach are also reported by
tick class (step0 vs e3). C3 failing says the proxy is live but not harm-predictive
(expected if SD-PP-B9 holds); it does not say a harm-forward cost is the wrong proxy.

claim_ids = [] (substrate readiness; experiment_purpose diagnostic; non_contributory).
Ethics preflight: all involvement flags false, decision allow (V3 pre-ethical instrumentation).
red-team: see queue entry note.
"""
from __future__ import annotations

import argparse
import copy
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import reset_all_rng  # noqa: E402
from experiments._lib.run_id import make_run_id  # noqa: E402
from experiments._lib.stats import spearman  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_1104_sd032b_effort_proxy_validation"
QUEUE_ID = "V3-EXQ-1104"
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS: List[str] = []
SUBSTRATE_ENTRY = "sd032b-candidate-effort-proxy"

SEEDS = [42, 43, 45]
ENV_KW = dict(size=8, num_hazards=8, num_resources=3, num_waypoints=0)
# P2 = 60: P2 episodes typically end by health depletion after 3-10 steps (red-team pass 2),
# so the fresh-tick sample comes mostly from episode-start ticks; 60 episodes target >= 40.
P0_EPS, P1_EPS, P2_EPS, STEPS_PER_EP = 5, 30, 60, 120
E2_HARM_A_LR = 5e-4
BEHAVIOUR_COST = 1.0
# dacc_weight = 0.5, NOT 1.0. The dACC payoff proxy is -e3.last_scores of the PREVIOUS
# tick (agent.py select_action) and DACCtoE3Adapter adds -mode_ev back onto the E3
# scores, so e3.last_scores feeds back into itself with gain dacc_weight. At 1.0 that is
# an undamped integrator: |last_scores| max measured 2553 -> 11833 -> 23404 over one
# 1500-step run (seed 42), so any bounded effort manipulation is compared against a term
# whose scale is set by elapsed ticks. At 0.5 it is a stable geometric feedback (~265,
# flat). Red-team (fable) finding 1, verified 2026-09-25; the integrator itself is
# registered as a separate substrate_queue item.
DACC_WEIGHT = 0.5
COST_GRID = [0.1, 1.0, 10.0]
DEFAULT_COST = 0.1
HORIZON_CONTROL_COST = 10.0
SAT_STRENGTH, SAT_WINDOW, SAT_GRACE = 0.3, 8, 2
F_GRID = [1.0 / (1.0 + SAT_STRENGTH * k) for k in range(SAT_WINDOW - SAT_GRACE + 1)]
# Positive control: effort injected on the real argmin candidate only, sized so its added
# cost is gap/sqrt(F_GRID[-1]) at f=1 (> gap) and gap*sqrt(F_GRID[-1]) at f=F_GRID[-1] (< gap).

# Pre-registered thresholds.
SPREAD_EPS = 1e-6
C1_TICK_FRAC = 0.9
C2_FLIP_FRAC = 0.05
C3_RHO_MIN = 0.1
C3_N_MIN = 20
SEEDS_REQUIRED = 2
N_FRESH_MIN = 40
CF_FIDELITY_MIN = 0.95
POS_CONTROL_FLIP_MIN = 0.8
# Tolerance for a float rounding reorder of a near-tie under the horizon control's exact
# uniform shift. A broken counterfactual instrument flips broadly, far above this.
NEG_CONTROL_FLIP_MAX = 0.02
SCORES_BOUNDED_RATIO_MAX = 1.5  # |last_scores| max, last vs first decile of P2 fresh ticks


def _build(obs: Dict[str, torch.Tensor]) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=obs["body_state"].shape[-1],
        world_obs_dim=obs["world_state"].shape[-1],
        action_dim=4,
        alpha_world=0.3,  # matches V3-EXQ-729 / the mech_268 probes; z_world fidelity is not the DV
        use_dacc=True,
        use_affective_harm_stream=True,
        use_salience_coordinator=True,
        use_lateral_pfc_analog=True,
        dacc_saturation_enabled=True,
        dacc_saturation_window=SAT_WINDOW,
        dacc_saturation_strength=SAT_STRENGTH,
        dacc_saturation_grace=SAT_GRACE,
        use_e2_harm_a=True,
        e2_harm_a_lr=E2_HARM_A_LR,
        dacc_weight=DACC_WEIGHT,
        dacc_effort_cost=BEHAVIOUR_COST,
        dacc_candidate_effort_source="harm_a_forward",
    )
    return REEAgent(cfg)


def _rng_save():
    return (torch.get_rng_state(), np.random.get_state(), random.getstate())


def _rng_restore(st) -> None:
    torch.set_rng_state(st[0])
    np.random.set_state(st[1])
    random.setstate(st[2])


def _cf(agent: REEAgent, cands, ticks, *, f: Optional[float] = None,
        cost: Optional[float] = None, source: Optional[str] = None,
        injected: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    """Replay select_action on a deep copy; return argmin and the copy's dACC readouts.

    The live agent and the global RNG streams are untouched (state restored after)."""
    st = _rng_save()
    try:
        cp = copy.deepcopy(agent, {id(random): random})
        if source is not None:
            cp.config.dacc_candidate_effort_source = source
        if cost is not None and cp.dacc is not None:
            cp.dacc.config.dacc_effort_cost = float(cost)
        if f is not None and cp.dacc is not None:
            cp.dacc._saturation_factor = (lambda cls=None, _f=float(f): (_f, 0))
        if injected is not None:
            cp._dacc_candidate_effort = (
                lambda c, dtype, _v=injected: (_v.to(dtype=dtype), "injected"))
        # Freshness by OBJECT IDENTITY, never by clearing: e3.last_scores is the dACC's
        # payoff source and _dacc_last_bundle is read (previous tick's) before the dACC
        # runs, so clearing either would perturb the very selection being replayed.
        ls_prev, b_prev = cp.e3.last_scores, cp._dacc_last_bundle
        cp.select_action(cands, dict(ticks))
        ls = cp.e3.last_scores if cp.e3.last_scores is not ls_prev else None
        b = cp._dacc_last_bundle if cp._dacc_last_bundle is not b_prev else None
        out: Dict[str, Any] = {
            "argmin": (int(torch.argmin(ls).item())
                       if ls is not None and ls.numel() == len(cands) else None),
            "pe_unsat": (float(cp.dacc._last_pe_unsaturated)
                         if cp.dacc is not None and cp.dacc._last_pe_unsaturated is not None
                         else None),
            "payoff_range": None,
            "scores": (ls.detach().clone() if ls is not None and ls.numel() == len(cands)
                       else None),
        }
        if b is not None:
            pay = b["mode_ev"] + b["effort_term"]
            out["payoff_range"] = float(pay.max() - pay.min())
        return out
    finally:
        _rng_restore(st)


def _flip(argmins: List[Optional[int]]) -> Optional[bool]:
    vals = [a for a in argmins if a is not None]
    if len(vals) < 2:
        return None
    return len(set(vals)) > 1


def _frac(num: int, den: int) -> Optional[float]:
    return (num / den) if den > 0 else None


def run_seed(seed: int, dry_run: bool, zg: ZGoalStreamAccumulator) -> Dict[str, Any]:
    reset_all_rng(seed)
    env = CausalGridWorldV2(seed=seed, **ENV_KW)
    _, obs = env.reset()
    agent = _build(obs)
    opt = torch.optim.Adam(agent.e2_harm_a.parameters(), lr=E2_HARM_A_LR)
    wd = agent.config.latent.world_dim
    total_eps = P0_EPS + P1_EPS + P2_EPS

    p1_loss_by_ep: List[float] = []
    fresh = latched = 0
    spread_ticks = pe_live = payoff_live = 0
    fid_n = fid_agree = 0
    flips_on = {c: 0 for c in COST_GRID}
    flips_on_n = {c: 0 for c in COST_GRID}
    neg_n = neg_flip = 0
    pos_n = pos_flip = 0
    pos_meff: List[float] = []
    fresh_without_cf = 0
    flips_cls = {k: {c: 0 for c in COST_GRID} for k in ("step0", "e3")}
    flips_cls_n = {k: {c: 0 for c in COST_GRID} for k in ("step0", "e3")}
    reach_cls_n = {"step0": 0, "e3": 0}
    reach_cls_gt = {k: {c: 0 for c in COST_GRID} for k in ("step0", "e3")}
    reach_n = 0
    reach_gt_gap = {c: 0 for c in COST_GRID}
    score_absmax: List[float] = []
    pe_start: List[float] = []
    pe_mid: List[float] = []
    spreads: List[float] = []
    effort_term_ranges: List[float] = []
    payoff_ranges: List[float] = []
    pe_vals: List[float] = []
    sel_centred_effort: List[float] = []
    realised_dharm: List[float] = []
    pending: Optional[Dict[str, Any]] = None
    mse_model: List[float] = []
    mse_persist: List[float] = []
    action_counts = [0, 0, 0, 0]

    def _zha_norm() -> Optional[float]:
        z = agent._current_latent.z_harm_a if agent._current_latent is not None else None
        return float(z.detach().norm().item()) if z is not None else None

    for ep in range(total_eps):
        phase = "P0" if ep < P0_EPS else ("P1" if ep < P0_EPS + P1_EPS else "P2")
        agent.reset()
        _, obs = env.reset()
        prev_z = prev_a = None
        ep_losses: List[float] = []
        pending = None
        post_norms: List[float] = []
        for _step in range(STEPS_PER_EP):
            with torch.no_grad():
                lat = agent.sense(obs["body_state"], obs["world_state"],
                                  obs_harm_a=obs.get("harm_obs_a"))
            z_now = lat.z_harm_a.detach().clone() if lat.z_harm_a is not None else None
            # E2_harm_a transition (prev_z, prev_a) -> z_now
            if prev_z is not None and z_now is not None:
                if phase == "P1":
                    pred = agent.e2_harm_a(prev_z, prev_a)
                    loss = agent.e2_harm_a.compute_loss(pred, z_now)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    ep_losses.append(float(loss.item()))
                elif phase == "P2":
                    with torch.no_grad():
                        pred = agent.e2_harm_a(prev_z, prev_a)
                    mse_model.append(float(((pred - z_now) ** 2).mean().item()))
                    mse_persist.append(float(((prev_z - z_now) ** 2).mean().item()))
            zn = _zha_norm()
            if pending is not None and zn is not None:
                post_norms.append(zn)
            with torch.no_grad():
                ticks = agent.clock.advance()
                e1p = agent._e1_tick(lat) if ticks.get("e1_tick") else torch.zeros(1, wd)
                cands = agent.generate_trajectories(lat, e1p, ticks)
                cf_rows = None
                snap = None
                # The dACC also runs on step 0 of every episode without an E3 tick
                # (select_action's early return is skipped after reset); include it so
                # C1 and C2 share one denominator (red-team finding 2).
                if phase == "P2" and (ticks.get("e3_tick") or _step == 0):
                    # ONE pre-call snapshot; every counterfactual below copies THIS, never the
                    # post-call agent (red-team pass 2, finding 1: the post-call state has
                    # _last_action set -> the replay early-returns on step-0 ticks -- and
                    # _harm_a_pred_prev / last_scores already advanced -> a different tick).
                    _st = _rng_save()
                    snap = copy.deepcopy(agent, {id(random): random})
                    _rng_restore(_st)
                    cf_rows = {"fid": _cf(snap, cands, ticks)}
                # Freshness by OBJECT IDENTITY (see _cf): the E3/dACC diagnostics latch on
                # non-E3 steps, and clearing them would perturb selection.
                b_prev, ls_prev = agent._dacc_last_bundle, agent.e3.last_scores
                action = agent.select_action(cands, ticks)
                b_fresh = agent._dacc_last_bundle is not b_prev and agent._dacc_last_bundle is not None
                ls_fresh = agent.e3.last_scores is not ls_prev and agent.e3.last_scores is not None
            a_idx = int(action.argmax(dim=-1).flatten()[0].item())
            if phase == "P2":
                action_counts[a_idx] += 1
            if phase == "P2":
                eff = agent._dacc_last_effort
                bundle = agent._dacc_last_bundle
                if not b_fresh or eff is None or bundle is None:
                    latched += 1
                elif cf_rows is None:
                    fresh_without_cf += 1
                else:
                    fresh += 1
                    # close the previous fresh tick's realised-harm window
                    if pending is not None and post_norms:
                        sel_centred_effort.append(pending["centred"])
                        realised_dharm.append(float(np.mean(post_norms)) - pending["zn"])
                    pending = None
                    post_norms = []
                    sp = float(eff.max() - eff.min())
                    spreads.append(sp)
                    spread_ticks += int(sp > SPREAD_EPS)
                    et = bundle["effort_term"]
                    effort_term_ranges.append(float(et.max() - et.min()))
                    pay = bundle["mode_ev"] + et
                    pr = float(pay.max() - pay.min())
                    payoff_ranges.append(pr)
                    pe_u = agent.dacc._last_pe_unsaturated
                    pe_u = float(pe_u) if pe_u is not None else 0.0
                    pe_vals.append(pe_u)
                    (pe_start if _step == 0 else pe_mid).append(pe_u)
                    if ls_fresh:
                        score_absmax.append(float(agent.e3.last_scores.abs().max()))
                    pe_live += int(pe_u > 1e-6)
                    payoff_live += int(pr > 1e-6)
                    real_arg = (int(torch.argmin(agent.e3.last_scores).item())
                                if ls_fresh and agent.e3.last_scores.numel() == len(cands)
                                else None)
                    if cf_rows is not None:
                        fid_n += 1
                        fid_agree += int(cf_rows["fid"]["argmin"] == real_arg and real_arg is not None)
                        _cls = "step0" if _step == 0 else "e3"
                        for c in COST_GRID:
                            fl = _flip([_cf(snap, cands, ticks, f=f, cost=c,
                                            source="harm_a_forward")["argmin"] for f in F_GRID])
                            if fl is not None:
                                flips_on_n[c] += 1
                                flips_on[c] += int(fl)
                                flips_cls_n[_cls][c] += 1
                                flips_cls[_cls][c] += int(fl)
                        fl = _flip([_cf(snap, cands, ticks, f=f, cost=HORIZON_CONTROL_COST,
                                        source="horizon")["argmin"] for f in F_GRID])
                        if fl is not None:
                            neg_n += 1
                            neg_flip += int(fl)
                        # Baseline = the CONSTANT horizon effort (a uniform shift at every f),
                        # so f moves ONLY the increment injected on candidate a.
                        kc = len(cands)
                        const = torch.full((kc,), float(cands[0].actions.shape[1]))
                        ref = _cf(snap, cands, ticks, f=1.0, cost=BEHAVIOUR_COST,
                                  injected=const)
                        sc = ref["scores"]
                        if pe_u > 1e-6 and sc is not None and sc.numel() >= 2:
                            order = torch.argsort(sc)
                            ia, ib = int(order[0]), int(order[1])
                            gap = float(sc[ib] - sc[ia])
                            if gap > 1e-9:
                                unit = torch.zeros(kc)
                                unit[ia] = 1.0
                                s_unit = _cf(snap, cands, ticks, f=1.0, cost=BEHAVIOUR_COST,
                                             injected=const + unit)["scores"]
                                m_eff = (float(s_unit[ia] - sc[ia]) if s_unit is not None
                                         else 0.0)
                                pos_n += 1
                                pos_meff.append(m_eff)
                                # REACH (red-team finding 3): the largest cross-candidate
                                # shift the f sweep can produce with the REAL effort vector,
                                # in score units, vs the real top-2 gap. gain = score shift
                                # per unit of control_required * effort (measured at f=1).
                                real_sc = cf_rows["fid"]["scores"]
                                if m_eff > 1e-12 and real_sc is not None and real_sc.numel() >= 2:
                                    gain = m_eff / (pe_u * BEHAVIOUR_COST)
                                    ro = torch.sort(real_sc).values
                                    real_gap = float(ro[1] - ro[0])
                                    reach_n += 1
                                    reach_cls_n[_cls] += 1
                                    for c in COST_GRID:
                                        reach = (1.0 - F_GRID[-1]) * pe_u * c * gain * sp
                                        reach_gt_gap[c] += int(reach > real_gap)
                                        reach_cls_gt[_cls][c] += int(reach > real_gap)
                                if m_eff > 1e-12:
                                    x = gap / (m_eff * float(np.sqrt(F_GRID[-1])))
                                    fl = _flip([_cf(snap, cands, ticks, f=f,
                                                    cost=BEHAVIOUR_COST,
                                                    injected=const + unit * x)["argmin"]
                                                for f in F_GRID])
                                    pos_flip += int(bool(fl))
                                # m_eff <= 0: injected effort did not raise a's score -> the
                                # dACC bias is not reaching E3; counted as a control FAILURE.
                    sel = agent.e3.last_selected_idx if ls_fresh else None
                    if sel is not None and zn is not None and 0 <= int(sel) < eff.numel():
                        pending = {"centred": float(eff[int(sel)] - eff.mean()), "zn": zn}
            prev_z = z_now
            prev_a = action.detach().clone()
            _, _, done, _, obs = env.step(a_idx)
            if done:
                break
        if pending is not None and post_norms:
            sel_centred_effort.append(pending["centred"])
            realised_dharm.append(float(np.mean(post_norms)) - pending["zn"])
        if phase == "P1":
            p1_loss_by_ep.append(float(np.mean(ep_losses)) if ep_losses else float("nan"))
        if (ep + 1) % 5 == 0 or dry_run:
            print(f"  [train] seed={seed} phase={phase} ep {ep+1}/{total_eps} fresh_e3={fresh}",
                  flush=True)
    zg.observe(agent)

    rho = spearman(sel_centred_effort, realised_dharm)  # None on degenerate input
    skill = (1.0 - float(np.mean(mse_model)) / float(np.mean(mse_persist))
             if mse_model and mse_persist and float(np.mean(mse_persist)) > 0 else None)
    flip_frac = {str(c): _frac(flips_on[c], flips_on_n[c]) for c in COST_GRID}
    best = max((v for v in flip_frac.values() if v is not None), default=None)
    row = {
        "seed": seed,
        "n_fresh_e3_ticks": fresh - fresh_without_cf,
        "n_latched_ticks": latched,
        "spread_frac": _frac(spread_ticks, fresh),
        "n_fresh_dacc_ticks_all": fresh,
        "effort_spread_mean": float(np.mean(spreads)) if spreads else None,
        "effort_term_range_mean": float(np.mean(effort_term_ranges)) if effort_term_ranges else None,
        "payoff_range_mean": float(np.mean(payoff_ranges)) if payoff_ranges else None,
        "pe_unsat_mean": float(np.mean(pe_vals)) if pe_vals else None,
        "pe_live_frac": _frac(pe_live, fresh),
        "payoff_live_frac": _frac(payoff_live, fresh),
        "cf_fidelity": _frac(fid_agree, fid_n),
        "cf_fidelity_n": fid_n,
        "on_flip_frac_by_cost": flip_frac,
        "on_flip_n_by_cost": {str(c): flips_on_n[c] for c in COST_GRID},
        "on_flip_frac_best": best,
        "on_flip_frac_default_cost": flip_frac[str(DEFAULT_COST)],
        "neg_control_horizon_flip_frac": _frac(neg_flip, neg_n),
        "neg_control_n": neg_n,
        "pos_control_injected_flip_frac": _frac(pos_flip, pos_n),
        "pos_control_n": pos_n,
        "pos_control_unit_shift_mean": float(np.mean(pos_meff)) if pos_meff else None,
        "n_fresh_without_cf": fresh_without_cf,
        "c2_denominator_min_over_costs": min(flips_on_n.values()) if flips_on_n else 0,
        "c1_minus_c2_denominator": (fresh - fresh_without_cf) - (min(flips_on_n.values()) if flips_on_n else 0),
        "on_flip_frac_by_tick_class": {k: {str(c): _frac(flips_cls[k][c], flips_cls_n[k][c])
                                           for c in COST_GRID} for k in flips_cls},
        "on_flip_n_by_tick_class": {k: {str(c): flips_cls_n[k][c] for c in COST_GRID}
                                    for k in flips_cls_n},
        "reach_gt_gap_frac_by_tick_class": {k: {str(c): _frac(reach_cls_gt[k][c], reach_cls_n[k])
                                                for c in COST_GRID} for k in reach_cls_gt},
        "reach_n": reach_n,
        "reach_gt_gap_frac_by_cost": {str(c): _frac(reach_gt_gap[c], reach_n) for c in COST_GRID},
        "pe_unsat_mean_episode_start": float(np.mean(pe_start)) if pe_start else None,
        "pe_unsat_mean_mid_episode": float(np.mean(pe_mid)) if pe_mid else None,
        "last_scores_absmax_first_decile": (float(np.mean(score_absmax[:max(1, len(score_absmax)//10)]))
                                            if score_absmax else None),
        "last_scores_absmax_last_decile": (float(np.mean(score_absmax[-max(1, len(score_absmax)//10):]))
                                           if score_absmax else None),
        "c3_rho": rho,
        "c3_n_pairs": len(sel_centred_effort),
        "e2_harm_a_skill_vs_persistence": skill,
        "e2_harm_a_mse_model_p2": float(np.mean(mse_model)) if mse_model else None,
        "e2_harm_a_mse_persist_p2": float(np.mean(mse_persist)) if mse_persist else None,
        "e2_harm_a_p1_loss_by_ep": p1_loss_by_ep,
        "p2_action_counts": action_counts,
    }
    f0, f1 = row["last_scores_absmax_first_decile"], row["last_scores_absmax_last_decile"]
    row["last_scores_growth_ratio"] = (f1 / f0) if (f0 and f1 is not None and f0 > 0) else None
    row["c1_seed_pass"] = bool(row["spread_frac"] is not None and row["spread_frac"] >= C1_TICK_FRAC)
    row["c2_seed_pass"] = bool(best is not None and best >= C2_FLIP_FRAC)
    row["c2_default_seed_pass"] = bool(row["on_flip_frac_default_cost"] is not None
                                       and row["on_flip_frac_default_cost"] >= C2_FLIP_FRAC)
    row["c3_seed_pass"] = bool(rho is not None and len(sel_centred_effort) >= C3_N_MIN
                               and rho > C3_RHO_MIN)
    return row


def _worst(rows: List[Dict[str, Any]], key: str, mode: str = "min"):
    vals = [(r[key], r["seed"]) for r in rows if r.get(key) is not None]
    if not vals:
        return None, None
    v = min(vals) if mode == "min" else max(vals)
    return v[0], v[1]


def main(dry_run: bool = False) -> Dict[str, Any]:
    global P0_EPS, P1_EPS, P2_EPS, STEPS_PER_EP
    started_at = time.perf_counter()
    seeds = SEEDS[:1] if dry_run else list(SEEDS)
    n_fresh_min = 3 if dry_run else N_FRESH_MIN
    if dry_run:
        P0_EPS, P1_EPS, P2_EPS, STEPS_PER_EP = 1, 2, 2, 40
    zg = ZGoalStreamAccumulator()
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        print(f"Seed {seed} Condition within_tick_counterfactual", flush=True)
        r = run_seed(seed, dry_run, zg)
        rows.append(r)
        print(f"verdict: {'PASS' if (r['c1_seed_pass'] and r['c2_seed_pass']) else 'FAIL'}", flush=True)

    req = min(SEEDS_REQUIRED, len(seeds))
    n_c1 = sum(r["c1_seed_pass"] for r in rows)
    n_c2 = sum(r["c2_seed_pass"] for r in rows)
    n_c2d = sum(r["c2_default_seed_pass"] for r in rows)
    n_c3 = sum(r["c3_seed_pass"] for r in rows)
    c1, c2 = n_c1 >= req, n_c2 >= req
    c2d, c3 = n_c2d >= req, n_c3 >= req

    fresh_w, fresh_s = _worst(rows, "c2_denominator_min_over_costs")
    dd_w, dd_s = _worst(rows, "c1_minus_c2_denominator", "max")
    fid_w, fid_s = _worst(rows, "cf_fidelity")
    neg_w, neg_s = _worst(rows, "neg_control_horizon_flip_frac", "max")
    pos_w, pos_s = _worst(rows, "pos_control_injected_flip_frac")
    grow_w, grow_s = _worst(rows, "last_scores_growth_ratio", "max")

    def _pc(name, measured, threshold, seed, control, direction="lower", comparator=None):
        met = (measured is not None and
               (measured <= threshold if direction == "upper" else measured >= threshold))
        d = {"name": name, "measured": measured, "threshold": threshold, "direction": direction,
             "offending_cell": f"seed={seed}", "control": control, "met": bool(met)}
        if comparator:
            d["comparator"] = comparator
        return d

    preconditions = [
        _pc("c2_denominator_min", fresh_w, n_fresh_min, fresh_s,
            "worst seed; C2's OWN denominator (ticks with a valid f-sweep, min over costs) -- "
            "the sample the load-bearing fraction is computed on"),
        _pc("c1_c2_denominator_gap", dd_w, 0, dd_s,
            "max over seeds; C1 ticks minus C2 ticks -- C1 and C2 must be read on the SAME "
            "fresh ticks (a dropped tick class would bias C2)", direction="upper",
            comparator="<="),
        _pc("counterfactual_fidelity", fid_w, CF_FIDELITY_MIN, fid_s,
            "worst seed; an UNPATCHED deep copy must reproduce the real call's argmin"),
        _pc("neg_control_horizon_flip_frac", neg_w, NEG_CONTROL_FLIP_MAX, neg_s,
            "max over seeds; horizon effort is constant -> argmin-invariant by arithmetic; "
            "any flip means the counterfactual instrument is broken", direction="upper",
            comparator="<="),
        _pc("pos_control_injected_flip_frac", pos_w, POS_CONTROL_FLIP_MIN, pos_s,
            "worst seed; effort injected on the argmin candidate, sized from a measured unit "
            "shift to cross the runner-up gap between f_min and f=1, must flip the argmin"),
        _pc("last_scores_bounded_growth_ratio", grow_w, SCORES_BOUNDED_RATIO_MAX, grow_s,
            "max over seeds; |e3.last_scores| last/first decile of P2 -- the payoff "
            "self-feedback must be at a stationary operating point, not integrating",
            direction="upper", comparator="<="),
    ]
    ready = all(p["met"] for p in preconditions)

    c1_nd = bool(any(r["spread_frac"] for r in rows if r["spread_frac"] is not None))
    # C2 discriminates iff the instrument could have registered a flip: fidelity, the
    # negative control and the guaranteed-flip positive control all met (the readiness
    # preconditions). A measured zero with those green is a genuine null, not degeneracy.
    # ...AND the real manipulation can reach the top-2 gap on some ticks (red-team
    # finding 3): a positive control sized to the gap passes for any gap, so it cannot
    # alone certify that C2 was reachable.
    def _reach_any(cost_key):
        return any((r["reach_gt_gap_frac_by_cost"].get(cost_key) or 0) > 0 for r in rows)
    c2_nd = bool(ready and any(_reach_any(str(c)) for c in COST_GRID))
    c2d_nd = bool(ready and _reach_any(str(DEFAULT_COST)))
    criteria = [
        {"name": "C1_effort_spread_live", "load_bearing": True, "passed": bool(c1),
         "measured": n_c1, "threshold": req, "per_seed_threshold": C1_TICK_FRAC,
         "per_seed_measured": {str(r["seed"]): r["spread_frac"] for r in rows}},
        {"name": "C2_argmin_responsive_to_control", "load_bearing": True, "passed": bool(c2),
         "measured": n_c2, "threshold": req, "per_seed_threshold": C2_FLIP_FRAC,
         "per_seed_measured": {str(r["seed"]): r["on_flip_frac_best"] for r in rows}},
        {"name": "C2_default_cost_responsive", "load_bearing": False, "passed": bool(c2d),
         "measured": n_c2d, "threshold": req, "per_seed_threshold": C2_FLIP_FRAC,
         "per_seed_measured": {str(r["seed"]): r["on_flip_frac_default_cost"] for r in rows}},
        {"name": "C3_effort_predicts_realised_harm", "load_bearing": False, "passed": bool(c3),
         "measured": n_c3, "threshold": req, "per_seed_threshold": C3_RHO_MIN,
         "per_seed_measured": {str(r["seed"]): r["c3_rho"] for r in rows}},
    ]
    combination_rule = ("outcome PASS iff C1 AND C2 (the substrate_queue failure_record target) and "
                        "all readiness preconditions met; C2_default and C3 route the label only")

    if not ready:
        label, outcome = "substrate_not_ready_requeue", "FAIL"
    elif not c1:
        label, outcome = "effort_proxy_not_live", "FAIL"
    elif not c2 and not c2_nd:
        label, outcome = "effort_proxy_live_c2_unreachable", "FAIL"
    elif not c2:
        label, outcome = "effort_proxy_live_scale_inert", "FAIL"
    elif c3:
        label, outcome = "effort_proxy_live_and_harm_predictive", "PASS"
    else:
        label, outcome = "effort_proxy_live_not_harm_predictive", "PASS"
    summary = (f"C1 {n_c1}/{len(rows)} seeds, C2 {n_c2}/{len(rows)} (best cost), "
               f"C2_default {n_c2d}/{len(rows)}, C3 {n_c3}/{len(rows)}; ready={ready}")
    routing = {
        "effort_proxy_live_and_harm_predictive": "mark substrate entry implemented_validated",
        "effort_proxy_live_not_harm_predictive": ("liveness validated; validity routes to SD-PP-B9 "
                                                   "(E2_harm_a below persistence)"),
        "effort_proxy_live_scale_inert": ("calibration: the effort term COULD reach the top-2 gap "
                                          "on some ticks yet the argmin did not respond at the "
                                          "pre-registered costs; puzzle (known rules)"),
        "effort_proxy_live_c2_unreachable": ("DEGENERATE C2: the real manipulation never reached "
                                             "the top-2 gap at any cost, so C2 could not flip -- "
                                             "not a calibration null; see per-tick-class reach"),
        "effort_proxy_not_live": "substrate defect in the effort proxy; /implement-substrate",
        "substrate_not_ready_requeue": "fix the failing precondition and re-queue a new letter",
    }[label]

    def _flat(v: Any) -> Optional[float]:
        if isinstance(v, bool):
            return float(int(v))
        if isinstance(v, (int, float)) and np.isfinite(v):
            return float(v)
        return None

    raw = {
        "n_seeds_c1": n_c1, "n_seeds_c2": n_c2, "n_seeds_c2_default": n_c2d, "n_seeds_c3": n_c3,
        "seeds_required": req, "c1_pass": c1, "c2_pass": c2, "ready": ready,
        "worst_c2_denominator": fresh_w, "max_c1_minus_c2_denominator": dd_w, "worst_cf_fidelity": fid_w,
        "max_neg_control_flip_frac": neg_w, "worst_pos_control_flip_frac": pos_w,
    }
    for c in COST_GRID:
        vals = [r["on_flip_frac_by_cost"][str(c)] for r in rows
                if r["on_flip_frac_by_cost"][str(c)] is not None]
        raw[f"mean_on_flip_frac_cost_{c}"] = float(np.mean(vals)) if vals else None
    skills = [r["e2_harm_a_skill_vs_persistence"] for r in rows
              if r["e2_harm_a_skill_vs_persistence"] is not None]
    raw["mean_e2_harm_a_skill_vs_persistence"] = float(np.mean(skills)) if skills else None
    rhos = [r["c3_rho"] for r in rows if r["c3_rho"] is not None]
    raw["mean_c3_rho"] = float(np.mean(rhos)) if rhos else None
    for c in COST_GRID:
        vals = [r["reach_gt_gap_frac_by_cost"][str(c)] for r in rows
                if r["reach_gt_gap_frac_by_cost"][str(c)] is not None]
        raw[f"mean_reach_gt_gap_frac_cost_{c}"] = float(np.mean(vals)) if vals else None
    raw["max_last_scores_growth_ratio"] = grow_w
    readout = {k: fv for k, fv in ((k, _flat(v)) for k, v in raw.items()) if fv is not None}

    manifest: Dict[str, Any] = {
        "run_id": make_run_id(EXPERIMENT_TYPE, queue_id=QUEUE_ID),
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "validates_substrate": SUBSTRATE_ENTRY,
        "evidence_direction": "non_contributory",
        "evidence_direction_note": "diagnostic substrate-readiness validation; no claim credit",
        "outcome": outcome,
        "timestamp_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "dry_run": bool(dry_run),
        "per_seed": rows,
        "criteria": criteria,
        "combination_rule": combination_rule,
        "readout": readout,
        "interpretation": {
            "label": label,
            "summary": summary,
            "routing": routing,
            "preconditions": preconditions,
            "criteria_non_degenerate": {
                "C1_effort_spread_live": c1_nd,
                "C2_argmin_responsive_to_control": c2_nd,
                "C2_default_cost_responsive": c2d_nd,
                "C3_effort_predicts_realised_harm": bool(
                    any(r["c3_n_pairs"] >= C3_N_MIN for r in rows)),
            },
            "what_a_null_does_not_mean": (
                "C2 failing at every cost is a calibration finding (the effort term cannot compete "
                "with the E3 payoff range), not evidence against SD-032b; C3 failing says the proxy "
                "is live but not harm-predictive (expected under SD-PP-B9), not that a harm-forward "
                "cost is the wrong proxy."),
            "dv_symmetry": ("ON: f_sat x cost scales a per-candidate effort vector -> not a uniform "
                            "shift, argmin not invariant. horizon control: constant effort -> "
                            "broadcast scalar, argmin-invariant by arithmetic -> scoped out of "
                            "scoring, used only as the negative-control precondition."),
        },
        "non_degenerate": bool(ready and c1_nd and c2_nd),
        "degeneracy_reason": ("" if (ready and c1_nd and c2_nd) else
                              "precondition unmet, effort spread pinned, or C2 unreachable "
                              "(real manipulation never exceeded the top-2 gap)"),
        "pre_registered_thresholds": {
            "SPREAD_EPS": SPREAD_EPS, "C1_TICK_FRAC": C1_TICK_FRAC, "C2_FLIP_FRAC": C2_FLIP_FRAC,
            "C3_RHO_MIN": C3_RHO_MIN, "C3_N_MIN": C3_N_MIN, "SEEDS_REQUIRED": SEEDS_REQUIRED,
            "N_FRESH_MIN": N_FRESH_MIN,
            "CF_FIDELITY_MIN": CF_FIDELITY_MIN, "POS_CONTROL_FLIP_MIN": POS_CONTROL_FLIP_MIN,
            "NEG_CONTROL_FLIP_MAX": NEG_CONTROL_FLIP_MAX, "COST_GRID": COST_GRID,
            "SCORES_BOUNDED_RATIO_MAX": SCORES_BOUNDED_RATIO_MAX, "DACC_WEIGHT": DACC_WEIGHT,
            "F_GRID": F_GRID,
        },
        "n_latched_ticks_total": sum(r["n_latched_ticks"] for r in rows),
        "ethics_preflight": {
            "involves_negative_valence": False, "involves_suffering_like_state": False,
            "involves_self_model": False, "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False, "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False, "decision": "allow",
        },
        "custom_information": {
            "sd_pp_b9_note": ("E2_harm_a skill-vs-persistence recorded per seed "
                              "(e2_harm_a_skill_vs_persistence); P1 loss trajectory recorded "
                              "(e2_harm_a_p1_loss_by_ep) per SD-PP-B9 pieces 1-2."),
            "counterfactual_method": ("copy.deepcopy(agent, {id(random): random}); torch/numpy/"
                                      "random RNG saved+restored around every copy; f_sat forced "
                                      "via dacc._saturation_factor patch on the copy"),
        },
    }
    out_path = write_flat_manifest(
        manifest,
        dry_run=dry_run,
        config={
            "env": ENV_KW, "schedule": {"p0": P0_EPS, "p1": P1_EPS, "p2": P2_EPS,
                                        "steps_per_ep": STEPS_PER_EP},
            "behaviour": {"dacc_candidate_effort_source": "harm_a_forward",
                          "dacc_effort_cost": BEHAVIOUR_COST, "dacc_weight": DACC_WEIGHT,
                          "e2_harm_a_lr": E2_HARM_A_LR,
                          "saturation": [SAT_WINDOW, SAT_STRENGTH, SAT_GRACE]},
            "thresholds": manifest["pre_registered_thresholds"],
        },
        seeds=seeds,
        script_path=Path(__file__),
        started_at=started_at,
        z_goal_stream_stats=zg.stats(),
    )
    manifest["_out_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V3-EXQ-1104 SD-032b effort proxy validation")
    parser.add_argument("--dry-run", action="store_true",
                        help="1 seed, tiny schedule; manifest relocated out of evidence/")
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)
    out_path = result["_out_path"]
    print()
    print("=== V3-EXQ-1104 SD-032b effort proxy validation ===")
    print(f"label:   {result['interpretation']['label']}")
    print(f"outcome: {result['outcome']}")
    print(f"summary: {result['interpretation']['summary']}")
    for r in result["per_seed"]:
        print(f"  seed {r['seed']}: fresh={r['n_fresh_e3_ticks']} spread_frac={r['spread_frac']} "
              f"flip_by_cost={r['on_flip_frac_by_cost']} neg={r['neg_control_horizon_flip_frac']} "
              f"pos={r['pos_control_injected_flip_frac']} fid={r['cf_fidelity']} rho={r['c3_rho']} "
              f"skill={r['e2_harm_a_skill_vs_persistence']}")
    print(f"manifest: {out_path}")
    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
