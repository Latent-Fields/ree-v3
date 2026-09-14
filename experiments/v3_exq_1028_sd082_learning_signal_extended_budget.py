"""V3-EXQ-1028 -- SD-082 fan-out legs 2 + 4 (H-learning-signal-sign, axis
measurement; H-learning-signal-noisy, axis process): the V3-EXQ-1020 learning-
signal instrument at an EXTENDED P1 budget (T = 500 REINFORCE updates, 1020 ran
70), read with a WINDOWED persistence profile (so a signal that was directed and
then exhausted cannot masquerade as noise) and an EPISODE-LEVEL advantage-on-flip
statistic labelled by the frozen INIT head (so flips cannot self-extinguish out
of the sample as the head trains).

RED-TEAM (fable, Step 4.5): CONTESTED -> 4 fixed (C2 windowed, C3 episode-level); see RED_TEAM_VERDICT.

================================================================================
WHERE THIS COMES FROM
================================================================================
GOV-FANOUT-1 portfolio routed by CONFIRMED failure_autopsy_V3-EXQ-1020_2026-09-11
(REE_assembly 92751187b1), ratified by governance gov-20260911-1612, all legs
user-selected at the autopsy's Step 8 gate; hypotheses pre-registered in
hypothesis_space_registry.v1.json qid sd082_candidate_discriminating_readout_locus
(NOT re-registered). Chip: chip-20260911-sd082-fanout-portfolio-v2.
Siblings: V3-EXQ-1027 (H-replay-rule-state-mismatch), V3-EXQ-1029
(H-selection-authority-bounded).

WHY TWO LEGS SHARE ONE RUN. The autopsy's sketches name the same manipulation
(leg 2 "raise P1 budget and/or flip yield"; leg 4 "extend P1 past T~300"). One
run carries both readings; they are scored independently (criteria_aggregation
"any") with separate non-degeneracy gates.

WHAT 1020 LEFT UNSETTLED (autopsy sec 3.2): C2 (persistence low) passed on a bare
3-of-5 majority against an unexcluded rival (1027 owns it); C3 (advantage on
flips) was STARVED -- 4..61 fresh-select flip samples per cell against a floor of
20.

================================================================================
DESIGN
================================================================================
IDENTICAL to V3-EXQ-1020's ARM_ON cell (centering True; seeds 611/622/633/644/
655; P0 60; 48 steps; lr/batch/buffer/temperature/advantage filter/EMA IMPORTED
from the 1020 module) except:
  * P1 = 500 updates (one per P1 episode, as in 1020).
  * ONE arm (centering True); 1020's second arm carried no criterion.
  * Persistence snapshots of the cumulative step sum at update counts 70, 150,
    300 and the realised end, giving WINDOWS W1=[0,70], W2=[70,150],
    W3=[150,300], W4=[300,end].
  * 1020's two in-run controls (dense-synthetic-credit ceiling, pure-noise Adam
    floor; imported) run per cell at each WINDOW LENGTH (70, 80, 150, and the
    realised W4 length) and at T=500 for the cumulative reading. Their update
    count is set through the 1020 module's SYNTH_UPDATES global for the call and
    restored; noise_n_steps is asserted equal to the intended length.
  * Every fresh, rule-live P1 tick is also labelled with the INIT head's flip
    (argmax(raw | rule_state) != argmax(raw | 0) through a frozen copy of the
    cell's own init head). Measurement only; the optimisation is 1020's.

================================================================================
LEG 4 -- H-learning-signal-noisy, WINDOWED (red-team F1 on the first draft)
================================================================================
The cumulative persistence ||sum of steps|| / sum ||steps|| decays toward the
Adam noise floor for TWO reasons: a direction-inconsistent gradient (the
hypothesis) or a consistent gradient whose learnable structure is exhausted, after
which Adam keeps stepping at ~constant norm around the optimum. An endpoint read
at T=500 cannot tell them apart -- the synthetic control itself falls 0.73 ->
0.54 from T=70 to T=500 for exactly the second reason. So C2 reads each WINDOW
against a bracket measured at that window's length:
    W persistence = ||S_b - S_a|| / (N_b - N_a),  S = cumulative step sum,
    N = cumulative step norm;  midpoint(L) = floor(L) + 0.5 (ceiling(L) - floor(L)).
Per seed:
    noisy      W1 < midpoint(70)  AND  W4 < midpoint(L4)
    exhausted  W1 >= midpoint(70) AND  W4 < midpoint(L4)
    persistent W4 >= midpoint(L4)
  C2 (load-bearing): >= 3 seeds "noisy" -> H-learning-signal-noisy supported: the
     gradient is direction-inconsistent at the start AND stays so.
  Recorded readings select the label otherwise: >= 3 "exhausted" ->
     directed_then_exhausted (H-noisy refuted; 1020's T=70 C2 reading was the
     start of a saturating curve); >= 3 "persistent" -> persistent_directed
     (H-noisy refuted outright); else c2_mixed.
  Window brackets come from FRESH-init-head controls of length L; the real run's
  late windows step a mid-training Adam whose bias-corrected moments have settled.
  Recorded as a known approximation: the pure-noise floor is invariant over
  gradient scale (1020) and moves with L, which is what the window matches.
  Non-degenerate only if every cell's brackets at L=70 and L4 are valid
  (ceiling > floor) and every cell realised >= 95% of the intended updates.

================================================================================
LEG 2 -- H-learning-signal-sign, EPISODE-LEVEL (red-team F2/F3 on the first draft)
================================================================================
1020's C3 statistic (drawn fresh flip samples, trained-head labels) cannot scale
with budget: 1020's own per-episode flip rates collapse to ~0 within ~15-30 P1
episodes on 4 of 5 seeds -- the phenomenon under test (trained flips < init)
starves its own criterion -- and what samples remain sit in early P1, where the
sign reduces to a return trend. It is still computed and RECORDED unchanged.
The load-bearing C3 instead reads the teaching signal directly:
  * With use_modulatory_selection_authority False the head's bias never changes a
    committed action (1020 per_episode_returns bit-identical across arms; 1029
    measures it), so per-episode return does not depend on the head, and a flip
    label from the FROZEN INIT head is available on every episode of P1. That is
    the signal whose sign drives early REINFORCE toward fewer flips.
  * Advantage is an EPISODE quantity (every tick in an episode shares one return),
    so the statistic is episode-level: Pearson r across P1 episodes between the
    episode's init-head flip fraction (among its fresh rule-live ticks) and its
    LOCAL advantage = return minus the mean return of the +/-10 neighbouring
    episodes (drift-robust). A grand-mean-advantage r is computed alongside.
  * Per seed: negative iff r_local <= -2/sqrt(n_eps) AND r_grand < 0; positive
    iff r_local >= +2/sqrt(n_eps) AND r_grand > 0 (n_eps = episodes with >= 1
    fresh rule-live tick). A +/- 2 SE band, so a true-zero effect does not route
    to a signed label by coin-flip.
  * Scoreable iff >= 50 init-head flip ticks in EACH half of P1 and n_eps >= 100.
  C3 (load-bearing): >= 3 scoreable seeds negative -> H-learning-signal-sign
     supported (flip-heavy episodes carry lower advantage).
  Recorded: >= 3 positive -> sign_positive (refuted); >= 3 scoreable, neither ->
     sign_null; < 3 scoreable -> c3_starved; C4 (return variance) failing ->
     c3_degenerate (red-team F3: never a directed label on a degenerate C3).
  Correlational by construction at authority OFF: a flip has no causal path to the
  return; the statistic is about which states produce flips, i.e. the sign of the
  credit REINFORCE receives, which is what the hypothesis asserts.
  Also recorded (H-sign's second clause): Spearman of the TRAINED head's
  per-episode flip rate against episode index, and per-100-episode block means.

READINESS (R): control bracket separation >= 0.20 at every window length in >=
0.8 of cells; every cell realised >= 95% of intended updates (red-team F4);
0 candidate-summary fallbacks; >= 200 measured P1 ticks.
C1 (load-bearing, liveness): median pre-clip gradient norm > 1e-6 on >= 3/5.
OUTCOME: PASS iff R AND C1 (1020's rule); the LABEL carries the leg readings,
label = "c2_<noisy|directed_then_exhausted|persistent_directed|mixed>__c3_<sign_negative|
sign_positive|sign_null|starved|degenerate>" (or substrate_not_ready_requeue /
training_signal_absent_no_gradient).
Directions pinned "unknown" (diagnostic); nothing here moves SD-082's status.
A C2 "noisy" reading still does not exclude H-replay-rule-state-mismatch; read
with V3-EXQ-1027.

DV-SYMMETRY DECLARATION (single arm): windowed persistence is invariant only
under a global rotation/rescaling of the steps inside the window, which a longer
budget is not; Pearson r is invariant under positive affine maps of either
variable, not under relabelling which episodes are flip-heavy. Not invariant.

================================================================================
QUEUE-EXPERIMENT GATE DISPOSITIONS (2026-09-14)
================================================================================
Step 2.4 GOV-REUSE-1: 1020 recorded T=70 only and no init-head flip labels ->
not recoverable, run.
Step 2.5/2.5a: substrate and path identical to V3-EXQ-1020 (imported).
Step 2.5b brake: SD-082 counts 3 (822b/c/d), released by the 822f autopsy; the
1020 autopsy refuses further LETTERED iterations; new EXQ number on the budget
axis it licensed, diagnostic -- "Not braked".
Step 2.5c: module footprint identical to 1020; its call-trace disposition carried
forward.
Step 2.6 ethics: all-false / allow.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest, flat_readout  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.readiness_anchor import assert_anchor_reachable  # noqa: E402
import experiments.v3_exq_1020_sd082_learning_signal_probe as x1020  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_1028_sd082_learning_signal_extended_budget"
QUEUE_ID = "V3-EXQ-1028"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS = ["SD-082"]
FANOUT_QID = "sd082_candidate_discriminating_readout_locus"
FANOUT_HYPOTHESES = ["H-learning-signal-sign", "H-learning-signal-noisy"]
FANOUT_AXES = {"H-learning-signal-sign": "measurement", "H-learning-signal-noisy": "process"}
FANOUT_SOURCE_AUTOPSY = "failure_autopsy_V3-EXQ-1020_2026-09-11"
SOURCE_CHIP_REF = "chip-20260911-sd082-fanout-portfolio-v2"
COMPARES_AGAINST_RUN_ID = "v3_exq_1020_sd082_learning_signal_probe_20260911T003146Z_v3"
RED_TEAM_VERDICT = (
    "red-team (fable): CONTESTED -> 4 findings, all FIXED. F1 the T=500 endpoint C2 aliased "
    "a directed-then-exhausted signal with noise (cumulative persistence decays toward the "
    "Adam floor either way; the synthetic control itself falls 0.73 -> 0.54) -> C2 now reads "
    "WINDOWED persistence W1=[0,70] and W4=[300,end] against brackets measured at each "
    "window's length, with noisy / directed_then_exhausted / persistent_directed labels. F2 "
    "C3's sample count cannot scale with T (trained-head flips extinguish within ~15-30 P1 "
    "episodes on 4/5 seeds; probe seed 611 froze at 4 samples) -> load-bearing C3 is now "
    "episode-level Pearson r between init-head flip fraction and local advantage, +/-2 SE "
    "band, scoreable only with >= 50 init-head flip ticks in each half of P1; 1020's "
    "statistic recorded. F3 a C4 failure could still yield a directed label -> "
    "c3_degenerate label. F4 controls ran at intended T while real updates can be skipped "
    "-> readiness requires >= 95% realised updates and window controls use realised "
    "lengths.")

SEEDS = list(x1020.SEEDS)
ARM = "ARM_ON"
CENTERING = True
P0_WARMUP_EPISODES = x1020.P0_WARMUP_EPISODES
P1_BIAS_TRAIN_EPISODES = 500
TOTAL_EPISODES = P0_WARMUP_EPISODES + P1_BIAS_TRAIN_EPISODES
STEPS_PER_EPISODE = x1020.STEPS_PER_EPISODE
CHECKPOINTS = [70, 150, 300]          # cumulative update counts; the realised end is added
CUMULATIVE_CONTROL_T = 500

# Pre-registered.
BRACKET_SEPARATION_FLOOR = 0.20
SEED_MAJORITY = 3
MIN_UPDATE_REALISATION = 0.95
LOCAL_BASELINE_HALF_WIDTH = 10
R_SE_MULTIPLIER = 2.0
C3_MIN_FLIP_TICKS_PER_HALF = 50
C3_MIN_EPISODES = 100
C3_MIN_SCOREABLE_SEEDS = 3

# Controls measured on this head at authoring (darwin-arm64, 2026-09-14, init heads
# of seeds 611 and 633): synthetic persistence / noise floor at T = 70, 150, 300, 500.
CONTROL_REFERENCE_CELLS = [
    {"L": 70, "synth": 0.7327, "noise": 0.4488}, {"L": 150, "synth": 0.7197, "noise": 0.3304},
    {"L": 300, "synth": 0.6408, "noise": 0.2457}, {"L": 500, "synth": 0.5378, "noise": 0.1915},
    {"L": 70, "synth": 0.7573, "noise": 0.4488}, {"L": 150, "synth": 0.7442, "noise": 0.3304},
    {"L": 300, "synth": 0.6524, "noise": 0.2457}, {"L": 500, "synth": 0.5411, "noise": 0.1915},
]
REFERENCE_SOURCE = ("V3-EXQ-1028 authoring probe 2026-09-14 darwin-arm64: 1020's two controls "
                    "at T=70/150/300/500 on the init heads of seeds 611 and 633")


def _bracket_ok(c: Dict[str, Any]) -> bool:
    """THE SHIPPED bracket predicate (live cells and anchor guard)."""
    return bool((float(c["synth"]) - float(c["noise"])) >= BRACKET_SEPARATION_FLOOR)


def _assert_anchors() -> List[Dict[str, Any]]:
    return [assert_anchor_reachable(
        anchor_name="control_bracket_separates_at_every_window_length",
        reference_cells=CONTROL_REFERENCE_CELLS, score_fn=_bracket_ok,
        threshold=x1020.CONTROL_READY_FRACTION_FLOOR, reference_source=REFERENCE_SOURCE)]


@contextlib.contextmanager
def _updates(T: int):
    saved = x1020.SYNTH_UPDATES
    x1020.SYNTH_UPDATES = int(T)
    try:
        yield
    finally:
        x1020.SYNTH_UPDATES = saved


def _controls_at(agent, init_head, wd, device, T: int) -> Dict[str, Any]:
    with _updates(T):
        c = x1020._synthetic_credit_control(agent, init_head, wd, device)
        n = x1020._noise_persistence_control(init_head, device)
    if int(n["noise_n_steps"]) != int(T):
        raise RuntimeError(f"noise control ran {n['noise_n_steps']} steps, expected {T}")
    synth = float(c["synth_persistence_pooled"])
    noise = float(n["noise_persistence_pooled"])
    return {"L": int(T), "synth": synth, "noise": noise,
            "midpoint": noise + x1020.C2_MIDPOINT_FRACTION * max(synth - noise, 0.0),
            "bracket_valid": bool(synth > noise), "separation": synth - noise,
            "synth_task_gain": float(c["synth_task_gain"]),
            "synth_n_updates": int(c["synth_n_updates"])}


def _config_slice(p1: int) -> Dict[str, Any]:
    s = x1020._config_slice(CENTERING)
    s["schedule"] = {"p0": P0_WARMUP_EPISODES, "p1": int(p1), "steps": STEPS_PER_EPISODE}
    s["checkpoints"] = list(CHECKPOINTS)
    return s


def _pearson(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 3:
        return None
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if float(xa.std()) == 0.0 or float(ya.std()) == 0.0:
        return None
    r = float(np.corrcoef(xa, ya)[0, 1])
    return r if math.isfinite(r) else None


def _c3_episode_stats(ep_returns: List[float], ep_ticks: List[int],
                      ep_init_flips: List[int]) -> Dict[str, Any]:
    n = len(ep_returns)
    rets = np.asarray(ep_returns, dtype=float)
    grand = float(rets.mean()) if n else 0.0
    xs_local, xs_grand, fr = [], [], []
    for i in range(n):
        if ep_ticks[i] <= 0:
            continue
        lo = max(0, i - LOCAL_BASELINE_HALF_WIDTH)
        hi = min(n, i + LOCAL_BASELINE_HALF_WIDTH + 1)
        neigh = [rets[j] for j in range(lo, hi) if j != i]
        if not neigh:
            continue
        xs_local.append(float(rets[i] - np.mean(neigh)))
        xs_grand.append(float(rets[i] - grand))
        fr.append(float(ep_init_flips[i]) / ep_ticks[i])
    n_eps = len(fr)
    half = n // 2
    flips_first = int(sum(ep_init_flips[:half]))
    flips_second = int(sum(ep_init_flips[half:]))
    r_local = _pearson(fr, xs_local)
    r_grand = _pearson(fr, xs_grand)
    se_band = (R_SE_MULTIPLIER / math.sqrt(n_eps)) if n_eps > 0 else None
    scoreable = bool(n_eps >= C3_MIN_EPISODES and flips_first >= C3_MIN_FLIP_TICKS_PER_HALF
                     and flips_second >= C3_MIN_FLIP_TICKS_PER_HALF
                     and r_local is not None and r_grand is not None)
    neg = bool(scoreable and r_local <= -se_band and r_grand < 0.0)
    pos = bool(scoreable and r_local >= se_band and r_grand > 0.0)
    return {"c3_n_episodes": n_eps, "c3_init_flip_ticks_first_half": flips_first,
            "c3_init_flip_ticks_second_half": flips_second, "c3_r_local": r_local,
            "c3_r_grand": r_grand, "c3_se_band": se_band, "c3_scoreable": scoreable,
            "c3_negative": neg, "c3_positive": pos,
            "c3_init_flip_fraction_overall": (float(sum(ep_init_flips)) / max(1, sum(ep_ticks)))}


def _run_cell(seed: int, episodes: Dict[str, int], zg: ZGoalStreamAccumulator,
              checkpoints: List[int], cumulative_T: int) -> Dict[str, Any]:
    p0, p1 = episodes["p0"], episodes["p1"]
    total_eps = p0 + p1
    with arm_cell(seed, config_slice=_config_slice(p1), script_path=Path(__file__),
                  extra_substrate_paths=[Path(x1020.__file__)]) as cell:
        env = x1020._build_env(seed)
        agent = x1020._make_agent(env, CENTERING)
        wd = agent.config.latent.world_dim
        lpfc = agent.lateral_pfc
        device = agent.device
        init_head = copy.deepcopy(lpfc.rule_bias_head)
        for p in init_head.parameters():
            p.requires_grad_(False)
        bias_opt = torch.optim.Adam(list(lpfc.bias_head_parameters()), lr=x1020.LR_LPFC_BIAS)
        names = [n for n, _ in lpfc.rule_bias_head.named_parameters()]
        acc = x1020._StepAccumulator(names)
        tel: Dict[str, Any] = {"grad_norms": [], "sample_flip_adv": [],
                               "n_updates": 0, "n_adv_filtered": 0,
                               "n_grad_nonfinite": 0, "sample_flip_ret": [],
                               "rule_state_norm_at_update": []}
        summary_counters = {"dispatch": 0, "fallback": 0}
        outcome_buf: List[Tuple] = []
        baseline = 0.0
        ep_returns: List[float] = []
        ep_flip_rate: List[float] = []
        ep_init_ticks: List[int] = []
        ep_init_flips: List[int] = []
        n_flip_ticks = n_measured_ticks = n_rule_live_ticks = 0
        n_fresh_ticks = n_fresh_flip_ticks = n_latched_ticks = 0
        snaps: Dict[int, Tuple[torch.Tensor, float]] = {}
        pending = sorted(int(c) for c in checkpoints)

        print(f"Seed {seed} Condition {ARM}", flush=True)
        for ep in range(total_eps):
            is_p1 = (ep >= p0)
            _, obs = env.reset()
            agent.reset()
            ep_reward = 0.0
            ep_buf: List[Tuple] = []
            ep_ticks = ep_flips = 0
            e_init_ticks = e_init_flips = 0
            last_flip = False
            last_flip_valid = False
            for _step in range(episodes["steps"]):
                latent = agent.sense(obs["body_state"], obs["world_state"])
                ticks = agent.clock.advance()
                is_fresh = bool(ticks.get("e3_tick", False))
                e1 = (agent._e1_tick(latent) if ticks.get("e1_tick")
                      else torch.zeros(1, wd, device=device))
                candidates = agent.generate_trajectories(latent, e1, ticks)
                snap: Optional[torch.Tensor] = None
                was_flip = False
                if is_p1 and candidates and len(candidates) >= 2:
                    cs = x1020._candidate_summaries(agent, candidates, summary_counters)
                    if cs is not None and torch.isfinite(cs).all():
                        snap = cs.clone()
                        rule_live = float(lpfc.rule_state.norm()) > x1020.RULE_STATE_LIVE_FLOOR
                        if rule_live:
                            n_rule_live_ticks += 1
                        if rule_live and is_fresh:
                            rf = x1020._raw_ratio_and_flip(lpfc, snap)
                            if rf is not None:
                                last_flip = bool(rf[1] > 0.5)
                                last_flip_valid = True
                            else:
                                last_flip_valid = False
                            if last_flip_valid:
                                n_fresh_ticks += 1
                                if last_flip:
                                    n_fresh_flip_ticks += 1
                            rfi = x1020._raw_ratio_and_flip(lpfc, snap, head=init_head)
                            if rfi is not None:
                                e_init_ticks += 1
                                e_init_flips += int(rfi[1] > 0.5)
                        if rule_live and last_flip_valid:
                            was_flip = last_flip
                            ep_ticks += 1
                            n_measured_ticks += 1
                            if not is_fresh:
                                n_latched_ticks += 1
                            if was_flip:
                                ep_flips += 1
                                n_flip_ticks += 1
                action = agent.select_action(candidates, ticks)
                if action is None:
                    action = torch.zeros(1, 4, device=device)
                    action[0, int(np.random.randint(0, 4))] = 1.0
                    agent._last_action = action
                committed_class = int(action[0].argmax().item())
                if snap is not None:
                    sel = 0
                    for ci, c in enumerate(candidates):
                        if (getattr(c, "actions", None) is not None
                                and c.actions.shape[1] >= 1
                                and int(c.actions[:, 0, :].argmax(-1).reshape(-1)[0].item())
                                == committed_class):
                            sel = min(ci, snap.shape[0] - 1)
                            break
                    ep_buf.append((snap, sel, was_flip, is_fresh))
                _, _h, done, _info, obs = env.step(int(action.argmax(dim=-1).item()))
                if is_p1:
                    ep_reward += float(_h)
                if done:
                    break
            if is_p1:
                baseline = x1020.EMA_DECAY * baseline + (1.0 - x1020.EMA_DECAY) * ep_reward
                for cand_features, sel, flip_lbl, fresh_lbl in ep_buf:
                    outcome_buf.append((cand_features, sel, ep_reward, flip_lbl, None, fresh_lbl))
                if len(outcome_buf) > x1020.OUTCOME_BUF_MAX:
                    outcome_buf = outcome_buf[-x1020.OUTCOME_BUF_MAX:]
                x1020._reinforce_step(lpfc, bias_opt, outcome_buf, baseline, device, acc, tel)
                ep_returns.append(float(ep_reward))
                ep_flip_rate.append(float(ep_flips) / ep_ticks if ep_ticks > 0 else 0.0)
                ep_init_ticks.append(e_init_ticks)
                ep_init_flips.append(e_init_flips)
                while pending and int(tel["n_updates"]) >= pending[0]:
                    cp = pending.pop(0)
                    vec = acc.pooled_vec.detach().clone() if acc.pooled_vec is not None else None
                    snaps[cp] = (vec, float(acc.pooled_norm))
            if (ep + 1) % 25 == 0 or (ep + 1) == total_eps:
                print(f"  [train] {ARM} seed={seed} ep {ep+1}/{total_eps} "
                      f"updates={tel['n_updates']}", flush=True)

        n_final = int(tel["n_updates"])
        final_vec = acc.pooled_vec.detach().clone() if acc.pooled_vec is not None else None
        bounds = [0] + [c for c in sorted(snaps)] + [n_final]
        bounds = sorted(set(b for b in bounds if b <= n_final))
        windows: List[Dict[str, Any]] = []
        prev_vec, prev_norm, prev_n = None, 0.0, 0
        for b in bounds[1:]:
            vec, nrm = (snaps[b] if b in snaps else (final_vec, float(acc.pooled_norm)))
            if vec is None:
                continue
            dv = vec if prev_vec is None else vec - prev_vec
            dn = nrm - prev_norm
            pers = float(dv.norm().item()) / dn if dn > 0 else None
            windows.append({"start": prev_n, "end": b, "L": b - prev_n, "persistence": pers})
            prev_vec, prev_norm, prev_n = vec, nrm, b
        controls: Dict[str, Dict[str, Any]] = {}
        for w in windows:
            L = int(w["L"])
            if L > 0 and str(L) not in controls:
                controls[str(L)] = _controls_at(agent, init_head, wd, device, L)
            w["control"] = controls.get(str(L))
        if str(cumulative_T) not in controls:
            controls[str(cumulative_T)] = _controls_at(agent, init_head, wd, device, cumulative_T)
        zg.observe(agent)

        row: Dict[str, Any] = {"arm_id": ARM, "seed": seed, "centering": CENTERING,
                               "p1_updates_intended": int(p1)}
        row.update(x1020._summarise_cell(tel, acc.result(), ep_returns, ep_flip_rate,
                                         n_flip_ticks, n_measured_ticks))
        row["windows"] = windows
        row["controls_by_L"] = controls
        first = windows[0] if windows else None
        last = windows[-1] if len(windows) >= 2 else None
        row["w1_persistence"] = first["persistence"] if first else None
        row["w1_midpoint"] = first["control"]["midpoint"] if first and first.get("control") else None
        row["w4_persistence"] = last["persistence"] if last else None
        row["w4_midpoint"] = last["control"]["midpoint"] if last and last.get("control") else None
        row["w4_length"] = last["L"] if last else None
        row["update_realisation"] = (n_final / float(p1)) if p1 else None
        row["brackets_valid"] = bool(all(w.get("control") and w["control"]["bracket_valid"]
                                         for w in windows))
        row["brackets_separate"] = bool(all(w.get("control") and _bracket_ok(w["control"])
                                            for w in windows))
        cum = controls[str(cumulative_T)]
        row["cumulative_midpoint_T"] = cum["midpoint"]
        row["cumulative_below_midpoint_T"] = bool(row["persistence_pooled"] < cum["midpoint"])
        w1b = row["w1_persistence"] is not None and row["w1_midpoint"] is not None
        w4b = row["w4_persistence"] is not None and row["w4_midpoint"] is not None
        row["c2_class"] = (
            None if not (w1b and w4b) else
            ("persistent" if row["w4_persistence"] >= row["w4_midpoint"] else
             ("noisy" if row["w1_persistence"] < row["w1_midpoint"] else "exhausted")))
        row.update(_c3_episode_stats(ep_returns, ep_init_ticks, ep_init_flips))
        row["per_episode_init_flip_ticks"] = ep_init_flips
        row["per_episode_init_live_fresh_ticks"] = ep_init_ticks
        blocks = [float(np.mean(ep_flip_rate[i:i + 100])) for i in range(0, len(ep_flip_rate), 100)]
        row["trained_flip_rate_block_means_100"] = blocks
        row["n_rule_live_ticks"] = n_rule_live_ticks
        row["n_fresh_select_ticks"] = n_fresh_ticks
        row["n_latched_ticks"] = n_latched_ticks
        row["n_fresh_flip_ticks"] = n_fresh_flip_ticks
        row["summary_dispatch_calls"] = summary_counters["dispatch"]
        row["summary_fallback_calls"] = summary_counters["fallback"]
        row["cell_ready"] = bool(row["brackets_separate"]
                                 and (row["update_realisation"] or 0.0) >= MIN_UPDATE_REALISATION)
        cell.stamp(row)
    print(f"verdict: {'PASS' if row['cell_ready'] else 'FAIL'}", flush=True)
    return row


def run_experiment(episodes: Dict[str, int], checkpoints: List[int], cumulative_T: int,
                   dry_run: bool) -> Dict[str, Any]:
    t0 = time.perf_counter()
    anchors = _assert_anchors()
    zg = ZGoalStreamAccumulator()
    rows = [_run_cell(seed, episodes, zg, checkpoints, cumulative_T) for seed in SEEDS]
    n = max(len(rows), 1)

    sep_frac = sum(1 for r in rows if r["brackets_separate"]) / n
    worst_real, worst_real_cell = x1020._worst(rows, "update_realisation", "min")
    worst_fb, worst_fb_cell = x1020._worst(rows, "summary_fallback_calls", "max")
    worst_ticks, worst_ticks_cell = x1020._worst(rows, "n_measured_ticks", "min")
    r_sep = bool(sep_frac >= x1020.CONTROL_READY_FRACTION_FLOOR)
    r_real = bool(worst_real >= MIN_UPDATE_REALISATION)
    r_dispatch = bool(worst_fb <= 0.0)
    r_ticks = bool(worst_ticks >= x1020.FLIP_TICKS_FLOOR)
    ready = bool(r_sep and r_real and r_dispatch and r_ticks)

    n_c1 = sum(1 for r in rows if r["median_grad_norm"] > x1020.GRAD_NORM_FLOOR)
    n_noisy = sum(1 for r in rows if r["c2_class"] == "noisy")
    n_exh = sum(1 for r in rows if r["c2_class"] == "exhausted")
    n_pers = sum(1 for r in rows if r["c2_class"] == "persistent")
    n_c4 = sum(1 for r in rows if r["return_variance"] > x1020.RETURN_VAR_FLOOR)
    n_scoreable = sum(1 for r in rows if r["c3_scoreable"])
    n_neg = sum(1 for r in rows if r["c3_negative"])
    n_pos = sum(1 for r in rows if r["c3_positive"])
    c1 = n_c1 >= SEED_MAJORITY
    c4 = n_c4 >= SEED_MAJORITY
    c2 = n_noisy >= SEED_MAJORITY
    c3_ok = bool(c4 and n_scoreable >= C3_MIN_SCOREABLE_SEEDS)
    c3 = bool(c3_ok and n_neg >= SEED_MAJORITY)

    if c2:
        c2_part = "noisy"
    elif n_exh >= SEED_MAJORITY:
        c2_part = "directed_then_exhausted"
    elif n_pers >= SEED_MAJORITY:
        c2_part = "persistent_directed"
    else:
        c2_part = "mixed"
    if not c4:
        c3_part = "degenerate"
    elif n_scoreable < C3_MIN_SCOREABLE_SEEDS:
        c3_part = "starved"
    elif c3:
        c3_part = "sign_negative"
    elif n_pos >= SEED_MAJORITY:
        c3_part = "sign_positive"
    else:
        c3_part = "sign_null"
    if not ready:
        label = "substrate_not_ready_requeue"
    elif not c1:
        label = "training_signal_absent_no_gradient"
    else:
        label = f"c2_{c2_part}__c3_{c3_part}"
    outcome = "PASS" if (ready and c1) else "FAIL"

    worst_grad, worst_grad_cell = x1020._worst(rows, "median_grad_norm", "min")
    preconditions = [
        {"name": "control_bracket_separates_at_every_window_length", "kind": "readiness",
         "description": ("fraction of cells whose synthetic-credit ceiling exceeds the pure-noise "
                         "floor by >= BRACKET_SEPARATION_FLOOR at EVERY window length"),
         "control": "1020's two in-run controls, imported, at each window's length",
         "measured": sep_frac, "threshold": x1020.CONTROL_READY_FRACTION_FLOOR,
         "direction": "lower", "met": r_sep},
        {"name": "updates_realised", "kind": "readiness",
         "description": "worst cell realised REINFORCE updates / intended P1 updates",
         "measured": worst_real, "threshold": MIN_UPDATE_REALISATION, "direction": "lower",
         "offending_cell": worst_real_cell, "met": r_real},
        {"name": "candidate_summary_dispatch_engaged", "kind": "readiness",
         "description": "proposer_post_action supplies every summary (0 manual fallbacks)",
         "measured": worst_fb, "threshold": 0.0, "direction": "upper",
         "offending_cell": worst_fb_cell, "met": r_dispatch},
        {"name": "flip_fraction_resolvable", "kind": "readiness",
         "description": "worst cell measured P1 ticks >= FLIP_TICKS_FLOOR",
         "measured": worst_ticks, "threshold": float(x1020.FLIP_TICKS_FLOOR),
         "direction": "lower", "offending_cell": worst_ticks_cell, "met": r_ticks},
    ]
    criteria = [
        {"name": "C1_gradient_present", "load_bearing": True, "passed": bool(c1),
         "measured": float(n_c1), "threshold": float(SEED_MAJORITY), "comparator": ">=",
         "worst_median_grad_norm": worst_grad, "per_cell_floor": x1020.GRAD_NORM_FLOOR,
         "offending_cell": worst_grad_cell},
        {"name": "C2_windowed_persistence_noisy", "load_bearing": True, "passed": bool(c2),
         "measured": float(n_noisy), "threshold": float(SEED_MAJORITY), "comparator": ">=",
         "hypothesis": "H-learning-signal-noisy",
         "n_exhausted": n_exh, "n_persistent": n_pers,
         "per_seed": {str(r["seed"]): {"class": r["c2_class"], "w1": r["w1_persistence"],
                                        "w1_mid": r["w1_midpoint"], "w4": r["w4_persistence"],
                                        "w4_mid": r["w4_midpoint"]} for r in rows}},
        {"name": "C3_episode_advantage_on_init_flips_negative", "load_bearing": True,
         "passed": bool(c3), "measured": float(n_neg), "threshold": float(SEED_MAJORITY),
         "comparator": ">=", "hypothesis": "H-learning-signal-sign",
         "n_scoreable": n_scoreable, "n_positive": n_pos,
         "per_seed": {str(r["seed"]): {"r_local": r["c3_r_local"], "r_grand": r["c3_r_grand"],
                                        "se_band": r["c3_se_band"],
                                        "scoreable": r["c3_scoreable"]} for r in rows}},
        {"name": "C4_return_variance_present", "load_bearing": False, "passed": bool(c4),
         "measured": float(n_c4), "threshold": float(SEED_MAJORITY), "comparator": ">=",
         "per_cell_floor": x1020.RETURN_VAR_FLOOR},
        {"name": "R1020_C3_trained_head_drawn_samples_recorded", "load_bearing": False,
         "threshold_not_applicable": "1020's statistic, recorded for comparability only",
         "per_seed_adv_flip_minus_nonflip": {str(r["seed"]): r["adv_flip_minus_nonflip"]
                                              for r in rows},
         "per_seed_n_flip_samples": {str(r["seed"]): r["n_flip_samples"] for r in rows}},
    ]
    combination_rule = ("outcome PASS iff readiness AND C1. C2 (windowed H-learning-signal-noisy) "
                        "and C3 (episode-level H-learning-signal-sign) are INDEPENDENT legs "
                        "(criteria_aggregation any); the label carries both readings. C3 is "
                        "scored only with C4 and >= 3 scoreable seeds.")
    non_degen = {
        "C1_gradient_present": bool(all(r["n_p1_updates"] > 0 for r in rows)),
        "C2_windowed_persistence_noisy": bool(all(r["brackets_valid"] for r in rows) and r_real),
        "C3_episode_advantage_on_init_flips_negative": bool(c3_ok),
        "C4_return_variance_present": bool(all(len(r["per_episode_returns"]) >= 2 for r in rows)),
    }
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "unknown",
        "evidence_direction_note": "diagnostic fan-out legs; directions pinned unknown",
        "outcome": outcome,
        "timestamp_utc": ts,
        "dry_run": bool(dry_run),
        "compares_against_run_id": COMPARES_AGAINST_RUN_ID,
        "fanout_qid": FANOUT_QID,
        "fanout_hypotheses": FANOUT_HYPOTHESES,
        "fanout_axes": FANOUT_AXES,
        "fanout_source_autopsy": FANOUT_SOURCE_AUTOPSY,
        "source_chip_ref": SOURCE_CHIP_REF,
        "red_team_verdict": RED_TEAM_VERDICT,
        "criteria": criteria,
        "combination_rule": combination_rule,
        "interpretation": {
            "label": label,
            "criteria_aggregation": "any",
            "preconditions": preconditions,
            "criteria_non_degenerate": non_degen,
            "combination_rule": combination_rule,
            "unexcluded_rivals": [{
                "name": "H-replay-rule-state-mismatch", "excluded_by_this_run": False,
                "owned_by": "V3-EXQ-1027",
                "description": "a C2 noisy reading does not exclude the replay rule_state mismatch"}],
        },
        "arm_results": rows,
        "diagnostics": {"anchor_reachability": anchors},
    }
    readout: Dict[str, Any] = {
        "readiness_met": ready, "c1_gradient_present": c1, "c2_noisy": c2,
        "c3_sign_negative": c3, "c4_return_variance_present": c4,
        "n_c2_noisy": n_noisy, "n_c2_exhausted": n_exh, "n_c2_persistent": n_pers,
        "n_c3_scoreable": n_scoreable, "n_c3_negative": n_neg, "n_c3_positive": n_pos,
        "bracket_separation_fraction": sep_frac, "worst_update_realisation": worst_real,
    }
    for r in rows:
        s = r["seed"]
        readout[f"w1_persistence_seed{s}"] = r["w1_persistence"]
        readout[f"w1_midpoint_seed{s}"] = r["w1_midpoint"]
        readout[f"w4_persistence_seed{s}"] = r["w4_persistence"]
        readout[f"w4_midpoint_seed{s}"] = r["w4_midpoint"]
        readout[f"c3_r_local_seed{s}"] = r["c3_r_local"]
        readout[f"c3_se_band_seed{s}"] = r["c3_se_band"]
    manifest["readout"] = flat_readout(readout)
    manifest["_zg"] = zg
    manifest["_t0"] = t0
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        episodes = {"p0": 4, "p1": 12, "steps": 12}
        checkpoints, cumulative_T = [3, 6], 12
    else:
        episodes = {"p0": P0_WARMUP_EPISODES, "p1": P1_BIAS_TRAIN_EPISODES,
                    "steps": STEPS_PER_EPISODE}
        checkpoints, cumulative_T = list(CHECKPOINTS), CUMULATIVE_CONTROL_T
    manifest = run_experiment(episodes, checkpoints, cumulative_T, args.dry_run)
    zg = manifest.pop("_zg")
    t0 = manifest.pop("_t0")
    out_path = write_flat_manifest(
        manifest, dry_run=args.dry_run,
        config={"arm": ARM, "episodes": episodes, "checkpoints": checkpoints,
                "cumulative_T": cumulative_T, **_config_slice(episodes["p1"]),
                "bracket_separation_floor": BRACKET_SEPARATION_FLOOR,
                "local_baseline_half_width": LOCAL_BASELINE_HALF_WIDTH,
                "r_se_multiplier": R_SE_MULTIPLIER},
        seeds=SEEDS, script_path=Path(__file__), started_at=t0,
        z_goal_stream_stats=zg.stats(),
    )
    print(f"manifest: {out_path}", flush=True)
    print(f"label: {manifest['interpretation']['label']}", flush=True)
    print(f"outcome: {manifest['outcome']}", flush=True)
    return manifest, out_path, args.dry_run


if __name__ == "__main__":
    _manifest, _out_path, _dry = main()
    _o = str(_manifest["outcome"]).upper()
    emit_outcome(outcome=_o if _o in ("PASS", "FAIL") else "FAIL",
                 manifest_path=_out_path, dry_run=_dry)
