"""V3-EXQ-1089: MECH-268 closure-cadence dose -- MODE-REGISTER DV, reset ON/OFF bracket.

Purpose: evidence. Tests MECH-268's registered PURPOSE -- that dACC PE saturation lets
the SD-032a mode register relax out of internal_planning -- in the LIVE loop of a
TRAINED agent, with SD-034 closure cadence as the ecological dose and the closure ->
dACC-FIFO coupling (ClosureOperator reset_outcome_history) as the bracketing control.

RED-TEAM (fable, Step 4.5): CONTESTED -> all findings fixed. F1 no-headroom routed to
  non_contributory (was weakens) + headroom precondition; F2 C3 redefined on floor-arm
  release fractions; F3 C2 relabelled manipulation-consistency (sign entailed by its gate);
  F4 K_SHORT 3 -> 6 (fire timing diverged ON vs OFF at K=3); F5 closure/FIFO bookkeeping on
  every tick; F6 scope note (constant FIFO class; shared pe stream).

WHY THIS DESIGN (provenance, do not re-derive):
  * chip-20260917-mech268-closure-cadence-dose + its 2026-09-24 PRE-FLIGHT VERDICT
    (AMBER, named changes 1-3) + REE_assembly docs/architecture/
    mech_268_dacc_saturation_form.md (c12a11eafb9) + GFLAG-0327 / GFLAG-0330.
  * Orchestrator decision orchestrate-20260924-0808 "DECIDED Q-MECH268 -> A": the
    chip's recommended DV (sat-rung distribution) is FIXED ARITHMETICALLY by the
    closure-interval histogram (harm_class_fraction 0.946-1.000 -> n_rec at a fresh
    tick = min(ticks since last FIFO reset, W)), the same gate-entails-criterion shape
    GFLAG-0330 refused V3-EXQ-1051 for. So the sat-rung distribution is recorded ONLY
    as a manipulation / consistency check, and the LOAD-BEARING criterion is the MODE
    REGISTER (internal_planning occupancy; mode_switch_trigger rate reported), with a
    reset ON/OFF bracket inside each cadence arm. Strength 0.5 is primary; 0.3 is the
    replication of the form doc's single-seed 0.3/0.5 boundary (decision
    dec-20260923T185804-MECH-268 is waiting on exactly that multi-seed replication).
  * Thresholds were pre-registered from a 1-seed PILOT of this driver (--pilot) plus
    the SD-of-delta margin rule (margin = max(K_SD * pstdev(delta), ABS_FLOOR)).

MECHANISM (source, ree-v3):
  f_sat = 1/(1 + s * max(0, n_rec - G)), n_rec = count of the current class in the last
  W outcomes (dacc.py _saturation_factor); FIFO written once per E3 tick by the
  select_action tail (agent.py record_outcome site, gated on dacc_saturation_enabled);
  cleared by agent.reset() (dacc.reset) and by ClosureOperator._fire() when
  config.reset_outcome_history is True (closure_operator.py step (e)).
  pe (post-saturation) -> SalienceCoordinator.tick as dacc_pe: affinity
  {internal_planning: 1.0, internal_replay: 0.5} against external_task_bias 1.0, and
  salience_weights dacc_pe 1.0 against switch_threshold 1.0.

THE RESET ON/OFF BRACKET (the attribution control, pre-flight CHANGE 1/2):
  A closure fire does five things at once (beta release, No-Go, residue discharge,
  closure_event salience signal, pe_ema reset) PLUS the FIFO reset. A cadence dose
  alone cannot attribute a mode-register effect to MECH-268. Within each cadence arm the
  ONLY difference between ON and OFF is closure_operator.config.reset_outcome_history,
  so ON-OFF isolates the closure -> FIFO -> f_sat path.
  NOTE: REEConfig.closure_reset_outcome_history is NEVER READ (agent.py builds
  ClosureOperatorConfig without it), so the OFF arm patches the built operator's
  config post-build and ASSERTS every fired ClosureEvent has outcome_history_reset
  False (ON arms: True).

PREDICTION (pre-registered): ON keeps re-entering f_sat at 1.0 on every closure, so
  dacc_pe stays high and argmax(operating_mode) stays internal_planning; OFF lets the
  FIFO fill so f_sat reaches its floor (0.25 at s=0.5) and pe drops below the ~1.0
  critical value, releasing the register. So occupancy(ON) > occupancy(OFF) at s=0.5
  (C1, load-bearing); the gap shrinks when closures are rarer (C2, cadence
  manipulation-consistency, reported); and the s=0.5 floor arm releases more ticks than the
  s=0.3 floor arm (C3, boundary replication, reported: the s=0.3 floor 0.357 sits above the
  ~0.275 requirement).
  VERDICT GRID: not ready -> non_contributory (substrate_not_ready_requeue); no headroom
  (S050_KS_OFF release < 0.05 on too many seeds) -> non_contributory
  (calibration_no_headroom_s050); C1 pass -> supports; C1 mean <= 0 with headroom ->
  weakens (closure resets do not re-engage the register); else mixed.

WHY C1 CAN FAIL ON A GATE-GREEN RUN (GFLAG-0330 lesson): the gates certify only that
  closures fire, that the FIFO is populated, and that there is internal_planning to
  release (SATOFF arm). Whether the attenuated pe actually crosses the register's
  critical value in the closed loop depends on the LIVE pe_unsaturated scale (measured
  ~0.99 untrained, ~3.6 trained, ~16-17 on another config -- form doc Line 1), on
  dacc_foraging / dacc_difficulty, and on how often closure resets in practice. At
  pe_unsat ~16 even the floor leaves pe 4x above critical (both arms internal_planning,
  delta 0 -> weakens); at pe_unsat < 1 both arms sit in external_task (the readiness
  gate catches this). None of that is fixed by the manipulation.

CONFOUND CONTROLS:
  * closure_signal_affinity_internal_planning = 0.0 in EVERY arm (pre-flight CHANGE 2:
    closure_event is written once and never cleared until SalienceCoordinator.reset(),
    a sticky +0.5 internal_planning logit after the first fire). Zero also matches the
    operating point the form doc's 0.275 requirement was measured at (729 fixture, no
    closure operator). Per-episode first-fire tick is still recorded.
  * Training is shared per seed (one P0 run, saturation OFF), snapshotted via
    state_dict + e3._running_variance (a plain float, NOT in state_dict -- it sets
    precision, which scales pe). Every arm builds a FRESH agent and loads the snapshot,
    so no non-parameter state (FIFO, closure detector, BetaGate latch, salience mode)
    carries across arms. Eval env seed is pinned per seed and identical across arms.
  * pe_cap_after_closure left None (it writes dacc.config in place).
  * E3 latch idiom: dacc._last_pe_unsaturated and agent._salience_last_tick are cleared
    immediately before select_action; a tick with no fresh dACC forward records nothing
    and increments n_latched_ticks (true denominator auditable).
  * record_outcome spy installed BEFORE the eval loop.
  * dacc_weight stays 0.0 (dACC bias into E3 is the zero vector): the DV is the mode
    register, NEVER E3 candidate selection (Consumer A is argmin-invariant structurally,
    form doc design constraint 1).

DV-SYMMETRY DECLARATION (per arm): the DV is a per-tick OCCUPANCY of one mode label
  (argmax of a softmax over mode logits). The manipulation (FIFO reset on/off) changes
  the MAGNITUDE of one logit input (dacc_pe) relative to a FIXED competitor
  (external_task_bias), not a uniform shift across all mode logits, so it is not
  invariant under the argmax's shift symmetry; it is not a monotone rescaling of all
  logits; and it does not permute interchangeable units. Strength arms rescale the same
  single input. The SATOFF arm is the reference, not a treatment.

Arms (all eval-only; one shared training per seed):
  SATOFF        saturation disabled, cadence K_SHORT          (readiness positive control)
  S050_KS_ON/OFF  s=0.5, closure_stable_ticks=K_SHORT(6), reset ON / OFF
  S050_KL_ON/OFF  s=0.5, closure_stable_ticks=K_LONG(24), reset ON / OFF
  (training: one P0 per seed at the production-default closure_stable_ticks 3)
  S030_KS_ON/OFF  s=0.3, K_SHORT, reset ON / OFF
  S030_KL_ON/OFF  s=0.3, K_LONG,  reset ON / OFF
Seeds: 5. One verdict line per seed.

Eval schedule: EVAL_EPISODES "lives" of EVAL_STEPS_PER_EP env steps per arm. agent.reset()
at each life start; on env death the env respawns WITHOUT agent.reset (V3-EXQ-729
continuous-exposure convention), so the FIFO / closure detector / salience state persist
within a life. (Pilot 1 with per-death agent.reset gave ~14-step lives and ZERO closure
fires -- rule_state never stayed stable for 3 ticks.)

PILOT (seed 42, 4 lives x 200 steps/arm, same training as the real run; P0 converged at
30 eps, rv 1.1e-6, precision_norm 3.0 = capped):
  pe_unsaturated p10/50/90 = 3.78 / 4.15 / 4.40 (identical in every arm: dacc_weight 0
  means the manipulation never changes behaviour, so the env/pe stream is shared).
  occ_ip_argmax: SATOFF 1.000;  s=0.5 ON/OFF at K=3 0.917/0.674 (delta 0.243),
  K=6 0.909/0.680 (0.229), K=12 0.818/0.689 (0.129), K=24 0.729/0.684 (0.045);
  s=0.3 K=3 1.000/1.000 (delta 0.000).  Closure fires per 760 fresh ticks: 146-163 at K=3,
  73 at K=6, 36 at K=12, 18 at K=24; median inter-reset interval = K.  Floor-arm pe_post
  p50 1.04 (s=0.5) -- just above the ~1.0 critical value, which is why OFF releases only
  ~1/3 of ticks: the operating point sits near the knife-edge the form doc predicts.
  Thresholds set from these: C1 floor 0.10 (~40% of the pilot delta), C2 floor 0.05
  (pilot 0.198), C3 floor 0.05 (pilot 0.243); K_SD 2.0 on the across-seed SD of delta.

DISCRETE REGISTER (current_mode) AND mode_switch_trigger ARE STRUCTURALLY PINNED HERE --
recorded, NOT load-bearing. Pilot: occ_ip_current = 1.000 and switch_rate = 0.0053
(the life-start entries only) in EVERY arm incl. SATOFF. Source reason
(salience_coordinator.py tick): a switch out of internal_planning needs
salience_aggregate > enter_threshold (1.0 x stability) AND argmax(operating_mode) ==
external_task. With aic_salience 0, salience_aggregate = dacc_pe; argmax flips to
external_task only when dacc_pe + foraging + 0.5*difficulty < 1 + drive_level. With
drive_level ~0 those two conditions are mutually exclusive, so the discrete register can
never exit internal_planning by dacc_pe alone -- saturation or not. That is an SD-032a
exit-deadlock property independent of MECH-268 (it pins the SATOFF reference too), so it
cannot discriminate this manipulation and is reported as a finding, not scored.

Run:   /opt/local/bin/python3 experiments/v3_exq_1089_mech268_closure_cadence_mode_register.py
Smoke: ... --dry-run          Pilot: ... --pilot --pilot-out <path>
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.committed_mode_curriculum import run_p0_warmup  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell, reset_all_rng  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._metrics import check_degeneracy  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_1089_mech268_closure_cadence_mode_register"
QUEUE_ID = "V3-EXQ-1089"
CLAIM_IDS = ["MECH-268"]
EXPERIMENT_PURPOSE = "evidence"
SEEDS = [42, 43, 44, 45, 46]

# --- MECH-268 saturation knobs -------------------------------------------------
SAT_WINDOW = 8
SAT_GRACE = 2
STRENGTH_PRIMARY = 0.5
STRENGTH_REPL = 0.3
CONTEXTUAL_SAFETY_HARM_THRESHOLD = 0.05   # substrate default, reported only

# --- Cadence dose (closure_stable_ticks = lower bound on inter-fire interval in E3
# ticks). Levels chosen from the pilot so realised inter-RESET intervals straddle W.
K_SHORT = 6        # RED-TEAM F4: at K=3 fire TIMING diverged ON vs OFF on 130/760 pilot
                   # ticks (soft operating_mode -> sd_033a write gate -> lateral-PFC EMA
                   # -> rule_state stability), so the bracket was not a pure FIFO
                   # contrast; at K=6/12/24 fire timing agreed on 760/760.
TRAIN_STABLE_TICKS = 3   # training uses the production-default cadence (arm-independent)
K_LONG = 24        # PILOT-SET: median inter-reset 24 E3 ticks (3W) in the pilot
CLOSURE_AFFINITY_IP = 0.0   # sticky closure_event affinity removed (CHANGE 2)

# --- Env geometry (V3-EXQ-729 lineage) ------------------------------------------
ENV_SIZE = 7
ENV_RESOURCES = 2
ENV_WAYPOINTS = 1
TRAIN_HAZARDS = 3
EVAL_HAZARDS = 6

# --- Schedule --------------------------------------------------------------------
P0_BUDGET = 120
P0_STEPS_PER_EP = 200
P0_PROBE_INTERVAL = 10
EVAL_EPISODES = 6
EVAL_STEPS_PER_EP = 200
# smoke
P0_BUDGET_SMOKE = 3
P0_STEPS_SMOKE = 20
EVAL_EPISODES_SMOKE = 1
EVAL_STEPS_SMOKE = 30
# pilot (1 seed, same training as the real run, short eval, wider cadence scan)
EVAL_EPISODES_PILOT = 4
PILOT_CADENCES = [3, 6, 12, 24]   # pilot scan (pilot S030 arms used K=3)

# --- Pre-registered criteria (constants; PILOT-SET values, see queue note) --------
K_SD = 2.0                  # margin = max(K_SD * pstdev(delta over ready seeds), floor)
C1_ABS_FLOOR = 0.10         # s=0.5, K_SHORT: occ_ip(ON) - occ_ip(OFF)
C2_ABS_FLOOR = 0.05         # s=0.5: delta(K_SHORT) - delta(K_LONG)
C3_ABS_FLOOR = 0.05         # K_SHORT: release(S050 OFF) - release(S030 OFF)  [release = 1 - occ_ip]
# RED-TEAM F1 headroom gate (run level, intended n): the floor arm must actually be able to
# release the register. Without it C1's delta is 0 by arithmetic (shared pe stream), which
# is a CALIBRATION null (pe_unsat scale vs the ~1.0 critical value), never MECH-268 evidence.
R_HEADROOM_RELEASE_MIN = 0.05   # release fraction in S050_KS_OFF, per seed
# Readiness (per seed) -- denominated on the INTENDED seed count.
R_SATOFF_IP_FLOOR = 0.50    # SATOFF argmax internal_planning occupancy (something to release)
R_MIN_FIRES = 5             # closure fires per ON/OFF arm over the eval
R_MIN_FRESH = 50            # fresh E3 ticks per arm
MIN_READY_SEEDS = 4         # of len(SEEDS)=5
# C2 applies only if the cadence arms actually separate around W (manipulation check):
C2_KS_MEDIAN_RESET_INTERVAL_MAX = SAT_WINDOW   # K_SHORT ON: median inter-reset < W
C2_KL_MEDIAN_RESET_INTERVAL_MIN = SAT_WINDOW   # K_LONG ON: median inter-reset > W


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _pstdev(xs: List[float]) -> float:
    return float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _median(xs: List[float]) -> Optional[float]:
    return float(statistics.median(xs)) if xs else None


# ----------------------------------------------------------------------------------
# Arms
# ----------------------------------------------------------------------------------
def _arm(name: str, sat: bool, strength: float, k: int, reset: bool) -> dict:
    return {"name": name, "sat_enabled": sat, "strength": strength,
            "stable_ticks": k, "reset_outcome_history": reset}


def main_arms() -> List[dict]:
    arms = [_arm("SATOFF", False, STRENGTH_PRIMARY, K_SHORT, True)]
    for s, stag in ((STRENGTH_PRIMARY, "S050"), (STRENGTH_REPL, "S030")):
        for k, ktag in ((K_SHORT, "KS"), (K_LONG, "KL")):
            for reset, rtag in ((True, "ON"), (False, "OFF")):
                arms.append(_arm(f"{stag}_{ktag}_{rtag}", True, s, k, reset))
    return arms


def pilot_arms() -> List[dict]:
    arms = [_arm("SATOFF", False, STRENGTH_PRIMARY, 3, True)]
    for k in PILOT_CADENCES:
        for reset, rtag in ((True, "ON"), (False, "OFF")):
            arms.append(_arm(f"S050_K{k}_{rtag}", True, STRENGTH_PRIMARY, k, reset))
    for reset, rtag in ((True, "ON"), (False, "OFF")):
        arms.append(_arm(f"S030_K3_{rtag}", True, STRENGTH_REPL, 3, reset))
    return arms


# ----------------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------------
def _env(seed: int, hazards: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed, size=ENV_SIZE, num_hazards=hazards,
        num_resources=ENV_RESOURCES, num_waypoints=ENV_WAYPOINTS,
    )


def _train_env_seed(seed: int) -> int:
    return int(seed) * 1000 + 1


def _eval_env_seed(seed: int) -> int:
    return int(seed) * 1000 + 7


def _build_agent(world_obs_dim: int, arm: Optional[dict]) -> REEAgent:
    """REEAgent with the full MECH-268 arming chain. arm=None -> training config
    (saturation OFF, default cadence)."""
    sat = bool(arm["sat_enabled"]) if arm else False
    strength = float(arm["strength"]) if arm else STRENGTH_PRIMARY
    k = int(arm["stable_ticks"]) if arm else TRAIN_STABLE_TICKS
    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=world_obs_dim,
        action_dim=4,
        use_dacc=True,
        use_affective_harm_stream=True,
        use_salience_coordinator=True,
        use_lateral_pfc_analog=True,
        use_closure_operator=True,
        closure_stable_ticks=k,
        closure_signal_affinity_internal_planning=CLOSURE_AFFINITY_IP,
        dacc_saturation_enabled=sat,
        dacc_saturation_window=SAT_WINDOW,
        dacc_saturation_strength=strength,
        dacc_saturation_grace=SAT_GRACE,
    )
    cfg.heartbeat.beta_gate_bistable = True
    # from_dims can swallow unknown kwargs silently -- assert the knobs landed.
    assert cfg.use_closure_operator is True
    assert int(cfg.closure_stable_ticks) == k
    assert float(cfg.closure_signal_affinity_internal_planning) == CLOSURE_AFFINITY_IP
    assert bool(cfg.dacc_saturation_enabled) == sat
    assert float(cfg.dacc_saturation_strength) == strength
    agent = REEAgent(cfg)
    assert agent.closure_operator is not None, "closure operator not built"
    assert agent.dacc is not None and agent.salience is not None
    assert int(agent.closure_operator.config.completion_stable_ticks) == k
    assert bool(agent.dacc.config.dacc_saturation_enabled) == sat
    assert float(agent.dacc.config.dacc_saturation_strength) == strength
    assert int(agent.dacc.config.dacc_saturation_window) == SAT_WINDOW
    assert int(agent.dacc.config.dacc_saturation_grace) == SAT_GRACE
    if arm is not None:
        # REEConfig.closure_reset_outcome_history is never read -> patch the operator.
        agent.closure_operator.config.reset_outcome_history = bool(
            arm["reset_outcome_history"]
        )
    return agent


# ----------------------------------------------------------------------------------
# Training (shared per seed)
# ----------------------------------------------------------------------------------
def train_seed(seed: int, device: torch.device, p0_budget: int, p0_steps: int) -> dict:
    reset_all_rng(seed)
    env = _env(_train_env_seed(seed), TRAIN_HAZARDS)
    agent = _build_agent(env.world_obs_dim, None).to(device)
    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        p0 = run_p0_warmup(
            agent, env, device, budget=p0_budget, steps_per_episode=p0_steps,
            probe_interval=P0_PROBE_INTERVAL,
        )
    snap = {k: v.detach().clone() for k, v in agent.state_dict().items()}
    rv = float(agent.e3._running_variance)
    info = {
        "p0_converged": bool(p0.converged), "p0_aborted": bool(p0.aborted),
        "p0_abort_reason": p0.abort_reason, "p0_episodes": int(p0.n_episodes),
        "p0_final_rv": rv, "precision": float(agent.e3.current_precision),
        "precision_norm": float(min(agent.e3.current_precision
                                    / float(agent.dacc.config.dacc_precision_scale), 3.0)),
        "p0_probe_log": list(p0.probe_log),
        "p0_seconds": round(time.perf_counter() - t0, 1),
        "world_obs_dim": int(env.world_obs_dim),
    }
    return {"state_dict": snap, "running_variance": rv, "info": info}


# ----------------------------------------------------------------------------------
# Eval (one arm)
# ----------------------------------------------------------------------------------
def _install_spies(agent: REEAgent) -> dict:
    rec = {"n": 0, "classes": []}
    orig_rec = agent.dacc.record_outcome

    def _rec_spy(outcome_class: int) -> None:
        rec["n"] += 1
        rec["classes"].append(int(outcome_class))
        return orig_rec(outcome_class)

    agent.dacc.record_outcome = _rec_spy
    clo = {"events": []}
    orig_tick = agent.closure_operator.tick

    def _tick_spy(*a, **kw):
        evt = orig_tick(*a, **kw)
        clo["events"].append(evt)
        return evt

    agent.closure_operator.tick = _tick_spy
    return {"rec": rec, "clo": clo}


def eval_arm(agent: REEAgent, seed: int, arm: dict, device: torch.device,
             n_episodes: int, steps_per_ep: int, progress: dict) -> dict:
    agent.eval()
    world_dim = agent.config.latent.world_dim
    spies = _install_spies(agent)          # BEFORE the eval loop
    env = _env(_eval_env_seed(seed), EVAL_HAZARDS)
    ticks_rows: List[dict] = []
    n_latched = 0
    n_env_steps = 0
    fire_reasons: Dict[str, int] = {}
    fired_reset_flags: List[bool] = []
    reset_intervals: List[int] = []        # completed inter-FIFO-reset intervals (fresh ticks)
    fire_intervals: List[int] = []         # completed inter-closure-fire intervals
    first_fire_tick: List[Optional[int]] = []
    n_episodes_done = 0
    n_respawns = 0
    n_fires_on_latched = 0
    stable_hist: Dict[str, int] = {}
    with torch.no_grad():
        for ep in range(n_episodes):
            _, obs_dict = env.reset()
            agent.reset()
            j_since_reset = 0              # record_outcome calls since last FIFO reset
            t_since_fire = None
            t_since_reset = 0
            fresh_i = 0
            ep_first_fire = None
            for _step in range(steps_per_ep):
                obs_body = obs_dict["body_state"].to(device)
                obs_world = obs_dict["world_state"].to(device)
                obs_harm_a = obs_dict["harm_obs_a"].to(device) if "harm_obs_a" in obs_dict else None
                latent = agent.sense(obs_body, obs_world, obs_harm_a=obs_harm_a)
                ticks = agent.clock.advance()
                e1_prior = (agent._e1_tick(latent) if ticks.get("e1_tick")
                            else torch.zeros(1, world_dim, device=device))
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                _rs_prev = agent.lateral_pfc.rule_state.detach().clone()
                n_ev_before = len(spies["clo"]["events"])
                n_rec_before = spies["rec"]["n"]
                agent.dacc._last_pe_unsaturated = None      # E3-latch clear
                agent._salience_last_tick = None
                action = agent.select_action(candidates, ticks)
                n_env_steps += 1
                # RED-TEAM F5: closure/record bookkeeping on EVERY tick (a fire or a
                # FIFO write on a latched tick must still be counted); rows only on
                # fresh ticks.
                new_evts = spies["clo"]["events"][n_ev_before:]
                fired = [e for e in new_evts if e.fired]
                for e in new_evts:
                    key = e.reason.split("(")[0]
                    fire_reasons[key] = fire_reasons.get(key, 0) + 1
                    if "(" in e.reason:
                        stable_hist[e.reason] = stable_hist.get(e.reason, 0) + 1
                fresh = agent.dacc._last_pe_unsaturated is not None
                if not fresh:
                    n_latched += 1
                    if fired:
                        n_fires_on_latched += 1
                else:
                    sal = agent._salience_last_tick or {}
                    om = sal.get("operating_mode") or {}
                    argmax_mode = max(om.items(), key=lambda kv: kv[1])[0] if om else None
                    bundle = agent._dacc_last_bundle or {}
                    row = {
                        "ep": ep, "i": fresh_i,
                        "pe_unsat": float(agent.dacc._last_pe_unsaturated),
                        "pe": float(bundle.get("pe", float("nan"))),
                        # READ FROM THE BUNDLE, not dacc._last_*: on a tick where
                        # closure fires, reset_outcome_history() (called AFTER this
                        # tick's forward) overwrites _last_saturation_factor=1.0 /
                        # _last_outcome_recurrence=0 -- the pilot caught exactly
                        # that (closed-form mismatches == fire count).
                        "sat": float(bundle.get("saturation_factor", float("nan"))),
                        "n_rec": int(bundle.get("outcome_recurrence", -1)),
                        "j_pred": int(min(j_since_reset, SAT_WINDOW)),
                        "argmax_mode": argmax_mode,
                        "current_mode": sal.get("current_mode"),
                        "p_ip": float(om.get("internal_planning", 0.0)),
                        "trigger": bool(sal.get("mode_switch_trigger", False)),
                        "sal_agg": float(sal.get("salience_aggregate", 0.0)),
                        "foraging": float(bundle.get("foraging_value", 0.0)),
                        "fired": bool(fired),
                        "beta_elev": bool(agent.beta_gate.is_elevated),
                        "rule_delta": float((agent.lateral_pfc.rule_state.detach()
                                             - _rs_prev).norm().item()),
                    }
                    ticks_rows.append(row)
                    t_since_reset += 1
                    if t_since_fire is not None:
                        t_since_fire += 1
                    fresh_i += 1
                # counters AFTER this tick (forward preceded record + closure)
                if spies["rec"]["n"] > n_rec_before:
                    j_since_reset += spies["rec"]["n"] - n_rec_before
                if fired:
                    for e in fired:
                        fired_reset_flags.append(bool(e.outcome_history_reset))
                    if t_since_fire is not None:
                        fire_intervals.append(t_since_fire)
                    t_since_fire = 0
                    if ep_first_fire is None:
                        ep_first_fire = fresh_i
                    if any(e.outcome_history_reset for e in fired):
                        reset_intervals.append(t_since_reset)
                        t_since_reset = 0
                        j_since_reset = 0
                _, _, done, _, obs_dict = env.step(int(action.argmax(dim=-1).item()))
                if done:
                    # respawn inside the life: FIFO / closure detector / salience
                    # mode persist (V3-EXQ-729 continuous-exposure convention)
                    _, obs_dict = env.reset()
                    n_respawns += 1
            first_fire_tick.append(ep_first_fire)
            n_episodes_done += 1
            progress["ep"] += 1
            print(f"  [train] seed={seed} arm={arm['name']} ep {progress['ep']}/{progress['total']}",
                  flush=True)
    out = _summarise(arm, ticks_rows, n_latched, n_env_steps, spies, fire_reasons,
                     fired_reset_flags, reset_intervals, fire_intervals, first_fire_tick)
    out["n_respawns"] = int(n_respawns)
    out["n_fires_on_latched_ticks"] = int(n_fires_on_latched)
    out["n_closure_fires_all_ticks"] = int(len(fired_reset_flags))
    out["closure_skip_reason_detail_hist"] = stable_hist
    return out


def _frac(rows: List[dict], pred) -> Optional[float]:
    return (sum(1 for r in rows if pred(r)) / len(rows)) if rows else None


def _summarise(arm, rows, n_latched, n_env_steps, spies, fire_reasons, fired_reset_flags,
               reset_intervals, fire_intervals, first_fire_tick) -> dict:
    n = len(rows)
    pe_un = [r["pe_unsat"] for r in rows]
    floor = 1.0 / (1.0 + float(arm["strength"]) * (SAT_WINDOW - SAT_GRACE))
    classes = spies["rec"]["classes"]
    rung_hist: Dict[str, int] = {}
    for r in rows:
        rung_hist[str(r["n_rec"])] = rung_hist.get(str(r["n_rec"]), 0) + 1
    fired_n = sum(1 for r in rows if r["fired"])
    out = {
        "arm": arm["name"], "arm_spec": dict(arm),
        "n_fresh_ticks": n, "n_latched_ticks": int(n_latched), "n_env_steps": int(n_env_steps),
        "occ_ip_argmax": _frac(rows, lambda r: r["argmax_mode"] == "internal_planning"),
        "occ_ip_current": _frac(rows, lambda r: r["current_mode"] == "internal_planning"),
        "occ_ext_argmax": _frac(rows, lambda r: r["argmax_mode"] == "external_task"),
        "switch_rate": _frac(rows, lambda r: r["trigger"]),
        "mean_p_ip": _mean([r["p_ip"] for r in rows]) if rows else None,
        "pe_unsat_p10": sorted(pe_un)[int(0.1 * (n - 1))] if n else None,
        "pe_unsat_p50": _median(pe_un),
        "pe_unsat_p90": sorted(pe_un)[int(0.9 * (n - 1))] if n else None,
        "pe_post_p50": _median([r["pe"] for r in rows]),
        "frac_pe_post_below_1": _frac(rows, lambda r: r["pe"] < 1.0),
        "mean_sat": _mean([r["sat"] for r in rows]) if rows else None,
        "frac_sat_at_floor": _frac(rows, lambda r: abs(r["sat"] - floor) < 1e-9),
        "sat_floor": floor,
        "n_rec_hist": rung_hist,
        "n_rec_matches_closed_form": _frac(rows, lambda r: r["n_rec"] == r["j_pred"]),
        "mean_foraging": _mean([r["foraging"] for r in rows]) if rows else None,
        "rule_delta_p10": sorted([r["rule_delta"] for r in rows])[int(0.1 * (n - 1))] if n else None,
        "rule_delta_p50": _median([r["rule_delta"] for r in rows]),
        "frac_rule_delta_below_closure_thr": _frac(rows, lambda r: r["rule_delta"] < 0.001),
        "frac_beta_elevated": _frac(rows, lambda r: r["beta_elev"]),
        "n_closure_fires": int(fired_n),
        "closure_fires_per_100_fresh": (100.0 * fired_n / n) if n else None,
        "closure_reason_hist": fire_reasons,
        "fired_outcome_history_reset_all_true": bool(fired_reset_flags) and all(fired_reset_flags),
        "release_frac": (1.0 - _frac(rows, lambda r: r["argmax_mode"] == "internal_planning"))
                        if rows else None,
        "fired_outcome_history_reset_all_false": bool(fired_reset_flags) and not any(fired_reset_flags),
        "reset_intervals": reset_intervals,
        "median_reset_interval": _median(reset_intervals),
        "fire_intervals": fire_intervals,
        "median_fire_interval": _median(fire_intervals),
        "first_fire_tick_per_episode": first_fire_tick,
        "live_record_calls": int(spies["rec"]["n"]),
        "live_record_harm_fraction": (sum(classes) / len(classes)) if classes else None,
        "per_tick": [
            {k: r[k] for k in ("ep", "i", "pe_unsat", "sat", "n_rec", "argmax_mode",
                               "current_mode", "trigger", "fired", "rule_delta")}
            for r in rows
        ],
    }
    return out


# ----------------------------------------------------------------------------------
# Seed / run
# ----------------------------------------------------------------------------------
def _config_slice(arm: dict, sched: dict) -> dict:
    return {
        "env_train": {"size": ENV_SIZE, "num_hazards": TRAIN_HAZARDS,
                      "num_resources": ENV_RESOURCES, "num_waypoints": ENV_WAYPOINTS},
        "env_eval": {"size": ENV_SIZE, "num_hazards": EVAL_HAZARDS,
                     "num_resources": ENV_RESOURCES, "num_waypoints": ENV_WAYPOINTS},
        "env_seed_rule": "train=seed*1000+1, eval=seed*1000+7",
        "schedule": dict(sched),
        "agent": {"body_obs_dim": 12, "action_dim": 4, "use_dacc": True,
                  "use_affective_harm_stream": True, "use_salience_coordinator": True,
                  "use_lateral_pfc_analog": True, "use_closure_operator": True,
                  "beta_gate_bistable": True,
                  "closure_signal_affinity_internal_planning": CLOSURE_AFFINITY_IP,
                  "training_saturation": False,
                  # training build (arm=None) reads these defaults
                  "training_closure_stable_ticks": TRAIN_STABLE_TICKS,
                  "training_strength_default": STRENGTH_PRIMARY},
        "saturation": {"enabled": bool(arm["sat_enabled"]), "window": SAT_WINDOW,
                       "grace": SAT_GRACE, "strength": float(arm["strength"])},
        "closure": {"stable_ticks": int(arm["stable_ticks"]),
                    "reset_outcome_history": bool(arm["reset_outcome_history"])},
    }


def run_seed(seed: int, arms: List[dict], device: torch.device, sched: dict,
             zg: ZGoalStreamAccumulator) -> dict:
    print(f"Seed {seed} Condition closure_cadence_mode_register", flush=True)
    total = sched["p0_budget"] + len(arms) * sched["eval_episodes"]
    cache = sched.get("snapshot_cache")
    if cache and Path(cache).exists():
        tr = torch.load(cache, weights_only=False)
    else:
        tr = train_seed(seed, device, sched["p0_budget"], sched["p0_steps"])
        if cache:
            torch.save(tr, cache)
    progress = {"ep": sched["p0_budget"], "total": total}
    print(f"  [train] seed={seed} P0 ep {progress['ep']}/{total}"
          f" (P0 ran {tr['info']['p0_episodes']} eps, converged={tr['info']['p0_converged']},"
          f" rv={tr['running_variance']:.6f}, prec_norm={tr['info']['precision_norm']:.3f},"
          f" {tr['info']['p0_seconds']}s)", flush=True)
    arm_rows = {}
    for arm in arms:
        with arm_cell(seed, config_slice=_config_slice(arm, sched), script_path=Path(__file__),
                      config_slice_declared=True, include_driver_script_in_hash=False) as cell:
            agent = _build_agent(tr["info"]["world_obs_dim"], arm).to(device)
            agent.load_state_dict(tr["state_dict"], strict=True)
            agent.e3._running_variance = tr["running_variance"]
            assert abs(float(agent.e3._running_variance) - tr["running_variance"]) < 1e-12
            res = eval_arm(agent, seed, arm, device, sched["eval_episodes"],
                           sched["eval_steps"], progress)
            zg.observe(agent)
            cell.stamp(res)
        arm_rows[arm["name"]] = res
        print(f"  [arm {arm['name']}] seed={seed} fresh={res['n_fresh_ticks']}"
              f" latched={res['n_latched_ticks']} occ_ip={res['occ_ip_argmax']}"
              f" occ_ip_cur={res['occ_ip_current']} sw={res['switch_rate']}"
              f" pe_un_p50={res['pe_unsat_p50']} mean_sat={res['mean_sat']}"
              f" fires={res['n_closure_fires']} med_reset_int={res['median_reset_interval']}",
              flush=True)
    return {"seed": seed, "training": tr["info"], "arms": arm_rows}


def _ready(sr: dict) -> dict:
    arms = sr["arms"]
    sat_off = arms["SATOFF"]
    checks = {
        "satoff_ip_occupancy": (sat_off["occ_ip_argmax"] or 0.0) >= R_SATOFF_IP_FLOOR,
        "min_fires_all_sat_arms": all(a["n_closure_fires_all_ticks"] >= R_MIN_FIRES
                                      for k, a in arms.items() if k != "SATOFF"),
        "min_fresh_all_arms": all(a["n_fresh_ticks"] >= R_MIN_FRESH for a in arms.values()),
        "fifo_live_populated": all((a["live_record_calls"] or 0) > 0
                                   for k, a in arms.items() if k != "SATOFF"),
        "reset_flag_on_arms_true": all(a["fired_outcome_history_reset_all_true"]
                                       for k, a in arms.items() if k.endswith("_ON")),
        "reset_flag_off_arms_false": all(a["fired_outcome_history_reset_all_false"]
                                         for k, a in arms.items() if k.endswith("_OFF")),
    }
    return {"ready": all(checks.values()), "checks": checks}


def _delta(sr: dict, on: str, off: str, key: str = "occ_ip_argmax") -> float:
    return float(sr["arms"][on][key] or 0.0) - float(sr["arms"][off][key] or 0.0)


def build_manifest(seed_results: List[dict], arms: List[dict], sched: dict,
                   smoke: bool) -> dict:
    ready_map = {sr["seed"]: _ready(sr) for sr in seed_results}
    ready = [sr for sr in seed_results if ready_map[sr["seed"]]["ready"]]
    n_ready = len(ready)
    intended = 1 if smoke else len(SEEDS)
    min_ready = 1 if smoke else MIN_READY_SEEDS
    gate_green = n_ready >= min_ready

    d_ks5 = [_delta(sr, "S050_KS_ON", "S050_KS_OFF") for sr in ready]
    d_kl5 = [_delta(sr, "S050_KL_ON", "S050_KL_OFF") for sr in ready]
    d_ks3 = [_delta(sr, "S030_KS_ON", "S030_KS_OFF") for sr in ready]
    d_kl3 = [_delta(sr, "S030_KL_ON", "S030_KL_OFF") for sr in ready]
    c2_diff = [a - b for a, b in zip(d_ks5, d_kl5)]
    # RED-TEAM F2: C3 compares the FLOOR arms' release fractions (the form-doc 0.3/0.5
    # boundary is about whether the floor alone releases the register), not the ON-OFF
    # deltas (with S030 pinned, delta(0.3) == 0 and a delta-based C3 restates C1).
    rel5 = [float(sr["arms"]["S050_KS_OFF"]["release_frac"] or 0.0) for sr in ready]
    rel3 = [float(sr["arms"]["S030_KS_OFF"]["release_frac"] or 0.0) for sr in ready]
    c3_diff = [a - b for a, b in zip(rel5, rel3)]
    # RED-TEAM F1: headroom gate on the intended n (all seeds that ran, not only ready ones)
    head_rel = [float(sr["arms"]["S050_KS_OFF"]["release_frac"] or 0.0) for sr in seed_results]
    n_headroom = sum(1 for v in head_rel if v >= R_HEADROOM_RELEASE_MIN)
    headroom_green = n_headroom >= min_ready
    # RED-TEAM F4 manipulation check: fire-TIMING agreement ON vs OFF per cadence pair
    def _fire_agree(sr, on, off):
        a = [r["fired"] for r in sr["arms"][on]["per_tick"]]
        b = [r["fired"] for r in sr["arms"][off]["per_tick"]]
        n = min(len(a), len(b))
        return (sum(1 for x, y in zip(a[:n], b[:n]) if x == y) / n) if n else None
    fire_agree = {pair: [_fire_agree(sr, pair + "_ON", pair + "_OFF") for sr in seed_results]
                  for pair in ("S050_KS", "S050_KL", "S030_KS", "S030_KL")}
    # secondary discrete-register readouts (reported, not gating)
    d_ks5_cur = [_delta(sr, "S050_KS_ON", "S050_KS_OFF", "occ_ip_current") for sr in ready]
    d_ks5_sw = [_delta(sr, "S050_KS_ON", "S050_KS_OFF", "switch_rate") for sr in ready]

    def _crit(name, vals, floor, load_bearing, applies=True, note=""):
        m = _mean(vals) if vals else float("nan")
        margin = max(K_SD * _pstdev(vals), floor)
        passed = bool(applies and gate_green and vals and m > margin)
        return {"name": name, "load_bearing": load_bearing, "applies": bool(applies),
                "measured": m, "threshold": margin, "abs_floor": floor, "k_sd": K_SD,
                "pstdev": _pstdev(vals), "n": len(vals), "per_seed": vals,
                "passed": passed, "note": note}

    ks_int = [sr["arms"]["S050_KS_ON"]["median_reset_interval"] for sr in ready]
    kl_int = [sr["arms"]["S050_KL_ON"]["median_reset_interval"] for sr in ready]
    c2_applies = bool(ks_int and kl_int
                      and all(x is not None and x < C2_KS_MEDIAN_RESET_INTERVAL_MAX for x in ks_int)
                      and all(x is not None and x > C2_KL_MEDIAN_RESET_INTERVAL_MIN for x in kl_int))
    c1 = _crit("C1_reset_bracket_s050_Kshort", d_ks5, C1_ABS_FLOOR, True,
               note="occ_ip_argmax(S050_KS_ON) - occ_ip_argmax(S050_KS_OFF), per ready seed")
    c2 = _crit("C2_cadence_consistency_s050", c2_diff, C2_ABS_FLOOR, False, applies=c2_applies,
               note="MANIPULATION-CONSISTENCY, not evidence (RED-TEAM F3): delta(K_SHORT) - "
                    "delta(K_LONG) at s=0.5. Its SIGN is entailed by its own applies-gate "
                    "(K_SHORT ON median inter-reset < W, K_LONG ON median > W) whenever headroom "
                    "> 0, because floor occupancy grows with the reset interval by construction.")
    c3 = _crit("C3_strength_boundary_release_Kshort", c3_diff, C3_ABS_FLOOR, False,
               note="release_frac(S050_KS_OFF) - release_frac(S030_KS_OFF), release = 1 - "
                    "argmax internal_planning occupancy: multi-seed replication of the form "
                    "doc's single-seed 0.3/0.5 floor boundary (dec-20260923T185804-MECH-268). "
                    "Contingent on the live pe_unsat scale (release needs pe_unsat*floor < ~1).")
    criteria = [c1, c2, c3]

    m1 = c1["measured"]
    if not gate_green:
        outcome, direction, label = "FAIL", "non_contributory", "substrate_not_ready_requeue"
    elif not headroom_green:
        # The floor cannot bring pe below the register's critical value at this operating
        # point: C1 is 0 by arithmetic. A calibration finding (bears on
        # dec-20260923T185804-MECH-268), never a MECH-268 weakens (RED-TEAM F1).
        outcome, direction, label = "FAIL", "non_contributory", "calibration_no_headroom_s050"
    elif c1["passed"]:
        outcome, direction, label = "PASS", "supports", "mode_register_released_by_saturation"
    elif m1 <= 0.0:
        # Reachable only WITH headroom present: the floor releases the register but closure
        # resets do not re-engage it -- the claimed closure -> FIFO reset coupling is not
        # operative in the live loop.
        outcome, direction, label = "FAIL", "weakens", "no_release_reset_bracket_null_or_inverted"
    else:
        outcome, direction, label = "FAIL", "mixed", "positive_but_below_margin_inconclusive"

    all_on_off = [[sr["arms"][a]["occ_ip_argmax"] or 0.0 for a in
                   ("S050_KS_ON", "S050_KS_OFF")] for sr in ready]
    degeneracy = check_degeneracy({
        "occ_ip_argmax_s050_KS_on_off": {"groups": all_on_off} if all_on_off else {"values": [0.0]},
    }) if not smoke else {"non_degenerate": True, "degeneracy_reason": "", "degenerate_metrics": {}}

    def _flat(v):
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (int, float)) and v == v and v not in (float("inf"), float("-inf")):
            return v
        return None

    readout = {
        "n_seeds_ready": n_ready, "n_seeds_intended": intended, "gate_green": int(gate_green),
        "C1_measured": c1["measured"], "C1_threshold": c1["threshold"], "C1_pass": int(c1["passed"]),
        "C2_measured": c2["measured"], "C2_threshold": c2["threshold"], "C2_pass": int(c2["passed"]),
        "C2_applies": int(c2_applies),
        "C3_measured": c3["measured"], "C3_threshold": c3["threshold"], "C3_pass": int(c3["passed"]),
        "delta_s050_KL_mean": _mean(d_kl5) if d_kl5 else None,
        "delta_s030_KS_mean": _mean(d_ks3) if d_ks3 else None,
        "delta_s030_KL_mean": _mean(d_kl3) if d_kl3 else None,
        "delta_s050_KS_occ_current_mean": _mean(d_ks5_cur) if d_ks5_cur else None,
        "delta_s050_KS_switch_rate_mean": _mean(d_ks5_sw) if d_ks5_sw else None,
        "headroom_green": int(headroom_green), "n_seeds_headroom": n_headroom,
        "release_frac_S050_KS_OFF_mean": _mean(head_rel) if head_rel else None,
        "release_frac_S030_KS_OFF_mean": _mean(rel3) if rel3 else None,
    }
    for pair, vals in fire_agree.items():
        vv = [v for v in vals if v is not None]
        readout[f"fire_timing_agreement_{pair}_min"] = min(vv) if vv else None
    for a in arms:
        vals = [sr["arms"][a["name"]]["occ_ip_argmax"] for sr in seed_results]
        vals = [v for v in vals if v is not None]
        readout[f"occ_ip_argmax_{a['name']}_mean"] = _mean(vals) if vals else None
        pes = [sr["arms"][a["name"]]["pe_unsat_p50"] for sr in seed_results]
        pes = [v for v in pes if v is not None]
        readout[f"pe_unsat_p50_{a['name']}_mean"] = _mean(pes) if pes else None
        for key in ("occ_ip_current", "switch_rate", "n_closure_fires",
                    "median_reset_interval", "frac_sat_at_floor", "n_rec_matches_closed_form"):
            vv = [sr["arms"][a["name"]][key] for sr in seed_results]
            vv = [v for v in vv if v is not None]
            readout[f"{key}_{a['name']}_mean"] = _mean(vv) if vv else None
    all_cur = [sr["arms"][a["name"]]["occ_ip_current"] for sr in seed_results for a in arms]
    readout["discrete_register_pinned_all_arms"] = bool(all_cur) and all(
        v is not None and v >= 0.999 for v in all_cur)
    readout = {k: _flat(v) for k, v in readout.items() if _flat(v) is not None}

    preconditions = [
        {"name": "n_seeds_ready_intended", "measured": n_ready, "threshold": min_ready,
         "direction": "lower",
         "control": "per-seed readiness: SATOFF argmax internal_planning occupancy >= "
                    f"{R_SATOFF_IP_FLOOR} (positive control: something to release, same "
                    "statistic as the DV), closures fire >= "
                    f"{R_MIN_FIRES} per saturation arm, FIFO live-populated, reset flags "
                    "correct per arm; denominated on the INTENDED seed count",
         "met": gate_green},
        {"name": "headroom_s050_floor_releases", "measured": n_headroom, "threshold": min_ready,
         "direction": "lower",
         "control": f"seeds whose S050_KS_OFF release fraction >= {R_HEADROOM_RELEASE_MIN} "
                    "(the floor arm can move the DV at all); counted over every seed that ran",
         "met": headroom_green},
    ]
    run_id = f"{EXPERIMENT_TYPE}_{_utc_stamp()}_v3"
    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": _utc_iso(),
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "result": outcome,
        "outcome": outcome,
        "evidence_direction": direction,
        "evidence_direction_per_claim": {"MECH-268": direction},
        "criteria": criteria,
        "combination_rule": "PASS iff gate_green AND headroom_green AND C1 (load-bearing). "
                            "C2 (cadence manipulation-consistency) and C3 (0.3/0.5 floor "
                            "boundary replication) are reported and do not gate the verdict. "
                            "Not ready -> non_contributory; no headroom -> non_contributory "
                            "(calibration); C1 mean<=0 with headroom -> weakens; "
                            "0<mean<=margin -> mixed (inconclusive).",
        "scope_note": "RED-TEAM F6: the FIFO class is constant live (harm_class_fraction ~1.0), "
                      "so a PASS supports only the closure -> FIFO -> f_sat -> mode-register "
                      "path; it is silent on MECH-268's outcome-recurrence / class-transition half. "
                      "dacc_weight=0 means the pe stream is shared across arms (behaviour is not "
                      "changed by the manipulation); C1's sign is fixed by f_ON >= f_OFF on that "
                      "shared stream, its MAGNITUDE (pe_unsat scale vs the ~1.0 critical value, "
                      "and live closure cadence) is the measurement.",
        "fire_timing_agreement": {k: v for k, v in fire_agree.items()},
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": {"C1": bool(degeneracy.get("non_degenerate", True))},
        },
        "readiness_per_seed": {str(k): v for k, v in ready_map.items()},
        "readout": readout,
        "thresholds": {
            "K_SD": K_SD, "C1_ABS_FLOOR": C1_ABS_FLOOR, "C2_ABS_FLOOR": C2_ABS_FLOOR,
            "C3_ABS_FLOOR": C3_ABS_FLOOR, "R_SATOFF_IP_FLOOR": R_SATOFF_IP_FLOOR,
            "R_MIN_FIRES": R_MIN_FIRES, "R_MIN_FRESH": R_MIN_FRESH,
            "MIN_READY_SEEDS": MIN_READY_SEEDS,
            "sat_window": SAT_WINDOW, "sat_grace": SAT_GRACE,
            "strength_primary": STRENGTH_PRIMARY, "strength_repl": STRENGTH_REPL,
            "K_SHORT": K_SHORT, "K_LONG": K_LONG,
            "closure_signal_affinity_internal_planning": CLOSURE_AFFINITY_IP,
            "contextual_safety_harm_threshold": CONTEXTUAL_SAFETY_HARM_THRESHOLD,
        },
        "schedule": dict(sched),
        "arms": [dict(a) for a in arms],
        "arm_results": [dict(sr["arms"][a["name"]], seed=sr["seed"])
                        for sr in seed_results for a in arms],
        "per_seed_training": {str(sr["seed"]): sr["training"] for sr in seed_results},
        "ethics_preflight": {
            "involves_negative_valence": False, "involves_suffering_like_state": False,
            "involves_self_model": False, "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False, "decision": "allow",
        },
        "smoke": smoke,
        "notes": "MECH-268 closure-cadence dose, mode-register DV, reset ON/OFF bracket. "
                 "See module docstring for provenance (chip-20260917-mech268-closure-"
                 "cadence-dose pre-flight; orchestrator decision Q-MECH268 -> A).",
    }
    manifest.update(degeneracy)
    return manifest


def main(mode: str, pilot_out: Optional[str]) -> Optional[tuple]:
    device = torch.device("cpu")
    t0 = time.perf_counter()
    if mode == "smoke":
        seeds, arms = SEEDS[:1], main_arms()
        sched = {"p0_budget": P0_BUDGET_SMOKE, "p0_steps": P0_STEPS_SMOKE,
                 "eval_episodes": EVAL_EPISODES_SMOKE, "eval_steps": EVAL_STEPS_SMOKE}
    elif mode == "pilot":
        seeds, arms = SEEDS[:1], pilot_arms()
        sched = {"p0_budget": P0_BUDGET, "p0_steps": P0_STEPS_PER_EP,
                 "eval_episodes": EVAL_EPISODES_PILOT, "eval_steps": EVAL_STEPS_PER_EP,
                 "snapshot_cache": str(Path(pilot_out).with_suffix(".snapshot.pt"))}
    else:
        seeds, arms = SEEDS, main_arms()
        sched = {"p0_budget": P0_BUDGET, "p0_steps": P0_STEPS_PER_EP,
                 "eval_episodes": EVAL_EPISODES, "eval_steps": EVAL_STEPS_PER_EP}
    zg = ZGoalStreamAccumulator()
    seed_results = []
    for s in seeds:
        sr = run_seed(s, arms, device, sched, zg)
        seed_results.append(sr)
        if mode == "pilot":
            print("verdict: FAIL", flush=True)   # pilot has no verdict
            continue
        rd = _ready(sr)
        ok = rd["ready"] and _delta(sr, "S050_KS_ON", "S050_KS_OFF") > 0.0
        print(f"verdict: {'PASS' if ok else 'FAIL'}", flush=True)

    if mode == "pilot":
        slim = [{"seed": sr["seed"], "training": sr["training"],
                 "arms": {k: {kk: vv for kk, vv in v.items() if kk != "per_tick"}
                          for k, v in sr["arms"].items()}} for sr in seed_results]
        Path(pilot_out).parent.mkdir(parents=True, exist_ok=True)
        Path(pilot_out).write_text(json.dumps({"pilot": slim, "per_tick": {
            k: v["per_tick"] for k, v in seed_results[0]["arms"].items()}}, indent=1, default=str))
        print(f"pilot written to {pilot_out} ({time.perf_counter() - t0:.0f}s)", flush=True)
        return None

    manifest = build_manifest(seed_results, arms, sched, smoke=(mode == "smoke"))
    print(f"=== {QUEUE_ID} {EXPERIMENT_TYPE} ===", flush=True)
    print(f"outcome: {manifest['outcome']} direction={manifest['evidence_direction']}"
          f" label={manifest['interpretation']['label']}"
          f" C1={manifest['criteria'][0]['measured']:.4f} vs {manifest['criteria'][0]['threshold']:.4f}",
          flush=True)
    full_config = {"schedule": sched, "arms": arms, "thresholds": manifest["thresholds"]}
    out_path = write_flat_manifest(
        manifest, None, dry_run=(mode == "smoke"), config=full_config, seeds=list(seeds),
        script_path=Path(__file__), started_at=t0, z_goal_stream_stats=zg.stats(),
    )
    print(f"Result written to: {out_path}", flush=True)
    return manifest["outcome"], out_path, manifest["run_id"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Smoke run (tiny budgets).")
    parser.add_argument("--pilot", action="store_true",
                        help="1-seed pilot (real training, short eval, cadence scan); "
                             "writes JSON to --pilot-out, no manifest, no outcome.")
    parser.add_argument("--pilot-out", default=None)
    args = parser.parse_args()
    if args.pilot:
        assert args.pilot_out, "--pilot requires --pilot-out"
        main("pilot", args.pilot_out)
        sys.exit(0)
    result = main("smoke" if args.dry_run else "full", None)
    _outcome, _out_path, _run_id = result
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path, run_id=_run_id, queue_id=QUEUE_ID,
        exit_reason="ok" if _outcome == "PASS" else "fail", dry_run=args.dry_run,
    )
    sys.exit(0)
