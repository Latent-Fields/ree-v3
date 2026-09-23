"""V3-EXQ-1075 -- SD-PP-B5 validation: does the action-margin loss make
e2.world_forward read its action?

PURPOSE: diagnostic (substrate-readiness validation for SD-PP-B5). Non-contributory
to governance confidence by design.

SLEEP DRIVER: not applicable (no sleep machinery; P0/P1 world_forward training only).

RED-TEAM (Step 4.5): see the queue entry note for V3-EXQ-1075.

WHAT THIS VALIDATES
-------------------
V3-EXQ-1073 measured on seeds 42/123/456 that the converged e2.world_forward head
does NOT read its action: skill vs the copy-the-input predictor
-0.071/+0.227/-0.007 (MECH-573), and an INVERTED-action-map battery was EASIER
than the original rule (ratio 0.760/0.901/0.881, all < 1). An action-map inversion
is therefore not a contradiction on that head, and its condition 3 was unposeable
(self-route confidently_wrong_condition_unposeable_on_this_head).

SD-PP-B5 (landed 2026-09-22, ree-v3 8f10214) ported SD-013's contrastive
interventional margin loss to that head. This run asks the only question that
matters for the substrate: at matched budget, does turning the term ON move the
head's action-sensitivity across the readability bar, WITHOUT destroying
reconstruction (the V3-EXQ-701b direction)?

DESIGN
------
5 arms x 3 seeds = 15 cells. The ONLY difference between arms is the margin term
in the P0 objective; rollout, budget, optimiser regime and battery are identical.
That invariance is enforced by TWO rng streams: the rollout/battery/batch stream is
seeded identically in every arm, and the interventional-fraction coin has its own
stream, so an ON arm does not shift the shared stream and silently train on a
different trajectory than OFF.

The DV is the INVERTED-ACTION-MAP battery ratio -- the same instrument V3-EXQ-1073
measured (0.760/0.901/0.881) -- not an action-shuffle of the original battery. The
two are different measurements: under shuffle the same OFF condition reads ~1.05,
ABOVE the 1.0 bar that 1073's inverted-rule values were what justified.

  ARM_OFF        use_world_interventional=False   (must reproduce 1073's blindness)
  ARM_ON_M001    margin 0.01     ladder rung
  ARM_ON_M005    margin 0.05     ladder rung
  ARM_ON_M010    margin 0.10     ladder rung (the config default)
  ARM_ON_M050    margin 0.50     ladder rung

The ladder exists because the margin is in z_world L2 units and z_world moves only
~3e-3 RMS/dim per step on this env: a margin far above the natural displacement
would buy separation by wrecking reconstruction, which is the failure this run must
be able to SEE rather than assume away. A3/A4 are the guards that see it, and they
are read against rungs meeting A1 ALONE -- an earlier draft conditioned them on
A1+A2 and that made them DEAD CODE, since A2 (model_mse < identity_mse ~1.3e-5) is
strictly tighter than A3 (~7.4e-5) and A4 (~4.8e-5), so any rung that could trip
the guard had already failed A2 and was labelled something else.

DECLARED FAILURE ROUTES (both directions, per CLAUDE.md)
--------------------------------------------------------
TWO AXES, deliberately separated -- only ONE of them gates.
  AXIS 1  action-sensitivity (A1, the inverted-rule ratio). What the margin loss
          actually targets, the only axis it can move, and the ONLY axis that
          gates PASS. This driver's OFF arm reproduces V3-EXQ-1073 on it:
          0.833/0.879 here against 0.760/0.901 there.
  AXIS 2  readability (A2, skill vs the trivial predictor -- MECH-573). RECORDED,
          NOT GATED (user decision rec-20260923-cb59ede6). It is bounded by how
          far z_world moves per step, i.e. by the ENCODER, which this loss cannot
          touch -- AND this driver's OFF arm does NOT reproduce 1073 on it
          (-5.79/-2.24 here against -0.071/+0.227 there; cause not identified, the
          obs_harm hypothesis was tested and eliminated). Gating on an axis whose
          own control does not reproduce its reference would charge the encoder
          for a regime difference, which is exactly the error this design was
          revised to avoid. MECH-573 readability is therefore deferred to the
          encoder work and this run tags NO claim.

  PASS          A1 met on >= 2/3 seeds at >= 1 ON rung, with A3+A4 intact there.
                The margin loss makes an action-map inversion a contradiction
                again, so a contradiction-based consolidation test becomes
                posable on this head.
  FAIL-a        A1 unmet at EVERY rung -> the margin FORM did not raise
                action-sensitivity. A finding about SD-013's shape on this head.
                Routes to the ENCODER remedy (SD-018-amend / SD-009 / SD-106) as
                a new substrate entry -- the escalation branch of the user's
                2026-09-22 scope decision.
  FAIL-b        A1 met but A3/A4 violated -> action-sensitivity bought by
                destroying reconstruction: the V3-EXQ-701b direction reproduced on
                the MARGIN form, which would falsify the "margin goes silent once
                separated" rationale for preferring it over SD-056's InfoNCE form.
  UNDETERMINED  any non-degeneracy precondition unmet (N1/N2/N3). NOT a FAIL.
                Self-routes substrate_not_ready_requeue.

DV-SYMMETRY (Step 3.5, per arm): the DV is a RATIO of two MSEs on a fixed frozen
battery. The manipulation is an added training-time loss term that changes the
head's PARAMETERS, so it is not invariant under any symmetry of that DV: the
battery, its targets and the evaluation are bit-identical across arms, and only the
trained weights differ. A uniform additive constant on the predictions would cancel
in neither MSE (MSE is not shift-invariant in the residual), and no monotone
reparameterisation is applied to either numerator or denominator.

WHY NOT REANALYSIS (Step 2.4, GOV-REUSE-1): the decisive readout is
action_sensitivity_ratio on a head trained WITH the margin term. That term did not
exist in any substrate before ree-v3 8f10214 (2026-09-22), so no recorded manifest
on any substrate_hash can carry it. Checked 1073 and 1063 (the only runs carrying
identity_predictor_mse): both are OFF-arm only. Not recoverable -> run.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_protocol import emit_outcome
from experiments._lib.action_sensitivity_gate import (
    check_canary, format_verdict, identity_predictor_mse, readiness_verdict)
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiments.pack_writer import write_flat_manifest
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

EXPERIMENT_PURPOSE = "diagnostic"
EXPERIMENT_TYPE = "v3_exq_1075_sdppb5_action_sensitivity_validation"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
# DELIBERATELY EMPTY. An earlier draft tagged MECH-573, but after A2 (skill vs the
# trivial predictor) was WITHDRAWN as a gate this run no longer tests MECH-573's
# CONFIRMING clause -- that needs a matched-conv_rel_drop comparison between a
# skill>0 and a skill<=0 base, which this design does not perform. The skill score
# is still RECORDED per cell as a descriptive readout. Tagging a claim the run does
# not test corrupts governance confidence, so it is not tagged.
CLAIM_IDS: List[str] = []
VALIDATES_SUBSTRATE = "SD-PP-B5-z-world-per-step-displacement-range"

# Every anchor precondition below is computed by the SAME shipped code that
# produces the verdicts it anchors (readiness_verdict / check_canary), never by a
# hand-written re-implementation of them, so the "predicate narrower than the state
# it anchors to" failure mode has no surface here: there is no second scoring
# predicate that could be narrower. The counting anchors (row count, distinct-action
# count, cells returning cannot_determine, OFF cells classified action_blind) are
# reachable by construction at any non-degenerate battery, and the canary anchor IS
# the degeneracy definition.
ANCHOR_REACHABILITY_EXEMPT = (
    "anchors are computed by the shipped readiness_verdict/check_canary code, not a "
    "re-implementation; counting anchors are reachable by construction and the canary "
    "anchor is itself the degeneracy definition")

# ---- operating point: matched to V3-EXQ-1073 so ARM_OFF can reproduce it -------
SEEDS = [42, 123, 456]
SELF_DIM = 16
WORLD_DIM = 16
# Every constant below is COPIED from V3-EXQ-1073 (its lines 527-551), because
# N3 asserts ARM_OFF reproduces that run's measurement. An earlier draft used a
# different env and training regime while claiming a match; its OFF arm came back
# at skill -0.83/-0.35 against 1073's -0.071/+0.227/-0.007, which would have
# charged "the encoder" for what was really a regime difference.
GRID_SIZE = 5
N_HAZARDS = 1
N_RESOURCES = 1
P0_STEPS = 3600
STEPS_PER_EPISODE = 90
EPISODES_PER_RUN = P0_STEPS // STEPS_PER_EPISODE      # 40 -- the [train] denominator
BATCH_K = 8
MIN_BUF_BEFORE_TRAIN = 16
BUF_CAP = 4096
LR = 1e-3            # 1073's E2_LR
MAX_GRAD_NORM = 1.0
BATTERY_N = 64
INTERVENTIONAL_FRACTION = 0.3

_ZG = ZGoalStreamAccumulator()

ARMS: List[Tuple[str, Optional[float]]] = [
    ("ARM_OFF", None),
    ("ARM_ON_M001", 0.01),
    ("ARM_ON_M005", 0.05),
    ("ARM_ON_M010", 0.10),
    ("ARM_ON_M050", 0.50),
]

# ---- PRE-REGISTERED thresholds (constants, never derived from this run) --------
# AXIS 1 -- action-sensitivity. What the margin loss actually targets.
RATIO_BAR = 1.0        # A1: inverted-RULE battery MSE must EXCEED original-rule MSE
# AXIS 2 -- readability (MECH-573). RECORDED, NOT GATED (user decision
# rec-20260923-cb59ede6). It is bounded by how far z_world moves per step, i.e. by
# the ENCODER, which this loss cannot touch; and this driver's OFF arm does not
# reproduce V3-EXQ-1073's skill baseline (-5.79/-2.24 here vs -0.071/+0.227 there,
# cause not identified -- the obs_harm hypothesis was tested and eliminated), so a
# skill verdict here would not be comparable to 1073's. Readability is deferred to
# the encoder work. SKILL_BAR is kept only to label the recorded readout.
SKILL_BAR = 0.0        # A2: RECORDED ONLY -- does not gate PASS
# GUARD -- reconstruction. Evaluated against rungs meeting A1 ALONE.
# A2 would subsume both of these (A2 needs mse < identity_mse ~1.3e-5, while A3
# allows ~7.4e-5 and A4 ~4.8e-5), so conditioning the guard on A1+A2 made it dead
# code and the "readability bought at reconstruction cost" route unreachable.
CONV_REL_DROP_BAR = 0.99   # A3
MSE_INFLATION_BAR = 2.0    # A4
SEEDS_REQUIRED = 2         # of 3
MIN_DISTINCT_ACTIONS = 2   # N2
MIN_BATTERY_ROWS = 16      # N2


def _to_b(x: Any, device: Any) -> torch.Tensor:
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)
    return (t.unsqueeze(0) if t.ndim == 1 else t).to(device)


def _make_env(seed: int, invert_action_map: bool = False) -> CausalGridWorldV2:
    env = CausalGridWorldV2(seed=seed, size=GRID_SIZE, num_hazards=N_HAZARDS,
                            num_resources=N_RESOURCES, use_proxy_fields=True)
    if invert_action_map:
        _invert_action_map(env)
    return env


def _invert_action_map(env: CausalGridWorldV2) -> None:
    """Rule R1 = R0 with the action map inverted: swap 0<->1 and 2<->3.

    Copied from V3-EXQ-1073 so the counterfactual battery is the SAME instrument
    whose values this run's N3 precondition cites (0.760/0.901/0.881). An earlier
    draft used an action-SHUFFLE on the original battery instead, which is a
    different measurement: the smoke put the OFF arm at ratio 1.05 under shuffle
    while 1073 measured 0.76-0.88 under rule inversion, so the pre-registered
    1.0 bar was not comparable to the numbers justifying it.
    """
    am = env._action_map
    env._action_map = {
        0: am[1], 1: am[0], 2: am[3], 3: am[2],
        **{k: v for k, v in am.items() if k > 3},
    }


def _config_slice(margin: Optional[float]) -> Dict[str, Any]:
    """Declares ONLY what the P0 world_forward computation reads."""
    return {
        "env": {"size": GRID_SIZE, "num_hazards": N_HAZARDS,
                "num_resources": N_RESOURCES, "use_proxy_fields": True},
        "dims": {"self_dim": SELF_DIM, "world_dim": WORLD_DIM},
        "schedule": {"p0_steps": P0_STEPS, "batch_k": BATCH_K, "lr": LR,
                     "min_buf": MIN_BUF_BEFORE_TRAIN, "max_grad_norm": MAX_GRAD_NORM},
        "margin": {"use_world_interventional": margin is not None,
                   "world_interventional_margin": margin,
                   "world_interventional_fraction":
                       INTERVENTIONAL_FRACTION if margin is not None else None},
    }


def _build(seed: int, margin: Optional[float]) -> Tuple[CausalGridWorldV2, REEAgent]:
    torch.manual_seed(seed)
    env = _make_env(seed)
    kw: Dict[str, Any] = dict(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim, self_dim=SELF_DIM, world_dim=WORLD_DIM)
    if margin is not None:
        kw.update(use_world_interventional=True,
                  world_interventional_margin=float(margin),
                  world_interventional_fraction=INTERVENTIONAL_FRACTION)
    cfg = REEConfig.from_dims(**kw)
    # MECH-307 guard: from_dims silently swallows unknown kwargs. Assert the knob
    # actually landed rather than trusting the call.
    if margin is not None:
        if not getattr(cfg.e2, "use_world_interventional", False):
            raise RuntimeError("from_dims did not thread use_world_interventional")
        if float(getattr(cfg.e2, "world_interventional_margin", -1.0)) != float(margin):
            raise RuntimeError("from_dims did not thread world_interventional_margin")
    return env, REEAgent(cfg)


def _sense_zworld(agent: REEAgent, obs: Dict[str, Any]) -> torch.Tensor:
    """obs_harm is passed, matching V3-EXQ-1073's _sense. Omitting it changes the
    encoder's input and therefore z_world's scale, which moves identity_mse and
    the skill score off 1073's reference even at an otherwise matched config."""
    obs_harm = obs.get("harm_obs", None)
    return agent.sense(_to_b(obs["body_state"], agent.device),
                       _to_b(obs["world_state"], agent.device),
                       obs_harm=(_to_b(obs_harm, agent.device)
                                 if obs_harm is not None else None)
                       ).z_world.detach().reshape(-1).clone()


def _collect_battery(agent: REEAgent, env: CausalGridWorldV2, rng: random.Random,
                     n: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Frozen held-out battery, captured ONCE before P0. Nothing here trains the
    encoder, so these tensors are a fixed latent reference for the whole cell and
    are identical across arms at a given seed by construction."""
    obs = env.reset()
    obs = obs[-1] if isinstance(obs, tuple) else obs
    rows: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    prev: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    guard = 0
    while len(rows) < n and guard < n * 8:
        guard += 1
        z = _sense_zworld(agent, obs)
        if prev is not None and bool(torch.isfinite(z).all()):
            rows.append((prev[0], prev[1], z))
        idx = rng.randrange(env.action_dim)
        a = torch.zeros(env.action_dim, dtype=torch.float32)
        a[idx] = 1.0
        prev = (z, a)
        out = env.step(a.unsqueeze(0).to(agent.device))
        obs = out[-1]
    z0 = torch.stack([r[0] for r in rows])
    acts = torch.stack([r[1] for r in rows])
    z1 = torch.stack([r[2] for r in rows])
    return z0, acts, z1


def _draw_cf(acts: torch.Tensor) -> torch.Tensor:
    """a_cf from the COMPLEMENT of a_actual, per row. REE agents are monostrategy
    for long stretches, so a naive draw collides on most rows and those rows carry
    a constant `margin` with no useful gradient direction."""
    a_dim = acts.shape[-1]
    idx = acts.argmax(-1)
    offset = torch.randint(1, a_dim, idx.shape) if a_dim > 1 else torch.zeros_like(idx)
    cf_idx = (idx + offset) % a_dim
    return F.one_hot(cf_idx, a_dim).float()


def _battery_mse(agent: REEAgent, z0: torch.Tensor, acts: torch.Tensor,
                 z1: torch.Tensor) -> float:
    with torch.no_grad():
        return float(((agent.e2.world_forward(z0, acts) - z1) ** 2).mean().item())


def _run_cell(arm: str, margin: Optional[float], seed: int) -> Dict[str, Any]:
    print(f"Seed {seed} Condition {arm}", flush=True)
    with arm_cell(seed, config_slice=_config_slice(margin),
                  script_path=Path(__file__)) as cell:
        env, agent = _build(seed, margin)
        # TWO streams, deliberately. `rng` drives the rollout, the battery and the
        # batch sampling and is therefore IDENTICAL across arms at a given seed;
        # `coin` draws the interventional-fraction decision and is consumed ONLY
        # by ON arms. Sharing one stream made every ON arm train on a different
        # trajectory than OFF, which silently falsified this script's own claim
        # that the margin term is the only difference between arms (and
        # mis-specified A4's per-seed ON/OFF pairing).
        rng = random.Random(seed)
        coin = random.Random(seed ^ 0x5D99B5)
        z0, acts, z1 = _collect_battery(agent, env, rng, BATTERY_N)
        # Counterfactual battery: SAME seed, SAME collection rng stream, but the
        # env's action map inverted -- V3-EXQ-1073's instrument.
        cf_env = _make_env(seed, invert_action_map=True)
        cf_batt = _collect_battery(agent, cf_env, random.Random(seed), BATTERY_N)
        n_rows = int(z0.shape[0])
        n_distinct = int(torch.unique(acts, dim=0).shape[0])
        id_mse = identity_predictor_mse(z0, z1)
        mse_init = _battery_mse(agent, z0, acts, z1)

        opt = torch.optim.Adam(agent.e2.parameters(), lr=LR)
        buf: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = deque(maxlen=BUF_CAP)
        obs = env.reset()
        obs = obs[-1] if isinstance(obs, tuple) else obs
        prev: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        margin_steps = 0
        margin_loss_sum = 0.0
        for step in range(1, P0_STEPS + 1):
            if step % STEPS_PER_EPISODE == 0:
                ep = step // STEPS_PER_EPISODE
                print(f"  [train] {arm} seed={seed} ep {ep}/{EPISODES_PER_RUN} "
                      f"phase=P0", flush=True)
            z = _sense_zworld(agent, obs)
            if prev is not None and bool(torch.isfinite(z).all()):
                buf.append((prev[0], prev[1], z))
            idx = rng.randrange(env.action_dim)
            a = torch.zeros(env.action_dim, dtype=torch.float32)
            a[idx] = 1.0
            prev = (z, a)
            out = env.step(a.unsqueeze(0).to(agent.device))
            obs = out[-1]

            if len(buf) < MIN_BUF_BEFORE_TRAIN:
                continue
            pool = list(buf)
            batch = pool if len(pool) <= BATCH_K else rng.sample(pool, BATCH_K)
            b0 = torch.stack([t[0] for t in batch]).to(agent.device)
            ba = torch.stack([t[1] for t in batch]).to(agent.device)
            b1 = torch.stack([t[2] for t in batch]).to(agent.device)
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(agent.e2.world_forward(b0, ba), b1)
            if margin is not None and coin.random() < INTERVENTIONAL_FRACTION:
                m = agent.e2.compute_world_interventional_loss(b0, ba, _draw_cf(ba))
                margin_loss_sum += float(m.detach().item())
                margin_steps += 1
                loss = loss + m
            if not math.isfinite(float(loss.detach().item())):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.e2.parameters(), max_norm=MAX_GRAD_NORM)
            opt.step()

        mse_final = _battery_mse(agent, z0, acts, z1)
        conv_rel_drop = (1.0 - mse_final / mse_init) if mse_init > 0 else float("nan")
        verdict = readiness_verdict(
            agent.e2.world_forward, z0, acts, z1,
            counterfactual_battery=cf_batt,
            min_rows=MIN_BATTERY_ROWS, min_distinct_actions=MIN_DISTINCT_ACTIONS,
            ratio_floor=RATIO_BAR,
            # A2 withdrawn as a gate: -inf makes the skill conjunct always true so
            # the verdict turns on the ratio alone. `verdict.skill` is still
            # computed and recorded; only its GATING role is removed.
            skill_floor=float("-inf"))
        print(format_verdict(verdict, f"{arm} seed={seed}"), flush=True)

        row: Dict[str, Any] = {
            "arm": arm, "seed": seed, "margin": margin,
            "gate_status": verdict.status,
            "gate_reason": verdict.reason,
            "action_sensitivity_ratio": verdict.ratio,
            "skill_vs_identity": verdict.skill,
            "battery_mse_init": mse_init,
            "battery_mse_final": mse_final,
            "identity_predictor_mse": id_mse,
            "conv_rel_drop": conv_rel_drop,
            "n_rows": n_rows,
            "n_distinct_actions": n_distinct,
            "margin_steps_applied": margin_steps,
            "margin_loss_mean": (margin_loss_sum / margin_steps) if margin_steps else None,
        }
        cell.stamp(row)
        _ZG.observe(agent)   # AFTER stepping -- reads the counters at call time
    print(f"verdict: {'PASS' if verdict.status == 'ready' else 'FAIL'}", flush=True)
    return row


def _seeds_meeting(rows: List[Dict[str, Any]], key: str, bar: float) -> int:
    n = 0
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv > bar or (fv == math.inf and bar < math.inf):
            n += 1
    return n


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    # 2 seeds, not 1: SEEDS_REQUIRED is a PRE-REGISTERED constant, so a
    # 1-seed smoke makes N3 structurally unsatisfiable and the dry run
    # could never exercise its own PASS route. Thresholds never move.
    seeds = SEEDS[:2] if dry_run else SEEDS
    arms = ARMS[:2] if dry_run else ARMS

    canary = check_canary()
    print(f"[canary] V3-EXQ-1073 replay ok={canary['ok']} "
          f"n_seeds={canary['n_seeds']}", flush=True)

    rows: List[Dict[str, Any]] = []
    for arm, margin in arms:
        for seed in seeds:
            rows.append(_run_cell(arm, margin, seed))

    off_rows = [r for r in rows if r["arm"] == "ARM_OFF"]
    on_arms = sorted({r["arm"] for r in rows if r["arm"] != "ARM_OFF"})

    # --- N1/N2: did the battery let the gate speak, in EVERY cell? --------------
    n_cannot = sum(1 for r in rows if r["gate_status"] == "cannot_determine")
    worst_distinct = min((r["n_distinct_actions"] for r in rows), default=0)
    worst_rows = min((r["n_rows"] for r in rows), default=0)
    # --- N3: the canary at experiment scale -- OFF must reproduce 1073 ----------
    off_blind = sum(1 for r in off_rows if r["gate_status"] == "action_blind")

    off_mse = {r["seed"]: r["battery_mse_final"] for r in off_rows}

    per_arm: Dict[str, Any] = {}
    for arm in on_arms:
        ar = [r for r in rows if r["arm"] == arm]
        a1 = _seeds_meeting(ar, "action_sensitivity_ratio", RATIO_BAR)
        a2 = _seeds_meeting(ar, "skill_vs_identity", SKILL_BAR)
        # Counted per SEED, not per criterion: a rung where seed 42 clears the
        # ratio and seed 123 clears the skill has no seed that is actually
        # readable, and must not read as 2/3 on both.
        both = sum(1 for r in ar
                   if r["action_sensitivity_ratio"] is not None
                   and r["skill_vs_identity"] is not None
                   and float(r["action_sensitivity_ratio"]) > RATIO_BAR
                   and float(r["skill_vs_identity"]) > SKILL_BAR)
        a3 = sum(1 for r in ar if r["conv_rel_drop"] is not None
                 and float(r["conv_rel_drop"]) >= CONV_REL_DROP_BAR)
        infl = [float(r["battery_mse_final"]) / float(off_mse[r["seed"]])
                for r in ar if off_mse.get(r["seed"])]
        a4 = sum(1 for v in infl if v <= MSE_INFLATION_BAR)
        per_arm[arm] = {
            "margin": ar[0]["margin"],
            "A1_ratio_seeds": a1, "A2_skill_seeds": a2,
            "A3_conv_seeds": a3, "A4_mse_seeds": a4,
            "max_mse_inflation": max(infl) if infl else None,
            "A1A2_same_seed": both,
            "action_sensitive": a1 >= SEEDS_REQUIRED,
            "readable_same_seed": both >= SEEDS_REQUIRED,
            "reconstruction_ok": a3 >= SEEDS_REQUIRED and a4 >= SEEDS_REQUIRED,
            "rung_pass": (a1 >= SEEDS_REQUIRED and a3 >= SEEDS_REQUIRED
                          and a4 >= SEEDS_REQUIRED),
        }

    passing = [a for a, v in per_arm.items() if v["rung_pass"]]
    # FAIL-b: the guard is read against rungs that met A1 ALONE. Conditioning it
    # on A1+A2 made it unreachable, because A2 (mse < identity_mse) is strictly
    # tighter than both A3 and A4.
    sensitive_but_broken = [a for a, v in per_arm.items()
                            if v["action_sensitive"] and not v["reconstruction_ok"]]
    # The distinction the old single label could not carry: a margin loss that
    # DID raise action-sensitivity but left the head unreadable is an ENCODER
    # cap (escalate per the 2026-09-22 scope decision), NOT a failure of the
    # margin form -- they route to different places.
    # Descriptive only now that A2 is withdrawn as a gate: a rung that raised the
    # ratio and kept reconstruction, but whose head still does not beat the trivial
    # predictor. Recorded so the encoder escalation has its evidence, but it no
    # longer changes the outcome.
    sensitive_not_readable = [a for a, v in per_arm.items()
                              if v["action_sensitive"] and v["reconstruction_ok"]
                              and not v["readable_same_seed"]]

    preconditions = [
        {"name": "gate_spoke_in_every_cell", "description":
         "no cell returned cannot_determine (N1)",
         "measured": float(n_cannot), "threshold": 0.0, "direction": "upper",
         "control": "the gate's own third value; a monostrategy battery cannot test",
         "met": n_cannot == 0},
        {"name": "battery_distinct_actions", "description":
         "worst cell's distinct-action count (N2 denominator)",
         "measured": float(worst_distinct), "threshold": float(MIN_DISTINCT_ACTIONS),
         "direction": "lower", "control": "random-action battery collection", "met": worst_distinct >= MIN_DISTINCT_ACTIONS},
        {"name": "battery_rows", "description": "worst cell's battery row count (N2)",
         "measured": float(worst_rows), "threshold": float(MIN_BATTERY_ROWS),
         "direction": "lower", "control": "frozen battery captured before P0", "met": worst_rows >= MIN_BATTERY_ROWS},
        {"name": "off_arm_reproduces_1073_blindness", "description":
         "ARM_OFF must be action_blind on >= 2/3 seeds (N3); if OFF comes back "
         "ready the operating point has drifted since 1073 and the ON/OFF "
         "comparison is void",
         "measured": float(off_blind), "threshold": float(SEEDS_REQUIRED),
         "direction": "lower",
         "control": "V3-EXQ-1073 measured ratio 0.760/0.901/0.881, skill "
                    "-0.071/+0.227/-0.007 at this operating point",
         "met": off_blind >= SEEDS_REQUIRED},
        {"name": "canary_replays", "description":
         "the gate's own known-baseline replay reproduces 1073's classification",
         "measured": 1.0 if canary["ok"] else 0.0, "threshold": 1.0,
         "direction": "lower",
         "control": "check_canary() over the landed autopsy values",
         "met": bool(canary["ok"])},
    ]
    all_pre_met = all(p["met"] for p in preconditions)

    if not all_pre_met:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
    elif passing:
        label = "margin_loss_makes_world_forward_action_sensitive"
        outcome = "PASS"
    elif sensitive_but_broken:
        label = "readability_bought_at_reconstruction_cost"   # FAIL-b
        outcome = "FAIL"
    else:
        label = "margin_loss_did_not_raise_action_sensitivity"  # FAIL-a
        outcome = "FAIL"

    best = max(per_arm.values(), key=lambda v: (v["A1_ratio_seeds"], v["A2_skill_seeds"]),
               default=None) if per_arm else None
    criteria = [
        {"name": "A1_action_sensitivity_ratio", "load_bearing": True,
         "measured": float(best["A1_ratio_seeds"]) if best else 0.0,
         "threshold": float(SEEDS_REQUIRED),
         "description": "seeds with ratio > 1.0 at the best ON rung",
         "passed": bool(best and best["A1_ratio_seeds"] >= SEEDS_REQUIRED)},
        {"name": "A2_skill_vs_identity_RECORDED_NOT_GATED", "load_bearing": False,
         "measured": float(best["A1A2_same_seed"]) if best else 0.0,
         "threshold": float(SEEDS_REQUIRED),
         "description": "seeds where the SAME seed clears both the ratio bar and "
                        "skill > 0 at the best ON rung (MECH-573)",
         "passed": bool(best and best["A1A2_same_seed"] >= SEEDS_REQUIRED)},
        {"name": "A3_conv_rel_drop_preserved", "load_bearing": True,
         "measured": float(best["A3_conv_seeds"]) if best else 0.0,
         "threshold": float(SEEDS_REQUIRED),
         "description": "seeds still converging >= 0.99 at the best ON rung",
         "passed": bool(best and best["A3_conv_seeds"] >= SEEDS_REQUIRED)},
        {"name": "A4_mse_not_inflated", "load_bearing": True,
         "measured": float(best["A4_mse_seeds"]) if best else 0.0,
         "threshold": float(SEEDS_REQUIRED),
         "description": "seeds with ON battery MSE <= 2x OFF at the best ON rung",
         "passed": bool(best and best["A4_mse_seeds"] >= SEEDS_REQUIRED)},
    ]
    combination_rule = (
        "PASS iff at least ONE ON rung meets A1 (inverted-rule ratio > 1.0) AND "
        "A3 AND A4, each on >= 2/3 seeds, AND every precondition N1/N2/N3 is met. "
        "A1 met with A3/A4 violated is FAIL-b (action-sensitivity bought by "
        "destroying reconstruction), NOT a pass. A2 (skill vs the trivial "
        "predictor) is RECORDED but does NOT gate: it is encoder-bound and this "
        "driver's OFF arm does not reproduce V3-EXQ-1073's skill baseline, so a "
        "skill verdict here would not be comparable to that run.")

    ratios = [r["action_sensitivity_ratio"] for r in rows
              if r["action_sensitivity_ratio"] is not None]
    non_degenerate = all_pre_met and len(set(
        round(float(v), 9) for v in ratios if math.isfinite(float(v)))) > 1

    flat: Dict[str, float] = {
        "n_cells": float(len(rows)),
        "n_cannot_determine": float(n_cannot),
        "off_arm_action_blind_seeds": float(off_blind),
        "worst_distinct_actions": float(worst_distinct),
        "worst_battery_rows": float(worst_rows),
        "canary_ok": 1.0 if canary["ok"] else 0.0,
        "n_rungs_passing": float(len(passing)),
        "n_rungs_sensitive_but_broken": float(len(sensitive_but_broken)),
        "n_rungs_sensitive_not_readable": float(len(sensitive_not_readable)),
        "n_rungs_action_sensitive": float(sum(1 for v in per_arm.values() if v["action_sensitive"])),
        "best_rung_A1_seeds": float(best["A1_ratio_seeds"]) if best else 0.0,
        "best_rung_A2_seeds": float(best["A2_skill_seeds"]) if best else 0.0,
        "best_rung_A1A2_same_seed": float(best["A1A2_same_seed"]) if best else 0.0,
        "all_preconditions_met": 1.0 if all_pre_met else 0.0,
    }
    for arm in on_arms:
        v = per_arm[arm]
        if v["max_mse_inflation"] is not None and math.isfinite(v["max_mse_inflation"]):
            flat[f"{arm}_max_mse_inflation"] = float(v["max_mse_inflation"])
    flat = {k: v for k, v in flat.items() if v is not None and math.isfinite(v)}

    manifest: Dict[str, Any] = {
        "validates_substrate": VALIDATES_SUBSTRATE,
        "run_id": f"{EXPERIMENT_TYPE}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "non_contributory",
        "outcome": outcome,
        "timestamp_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "sleep_driver_pattern": "not_applicable",
        "non_degenerate": bool(non_degenerate),
        "degeneracy_reason": (None if non_degenerate else
                              "preconditions unmet or ratio identical across all arms"),
        "readout": flat,
        "arm_results": rows,
        "per_arm": per_arm,
        "canary": canary,
        "interpretation": {
            "label": label,
            "combination_rule": combination_rule,
            "preconditions": preconditions,
            "criteria": criteria,
            "criteria_non_degenerate": {
                c["name"]: bool(all_pre_met and len(rows) > 1) for c in criteria},
            "passing_rungs": passing,
            "sensitive_but_broken_rungs": sensitive_but_broken,
            "sensitive_not_readable_rungs": sensitive_not_readable,
        },
    }
    manifest["elapsed_seconds"] = time.perf_counter() - t0
    return manifest, t0


if __name__ == "__main__":
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--dry-run", action="store_true")
    _args = _ap.parse_args()

    _manifest, _t0 = run_experiment(dry_run=_args.dry_run)
    _out_path = write_flat_manifest(
        _manifest, dry_run=_args.dry_run,
        config={"arms": [{"arm": a, "margin": m} for a, m in ARMS],
                "seeds": SEEDS, "p0_steps": P0_STEPS, "world_dim": WORLD_DIM,
                "self_dim": SELF_DIM, "battery_n": BATTERY_N,
                "interventional_fraction": INTERVENTIONAL_FRACTION,
                "env": {"size": GRID_SIZE, "num_hazards": N_HAZARDS,
                        "num_resources": N_RESOURCES, "use_proxy_fields": True}},
        seeds=SEEDS, script_path=Path(__file__), started_at=_t0,
        z_goal_stream_stats=_ZG.stats())

    print(f"[{EXPERIMENT_TYPE}] outcome={_manifest['outcome']} "
          f"label={_manifest['interpretation']['label']}", flush=True)
    _o = str(_manifest["outcome"]).upper()
    emit_outcome(outcome=_o if _o in ("PASS", "FAIL") else "FAIL",
                 manifest_path=_out_path, dry_run=_args.dry_run)
