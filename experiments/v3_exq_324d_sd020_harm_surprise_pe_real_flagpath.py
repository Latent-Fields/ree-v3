"""
V3-EXQ-324d -- SD-020: harm_surprise_pe REAL FLAG-PATH retest (OFF vs ON)

Claims: SD-020   (evidence)
EXPERIMENT_PURPOSE = "evidence"

WHY THIS RUN EXISTS (2026-08-09 governance, GFLAG-0005 demotion). SD-020's
promoting run V3-EXQ-324b was a STANDALONE BENCH REIMPLEMENTATION -- zero
references to REEAgent / harm_surprise_pe_enabled / compute_harm_accum_loss. It
validated an inline PE-vs-EMA comparison against a fresh linear head, never the
shipped flag path, so SD-020 was demoted stable -> candidate. The only run that
DID drive the real flag path, V3-EXQ-324, FAILED at 1/5 seeds -- attributed to
underpowering (10-20 harm events/seed, harm_obs_a_variance ~3e-4 on the EMA
substrate), not a clean falsification. Even 324, on audit, constructs a REEAgent
but its training loop reimplements the PE target manually against a standalone
head and never calls agent.compute_harm_accum_loss() -- so the shipped
implementation of SD-020 has NEVER been cleanly tested. This run closes that gap.

WHAT MAKES THIS THE REAL FLAG PATH (not another bench). Every arm builds a real
REEAgent with use_affective_harm_stream=True + harm_history_len=10, and trains
z_harm_a through the SHIPPED loss agent.compute_harm_accum_loss(accum, latent)
(ree_core/agent.py) via the proven affective_fishtank.warmup_train harness. When
harm_surprise_pe_enabled=True, that loss:
  - maintains agent._harm_obs_ema (updated at harm_obs_ema_alpha=0.1),
  - computes harm_pe = |accumulated_harm - _harm_obs_ema|,
  - scales by precision_norm = min(e3.current_precision/500, 3.0)  (ARC-016),
  - trains latent_state.harm_accum_pred (the AffectiveHarmEncoder head) on that
    surprise target instead of the raw accumulated-harm magnitude.
The PE-BRANCH DIFFERENTIATION gate below proves the branch actually fired:
agent._harm_obs_ema is mutated ONLY inside the PE branch and agent.reset() never
clears it, so after warmup ARM_ON's _harm_obs_ema must be > PE_EMA_FLOOR and
ARM_OFF's must be exactly 0.0. If not, ARM_ON silently fell through to the EMA
target and the manipulation never took (verified in design probes 2026-08-09:
ARM_OFF ema=0.0, ARM_ON ema=0.0159 after a 20-episode warmup).

SUBSTRATE (SD-022, limb_damage_enabled=True). The 2026-04-11 note and the
substrate_queue SD-020 metric_trajectory both prescribe the SD-022 substrate:
with limb_damage_enabled=True the env re-sources harm_obs_a from body damage
state (7 dims: damage[4] + max + mean + residual_pain) instead of the near-zero
proxy field, fixing 324's harm_obs_a_variance ~3e-4 confound. Design probes
2026-08-09 confirm this substrate is WELL-POWERED: 367/400 eval steps carry a
harm signal (>> the 100/seed floor the metric_trajectory asks for), against
324's 10-20/seed.

DECISIVE READOUT + PASS. SD-020 predicts the flag makes z_harm_a encode affective
SURPRISE, not raw accumulated-harm MAGNITUDE. Primary DV is measured on the
head's own output (latent.harm_accum_pred), which is what the target-swap
directly trains and is not masked by z_harm_a-norm saturation:
  surprise_pref = corr(harm_accum_pred, harm_pe) - corr(harm_accum_pred, accum)
Per seed:
  C1  ARM_ON surprise_pref > ARM_OFF surprise_pref      (flag shifts head toward surprise)
  C2  ARM_ON corr(harm_accum_pred, harm_pe) >= SURPRISE_CORR_FLOOR  (head genuinely tracks surprise)
  C3  no collapse: harm_accum_pred varies (hap_std > HAP_STD_FLOOR) and the PE
      branch fired for this seed's ARM_ON cell
Experiment PASS (SD-020 real-path supported) = C1 and C2 and C3 in >= PASS_MIN_SEEDS
of SEEDS.

NON-DEGENERACY (self-routes rather than emit a false verdict). The whole run is
recorded non_degenerate=false / non_contributory (NOT weakens) when any of:
  * PE branch not differentiated  (ARM_OFF ema != 0 on any seed, or ARM_ON ema
    <= PE_EMA_FLOOR on any seed) -> the shipped flag path did not run.
  * underpowered  (any cell < HARM_EVENT_FLOOR harm events) -> 324's confound.
  * DV not measurable  (any ARM_ON cell hap_std <= HAP_STD_FLOOR, i.e. the head
    is a constant, so no correlation can be read).
  * manipulation had zero representational effect  (all seeds
    |ARM_ON pref - ARM_OFF pref| < MANIP_EPS) -> the flag ran but produced no
    measurable change; honest "shipped implementation does not demonstrably do
    what SD-020 claims, on this substrate/schedule" rather than a false weakens.
A degenerate run keeps SD-020 at candidate and is auditable, not a re-promotion
and not a falsification.

DV-SYMMETRY. The manipulation (harm_surprise_pe_enabled) changes the z_harm_a
TRAINING TARGET and hence the learned AffectiveHarmEncoder weights; corr() of the
resulting head output with harm_pe / accum is a function of those weights, NOT
invariant under any broadcast / monotone-rescale / permutation of the latent. So
a nonzero ARM_ON-vs-ARM_OFF preference delta is a genuine measurement, not an
arithmetic identity.

Output: REE_assembly/evidence/experiments/<run_id>.json (flat manifest).
Estimated runtime: ~90 min cloud CPU (2 arms x 5 seeds x 50 warmup + eval x 200
steps, full affective stack).
"""

import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.baselines import affective_fishtank as AF  # noqa: E402


EXPERIMENT_TYPE    = "v3_exq_324d_sd020_harm_surprise_pe_real_flagpath"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS          = ["SD-020"]

# z_goal IS driven -- AF.warmup_train (imported from _lib/baselines/affective_fishtank)
# calls agent.update_z_goal(...) every training step. The dead-z_goal static lint scans
# only this file, which inlines the rich config (so it textually sets z_goal_enabled)
# but delegates stepping to the shared harness (so the writer is not textually here) --
# a false positive, unlike sibling V3-EXQ-856 which also delegates the *config*. The
# stream is recorded via _ZG (z_goal_stream_stats) so a genuine dead stream would still
# be visible. z_goal is inherited substrate scaffolding here, not a DV of this SD-020 run.
DEAD_Z_GOAL_STREAM_EXEMPT = (
    "z_goal driven by AF.warmup_train's update_z_goal (imported harness, not textual "
    "here); inherited affective_fishtank scaffolding, not a DV; recorded via _ZG"
)

SEEDS             = [0, 1, 2, 3, 4]
WARMUP_EPISODES   = 50          # affective_fishtank canonical schedule
EVAL_STEPS        = 400         # matches V3-EXQ-324b eval length
STEPS_PER_EPISODE = 200
HARM_EMA_ALPHA    = 0.1         # must equal cfg.harm_obs_ema_alpha (agent-internal PE ema)

ARMS = [
    {"arm_id": "ARM_OFF", "harm_surprise_pe_enabled": False},  # z_harm_a trained on EMA magnitude
    {"arm_id": "ARM_ON",  "harm_surprise_pe_enabled": True},   # z_harm_a trained on precision-weighted PE
]

# --- pre-registered thresholds (constants; not derived from run statistics) --
SURPRISE_CORR_FLOOR = 0.20      # C2: ARM_ON head genuinely tracks surprise
PE_EMA_FLOOR        = 1e-4      # ARM_ON _harm_obs_ema must exceed this after warmup
HARM_EVENT_FLOOR    = 100       # per-cell harm events (the metric_trajectory power bar)
HAP_STD_FLOOR       = 1e-5      # harm_accum_pred must vary for a correlation to be readable
MANIP_EPS           = 0.02      # |ON pref - OFF pref| below this on ALL seeds = no effect
PASS_MIN_SEEDS      = 3

# SD-022 substrate = the harsh affective_fishtank env + limb_damage re-sourcing
SD022_ENV_KWARGS = dict(AF.HARSH_ENV_KWARGS)
SD022_ENV_KWARGS.update(limb_damage_enabled=True)

# z_goal liveness accumulator (config sets z_goal_enabled=True): observed per cell
# so a dead z_goal stream stays visible in the manifest (V3-EXQ-626/830).
_ZG = ZGoalStreamAccumulator()


# ---------------------------------------------------------------------------
# Agent + env construction (affective_fishtank rich config + SD-022 limb_damage)
# ---------------------------------------------------------------------------

def make_agent_and_env(seed, harm_surprise_pe_enabled):
    """Real REEAgent on the SD-022 limb_damage substrate.

    Ports affective_fishtank.make_agent_and_env's rich config verbatim (so the
    proven warmup_train harness has every feature it calls), and additionally
    passes limb_damage_enabled=True to BOTH the env AND REEConfig.from_dims. The
    from_dims flag is what auto-sets harm_obs_a_dim=7 (config.py); passing it only
    to the env would leave the encoder expecting the 50-dim proxy and shape-mismatch
    at sense(). body/world dims are read back from the env so they always match.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    env = CausalGridWorldV2(seed=seed, **SD022_ENV_KWARGS)

    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=AF.SELF_DIM,
        world_dim=AF.WORLD_DIM,
        harm_dim=AF.HARM_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=env.action_dim,
        use_harm_stream=True,
        z_harm_dim=AF.HARM_DIM,
        use_affective_harm_stream=True,
        z_harm_a_dim=AF.HARM_A_DIM,
        harm_history_len=AF.HARM_HISTORY_LEN,
        limb_damage_enabled=True,                       # -> harm_obs_a_dim auto 7 (SD-022)
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        benefit_eval_enabled=True,
        benefit_weight=1.0,
        z_goal_enabled=True,
        goal_weight=0.5,
        drive_weight=2.0,
        e1_goal_conditioned=True,
        use_tonic_vigor=True,
        use_blocked_agency=True,
        use_pag_freeze_gate=True,
    )
    cfg.e3.commitment_threshold = 0.5
    cfg.heartbeat.beta_gate_bistable = True
    cfg.harm_descending_mod_enabled = True
    cfg.descending_attenuation_factor = 0.5
    cfg.latent.use_harm_un = True
    cfg.pag_max_freeze_duration = AF.PAG_MAX_FREEZE
    cfg.use_broadcast_override = True
    cfg.surprise_gated_replay = True
    cfg.use_mech307_split_surprise = True
    cfg.use_control_vector_logging = True

    # --- the SD-020 axis: shipped precision-weighted-PE training target ---
    cfg.harm_surprise_pe_enabled = bool(harm_surprise_pe_enabled)
    cfg.harm_obs_ema_alpha = HARM_EMA_ALPHA

    agent = REEAgent(cfg)
    return agent, env


def arm_config_slice(harm_surprise_pe_enabled):
    """Declared config slice for this arm's fingerprint. Captures the resolved env
    kwargs, the schedule, and the one flag that differs between arms. The rest of
    the substrate config is fixed for the run and folded into substrate_hash via
    the ree_core glob."""
    return {
        "baseline_id": "sd020_real_flagpath_sd022_v1",
        "lineage": "sd020_harm_surprise_pe_real_flagpath",
        "env_kwargs": dict(SD022_ENV_KWARGS),
        "schedule": {"warmup": WARMUP_EPISODES, "eval_steps": EVAL_STEPS,
                     "steps_per_episode": STEPS_PER_EPISODE},
        "substrate_dims": {
            "world_dim": AF.WORLD_DIM, "self_dim": AF.SELF_DIM,
            "harm_dim": AF.HARM_DIM, "harm_a_dim": AF.HARM_A_DIM,
            "harm_history_len": AF.HARM_HISTORY_LEN, "pag_max_freeze": AF.PAG_MAX_FREEZE,
            "limb_damage_enabled": True,
        },
        "harm_obs_ema_alpha": HARM_EMA_ALPHA,
        "harm_surprise_pe_enabled": bool(harm_surprise_pe_enabled),
    }


# ---------------------------------------------------------------------------
# Eval: surprise-vs-magnitude tracking of the real head + z_harm_a
# ---------------------------------------------------------------------------

def _corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or x.std() < 1e-8 or y.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def eval_surprise_tracking(agent, env, seed, eval_steps=EVAL_STEPS):
    """Per eval step read the REAL agent's latent.harm_accum_pred (head output) and
    latent.z_harm_a norm; compute harm_pe = |accumulated_harm - ema| with the SAME
    alpha the agent's PE branch uses (persistent ema, not reset per episode, matching
    agent._harm_obs_ema semantics). magnitude = accumulated_harm."""
    agent.eval()
    hap_vals = []      # harm_accum_pred (head output)
    zna_vals = []      # z_harm_a norm
    pe_vals = []       # surprise target |accum - ema|
    mag_vals = []      # accumulated_harm (magnitude)
    ema = 0.0
    harm_events = 0

    _, obs = env.reset()
    agent.reset()
    for _ in range(eval_steps):
        oh = obs.get("harm_obs")
        oha = obs.get("harm_obs_a")
        ohh = obs.get("harm_history")
        accum = float(obs.get("accumulated_harm") or 0.0)
        with torch.no_grad():
            latent = agent.sense(
                obs["body_state"], obs["world_state"],
                obs_harm=oh, obs_harm_a=oha, obs_harm_history=ohh,
            )
            zna = float(latent.z_harm_a.norm().item()) if latent.z_harm_a is not None else 0.0
            hap = (float(latent.harm_accum_pred.reshape(-1)[0].item())
                   if latent.harm_accum_pred is not None else 0.0)

        harm_pe = abs(accum - ema)
        ema = (1.0 - HARM_EMA_ALPHA) * ema + HARM_EMA_ALPHA * accum

        hap_vals.append(hap)
        zna_vals.append(zna)
        pe_vals.append(harm_pe)
        mag_vals.append(accum)

        av = torch.zeros(1, env.action_dim)
        av[0, random.randint(0, env.action_dim - 1)] = 1.0
        _, harm_signal, done, _info, obs = env.step(av)
        if float(harm_signal) < 0:
            harm_events += 1
        if done:
            _, obs = env.reset()
            agent.reset()

    zn = np.asarray(zna_vals, dtype=float)
    hap_surprise = _corr(hap_vals, pe_vals)
    hap_mag = _corr(hap_vals, mag_vals)
    zna_surprise = _corr(zna_vals, pe_vals)
    zna_mag = _corr(zna_vals, mag_vals)
    return {
        "hap_surprise_corr": hap_surprise,
        "hap_magnitude_corr": hap_mag,
        "hap_surprise_pref": hap_surprise - hap_mag,
        "zna_surprise_corr": zna_surprise,
        "zna_magnitude_corr": zna_mag,
        "zna_surprise_pref": zna_surprise - zna_mag,
        "hap_std": float(np.std(hap_vals)),
        "zna_cov": float(zn.std() / zn.mean()) if zn.mean() > 1e-9 else 0.0,
        "zna_mean_norm": float(zn.mean()),
        "accum_std": float(np.std(mag_vals)),
        "harm_events": int(harm_events),
        "n_eval_steps": len(hap_vals),
    }


# ---------------------------------------------------------------------------
# Per (arm, seed) cell
# ---------------------------------------------------------------------------

def run_cell(arm, seed, dry_run=False):
    warmup_eps = 3 if dry_run else WARMUP_EPISODES
    steps = 30 if dry_run else STEPS_PER_EPISODE
    eval_steps = 40 if dry_run else EVAL_STEPS

    print(f"Seed {seed} Condition {arm['arm_id']}", flush=True)

    agent, env = make_agent_and_env(seed, arm["harm_surprise_pe_enabled"])
    tag = f" {arm['arm_id']}"
    AF.warmup_train(agent, env, warmup_eps, steps, seed, tag=tag)
    ema_after = float(getattr(agent, "_harm_obs_ema", 0.0))
    prec = float(agent.e3.current_precision)

    ev = eval_surprise_tracking(agent, env, seed, eval_steps=eval_steps)
    _ZG.observe(agent)

    print(f"[324d]{tag} seed={seed} ema={ema_after:.6f} prec={prec:.1f} "
          f"hap_pref={ev['hap_surprise_pref']:+.4f} "
          f"hap_surp={ev['hap_surprise_corr']:+.4f} hap_mag={ev['hap_magnitude_corr']:+.4f} "
          f"hap_std={ev['hap_std']:.6f} harm_events={ev['harm_events']}", flush=True)
    print(f"verdict: {'PASS' if ev['n_eval_steps'] > 0 else 'FAIL'}", flush=True)

    row = {
        "arm_id": arm["arm_id"],
        "seed": seed,
        "harm_surprise_pe_enabled": bool(arm["harm_surprise_pe_enabled"]),
        "harm_obs_ema_after_warmup": ema_after,
        "e3_current_precision_after_warmup": prec,
        "precision_norm_after_warmup": float(min(prec / 500.0, 3.0)),
    }
    row.update(ev)
    return row


# ---------------------------------------------------------------------------
# Aggregate + verdict
# ---------------------------------------------------------------------------

def _seed_pass(on_row, off_row):
    c1 = on_row["hap_surprise_pref"] > off_row["hap_surprise_pref"]
    c2 = on_row["hap_surprise_corr"] >= SURPRISE_CORR_FLOOR
    c3 = (on_row["hap_std"] > HAP_STD_FLOOR
          and on_row["harm_obs_ema_after_warmup"] > PE_EMA_FLOOR)
    return bool(c1 and c2 and c3), {"C1_pref_higher": bool(c1),
                                    "C2_abs_surprise_corr": bool(c2),
                                    "C3_no_collapse": bool(c3)}


def run(seeds=None, dry_run=False):
    if seeds is None:
        seeds = SEEDS
    import time
    t0 = time.perf_counter()

    print(f"[V3-EXQ-324d] SD-020 real flag-path retest (SD-022 substrate)\n"
          f"  Seeds: {seeds}  Arms: {[a['arm_id'] for a in ARMS]}\n"
          f"  SURPRISE_CORR_FLOOR={SURPRISE_CORR_FLOOR} PE_EMA_FLOOR={PE_EMA_FLOOR} "
          f"HARM_EVENT_FLOOR={HARM_EVENT_FLOOR} MANIP_EPS={MANIP_EPS}", flush=True)

    arm_results = []
    for arm in ARMS:
        for seed in seeds:
            slice_ = arm_config_slice(arm["harm_surprise_pe_enabled"])
            with arm_cell(
                seed,
                config_slice=slice_,
                script_path=Path(__file__),
                config_slice_declared=True,
            ) as cell:
                row = run_cell(arm, seed, dry_run=dry_run)
                cell.stamp(row)
            arm_results.append(row)

    off = {r["seed"]: r for r in arm_results if r["arm_id"] == "ARM_OFF"}
    on = {r["seed"]: r for r in arm_results if r["arm_id"] == "ARM_ON"}

    # --- non-degeneracy gates -------------------------------------------------
    off_ema_zero = all(r["harm_obs_ema_after_warmup"] == 0.0 for r in off.values())
    on_ema_live = all(r["harm_obs_ema_after_warmup"] > PE_EMA_FLOOR for r in on.values())
    pe_differentiated = bool(off_ema_zero and on_ema_live)

    well_powered = all(r["harm_events"] >= HARM_EVENT_FLOOR for r in arm_results)
    dv_measurable = all(r["hap_std"] > HAP_STD_FLOOR for r in on.values())

    seed_flags = {}
    seed_passes = 0
    manip_deltas = []
    for s in seeds:
        if s not in on or s not in off:
            continue
        ok, flags = _seed_pass(on[s], off[s])
        seed_flags[s] = flags
        seed_passes += int(ok)
        manip_deltas.append(abs(on[s]["hap_surprise_pref"] - off[s]["hap_surprise_pref"]))
    manip_had_effect = any(d >= MANIP_EPS for d in manip_deltas)

    non_degenerate = True
    ng_label = None
    degeneracy_reason = None
    if not pe_differentiated:
        non_degenerate = False
        ng_label = "pe_branch_not_differentiated"
        degeneracy_reason = (
            f"shipped flag path did not run cleanly: ARM_OFF _harm_obs_ema all zero="
            f"{off_ema_zero}, ARM_ON _harm_obs_ema all>{PE_EMA_FLOOR}={on_ema_live}."
        )
    elif not well_powered:
        non_degenerate = False
        ng_label = "underpowered_harm_events"
        degeneracy_reason = (
            f"a cell recorded < {HARM_EVENT_FLOOR} harm events (324's confound); "
            f"min={min(r['harm_events'] for r in arm_results)}."
        )
    elif not dv_measurable:
        non_degenerate = False
        ng_label = "dv_not_measurable"
        degeneracy_reason = (
            f"ARM_ON harm_accum_pred is constant (hap_std <= {HAP_STD_FLOOR}) on some "
            f"seed; no correlation can be read."
        )
    elif not manip_had_effect:
        non_degenerate = False
        ng_label = "manipulation_no_representational_effect"
        degeneracy_reason = (
            f"the flag path ran (ema differentiated) but produced no measurable "
            f"representational change: |ARM_ON pref - ARM_OFF pref| < {MANIP_EPS} on "
            f"all {len(manip_deltas)} seeds (max delta={max(manip_deltas) if manip_deltas else 0.0:.4f}). "
            f"Shipped SD-020 target-swap does not demonstrably alter z_harm_a on this "
            f"substrate/schedule -- keeps SD-020 at candidate, not a falsification."
        )

    # --- verdict (only meaningful when non-degenerate) -----------------------
    if non_degenerate:
        experiment_passes = seed_passes >= PASS_MIN_SEEDS
        outcome = "PASS" if experiment_passes else "FAIL"
        evidence_direction = "supports" if experiment_passes else "does_not_support"
        label = ("sd020_real_flagpath_supported" if experiment_passes
                 else "sd020_real_flagpath_not_supported")
    else:
        experiment_passes = False
        outcome = "FAIL"
        evidence_direction = "non_contributory"
        label = ng_label

    print(f"\n=== V3-EXQ-324d {outcome} (evidence_direction={evidence_direction}) ===", flush=True)
    print(f"  seeds passing C1&C2&C3: {seed_passes}/{len(seeds)}  "
          f"non_degenerate={non_degenerate}", flush=True)

    metrics = {
        "n_seeds": float(len(seeds)),
        "seeds_passing": float(seed_passes),
        "pe_differentiated": 1.0 if pe_differentiated else 0.0,
        "well_powered": 1.0 if well_powered else 0.0,
        "dv_measurable": 1.0 if dv_measurable else 0.0,
        "manip_had_effect": 1.0 if manip_had_effect else 0.0,
        "off_mean_hap_surprise_pref": float(np.mean([r["hap_surprise_pref"] for r in off.values()])) if off else 0.0,
        "on_mean_hap_surprise_pref": float(np.mean([r["hap_surprise_pref"] for r in on.values()])) if on else 0.0,
        "on_mean_hap_surprise_corr": float(np.mean([r["hap_surprise_corr"] for r in on.values()])) if on else 0.0,
        "on_mean_harm_obs_ema": float(np.mean([r["harm_obs_ema_after_warmup"] for r in on.values()])) if on else 0.0,
        "mean_zna_cov": float(np.mean([r["zna_cov"] for r in arm_results])) if arm_results else 0.0,
        "min_harm_events": float(min(r["harm_events"] for r in arm_results)) if arm_results else 0.0,
        "max_manip_delta": float(max(manip_deltas)) if manip_deltas else 0.0,
    }

    interpretation = {
        "label": label,
        "seed_flags": seed_flags,
        "criteria": [
            {"name": "C1_on_surprise_pref_gt_off", "load_bearing": True},
            {"name": "C2_on_surprise_corr_ge_floor", "load_bearing": True},
            {"name": "C3_no_collapse", "load_bearing": True},
        ],
        "note": (
            "Real shipped flag path (agent.compute_harm_accum_loss under "
            "harm_surprise_pe_enabled) on the SD-022 limb_damage substrate. PASS = the "
            "flag shifts the harm_accum head toward encoding harm surprise (PE) over raw "
            "magnitude in >=3/5 seeds. Self-routes non_contributory when the flag path "
            "did not differentiate, the run is underpowered, the DV is unmeasurable, or "
            "the manipulation produced no representational change. Directly addresses "
            "Q-036 (is surprise necessary in addition to temporal integration)."
        ),
    }

    result = {
        "experiment_type": EXPERIMENT_TYPE,
        "status": outcome,
        "outcome": outcome,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": evidence_direction,
        "supersedes": "v3_exq_324b_sd020_harm_surprise_pe",
        "registered_thresholds": {
            "SURPRISE_CORR_FLOOR": SURPRISE_CORR_FLOOR,
            "PE_EMA_FLOOR": PE_EMA_FLOOR,
            "HARM_EVENT_FLOOR": HARM_EVENT_FLOOR,
            "HAP_STD_FLOOR": HAP_STD_FLOOR,
            "MANIP_EPS": MANIP_EPS,
            "PASS_MIN_SEEDS": PASS_MIN_SEEDS,
        },
        "metrics": metrics,
        "arm_results": arm_results,
        "interpretation": interpretation,
        "non_degenerate": bool(non_degenerate),
        "substrate": "SD-022 limb_damage_enabled=True (harm_obs_a re-sourced from damage state, 7-dim)",
        "elapsed_seconds": float(time.perf_counter() - t0),
    }
    if degeneracy_reason is not None:
        result["degeneracy_reason"] = degeneracy_reason
    return result


if __name__ == "__main__":
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run(seeds=args.seeds, dry_run=args.dry_run)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = (f"{EXPERIMENT_TYPE}_dry_v3" if args.dry_run
              else f"{EXPERIMENT_TYPE}_{ts}_v3")
    result["run_timestamp"] = ts
    result["timestamp_utc"] = ts
    result["run_id"] = run_id
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    out_path = write_flat_manifest(
        result,
        dry_run=args.dry_run,
        config={
            "seeds": args.seeds,
            "arms": [a["arm_id"] for a in ARMS],
            "warmup_episodes": WARMUP_EPISODES,
            "eval_steps": EVAL_STEPS,
            "steps_per_episode": STEPS_PER_EPISODE,
            "harm_obs_ema_alpha": HARM_EMA_ALPHA,
            "env_kwargs": dict(SD022_ENV_KWARGS),
            "registered_thresholds": result["registered_thresholds"],
        },
        seeds=args.seeds,
        script_path=Path(__file__),
        z_goal_stream_stats=_ZG.stats(),
    )

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    print(f"final_outcome: {result['outcome']}", flush=True)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=bool(args.dry_run),
    )
