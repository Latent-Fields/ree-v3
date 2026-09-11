"""V3-EXQ-1023 -- SD-106 VALIDATION: does a generic bottleneck preservation pressure at the
observation->z_world encoder reach PCA-32 parity at the consumer rung?

SD-106 (`encoder.generic_bottleneck_variance_preservation`, landed 2026-09-11) is the successor
SHAPE to SD-018, minted by /governance gov-20260911 from the CONFIRMED autopsy
`failure_autopsy_V3-EXQ-1010_2026-09-11.json` (REE_assembly 652ababa92, user-gated). This run is
its pre-set acceptance measurement.

  MANIPULATION: the observation->z_world TRAINING OBJECTIVE (and the encoder's ability to
                express a linear map), and nothing else.
  HELD FIXED:   the decoder ladder, the dataset recipe, the seeds, the held-out episode split,
                the standardiser, the fit protocol (Adam, ADAPTER_LR, ADAPTER_BATCH,
                ADAPTER_PASSES), the oracle labels, and every threshold. All imported from
                V3-EXQ-1010 / V3-EXQ-1008 / V3-EXQ-1002, never re-defined.

EXPERIMENT_PURPOSE = "diagnostic" -- substrate-readiness validation of a landed SD, per
/implement-substrate Step 8. It is not governance evidence for any mechanism claim.

SLEEP DRIVER: not applicable -- no sleep flag is set (the x734 all-ON stack at this rung enables
no sleep loop). Recorded as sleep_driver_pattern="none".

red-team (fable): see the queue entry note and the RED-TEAM RECORD at the end of this docstring.

=== THE ACCEPTANCE TARGET, PRE-SET AND NOT INVENTED HERE ===

From the substrate_queue SD-106 entry and the autopsy that minted it, verbatim in substance:

    the observation->z_world latent should reach PCA-32 parity at the consumer rung -- >= 0.85
    held-out oracle-action agreement on a seed majority, re-measured by re-running
    v3_exq_1010_zworld_overcapacity_decoder_sweep.py unchanged

`SD106_PARITY_BAR = 0.85` and `CONSUMER_RUNG = "mlp128"` (x734.PPOPolicyNet at
PPO_TRUNK_HIDDEN) are that target, transcribed. `SEED_MAJORITY = 2` of 3 is x1002's own.

=== WHY THIS IS A NEW DRIVER AND NOT AN EDIT TO 1010 ===

The acceptance target says "re-running 1010 UNCHANGED". Taken literally that is impossible: 1010
builds its OFF latent from `ZWorldP0Config()` defaults, so nothing in it can express an
SD-106-ON warmup. What the target MEANS -- and what this driver honours -- is that the
MEASUREMENT INSTRUMENT is unchanged: the capacity ladder, the dataset recipe, the standardiser,
the fit protocol, the calibration anchor and the thresholds are IMPORTED FROM x1010 AND x1008,
never re-implemented. This driver adds exactly one thing: a fourth representation track whose
encoder was trained with the SD-106 objective. 1010 itself is not modified, and continues to
reproduce its own published numbers.

Concretely: `x1010._fit_track_cell` performs every cell's transform -> decoder fit -> agreement
readout here, exactly as it does in 1010. It reads only `track` and `rung` off the arm id, both
parsed from the `<track>__<rung>` string, so the new track composes with it without an edit.

=== THE FOUR TRACKS ===

  `zworld_sd106`      z_world from an SD-106-ON warmup (32 dims): the encoder built with
                      `use_world_encoder_skip=True` and the P0 run at
                      `ZWorldP0Config(preservation_weight=PRESERVATION_WEIGHT)`.
                      ** THE SUBJECT. **
  `zworld_off`        the frozen 978-OFF z_world (32 dims), re-warmed IN THIS RUN from the same
                      imported recipe 1010 used. ** THE PRE-SD-106 COMPARISON. ** Re-run rather
                      than cited so ON-vs-OFF is PAIRED at the same seeds, the same dataset and
                      the same fit -- the single comparison this run exists to make. 1010
                      measured 0.6839 / 0.6846 / 0.6656 best-over-capacity.
  `ws250_pca`         PCA-32 of the encoder's own world_state input at the encoder's own width,
                      the same train-split-fitted projection 1008 and 1010 used.
                      ** THE CALIBRATION ANCHOR ** -- it certifies the protocol AND defines
                      "parity". 1010 measured 0.8836 / 0.8729 / 0.8763.
  `rawfield_ceiling`  the raw 25-dim resource field at the consumer's width, single rung.
                      ** THE INSTRUMENT GATE ** (x1002's ARM_RAW, imported). 1010: 0.9735-0.9832.

The UNTRAINED track from 1010 is deliberately NOT re-run. Its job there was to attribute an
ELIMINATED verdict for H-F; here the verdict is a parity comparison against the anchor, and the
anchor plus the paired OFF arm carry it. Cost, not doubt, is the reason -- stated so a later
reader does not infer the control was dropped to flatter the result.

=== WHAT A NULL HERE WOULD AND WOULD NOT MEAN ===

If `zworld_sd106` does NOT clear 0.85 at the consumer rung:
  IT WOULD mean the SD-106 objective, at PRESERVATION_WEIGHT on this rung, does not recover the
  oracle-relevant content -- so the proxies the build was designed against (linear-decodable
  world_obs R^2 0.9974/0.9978 and resource_field R^2 0.9857/0.9908 vs the PCA-32 anchor's
  0.9983/0.9878) do NOT transfer to oracle-action agreement, which is itself the finding.
  IT WOULD NOT mean preservation is the wrong lever: the weight is one pre-registered point on a
  curve the design probe measured only at 50/200/500, and a null at 200 leaves the curve open.
  A null therefore routes to /failure-autopsy, NOT to an automatic re-queue at another weight.

If the ANCHOR fails its own gate the run is an instrument failure, not a verdict on SD-106, and
self-routes `substrate_not_ready_requeue`.

=== RED-TEAM RECORD ===
red-team (opus-5; fable requested first and refused by a provider spend limit, re-spawned once
on the session model per /queue-experiment Step 4.5): **CONTESTED -> all five findings FIXED**,
none dismissed. Verified against source before acting, per the skill's "a finding is a lead,
not a verdict".

  F1 (family 4, headline) CONFIRMED. `REF_PCA_CONSUMER_1010` was labelled "ws250_pca at mlp128"
     but held 1010's `anchor_best_agreement` -- the MAX OVER FIVE RUNGS. The gate it certifies
     measures the CONSUMER RUNG, where 1010's real values are 0.8836 / 0.8578 / 0.8702. So the
     reachability certificate scored a systematically HIGHER statistic than the gate, and
     passed vacuously; worse, the shipped predicate was `min` over seeds, under which 1010's
     OWN anchor data FAILS (0.8578 < 0.85) -- unmeetable by construction, so ordinary seed
     jitter would have labelled a good SD-106 run `substrate_not_ready_requeue`. FIXED: the
     literals are the consumer-rung values read from 1010's manifest, and the predicate is a
     SEED-MAJORITY rule matching both the certificate's aggregation and the pre-set criterion.
  F2 (family 3) CONFIRMED by execution. With the instrument gate failed, no SD-106 cell runs,
     `skip_norms` is empty, and a `None` `measured` made `p0_readiness_gate` raise an UNCAUGHT
     TypeError -- converting the designed `substrate_not_ready_requeue` self-route into an
     ERROR with no manifest at all. FIXED: `measured` is 0.0 in that case, which is also the
     correct reading (the bypass demonstrably did not train, because nothing ran).
  F3 (family 4) CONFIRMED. C2 witnessed non-degeneracy with `max|sd106 - off| > 1e-9`, which
     cannot fail: the arms are built at different points in the torch RNG stream and 1010's own
     OFF arm varies by ~1e-2 across seeds. C1's `criteria_non_degenerate` was built on that
     un-failable quantity. FIXED: C2 is now the mechanism fact -- the preservation head ran on
     every SD-106 cell and on no OFF cell -- which is false exactly when the manipulation
     silently did not happen.
  F4 (family 3) CONFIRMED. The per-arm encoder-health specs 1010 gates on were dropped, which
     is 1010's own red-team F1 re-committed. FIXED: `sd106_encoder_trained_in_p0` and
     `sd106_latent_not_collapsed` restored as preconditions (the latter matters specifically
     because preservation_weight=200.0 reweights SD-070's variance=25 / covariance=50
     anti-collapse terms). `ANCHOR_DEGRADE_TOL` was imported and written into the manifest
     config while no check used it -- advertising a Guard 1 this run does not perform -- and
     is now removed rather than faked.
  F5 (minor) CONFIRMED on all three counts. `REF_RAWFIELD_1010`'s third value was not a
     measured seed (actuals 0.9832 / 0.9738 / 0.9735); `offending_cell` was always null because
     per-seed dicts carry no `cell_id`; and `insufficient_seeds_for_majority` was unreachable
     because `parity_met` already conjoins `seeds_sufficient`. All three fixed.

  Found while fixing F4, not by the red-team: the restored weight-delta gate read a
  non-existent key `n_changed` (the real one is `n_world_encoder_changed`) -- the same
  unmeetable-by-construction shape as F1, and caught only because the dry-run manifest showed
  0.0 against a bypass norm of 0.319. Fixed and verified at 4 of 4 tensors changed.

=== KNOWN OPEN SUBSTRATE DEFECTS THIS RUN EXERCISES (Step 2.5c, user exception) ===
Three `corrupting`-severity substrate_queue entries are open on modules any all-ON agent driver
exercises: `mode-governance-engagement` (ree_core/agent.py, utils/config.py,
salience_coordinator.py), `contextmemory-write-path-addressing-degeneracy`
(e1_deep.py::ContextMemory.write) and `SD-e1-rollout-consistency-training` (e1_deep.py::forward,
::predict_long_horizon). A scoped user exception was granted 2026-09-11 on the ground that
V3-EXQ-1010 -- the run governance pre-set as this acceptance measurement -- ALREADY EXECUTED
under these identical three defects, and that the comparison here is PAIRED (ON vs OFF vs anchor
vs raw, all under the same defects), which is the case where a shared defect is least likely to
flip a parity verdict. Recorded so the manifest and any later autopsy see the limitation.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.capability_eval import evaluate_seed  # noqa: E402
from experiments._metrics import p0_readiness_gate, P0NotReady  # noqa: E402
from experiments._lib.readiness_anchor import assert_anchor_reachable  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.zworld_encoder_guard import (  # noqa: E402
    latent_stack_snapshot, latent_stack_weight_delta,
)
from ree_core.latent.zworld_p0 import ZWorldP0Config  # noqa: E402

import experiments.v3_exq_724_competence_localization_diagnostic as x724  # noqa: E402
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734  # noqa: E402
import experiments.v3_exq_1002_zworld_actor_adequacy_oracle_adapter as x1002  # noqa: E402
import experiments.v3_exq_1008_zworld_adequacy_portfolio_ws250_rebasis as x1008  # noqa: E402
import experiments.v3_exq_1010_zworld_overcapacity_decoder_sweep as x1010  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_1023_sd106_bottleneck_preservation_validation"
QUEUE_ID = "V3-EXQ-1023"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS = ["SD-106"]

# ---- IMPORTED, NEVER REDEFINED: instrument, dataset recipe, thresholds ----------------
DEVICE = x1002.DEVICE
RUNG = x1002.RUNG
RUNG_ID = x1002.RUNG_ID
LEVEL_ID = x1002.LEVEL_ID
ZWORLD_P0_EPISODES = x1002.ZWORLD_P0_EPISODES
P0_WARMUP_EPISODES = x1002.P0_WARMUP_EPISODES
P1_REINFORCE_EPISODES = x1002.P1_REINFORCE_EPISODES
EVAL_EPISODES = x1002.EVAL_EPISODES
STEPS_PER_EPISODE = x1002.STEPS_PER_EPISODE
BC_EPISODES = x1002.BC_EPISODES
BC_RANDOM_EPISODES = x1002.BC_RANDOM_EPISODES
ADAPTER_PASSES = x1002.ADAPTER_PASSES
ADAPTER_LR = x1002.ADAPTER_LR
ADAPTER_BATCH = x1002.ADAPTER_BATCH
SEED_MAJORITY = x1002.SEED_MAJORITY
RAW_FIELD_CONTROL_FLOOR = x1002.RAW_FIELD_CONTROL_FLOOR
PROJECTION_DIM = x1010.PROJECTION_DIM
CAPACITY_LADDER = x1010.CAPACITY_LADDER
RUNG_IDS = x1010.RUNG_IDS
CONSUMER_RUNG = x1010.CONSUMER_RUNG
MAX_CAPACITY_RUNG = x1010.MAX_CAPACITY_RUNG
ARM_RAW = x1010.ARM_RAW

DRY_RUN_SEEDS = x1010.DRY_RUN_SEEDS
DRY_RUN_ZWORLD_P0 = x1010.DRY_RUN_ZWORLD_P0
DRY_RUN_P0 = x1010.DRY_RUN_P0
DRY_RUN_P1 = x1010.DRY_RUN_P1
DRY_RUN_EVAL = x1010.DRY_RUN_EVAL
DRY_RUN_STEPS = x1010.DRY_RUN_STEPS
DRY_RUN_BC_EPISODES = x1010.DRY_RUN_BC_EPISODES
DRY_RUN_BC_RANDOM_EPISODES = x1010.DRY_RUN_BC_RANDOM_EPISODES
DRY_RUN_ADAPTER_PASSES = x1010.DRY_RUN_ADAPTER_PASSES

SEEDS = [42, 43, 44]

# ---- SD-106's own pre-registered constants -------------------------------------------
# THE ACCEPTANCE BAR, transcribed from the substrate_queue SD-106 entry. NOT re-derived here
# and NOT to be adjusted from this run's own statistics.
SD106_PARITY_BAR = 0.85
# The operating point the SD-106 design probe measured at PCA-32 parity on 2/2 seeds
# (world_obs R^2 0.9974/0.9978; resource_field R^2 0.9857/0.9908 vs anchor 0.9878/0.9894).
# Pre-registered: this run measures THIS weight, and a null at it does not license a silent
# re-queue at another (see "WHAT A NULL HERE WOULD AND WOULD NOT MEAN").
PRESERVATION_WEIGHT = 200.0
USE_WORLD_ENCODER_SKIP = True

# ---- ANCHOR REACHABILITY REFERENCES (V3-EXQ-1010's own recorded per-seed values) ------
# Each gate below self-routes to `substrate_not_ready_requeue`, so a gate NARROWER than the
# state it anchors to would report met=false forever and mislabel an instrument-specification
# gap as a substrate verdict (V3-EXQ-778d). These literals are 1010's published numbers on
# seeds 42/43/44, and `_assert_gates_reachable()` re-scores them at setup with THE SHIPPED
# PREDICATE -- not a copy -- so a drifted bar is refused before any compute is spent.
# RED-TEAM F1 (CONTESTED -> FIXED). An earlier draft used [0.8836, 0.8729, 0.8763] here and
# labelled them "ws250_pca at mlp128". They are NOT: those are 1010's `anchor_best_agreement`,
# the MAX OVER FIVE CAPACITY RUNGS. The gate below measures the CONSUMER RUNG only, where
# 1010's actual values are 0.8836 / 0.8578 / 0.8702 -- seed 43 sits BELOW the 0.85 bar. So the
# old certificate was scored on a different, systematically higher statistic than the gate it
# certified, and passed vacuously. Two consequences, both fixed:
#   (a) the literals are now the consumer-rung values, read from 1010's own manifest;
#   (b) the shipped predicate is a SEED-MAJORITY rule, not `min` over all seeds. Under a
#       min rule the anchor gate is UNMEETABLE against its own reference (0.8578 < 0.85), so
#       ordinary seed jitter would label a good SD-106 run `substrate_not_ready_requeue`.
#       Majority also matches the pre-set acceptance criterion's own 2-of-3 semantics.
REF_PCA_CONSUMER_1010 = [0.8836, 0.8578, 0.8702]   # ws250_pca AT mlp128 (1010, seeds 42/43/44)
REF_RAWFIELD_1010 = [0.9832, 0.9738, 0.9735]       # rawfield_ceiling (1010, seeds 42/43/44)
REF_OFF_PARTICIPATION_1010 = [4.5657, 4.2228, 5.6651]   # zworld_off PR (1010, seeds 42/43/44)
# SD-070's own absolute anti-collapse floor (V3-EXQ-783 gate), re-used unchanged as the
# "the preservation term did not collapse the latent" guard -- RED-TEAM F4.
PARTICIPATION_RATIO_FLOOR = 2.0

# PRESERVATION_WEIGHT *is* declared in the arm fingerprint's config_slice -- as the
# `preservation_weight` key built by `_config_slice()` (see that function). The static check
# cannot follow the value through a helper that builds the dict and returns it, so it reads
# as omitted. Exempting rather than inlining the dict at the call site: the same helper also
# carries `use_world_encoder_skip`, and `_run_self_test` asserts the ON and OFF slices do NOT
# collide, which is the property the check exists to protect.
CONFIG_SLICE_DECLARATION_EXEMPT = (
    "PRESERVATION_WEIGHT is declared as config_slice['preservation_weight'] in _config_slice(); "
    "the static scan cannot follow it through the helper. _run_self_test asserts the SD-106 and "
    "OFF slices differ, so the false-cache-HIT this check guards cannot occur."
)

TRACK_SD106 = "zworld_sd106"
TRACK_OFF = x1010.TRACK_OFF
TRACK_PCA = x1010.TRACK_PCA
SWEPT_TRACKS = [TRACK_SD106, TRACK_OFF, TRACK_PCA]


def _arm_id(track: str, rung: str) -> str:
    return "%s__%s" % (track, rung)


SD106_ARM_IDS = [_arm_id(TRACK_SD106, r) for r in RUNG_IDS]
OFF_ARM_IDS = [_arm_id(TRACK_OFF, r) for r in RUNG_IDS]
PCA_ARM_IDS = [_arm_id(TRACK_PCA, r) for r in RUNG_IDS]
SWEPT_ARM_IDS = SD106_ARM_IDS + OFF_ARM_IDS + PCA_ARM_IDS
ARM_IDS = [ARM_RAW] + SWEPT_ARM_IDS          # 1 + 15 = 16 arms per seed

_EXTRA_SUBSTRATE_PATHS = [Path(m.__file__) for m in (x1002, x1008, x1010, x724, x734)]
_ZG = ZGoalStreamAccumulator()


def _ctx(arm_id: str) -> Dict[str, Any]:
    """Local arm context. `x1010._fit_track_cell` reads only `track` and `rung`, both parsed
    from the `<track>__<rung>` arm id, so the new track needs no edit to that helper."""
    track = ARM_RAW if arm_id == ARM_RAW else arm_id.split("__", 1)[0]
    rung = None if arm_id == ARM_RAW else arm_id.split("__", 1)[1]
    return {
        "id": arm_id, "arm_id": arm_id, "track": track, "rung": rung,
        "is_sd106_track": bool(arm_id in SD106_ARM_IDS),
        "is_off_track": bool(arm_id in OFF_ARM_IDS),
        "is_anchor_track": bool(arm_id in PCA_ARM_IDS),
        "is_consumer_rung": bool(rung == CONSUMER_RUNG),
        "is_max_capacity": bool(rung == MAX_CAPACITY_RUNG),
    }


# --------------------------------------------------------------------------------------
# WARMUPS
# --------------------------------------------------------------------------------------
def _make_sd106_agent(env):
    """x1002's all-ON stack with SD-106's encoder bypass enabled, and NOTHING else changed.

    Built from `x724._base_config_kwargs` + `_all_on_extra_kwargs` + `use_resource_field_head`
    exactly as `x1002._make_agent` does -- 978's own F1 correction, which must be preserved or
    this stops being comparable with the OFF arm -- then the ONE SD-106 flag on top.
    """
    kwargs = x724._base_config_kwargs(env)
    kwargs.update(x724._all_on_extra_kwargs())
    kwargs["use_resource_field_head"] = True
    kwargs["use_world_encoder_skip"] = bool(USE_WORLD_ENCODER_SKIP)
    cfg = x724.REEConfig.from_dims(**kwargs)
    agent = x724.REEAgent(cfg)
    # `REEConfig.from_dims` silently swallows kwargs it does not recognise (memory:
    # reference-reeconfig-from-dims-silent-kwargs), so assert the flag actually reached the
    # substrate rather than trusting that it did. A silent swallow here would make the ON arm
    # a bit-identical copy of the OFF arm and the whole run vacuous.
    if USE_WORLD_ENCODER_SKIP:
        skip = getattr(agent.latent_stack.split_encoder, "world_encoder_skip", None)
        if skip is None:
            raise RuntimeError(
                "SD-106: use_world_encoder_skip=True did not reach SplitEncoder "
                "(world_encoder_skip is None). The config kwarg was swallowed; the ON arm "
                "would be identical to OFF. Refusing to run."
            )
    return agent


def _warm_sd106_agent(seed: int, env_kwargs: Dict[str, Any], sched: Dict[str, int],
                      dry_run: bool):
    """The SD-106-ON warmup: the SAME warmup family as `x1010._warm_off_agent`, differing ONLY
    in the encoder flag and the P0 objective weights. Paid once per seed, shared by five rungs.
    """
    warm_env = x734._make_env(seed, env_kwargs)
    agent = _make_sd106_agent(warm_env)
    before = latent_stack_snapshot(agent)
    stats = x734._train_all_on_agent(
        agent, warm_env, seed=seed, p0_episodes=sched["p0"], p1_episodes=sched["p1"],
        steps_per_episode=sched["steps"], rung_id=RUNG_ID,
        total_denominator=(sched["p0"] + sched["p1"]),
        zworld_p0_episodes=sched["zworld_p0"],
        zworld_p0_env=(x734._make_env(seed, env_kwargs) if sched["zworld_p0"] > 0 else None),
        zworld_p0_dry_run=dry_run,
        # SD-106: the manipulation. resource_field_weight stays 0.0 (978's OFF arm) and is
        # carried INSIDE the config -- run_zworld_p0 refuses both sources at once.
        zworld_p0_config=ZWorldP0Config(preservation_weight=float(PRESERVATION_WEIGHT),
                                        resource_field_weight=0.0),
    )
    guard = latent_stack_weight_delta(agent, before)
    _ZG.observe(agent)
    # The bypass is zero-initialised, so if P0 never stepped it the flag is on but the
    # mechanism inert -- exactly the "reads as enabled while unreachable" failure the
    # substrate-readiness rule exists to catch. Measured, recorded, and gated below.
    skip = getattr(agent.latent_stack.split_encoder, "world_encoder_skip", None)
    skip_norm = float(skip.weight.detach().norm()) if skip is not None else 0.0
    return agent, stats, guard, skip_norm


# --------------------------------------------------------------------------------------
# CELLS
# --------------------------------------------------------------------------------------
def _config_slice(base: Dict[str, Any], arm_id: str) -> Dict[str, Any]:
    ctx = _ctx(arm_id)
    kind, hidden, depth = ((None, None, None) if arm_id == ARM_RAW
                           else x1010._rung_spec(str(ctx["rung"])))
    d = dict(base)
    d["arm_id"] = arm_id
    d["arm_track"] = ctx["track"]
    d["arm_capacity_rung"] = ctx["rung"] or CONSUMER_RUNG
    d["arm_decoder_kind"] = kind or "mlp"
    d["arm_decoder_hidden"] = (int(hidden) if hidden else (int(x734.PPO_TRUNK_HIDDEN)
                                                           if arm_id == ARM_RAW else 0))
    d["arm_decoder_depth"] = (int(depth) if depth is not None else 2)
    # SD-106's manipulation belongs in the slice: two cells differing in it are NOT the same
    # computation, and omitting it would make the ON and OFF fingerprints collide.
    d["preservation_weight"] = (float(PRESERVATION_WEIGHT)
                                if ctx["is_sd106_track"] else 0.0)
    d["use_world_encoder_skip"] = bool(USE_WORLD_ENCODER_SKIP and ctx["is_sd106_track"])
    return d


def run_cell(arm_id: str, seed: int, data: Dict[str, Any], feats: Dict[str, Any],
             frozen: Dict[str, Any], action_dim: int, sched: Dict[str, int],
             env_kwargs: Dict[str, Any], cfg_base: Dict[str, Any],
             dry_run: bool) -> Dict[str, Any]:
    ctx = _ctx(arm_id)
    print("Seed %d Condition %s:%s" % (seed, RUNG_ID, arm_id), flush=True)
    cfg_slice = _config_slice(cfg_base, arm_id)
    passes = sched["passes"]
    # A z_world cell shares its seed's frozen agent with its four sibling rungs, so the cells
    # are NOT independent -- stamped reuse-ineligible for exactly that reason.
    ineligible = (["frozen_agent_shared_across_capacity_rungs"]
                  if (ctx["is_off_track"] or ctx["is_sd106_track"]) else [])

    with arm_cell(seed, config_slice=cfg_slice, script_path=Path(__file__),
                  config_slice_declared=True, include_driver_script_in_hash=False,
                  extra_substrate_paths=_EXTRA_SUBSTRATE_PATHS,
                  extra_ineligible_reasons=ineligible) as cell:
        y_tr, y_te, yr_te = feats["y_tr"], feats["y_te"], feats["yr_te"]
        f_tr, f_te = feats["field"]["tr"], feats["field"]["te"]

        if arm_id == ARM_RAW:
            x_tr, x_te, xr_te = f_tr, f_te, feats["field"]["r"]
            row = x1010._fit_track_cell(arm_id, seed, data, action_dim, passes,
                                        x_tr, y_tr, x_te, y_te, xr_te, yr_te,
                                        x1008._DiagZ(x_tr), None, None)
        elif ctx["is_anchor_track"]:
            x_tr, x_te, xr_te = feats["ws"]["tr"], feats["ws"]["te"], feats["ws"]["r"]
            if "pca" not in frozen:
                W, stats = x1008._world_state_pca_stats(x_tr, PROJECTION_DIM)
                frozen["pca"] = x1008._LinearProjection(x_tr, W, "pca_32", extra=stats)
            row = x1010._fit_track_cell(arm_id, seed, data, action_dim, passes,
                                        x_tr, y_tr, x_te, y_te, xr_te, yr_te,
                                        frozen["pca"], f_tr, f_te)
        else:
            key = "sd106" if ctx["is_sd106_track"] else "off"
            if key not in frozen:
                if key == "sd106":
                    agent, wstats, guard, skip_norm = _warm_sd106_agent(
                        seed, env_kwargs, sched, dry_run)
                else:
                    agent, wstats, guard = x1010._warm_off_agent(
                        seed, env_kwargs, sched, dry_run)
                    skip_norm = None
                frozen[key] = {"agent": agent, "warm_stats": wstats, "warm_guard": guard,
                               "skip_norm": skip_norm, "z": x1008._z_feats(agent, data)}
            fz = frozen[key]
            z = fz["z"]
            x_tr, x_te, xr_te = z["tr"], z["te"], z["r"]
            row = x1010._fit_track_cell(arm_id, seed, data, action_dim, passes,
                                        x_tr, y_tr, x_te, y_te, xr_te, yr_te,
                                        x1008._DiagZ(x_tr), f_tr, f_te)
            row["warmup_skipped"] = False
            row["warmup_stats"] = fz.get("warm_stats")
            row["zworld_weight_delta"] = fz.get("warm_guard")
            row["zworld_participation_ratio"] = x1002._participation_ratio(x_tr)
            if ctx["is_sd106_track"]:
                row["sd106_skip_weight_norm"] = fz.get("skip_norm")
                # `_train_all_on_agent` nests the SD-070 warmup stats under "zworld_p0"
                # with p0a_* keys (NOT "p0a_stats" -- a wrong path here would read None
                # forever and hide an inert leg).
                p0a = (fz.get("warm_stats") or {}).get("zworld_p0") or {}
                pres = p0a.get("p0a_preservation_holdout")
                row["sd106_preservation_holdout_r2"] = (
                    pres.get("r2") if isinstance(pres, dict) else None)
                row["sd106_used_preservation_head"] = p0a.get("p0a_used_preservation_head")
                row["sd106_used_world_encoder_skip"] = p0a.get("p0a_used_world_encoder_skip")

        cell.stamp(row)
    x1010._print_verdict(row)
    return row


def _assert_gates_reachable() -> List[Dict[str, Any]]:
    """Refuse, at setup, any gate its own known-positive control cannot clear.

    `sd106_bypass_trained_off_zero` is deliberately NOT guarded here: its predicate
    (||W_skip|| > 0) IS the definition of "the mechanism ran at all", reachable by
    construction, and there is no recorded reference for a quantity this run is the first
    to produce. Stated rather than silently omitted.
    """
    out = []
    out.append(assert_anchor_reachable(
        anchor_name="anchor_pca32_reaches_parity_bar_on_majority",
        reference_cells=list(REF_PCA_CONSUMER_1010),
        score_fn=lambda v: float(v) >= SD106_PARITY_BAR,   # THE SHIPPED PREDICATE
        threshold=float(SEED_MAJORITY) / 3.0,              # = the shipped MAJORITY aggregation
        reference_source=("V3-EXQ-1010 ws250_pca AT THE CONSUMER RUNG mlp128, seeds 42/43/44 "
                          "(0.8836 / 0.8578 / 0.8702) -- 2 of 3 clear 0.85, which is exactly "
                          "the majority the shipped gate requires and the margin it survives "
                          "on. NOT the best-over-rungs series, which is a different and "
                          "systematically higher statistic (red-team F1).")))
    out.append(assert_anchor_reachable(
        anchor_name="instrument_rawfield_control_supra_floor",
        reference_cells=list(REF_RAWFIELD_1010),
        score_fn=lambda v: float(v) >= RAW_FIELD_CONTROL_FLOOR,   # THE SHIPPED PREDICATE
        threshold=1.0,
        reference_source="V3-EXQ-1010 rawfield_ceiling reported range 0.9735-0.9832"))
    return out


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------
def run_experiment(seeds: List[int], dry_run: bool = False) -> Dict[str, Any]:
    sched = {
        "zworld_p0": DRY_RUN_ZWORLD_P0 if dry_run else ZWORLD_P0_EPISODES,
        "p0": DRY_RUN_P0 if dry_run else P0_WARMUP_EPISODES,
        "p1": DRY_RUN_P1 if dry_run else P1_REINFORCE_EPISODES,
        "eval_eps": DRY_RUN_EVAL if dry_run else EVAL_EPISODES,
        "steps": DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE,
        "bc_eps": DRY_RUN_BC_EPISODES if dry_run else BC_EPISODES,
        "bc_rand": DRY_RUN_BC_RANDOM_EPISODES if dry_run else BC_RANDOM_EPISODES,
        "passes": DRY_RUN_ADAPTER_PASSES if dry_run else ADAPTER_PASSES,
    }
    anchor_reachability = _assert_gates_reachable()
    seeds_sufficient = bool(len(seeds) >= SEED_MAJORITY)
    majority = int(SEED_MAJORITY)
    env_kwargs = x734._env_kwargs_for_rung(RUNG)
    cfg_base = x1002._off_path_config_slice(
        dry_run, sched["zworld_p0"], sched["p0"], sched["p1"], sched["steps"],
        sched["bc_eps"], sched["bc_rand"], sched["passes"], sched["eval_eps"])
    probe_env = x734._make_env(seeds[0], env_kwargs)
    action_dim = int(probe_env.action_dim)

    # ---- the 1002 dataset, re-collected from its deterministic recipe, once per seed ----
    per_seed_data: Dict[int, Dict[str, Any]] = {}
    per_seed_feats: Dict[int, Dict[str, Any]] = {}
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        oracle_eps = x1002._collect_episodes(s, env_kwargs, "oracle", sched["bc_eps"],
                                             sched["steps"])
        rand_eps = x1002._collect_episodes(s, env_kwargs, "random", sched["bc_rand"],
                                           sched["steps"])
        tr, te = x1002._split_episodes(oracle_eps)
        per_seed_data[s] = {"train": tr, "test": te, "random": rand_eps}
        f_tr, y_tr = x1002._rawfield_features(tr)
        f_te, y_te = x1002._rawfield_features(te)
        fr_te, yr_te = x1002._rawfield_features(rand_eps)
        w_tr, _ = x1008._world_state_features(tr)
        w_te, _ = x1008._world_state_features(te)
        wr_te, _ = x1008._world_state_features(rand_eps)
        per_seed_feats[s] = {"y_tr": y_tr, "y_te": y_te, "yr_te": yr_te,
                             "field": {"tr": f_tr, "te": f_te, "r": fr_te},
                             "ws": {"tr": w_tr, "te": w_te, "r": wr_te}}

    # ---- instrument gate first, on every seed ------------------------------------------
    frozen: Dict[int, Dict[str, Any]] = {s: {} for s in seeds}
    raw_rows = [run_cell(ARM_RAW, s, per_seed_data[s], per_seed_feats[s], frozen[s],
                         action_dim, sched, env_kwargs, cfg_base, dry_run) for s in seeds]
    raw_worst, raw_worst_cell = x1002._worst_cell(raw_rows, "oracle_action_agreement", "min")
    nte_worst, _nc = x1002._worst_cell(raw_rows, "n_heldout_steps", "min")
    instrument_ready = bool(raw_worst is not None and raw_worst >= RAW_FIELD_CONTROL_FLOOR)

    # ---- the swept cells. Anchor first (it defines parity and gates the verdict), then the
    #      paired OFF and SD-106 tracks. `or dry_run`: the smoke MUST execute every arm
    #      including both warmups -- at dry scale the control cannot reach its floor and the
    #      gate would short-circuit past the code this run exists to exercise (V3-EXQ-591g).
    other_rows: List[Dict[str, Any]] = []
    for s in seeds:
        if instrument_ready or dry_run:
            for aid in PCA_ARM_IDS + OFF_ARM_IDS + SD106_ARM_IDS:
                other_rows.append(run_cell(aid, s, per_seed_data[s], per_seed_feats[s],
                                           frozen[s], action_dim, sched, env_kwargs,
                                           cfg_base, dry_run))
        frozen[s].clear()          # release the seed's agents before the next seed

    rows = raw_rows + other_rows

    # ---- per-seed readouts --------------------------------------------------------------
    per_seed: List[Dict[str, Any]] = []
    for s in seeds:
        sd_best, sd_best_rung = x1010._best_over_rungs(rows, TRACK_SD106, s)
        off_best, off_best_rung = x1010._best_over_rungs(rows, TRACK_OFF, s)
        pca_best, pca_best_rung = x1010._best_over_rungs(rows, TRACK_PCA, s)
        sd_c = x1010._cell(rows, _arm_id(TRACK_SD106, CONSUMER_RUNG), s)
        off_c = x1010._cell(rows, _arm_id(TRACK_OFF, CONSUMER_RUNG), s)
        pca_c = x1010._cell(rows, _arm_id(TRACK_PCA, CONSUMER_RUNG), s)
        sd_cons = (sd_c or {}).get("oracle_action_agreement")
        off_cons = (off_c or {}).get("oracle_action_agreement")
        pca_cons = (pca_c or {}).get("oracle_action_agreement")
        per_seed.append({
            "seed": int(s),
            # THE LOAD-BEARING READOUT: the consumer rung, which is what the bar names.
            "sd106_consumer_agreement": sd_cons,
            "off_consumer_agreement": off_cons,
            "pca_consumer_agreement": pca_cons,
            "sd106_clears_parity_bar": (None if sd_cons is None
                                        else bool(float(sd_cons) >= SD106_PARITY_BAR)),
            "sd106_minus_off_consumer": (None if (sd_cons is None or off_cons is None)
                                         else float(sd_cons) - float(off_cons)),
            "sd106_minus_pca_consumer": (None if (sd_cons is None or pca_cons is None)
                                         else float(sd_cons) - float(pca_cons)),
            "sd106_best_over_capacity": sd_best, "sd106_best_rung": sd_best_rung,
            "off_best_over_capacity": off_best, "off_best_rung": off_best_rung,
            "pca_best_over_capacity": pca_best, "pca_best_rung": pca_best_rung,
            "sd106_skip_weight_norm": (sd_c or {}).get("sd106_skip_weight_norm"),
            "sd106_preservation_holdout_r2": (sd_c or {}).get(
                "sd106_preservation_holdout_r2"),
        })

    n_clear = sum(1 for p in per_seed if p["sd106_clears_parity_bar"])
    parity_met = bool(seeds_sufficient and n_clear >= majority)

    # ---- preconditions -------------------------------------------------------------------
    # (1) INSTRUMENT: the raw-field positive control must clear its floor.
    # (2) ANCHOR: PCA-32 must itself clear the parity bar -- it DEFINES parity, so if it does
    #     not reach the bar on this run the bar is unreachable and no verdict on SD-106 is
    #     available. This is the gate that keeps a null attributable.
    # (3) MECHANISM LIVE: the zero-init bypass must have moved off zero, else the flag reads
    #     as enabled while inert and the ON arm is not actually an ON arm.
    # RED-TEAM F1/F5b: score the anchor by the SAME majority rule the certificate proves, and
    # name the offending seed explicitly (x1002._worst_cell keys on `cell_id`, which per-seed
    # dicts do not carry, so it returned offending_cell=None on every run).
    pca_vals = [(p["seed"], p["pca_consumer_agreement"]) for p in per_seed
                if p.get("pca_consumer_agreement") is not None]
    pca_n_clear = sum(1 for _sd, v in pca_vals if float(v) >= SD106_PARITY_BAR)
    pca_worst_seed = (min(pca_vals, key=lambda t: float(t[1]))[0] if pca_vals else None)
    pca_worst = (min(float(v) for _sd, v in pca_vals) if pca_vals else None)
    skip_norms = [p["sd106_skip_weight_norm"] for p in per_seed
                  if p.get("sd106_skip_weight_norm") is not None]
    # RED-TEAM F4: 1010 gates per-arm on x1002's encoder-health specs; an earlier draft of this
    # driver dropped them, which is 1010's OWN red-team finding F1 re-committed. The two that
    # bear on SD-106 are restored here as flat checks: a silently UNTRAINED encoder (weight
    # delta 0) and a COLLAPSED latent. The second matters specifically because
    # preservation_weight=200.0 sits against variance_weight=25.0 / covariance_weight=50.0,
    # an 8:1-to-4:1 reweighting of the very anti-collapse terms SD-070 was built around -- so
    # "the preservation term collapsed z_world" is a live failure mode this run must not read
    # as an SD-106 verdict.
    prs = [r.get("zworld_participation_ratio") for r in rows
           if r.get("track") == TRACK_SD106 and r.get("zworld_participation_ratio") is not None]
    pr_worst = min(prs) if prs else None
    deltas_ok = [r.get("zworld_weight_delta") for r in rows
                 if r.get("track") == TRACK_SD106 and isinstance(r.get("zworld_weight_delta"), dict)]
    n_changed = None
    for g in deltas_ok:
        # `latent_stack_weight_delta` reports `n_world_encoder_changed` (of
        # `n_world_encoder_tensors`). An earlier draft read a non-existent `n_changed`, which
        # is the same unmeetable-by-construction defect the red-team found at F1: it returns
        # 0.0 forever and fails the gate on every run. Verified against a dry-run manifest:
        # 4 of 4 world-encoder tensors changed, n_world_path_tensors=6 (the bypass is in the
        # trained world path).
        v = g.get("n_world_encoder_changed") if isinstance(g, dict) else None
        if v is not None:
            n_changed = int(v) if n_changed is None else min(n_changed, int(v))

    checks = [
        {"name": "instrument_rawfield_control_supra_floor",
         "description": ("The raw 25-dim resource field decoded at the consumer's width must "
                         "clear RAW_FIELD_CONTROL_FLOOR, else the dataset/labels are blind and "
                         "no track's reading means anything."),
         "measured": raw_worst, "threshold": float(RAW_FIELD_CONTROL_FLOOR),
         "direction": "lower",
         "control": "rawfield_ceiling arm, worst seed (x1002's imported positive control)",
         "offending_cell": raw_worst_cell},
        {"name": "anchor_pca32_reaches_parity_bar_on_majority",
         "description": ("The PCA-32 calibration anchor must itself reach SD106_PARITY_BAR at "
                         "the consumer rung on a SEED MAJORITY -- the same aggregation the "
                         "acceptance criterion uses, and the one its reachability certificate "
                         "proves. The anchor DEFINES parity: if it cannot clear the bar on this "
                         "run, an SD-106 null is an instrument failure, not a verdict."),
         "measured": int(pca_n_clear), "threshold": int(majority), "direction": "lower",
         "control": ("ws250_pca at mlp128 -- a task-agnostic linear compression of the "
                     "encoder's own input at the encoder's own width; 1010 measured "
                     "0.8836 / 0.8578 / 0.8702"),
         "offending_cell": ("seed%s" % pca_worst_seed if pca_worst_seed is not None else None)},
        {"name": "sd106_bypass_trained_off_zero",
         "description": ("SD-106's linear bypass is ZERO-INITIALISED, so a warmup that never "
                         "stepped it leaves the flag enabled and the mechanism inert. The "
                         "worst seed's ||W_skip|| must be strictly above zero."),
         # RED-TEAM F2: `measured` MUST be a real number. When the instrument gate fails, no
         # SD-106 cell runs, skip_norms is empty, and a None here made p0_readiness_gate raise
         # an UNCAUGHT TypeError -- turning the designed `substrate_not_ready_requeue`
         # self-route into an ERROR with no manifest at all. 0.0 is the correct reading: the
         # bypass demonstrably did not train, because nothing ran.
         "measured": (min(skip_norms) if skip_norms else 0.0), "threshold": 1e-8,
         "direction": "lower", "comparator": ">",
         "control": "SD-106 ON arm's world_encoder_skip weight norm after P0, worst seed"},
        {"name": "sd106_encoder_trained_in_p0",
         "description": ("The SD-106 arm's z_world path must have CHANGED during P0 -- the "
                         "SD-070 / V3-EXQ-783 readiness signature. A zero weight delta means "
                         "the encoder never trained, and the ON-vs-OFF delta this run reports "
                         "would then be an artefact rather than an SD-106 effect."),
         "measured": (float(n_changed) if n_changed is not None else 0.0), "threshold": 1.0,
         "direction": "lower",
         "control": "latent_stack_weight_delta n_changed on the SD-106 arm, worst seed"},
        {"name": "sd106_latent_not_collapsed",
         "description": ("z_world must not have COLLAPSED under the reweighting: "
                         "preservation_weight=200.0 against variance_weight=25.0 / "
                         "covariance_weight=50.0. SD-070's own absolute floor (V3-EXQ-783); "
                         "1010's OFF arm measured 4.22-5.67 on these seeds."),
         "measured": (float(pr_worst) if pr_worst is not None else 0.0),
         "threshold": float(PARTICIPATION_RATIO_FLOOR), "direction": "lower",
         "control": "participation ratio of the SD-106 z_world, worst seed"},
    ]
    try:
        preconditions = p0_readiness_gate(checks)
        gate_green, gate_reason = True, ""
    except P0NotReady as e:
        preconditions = list(e.preconditions)
        gate_green = False
        gate_reason = "preconditions unmet: " + ", ".join(
            str(p.get("name")) for p in preconditions if not p.get("met"))

    # ---- non-degeneracy: the ON and OFF tracks must not be bit-identical -----------------
    deltas = [p["sd106_minus_off_consumer"] for p in per_seed
              if p["sd106_minus_off_consumer"] is not None]
    # RED-TEAM F3 (CONTESTED -> FIXED). An earlier draft witnessed non-degeneracy with
    # `max|sd106 - off| > 1e-9`. That CANNOT FAIL: the two agents are constructed at different
    # points in the global torch RNG stream (the ON stack additionally consumes an extra
    # nn.Linear init draw for the bypass), and 1010's own OFF arm varies by ~1e-2 across seeds
    # -- seven orders of magnitude above the tolerance. A certificate that cannot fail is not
    # a certificate, and C1's own `criteria_non_degenerate` was built on it.
    # The witness is now the MECHANISM FACT the manipulation consists of: the preservation head
    # ran on the ON arm and did NOT run on the OFF arm. That is checkable, and it is false
    # exactly when the manipulation silently did not happen.
    on_used = {bool(r.get("sd106_used_preservation_head")) for r in rows
               if r.get("track") == TRACK_SD106}
    off_used = set()
    for r in rows:
        if r.get("track") != TRACK_OFF:
            continue
        p0a = (r.get("warmup_stats") or {}).get("zworld_p0") or {}
        if p0a:
            off_used.add(bool(p0a.get("p0a_used_preservation_head")))
    arms_differ = bool(on_used == {True} and off_used == {False})
    arms_differ_detail = {"sd106_used_preservation_head": sorted(on_used),
                          "off_used_preservation_head": sorted(off_used),
                          "consumer_deltas": deltas}

    if not gate_green:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
    elif not seeds_sufficient:
        # Checked BEFORE parity: `parity_met` already conjoins seeds_sufficient, so testing it
        # afterwards made this branch unreachable (red-team F5).
        label = "insufficient_seeds_for_majority"
        outcome = "FAIL"
    elif parity_met:
        label = "sd106_reaches_pca32_parity"
        outcome = "PASS"
    else:
        label = "sd106_below_pca32_parity"
        outcome = "FAIL"

    flat = {
        "n_seeds": len(seeds),
        "n_seeds_clearing_parity_bar": int(n_clear),
        "seed_majority_required": int(majority),
        "sd106_parity_bar": float(SD106_PARITY_BAR),
        "preservation_weight": float(PRESERVATION_WEIGHT),
        "parity_met": 1 if parity_met else 0,
        "gate_green": 1 if gate_green else 0,
        "arms_differ": 1 if arms_differ else 0,
        "instrument_rawfield_worst": raw_worst,
        "anchor_pca_consumer_worst": pca_worst,
        "anchor_pca_n_seeds_clearing_bar": int(pca_n_clear),
        "sd106_participation_ratio_worst": pr_worst,
        "heldout_steps_worst": nte_worst,
    }
    for tag, key in (("sd106", "sd106_consumer_agreement"),
                     ("off", "off_consumer_agreement"),
                     ("pca", "pca_consumer_agreement")):
        vals = [p[key] for p in per_seed if p.get(key) is not None]
        if vals:
            flat["%s_consumer_agreement_mean" % tag] = float(sum(vals) / len(vals))
            flat["%s_consumer_agreement_min" % tag] = float(min(vals))
    # Non-finite / None are DROPPED rather than emitted: a nan is numeric to the indexer and
    # would pollute a delta, while an absent key correctly reads as unmeasured.
    flat = {k: v for k, v in flat.items()
            if v is not None and (not isinstance(v, float) or v == v)}

    return {
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": list(CLAIM_IDS),
        "sleep_driver_pattern": "none",
        "outcome": outcome,
        "readout": flat,
        "per_seed_results": per_seed,
        "arm_results": rows,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria": [
                {"name": "C1_sd106_reaches_parity_on_seed_majority",
                 "load_bearing": True,
                 "passed": bool(parity_met),
                 "measured": int(n_clear), "threshold": int(majority),
                 "measured_agreement_min": flat.get("sd106_consumer_agreement_min"),
                 "threshold_agreement": float(SD106_PARITY_BAR)},
                {"name": "C2_manipulation_engaged_on_but_not_off",
                 "load_bearing": False,
                 "passed": bool(arms_differ),
                 "measured": int(bool(on_used == {True})) + int(bool(off_used == {False})),
                 "threshold": 2,
                 "detail": arms_differ_detail,
                 "threshold_note": ("both halves must hold: the preservation head RAN on every "
                                    "SD-106 cell and did NOT run on any OFF cell.")},
            ],
            "combination_rule": ("C1 alone decides the verdict. C2 is a non-degeneracy witness "
                                 "that the manipulation actually engaged -- the preservation "
                                 "head ran on every SD-106 cell and on no OFF cell -- and it "
                                 "gates C1's own non-degeneracy rather than the verdict."),
            "criteria_non_degenerate": {
                "C1_sd106_reaches_parity_on_seed_majority": bool(gate_green and arms_differ),
                "C2_on_and_off_arms_differ": bool(len(deltas) > 0),
            },
            "gate_reason": gate_reason,
            "anchor_reachability": anchor_reachability,
            "null_reading": ("A null means the SD-106 objective at preservation_weight=%.1f "
                             "does not recover oracle-relevant content on this rung -- NOT "
                             "that preservation is the wrong lever. Route to /failure-autopsy, "
                             "not to an automatic re-queue at another weight."
                             % float(PRESERVATION_WEIGHT)),
        },
        "non_degenerate": bool(gate_green and arms_differ),
        "degeneracy_reason": (None if (gate_green and arms_differ)
                              else ("preconditions unmet" if not gate_green
                                    else "SD-106 and OFF arms produced identical readouts")),
    }


def _run_self_test() -> int:
    fails = 0
    if SD106_PARITY_BAR != 0.85:
        print("  [self-test] FAIL parity bar drifted from the pre-set 0.85", flush=True)
        fails += 1
    if CONSUMER_RUNG != "mlp128":
        print("  [self-test] FAIL consumer rung is not the imported mlp128", flush=True)
        fails += 1
    if len(ARM_IDS) != 1 + 3 * len(RUNG_IDS):
        print("  [self-test] FAIL arm count %d" % len(ARM_IDS), flush=True)
        fails += 1
    c = _ctx(_arm_id(TRACK_SD106, CONSUMER_RUNG))
    if not (c["track"] == TRACK_SD106 and c["rung"] == CONSUMER_RUNG and c["is_sd106_track"]):
        print("  [self-test] FAIL arm-id parsing: %r" % (c,), flush=True)
        fails += 1
    a = _config_slice({}, _arm_id(TRACK_SD106, CONSUMER_RUNG))
    b = _config_slice({}, _arm_id(TRACK_OFF, CONSUMER_RUNG))
    if a == b:
        print("  [self-test] FAIL ON/OFF config slices collide -- fingerprints would too",
              flush=True)
        fails += 1
    print("[self-test] %d failure(s)" % fails, flush=True)
    return fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args()

    if args.self_test:
        sys.exit(_run_self_test())

    t0 = time.perf_counter()
    seeds = args.seeds if args.seeds else (DRY_RUN_SEEDS if args.dry_run else SEEDS)
    print("%s: seeds=%s dry_run=%s" % (EXPERIMENT_TYPE, seeds, bool(args.dry_run)), flush=True)

    result = run_experiment(list(seeds), dry_run=bool(args.dry_run))

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["timestamp_utc"] = ts
    result["run_id"] = "%s_%s_v3" % (EXPERIMENT_TYPE, ts)
    result["architecture_epoch"] = ARCHITECTURE_EPOCH
    result["queue_id"] = QUEUE_ID

    full_config = {
        "rung": RUNG, "level_id": LEVEL_ID,
        "env_kwargs": x734._env_kwargs_for_rung(RUNG),
        "zworld_p0_episodes": (DRY_RUN_ZWORLD_P0 if args.dry_run else ZWORLD_P0_EPISODES),
        "p0_warmup_episodes": (DRY_RUN_P0 if args.dry_run else P0_WARMUP_EPISODES),
        "p1_reinforce_episodes": (DRY_RUN_P1 if args.dry_run else P1_REINFORCE_EPISODES),
        "eval_episodes": (DRY_RUN_EVAL if args.dry_run else EVAL_EPISODES),
        "steps_per_episode": (DRY_RUN_STEPS if args.dry_run else STEPS_PER_EPISODE),
        "bc_episodes": (DRY_RUN_BC_EPISODES if args.dry_run else BC_EPISODES),
        "bc_random_episodes": (DRY_RUN_BC_RANDOM_EPISODES if args.dry_run
                               else BC_RANDOM_EPISODES),
        "bc_train_frac": x1002.BC_TRAIN_FRAC,
        "adapter_passes": (DRY_RUN_ADAPTER_PASSES if args.dry_run else ADAPTER_PASSES),
        "adapter_batch": ADAPTER_BATCH, "adapter_lr": ADAPTER_LR,
        # THE MANIPULATED VARIABLE, declared in the config the manifest carries.
        "sd106_preservation_weight": float(PRESERVATION_WEIGHT),
        "sd106_use_world_encoder_skip": bool(USE_WORLD_ENCODER_SKIP),
        "sd106_parity_bar": float(SD106_PARITY_BAR),
        "capacity_ladder": [{"rung": r, "kind": k, "hidden": h, "depth": d}
                            for r, k, h, d in CAPACITY_LADDER],
        "capacity_consumer_rung": CONSUMER_RUNG,
        "capacity_max_rung": MAX_CAPACITY_RUNG,
        "consumer_trunk_hidden": int(x734.PPO_TRUNK_HIDDEN),
        "tracks": list(SWEPT_TRACKS),
        "projection_dim": PROJECTION_DIM,
        "raw_field_control_floor": RAW_FIELD_CONTROL_FLOOR,
        "seed_majority": SEED_MAJORITY, "arms": ARM_IDS,
        "dry_run": bool(args.dry_run),
    }
    out_path = write_flat_manifest(
        result, None, dry_run=args.dry_run,
        config=full_config, seeds=seeds, script_path=Path(__file__),
        started_at=t0, z_goal_stream_stats=_ZG.stats(),
    )
    print("outcome: %s (%s)" % (result["outcome"], result["interpretation"]["label"]),
          flush=True)

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
