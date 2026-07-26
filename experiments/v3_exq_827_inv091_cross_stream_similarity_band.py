#!/opt/local/bin/python3
"""V3-EXQ-827 -- INV-091 cross-stream similarity BAND falsifier (EVB-0459, IGW-20260726-219).

Queue ID note: originally drafted as V3-EXQ-824, but a concurrent session claimed that
number for a different (Q-081-direct) experiment before either was committed; renumbered
to 827 (next free after 826) before commit, all internal references updated.

INV-091 asserts that integration and protected non-equivalence are JOINTLY necessary:
cross-stream similarity has a viable BAND. Too little shared organisation is fragmentation;
too much is representational collapse. FALSIFIER (claims.yaml, verbatim): an ablation series
showing a NON-MONOTONIC relation between cross-stream similarity and task-plus-anti-collapse
function. Monotonic improvement with similarity refutes the band one way; monotonic
degradation refutes it the other way. NON-DEGENERACY GUARD: at least one ablation must move
the similarity statistic measurably, or the series tested nothing.

SCOPE (a targeted probe, not the full 7-manipulation falsifier). The claim names seven
possible ablations (remove event broadcasts / remove mode conditioning / remove commitment
landmarks / remove residue feedback / force lockstep / randomise rates / collapse the harm
streams). This run exercises TWO of them, chosen to move cross-stream similarity in OPPOSITE
directions from a common intact baseline:

  * DECOUPLE  (predicted LOW similarity / fragmentation direction) -- "remove commitment
    landmarks", via the already-built q081_landmark_removal yoked-permutation scrambler.
  * LOCKSTEP  (predicted HIGH similarity / collapse direction) -- "force lockstep", via
    e2_steps_per_tick=e3_steps_per_tick=1 (all streams update every env step instead of the
    configured 1:3:10 separation).

A PASS here is informative for INV-091 (both extremes worse than the middle, on two
independent manipulations pulling similarity in opposite directions). A null does NOT
exhaust the falsifier -- the remaining five manipulations (event broadcasts, residue
feedback, randomised rates, harm-stream collapse) are untested by this run. Both this run's
outcome and its partial coverage are recorded in interpretation.note.

WHY A COMMON TRAINED SUBSTRATE, NOT THREE SEPARATE WARMUPS. All three arms share ONE
warmed-up agent (E1/E2 world-forward + E3 harm-eval head trained once per seed via
_lib.goal_pipeline_tier1.warmup_train under the INTACT config); the manipulations are then
applied ONLY at eval/recording time -- DECOUPLE via the landmark scrambler, LOCKSTEP by
mutating agent.clock.e2_steps_per_tick / _current_e3_steps between eval passes. This isolates
the manipulation to the eval window and removes "the arms trained differently" as a
confound, which three independently-warmed agents could not do. Each arm's eval env is a
FRESH CausalGridWorldV2 instance built with the SAME seed, so the three arms see matched
episode layouts (a paired design at fixed seed count).

CROSS-STREAM SIMILARITY STATISTIC. Mean of |xcorr| (q081_surrogate.cross_stream_xcorr) over
four pairs spanning different subsystems and native rates, all populated by the Q-081
recording profile on a common trained substrate:
    (z_world, z_self)          E1-rate vs E1-rate, distinct subsystems (world vs self model)
    (z_world, operating_mode)  E1-rate vs E3-rate (MECH-321 salience coordinator)
    (e1_hidden, e3_commitment) E1-rate vs E3-rate (sensorium vs commitment state)
    (z_world, z_goal)          E1-rate vs loop-driven (goal stream; StepHarness calls
                                update_z_goal every tick, so this is genuinely driven, not
                                flat -- see q081_profile.LOOP_DRIVEN_REQUIREMENTS)
Validated per Q-081's "validate the null before using it" requirement: screen_statistic
confirms this statistic is NOT ruled_out (varies across the constrained-realisation
surrogate ensemble) while q081_surrogate.artefactual_rate_statistic on the same arrays IS
ruled_out (a pure function of the two streams' update periods, killed by construction) --
run once on the intact arm's seed[0] trace before any p-value is read.

ANTI-COLLAPSE (protected non-equivalence) STATISTIC. 1 - mean(|cosine(z_self_t, z_world_t)|)
over eval steps where both are fresh (every step, both E1-rate). Operationalises MECH-063 /
SD-011's per-axis guards as a single continuous readout of the axis INV-091 binds: does the
self/world distinction survive, or does raising cross-stream coupling collapse the two
representations onto each other. Higher = more protected non-equivalence.

TASK METRIC. Mean per-episode reward (env harm_signal summed per episode; identical
accounting to _lib.capability_eval.rollout_episode's `episode_reward`).

COMPOSITE (task_plus_anticollapse). INV-091's own wording is a CONJUNCTIVE requirement --
"satisfy both constraints at once ... not merely satisfy each on a separate axis" -- so the
composite is min(z_task, z_anticollapse), z-scored by pooling all 9 (arm x seed) cells. A
sum would let either axis compensate for the other, which is exactly what the claim says an
adequate architecture must NOT be allowed to do.

PRE-REGISTERED CRITERIA:
  C1 (non-degeneracy, INV-091's own guard): the per-arm mean cross_stream_similarity has a
     measurable spread across {intact, decouple, lockstep} (checked via _metrics.
     check_degeneracy) AND moves in the pre-registered direction:
     similarity(decouple) < similarity(intact) < similarity(lockstep), each gap >=
     SIMILARITY_MOVE_FLOOR. C1 false -> self-route substrate_not_ready_requeue (the
     manipulation didn't move the dial; the series tested nothing, per the claim's own
     non-degeneracy clause). NEVER a verdict on INV-091 itself.
  C2 (the falsifier): composite(intact) exceeds BOTH composite(decouple) and
     composite(lockstep) by >= COMPOSITE_MARGIN. This is the non-monotonic-in-similarity
     shape the claim predicts (both extremes worse than the middle, despite similarity
     moving in opposite directions to produce them).

INTERPRETATION (only reached when C1 holds):
  PASS  (supports INV-091): C1 and C2 both true.
  FAIL: C1 true, C2 false. If composite is additionally MONOTONIC in similarity across the
     three arms (matching either direction the claim names), record which direction in
     interpretation.monotonic_direction -- that is a stronger, direction-specific refutation
     the claim explicitly anticipates ("monotonic improvement ... refutes the band ...
     monotonic degradation ... refutes it in the other direction").

DV-SYMMETRY (per-arm, mandatory declaration -- REE_Working/CLAUDE.md queue-experiment skill
Step 3.5). Neither manipulation is a broadcast-scalar / monotone-rescaling / interchangeable-
unit permutation of the kind that can leave a DV invariant by construction:
  DECOUPLE:  the scrambler permutes boundary-event TIMES and replays them into a live closed
             loop (MECH-269 write_anchor binds segment ids to z_world AT EMISSION TIME), so a
             retimed landmark binds a DIFFERENT z_world state and causally alters downstream
             V_s / anchor gating / E3 selection / actions -- not a symmetry of any streamed
             vector or of the composite.
  LOCKSTEP:  e2/e3 tick cadence is load-bearing for WHEN candidates are generated, scored and
             committed; collapsing it to every-step changes the actual computation performed,
             not merely a label on it.
Neither arm's DV is a pure function of a manipulation the DV cannot see.

ethics_preflight:
  involves_negative_valence: false
  involves_suffering_like_state: false
  involves_self_model: false
  involves_inescapability_or_helplessness: false
  involves_offline_replay_over_harm: false
  involves_social_mind_or_language: false
  involves_human_data_or_clinical_context: false
  decision: allow

SLEEP DRIVER: none (use_sleep_loop stays at the Q-081 profile default; the loop never calls
run_sleep_cycle(), so no sleep cycle fires during this run).

MINT (mint-as-you-go). This experiment writes no reusable OFF-arm baseline module: the
common-trained-substrate design means all three arms consume the SAME warmup, so there is no
separate "OFF path" to bank, and each cell's arm_fingerprint is emitted reuse-eligible
(include_driver_script_in_hash=False) as routine provenance, not as a lineage mint.

GOV-REUSE-1: no prior manifest tags INV-091 or Q-081 (reanalysis_query.py query --claim
INV-091 / --claim Q-081, evidence/experiments/, 2026-07-26: 0 manifests scanned after the
claim filter). Not recoverable; this run is the first.

ASCII-only in all runtime strings (repo rule).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._harness import StepHarness, StepHooks  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.goal_pipeline_tier1 import warmup_train  # noqa: E402
from experiments._lib.q081_landmark_removal import (  # noqa: E402
    LandmarkRemovalConfig,
    LandmarkScrambler,
    assert_behavioural_reach,
    input_statistics_divergence,
)
from experiments._lib.q081_profile import (  # noqa: E402
    q081_profile_kwargs,
    q081_substrate_declaration,
)
from experiments._lib import q081_surrogate as surro  # noqa: E402
from experiments._lib.stream_recorder import StreamTraceRecorder  # noqa: E402
from experiments._lib.trace_store import TraceStore  # noqa: E402
from experiments._metrics import check_degeneracy  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_827_inv091_cross_stream_similarity_band"
QUEUE_ID = "V3-EXQ-827"
CLAIM_IDS: List[str] = ["INV-091"]
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

SEEDS: Tuple[int, ...] = (11, 23, 37)
ARMS: Tuple[str, ...] = ("intact", "decouple", "lockstep")
OFF_ARM = "intact"

WARMUP_EPISODES = 40
EVAL_EPISODES = 10
STEPS_PER_EPISODE = 150
DRY_WARMUP_EPISODES = 2
DRY_EVAL_EPISODES = 2
DRY_STEPS_PER_EPISODE = 20

HARM_HISTORY_LEN = 10
ENV_KWARGS: Dict[str, Any] = dict(
    size=10, num_hazards=1, num_resources=5, harm_history_len=HARM_HISTORY_LEN,
)

# Pairs spanning different subsystems / native rates; see module docstring.
SIMILARITY_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("z_world", "z_self"),
    ("z_world", "operating_mode"),
    ("e1_hidden", "e3_commitment"),
    ("z_world", "z_goal"),
)
SIMILARITY_STREAM_NAMES: Tuple[str, ...] = tuple(sorted(
    {s for pair in SIMILARITY_PAIRS for s in pair}
))

# Pre-registered thresholds (never derived from the run's own statistics).
SIMILARITY_MOVE_FLOOR = 0.02   # C1: each pairwise arm-mean gap must clear this
COMPOSITE_MARGIN = 0.15        # C2: intact must beat both extremes by this many pooled-SD units
MIN_FRESH_COS_SAMPLES = 8
N_SURROGATES = 199


def _env_kwargs() -> Dict[str, Any]:
    return dict(ENV_KWARGS)


def _build_cfg(env: CausalGridWorldV2) -> REEConfig:
    """Shared INTACT-rate config for all three arms. use_anchor_sets=True gives the
    landmark scrambler real reach into z_world via MECH-269 write_anchor (assert_
    behavioural_reach is satisfied by use_invalidation_trigger alone, but the anchor path
    is the one that actually touches z_world, which is what this experiment measures)."""
    kwargs = dict(q081_profile_kwargs())
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        harm_history_len=HARM_HISTORY_LEN,
        **kwargs,
    )
    cfg.hippocampal.use_anchor_sets = True
    return cfg


def _config_slice(arm: str, warmup_episodes: int, eval_episodes: int, steps: int) -> Dict[str, Any]:
    """Declared config_slice for the arm fingerprint -- ONLY what this cell computes."""
    return {
        "arm_id": arm,
        "experiment_type": EXPERIMENT_TYPE,
        "env_kwargs": _env_kwargs(),
        "q081_profile": dict(q081_profile_kwargs()),
        "alpha_world": 0.9,
        "use_anchor_sets": True,
        "warmup_episodes": int(warmup_episodes),
        "eval_episodes": int(eval_episodes),
        "steps_per_episode": int(steps),
        "lockstep": arm == "lockstep",
        "landmark_scramble_mode": "iei_permute" if arm == "decouple" else (
            "off" if arm == "intact" else "none"
        ),
        "similarity_pairs": list(SIMILARITY_PAIRS),
    }


def _restore_intact_rates(agent: REEAgent) -> None:
    agent.clock.e1_steps_per_tick = 1
    agent.clock.e2_steps_per_tick = 3
    agent.clock._e3_base_steps = 10
    agent.clock._current_e3_steps = 10


def _force_lockstep(agent: REEAgent) -> None:
    agent.clock.e1_steps_per_tick = 1
    agent.clock.e2_steps_per_tick = 1
    agent.clock._e3_base_steps = 1
    agent.clock._current_e3_steps = 1


def _cross_stream_similarity_stat(arrays) -> float:
    vals: List[float] = []
    for a, b in SIMILARITY_PAIRS:
        try:
            r, _lag = surro.cross_stream_xcorr(arrays, a, b, max_lag_ticks=8, reduce="first")
        except Exception:
            r = float("nan")
        if np.isfinite(r):
            vals.append(float(r))
    return float(np.mean(vals)) if vals else float("nan")


def _anti_collapse_stat(arrays) -> float:
    zs = np.asarray(arrays["z_self"], dtype=np.float64)
    zw = np.asarray(arrays["z_world"], dtype=np.float64)
    fresh = (
        np.asarray(arrays["z_self__fresh"], dtype=bool)
        & np.asarray(arrays["z_world__fresh"], dtype=bool)
        & np.asarray(arrays["z_self__valid"], dtype=bool)
        & np.asarray(arrays["z_world__valid"], dtype=bool)
    )
    idx = np.flatnonzero(fresh)
    if idx.size < MIN_FRESH_COS_SAMPLES:
        return float("nan")
    a = zs[idx]
    b = zw[idx]
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    den = np.where(den <= 1e-12, np.nan, den)
    cos = num / den
    cos = cos[np.isfinite(cos)]
    if cos.size == 0:
        return float("nan")
    return float(1.0 - np.mean(np.abs(cos)))


def _eval_pass(
    agent: REEAgent,
    env: CausalGridWorldV2,
    arm: str,
    seed: int,
    n_episodes: int,
    steps_per_episode: int,
    store: TraceStore,
    scrambler: Optional[LandmarkScrambler] = None,
    donor_trains: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Run n_episodes of eval on `agent`/`env`, recording the Q-081 trace throughout.

    `agent`'s weights are never updated here (train_mode=False -> torch.no_grad via the
    harness). `scrambler`, if given, is begin_episode/end_episode-bracketed per episode
    (mode="off" for intact banks donor trains; mode="iei_permute" for decouple consumes
    `donor_trains[episode_index]`).
    """
    rec = StreamTraceRecorder(
        agent, run_id=f"{EXPERIMENT_TYPE}_{arm}_seed{seed}_v3",
        store=store, substrate_declaration=q081_substrate_declaration(agent.config),
    )
    acc: Dict[str, Any] = {
        "reward_total": 0.0, "episode_rewards": [], "action_counts": {},
        "state_visitation": {}, "harm_events": 0, "n_steps": 0,
        "episode_lengths": [],
    }

    def on_post_step(*, agent, latent, action, obs_dict, next_obs_dict,
                      harm_signal, done, ticks, residue_metrics, step, **kw) -> None:
        rec.on_step(extras={"reward": float(harm_signal)})
        aidx = int(action.argmax(dim=-1).item())
        acc["action_counts"][aidx] = acc["action_counts"].get(aidx, 0) + 1
        pos_key = f"{int(getattr(env, 'agent_x', -1))},{int(getattr(env, 'agent_y', -1))}"
        acc["state_visitation"][pos_key] = acc["state_visitation"].get(pos_key, 0) + 1
        if float(harm_signal) < 0.0:
            acc["harm_events"] += 1
        acc["n_steps"] += 1

    hooks = StepHooks(on_post_step=on_post_step)
    harness = StepHarness(agent, env, train_mode=False, hooks=hooks, seed=seed)
    agent.eval()

    for ep in range(int(n_episodes)):
        _flat, obs_dict = env.reset()
        agent.reset()
        harness.reset()
        if scrambler is not None:
            donor = donor_trains[ep] if (donor_trains is not None and ep < len(donor_trains)) else None
            scrambler.begin_episode(ep, n_steps=int(steps_per_episode), seed=seed, donor=donor)

        ep_reward = 0.0
        ep_len = 0
        for _t in range(int(steps_per_episode)):
            result = harness.step(obs_dict)
            ep_reward += float(result.harm_signal)
            ep_len += 1
            obs_dict = result.next_obs_dict
            if result.done:
                break
        rec.on_episode_end()
        if scrambler is not None:
            scrambler.end_episode()
        acc["reward_total"] += ep_reward
        acc["episode_rewards"].append(round(ep_reward, 6))
        acc["episode_lengths"].append(ep_len)
        if (ep + 1) % 5 == 0 or (ep + 1) == n_episodes:
            print(f"  [eval] {arm} seed={seed} ep {ep + 1}/{n_episodes}", flush=True)

    n_ep = max(1, int(n_episodes))
    acc["mean_episode_reward"] = float(acc["reward_total"]) / n_ep

    pointer = rec.finalize()
    loaded = store.get(pointer)
    arrays = loaded["arrays"]
    acc["cross_stream_similarity"] = _cross_stream_similarity_stat(arrays)
    acc["anti_collapse"] = _anti_collapse_stat(arrays)
    acc["trace_pointer"] = pointer
    acc["_arrays"] = arrays  # consumed for null validation / donor banking; stripped before manifest write
    return acc


def _run_seed(seed: int, warmup_episodes: int, eval_episodes: int, steps_per_episode: int,
              store: TraceStore, script_path: Path) -> Dict[str, Any]:
    env_warm = CausalGridWorldV2(seed=seed, **_env_kwargs())
    cfg = _build_cfg(env_warm)
    agent = REEAgent(cfg)

    print(f"Seed {seed} Condition warmup_shared", flush=True)
    warmup_train(
        agent, env_warm,
        num_episodes=int(warmup_episodes), steps_per_episode=int(steps_per_episode),
        label=f"warmup_shared seed={seed}", progress_total_episodes=int(warmup_episodes),
    )

    results: Dict[str, Dict[str, Any]] = {}
    donor_trains: List[Any] = []

    with arm_cell(seed, config_slice=_config_slice("intact", warmup_episodes, eval_episodes,
                                                     steps_per_episode),
                  script_path=script_path, config_slice_declared=True,
                  include_driver_script_in_hash=False) as cell:
        print(f"Seed {seed} Condition intact", flush=True)
        _restore_intact_rates(agent)
        scr = LandmarkScrambler(LandmarkRemovalConfig(mode="off", seed=seed))
        scr.attach(agent)
        row = _eval_pass(agent, CausalGridWorldV2(seed=seed, **_env_kwargs()),
                         "intact", seed, eval_episodes, steps_per_episode, store, scrambler=scr)
        donor_trains = list(scr.recorded_trains())
        scr.detach()
        cell.stamp(row)
    results["intact"] = row

    with arm_cell(seed, config_slice=_config_slice("lockstep", warmup_episodes, eval_episodes,
                                                     steps_per_episode),
                  script_path=script_path, config_slice_declared=True,
                  include_driver_script_in_hash=False) as cell:
        print(f"Seed {seed} Condition lockstep", flush=True)
        _force_lockstep(agent)
        row = _eval_pass(agent, CausalGridWorldV2(seed=seed, **_env_kwargs()),
                         "lockstep", seed, eval_episodes, steps_per_episode, store)
        _restore_intact_rates(agent)
        cell.stamp(row)
    results["lockstep"] = row

    with arm_cell(seed, config_slice=_config_slice("decouple", warmup_episodes, eval_episodes,
                                                     steps_per_episode),
                  script_path=script_path, config_slice_declared=True,
                  include_driver_script_in_hash=False,
                  extra_ineligible_reasons=["shared_warmed_agent_across_arms"]) as cell:
        print(f"Seed {seed} Condition decouple", flush=True)
        reach = assert_behavioural_reach(agent, strict=True)
        scr = LandmarkScrambler(LandmarkRemovalConfig(mode="iei_permute", seed=seed))
        scr.attach(agent)
        row = _eval_pass(agent, CausalGridWorldV2(seed=seed, **_env_kwargs()),
                         "decouple", seed, eval_episodes, steps_per_episode, store,
                         scrambler=scr, donor_trains=donor_trains)
        scr.detach()
        row["behavioural_reach"] = reach
        row["preservation_report"] = scr.preservation_report()
        row["input_statistics_divergence"] = input_statistics_divergence(
            {"state_visitation": results["intact"]["state_visitation"],
             "action_counts": results["intact"]["action_counts"],
             "harm_events": results["intact"]["harm_events"],
             "n_steps": results["intact"]["n_steps"],
             "episode_lengths": results["intact"]["episode_lengths"]},
            {"state_visitation": row["state_visitation"],
             "action_counts": row["action_counts"],
             "harm_events": row["harm_events"],
             "n_steps": row["n_steps"],
             "episode_lengths": row["episode_lengths"]},
        )
        cell.stamp(row)
    results["decouple"] = row

    print("verdict: PASS", flush=True)
    return results


def _null_validation(intact_arrays) -> Dict[str, Any]:
    """Q-081's mandatory 'validate the null before using it' check, run once on the
    intact arm's seed[0] trace. Confirms the real statistic is admissible (varies across
    the surrogate ensemble) while the deliberately artefactual rate statistic is ruled out
    (a pure function of the two streams' update periods, killed by construction)."""
    try:
        real_screen = surro.screen_statistic(
            intact_arrays, list(SIMILARITY_STREAM_NAMES), _cross_stream_similarity_stat,
            n_surrogates=64, seed=1,
        )
    except surro.SurrogateDesignError as exc:
        return {"checked": False, "reason": str(exc)}

    def _artefactual(arrays):
        return surro.artefactual_rate_statistic(arrays, "z_world", "operating_mode")

    try:
        artefact_screen = surro.screen_statistic(
            intact_arrays, ["z_world", "operating_mode"], _artefactual,
            n_surrogates=64, seed=1,
        )
    except surro.SurrogateDesignError as exc:
        return {"checked": False, "reason": str(exc)}

    return {
        "checked": True,
        "real_statistic_verdict": real_screen["verdict"],
        "real_statistic_admissible": real_screen["verdict"] == "admissible",
        "artefactual_statistic_verdict": artefact_screen["verdict"],
        "artefactual_statistic_ruled_out": artefact_screen["verdict"] == "ruled_out",
        "null_validated": (
            real_screen["verdict"] == "admissible"
            and artefact_screen["verdict"] == "ruled_out"
        ),
        "detail": {"real": real_screen, "artefactual": artefact_screen},
    }


def _surrogate_p_values(per_arm_arrays: Dict[str, List[Any]]) -> Dict[str, Any]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for arm, arrays_list in per_arm_arrays.items():
        rows = []
        for i, arrays in enumerate(arrays_list):
            try:
                res = surro.evaluate_against_null(
                    arrays, list(SIMILARITY_STREAM_NAMES), _cross_stream_similarity_stat,
                    n_surrogates=N_SURROGATES, seed=100 + i,
                )
                rows.append({"seed_index": i, "p_value": res["p_value"],
                            "observed": res["observed"], "surrogate_mean": res["surrogate_mean"]})
            except surro.SurrogateDesignError as exc:
                rows.append({"seed_index": i, "error": str(exc)})
        out[arm] = rows
    return out


def main(dry_run: bool) -> Dict[str, Any]:
    t0 = time.perf_counter()
    warmup_episodes = DRY_WARMUP_EPISODES if dry_run else WARMUP_EPISODES
    eval_episodes = DRY_EVAL_EPISODES if dry_run else EVAL_EPISODES
    steps_per_episode = DRY_STEPS_PER_EPISODE if dry_run else STEPS_PER_EPISODE
    script_path = Path(__file__)

    store = TraceStore()
    per_seed: Dict[int, Dict[str, Any]] = {}
    for seed in SEEDS:
        per_seed[seed] = _run_seed(seed, warmup_episodes, eval_episodes, steps_per_episode,
                                   store, script_path)

    # ---- null validation (once, intact seed[0]) ----
    null_report = _null_validation(per_seed[SEEDS[0]]["intact"]["_arrays"])

    # ---- per-arm surrogate p-values (all cells; cheap, array-only) ----
    per_arm_arrays = {arm: [per_seed[s][arm]["_arrays"] for s in SEEDS] for arm in ARMS}
    surrogate_report = _surrogate_p_values(per_arm_arrays)

    # ---- per-arm means ----
    per_arm_similarity = {arm: float(np.mean([per_seed[s][arm]["cross_stream_similarity"]
                                             for s in SEEDS])) for arm in ARMS}
    per_arm_anti_collapse = {arm: float(np.mean([per_seed[s][arm]["anti_collapse"]
                                                for s in SEEDS])) for arm in ARMS}
    per_arm_task = {arm: float(np.mean([per_seed[s][arm]["mean_episode_reward"]
                                       for s in SEEDS])) for arm in ARMS}

    # ---- C1: non-degeneracy + directional move ----
    degeneracy = check_degeneracy({
        "cross_stream_similarity_arm_means": [per_arm_similarity[a] for a in ARMS],
    })
    gap_decouple_intact = per_arm_similarity["intact"] - per_arm_similarity["decouple"]
    gap_intact_lockstep = per_arm_similarity["lockstep"] - per_arm_similarity["intact"]
    c1_direction_ok = (
        gap_decouple_intact >= SIMILARITY_MOVE_FLOOR
        and gap_intact_lockstep >= SIMILARITY_MOVE_FLOOR
    )
    c1_pass = bool(degeneracy["non_degenerate"]) and c1_direction_ok

    # ---- composite: min(z_task, z_anticollapse), pooled across all 9 cells ----
    task_pool = np.asarray([per_seed[s][a]["mean_episode_reward"] for a in ARMS for s in SEEDS])
    ac_pool = np.asarray([per_seed[s][a]["anti_collapse"] for a in ARMS for s in SEEDS])
    task_mu, task_sd = float(np.nanmean(task_pool)), float(np.nanstd(task_pool, ddof=1) or 1.0)
    ac_mu, ac_sd = float(np.nanmean(ac_pool)), float(np.nanstd(ac_pool, ddof=1) or 1.0)
    task_sd = task_sd if task_sd > 1e-9 else 1.0
    ac_sd = ac_sd if ac_sd > 1e-9 else 1.0

    composite_by_cell: Dict[str, List[float]] = {a: [] for a in ARMS}
    for a in ARMS:
        for s in SEEDS:
            zt = (per_seed[s][a]["mean_episode_reward"] - task_mu) / task_sd
            za = (per_seed[s][a]["anti_collapse"] - ac_mu) / ac_sd
            composite_by_cell[a].append(min(zt, za))
    per_arm_composite = {a: float(np.mean(composite_by_cell[a])) for a in ARMS}

    # ---- C2: intact composite beats both extremes ----
    c2_vs_decouple = per_arm_composite["intact"] - per_arm_composite["decouple"]
    c2_vs_lockstep = per_arm_composite["intact"] - per_arm_composite["lockstep"]
    c2_pass = (c2_vs_decouple >= COMPOSITE_MARGIN) and (c2_vs_lockstep >= COMPOSITE_MARGIN)

    monotonic_direction = "none"
    if per_arm_composite["decouple"] < per_arm_composite["intact"] < per_arm_composite["lockstep"]:
        monotonic_direction = "improves_with_similarity"
    elif per_arm_composite["decouple"] > per_arm_composite["intact"] > per_arm_composite["lockstep"]:
        monotonic_direction = "degrades_with_similarity"

    if not c1_pass:
        outcome = "FAIL"
        evidence_direction = "unknown"
        label = "substrate_not_ready_requeue"
        note = ("C1 (non-degeneracy) failed: the manipulation did not move the cross-stream "
                "similarity statistic measurably in the pre-registered direction "
                f"(decouple={per_arm_similarity['decouple']:.4f}, "
                f"intact={per_arm_similarity['intact']:.4f}, "
                f"lockstep={per_arm_similarity['lockstep']:.4f}; "
                f"degeneracy_reason={degeneracy['degeneracy_reason']!r}). Per INV-091's own "
                "non-degeneracy guard, the series tested nothing. Not a verdict on INV-091.")
    elif c2_pass:
        outcome = "PASS"
        evidence_direction = "supports"
        label = "cross_stream_similarity_band_supported"
        note = ("C1 and C2 both hold: the intact (moderate-similarity) arm's composite "
                f"({per_arm_composite['intact']:.3f}) exceeds both the decoupled "
                f"({per_arm_composite['decouple']:.3f}) and lockstep "
                f"({per_arm_composite['lockstep']:.3f}) extremes by >= {COMPOSITE_MARGIN}, "
                "despite the two manipulations moving cross-stream similarity in opposite "
                "directions from intact -- a non-monotonic relation. Partial falsifier "
                "coverage: 2 of 7 named ablations tested (landmark removal, forced "
                "lockstep); event broadcasts / mode conditioning / residue feedback / "
                "randomised rates / harm-stream collapse are untested by this run.")
    else:
        outcome = "FAIL"
        evidence_direction = "weakens"
        label = "cross_stream_similarity_band_not_supported"
        note = ("C1 held (similarity moved as designed) but C2 failed: intact's composite "
                f"({per_arm_composite['intact']:.3f}) did not exceed both extremes by the "
                f"pre-registered margin (decouple={per_arm_composite['decouple']:.3f}, "
                f"lockstep={per_arm_composite['lockstep']:.3f}, "
                f"monotonic_direction={monotonic_direction!r}). Partial falsifier coverage: "
                "2 of 7 named ablations tested; see interpretation.note in the manifest.")

    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "evidence_class": "experimental",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "queue_id": QUEUE_ID,
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "similarity_pairs": [list(p) for p in SIMILARITY_PAIRS],
        "thresholds": {
            "similarity_move_floor": SIMILARITY_MOVE_FLOOR,
            "composite_margin": COMPOSITE_MARGIN,
        },
        "per_arm_cross_stream_similarity": per_arm_similarity,
        "per_arm_anti_collapse": per_arm_anti_collapse,
        "per_arm_task_reward": per_arm_task,
        "per_arm_composite": per_arm_composite,
        "criteria": [
            {"name": "C1_similarity_moves_as_designed", "load_bearing": True, "passed": c1_pass},
            {"name": "C2_intact_composite_beats_both_extremes", "load_bearing": True, "passed": c2_pass},
        ],
        "criteria_non_degenerate": {
            "C1_similarity_moves_as_designed": bool(degeneracy["non_degenerate"]),
            "C2_intact_composite_beats_both_extremes": bool(np.isfinite(c2_vs_decouple)
                                                            and np.isfinite(c2_vs_lockstep)),
        },
        "monotonic_direction": monotonic_direction,
        "null_validation": null_report,
        "surrogate_p_values": surrogate_report,
        "interpretation": {"label": label, "note": note},
        "arm_results": [
            {
                "arm_id": a, "seed": s,
                "mean_episode_reward": per_seed[s][a]["mean_episode_reward"],
                "cross_stream_similarity": per_seed[s][a]["cross_stream_similarity"],
                "anti_collapse": per_seed[s][a]["anti_collapse"],
                "episode_rewards": per_seed[s][a]["episode_rewards"],
                "n_steps": per_seed[s][a]["n_steps"],
                "trace_pointer": per_seed[s][a]["trace_pointer"],
                "arm_fingerprint": per_seed[s][a]["arm_fingerprint"],
                **({"behavioural_reach": per_seed[s][a]["behavioural_reach"],
                    "preservation_report": per_seed[s][a]["preservation_report"],
                    "input_statistics_divergence": per_seed[s][a]["input_statistics_divergence"]}
                   if a == "decouple" else {}),
            }
            for a in ARMS for s in SEEDS
        ],
        "aggregates": {
            "per_level": {
                a: {
                    "cross_stream_similarity_mean": per_arm_similarity[a],
                    "anti_collapse_mean": per_arm_anti_collapse[a],
                    "task_reward_mean": per_arm_task[a],
                    "composite_mean": per_arm_composite[a],
                }
                for a in ARMS
            },
        },
        "dose_key": "cross_stream_similarity_mean",
        "supersedes": None,
    }
    manifest.update(degeneracy)

    config_snapshot = {
        "env_kwargs": _env_kwargs(),
        "q081_profile": dict(q081_profile_kwargs()),
        "alpha_world": 0.9,
        "use_anchor_sets": True,
        "warmup_episodes": warmup_episodes,
        "eval_episodes": eval_episodes,
        "steps_per_episode": steps_per_episode,
        "similarity_pairs": [list(p) for p in SIMILARITY_PAIRS],
        "thresholds": manifest["thresholds"],
    }

    out_path = write_flat_manifest(
        manifest, dry_run=dry_run, config=config_snapshot, seeds=list(SEEDS),
        script_path=script_path, started_at=t0,
    )
    manifest["_out_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = main(dry_run=args.dry_run)
    print(f"outcome: {result['outcome']}")
    print(f"interpretation: {result['interpretation']['label']}")

    emit_outcome(
        outcome=result["outcome"],
        manifest_path=result["_out_path"],
        run_id=result["run_id"],
        dry_run=args.dry_run,
    )
