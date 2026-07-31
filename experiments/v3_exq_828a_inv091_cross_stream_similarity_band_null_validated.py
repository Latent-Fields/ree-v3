#!/opt/local/bin/python3
"""V3-EXQ-828a -- INV-091 cross-stream similarity BAND falsifier, null-validated re-run
of V3-EXQ-828 (SAME six-arm design; run-length correction ONLY; IGW-20260730-192 successor).

RUN-LENGTH CORRECTION, NOT A REDESIGN. This script is byte-for-byte the same experimental
design as V3-EXQ-828 (six arms: intact + broadcast_off + mode_decorrelated + residue_off +
rate_randomized + harm_collapse; same patches, same DVs, same C1/C2 criteria, same
thresholds). The ONLY change is the eval-phase episode/step budget: V3-EXQ-828 hardcoded
WARMUP_EPISODES=40 / EVAL_EPISODES=10 / STEPS_PER_EPISODE=150 (1500 eval-phase steps), which
left `null_validation.checked=False` in every one of 827/827a/828's manifests -- the Q-081
surrogate null (`q081_surrogate.plan_blocks`) needs a longer eval-phase array than any of the
three runs supplied (828's own worst-arm deficit: 1500 supplied, 2464-2848 needed). This run
imports the standing default from `experiments/_lib/inv091_driver_defaults.py`
(INV091_WARMUP_EPISODES=40, INV091_EVAL_EPISODES=24, INV091_STEPS_PER_EPISODE=150 -- 3600
eval-phase steps, 26% margin over the 2848-step worst-case deficit) instead of hardcoding its
own copy of the old, undersized numbers. See that module's docstring for the full per-run
deficit trace (827 / 827a / 828) and the `INV091-NULL-VALIDATION-RUN-LENGTH` substrate_queue
entry this run's defaults module was built to satisfy.

WHAT THIS DOES NOT DO. It does NOT re-litigate V3-EXQ-828's own scored verdict. Per
claims.yaml INV-091's evidence_quality_note (2026-07-29 governance, confirmed
failure_autopsy_batch-822a-826-817a-827_2026-07-26 + failure_autopsy_2026-07-28-sweep): 828
weakens the Goldilocks-band prediction (intact does not distinguish itself; residue_off has
the best composite reward) and that verdict is NOT retracted or superseded by this run -- 828
stays exactly as scored. This run's C1/C2 verdict is computed fresh from its own (longer,
null-validated) data, exactly as 828's was from its own data; it is a genuinely new,
higher-powered data point to be read ALONGSIDE 828, not a correction of it. The manifest
carries `null_validation_successor_of: "V3-EXQ-828"` (informational cite, NOT `supersedes` --
828's evidence is not invalidated) and `complements: ["V3-EXQ-827a", "V3-EXQ-828"]`.

Everything below this point (manipulations, DVs, thresholds, DV-symmetry declarations,
ethics preflight, mint policy) is UNCHANGED from V3-EXQ-828 -- see that script's docstring
for full mechanism detail; only the pointers that would drift (queue_id, experiment_type,
run-length source, supersedes/complements bookkeeping) are restated here.

SCOPE -- COMPLEMENTARY TO V3-EXQ-827/827a, NOT A REPLACEMENT (unchanged from 828). The claim
names seven possible ablations. V3-EXQ-827 (and its redesign V3-EXQ-827a) already exercised
two of them (remove commitment landmarks / force lockstep). This run, like 828, exercises the
remaining five: remove event broadcasts, remove mode conditioning, remove residue feedback,
randomise rates, collapse the harm streams.

C1 (non-degeneracy, INV-091's own guard, literal wording, unchanged from 828): at least one of
the five ablation arms must move mean cross_stream_similarity away from intact by >=
SIMILARITY_MOVE_FLOOR. C1 false -> substrate_not_ready_requeue. NEVER a verdict on INV-091
itself.
C2 (the falsifier, band shape, unchanged from 828): rank all six arms by mean
cross_stream_similarity; peak_arm (argmax composite) must be INTERIOR (neither the empirical
min- nor max-similarity arm) AND beat both empirical extremes by >= COMPOSITE_MARGIN.

THRESHOLDS UNCHANGED FROM 827/827a/828 (SIMILARITY_MOVE_FLOOR=0.02, COMPOSITE_MARGIN=0.15) --
properties of the DVs, not of run length or which manipulations produced them.

WHY A COMMON TRAINED SUBSTRATE, NOT SIX SEPARATE WARMUPS (unchanged from 827/827a/828). All
six arms share ONE warmed-up agent per seed (E1/E2 world-forward + E3 harm-eval head trained
once via _lib.goal_pipeline_tier1.warmup_train under the INTACT config); each manipulation is
applied ONLY at eval/recording time. Each arm's eval env is a FRESH CausalGridWorldV2 instance
built with the SAME seed, so all six arms see matched episode layouts.

THE FIVE MANIPULATIONS (unchanged from 828 -- BROADCAST_OFF, MODE_DECORRELATED, RESIDUE_OFF,
RATE_RANDOMIZED, HARM_COLLAPSE; see V3-EXQ-828's module docstring for the full per-manipulation
mechanism writeup, confirmed call-site line numbers, and DV-SYMMETRY declaration -- reproduced
verbatim in the DV-SYMMETRY block below since it is unaffected by the run-length change).

CROSS-STREAM SIMILARITY STATISTIC (unchanged from 828). Mean of |xcorr|
(q081_surrogate.cross_stream_xcorr, reduce="first") over the same FIVE pairs: (z_world,
z_self), (z_world, operating_mode), (e1_hidden, e3_commitment), (z_world, z_goal), (z_harm,
z_harm_a).

ANTI-COLLAPSE STATISTIC. Unchanged: 1 - mean(|cosine(z_self_t, z_world_t)|) over eval steps
where both are fresh.

TASK METRIC. Mean per-episode reward (env harm_signal summed per episode), unchanged.

COMPOSITE (task_plus_anticollapse). Unchanged: min(z_task, z_anticollapse), z-scored by
pooling all 18 (arm x seed) cells.

DV-SYMMETRY (per-arm, mandatory declaration -- REE_Working/CLAUDE.md queue-experiment skill
Step 3.5; reproduced verbatim from V3-EXQ-828 since the manipulations and DVs are unchanged --
only the eval-phase run length differs, and none of these arguments depends on run length):
  BROADCAST_OFF:     removes an entire computation path (invalidation_trigger.step() never
                     runs) -- a genuine change to what is computed, not a relabelling.
  MODE_DECORRELATED: replaces salience.tick()'s actual inputs with fresh independent noise
                     each tick -- a different operating_mode from different inputs every call.
  RESIDUE_OFF:       forcing hypothesis_tag=True changes ResidueField.accumulate's control
                     flow, changing what LATER ticks' residue-dependent reads see.
  RATE_RANDOMIZED:   per-episode-varying tick cadence changes WHEN E2/E3 candidates are
                     generated, scored and committed -- a resampled target is a change to what
                     is computed, not a relabelling.
  HARM_COLLAPSE:     NOTED CONSTRUCTION EFFECT, not a loophole -- the (z_harm, z_harm_a) pair's
                     xcorr is EXPECTED to move by construction (the intended mechanism for
                     reaching the high-similarity end of the band). What is NOT by construction
                     is whether the other four pairs also move and whether the five-pair mean
                     crosses SIMILARITY_MOVE_FLOOR and lands this arm at the correct end of C2's
                     ranking -- a single constructed pair contributing 1/5 of the mean cannot by
                     itself satisfy C1's spread check or C2's peak-interiority test.
None of the five arms' DVs is a pure function of a manipulation the DV cannot see. Run length
does not change any of these arguments -- longer eval windows do not make an invariant
statistic variant, or vice versa.

GOV-REUSE-1 (Step 2.4): the decisive readout this run adds is `null_validation.checked` (and
its dependent `real_statistic_verdict` / `artefactual_statistic_verdict` / `null_validated`
fields) at eval_episodes=24 / 3600 eval-phase steps -- this has never been computed at this
run length by any prior run (827: 1500 steps, checked=False, 12000 needed pre-phase-sync-fix;
827a: 1099 steps, checked=False, 2576 needed; 828: 1500 steps, checked=False, 2464-2848
needed). Not recoverable from any existing manifest by reanalysis -- the surrogate-block
construction genuinely requires the longer array, which no prior run recorded. Proceeds to
run. (Checked: 827, 827a, 828 -- all null_validation.checked=False at their respective run
lengths; no compatible-substrate manifest at eval_episodes>=24 for this driver family exists.)

RE-DERIVE BRAKE (Step 2.5b): counted 1 substrate_ceiling/non_contributory-category autopsy
target for INV-091 (`failure_autopsy_2026-07-28-sweep.json`) -- below the >=2 threshold, brake
does not fire.

ethics_preflight (unchanged from 827/827a/828):
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

MINT (mint-as-you-go). All six arms are emitted reuse-eligible by default
(include_driver_script_in_hash=False, config_slice_declared=True, no
extra_ineligible_reasons) -- unchanged policy from 828. The longer eval_episodes value is part
of `_config_slice`, so this run's fingerprints are naturally distinct from 828's (a different
config_slice), not a collision.

ASCII-only in all runtime strings (repo rule).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._harness import StepHarness, StepHooks  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.goal_pipeline_tier1 import warmup_train  # noqa: E402
from experiments._lib.inv091_driver_defaults import (  # noqa: E402
    INV091_WARMUP_EPISODES,
    INV091_EVAL_EPISODES,
    INV091_STEPS_PER_EPISODE,
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

EXPERIMENT_TYPE = "v3_exq_828a_inv091_cross_stream_similarity_band_null_validated"
QUEUE_ID = "V3-EXQ-828a"
CLAIM_IDS: List[str] = ["INV-091"]
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

SEEDS: Tuple[int, ...] = (11, 23, 37)
ARMS: Tuple[str, ...] = (
    "intact", "broadcast_off", "mode_decorrelated", "residue_off",
    "rate_randomized", "harm_collapse",
)
OFF_ARM = "intact"

# Run-length: imported from the standing driver-family default (was hardcoded
# WARMUP_EPISODES=40 / EVAL_EPISODES=10 / STEPS_PER_EPISODE=150 in 828 -- see module
# docstring "RUN-LENGTH CORRECTION, NOT A REDESIGN").
WARMUP_EPISODES = INV091_WARMUP_EPISODES
EVAL_EPISODES = INV091_EVAL_EPISODES
STEPS_PER_EPISODE = INV091_STEPS_PER_EPISODE
DRY_WARMUP_EPISODES = 2
DRY_EVAL_EPISODES = 2
DRY_STEPS_PER_EPISODE = 20

HARM_HISTORY_LEN = 10
ENV_KWARGS: Dict[str, Any] = dict(
    size=10, num_hazards=1, num_resources=5, harm_history_len=HARM_HISTORY_LEN,
)

# Pairs spanning different subsystems / native rates; see module docstring. Unchanged from
# 827/827a/828.
SIMILARITY_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("z_world", "z_self"),
    ("z_world", "operating_mode"),
    ("e1_hidden", "e3_commitment"),
    ("z_world", "z_goal"),
    ("z_harm", "z_harm_a"),
)
SIMILARITY_STREAM_NAMES: Tuple[str, ...] = tuple(sorted(
    {s for pair in SIMILARITY_PAIRS for s in pair}
))

# Pre-registered thresholds (never derived from the run's own statistics; unchanged from
# 827/827a/828).
SIMILARITY_MOVE_FLOOR = 0.02   # C1: largest |arm - intact| similarity gap must clear this
COMPOSITE_MARGIN = 0.15        # C2: peak-interior arm must beat both extremes by this many
                                # pooled-SD units
MIN_FRESH_COS_SAMPLES = 8
N_SURROGATES = 199
BASE_MAX_LAG_TICKS = 8

# Pre-registered per-episode resampling ranges for RATE_RANDOMIZED (inclusive). Unchanged
# from 828.
RATE_RANDOM_E2_RANGE: Tuple[int, int] = (1, 6)
RATE_RANDOM_E3_RANGE: Tuple[int, int] = (3, 18)


def _env_kwargs() -> Dict[str, Any]:
    return dict(ENV_KWARGS)


def _build_cfg(env: CausalGridWorldV2) -> REEConfig:
    """Shared INTACT-rate config for all six arms. use_anchor_sets=True (unchanged from
    827/827a/828) -- not exercised as a manipulation here, but load-bearing for MECH-269
    anchor writes to keep behaving exactly as the intact reference arm did."""
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
    slice_ = {
        "arm_id": arm,
        "experiment_type": EXPERIMENT_TYPE,
        "env_kwargs": _env_kwargs(),
        "q081_profile": dict(q081_profile_kwargs()),
        "alpha_world": 0.9,
        "use_anchor_sets": True,
        "warmup_episodes": int(warmup_episodes),
        "eval_episodes": int(eval_episodes),
        "steps_per_episode": int(steps),
        "similarity_pairs": list(SIMILARITY_PAIRS),
        "manipulation": arm,
        "base_max_lag_ticks": BASE_MAX_LAG_TICKS,
        "min_fresh_cos_samples": MIN_FRESH_COS_SAMPLES,
    }
    if arm == "broadcast_off":
        slice_["use_invalidation_trigger_eval"] = False
    elif arm == "mode_decorrelated":
        slice_["salience_tick_inputs"] = "iid_noise_per_tick"
    elif arm == "residue_off":
        slice_["update_residue_hypothesis_tag_forced"] = True
    elif arm == "rate_randomized":
        slice_["rate_random_e2_range"] = list(RATE_RANDOM_E2_RANGE)
        slice_["rate_random_e3_range"] = list(RATE_RANDOM_E3_RANGE)
    elif arm == "harm_collapse":
        slice_["z_harm_a_source"] = "z_harm_truncated_to_z_harm_a_dim"
    return slice_


def _restore_intact_rates(agent: REEAgent) -> None:
    agent.clock.e1_steps_per_tick = 1
    agent.clock.e2_steps_per_tick = 3
    agent.clock._e3_base_steps = 10
    agent.clock._current_e3_steps = 10


class _BroadcastOffPatch:
    """MECH-287 broadcast trigger off for the eval window: toggles
    agent.hippocampal.config.use_invalidation_trigger. Boundary events (MECH-288) and MECH-269
    anchor writes (use_anchor_sets) are completely untouched -- see module docstring
    BROADCAST_OFF."""

    def __init__(self, agent: REEAgent) -> None:
        self._agent = agent
        self._prev: Optional[bool] = None

    def attach(self) -> None:
        self._prev = bool(self._agent.hippocampal.config.use_invalidation_trigger)
        self._agent.hippocampal.config.use_invalidation_trigger = False

    def detach(self) -> None:
        if self._prev is not None:
            self._agent.hippocampal.config.use_invalidation_trigger = self._prev
        self._prev = None


class _ModeDecorrelatedPatch:
    """Decorrelate operating_mode from its true drivers for the eval window: patches
    agent.salience.tick so the dacc_bundle / drive_level / per_axis_drive it actually computes
    over are independent per-tick noise. See module docstring MODE_DECORRELATED."""

    def __init__(self, agent: REEAgent, seed: int) -> None:
        self._agent = agent
        self._rng = np.random.default_rng(seed + 900001)
        self._orig: Optional[Callable[..., Any]] = None

    def attach(self) -> None:
        self._orig = self._agent.salience.tick
        orig = self._orig
        rng = self._rng

        def _patched(dacc_bundle=None, drive_level: float = 0.0, is_offline: bool = False,
                     extra_signals=None, per_axis_drive=None, per_axis_combiner: str = "max"):
            fake_dacc = None
            if dacc_bundle is not None:
                fake_dacc = {
                    "pe": float(rng.standard_normal()),
                    "foraging_value": float(rng.uniform(0.0, 1.0)),
                    "choice_difficulty": float(rng.uniform(0.0, 1.0)),
                }
            fake_drive = float(rng.uniform(0.0, 1.0))
            fake_per_axis = None
            if per_axis_drive is not None:
                try:
                    n = len(per_axis_drive)
                except TypeError:
                    n = 1
                fake_per_axis = rng.uniform(0.0, 1.0, size=n)
            return orig(
                dacc_bundle=fake_dacc, drive_level=fake_drive, is_offline=is_offline,
                extra_signals=extra_signals, per_axis_drive=fake_per_axis,
                per_axis_combiner=per_axis_combiner,
            )

        self._agent.salience.tick = _patched

    def detach(self) -> None:
        if self._orig is not None:
            self._agent.salience.tick = self._orig
        self._orig = None


class _ResidueOffPatch:
    """Forces hypothesis_tag=True on every agent.update_residue call for the eval window --
    reuses MECH-094's own accumulation-block mechanism rather than a bespoke no-op. E3
    post_action_update and running-variance tracking inside update_residue are unaffected.
    See module docstring RESIDUE_OFF."""

    def __init__(self, agent: REEAgent) -> None:
        self._agent = agent
        self._orig: Optional[Callable[..., Any]] = None

    def attach(self) -> None:
        self._orig = self._agent.update_residue
        orig = self._orig

        def _patched(harm_signal, world_delta=None, hypothesis_tag: bool = False, owned: bool = True):
            return orig(harm_signal, world_delta=world_delta, hypothesis_tag=True, owned=owned)

        self._agent.update_residue = _patched

    def detach(self) -> None:
        if self._orig is not None:
            self._agent.update_residue = self._orig
        self._orig = None


class _HarmCollapsePatch:
    """Collapses z_harm_a onto (a truncation of) z_harm at the encoder output for the eval
    window: patches agent.latent_stack.encode so the returned LatentState's z_harm_a is
    overwritten with the SAME call's z_harm, truncated to z_harm_a's native dim. LatentState is
    a plain (non-frozen) dataclass, so every downstream consumer of the returned object sees
    the collapsed value. See module docstring HARM_COLLAPSE."""

    def __init__(self, agent: REEAgent) -> None:
        self._agent = agent
        self._orig: Optional[Callable[..., Any]] = None

    def attach(self) -> None:
        self._orig = self._agent.latent_stack.encode
        orig = self._orig

        def _patched(*args, **kwargs):
            state = orig(*args, **kwargs)
            if state.z_harm is not None and state.z_harm_a is not None:
                dim_a = state.z_harm_a.shape[-1]
                state.z_harm_a = state.z_harm[..., :dim_a].clone()
            return state

        self._agent.latent_stack.encode = _patched

    def detach(self) -> None:
        if self._orig is not None:
            self._agent.latent_stack.encode = self._orig
        self._orig = None


def _resample_rates(agent: REEAgent, rng: np.random.Generator) -> Dict[str, int]:
    """Draw fresh e2/e3 rates for one episode from the pre-registered ranges (module
    docstring RATE_RANDOMIZED). e1 stays fixed at 1."""
    e2 = int(rng.integers(RATE_RANDOM_E2_RANGE[0], RATE_RANDOM_E2_RANGE[1] + 1))
    e3 = int(rng.integers(RATE_RANDOM_E3_RANGE[0], RATE_RANDOM_E3_RANGE[1] + 1))
    agent.clock.e1_steps_per_tick = 1
    agent.clock.e2_steps_per_tick = e2
    agent.clock._e3_base_steps = e3
    agent.clock._current_e3_steps = e3
    return {"e1": 1, "e2": e2, "e3": e3}


def _cross_stream_similarity_stat(arrays) -> float:
    vals: List[float] = []
    for a, b in SIMILARITY_PAIRS:
        try:
            r, _lag = surro.cross_stream_xcorr(arrays, a, b, max_lag_ticks=BASE_MAX_LAG_TICKS, reduce="first")
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
    rate_rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Run n_episodes of eval on `agent`/`env`, recording the Q-081 trace throughout.

    `agent`'s weights are never updated here (train_mode=False -> torch.no_grad via the
    harness). When `rate_rng` is given (RATE_RANDOMIZED only), fresh e2/e3 rates are drawn at
    the start of EVERY episode. The returned `_arrays` (this arm's eval-phase trace) is what
    `_null_validation` consumes for the intact arm's seed[0] -- eval-phase throughout, never
    the warmup-phase trace (warmup uses a separate untraced pass via `warmup_train`, see
    `_run_seed`).
    """
    rec = StreamTraceRecorder(
        agent, run_id=f"{EXPERIMENT_TYPE}_{arm}_seed{seed}_v3",
        store=store, substrate_declaration=q081_substrate_declaration(agent.config),
    )
    acc: Dict[str, Any] = {
        "reward_total": 0.0, "episode_rewards": [], "action_counts": {},
        "state_visitation": {}, "harm_events": 0, "n_steps": 0,
        "episode_lengths": [], "sampled_rates": [],
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
        if rate_rng is not None:
            acc["sampled_rates"].append(_resample_rates(agent, rate_rng))

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
    acc["_arrays"] = arrays  # consumed for null validation; stripped before manifest write
    return acc


_PATCH_FACTORIES: Dict[str, Callable[[REEAgent, int], Any]] = {
    "broadcast_off": lambda agent, seed: _BroadcastOffPatch(agent),
    "mode_decorrelated": lambda agent, seed: _ModeDecorrelatedPatch(agent, seed),
    "residue_off": lambda agent, seed: _ResidueOffPatch(agent),
    "harm_collapse": lambda agent, seed: _HarmCollapsePatch(agent),
}


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

    for arm in ARMS:
        with arm_cell(seed, config_slice=_config_slice(arm, warmup_episodes, eval_episodes,
                                                         steps_per_episode),
                      script_path=script_path, config_slice_declared=True,
                      include_driver_script_in_hash=False) as cell:
            print(f"Seed {seed} Condition {arm}", flush=True)
            _restore_intact_rates(agent)

            patch = _PATCH_FACTORIES[arm](agent, seed) if arm in _PATCH_FACTORIES else None
            rate_rng = np.random.default_rng(seed + 900002) if arm == "rate_randomized" else None
            if patch is not None:
                patch.attach()
            try:
                row = _eval_pass(
                    agent, CausalGridWorldV2(seed=seed, **_env_kwargs()),
                    arm, seed, eval_episodes, steps_per_episode, store,
                    rate_rng=rate_rng,
                )
            finally:
                if patch is not None:
                    patch.detach()
                _restore_intact_rates(agent)

            cell.stamp(row)
        results[arm] = row

    print("verdict: PASS", flush=True)
    return results


def _null_validation(intact_arrays) -> Dict[str, Any]:
    """Q-081's mandatory 'validate the null before using it' check, run once on the
    intact arm's seed[0] EVAL-PHASE trace (see `_eval_pass`'s `_arrays` -- never warmup-phase;
    warmup runs through the separate untraced `warmup_train` path and never touches `store`).
    Confirms the real statistic is admissible (varies across the surrogate ensemble) while the
    deliberately artefactual rate statistic is ruled out (a pure function of the two streams'
    update periods, killed by construction). At EVAL_EPISODES=24 / STEPS_PER_EPISODE=150 (3600
    eval-phase steps -- see module docstring), this is expected to clear
    q081_surrogate.plan_blocks' minimum-array-length bound where 828's 1500-step eval-phase
    array (worst-arm deficit 2464-2848 needed) did not."""
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

    # ---- C1: non-degeneracy (N-arm generalisation; see module docstring) ----
    degeneracy = check_degeneracy({
        "cross_stream_similarity_arm_means": [per_arm_similarity[a] for a in ARMS],
    })
    moves_vs_intact = {a: per_arm_similarity[a] - per_arm_similarity["intact"]
                       for a in ARMS if a != "intact"}
    max_abs_move_arm = max(moves_vs_intact, key=lambda a: abs(moves_vs_intact[a]))
    max_abs_move = abs(moves_vs_intact[max_abs_move_arm])
    c1_pass = bool(degeneracy["non_degenerate"]) and (max_abs_move >= SIMILARITY_MOVE_FLOOR)

    # ---- composite: min(z_task, z_anticollapse), pooled across all 18 cells ----
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

    # ---- C2: band shape (N-arm generalisation of "intact beats both extremes") ----
    ranked_by_similarity = sorted(ARMS, key=lambda a: per_arm_similarity[a])
    sim_min_arm, sim_max_arm = ranked_by_similarity[0], ranked_by_similarity[-1]
    peak_arm = max(ARMS, key=lambda a: per_arm_composite[a])
    peak_is_interior = peak_arm not in (sim_min_arm, sim_max_arm)
    margin_vs_min = per_arm_composite[peak_arm] - per_arm_composite[sim_min_arm]
    margin_vs_max = per_arm_composite[peak_arm] - per_arm_composite[sim_max_arm]
    c2_pass = bool(
        peak_is_interior
        and np.isfinite(margin_vs_min) and margin_vs_min >= COMPOSITE_MARGIN
        and np.isfinite(margin_vs_max) and margin_vs_max >= COMPOSITE_MARGIN
    )

    # Secondary, NON-LOAD-BEARING diagnostic: does intact SPECIFICALLY beat both empirical
    # extremes (827/827a/828's own formula)? Only meaningful when intact is not itself an
    # extreme.
    intact_beats_both_extremes = None
    intact_vs_min = intact_vs_max = float("nan")
    if sim_min_arm != "intact" and sim_max_arm != "intact":
        intact_vs_min = per_arm_composite["intact"] - per_arm_composite[sim_min_arm]
        intact_vs_max = per_arm_composite["intact"] - per_arm_composite[sim_max_arm]
        intact_beats_both_extremes = bool(
            intact_vs_min >= COMPOSITE_MARGIN and intact_vs_max >= COMPOSITE_MARGIN
        )

    ranked_composites = [per_arm_composite[a] for a in ranked_by_similarity]
    is_nondecreasing = all(
        ranked_composites[i] <= ranked_composites[i + 1] + 1e-9
        for i in range(len(ranked_composites) - 1)
    )
    is_nonincreasing = all(
        ranked_composites[i] >= ranked_composites[i + 1] - 1e-9
        for i in range(len(ranked_composites) - 1)
    )
    monotonic_direction = "none"
    if is_nondecreasing and not is_nonincreasing:
        monotonic_direction = "improves_with_similarity"
    elif is_nonincreasing and not is_nondecreasing:
        monotonic_direction = "degrades_with_similarity"

    if not c1_pass:
        outcome = "FAIL"
        evidence_direction = "unknown"
        label = "substrate_not_ready_requeue"
        note = ("C1 (non-degeneracy) failed: no ablation moved the cross-stream similarity "
                f"statistic measurably away from intact (largest move: {max_abs_move_arm} at "
                f"{moves_vs_intact[max_abs_move_arm]:+.4f}, floor={SIMILARITY_MOVE_FLOOR}; "
                f"per-arm similarity={ {a: round(per_arm_similarity[a], 4) for a in ARMS} }; "
                f"degeneracy_reason={degeneracy['degeneracy_reason']!r}). Per INV-091's own "
                "non-degeneracy guard, the series tested nothing. Not a verdict on INV-091. "
                "Null-validated re-run of V3-EXQ-828's design (run-length correction only) -- "
                "read alongside V3-EXQ-827a and V3-EXQ-828, not in isolation.")
    elif c2_pass:
        outcome = "PASS"
        evidence_direction = "supports"
        label = "cross_stream_similarity_band_supported"
        note = ("C1 and C2 both hold: the interior-similarity arm "
                f"'{peak_arm}' (similarity={per_arm_similarity[peak_arm]:.4f}, composite="
                f"{per_arm_composite[peak_arm]:.3f}) exceeds both the empirical low-similarity "
                f"extreme '{sim_min_arm}' ({per_arm_composite[sim_min_arm]:.3f}) and the "
                f"high-similarity extreme '{sim_max_arm}' ({per_arm_composite[sim_max_arm]:.3f}) "
                f"by >= {COMPOSITE_MARGIN} -- a non-monotonic relation across five ablations "
                "(event broadcasts / mode conditioning / residue feedback / randomised rates / "
                "harm-stream collapse). Secondary diagnostic (non-load-bearing): "
                f"intact_beats_both_extremes={intact_beats_both_extremes}. Null-validated "
                "re-run of V3-EXQ-828's design (run-length correction only, NOT a redesign) -- "
                "read alongside V3-EXQ-827a and V3-EXQ-828, not in isolation.")
    else:
        outcome = "FAIL"
        evidence_direction = "weakens"
        label = "cross_stream_similarity_band_not_supported"
        note = ("C1 held (at least one ablation moved similarity) but C2 failed: the peak-"
                f"composite arm '{peak_arm}' is "
                f"{'interior' if peak_is_interior else 'AT AN EMPIRICAL EXTREME (' + peak_arm + ' == ' + (sim_min_arm if peak_arm == sim_min_arm else sim_max_arm) + ')'}"
                f", margin_vs_min={margin_vs_min:.3f}, margin_vs_max={margin_vs_max:.3f} "
                f"(threshold {COMPOSITE_MARGIN}); monotonic_direction={monotonic_direction!r}. "
                f"Secondary diagnostic (non-load-bearing): intact_beats_both_extremes="
                f"{intact_beats_both_extremes}. Null-validated re-run of V3-EXQ-828's design "
                "(run-length correction only, NOT a redesign) -- read alongside V3-EXQ-827a and "
                "V3-EXQ-828, not in isolation. Does NOT retract or supersede V3-EXQ-828's own "
                "scored weakens verdict; this is a fresh, higher-powered data point.")

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
            "base_max_lag_ticks": BASE_MAX_LAG_TICKS,
            "rate_random_e2_range": list(RATE_RANDOM_E2_RANGE),
            "rate_random_e3_range": list(RATE_RANDOM_E3_RANGE),
        },
        "per_arm_cross_stream_similarity": per_arm_similarity,
        "per_arm_anti_collapse": per_arm_anti_collapse,
        "per_arm_task_reward": per_arm_task,
        "per_arm_composite": per_arm_composite,
        "criteria": [
            {"name": "C1_at_least_one_ablation_moves_similarity", "load_bearing": True, "passed": c1_pass},
            {"name": "C2_interior_similarity_arm_beats_both_extremes", "load_bearing": True, "passed": c2_pass},
        ],
        "criteria_non_degenerate": {
            "C1_at_least_one_ablation_moves_similarity": bool(degeneracy["non_degenerate"]),
            "C2_interior_similarity_arm_beats_both_extremes": bool(
                np.isfinite(margin_vs_min) and np.isfinite(margin_vs_max)
            ),
        },
        "band_shape": {
            "ranked_by_similarity": ranked_by_similarity,
            "sim_min_arm": sim_min_arm,
            "sim_max_arm": sim_max_arm,
            "peak_composite_arm": peak_arm,
            "peak_is_interior": peak_is_interior,
            "margin_vs_min": margin_vs_min,
            "margin_vs_max": margin_vs_max,
            "intact_beats_both_extremes": intact_beats_both_extremes,
            "intact_vs_min": intact_vs_min,
            "intact_vs_max": intact_vs_max,
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
                **({"sampled_rates": per_seed[s][a]["sampled_rates"]}
                   if a == "rate_randomized" else {}),
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
        "null_validation_successor_of": "V3-EXQ-828",
        "complements": ["V3-EXQ-827a", "V3-EXQ-828"],
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
