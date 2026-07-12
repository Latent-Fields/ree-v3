#!/opt/local/bin/python3
"""Shared maturation-curriculum frozen-representation recipes + a frozen-prefix
tensor cache for the INV-064 / INV-088 / INV-089 family.

Members: V3-EXQ-740a (INV-064, z_world IV leg), V3-EXQ-744 (INV-088, z_world DV
coupling), V3-EXQ-743 (INV-089, z_harm leg), and their successors (e.g. a
higher-seed V3-EXQ-744a).

Design ref: REE_assembly/evidence/planning/arm_reuse_fingerprint_plan.md
(sec 6 "whole-cell reuse only"; sec 9 "no warm-start"; sec 10 this module).

WHY THIS MODULE EXISTS
----------------------
Each family member independently runs an EXPENSIVE maturation prefix -- a
warmup_train(onset) (z_world leg) or a HarmEncoder maturation (z_harm leg),
followed by a FIXED frozen-dataset collection -- then diverges only in a CHEAP
tail (which target it ridge-probes / trains an evaluator head on). The prefix is
a DETERMINISTIC pure function of (substrate, env_kwargs, recipe, seed, onset)
within a machine_class. Verified bit-identical 2026-07-12: two in-process runs of
v3_exq_744 _run_cell(42, 4) produced identical arm_fingerprint
(ddce40b7fbc89a6aa6df2900c50b315e38e033f9786bcd1028e5bc045636c8f5) and every
metric to full float precision. So the earlier
`frozen_representation_from_maturation_trajectory` reuse-ineligibility flag was
EMPIRICALLY FALSE (the trajectory is regenerated deterministically inside the
cell) and is dropped here.

Two things this unlocks:

  (1) SHARED RECIPE -> fingerprints match by construction. Every member builds its
      frozen prefix from THIS module. Because experiments/_lib/** is in the
      arm_fingerprint substrate-hash glob and drivers are excluded via
      include_driver_script_in_hash=False, differently-driven siblings that share a
      (seed, onset, env_kwargs, recipe) produce the SAME arm_fingerprint -- the
      precondition for any cross-sibling reuse.

  (2) FROZEN-PREFIX TENSOR CACHE (frozen_prefix_cache) memoises the frozen encoder
      (agent / HarmEncoder state_dict) + the collected dataset tensors, keyed on
      UPSTREAM-ONLY params, so a later same-(seed, onset) cell skips warmup+collect.

      This is a NEW mechanism, COMPLEMENTARY to experiments/_lib/arm_reuse.py:
        - arm_reuse.py reuses recorded scalar CELL METRICS (whole-cell,
          metrics-only, plan sec 9) -- it helps ONLY an exact-config re-run, because
          a differently-targeted sibling computes DIFFERENT metrics off the same
          frozen representation and so has nothing to read.
        - THIS cache memoises the shared TENSOR prefix, so a differently-targeted
          sibling still shares the expensive artifact and re-derives its own metrics
          from it cheaply.
      It is NOT the plan-sec-6-excluded partial-training warm-start: the downstream
      evaluator head still trains FRESH from a fixed init on each cell; only the
      deterministic frozen (encoder, dataset) prefix is memoised. Same Regime-A
      determinism arm_fingerprint already assumes.

GOVERNING ASYMMETRY (plan sec 2): a false cache HIT corrupts a scientific
conclusion; a false cache MISS only wastes compute. The cache key is therefore
OVER-inclusive -- substrate_hash (see DEPENDENCY-SCOPED SUBSTRATE HASHING below),
the FULL env_kwargs, every recipe scalar, machine_class, seed, onset, and the leg
tag. Over-inclusion causes only (cheap) false misses. The stored key is re-verified
on load; any mismatch / unreadable / partial file is treated as a MISS.

DEPENDENCY-SCOPED SUBSTRATE HASHING (plan sec 11, added 2026-07-12)
------------------------------------------------------------------
The prefix substrate_hash was originally the WHOLE ree_core/** + experiments/_lib/**
trees (121 files), so an edit to ANY module -- sleep, the hippocampal proposer, an
unrelated env, most of policy -- busted a prefix that never executes a line of it.
Given how continuously ree_core churns, that was the dominant source of (cheap but
real) false misses. This module now hashes ONLY the per-leg DECLARED SUBSTRATE SCOPE
(_LEG_SUBSTRATE_SCOPE): the closure the frozen prefix actually depends on -- 24 files
(world) / 19 files (harm) instead of 121. compute_substrate_hash's default (scope=
None) is unchanged, so the global sec-9 arm_fingerprint path and every existing
fingerprint are byte-identical; only THIS cache key narrows.

SOUNDNESS (the one thing that must not be wrong -- plan sec 2/11): narrowing is safe
ONLY if the declared scope is a provable SUPERSET of every file that can change the
frozen-prefix output. The scope = (a) the EXECUTED-file closure of build+warmup+
collect (captured by a call-trace: files whose code actually runs) UNION (b) its
transitive DATA-closure (any repo module a scope file value-imports a module-level
CONSTANT from -- here only ree_core.regulators SITE_* string labels). CODE
(class/function) imports of NON-executed modules are deliberately EXCLUDED: the
call-trace proves their functions never run, so their bodies cannot affect the
deterministic prefix. Two guards keep this an over-approximation (false-miss-only),
enforced by verify_scope_conservatism (run in the scope test + opt-in at runtime via
REE_PREFIX_SCOPE_GUARD=1): guard 1 (call-trace) asserts every executed repo file is in
scope; guard 2 (static AST) asserts the scope is a FIXPOINT of the data-closure
operator. The guard MACHINERY is scope-generic and lives in the shared
experiments/_lib/substrate_scope_guard.py module (promoted 2026-07-12 so the global
sec-9 arm_fingerprint path can run the same guards); this module only declares the
per-leg scope + thin leg-keyed wrappers. A refactor that moves executed code out of a declared file, or adds a
constant value-import from outside the scope, trips a guard loudly rather than
silently under-approximating. substrate_scope_declared + the glob list are recorded in
the cache blob + provenance for audit (mirrors config_slice_declared).

CALLER CONTRACT (bit-identity)
------------------------------
Call mature_and_collect_world / mature_and_collect_harm as the FIRST RNG-consuming
step inside an `arm_cell(...)` context, before any other RNG use in the cell. The
arm_cell __enter__ resets all RNG to `seed`; this module then builds env + agent in
the SAME order the inline 740a/744/743 scripts used, so the returned frozen agent,
the FRESH evaluator-head inits (captured pre-maturation), and the dataset tensors
are bit-identical to the inline path. The `fresh_head_inits` MUST be used to re-init
the evaluator head (the returned agent's own heads are post-maturation and must not
be used as the fresh init).

ASCII-only output (repo rule). torch is required (tensors); the cache-key helpers
reuse the stdlib-only arm_fingerprint primitives.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

_LIB_DIR = Path(__file__).resolve().parents[1]        # experiments/_lib
_EXP_DIR = _LIB_DIR.parent                             # experiments
_REPO_ROOT = _EXP_DIR.parent                           # ree-v3
for _p in (str(_REPO_ROOT), str(_EXP_DIR), str(_LIB_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Cache-key primitives -- reuse the arm_fingerprint content-hash machinery so the
# key obeys the same over-inclusion discipline (false-miss-only).
from arm_fingerprint import (  # noqa: E402
    _canonical_json,
    _sha256_hex,
    compute_substrate_hash,
    machine_class,
)
# Conservatism guards live in a shared, scope-generic module (plan sec 11) so any
# experiment -- not just this prototype -- can prove a declared substrate scope is a
# safe over-approximation. This module keeps only the per-leg scope declarations +
# thin leg-keyed wrappers; the guard MACHINERY (call-trace + static data-closure) is
# imported. Aliased to the historical private names so existing callers/tests are
# unaffected.
from substrate_scope_guard import (  # noqa: E402
    static_data_closure as _static_data_closure,
    traced_execution_files as _traced_execution_files,
    verify_scope_static as _verify_scope_static_files,
    verify_scope_conservatism as _verify_scope_conservatism_files,
)
from _lib.goal_pipeline_tier1 import ArmSpec, build_config, warmup_train  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.latent.stack import HarmEncoder  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# v2 = dependency-scoped substrate hashing (plan sec 11). The key now folds in the
# declared scope, so a v1 blob (whole-tree substrate_hash) can never collide with a
# v2 key -- old blobs simply cache-MISS (safe, false-miss-only).
PREFIX_CACHE_SCHEMA = "maturation_prefix/v2"

# Canonical collection RNG base shared across the family (740a/743/744 all use
# 70000 for the collection env + action RNG). Kept as the default; callers may
# override, and the value enters the cache key so a change never false-hits.
DEFAULT_COLLECT_SEED_BASE = 70000
# z_harm leg maturation RNG base (743 uses 60000 for the maturation env + actions).
DEFAULT_HARM_MATURE_SEED_BASE = 60000

# Evaluator-head attribute names captured as FRESH inits per leg (only those that
# exist on the built agent's e3 are captured).
_WORLD_HEAD_ATTRS = ("harm_eval_head", "benefit_eval_head")
_HARM_HEAD_ATTRS = ("harm_eval_z_harm_head",)


# ---------------------------------------------------------------------------
# Frozen-prefix tensor cache (machine-local; Regime A).
# ---------------------------------------------------------------------------

def _cache_dir(cache_dir: Optional[Path]) -> Path:
    if cache_dir is not None:
        d = Path(cache_dir)
    else:
        env = os.environ.get("REE_PREFIX_CACHE_DIR")
        d = Path(env) if env else (Path.home() / ".ree_maturation_prefix_cache")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_disabled() -> bool:
    return os.environ.get("REE_PREFIX_CACHE_DISABLE", "").strip() not in ("", "0", "false", "False")


# ---------------------------------------------------------------------------
# Declared substrate scope (plan sec 11). Each tuple is a set of repo-root-relative
# paths (exact, one file each -- valid single-match globs for compute_substrate_hash).
# It is the (executed-file closure) UNION (transitive data-closure) for that leg, so
# a provable SUPERSET of every file that can change the frozen prefix -- see the module
# docstring "DEPENDENCY-SCOPED SUBSTRATE HASHING". Grounded in a call-trace of
# build+warmup+collect and a leaf-kind AST data-closure; both re-verified by
# _verify_scope_conservatism (scope test + REE_PREFIX_SCOPE_GUARD=1). HARM subset of
# WORLD by construction (the harm leg runs no StepHarness / trajectory generation).
_WORLD_SUBSTRATE_SCOPE: Tuple[str, ...] = (
    "experiments/__init__.py",
    "experiments/_harness.py",
    "experiments/_lib/arm_fingerprint.py",
    "experiments/_lib/baselines/maturation_curriculum.py",
    "experiments/_lib/goal_pipeline_tier1.py",
    "ree_core/agent.py",
    "ree_core/cingulate/dacc.py",
    "ree_core/environment/causal_grid_world.py",
    "ree_core/goal.py",
    "ree_core/heartbeat/beta_gate.py",
    "ree_core/heartbeat/clock.py",
    "ree_core/hippocampal/module.py",
    "ree_core/latent/stack.py",
    "ree_core/latent/theta_buffer.py",
    "ree_core/neuromodulation/serotonin.py",
    "ree_core/predictors/e1_deep.py",
    "ree_core/predictors/e2_fast.py",
    "ree_core/predictors/e3_score_diversity.py",
    "ree_core/predictors/e3_selector.py",
    "ree_core/regulators/__init__.py",             # data-closure: SITE_* re-export chain
    "ree_core/regulators/simulation_mode_rule_gate.py",  # data-closure: SITE_* leaf literals
    "ree_core/residue/field.py",
    "ree_core/utils/config.py",
    "ree_core/utils/per_axis_drive.py",
)
_HARM_SUBSTRATE_SCOPE: Tuple[str, ...] = (
    "experiments/_harness.py",
    "experiments/_lib/arm_fingerprint.py",
    "experiments/_lib/baselines/maturation_curriculum.py",
    "ree_core/agent.py",
    "ree_core/environment/causal_grid_world.py",
    "ree_core/heartbeat/beta_gate.py",
    "ree_core/heartbeat/clock.py",
    "ree_core/hippocampal/module.py",
    "ree_core/latent/stack.py",
    "ree_core/latent/theta_buffer.py",
    "ree_core/neuromodulation/serotonin.py",
    "ree_core/predictors/e1_deep.py",
    "ree_core/predictors/e2_fast.py",
    "ree_core/predictors/e3_score_diversity.py",
    "ree_core/predictors/e3_selector.py",
    "ree_core/regulators/__init__.py",
    "ree_core/regulators/simulation_mode_rule_gate.py",
    "ree_core/residue/field.py",
    "ree_core/utils/config.py",
)
_LEG_SUBSTRATE_SCOPE: Dict[str, Tuple[str, ...]] = {
    "world": _WORLD_SUBSTRATE_SCOPE,
    "harm": _HARM_SUBSTRATE_SCOPE,
}


def _scope_provenance(leg: str) -> Dict[str, Any]:
    """Audit record of the declared substrate scope, mirroring config_slice_declared.
    Embedded in the cache blob + returned provenance so a reviewer / future consumer can
    confirm which reuse contract minted a blob (plan sec 11)."""
    scope = _LEG_SUBSTRATE_SCOPE.get(leg)
    return {
        "substrate_scope_declared": scope is not None,
        "substrate_scope": list(scope) if scope is not None else None,
    }


def _prefix_key(leg: str, upstream: Dict[str, Any]) -> str:
    """Content-addressed key over UPSTREAM-only params + SCOPED substrate + machine_class.

    Hashes ONLY the leg's declared substrate scope (_LEG_SUBSTRATE_SCOPE), not the whole
    ree_core/** + experiments/_lib/** trees, so an edit to a module the frozen prefix
    never executes (sleep, hippocampal proposer, most of policy, unrelated envs, the E3
    downstream heads) no longer busts the key -- while any change INSIDE the declared
    closure still refuses the cached prefix (plan sec 11). An unknown leg falls back to
    scope=None (hash everything -- the safe default). The driver script is EXCLUDED (this
    module is already in every scope), so differently-driven siblings sharing the same
    upstream produce the same key.
    """
    scope = _LEG_SUBSTRATE_SCOPE.get(leg)  # None -> compute_substrate_hash hashes ALL (safe)
    if scope is not None and os.environ.get("REE_PREFIX_SCOPE_GUARD", "").strip() not in ("", "0", "false", "False"):
        # Opt-in cheap static conservatism guard (guard 2). Raises loudly if the scope
        # is no longer data-closed or a declared file is missing. The expensive call-trace
        # guard (guard 1) lives in the scope test.
        _verify_scope_static(leg)
    sub = compute_substrate_hash(scope=scope)
    payload = {
        "schema": PREFIX_CACHE_SCHEMA,
        "leg": leg,
        "substrate_hash": sub["substrate_hash"],
        "substrate_scope_declared": scope is not None,
        # The declared scope is part of the key: narrowing / widening the reuse
        # contract must change the key (mirrors config_slice being in arm_fingerprint).
        "substrate_scope": list(scope) if scope is not None else None,
        "machine_class": machine_class(),
        "upstream": upstream,
    }
    return _sha256_hex(_canonical_json(payload).encode("utf-8"))


# ---------------------------------------------------------------------------
# Conservatism guards (plan sec 11). A declared scope that UNDER-approximates is a
# false-HIT bug, so the scope must be a provable over-approximation of everything the
# frozen prefix can execute or read. Two independent checks close the code-exec (a) and
# data-read (b) channels; both now live in the scope-generic substrate_scope_guard module
# (imported above) so any experiment can run them. These leg-keyed wrappers just look up
# the per-leg declared scope and delegate.
# ---------------------------------------------------------------------------


def _verify_scope_static(leg: str) -> None:
    """Guard 2 (cheap, static AST) for a leg: delegate to the shared guard with this leg's
    declared scope. Raises AssertionError loudly if the scope no longer exists or is not a
    data-closed fixpoint (plan sec 11)."""
    _verify_scope_static_files(_LEG_SUBSTRATE_SCOPE[leg], label="prefix-cache scope[%s]" % leg)


def verify_scope_conservatism(leg: str, run_once=None) -> Dict[str, Any]:
    """Full conservatism check for a leg's declared substrate scope (plan sec 11).

    guard 2 (always): static data-closure fixpoint + existence.
    guard 1 (only if `run_once` is given): execute a real (cheap) frozen-prefix cell under
    a call-trace and assert EVERY executed repo file is in the declared scope. `run_once`
    must invoke mature_and_collect_world / _harm once for this leg.

    Delegates to substrate_scope_guard.verify_scope_conservatism; returns its report dict
    augmented with the historical `leg` / `n_declared` keys. Raises AssertionError on any
    violation. This is the mechanism that makes author-declared narrowing safe."""
    scope = _LEG_SUBSTRATE_SCOPE[leg]
    report = _verify_scope_conservatism_files(scope, run_once, label="prefix-cache scope[%s]" % leg)
    report["leg"] = leg
    report["n_declared"] = len(scope)
    return report


def _cache_load(key: str, cache_dir: Optional[Path], logger) -> Optional[Dict[str, Any]]:
    if _cache_disabled():
        return None
    path = _cache_dir(cache_dir) / f"{key}.pt"
    if not path.is_file():
        return None
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # corrupt / partial / version skew -> treat as MISS
        logger("prefix_cache: unreadable (%s) -> MISS: %s" % (path.name, exc))
        return None
    # Defensive: re-verify the stored key matches (guards against any filename
    # collision or a truncated write). Any mismatch -> MISS.
    if not isinstance(blob, dict) or blob.get("key") != key:
        logger("prefix_cache: key mismatch on %s -> MISS" % path.name)
        return None
    return blob


def _cache_store(key: str, blob: Dict[str, Any], cache_dir: Optional[Path], logger) -> None:
    if _cache_disabled():
        return
    d = _cache_dir(cache_dir)
    path = d / f"{key}.pt"
    tmp = d / f"{key}.pt.tmp.{os.getpid()}"
    try:
        torch.save(blob, tmp)
        os.replace(tmp, path)  # atomic within a filesystem; safe under parallel workers
    except Exception as exc:
        logger("prefix_cache: store failed for %s: %s" % (path.name, exc))
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------

def _row_vec(x: Any) -> torch.Tensor:
    """Coerce an obs harm field to a [1, D] float32 row tensor (matches 740a/743/744)."""
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu().to(torch.float32)
    else:
        t = torch.as_tensor(np.asarray(x, dtype=np.float32))
    return t.reshape(1, -1)


# ---------------------------------------------------------------------------
# z_world leg (740a / 744 / 744a).
# ---------------------------------------------------------------------------

def build_world_agent(seed: int, env_kwargs: Dict[str, Any]) -> Tuple[REEAgent, CausalGridWorldV2]:
    """Bit-identical to 740a/744 `_build_agent`: ArmSpec("maturation") -> build_config
    -> REEAgent. Consumes global RNG for weight init, so call under a fresh reset."""
    env = CausalGridWorldV2(seed=seed, **env_kwargs)
    arm = ArmSpec(arm_id="maturation", gap4_operating=False)
    cfg = build_config(env, arm)  # from_dims path -> alpha_world=0.9 (SD-008)
    agent = REEAgent(cfg)
    return agent, env


def _freeze_world(agent: REEAgent) -> None:
    """Freeze E1 + E2.world_transition + E2.world_action_encoder + latent_stack
    (identical to 740a/744 `_freeze`)."""
    params = (
        list(agent.e1.parameters())
        + list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters())
        + list(agent.latent_stack.parameters())
    )
    for p in params:
        p.requires_grad_(False)


def collect_world_dataset(agent: REEAgent, seed: int, env_kwargs: Dict[str, Any],
                          steps_per_ep: int, n_episodes: int,
                          collect_seed_base: int = DEFAULT_COLLECT_SEED_BASE) -> Dict[str, torch.Tensor]:
    """Replay a FIXED action sequence through a FIXED collection env, encoding z_world
    with the (frozen) agent. TARGET-AGNOSTIC SUPERSET matching 740a's collect exactly
    (744's collect is the strict subset without Y; computing Y consumes no RNG and does
    not alter env stepping, so this is bit-identical to BOTH siblings for their fields):

      Z      = z_world[t]        (frozen encoding, the only per-arm variable)
      Y      = harm_target[t]    (realized harm abs(signal<0); 740a DV + pos-ctrl target)
      Hcur   = harm_obs[t]       (current harm-relevant world features)
      Hnext  = harm_obs[t+1]     (predictive IV / DV-target source: 51-dim)
      Zprev, A, Zcurr            (E2 forward-model transition triples)
    """
    collect_env = CausalGridWorldV2(seed=collect_seed_base + seed, **env_kwargs)
    act_rng = np.random.default_rng(collect_seed_base + seed)
    action_dim = collect_env.action_dim

    z_list = []
    y_list = []
    hcur_list = []
    hnext_list = []
    zprev_list = []
    a_list = []
    zcurr_list = []

    agent.eval()
    with torch.no_grad():
        for _ep in range(n_episodes):
            _, obs_dict = collect_env.reset()
            agent.reset()
            z_world_prev = None
            action_prev = None
            for _step in range(steps_per_ep):
                latent = agent.sense(
                    obs_dict["body_state"],
                    obs_dict["world_state"],
                    obs_harm=obs_dict.get("harm_obs"),
                    obs_harm_a=obs_dict.get("harm_obs_a"),
                    obs_harm_history=obs_dict.get("harm_history"),
                )
                z_world = latent.z_world.detach().cpu()

                action = int(act_rng.integers(0, action_dim))
                a_oh = torch.zeros(1, action_dim)
                a_oh[0, action] = 1.0

                _, harm_signal, done, _info, obs_next = collect_env.step(action)
                harm_target = abs(float(harm_signal)) if float(harm_signal) < 0 else 0.0

                z_list.append(z_world)
                y_list.append(harm_target)
                hcur_list.append(_row_vec(obs_dict.get("harm_obs")))
                hnext_list.append(_row_vec(obs_next.get("harm_obs")))

                if z_world_prev is not None and action_prev is not None:
                    zprev_list.append(z_world_prev)
                    a_list.append(action_prev)
                    zcurr_list.append(z_world)

                z_world_prev = z_world
                action_prev = a_oh
                obs_dict = obs_next
                if done:
                    break

    out = {
        "Z": torch.cat(z_list, dim=0),
        "Y": torch.tensor(y_list, dtype=torch.float32).unsqueeze(1),
        "Hcur": torch.cat(hcur_list, dim=0),
        "Hnext": torch.cat(hnext_list, dim=0),
    }
    if zprev_list:
        out["Zprev"] = torch.cat(zprev_list, dim=0)
        out["A"] = torch.cat(a_list, dim=0)
        out["Zcurr"] = torch.cat(zcurr_list, dim=0)
    return out


def _capture_fresh_heads(agent: REEAgent, attrs) -> Dict[str, Dict[str, Any]]:
    return {name: copy.deepcopy(getattr(agent.e3, name).state_dict())
            for name in attrs if hasattr(agent.e3, name)}


def mature_and_collect_world(
    seed: int,
    onset: int,
    *,
    env_kwargs: Dict[str, Any],
    steps_per_ep: int,
    collect_episodes: int,
    collect_seed_base: int = DEFAULT_COLLECT_SEED_BASE,
    warmup_progress_denom: Optional[int] = None,
    use_cache: bool = True,
    cache_dir: Optional[Path] = None,
    logger=print,
) -> Tuple[REEAgent, Dict[str, Dict[str, Any]], Dict[str, torch.Tensor], Dict[str, Any]]:
    """Mature (warmup_train) + freeze + collect the z_world frozen prefix.

    Returns (frozen_agent, fresh_head_inits, dataset, provenance).
      frozen_agent       : E1/E2/encoder frozen; warmed E2.world_forward available for
                           the E2-forward-R2 diagnostic. Its evaluator heads are
                           POST-maturation -- do NOT use them as the fresh init.
      fresh_head_inits   : {name: state_dict} of e3 eval heads (harm_eval_head,
                           benefit_eval_head) captured BEFORE maturation. Use these to
                           re-init the DV head to the fixed 740a/744 init.
      dataset            : collect_world_dataset superset (caller uses its subset).
      provenance         : {"cache": "hit"|"miss"|"disabled", "prefix_key": ...}.

    On a cache HIT the warmup+collect is skipped; a fresh agent is still rebuilt in the
    same RNG order (so fresh_head_inits are bit-identical) and the frozen params are
    loaded from the cache. On a MISS it runs warmup_train(onset) + collect and writes
    the cache. The downstream evaluator-head training re-seeds explicitly, so it is
    bit-identical on hit vs miss.
    """
    denom = warmup_progress_denom if warmup_progress_denom is not None else onset

    upstream = {
        "seed": int(seed),
        "onset": int(onset),
        "env_kwargs": env_kwargs,
        "steps_per_ep": int(steps_per_ep),
        "collect_episodes": int(collect_episodes),
        "collect_seed_base": int(collect_seed_base),
    }
    key = _prefix_key("world", upstream)

    # Build the fresh agent FIRST (consumes RNG identically on hit and miss) so the
    # fresh head inits are bit-identical to the inline path either way.
    agent, env = build_world_agent(seed, env_kwargs)
    fresh_heads = _capture_fresh_heads(agent, _WORLD_HEAD_ATTRS)

    blob = _cache_load(key, cache_dir, logger) if (use_cache and not _cache_disabled()) else None
    if blob is not None:
        agent.load_state_dict(blob["agent_state"])
        _freeze_world(agent)
        dataset = {k: v for k, v in blob["dataset"].items()}
        logger("prefix_cache: world HIT seed=%s onset=%s key=%s" % (seed, onset, key[:12]))
        return agent, fresh_heads, dataset, {"cache": "hit", "prefix_key": key,
                                             **_scope_provenance("world")}

    # MISS -> compute the prefix.
    warmup_train(
        agent, env,
        num_episodes=onset,
        steps_per_episode=steps_per_ep,
        label="mat seed=%s onset=%s" % (seed, onset),
        progress_total_episodes=denom,
    )
    _freeze_world(agent)
    dataset = collect_world_dataset(agent, seed, env_kwargs, steps_per_ep,
                                    collect_episodes, collect_seed_base)

    cache_state = "disabled" if _cache_disabled() else "miss"
    if use_cache and not _cache_disabled():
        _cache_store(
            key,
            {"key": key, "leg": "world", "upstream": upstream,
             "agent_state": agent.state_dict(), "dataset": dataset,
             **_scope_provenance("world")},
            cache_dir, logger,
        )
        logger("prefix_cache: world MISS-store seed=%s onset=%s key=%s" % (seed, onset, key[:12]))
    return agent, fresh_heads, dataset, {"cache": cache_state, "prefix_key": key,
                                         **_scope_provenance("world")}


# ---------------------------------------------------------------------------
# z_harm leg (743 and successors).
# ---------------------------------------------------------------------------

def build_harm_agent(seed: int, env_kwargs: Dict[str, Any], *, world_dim: int = 32,
                     z_harm_dim: int = 32, harm_obs_dim: int = 51) -> Tuple[REEAgent, CausalGridWorldV2]:
    """Bit-identical to 743 `_build_agent` (056c SD-010 wiring: world_dim = z_harm_dim
    so agent.e3.harm_eval_z_harm_head matches the standalone HarmEncoder width)."""
    env = CausalGridWorldV2(seed=seed, **env_kwargs)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=world_dim,
        alpha_world=0.9,
        alpha_self=0.3,
        use_harm_stream=True,
        harm_obs_dim=harm_obs_dim,
        z_harm_dim=z_harm_dim,
    )
    agent = REEAgent(cfg)
    return agent, env


def _mature_harm_encoder(harm_enc: HarmEncoder, seed: int, onset: int, steps_per_ep: int,
                         denom: int, env_kwargs: Dict[str, Any], *, z_harm_dim: int,
                         harm_obs_center: int, mature_lr: float, mature_hidden: int,
                         mature_seed_base: int) -> int:
    """Mature the HarmEncoder on the SD-010 label harm_obs[center] with a THROWAWAY
    proximity head (discarded). Bit-identical to 743 `_mature_harm_encoder`."""
    if onset <= 0:
        return 0
    mature_env = CausalGridWorldV2(seed=mature_seed_base + seed, **env_kwargs)
    act_rng = np.random.default_rng(mature_seed_base + seed)
    action_dim = mature_env.action_dim
    temp_head = nn.Sequential(
        nn.Linear(z_harm_dim, mature_hidden),
        nn.ReLU(),
        nn.Linear(mature_hidden, 1),
    )
    opt = torch.optim.Adam(list(harm_enc.parameters()) + list(temp_head.parameters()), lr=mature_lr)
    harm_enc.train()
    temp_head.train()
    n_steps = 0
    last_loss = float("nan")
    for ep in range(onset):
        _, obs_dict = mature_env.reset()
        for _step in range(steps_per_ep):
            harm_obs_t = _row_vec(obs_dict.get("harm_obs"))
            label = harm_obs_t[:, harm_obs_center:harm_obs_center + 1]
            z_harm = harm_enc(harm_obs_t)
            pred = temp_head(z_harm)
            loss = F.mse_loss(pred, label)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(harm_enc.parameters(), 0.5)
            opt.step()
            last_loss = float(loss.item())
            n_steps += 1

            action = int(act_rng.integers(0, action_dim))
            _, _harm_signal, done, _info, obs_dict = mature_env.step(action)
            if done:
                break
        if (ep + 1) % 5 == 0 or ep + 1 == onset:
            print("  [train] mat seed=%s onset=%s ep %s/%s loss=%.4f"
                  % (seed, onset, ep + 1, denom, last_loss), flush=True)
    return n_steps


def _hazard_state_targets(hazard_field: Any, ax: int, ay: int,
                          *, radius: int = 2) -> Tuple[float, float, float]:
    """Three STATE-DETERMINED raw-hazard_field targets at state s_t, read BEFORE the action
    so each aligns with the state z_harm encodes (matches 746's `_hazard_at_agent` timing).
    All clip the RAW field to [0, 1] (746 convention; the raw field, NOT the normalised
    harm_obs proxy view, so decoding them is a genuine differentiation test, not a
    label read-back). No RNG consumed (pure reads) -> the shared Zharm/Y/Prox collection
    stays bit-identical to 743's inline path.

      at_agent      : clip(hazard_field[ax, ay], 0, 1)  -- 746's single-cell target (SPARSE:
                      a random-walk agent rarely sits on a hot cell -> low raw variance, the
                      root of 746's starvation; kept as a secondary continuity leg).
      local_density : mean over the (2r+1)x(2r+1) neighbourhood EXCLUDING the centre cell,
                      in-bounds cells only -- DENSE neighbourhood harm exposure. High-variance
                      under a random walk because hazard_field is smooth/autocorrelated, so it
                      does not collapse to ~0 the way the single centre cell does. PRIMARY.
      next_step     : mean over the 4 orthogonally-adjacent in-bounds cells -- one-step-
                      reachable hazard exposure (a predictive variant). Secondary.
    """
    sx, sy = hazard_field.shape

    def _c(v: Any) -> float:
        return float(np.clip(v, 0.0, 1.0))

    at_agent = _c(hazard_field[ax, ay])
    dens = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = ax + dx, ay + dy
            if 0 <= nx < sx and 0 <= ny < sy:
                dens.append(_c(hazard_field[nx, ny]))
    local_density = float(np.mean(dens)) if dens else 0.0
    nxt = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = ax + dx, ay + dy
        if 0 <= nx < sx and 0 <= ny < sy:
            nxt.append(_c(hazard_field[nx, ny]))
    next_step = float(np.mean(nxt)) if nxt else 0.0
    return at_agent, local_density, next_step


def collect_harm_dataset(harm_enc: HarmEncoder, seed: int, env_kwargs: Dict[str, Any],
                         steps_per_ep: int, n_episodes: int, *, harm_obs_center: int,
                         collect_seed_base: int = DEFAULT_COLLECT_SEED_BASE,
                         hazard_neighbourhood_radius: int = 2) -> Dict[str, torch.Tensor]:
    """Replay a FIXED action sequence through a FIXED collection env, encoding z_harm
    with the (frozen) HarmEncoder. The Zharm / Y / Prox fields are bit-identical to 743
    `_collect_frozen_dataset`:
      Zharm = HarmEncoder(harm_obs[t]) ; Y = realized harm ; Prox = harm_obs[t][center].

    ADDITIONALLY collects three STATE-DETERMINED raw-hazard_field targets per step (added
    2026-07-12 for the harm leg's first wild consumer, V3-EXQ-746a -- 746 re-inlined a
    single such target inline and starved). These are pure no-RNG reads of
    collect_env.hazard_field at s_t, so appending them does NOT perturb the shared
    Zharm/Y/Prox RNG stream:
      Yat    = at-agent single cell (746's original; sparse)
      Ydens  = local neighbourhood density EXCLUDING centre (dense; PRIMARY for 746a)
      Ynext  = one-step-reachable mean (predictive)
    All raw-clipped to [0, 1]; the consumer applies its own variance precondition + probes.
    """
    collect_env = CausalGridWorldV2(seed=collect_seed_base + seed, **env_kwargs)
    act_rng = np.random.default_rng(collect_seed_base + seed)
    action_dim = collect_env.action_dim

    zh_list = []
    y_list = []
    prox_list = []
    yat_list = []
    ydens_list = []
    ynext_list = []

    harm_enc.eval()
    with torch.no_grad():
        for _ep in range(n_episodes):
            _, obs_dict = collect_env.reset()
            for _step in range(steps_per_ep):
                harm_obs_t = _row_vec(obs_dict.get("harm_obs"))
                z_harm = harm_enc(harm_obs_t).detach().cpu()
                prox = float(harm_obs_t[0, harm_obs_center].item())
                # STATE-DETERMINED raw-field targets aligned to s_t (read BEFORE the action).
                yat, ydens, ynext = _hazard_state_targets(
                    collect_env.hazard_field, int(collect_env.agent_x), int(collect_env.agent_y),
                    radius=hazard_neighbourhood_radius)

                action = int(act_rng.integers(0, action_dim))
                _, harm_signal, done, _info, obs_next = collect_env.step(action)
                harm_target = abs(float(harm_signal)) if float(harm_signal) < 0 else 0.0

                zh_list.append(z_harm)
                y_list.append(harm_target)
                prox_list.append(prox)
                yat_list.append(yat)
                ydens_list.append(ydens)
                ynext_list.append(ynext)

                obs_dict = obs_next
                if done:
                    break

    def _col(v):
        return torch.tensor(v, dtype=torch.float32).unsqueeze(1)

    return {
        "Zharm": torch.cat(zh_list, dim=0),
        "Y": _col(y_list),
        "Prox": _col(prox_list),
        "Yat": _col(yat_list),
        "Ydens": _col(ydens_list),
        "Ynext": _col(ynext_list),
    }


def mature_and_collect_harm(
    seed: int,
    onset: int,
    *,
    env_kwargs: Dict[str, Any],
    steps_per_ep: int,
    collect_episodes: int,
    world_dim: int = 32,
    z_harm_dim: int = 32,
    harm_obs_dim: int = 51,
    harm_obs_center: int = 12,
    mature_lr: float = 1e-3,
    mature_hidden: int = 32,
    mature_seed_base: int = DEFAULT_HARM_MATURE_SEED_BASE,
    collect_seed_base: int = DEFAULT_COLLECT_SEED_BASE,
    hazard_neighbourhood_radius: int = 2,
    mature_progress_denom: Optional[int] = None,
    use_cache: bool = True,
    cache_dir: Optional[Path] = None,
    logger=print,
) -> Tuple[REEAgent, HarmEncoder, Dict[str, Dict[str, Any]], Dict[str, torch.Tensor],
           int, Dict[str, Any]]:
    """Mature (standalone HarmEncoder) + freeze + collect the z_harm frozen prefix.

    Returns (agent, frozen_harm_encoder, fresh_head_inits, dataset, n_mature_steps,
    provenance). fresh_head_inits carries harm_eval_z_harm_head captured pre-maturation.
    Cache semantics mirror mature_and_collect_world.
    """
    denom = mature_progress_denom if mature_progress_denom is not None else onset

    upstream = {
        "seed": int(seed),
        "onset": int(onset),
        "env_kwargs": env_kwargs,
        "steps_per_ep": int(steps_per_ep),
        "collect_episodes": int(collect_episodes),
        "world_dim": int(world_dim),
        "z_harm_dim": int(z_harm_dim),
        "harm_obs_dim": int(harm_obs_dim),
        "harm_obs_center": int(harm_obs_center),
        "mature_lr": float(mature_lr),
        "mature_hidden": int(mature_hidden),
        "mature_seed_base": int(mature_seed_base),
        "collect_seed_base": int(collect_seed_base),
        "hazard_neighbourhood_radius": int(hazard_neighbourhood_radius),
    }
    key = _prefix_key("harm", upstream)

    # Build agent then HarmEncoder in 743's order (consumes RNG identically on hit/miss).
    agent, _env = build_harm_agent(seed, env_kwargs, world_dim=world_dim,
                                   z_harm_dim=z_harm_dim, harm_obs_dim=harm_obs_dim)
    harm_enc = HarmEncoder(harm_obs_dim=harm_obs_dim, z_harm_dim=z_harm_dim)
    fresh_heads = _capture_fresh_heads(agent, _HARM_HEAD_ATTRS)

    blob = _cache_load(key, cache_dir, logger) if (use_cache and not _cache_disabled()) else None
    if blob is not None:
        harm_enc.load_state_dict(blob["harm_encoder_state"])
        for p in harm_enc.parameters():
            p.requires_grad_(False)
        dataset = {k: v for k, v in blob["dataset"].items()}
        logger("prefix_cache: harm HIT seed=%s onset=%s key=%s" % (seed, onset, key[:12]))
        return agent, harm_enc, fresh_heads, dataset, int(blob.get("n_mature_steps", 0)), \
            {"cache": "hit", "prefix_key": key, **_scope_provenance("harm")}

    n_mature_steps = _mature_harm_encoder(
        harm_enc, seed, onset, steps_per_ep, denom, env_kwargs,
        z_harm_dim=z_harm_dim, harm_obs_center=harm_obs_center,
        mature_lr=mature_lr, mature_hidden=mature_hidden, mature_seed_base=mature_seed_base)
    for p in harm_enc.parameters():
        p.requires_grad_(False)
    dataset = collect_harm_dataset(harm_enc, seed, env_kwargs, steps_per_ep, collect_episodes,
                                   harm_obs_center=harm_obs_center, collect_seed_base=collect_seed_base,
                                   hazard_neighbourhood_radius=hazard_neighbourhood_radius)

    cache_state = "disabled" if _cache_disabled() else "miss"
    if use_cache and not _cache_disabled():
        _cache_store(
            key,
            {"key": key, "leg": "harm", "upstream": upstream,
             "harm_encoder_state": harm_enc.state_dict(), "dataset": dataset,
             "n_mature_steps": int(n_mature_steps), **_scope_provenance("harm")},
            cache_dir, logger,
        )
        logger("prefix_cache: harm MISS-store seed=%s onset=%s key=%s" % (seed, onset, key[:12]))
    return agent, harm_enc, fresh_heads, dataset, n_mature_steps, \
        {"cache": cache_state, "prefix_key": key, **_scope_provenance("harm")}


__all__ = [
    "PREFIX_CACHE_SCHEMA",
    "DEFAULT_COLLECT_SEED_BASE",
    "DEFAULT_HARM_MATURE_SEED_BASE",
    "build_world_agent",
    "collect_world_dataset",
    "mature_and_collect_world",
    "build_harm_agent",
    "collect_harm_dataset",
    "mature_and_collect_harm",
    # Dependency-scoped substrate hashing (plan sec 11): declared scope + guards.
    "verify_scope_conservatism",
]
