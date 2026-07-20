"""Canonical reusable-arm baseline for the MECH-457 actor-critic ON/OFF lineage.

Arm-reuse (arm_reuse_fingerprint_plan.md sections 7b, 9). This module is the SINGLE
SOURCE OF TRUTH for the OFF/baseline arm of the MECH-457 validation lineage
(V3-EXQ-742 -> future letters/successors): the ``bias_head_baseline`` arm, i.e. the
724-A0 all-ON REE stack (world-model warmup P0 + two-head REINFORCE P1, SD-056 e2
encoder frozen in P1) evaluated by capability_eval.REEForwardPolicy. That arm is the
actor-critic-OFF incompetence control -- it does NOT depend on any MECH-457 knob, so it
is IDENTICAL across every actor-critic ablation and every lettered iteration of the
lineage, which is exactly what makes it a durable reusable baseline.

WHY A SHARED MODULE. Both the V3-EXQ-742 consumer AND the V3-EXQ-742-m baseline mint
call ``run_off_cell`` (the computation) and ``off_path_config_slice`` (the fingerprint
slice) from THIS module, so their arm fingerprints match BY CONSTRUCTION and a mint
minted before/independently of the consumer can be reused later (order-independent
insurance). A false cache-miss is free; a false hit corrupts science -- this module is
matched by the arm-fingerprint substrate glob ``experiments/_lib/**/*.py``, so any edit
here correctly flips the substrate hash and refuses a stale reuse.

REUSE MACHINE-CLASS. A ree-cloud worker class -- since 2026-07-19 the tag includes the
TORCH BUILD, currently `linux-x86_64-py3.10-torch2.5.1+cu121`. The authority is
`machine_class()` in experiments/_lib/arm_fingerprint.py; treat any tag written here as
indicative only. The fingerprint carries machine_class, so a Mac-run consumer cannot match
a cloud mint and just re-runs -- reuse is intrinsically cloud-scoped. A fleet torch upgrade
now retires a banked baseline the same way an OS or python change always did, and any
baseline minted BEFORE the 2026-07-19 hard cut is DEAD: the pre-cut bank (1212 fingerprints)
cannot be migrated and must be re-minted under the new class (plan section 12).

MINT RUN_ID. Recorded by the parent session once V3-EXQ-742-m completes, then cited via
``reuse_baseline_from`` in a future consumer. Until then reuse_baseline_from=None and the
arm runs fresh.

The OFF computation itself is the 724-A0 recipe reused verbatim from
experiments/v3_exq_734_env_difficulty_competence_recovery_sweep.py (x734) -- this module
does NOT re-implement it, it delegates, so consumer and mint cannot drift from the
canonical warmup/train. ``off_path_config_slice`` is stdlib-only (importable without
ree_core); ``run_off_cell`` lazy-imports x734/capability_eval on call.

ASCII-only output (repo rule).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

LINEAGE = "mech457_actor_critic_onoff_bias_head_baseline"
# The reusable OFF arm of the lineage (the actor-critic-OFF 724-A0 all-ON control).
REUSABLE_ARM_ID = "bias_head_baseline"
# The eval metrics a consumer may read off a reused OFF cell (capability_eval row keys).
OFF_CELL_NEEDED_KEYS = (
    "foraging_competence",
    "competence_supra_floor",
    "survival_horizon",
    "goal_reach_rate",
    "planning_depth",
)


def off_path_config_slice(
    env_kwargs: Dict[str, Any],
    p0_warmup_episodes: int,
    p1_reinforce_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    zworld_p0_episodes: int = 0,
) -> Dict[str, Any]:
    """The declared fingerprint slice for the OFF (bias_head_baseline) arm.

    BOTH the V3-EXQ-742 consumer and the V3-EXQ-742-m mint pass this exact dict to the
    arm-fingerprint, so their fingerprints match by construction. It declares everything
    the OFF computation reads and NOTHING actor-critic-specific (no cotrain / sf / AC
    hyper-parameters -- those are treatment-arm properties, never this control arm). The
    caller passes env_kwargs derived from x734._env_kwargs_for_rung so env config cannot
    drift between the two sides.
    """
    return {
        "lineage": LINEAGE,
        "arm_id": REUSABLE_ARM_ID,
        "recipe": "724_A0_all_on_bias_head",  # world-model warmup + two-head REINFORCE
        "e2_train_in_p1": False,              # A0 recipe: SD-056 e2 frozen through P1
        "eval_policy": "REEForwardPolicy",
        "env_kwargs": dict(env_kwargs),
        # SD-070 P0a. LOAD-BEARING for reuse correctness: an OFF arm warmed with the encoder
        # recipe is a DIFFERENT arm from one whose z_world stayed a frozen random projection.
        # Omitting it would let a pre-SD-070 banked arm falsely cache-HIT a post-SD-070
        # consumer, silently comparing a trained-encoder treatment against an untrained
        # control. 0 (the default) reproduces every arm banked before this landed.
        "zworld_p0_episodes": int(zworld_p0_episodes),
        "p0_warmup_episodes": int(p0_warmup_episodes),
        "p1_reinforce_episodes": int(p1_reinforce_episodes),
        "eval_episodes": int(eval_episodes),
        "steps_per_episode": int(steps_per_episode),
    }


def run_off_cell(
    env_kwargs: Dict[str, Any],
    seed: int,
    p0_warmup_episodes: int,
    p1_reinforce_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    rung_id: str = "cell",
    zworld_p0_episodes: int = 0,
    zworld_p0_dry_run: bool = False,
) -> Dict[str, Any]:
    """Train + evaluate ONE (rung, seed) OFF cell and return the capability_eval row.

    Delegates to x734._make_all_on_agent / _train_all_on_agent (P0a SD-070 encoder warmup when
    enabled, then P0b e2 warmup + P1 two-head REINFORCE, e2 frozen in P1) and
    capability_eval.REEForwardPolicy eval -- byte-identical to the V3-EXQ-742
    bias_head_baseline arm because BOTH call this one function. The caller is responsible for
    RNG reset (arm_cell / reset_all_rng) before calling.

    `zworld_p0_episodes` MUST MATCH THE CONSUMING DRIVER'S SETTING. It is a member of
    `off_path_config_slice`, so a mismatch changes the fingerprint and correctly forces a
    cache MISS rather than silently comparing a trained-encoder arm against a frozen-random-
    projection baseline -- which would confound every ON/OFF contrast built on this baseline.
    Default 0 preserves the pre-SD-070 baseline exactly.
    """
    # Lazy imports so `import off_path_config_slice` stays stdlib-only.
    import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734
    from experiments._lib.capability_eval import REEForwardPolicy, evaluate_seed

    train_env = x734._make_env(seed, env_kwargs)
    agent = x734._make_all_on_agent(train_env)
    x734._train_all_on_agent(
        agent, train_env, seed=seed,
        p0_episodes=int(p0_warmup_episodes), p1_episodes=int(p1_reinforce_episodes),
        steps_per_episode=int(steps_per_episode), rung_id=rung_id,
        total_denominator=int(p0_warmup_episodes) + int(p1_reinforce_episodes),
        zworld_p0_episodes=int(zworld_p0_episodes),
        # Dedicated env instance: the P0a rollout consumes env RNG, so reusing train_env would
        # shift the layout sequence P0b/P1 then see.
        zworld_p0_env=(
            x734._make_env(seed, env_kwargs) if int(zworld_p0_episodes) > 0 else None
        ),
        zworld_p0_dry_run=bool(zworld_p0_dry_run),
    )
    eval_env = x734._make_env(seed, env_kwargs)
    return evaluate_seed(
        REEForwardPolicy(agent, name="bias_head_baseline"),
        eval_env, int(eval_episodes), int(steps_per_episode),
    )


__all__ = [
    "LINEAGE",
    "REUSABLE_ARM_ID",
    "OFF_CELL_NEEDED_KEYS",
    "off_path_config_slice",
    "run_off_cell",
]
