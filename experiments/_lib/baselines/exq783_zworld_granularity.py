"""Canonical OFF-arm baseline for the z_world granularity/training crossing lineage.

Lineage: exq783_zworld_granularity (first experiment: V3-EXQ-783).

The OFF / baseline arm of this lineage is D32_UNTRAINED -- world_dim=32 with the world
encoder left at initialisation. That cell is the CONTROL that reproduces, inside this
harness, the exact configuration measured by the 2026-07-18 offline characterisation
(`REE_assembly/evidence/experiments/zworld_near_static_characterisation_2026-07-18.md`)
and by convergence probe DREAMER-V3-P-008: world_dim=32 AND an untrained world encoder.

WHY THIS MODULE EXISTS. Per the /queue-experiment "Saving a baseline for reuse" rule, the
FIRST experiment of a lineage mints its own reusable baseline in-line: it factors the OFF
arm here, and emits that cell's fingerprint with include_driver_script_in_hash=False. A
later sibling with a DIFFERENT driver script then matches this fingerprint by construction
and can cite `reuse_baseline_from: <V3-EXQ-783 run_id>` instead of retraining the cell.
Being under experiments/_lib/** auto-binds this module into substrate_hash, so any edit
here correctly refuses a stale reuse.

CONFIG-SLICE DISCIPLINE. off_path_config_slice() declares ONLY what the OFF computation
reads: env kwargs + the measurement schedule + the substrate-operating agent config + the
OFF arm's own axis settings (world_dim=32, encoder_training=False). It must NEVER carry ON-arm
settings, acceptance thresholds, or criterion labels -- an under- or over-declared slice is
the false-hit / false-miss trap of arm_reuse_fingerprint_plan.md section 7b.
"""

from typing import Any, Dict

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

import experiments.v3_exq_724_competence_localization_diagnostic as x724

# The OFF arm's two axis coordinates. Named here (not in the driver) so every sibling
# that imports this module gets bit-identical values.
OFF_WORLD_DIM: int = 32
OFF_ENCODER_TRAINING: bool = False

# Schedule the OFF cell runs. Part of the config slice: changing it changes the computation,
# so it must bust the fingerprint. THIS MODULE IS THE SINGLE SOURCE OF TRUTH -- the driver
# imports these rather than declaring its own, because a slice that disagrees with what the
# cell actually ran is exactly the false-hit trap of arm_reuse_fingerprint_plan.md sec 7b.
#
# NOTE OFF_P0_ENCODER_EPISODES is 60, NOT 0. The untrained arm still runs the full P0
# rollout loop and computes the same auxiliary losses -- it simply never steps the encoder
# optimiser. Env exposure is therefore matched across all four arms, so the crossing varies
# weight updates ONLY, never the state distribution.
OFF_P0_ENCODER_EPISODES: int = 60
OFF_P1_E2_EPISODES: int = 60
# 40 measurement episodes matches the 2026-07-18 characterisation grid exactly (40 eval
# episodes per cell), keeping the manifold statistics directly comparable to its table.
OFF_MEASURE_EPISODES: int = 40
OFF_STEPS_PER_EPISODE: int = 150


def env_kwargs() -> Dict[str, Any]:
    """The D0_baseline_724 rung -- the RICHEST rung of the x734 ladder.

    x734's DIFFICULTY_RUNGS[0] is {"rung_id": "D0_baseline_724", "overrides": {}}, i.e.
    literally x724.ENV_KWARGS. The 2026-07-18 grid showed z_world contrast is FLAT across
    the full four-rung ladder (and highest at the most impoverished rung), so rung choice
    is not load-bearing for the contrast statistic -- but the richest rung is required for
    the event-selectivity and residue legs to have events to measure at all.
    """
    return dict(x724.ENV_KWARGS)


def make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **env_kwargs())


def agent_config_kwargs(env: CausalGridWorldV2, world_dim: int,
                        encoder_training: bool) -> Dict[str, Any]:
    """Agent config for one cell of the crossing.

    Built from the x724 all-ON config builders so the substrate superstructure is IDENTICAL
    to the configuration the 2026-07-18 diagnostic measured. Exactly ONE thing varies in the
    AGENT CONFIG across the 2x2 crossing:

      world_dim -- 32 vs 128   (the (a2) granularity axis)

    The (a1) encoder-training axis is NOT a config difference at all: it is whether the
    SD-070 P0 trainer's optimiser is stepped. Both training arms are therefore built from a
    BIT-IDENTICAL agent config, which is strictly cleaner than the previous design where the
    trained arms additionally carried use_event_classifier=True.

    use_event_classifier is now False on EVERY arm. SD-070 does not use the SD-009 event
    classifier head -- that head's CE target was measured to be undecodable from the world
    channel it reads (see the SD-070 doc), so the recipe replaces it with scene-structure
    grounding targets owned by the trainer. Leaving the head enabled would attach an
    untrained Linear(world_dim, 3) that nothing reads. This also keeps the OFF cell
    bit-identical to the measured 2026-07-18 configuration, which had it False.

    DELIBERATELY NOT REEConfig.large(). The `large` preset would move world_dim to 128 but
    ALSO self_dim 32->128, action_object_dim 16->64, e1/e2/hippocampal hidden_dim ->256,
    e3 hidden_dim ->128, residue num_basis_functions ->64, and alpha_world ->0.9. Using it
    would confound the dim axis with five other changes -- including alpha_world, which is
    SD-008 and precisely the mistagging the 2026-06-06 cluster autopsy called out. So the
    dim axis is moved by hand via from_dims(world_dim=...) with everything else held fixed.

    Note alpha_world is already 0.9 in the x724 base kwargs, which is what z_world-fidelity
    work requires (SD-008); it is held at 0.9 in BOTH dim arms.

    SD-018 (use_resource_proximity_head) is already True in the x724 base kwargs, but its
    loss is never computed in the x724/x734 P0 loop -- the head exists and is never trained.
    The SD-070 recipe RETAINS and actually trains that head: it is the one leg of the
    previously-prescribed P0 that does learn (held-out R^2 0.794 from raw obs).

    `encoder_training` is accepted for call-site symmetry and is deliberately NOT read: it
    selects whether the trainer's optimiser runs, not what the agent is built as.
    """
    kwargs = x724._base_config_kwargs(env)
    kwargs.update(x724._all_on_extra_kwargs())
    kwargs["world_dim"] = int(world_dim)
    kwargs["use_event_classifier"] = False
    return kwargs


def build_off_arm(seed: int) -> Dict[str, Any]:
    """Construct the OFF cell's env + agent (world_dim=32, encoder untrained)."""
    env = make_env(seed)
    cfg = REEConfig.from_dims(
        **agent_config_kwargs(env, OFF_WORLD_DIM, OFF_ENCODER_TRAINING)
    )
    return {"env": env, "agent": REEAgent(cfg)}


def off_path_config_slice() -> Dict[str, Any]:
    """The fingerprint-bearing declaration of everything the OFF computation reads.

    Consumed by compute_arm_fingerprint(config_slice=...) on the mint side and by
    try_reuse_cell(config_slice=...) on the consumer side. Both sides MUST also pass
    include_driver_script_in_hash=False or the drivers will never match.
    """
    return {
        "lineage": "exq783_zworld_granularity",
        "env_kwargs": env_kwargs(),
        "world_dim": OFF_WORLD_DIM,
        "encoder_training": OFF_ENCODER_TRAINING,
        "use_event_classifier": False,
        # The P0 recipe is part of the computation even on the OFF cell, because it
        # determines what the rollout buffers and therefore what a TRAINED sibling would
        # have done with the same exposure. Declaring it prevents a future consumer running
        # a different P0 from false-hitting this fingerprint.
        "p0_recipe": "sd070",
        "alpha_world": 0.9,
        "schedule": {
            "p0_encoder_episodes": OFF_P0_ENCODER_EPISODES,
            "p1_e2_episodes": OFF_P1_E2_EPISODES,
            "measure_episodes": OFF_MEASURE_EPISODES,
            "steps_per_episode": OFF_STEPS_PER_EPISODE,
        },
    }
