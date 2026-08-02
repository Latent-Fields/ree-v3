"""Canonical OFF/baseline arm for the MECH-332 efference-copy-vs-AIC lineage.

Lineage: v3_exq_878_mech332_efference_aic_dissociation (first of lineage).

WHY THIS MODULE EXISTS
----------------------
Per `/queue-experiment` "Saving a baseline for reuse", the FIRST experiment of a
lineage mints its own reusable baseline in-line rather than via a separate mint job.
Two things make that work, and both live here:

  1. the OFF arm (ARM_NEITHER: both MECH-332 pathways disabled) is CONSTRUCTED from
     this module, so a later sibling with a different driver builds a bit-identical
     cell by construction, not by coincidence; and
  2. the caller emits that cell's fingerprint with `include_driver_script_in_hash=False`
     (done in the driver), so a future consumer's distinct driver can still match.

`_lib/**` is inside the substrate glob, so any edit here correctly BUSTS a stale
reuse rather than silently serving one.

WHAT THE BASELINE IS
---------------------
ARM_NEITHER of the MECH-332 double dissociation: use_e2_harm_s_forward=False,
use_aic_analog=False, harm_descending_mod_enabled=False. Neither of MECH-332's two
candidate pathways (ARC-033 efference-copy comparator / SD-032c-subsumed descending
modulation) is active; z_harm_s reaches the agent unattenuated and no counterfactual
comparator is constructed. Only the arm-specific REEConfig flags below vary across
the lineage's other arms (ARM_E2_ONLY, ARM_AIC_ONLY, ARM_BOTH) -- the arena, the
SD-029 balanced-event curriculum, SD-022 body-damage substrate, and the schedule
are IDENTICAL across all four arms of this lineage, and live here.
"""

from typing import Any, Dict

# ------------------------------------------------------------------ #
# Schedule (part of the OFF-path config slice -- it changes the cell) #
# ------------------------------------------------------------------ #
STEPS_PER_EP = 150
P0_EPS = 80          # HarmEncoder + standard agent-loss warmup (no E2_harm_s training)
P1_EPS = 100         # E2_harm_s phased training (frozen-target MSE + SD-013 interventional)
EVAL_MAX_EPS = 50    # measurement phase, early-exit once trial targets are met
TARGET_TRIALS_PER_TYPE = 12   # agent_caused_hazard / env_caused_hazard trials, per seed

# ------------------------------------------------------------------ #
# Arena -- SD-022 body-damage substrate (z_harm_s / z_harm_a causal      #
# independence) + SD-029 balanced-event curriculum, EXQ-479 calibrated  #
# params (interval=10, prob=1.0, adjacent_only=True, hazard_harm=0.02)  #
# ------------------------------------------------------------------ #
ARENA: Dict[str, Any] = {
    "size": 8,
    "num_hazards": 2,
    "num_resources": 3,
    "hazard_harm": 0.02,
    "proximity_harm_scale": 0.1,
    "use_proxy_fields": True,
    "limb_damage_enabled": True,
    "harm_history_len": 10,
    "scheduled_external_hazard_enabled": True,
    "scheduled_external_hazard_interval": 10,
    "scheduled_external_hazard_prob": 1.0,
    "scheduled_external_hazard_adjacent_only": True,
}

# ------------------------------------------------------------------ #
# REEConfig kwargs shared by every arm (agent_kwargs merges arm-specific #
# pathway flags on top of this dict)                                    #
# ------------------------------------------------------------------ #
SELF_DIM = 32
WORLD_DIM = 32
Z_HARM_DIM = 32
Z_HARM_A_DIM = 16
LR = 1e-3
HARM_FWD_LR = 5e-4
INTERVENTIONAL_FRACTION = 0.3
INTERVENTIONAL_MARGIN = 0.15

COMMON_AGENT_KWARGS: Dict[str, Any] = {
    "self_dim": SELF_DIM,
    "world_dim": WORLD_DIM,
    "alpha_world": 0.9,
    "alpha_self": 0.3,
    "use_harm_stream": True,
    "z_harm_dim": Z_HARM_DIM,
    "use_affective_harm_stream": True,
    "z_harm_a_dim": Z_HARM_A_DIM,
    "limb_damage_enabled": True,
    "harm_history_len": 10,
    "descending_attenuation_factor": 0.5,
    "aic_baseline_alpha": 0.02,
    "aic_drive_coupling": 1.0,
    "aic_base_attenuation": 0.5,
    "aic_drive_protect_weight": 1.0,
}

# The OFF path: neither MECH-332 pathway active.
OFF_PATHWAY_FLAGS: Dict[str, Any] = {
    "use_e2_harm_s_forward": False,
    "use_aic_analog": False,
    "harm_descending_mod_enabled": False,
}


def env_kwargs(seed: int) -> Dict[str, Any]:
    """Env constructor kwargs at a seed. Identical across all four arms."""
    kw = dict(ARENA)
    kw["seed"] = int(seed)
    return kw


def agent_kwargs(pathway_flags: Dict[str, Any]) -> Dict[str, Any]:
    """REEConfig.from_dims kwargs (minus body_obs_dim/world_obs_dim/action_dim,
    which are read from the constructed env) for a given pathway-flag combination."""
    kw = dict(COMMON_AGENT_KWARGS)
    kw.update(pathway_flags)
    return kw


def off_path_config_slice(seed: int = 0) -> Dict[str, Any]:
    """The declared config slice for the ARM_NEITHER (fully-off) cell -- everything
    its build+collect path reads, and nothing else (never ON-arm gains or
    acceptance thresholds). `seed` only affects the seed field, not the shape."""
    return {
        "env_kwargs": env_kwargs(seed),
        "agent_kwargs": agent_kwargs(OFF_PATHWAY_FLAGS),
        "schedule": {
            "steps_per_ep": STEPS_PER_EP,
            "p0_eps": P0_EPS,
            "p1_eps": P1_EPS,
            "eval_max_eps": EVAL_MAX_EPS,
            "target_trials_per_type": TARGET_TRIALS_PER_TYPE,
        },
        "lr": LR,
        "harm_fwd_lr": HARM_FWD_LR,
        "interventional_fraction": INTERVENTIONAL_FRACTION,
        "interventional_margin": INTERVENTIONAL_MARGIN,
    }
