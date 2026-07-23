"""
Q-081 cross-stream recording config profile -- turns on the signals that are dark by default.

Why this exists
---------------
The Q-081 telemetry audit
(REE_assembly/evidence/planning/q081_cross_stream_telemetry_audit.md, section 4 item 3)
found that 4 of the 14 Q-081 streams DO NOT EXIST under stock defaults, and a fifth
(MECH-287 broadcasts) needs its own flag. A recorder run without this profile writes
nulls for roughly a quarter of the checklist and looks superficially fine while doing it
-- the "silent-null trap".

Every flag below was verified `= False` at its definition site in
ree_core/utils/config.py, and each is a real `REEConfig.from_dims` keyword with a
matching assignment site (checked; a typo'd kwarg would be swallowed silently -- see
MEMORY reference_reeconfig_from_dims_silent_kwargs).

    latent.use_harm_stream              config.py:86    -> z_harm_s   (signal 7)
    latent.use_affective_harm_stream    config.py:97    -> z_harm_a   (signal 8)
    use_salience_coordinator            config.py:2456  -> operating_mode (signal 10)
    hippocampal.use_event_segmenter     config.py:1827  -> boundary events (signal 12)
    hippocampal.use_invalidation_trigger config.py:1836 -> MECH-287 broadcasts
    use_tpj_comparator                  config.py:2713  -> E2-self error (signal 2)
    use_sleep_loop                      config.py:4236  -> sleep-phase markers
    z_goal_enabled                      goal.py:87      -> z_goal (signal 9)

E2 PE (audit section 2.1) -- decision recorded here, with a measured caveat
---------------------------------------------------------------------------
There is NO per-step E2-self prediction error on the default path. The decision taken
2026-07-22 is to enable `use_tpj_comparator`, which resolves a genuine E2
`predict_next_self` error into `agent._tpj_last_agency_signal`
(ree_core/agent.py:3047-3062), and to record it as the E2 stream.

The rejected alternative was substituting the E3 world-rollout error
(e3_selector.py:3286). That error is E3-rate and harm-event-sparse, so substituting it
would have inserted a second copy of the E3 stream into the trace at the E3 rate --
manufacturing exactly the shared periodic structure that Outcome B of the Q-081 taxonomy
exists to exclude. The recorder therefore never labels any E3-derived quantity "E2 PE".

CAVEAT, measured 2026-07-22 and NOT in the audit: the TPJ signal is E3-CADENCE, not
per-step. `_cache_tpj_prediction_for_action` is called at the END of select_action
(agent.py:7582), i.e. AFTER the between-tick short-circuit return at agent.py:5463, so no
efference copy is staged on a held step and the comparator resolves only on E3 ticks
(measured: 8 valid samples in 60 steps). What this buys over the rejected substitution is
therefore CONTENT, not cadence: a genuinely distinct E2-self quantity rather than a second
copy of E3's own error. It does NOT give Q-081 a true middle-rate stream. An analysis
treating it as one would be wrong; the recorder's freshness flag makes the real cadence
visible, and a contract pins it so this cannot drift back into an unqualified claim.

Sleep-phase markers (audit section 4 item 5) -- one audit correction
--------------------------------------------------------------------
The audit states SleepLoopManager "is not instantiated by REEAgent". That holds under
DEFAULTS only: REEAgent does construct it when `use_sleep_loop=True`
(ree_core/agent.py:2173, :2300), exposing `agent.sleep_loop.state.phase`. So the profile
turns the flag on rather than the harness owning a manager. `update_z_goal()` genuinely
IS loop-driven and stays the harness's responsibility -- z_goal is flat if the experiment
loop never calls it.

THE FLAGS ARE NOT SUFFICIENT -- three streams also need the LOOP to drive them
------------------------------------------------------------------------------
Turning a flag on is necessary but not sufficient for three streams. Each fails SILENTLY
as a well-formed null, which is the failure mode this whole profile exists to prevent, so
they are listed in `LOOP_DRIVEN_REQUIREMENTS` and asserted by a contract test.

  z_harm / z_harm_a  `agent.act()` and `act_with_split_obs()` call
                     `sense(obs_body, obs_world)` with NO harm channels, and
                     LatentStack.encode gates the harm encoders on
                     `harm_obs is not None` (latent/stack.py:1464, :1476). So on the
                     convenience act() path both stay None however the flags are set.
                     The loop must call `sense(..., obs_harm=, obs_harm_a=,
                     obs_harm_history=)` itself -- see `sense_kwargs_from_obs()` below,
                     and experiments/_harness.py:185-191 for the canonical shape.
                     (This is an addition to the audit, found 2026-07-22 by contract C5.)
  z_goal             needs BOTH `z_goal_enabled=True` (otherwise `agent.goal_state` is
                     None outright and the stream is absent, not merely zero) AND the
                     loop to call `agent.update_z_goal(...)` -- z_goal stays zero
                     otherwise (audit section 4 item 5).

NON-DEFAULT SUBSTRATE
---------------------
Any run using this profile describes a configuration REE does not normally run in, and
must say so in its manifest. `q081_substrate_declaration()` returns the block to merge;
`STREAMS_GATED_BY` maps each stream to the flag it needs, so a manifest can state which
streams would have been null without it.

ASCII-only output (repo rule). Stdlib only; no torch import at module scope.
"""

from __future__ import annotations

from typing import Any, Dict

Q081_PROFILE_ID = "q081_cross_stream_recording/v1"

# flag name (as a REEConfig.from_dims kwarg) -> (stock default, config.py line, why)
Q081_FLAGS: Dict[str, Any] = {
    "use_harm_stream": (False, 86, "z_harm_s sensory-discriminative harm latent (signal 7)"),
    "use_affective_harm_stream": (False, 97, "z_harm_a affective harm latent (signal 8)"),
    "use_salience_coordinator": (False, 2456, "operating_mode soft vector (signal 10)"),
    "use_event_segmenter": (False, 1827, "MECH-288 boundary events (signal 12)"),
    "use_invalidation_trigger": (False, 1836, "MECH-287 broadcast events"),
    "use_tpj_comparator": (False, 2713, "E2-self prediction error at E3 cadence (signal 2)"),
    "use_sleep_loop": (False, 4236, "sleep-phase markers via agent.sleep_loop.state.phase"),
    "z_goal_enabled": (False, 87, "z_goal persistent goal latent (signal 9); goal.py:87"),
}

# Which recorder stream each flag unlocks. Consumed by the manifest declaration so an
# analyst can see what would have been null under stock defaults.
STREAMS_GATED_BY: Dict[str, str] = {
    "z_harm": "use_harm_stream",
    "z_harm_a": "use_affective_harm_stream",
    "operating_mode": "use_salience_coordinator",
    "boundary_events": "use_event_segmenter",
    "broadcast_events": "use_invalidation_trigger",
    "e2_self_pe": "use_tpj_comparator",
    "sleep_phase": "use_sleep_loop",
    "z_goal": "z_goal_enabled",
}


# Streams whose flag is necessary but NOT sufficient: the experiment loop must also do
# something. Each of these fails silently as a well-formed null.
LOOP_DRIVEN_REQUIREMENTS: Dict[str, str] = {
    "z_harm": "loop must pass obs_harm into agent.sense(); act() does not",
    "z_harm_a": "loop must pass obs_harm_a into agent.sense(); act() does not",
    "z_goal": "loop must call agent.update_z_goal(); z_goal stays zero otherwise",
}

# Keys the environment's obs_dict uses for the harm channels.
HARM_OBS_KEYS = (
    ("obs_harm", "harm_obs"),
    ("obs_harm_a", "harm_obs_a"),
    ("obs_harm_history", "harm_history"),
)


def sense_kwargs_from_obs(obs_dict) -> Dict[str, Any]:
    """The harm kwargs `agent.sense()` needs but `act()` never passes.

    Usage in a Q-081 recording loop (the canonical shape is experiments/_harness.py:185):

        latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"],
                             **sense_kwargs_from_obs(obs_dict))

    Omitting this leaves z_harm and z_harm_a null no matter what the flags say.
    """
    return {kw: obs_dict.get(key) for kw, key in HARM_OBS_KEYS}


def q081_profile_kwargs() -> Dict[str, bool]:
    """The from_dims kwargs that enable every dark Q-081 signal.

    Usage:
        cfg = REEConfig.from_dims(..., **q081_profile_kwargs())

    Deliberately returns ONLY the recording-related flags. It sets no dimensions, no
    learning rates and no schedule, so it composes with any experiment's own config
    without silently moving a scientific knob.
    """
    return {name: True for name in Q081_FLAGS}


def verify_stock_defaults(config_cls) -> Dict[str, bool]:
    """Confirm every profile flag is still False on a default config.

    Returns {flag: observed_default}. A True value means the flag has been flipped in
    the substrate since the audit, and the "non-default" declaration is now wrong.
    Takes the REEConfig class (injected, so this module stays torch-free at import).
    """
    cfg = config_cls()
    observed = {}
    for name in Q081_FLAGS:
        observed[name] = bool(_read_flag(cfg, name))
    return observed


def _read_flag(cfg, name: str) -> bool:
    """Read a profile flag off a config, whichever sub-dataclass owns it."""
    for holder in (
        cfg,
        getattr(cfg, "latent", None),
        getattr(cfg, "hippocampal", None),
        getattr(cfg, "goal", None),
    ):
        if holder is not None and hasattr(holder, name):
            return bool(getattr(holder, name))
    raise AttributeError(f"flag {name} not found on config")


def q081_substrate_declaration(config=None) -> Dict[str, Any]:
    """The manifest block declaring this run as non-default substrate.

    Merge into the manifest under key "non_default_substrate". If `config` is given,
    each flag's ACTUAL effective value on that config is recorded too, so the manifest
    states what really ran rather than what the profile intended.
    """
    flags = {}
    for name, (stock_default, line, why) in sorted(Q081_FLAGS.items()):
        entry = {
            "stock_default": stock_default,
            "profile_value": True,
            "config_py_line": line,
            "unlocks": why,
        }
        if config is not None:
            entry["effective_value"] = bool(_read_flag(config, name))
        flags[name] = entry
    return {
        "profile_id": Q081_PROFILE_ID,
        "is_default_substrate": False,
        "reason": (
            "Q-081 cross-stream recording profile: 7 default-OFF flags enabled so the "
            "per-step trace is not silently null for a quarter of the stream checklist. "
            "Results describe a configuration REE does not normally run in."
        ),
        "flags": flags,
        "streams_gated_by": dict(STREAMS_GATED_BY),
        "loop_driven_requirements": dict(LOOP_DRIVEN_REQUIREMENTS),
        "e2_pe_decision": (
            "use_tpj_comparator enabled; the E2 stream is the true E2 predict_next_self "
            "error (agent._tpj_last_agency_signal). The E3 world-rollout error was NOT "
            "substituted -- doing so would duplicate the E3 stream at the E3 rate and "
            "manufacture Outcome B. CAVEAT: the comparator is staged at the end of "
            "select_action (agent.py:7582), after the between-tick short-circuit "
            "(agent.py:5463), so it resolves at E3 CADENCE, not per step. Distinct "
            "CONTENT from E3, not a distinct rate. Q-081 has no true middle-rate stream."
        ),
        "audit": "REE_assembly/evidence/planning/q081_cross_stream_telemetry_audit.md",
    }
