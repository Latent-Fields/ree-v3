"""C6/C7/C8: SD-032 cingulate cluster wiring contracts.

These test the WIRING, not the thresholds. When a substrate is disabled
(default), the agent attribute is None. When enabled, one tick of the
relevant agent method observably mutates the module's state in the
direction the spec says it should, and the plumbing into downstream
consumers (E3 score_bias, coordinator signal cache, descending gain) is
live.

No thresholds, no directional magnitudes copied from EXQ evidence. If
SD-032b / SD-032c / SD-032d / SD-032e are re-implemented under different
internal contracts, these tests should still pass as long as the
integration points survive.

Coverage:
  C6  SD-032b dACC   dacc._action_history grows after a full step
  C7  SD-032c AIC    aic.tick() runs in sense(); _aic_last_tick populated;
                     harm_s_gain in [0,1]
  C8a SD-032d PCC    pcc_stability injected into salience coordinator
  C8b SD-032e pACC   effective_drive(base) adds _drive_bias and clips
"""

import torch

from ree_core.agent import REEAgent

from tests.fixtures.seed_utils import set_all_seeds
from tests.fixtures.tiny_env import make_tiny_env
from tests.fixtures.tiny_configs import make_tiny_config
from tests.fixtures.tiny_loop import step_once


def _build_agent(seed: int = 0, **flags):
    set_all_seeds(seed)
    env = make_tiny_env(seed=seed)
    cfg = make_tiny_config(env, **flags)
    agent = REEAgent(cfg)
    agent.reset()
    _flat, obs_dict = env.reset()
    return agent, env, obs_dict


# --------------------------------------------------------------------------- #
# C6: SD-032b dACC wiring
# --------------------------------------------------------------------------- #

def test_dacc_disabled_by_default():
    agent, _env, _obs = _build_agent()
    assert agent.dacc is None, "dACC should be None when use_dacc not set"
    assert agent.dacc_adapter is None, "dACC adapter should be None when use_dacc not set"


def test_dacc_records_action_on_select():
    agent, env, obs = _build_agent(
        use_dacc=True,
        use_harm_stream=True,
        use_affective_harm_stream=True,
    )
    assert agent.dacc is not None, "use_dacc=True must instantiate DACCAdaptiveControl"
    assert len(agent.dacc._action_history) == 0

    step_once(agent, env, obs)

    assert len(agent.dacc._action_history) == 1, \
        "SD-032b MECH-260: select_action() must call dacc.record_action() on emitted action"


# --------------------------------------------------------------------------- #
# C7: SD-032c AIC wiring
# --------------------------------------------------------------------------- #

def test_aic_disabled_by_default():
    agent, _env, _obs = _build_agent()
    assert agent.aic is None, "AIC should be None when use_aic_analog not set"


def test_aic_ticks_on_sense():
    agent, env, obs = _build_agent(
        use_aic_analog=True,
        use_harm_stream=True,
        use_affective_harm_stream=True,
    )
    assert agent.aic is not None, "use_aic_analog=True must instantiate AICAnalog"
    assert agent._aic_last_tick is None

    step_once(agent, env, obs)

    assert agent._aic_last_tick is not None, \
        "SD-032c: sense() must tick AIC and cache _aic_last_tick"
    for key in ("aic_salience", "harm_s_gain", "urgency_signal"):
        assert key in agent._aic_last_tick, \
            f"SD-032c: AIC tick must expose '{key}' for coordinator/salience routing"
    gain = float(agent._aic_last_tick["harm_s_gain"])
    assert 0.0 <= gain <= 1.0, \
        f"SD-032c: harm_s_gain must be in [0,1], got {gain}"


# --------------------------------------------------------------------------- #
# C8a: SD-032d PCC wiring
# --------------------------------------------------------------------------- #

def test_pcc_disabled_by_default():
    agent, _env, _obs = _build_agent()
    assert agent.pcc is None, "PCC should be None when use_pcc_analog not set"


def test_pcc_stability_injected_into_salience():
    agent, env, obs = _build_agent(
        use_pcc_analog=True,
        use_salience_coordinator=True,
        use_harm_stream=True,
        use_affective_harm_stream=True,
    )
    assert agent.pcc is not None, "use_pcc_analog=True must instantiate PCCAnalog"
    assert agent.salience is not None, "use_salience_coordinator=True must instantiate SalienceCoordinator"

    step_once(agent, env, obs)

    # The coordinator is the architectural consumer of pcc_stability. If
    # the signal never lands in the input cache, MECH-259 cannot use it.
    signals = agent.salience._input_signals
    assert "pcc_stability" in signals, \
        "SD-032d MECH-259: select_action() must inject pcc_stability into coordinator._input_signals"
    stability = float(signals["pcc_stability"])
    assert 0.0 <= stability <= 1.0, \
        f"SD-032d: pcc_stability must be in [0,1], got {stability}"


# --------------------------------------------------------------------------- #
# C8b: SD-032e pACC wiring
# --------------------------------------------------------------------------- #

def test_pacc_disabled_by_default():
    agent, _env, _obs = _build_agent()
    assert agent.pacc is None, "pACC should be None when use_pacc_analog not set"


def test_pacc_effective_drive_applies_and_clips_bias():
    agent, _env, _obs = _build_agent(use_pacc_analog=True)
    assert agent.pacc is not None, "use_pacc_analog=True must instantiate PACCAnalog"

    # With zero bias, effective_drive should equal the base.
    assert agent.pacc.effective_drive(0.5) == 0.5

    # Additive bias within cap.
    agent.pacc._drive_bias = 0.2
    assert abs(agent.pacc.effective_drive(0.3) - 0.5) < 1e-6, \
        "SD-032e: effective_drive must add _drive_bias to the base"

    # Must clip to [0, 1].
    agent.pacc._drive_bias = 0.5
    assert agent.pacc.effective_drive(0.8) == 1.0, \
        "SD-032e: effective_drive must clip the positive tail at 1.0"
    agent.pacc._drive_bias = -0.5
    assert agent.pacc.effective_drive(0.2) == 0.0, \
        "SD-032e: effective_drive must clip the negative tail at 0.0"
