"""Contract tests for SD-059 / MECH-358 relief/safety escape-affordance bridge.

C1 default-off no-op (agent.escape_affordance_bridge is None; bit-identical OFF).
C2 relief credit: directed action under threat that drops z_harm_a credits class.
C3 safety credit: directed action after which threat is absent credits class.
C4 approach bias is negative (favoured) on credited classes under threat;
   no-op class never gets a bonus.
C5 threat-context gate: bias is exactly zero when safe (no swamping food/goal).
C6 MECH-094: update + approach bias are no-ops under simulation_mode.
C7 reset clears the within-episode trace but PRESERVES the learned tables.
C8 half toggles: relief-only / safety-only dissociate.
C9 trained-safety signal (603i fix): under retained threat (raw threat_scale>0),
   a directed action with a trained safety_signal >= threshold credits the safety
   half; flag OFF or signal below threshold does not (and is bit-identical).
C10 agent bit-identical: the escape_use_trained_safety_signal flag default-OFF
   leaves the action stream unchanged.
"""

import numpy as np
import torch

from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig
from ree_core.pfc.escape_affordance_bridge import (
    EscapeAffordanceBridge,
    EscapeAffordanceBridgeConfig,
)


def _bridge(**kw):
    cfg = EscapeAffordanceBridgeConfig(
        n_action_classes=5, threat_floor=0.1, threat_ref=0.5,
        relief_learn_rate=0.5, safety_learn_rate=0.5, leak_rate=0.0, **kw
    )
    return EscapeAffordanceBridge(cfg)


def test_c1_default_off_noop_and_bit_identical():
    def run(seed, **kw):
        torch.manual_seed(seed)
        np.random.seed(seed)
        cfg = REEConfig.from_dims(
            world_obs_dim=250, body_obs_dim=12, harm_obs_dim=50,
            harm_obs_a_dim=50, action_dim=5, **kw
        )
        ag = REEAgent(cfg)
        if not kw.get("use_escape_affordance_bridge", False):
            assert ag.escape_affordance_bridge is None
        acts = []
        for _ in range(8):
            a = ag.act_with_split_obs(torch.zeros(1, 12), torch.zeros(1, 250))
            acts.append(int(a.argmax(dim=-1).flatten()[0].item()))
        return acts

    assert run(0) == run(0, use_escape_affordance_bridge=False)


def test_c2_relief_credit():
    b = _bridge()
    b.update(0.4, last_action_class=2, last_action_directed=True)   # cache prev
    b.update(0.15, last_action_class=2, last_action_directed=True)  # harm dropped
    assert b.relief_affordance[2] > 0.0
    assert all(b.relief_affordance[i] == 0.0 for i in (0, 1, 3, 4))


def test_c3_safety_credit():
    b = _bridge()
    b.update(0.4, last_action_class=3, last_action_directed=True)   # cache prev
    b.update(0.02, last_action_class=3, last_action_directed=True)  # threat absent
    assert b.safety_affordance[3] > 0.0


def test_c4_approach_bias_favours_credited_class():
    b = _bridge()
    b.update(0.4, last_action_class=2, last_action_directed=True)
    b.update(0.15, last_action_class=2, last_action_directed=True)
    bias = b.compute_approach_bias(0.4, [0, 1, 2, 3, 4])
    assert bias[2] < 0.0                 # credited directed class favoured
    assert float(bias[0]) == 0.0         # no-op class never gets a bonus
    assert float(bias.abs().max()) <= b.config.bias_scale + 1e-9


def test_c5_threat_context_gate():
    b = _bridge()
    b.update(0.4, last_action_class=2, last_action_directed=True)
    b.update(0.15, last_action_class=2, last_action_directed=True)
    bias_safe = b.compute_approach_bias(0.05, [0, 1, 2, 3, 4])  # below threat_floor
    assert float(bias_safe.abs().max()) == 0.0


def test_c6_simulation_mode_noop():
    b = _bridge()
    b.update(0.4, last_action_class=2, last_action_directed=True)
    b.update(0.15, last_action_class=2, last_action_directed=True, simulation_mode=True)
    assert b.relief_affordance[2] == 0.0  # sim update did not credit
    bias = b.compute_approach_bias(0.4, [0, 1, 2, 3, 4], simulation_mode=True)
    assert float(bias.abs().max()) == 0.0


def test_c7_reset_preserves_tables_clears_trace():
    b = _bridge()
    b.update(0.4, last_action_class=2, last_action_directed=True)
    b.update(0.15, last_action_class=2, last_action_directed=True)
    before = b.relief_affordance[2]
    assert before > 0.0
    b.reset()
    assert b.relief_affordance[2] == before     # learned table persists
    assert b._z_harm_a_prev is None             # within-episode trace cleared


def test_c8_half_toggles_dissociate():
    relief_only = _bridge(use_safety_credit=False)
    relief_only.update(0.4, 2, True)
    relief_only.update(0.02, 2, True)   # harm dropped AND threat absent
    assert relief_only.relief_affordance[2] > 0.0
    assert relief_only.safety_affordance[2] == 0.0

    safety_only = _bridge(use_relief_credit=False)
    safety_only.update(0.4, 2, True)
    safety_only.update(0.02, 2, True)
    assert safety_only.safety_affordance[2] > 0.0
    assert safety_only.relief_affordance[2] == 0.0


def test_c9_trained_safety_signal_credits_under_retained_threat():
    # threat stays ABOVE threat_floor on the post-action tick (raw threat-absence
    # never fires -- the 603i starvation case), but a trained safety_signal above
    # threshold credits the safety half.
    b = _bridge(use_relief_credit=False, use_trained_safety_signal=True,
                safety_signal_threshold=0.5)
    b.update(0.4, last_action_class=3, last_action_directed=True)          # cache prev
    b.update(0.30, last_action_class=3, last_action_directed=True,        # threat retained
             safety_signal=0.8)                                           # trained-safe
    assert b.safety_affordance[3] > 0.0
    assert b.get_state()["mech358_n_safety_credit_trained"] >= 1

    # Signal BELOW threshold under retained threat -> no credit.
    b2 = _bridge(use_relief_credit=False, use_trained_safety_signal=True,
                 safety_signal_threshold=0.5)
    b2.update(0.4, 3, True)
    b2.update(0.30, 3, True, safety_signal=0.2)
    assert b2.safety_affordance[3] == 0.0

    # Flag OFF -> safety_signal ignored; retained threat -> no credit (bit-identical).
    b3 = _bridge(use_relief_credit=False, use_trained_safety_signal=False)
    b3.update(0.4, 3, True)
    b3.update(0.30, 3, True, safety_signal=0.99)
    assert b3.safety_affordance[3] == 0.0
    assert b3.get_state()["mech358_n_safety_credit_trained"] == 0


def test_c10_agent_trained_safety_flag_off_bit_identical():
    def run(seed, **kw):
        torch.manual_seed(seed)
        np.random.seed(seed)
        cfg = REEConfig.from_dims(
            world_obs_dim=250, body_obs_dim=12, harm_obs_dim=50,
            harm_obs_a_dim=50, action_dim=5,
            use_escape_affordance_bridge=True, **kw
        )
        ag = REEAgent(cfg)
        acts = []
        for _ in range(8):
            a = ag.act_with_split_obs(torch.zeros(1, 12), torch.zeros(1, 250))
            acts.append(int(a.argmax(dim=-1).flatten()[0].item()))
        return acts

    # bridge ON; the new trained-safety flag defaults OFF -> identical to passing
    # it explicitly False.
    assert run(0) == run(0, escape_use_trained_safety_signal=False)
