"""sd-allon-training-signal-absorption-telemetry contracts.

WHAT THIS GUARDS. `experiments/_lib/allon_training._train_all_on_agent` used to return four
keys -- n_p0_ticks, n_p1_ticks, n_e2_train_steps, zworld_p0 -- none of which says anything
about reward, advantage, or head movement. So a drive-axis null measured through this shared
all-ON recipe could not distinguish "the manipulation was ABSORBED (never reached a
parameter)" from "it could have CONVERTED into different committed behaviour and did not".
V3-EXQ-1039 is the confirmed case: a per-episode training return 250-500x its control's
reproduced that control bit-identically on every recorded channel in all 3 seeds, and the
readiness gate built to catch that (C0d) passed at 0.667.

THE INSTRUMENT'S NO-OP PROPERTY IS THE LOAD-BEARING HALF, and it holds by CONSTRUCTION
rather than by flag: additive keys in a returned dict, no optimizer step, no parameter write,
no RNG draw. So these contracts assert BIT-IDENTITY of the pre-existing behaviour first
(C1-C3) and only then that the new keys are present and honest (C4-C6).

C1  The two REINFORCE helpers are bit-identical with and without the new `adv_stats`
    accumulator -- same loss bits AND same numpy RNG state afterwards -- and every
    pre-existing 4-positional-argument call site still works.
C2  The module's RNG-draw inventory is unchanged, and no telemetry helper draws at all.
C3  Every telemetry helper is RNG-neutral (numpy and torch states byte-identical across it).
C4  `_train_all_on_agent` still returns the four legacy keys, plus `absorption` and
    `conversion`, and two identical-seed runs agree on the legacy keys.
C5  ABSENT IS None, NEVER 0.0/False. With capture_head_diagnostics OFF the dead-ReLU
    fraction reports unavailable rather than a measured zero.
C6  `candidate_summary_degenerate` is COMPUTED, not read back from lateral_pfc's own field
    (which is vacuous on this recipe: rule_readout_consumer defaults False and the all-ON
    recipe never sets it, so the field is an __init__ constant False for every run).
"""
import ast
import hashlib
import inspect
import pathlib

import numpy as np
import torch

from ree_core.agent import REEAgent
from tests.fixtures.seed_utils import set_all_seeds
from tests.fixtures.tiny_configs import make_tiny_config
from tests.fixtures.tiny_env import make_tiny_env

from experiments._lib import allon_training as allon
from experiments._lib.allon_training import (
    ADV_MIN_THRESHOLD,
    REINFORCE_BATCH_SIZE,
    _candidate_summary_spread,
    _dist_stats,
    _distinct_first_action_classes,
    _head_norm_delta,
    _head_norm_snapshot,
    _lpfc_reinforce_loss,
    _new_adv_stats,
    _ofc_deval_reinforce_loss,
    _train_all_on_agent,
)

MODULE_PATH = pathlib.Path(allon.__file__)
LEGACY_KEYS = ("n_p0_ticks", "n_p1_ticks", "n_e2_train_steps", "zworld_p0")

# Heads ON: the P1 two-head REINFORCE path is what the accumulator instruments, so a config
# without them would exercise none of it. from_dims swallows unknown kwargs silently
# ([memory] reference-reeconfig-from-dims-silent-kwargs), so every test that uses this
# ASSERTS the heads actually exist rather than assuming the flags took.
HEADS_ON = dict(
    use_lateral_pfc_analog=True,
    lateral_pfc_train_rule_bias_head=True,
    use_ofc_analog=True,
    use_ofc_devaluation_head=True,
    ofc_train_devaluation_head=True,
)


# --- stubs for C1 ---------------------------------------------------------------------
class _StubHead:
    """Deterministic, differentiable per-candidate bias. Draws no RNG when called."""

    def __init__(self, dim: int, seed: int = 0):
        torch.manual_seed(seed)
        self.lin = torch.nn.Linear(dim, 1)

    def _bias(self, feats):
        return self.lin(feats).squeeze(-1)

    compute_bias = _bias
    compute_devaluation_bias = _bias


class _StubAgent:
    def __init__(self, lateral_pfc=None, ofc=None):
        self.lateral_pfc = lateral_pfc
        self.ofc = ofc
        self.device = torch.device("cpu")


def _outcome_buf(n=40, dim=6, k=3):
    """Returns with a deliberate mix above and below ADV_MIN_THRESHOLD around baseline."""
    g = torch.Generator().manual_seed(7)
    buf = []
    for i in range(n):
        feats = torch.rand(k, dim, generator=g)
        ep_return = 0.5 + (0.0 if i % 3 == 0 else (0.2 if i % 3 == 1 else -0.2))
        buf.append((feats, i % k, ep_return))
    return buf


def _np_state_digest():
    s = np.random.get_state()
    return hashlib.sha256(
        s[1].tobytes() + repr((s[2], s[3], s[4])).encode()
    ).hexdigest()


def _torch_state_digest():
    return hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest()


# --- C1 -------------------------------------------------------------------------------
def _assert_loss_and_rng_identical(fn, agent, buf, baseline):
    np.random.seed(1234)
    loss_off = fn(agent, buf, baseline, agent.device)
    rng_off = _np_state_digest()

    stats = _new_adv_stats()
    np.random.seed(1234)
    loss_on = fn(agent, buf, baseline, agent.device, adv_stats=stats)
    rng_on = _np_state_digest()

    assert torch.equal(loss_off.detach(), loss_on.detach()), (
        f"{fn.__name__} changed its returned loss when handed the accumulator"
    )
    assert rng_off == rng_on, (
        f"{fn.__name__} shifted the numpy RNG stream -- every fixed-seed comparison "
        "spanning this commit would be silently confounded"
    )
    return stats


def test_c1_lpfc_loss_is_bit_identical_with_and_without_adv_stats():
    agent = _StubAgent(lateral_pfc=_StubHead(dim=6))
    buf = _outcome_buf()
    baseline = 0.5
    stats = _assert_loss_and_rng_identical(_lpfc_reinforce_loss, agent, buf, baseline)

    # And the counts are exactly the draw the helper made -- re-derived, not guessed.
    np.random.seed(1234)
    idxs = np.random.choice(len(buf), size=min(REINFORCE_BATCH_SIZE, len(buf)),
                            replace=False)
    expect_surviving = sum(
        1 for i in idxs if abs(buf[int(i)][2] - baseline) >= ADV_MIN_THRESHOLD
    )
    assert stats["n_sampled"] == len(idxs)
    assert stats["n_adv_surviving"] == expect_surviving
    assert 0 < expect_surviving < len(idxs), (
        "fixture must exercise BOTH branches of the advantage threshold"
    )


def test_c1_ofc_loss_is_bit_identical_with_and_without_adv_stats():
    agent = _StubAgent(ofc=_StubHead(dim=6, seed=1))
    buf = _outcome_buf()
    stats = _assert_loss_and_rng_identical(_ofc_deval_reinforce_loss, agent, buf, 0.5)
    assert stats["n_sampled"] > 0
    # The OFC head has a SECOND skip route; it must be counted separately so a zero
    # gradient is attributable to the threshold or to the head, not left ambiguous.
    assert "n_bias_skipped" in stats


def test_c1_existing_positional_call_sites_keep_working():
    """v3_exq_724 imports both helpers and calls them with 4 positional arguments."""
    for fn in (_lpfc_reinforce_loss, _ofc_deval_reinforce_loss):
        params = list(inspect.signature(fn).parameters.values())
        assert [p.name for p in params[:4]] == [
            "agent", "outcome_buf", "baseline", "device"
        ], f"{fn.__name__} reordered its positional arguments"
        assert params[4].name == "adv_stats" and params[4].default is None, (
            f"{fn.__name__}'s accumulator must be a trailing keyword defaulting to None"
        )


# --- C2 -------------------------------------------------------------------------------
_RNG_CALL_NAMES = {
    "choice", "randint", "shuffle", "rand", "randn", "random", "sample",
    "multinomial", "bernoulli", "randperm", "Random", "seed", "manual_seed",
}

_TELEMETRY_HELPERS = {
    "_dist_stats", "_new_adv_stats", "_head_norm_snapshot", "_head_norm_delta",
    "_candidate_summary_spread", "_distinct_first_action_classes",
}


def _rng_call_sites(node):
    out = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        fn = sub.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        if name in _RNG_CALL_NAMES:
            out.append((name, sub.lineno))
    return out


def test_c2_module_rng_draw_inventory_is_unchanged():
    """PINNED. The telemetry adds no RNG draw anywhere in the module; a new entry here is
    a silent invalidation of every fixed-seed comparison spanning the change."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    names = sorted(n for n, _ in _rng_call_sites(tree))
    assert names == ["Random", "choice", "choice", "randint", "shuffle"], names


def test_c2_no_telemetry_helper_draws_rng():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in _TELEMETRY_HELPERS:
            seen.add(node.name)
            assert _rng_call_sites(node) == [], (
                f"{node.name} draws RNG: {_rng_call_sites(node)}"
            )
    assert seen == _TELEMETRY_HELPERS, f"missing helpers: {_TELEMETRY_HELPERS - seen}"


# --- C3 -------------------------------------------------------------------------------
def test_c3_telemetry_helpers_are_rng_neutral():
    set_all_seeds(3)
    summaries = torch.rand(4, 5)
    head = torch.nn.Sequential(torch.nn.Linear(5, 4), torch.nn.ReLU(),
                               torch.nn.Linear(4, 1))
    np_before, torch_before = _np_state_digest(), _torch_state_digest()

    _dist_stats([1.0, 2.0, 3.0], "x")
    _dist_stats([], "y")
    _candidate_summary_spread(summaries)
    snap = _head_norm_snapshot(head)
    _head_norm_delta(snap, _head_norm_snapshot(head), "h")
    _head_norm_snapshot(None)
    _distinct_first_action_classes([])

    assert _np_state_digest() == np_before
    assert _torch_state_digest() == torch_before


def test_c3_dist_stats_reports_none_not_zero_on_an_empty_sample():
    empty = _dist_stats([], "q")
    assert empty["q_n"] == 0
    for k in ("q_mean", "q_sd", "q_min", "q_max"):
        assert empty[k] is None, "an empty sample must not be reported as a measured 0.0"
    filled = _dist_stats([1.0, 3.0], "q")
    assert filled == {"q_n": 2, "q_mean": 2.0, "q_sd": 1.0, "q_min": 1.0, "q_max": 3.0}


def test_c3_head_norm_delta_separates_absent_from_unmoved():
    absent = _head_norm_delta(None, None, "h")
    assert absent["h_present"] is False
    assert absent["h_head_weight_norm_delta"] is None
    head = torch.nn.Sequential(torch.nn.Linear(3, 1))
    snap = _head_norm_snapshot(head)
    unmoved = _head_norm_delta(snap, _head_norm_snapshot(head), "h")
    assert unmoved["h_present"] is True
    # A head that did not move reports a real 0.0 delta -- the distinction the boolean
    # `moved > 1e-9` readout could not make.
    assert unmoved["h_head_weight_norm_delta"] == 0.0
    assert isinstance(unmoved["h_last_linear_weight_norm_post"], float)


# --- C4 / C5 / C6 ---------------------------------------------------------------------
def _train(**overrides):
    set_all_seeds(0)
    env = make_tiny_env(seed=0)
    agent = REEAgent(make_tiny_config(env, **overrides))
    assert agent.lateral_pfc is not None and agent.ofc is not None, (
        "the heads must actually be built -- from_dims swallows unknown kwargs silently"
    )
    out = _train_all_on_agent(
        agent, env, seed=0, p0_episodes=2, p1_episodes=3, steps_per_episode=8,
        rung_id="contract", total_denominator=5,
    )
    return agent, out


def test_c4_legacy_keys_survive_and_the_two_telemetry_blocks_are_added():
    _agent, out = _train(**HEADS_ON)
    for k in LEGACY_KEYS:
        assert k in out, f"legacy return key {k} was dropped"
    for k in ("n_p0_ticks", "n_p1_ticks", "n_e2_train_steps"):
        assert isinstance(out[k], int)
    assert isinstance(out["zworld_p0"], dict)
    # The exact-set form is deliberate (it catches silent key CHURN, not only key loss), so a
    # genuinely new block has to be added here by hand rather than slipping in. `zharm_a_p0` is
    # the SD-011 P0h affective-harm-encoder warmup block, added 2026-09-18 -- the sibling of
    # `zworld_p0` for the other harm stream; see experiments/_lib/zharm_a_p0_warmup.py.
    assert set(out) == set(LEGACY_KEYS) | {"absorption", "conversion", "zharm_a_p0"}, (
        "the change must be ADDITIVE KEYS only"
    )
    assert isinstance(out["zharm_a_p0"], dict)


def test_c4_two_identical_seed_runs_agree_on_the_legacy_keys():
    _a1, out1 = _train(**HEADS_ON)
    _a2, out2 = _train(**HEADS_ON)
    assert {k: out1[k] for k in LEGACY_KEYS[:3]} == {k: out2[k] for k in LEGACY_KEYS[:3]}


def test_c4_absorption_answers_the_question_the_four_keys_could_not():
    """Did the run receive a return, and did any of it clear the gradient threshold?"""
    _agent, out = _train(**HEADS_ON)
    a = out["absorption"]
    assert a["p1_ep_return_n"] == 3 and a["p0_ep_return_n"] == 2
    assert a["adv_min_threshold"] == ADV_MIN_THRESHOLD
    for label in ("lpfc", "ofc"):
        assert a[f"{label}_adv_n_sampled"] > 0
        frac = a[f"{label}_adv_surviving_frac"]
        assert frac is not None and 0.0 <= frac <= 1.0
        assert a[f"{label}_adv_abs_mean"] is not None
    # Head MOVEMENT as a signed float, not a boolean.
    assert a["lpfc_bias_head_present"] is True
    assert isinstance(a["lpfc_bias_head_last_linear_weight_norm_delta"], float)
    assert isinstance(a["ofc_deval_head_head_weight_norm_delta"], float)


def test_c4_conversion_records_what_the_selection_layer_had_to_work_with():
    _agent, out = _train(**HEADS_ON)
    c = out["conversion"]
    assert c["n_p1_diag_ticks"] > 0
    assert c["e3_raw_score_range_mean_mean"] is not None
    # The quantity the entry says to record: the head's SHARE of the competition, not the
    # absolute bias bound.
    assert c["score_bias_to_raw_range_ratio_n"] > 0
    assert "modulatory_authority_ratio_competitive_mean" in c
    assert c["modulatory_authority_normalize_basis"] in ("range", "std", None)
    # A bias cannot convert into different behaviour when every candidate commits the same
    # first action, so the offered variety is part of the readout.
    assert c["distinct_first_action_classes_n"] > 0
    assert c["distinct_first_action_classes_mean"] >= 1.0


def test_c5_dead_relu_frac_reports_unavailable_rather_than_a_measured_zero():
    """capture_head_diagnostics defaults OFF, so `_last_hidden_dead_relu_frac` is an
    __init__ constant. Surfacing it as 0.0 would be the very defect this entry names."""
    agent, out = _train(**HEADS_ON)
    assert agent.lateral_pfc.config.capture_head_diagnostics is False
    a = out["absorption"]
    assert a["hidden_dead_relu_frac_available"] is False
    assert a["hidden_dead_relu_frac_n"] == 0
    assert a["hidden_dead_relu_frac_mean"] is None


def test_c6_candidate_summary_degeneracy_is_computed_not_read():
    agent, out = _train(**HEADS_ON)
    # The vacuity this guards against, asserted rather than assumed:
    assert agent.lateral_pfc.config.rule_readout_consumer is False, (
        "the all-ON recipe leaves the consumer off, so lateral_pfc's own "
        "candidate_summary_degenerate field is never assigned"
    )
    assert agent.lateral_pfc.get_state()["candidate_summary_degenerate"] is False
    c = out["conversion"]
    assert c["candidate_summary_degenerate_computed"] is True
    assert c["candidate_summary_post_pre_norm_ratio_n"] > 0
    assert c["candidate_summary_degenerate_frac"] is not None
    assert c["candidate_summary_degeneracy_floor"] > 0.0


def test_c6_module_never_reads_the_vacuous_field_back():
    text = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == "candidate_summary_degenerate":
            raise AssertionError(
                "allon_training must COMPUTE the degeneracy verdict, never read "
                "lateral_pfc.get_state()['candidate_summary_degenerate'], which is an "
                "__init__ constant on this recipe"
            )
    assert "_candidate_summary_spread" in text


def test_c6_spread_flags_a_degenerate_set_and_clears_a_differentiated_one():
    row = torch.arange(6, dtype=torch.float32)
    degenerate = row.unsqueeze(0).repeat(4, 1)          # identical candidates
    pre, post = _candidate_summary_spread(degenerate)
    assert pre > 0.0 and post <= 1e-4 * pre, (pre, post)

    differentiated = torch.eye(4, 6) * 3.0 + 1.0
    pre2, post2 = _candidate_summary_spread(differentiated)
    assert post2 > 1e-4 * pre2, (pre2, post2)

    assert _candidate_summary_spread(None) is None
    assert _candidate_summary_spread(row.unsqueeze(0)) is None   # < 2 candidates
