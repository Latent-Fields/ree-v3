"""MECH-268 f_sat: which outcome class the saturation count is taken against.

THE DEFECT THIS PINS. `DACC._saturation_factor` computes
`n_rec = count(cls in last window outcomes)` and `sat_factor = 1/(1 + strength *
max(0, n_rec - grace))`. Which class `cls` IS decides what the factor means, and
the spec (DACCConfig's MECH-268 block) says the CURRENT outcome class. Until
2026-09-18 REEAgent's only live dACC call never tagged it, so `_saturation_factor`
took its `cls is None` fallback -- `self._outcome_history[-1]`, the PREVIOUS
class -- on EVERY waking tick. Its docstring called that a first-tick case
("caller did not tag the outcome yet, e.g. first tick of an episode"), describing
a path production never took, which is how it stayed hidden.

WHY THE CLASS IS AVAILABLE AT READ TIME (the question this work had to settle
before touching anything). The dACC read happens during action SELECTION and the
FIFO write happens after the tick resolves, so the current tick's outcome looked
like it might genuinely not be knowable yet. It is: the FIFO writer does not read
a resolved outcome either. It thresholds `self._current_latent.z_harm_a` against
`contextual_safety_harm_threshold`, and `_current_latent` is assigned only in
`__init__` and the observe path -- never inside `select_action` -- so the same
attribute is already settled at the read. Same quantity, same derivation, earlier
in the same function. `test_l2_*` pins that the threaded class equals the class
the writer records.

WHAT THE FIX DOES NOT DO, pinned by `test_r4_*` so no later session claims
otherwise: E[sat_factor] is symmetric-unimodal in outcome-class density under
EITHER referent. An all-0 stream and an all-1 stream saturate identically; a
mixed stream saturates least. That is a property of counting recurrences of
whichever class is current, i.e. of the functional form, and the form is a
separate open question. The referent fix buys the TRANSITION case, not
monotonicity.

THE CLASS IS NEAR-CONSTANT LIVE, WHICH BOUNDS WHAT THIS FIX CAN BUY. With the
default `contextual_safety_harm_threshold` (0.05) the live `z_harm_a` norm on
CausalGridWorldV2 runs 0.354-0.890 and never crosses it, so the recorded class
does not flip (measured 2026-09-18). That agrees with the independent 2026-09-17
measurement recorded on `chip-20260917-mech268-closure-cadence-dose`:
`harm_class_fraction` 0.946-1.000 regardless of the threshold, i.e. the
threshold lever is INERT. Since both referents agree whenever the class is
constant, the corrected referent bites only on the minority off-class ticks and
at genuine transitions -- it is a spec-conformance fix with a small expected
ecological yield, not a differentiator.

DO NOT READ `test_l5_*` AS "f_sat IS PINNED AT ITS FLOOR LIVE" -- an earlier
draft of this file said that and it was WRONG. It holds only with SD-034 closure
OFF, which is the case in this file's fixture (`use_closure_operator` defaults
False, so no ClosureOperator is built and nothing ever clears the FIFO). On a
TRAINED agent with closure ON (`closure_reset_outcome_history` defaults True),
`ClosureOperator._fire()` calls `dacc.reset_outcome_history()`, `n_rec` re-ramps
from 0 and sweeps the full 0..8 range -- interior occupancy 0.911-0.946,
measured 2026-09-17 (REE_assembly e64d57908f). So the ecological driver of
graded saturation is CLOSURE CADENCE, not harm density, and f_sat is richly
exercised live. `test_l5_*` pins the closure-OFF mechanism only, and exists to
guard `test_l3_*`'s no-transition premise.

LEVER. `dacc_saturation_thread_current_class`, default False = the fallback
stands and behaviour is bit-identical (`test_l1_*`, `test_l3_*`).
"""

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from ree_core.agent import REEAgent
from ree_core.cingulate.dacc import DACCAdaptiveControl, DACCConfig
from ree_core.utils.config import REEConfig

WINDOW, STRENGTH, GRACE = 8, 0.3, 2


def _dacc(**over):
    cfg = dict(
        dacc_saturation_enabled=True,
        dacc_saturation_window=WINDOW,
        dacc_saturation_strength=STRENGTH,
        dacc_saturation_grace=GRACE,
    )
    cfg.update(over)
    return DACCAdaptiveControl(DACCConfig(**cfg))


def _expected(n_rec):
    excess = max(0, n_rec - GRACE)
    return 1.0 if excess <= 0 else 1.0 / (1.0 + STRENGTH * excess)


# ----------------------------------------------------------------------
# R: referent semantics, at the unit the defect lives in
# ----------------------------------------------------------------------
def test_r1_untagged_caller_habituates_to_the_PREVIOUS_class():
    """The fallback is 'previous class', not 'no class' -- and not first-tick."""
    d = _dacc()
    for c in [0, 0, 0, 0, 0, 0, 0, 0]:
        d.record_outcome(c)
    sat_none, n_none = d._saturation_factor(None)
    sat_prev, n_prev = d._saturation_factor(0)
    assert (sat_none, n_none) == (sat_prev, n_prev)
    assert n_none == WINDOW
    assert sat_none == pytest.approx(_expected(WINDOW))


def test_r2_the_two_referents_DIVERGE_at_a_class_transition():
    """The case habituation exists to spare: a novel outcome after a long run.

    Fallback scores it against the OLD class's 8 recurrences and attenuates it
    ~2.8x; the current-class referent counts 0 and leaves it untouched.
    """
    d = _dacc()
    for _ in range(WINDOW):
        d.record_outcome(0)
    sat_fallback, n_fallback = d._saturation_factor(None)   # cls <- 0 (previous)
    sat_current, n_current = d._saturation_factor(1)        # cls <- 1 (novel now)
    assert (n_fallback, n_current) == (WINDOW, 0)
    assert sat_fallback == pytest.approx(1.0 / (1.0 + STRENGTH * (WINDOW - GRACE)))
    assert sat_fallback == pytest.approx(0.35714285, abs=1e-6)
    assert sat_current == pytest.approx(1.0)
    assert sat_current / sat_fallback == pytest.approx(2.8, abs=1e-6)


def test_r3_the_referent_reaches_the_downstream_consumers():
    """Not just the factor: pe, control_required and mode_ev must move with it."""
    payoffs = torch.tensor([1.0, 0.5, 0.2])
    effort = torch.tensor([0.1, 0.2, 0.3])
    kw = dict(
        z_harm_a=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        z_harm_a_pred=torch.zeros(4),
        candidate_payoffs=payoffs,
        candidate_effort=effort,
        candidate_action_classes=[0, 1, 2],
        precision=100.0,
        drive_level=0.0,
    )
    out = {}
    for label, cls in (("fallback", None), ("current", 1)):
        d = _dacc()
        for _ in range(WINDOW):
            d.record_outcome(0)
        out[label] = d.forward(current_outcome_class=cls, **kw)

    assert float(out["current"]["pe"]) > float(out["fallback"]["pe"])
    # mode_ev = payoff - (pe * effort_cost) * effort, so a larger pe pushes every
    # candidate DOWN, and by more where effort is larger -- the spread moves too.
    assert not torch.allclose(out["current"]["mode_ev"], out["fallback"]["mode_ev"])
    assert float(out["current"]["mode_ev"][2]) < float(out["fallback"]["mode_ev"][2])
    assert out["current"]["choice_difficulty"] != out["fallback"]["choice_difficulty"]


@pytest.mark.parametrize("density", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_r4_the_referent_does_NOT_buy_monotonicity_in_harm_density(density):
    """HONEST NEGATIVE -- guards against over-claiming this fix.

    Saturation is symmetric-unimodal in class density under the current-class
    referent too: the pure streams both saturate hardest, the mixed stream
    least. Making it monotone would mean changing the FORM, which is a separate
    question and deliberately untouched here.
    """
    n_ones = int(round(density * WINDOW))
    hist = [1] * n_ones + [0] * (WINDOW - n_ones)
    d = _dacc()
    for c in hist:
        d.record_outcome(c)
    cls = hist[-1] if hist else 0
    sat, n_rec = d._saturation_factor(cls)
    assert n_rec == hist.count(cls)
    assert sat == pytest.approx(_expected(n_rec))
    if density in (0.0, 1.0):
        assert sat == pytest.approx(_expected(WINDOW))  # both extremes identical


def test_r4b_pure_streams_of_OPPOSITE_class_saturate_identically():
    sats = []
    for c in (0, 1):
        d = _dacc()
        for _ in range(WINDOW):
            d.record_outcome(c)
        sats.append(d._saturation_factor(c)[0])
    assert sats[0] == pytest.approx(sats[1])
    assert sats[0] == pytest.approx(_expected(WINDOW))


def test_r5_disabled_and_empty_history_still_short_circuit():
    assert _dacc(dacc_saturation_enabled=False)._saturation_factor(1) == (1.0, 0)
    assert _dacc()._saturation_factor(None) == (1.0, 0)  # empty buffer, no class


# ----------------------------------------------------------------------
# L: the live agent path -- the half that was actually broken
# ----------------------------------------------------------------------
def _live(thread, threshold=None, seed=7, steps=60):
    """Drive the canonical StepHarness loop; return the per-read trace.

    Uses agent.sense/select_action via StepHarness because that is the path the
    real drivers take -- act_with_split_obs does not feed obs_harm_a, so
    z_harm_a stays None and the outcome class is not exercised at all.
    """
    from _harness import StepHarness
    from ree_core.environment.causal_grid_world import CausalGridWorldV2

    torch.manual_seed(seed)
    env = CausalGridWorldV2(
        seed=seed, size=5, num_hazards=2, num_resources=1, use_proxy_fields=True
    )
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        self_dim=16,
        world_dim=16,
        use_dacc=True,
        use_affective_harm_stream=True,
        dacc_saturation_enabled=True,
        dacc_saturation_window=WINDOW,
        dacc_saturation_strength=STRENGTH,
        dacc_saturation_grace=GRACE,
        dacc_saturation_thread_current_class=thread,
    )
    if threshold is not None:
        cfg.contextual_safety_harm_threshold = float(threshold)
    agent = REEAgent(cfg)
    assert agent.dacc is not None, "use_dacc did not build the module"

    trace = []
    original = agent.dacc._affective_pe

    def spy(*a, **kw):
        zha = (
            agent._current_latent.z_harm_a
            if agent._current_latent is not None
            else None
        )
        pe = original(*a, **kw)
        trace.append(
            {
                "cls_arg": kw.get("current_outcome_class", "POSITIONAL"),
                "norm": None if zha is None else float(zha.detach().norm().item()),
                "sat": float(agent.dacc._last_saturation_factor),
                "pe": float(pe),
                "mode_ev_sum": None,
            }
        )
        return pe

    agent.dacc._affective_pe = spy
    harness = StepHarness(agent, env, train_mode=False, seed=seed)

    def on_step(_r):
        if trace and agent._dacc_last_bundle is not None:
            trace[-1]["mode_ev_sum"] = float(
                agent._dacc_last_bundle["mode_ev"].sum().item()
            )

    with torch.no_grad():
        harness.run_episode(max_steps=steps, on_step=on_step)
    return agent, cfg, trace


def _fingerprint(agent):
    """Parameter fingerprint + RNG state, for the bit-identity comparison."""
    ps = [p.detach().reshape(-1) for p in agent.parameters()]
    flat = torch.cat(ps) if ps else torch.zeros(1)
    return (
        int(flat.numel()),
        float(flat.sum().item()),
        float(flat.abs().max().item()),
        torch.get_rng_state().clone(),
    )


def test_l1_lever_OFF_passes_the_class_as_None_ie_the_pre_change_call():
    """OFF is not 'a similar call' -- it is argument-for-argument the old one."""
    _agent, _cfg, trace = _live(thread=False)
    assert trace, "the live dACC path did not run"
    assert all(r["cls_arg"] is None for r in trace)


def test_l2_lever_ON_passes_the_same_class_the_FIFO_writer_records():
    """The threaded class must be the writer's derivation, not a near-miss."""
    _agent, cfg, trace = _live(thread=True)
    assert trace
    thr = float(cfg.contextual_safety_harm_threshold)
    for r in trace:
        assert r["cls_arg"] in (0, 1)
        expected = 1 if (r["norm"] is not None and r["norm"] > thr) else 0
        assert r["cls_arg"] == expected


def test_l3_the_new_block_consumes_no_RNG_and_mutates_no_parameters():
    """Bit-identity, proven where the added code RUNS rather than where it is
    skipped. In a regime with no class transition the two referents agree, so
    ON exercises the whole new block and must still land on an identical
    trajectory, identical parameter fingerprint and identical RNG state. OFF
    (which skips the block entirely, and passes None per L1) is then identical
    a fortiori.
    """
    agent_off, _c0, t_off = _live(thread=False)
    agent_on, _c1, t_on = _live(thread=True)

    # Guard the premise rather than assume it: the default threshold must in
    # fact produce no transition, or this test proves nothing.
    classes = {
        1 if (r["norm"] or 0.0) > float(_c1.contextual_safety_harm_threshold) else 0
        for r in t_on
    }
    assert len(classes) == 1, f"premise broken: class flipped live ({classes})"

    assert len(t_off) == len(t_on)
    assert [r["sat"] for r in t_off] == [r["sat"] for r in t_on]
    assert [r["pe"] for r in t_off] == [r["pe"] for r in t_on]
    assert [r["mode_ev_sum"] for r in t_off] == [r["mode_ev_sum"] for r in t_on]

    f_off, f_on = _fingerprint(agent_off), _fingerprint(agent_on)
    assert f_off[:3] == f_on[:3]
    assert torch.equal(f_off[3], f_on[3]), "RNG state diverged"


def test_l4_lever_ON_MOVES_the_live_downstream_quantities_at_a_transition():
    """The ON path has to change something, not merely compute a new number.

    The threshold is taken from the run's own median norm rather than pinned,
    so the transition exists by construction on any machine.
    """
    _a, _c, probe = _live(thread=False)
    norms = sorted(r["norm"] for r in probe if r["norm"] is not None)
    assert norms, "no z_harm_a on the live path"
    median = norms[len(norms) // 2]

    _off_a, _off_c, t_off = _live(thread=False, threshold=median)
    _on_a, on_cfg, t_on = _live(thread=True, threshold=median)

    seen = {r["cls_arg"] for r in t_on}
    assert seen == {0, 1}, f"median split did not produce a transition: {seen}"

    sat_off = [r["sat"] for r in t_off]
    sat_on = [r["sat"] for r in t_on]
    assert sat_off != sat_on, "the referent change did not move sat_factor"

    diff = [i for i, (a, b) in enumerate(zip(sat_off, sat_on)) if a != b]
    assert diff
    i = diff[0]
    # At the transition the novel class has few recurrences, so the corrected
    # referent attenuates LESS -- higher sat, higher pe.
    assert sat_on[i] > sat_off[i]
    assert t_on[i]["pe"] > t_off[i]["pe"]

    ev_off = [r["mode_ev_sum"] for r in t_off]
    ev_on = [r["mode_ev_sum"] for r in t_on]
    assert ev_off != ev_on, "mode_ev -- the actual consumer -- did not move"


def test_l5_with_closure_OFF_a_constant_class_pins_f_sat_at_its_floor():
    """SCOPED DELIBERATELY: closure-OFF only. Not a claim about the live regime.

    Two things, and conflating them is the error an earlier draft of this file
    made. (1) The class really is constant here -- the live z_harm_a norm never
    approaches the default 0.05 boundary -- and that generalises: the 2026-09-17
    measurement puts harm_class_fraction at 0.946-1.000 with the threshold lever
    inert. (2) f_sat nonetheless pinning at its floor does NOT generalise: it
    follows from this fixture having no ClosureOperator (use_closure_operator
    defaults False), so nothing ever calls reset_outcome_history(). With closure
    ON, n_rec re-ramps from 0 on every rule completion and sweeps 0..8 (interior
    occupancy 0.911-0.946). This test pins the closure-OFF mechanism, and is
    what makes test_l3's no-transition premise auditable rather than assumed.
    """
    _agent, cfg, trace = _live(thread=True)
    thr = float(cfg.contextual_safety_harm_threshold)
    norms = [r["norm"] for r in trace if r["norm"] is not None]
    assert norms
    assert min(norms) > thr, "the class boundary is no longer degenerate -- requeue"
    assert _agent.closure_operator is None, (
        "premise broken: a ClosureOperator exists, so the FIFO can be reset and "
        "the floor assertion below no longer isolates the closure-OFF mechanism"
    )
    assert {r["cls_arg"] for r in trace} == {1}

    floor = _expected(WINDOW)
    tail = [r["sat"] for r in trace[WINDOW + 1 :]]
    assert tail, "run too short to observe the floor"
    assert all(s == pytest.approx(floor) for s in tail)
    assert floor == pytest.approx(0.35714285, abs=1e-6)


def test_l6_lever_is_reachable_through_from_dims():
    """from_dims ends in **kwargs and silently drops unknown names."""
    cfg = REEConfig.from_dims(
        body_obs_dim=8,
        world_obs_dim=8,
        action_dim=4,
        self_dim=16,
        world_dim=16,
        dacc_saturation_thread_current_class=True,
    )
    assert cfg.dacc_saturation_thread_current_class is True
    assert REEConfig.from_dims(
        body_obs_dim=8, world_obs_dim=8, action_dim=4, self_dim=16, world_dim=16
    ).dacc_saturation_thread_current_class is False
