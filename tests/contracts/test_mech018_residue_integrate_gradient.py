"""Contract: MECH-018 -- ResidueField.integrate() performs a real gradient step,
and is reachable from SleepLoopManager._run_cycle's WRITEBACK phase.

THE DEFECT THIS PINS (EXP-0755 / EVB-1391 release_condition, GFLAG-0306).
integrate() computed F.mse_loss(neural_field(pts), rbf_field(pts)) in a num_steps
loop and accumulated .item(), but never called loss.backward(), and the module
declared no optimizer in its 1259 lines. "integration_loss" read as training
progress while nothing trained. MECH-018 lists "the operation is geometrically
inert" as an explicit FALSIFYING outcome, so every MECH-018 run made before this
landed returned a CONFIDENT FALSE FALSIFICATION by construction.

WHY THE NO-ERASURE ASSERTION IS SHAPED THE WAY IT IS -- read before "simplifying".
The release_condition asks for "a contract test asserting neural_field params MOVE
and rbf_field weights at the recorded _harm_history locations DO NOT fall below
the 'cannot be erased' floor". Written literally, that SECOND half is an
ARITHMETIC IDENTITY and tests nothing: self.rbf_field and self.neural_field are
DISJOINT submodules, so an optimizer built over self.neural_field.parameters()
provably cannot reach rbf weights, and "rbf weights stay above the floor" passes
for every input, forever, by construction. It asserts the absence of a coupling
that does not exist.

So it is written here as two assertions that CAN fail instead:
  * C2 asserts the ISOLATION EXPLICITLY -- the optimizer's param set contains
    ONLY neural_field parameters and NO rbf_field parameter. That is the property
    the identity silently depends on, stated as a test, so a future change that
    widens the optimizer trips here rather than passing vacuously.
  * C5 asserts the no-erasure floor ON THE PATH WHERE ERASURE IS POSSIBLE --
    discharge_domain()'s in-place `.data` write with MIN_FLOOR = 1e-6 and its
    sign-preserving clamp. That is the only code in the module that can drive a
    recorded weight toward zero.
If the optimizer is ever widened to cover rbf_field.weights (making the literal
assertion falsifiable), the MIN_FLOOR clamp MUST be re-applied after every step()
-- autograd writes bypass the `.data` clamp entirely. C2 is what forces that
decision to be made deliberately.

Contracts:
  C1. Flag surface: ResidueConfig.use_offline_integration_gradient_step defaults False;
      REEConfig.use_sleep_residue_integration defaults False; and BOTH are
      reachable through REEConfig.from_dims (the MECH-307 trap: a from_dims
      kwarg never written back to the nested config runs silently OFF).
  C2. ISOLATION: the optimizer integrate() constructs holds exactly the
      neural_field parameters and no rbf_field parameter.
  C3. LIVENESS: neural_field params MOVE with the flag ON and do NOT move with
      it OFF; integration_loss falls from first step to last.
  C4. NO ERASURE via integrate(): rbf weights, active_mask and active_centers
      are bit-identical across the call in BOTH branches, rbf weights receive no
      gradient, and evaluate() at the recorded _harm_history locations never
      falls below the pure rbf core.
  C5. NO ERASURE floor where it is REACHABLE: discharge_domain's MIN_FLOOR
      sign-preserving clamp holds under an aggressive decay.
  C6. BIT-IDENTICAL OFF: the OFF branch returns exactly the three original keys
      and consumes the same RNG as the pre-MECH-018 loop.
  C7. WRITEBACK call site exists, is flag-gated, validates its step count, and
      the agent wires both knobs into SleepLoopManager.
  C8. The pairing trap is observable: the call site emits mech018_residue_trains
      so a run that fired an INERT integration is identifiable in the manifest.

GFLAG-0441 (2026-09-24) -- A SECOND, INDEPENDENT INERTNESS AT PRODUCTION
world_dim. Even with C3's gradient step wired, integrate()'s distillation
samples are drawn from an isotropic Gaussian whose PER-DIMENSION std equals
the kernel bandwidth. At the C1-C8 fixture's world_dim=8 that lands targets at
~4% of peak (loss falls, params move -- these contracts pass). At the
production world_dim=32 the same construction lands targets at ~1.5e-05 of
peak (bandwidth-invariant; mean target/peak is exactly 2^(-world_dim/2)): the
gradient step now fires but distills neural_field toward ~0 everywhere, so
integrate() still does nothing observable except erase evaluate()'s untrained
Softplus pedestal. C1-C8 are BLIND to this because they only ever build
world_dim=8 fields. C9-C11 pin the fix (ResidueConfig.
use_dim_scaled_integrate_sampling) at world_dim=32 without disturbing C1-C8.

  C9.  Flag surface: use_dim_scaled_integrate_sampling defaults False and is
       reachable through REEConfig.from_dims.
  C10. THE CHIP'S OWN CONTRACT: at world_dim=32, mean target/peak stays
       catastrophically small (~1.5e-05) with the flag OFF and clears a
       floor with it ON -- and C10b re-confirms the untouched world_dim=8
       contract (C3) still passes.
  C11. BIT-IDENTICAL RNG CONSUMPTION between ON and OFF (the same shape is
       drawn either way; only the scalar multiplier differs), mirroring C6.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import torch

from ree_core.residue.field import ResidueField
from ree_core.utils.config import REEConfig, ResidueConfig


def _field(trains: bool, seed: int = 7, n_harm: int = 6) -> ResidueField:
    torch.manual_seed(seed)
    rf = ResidueField(ResidueConfig(world_dim=8, use_offline_integration_gradient_step=trains))
    torch.manual_seed(seed + 1)
    for _ in range(n_harm):
        rf.accumulate(torch.randn(1, 8), harm_magnitude=1.0)
    return rf


# ---------------------------------------------------------------- C1

def test_c1_flag_surface_and_from_dims_reachability():
    assert ResidueConfig().use_offline_integration_gradient_step is False
    cfg = REEConfig()
    assert cfg.use_sleep_residue_integration is False
    assert cfg.sleep_residue_integration_steps == 10

    # The MECH-307 trap: from_dims silently swallows unknown kwargs, and the
    # gradient-step lever lives on the NESTED ResidueConfig, so a missing
    # write-back after cls() would run every driver with the feature OFF and no
    # error. Assert the value actually ARRIVES.
    on = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=250,
        action_dim=4,
        use_offline_integration_gradient_step=True,
        use_sleep_residue_integration=True,
        sleep_residue_integration_steps=4,
    )
    assert on.residue.use_offline_integration_gradient_step is True
    assert on.use_sleep_residue_integration is True
    assert on.sleep_residue_integration_steps == 4

    off = REEConfig.from_dims(body_obs_dim=12, world_obs_dim=250, action_dim=4)
    assert off.residue.use_offline_integration_gradient_step is False
    assert off.use_sleep_residue_integration is False


# ---------------------------------------------------------------- C2

def test_c2_optimizer_holds_only_neural_field_params():
    """The isolation the no-erasure identity depends on, asserted explicitly."""
    rf = _field(trains=True)
    captured = {}
    real_adam = torch.optim.Adam

    def _spy(params, *args, **kwargs):
        captured["params"] = list(params)
        return real_adam(captured["params"], *args, **kwargs)

    torch.optim.Adam = _spy
    try:
        rf.integrate(num_steps=2)
    finally:
        torch.optim.Adam = real_adam

    assert "params" in captured, "integrate() constructed no optimizer"
    got = {id(p) for p in captured["params"]}
    neural = {id(p) for p in rf.neural_field.parameters()}
    rbf = {id(p) for p in rf.rbf_field.parameters()}

    assert got == neural, "optimizer param set is not exactly neural_field's"
    assert not (got & rbf), (
        "optimizer reaches rbf_field parameters -- the recorded residue is now "
        "writable by autograd, which BYPASSES the MIN_FLOOR `.data` clamp in "
        "discharge_domain. Re-apply that clamp after every step() before "
        "relaxing this assertion."
    )
    # The premise of the isolation: the two submodules really are disjoint.
    assert neural and rbf and not (neural & rbf)


# ---------------------------------------------------------------- C3

def test_c3_liveness_params_move_on_and_not_off():
    for trains, expect_move in ((True, True), (False, False)):
        rf = _field(trains=trains)
        before = [p.detach().clone() for p in rf.neural_field.parameters()]
        torch.manual_seed(99)
        metrics = rf.integrate(num_steps=25)
        delta = sum(
            float((a.detach() - b).pow(2).sum().item())
            for b, a in zip(before, rf.neural_field.parameters())
        ) ** 0.5
        if expect_move:
            assert delta > 0.0, "INERT: flag ON but neural_field params did not move"
            assert metrics["trained"] == 1.0
            assert metrics["neural_param_delta_norm"] > 0.0
            assert metrics["integration_loss_last"] < metrics["integration_loss_first"], (
                "loss did not fall -- the optimizer step is not reducing the "
                "objective it claims to train on"
            )
        else:
            assert delta == 0.0, "flag OFF but params moved -- OFF is not inert"
            assert "trained" not in metrics


# ---------------------------------------------------------------- C4

def test_c4_integrate_never_erases_recorded_residue():
    for trains in (True, False):
        rf = _field(trains=trains)
        harm = torch.stack(rf._harm_history).squeeze(1)
        w_before = rf.rbf_field.weights.detach().clone()
        mask_before = rf.rbf_field.active_mask.detach().clone()
        centers_before = int(mask_before.sum().item())

        torch.manual_seed(99)
        rf.integrate(num_steps=25)

        assert torch.equal(w_before, rf.rbf_field.weights), (
            "integrate() moved recorded rbf weights (trains=%r)" % trains
        )
        assert torch.equal(mask_before, rf.rbf_field.active_mask)
        assert int(rf.rbf_field.active_mask.sum().item()) == centers_before
        assert rf.rbf_field.weights.grad is None, (
            "rbf weights received a gradient -- the no_grad target computation "
            "or the optimizer isolation has regressed"
        )
        # The "cannot be erased" content is the rbf core. The neural head is
        # Softplus-terminated, so evaluate() = rbf + 0.1 * neural can never sit
        # below it; assert that rather than assuming it.
        with torch.no_grad():
            core = rf.rbf_field(harm)
            assert float((rf.evaluate(harm) - core).min().item()) >= 0.0


# ---------------------------------------------------------------- C5

def test_c5_no_erasure_floor_holds_on_the_decay_path():
    """The floor, tested where erasure is actually REACHABLE.

    discharge_domain() is the only code in the module that drives recorded
    weights toward zero. It enforces MIN_FLOOR = 1e-6 with a sign-preserving
    clamp via an in-place `.data` write.
    """
    MIN_FLOOR = 1e-6
    rf = _field(trains=True)
    with torch.no_grad():
        active = rf.rbf_field.active_mask.clone()
        assert int(active.sum().item()) > 0
        idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
        # Force all three sign cases through the clamp.
        rf.rbf_field.weights.data[idx[0]] = 1e-5
        if idx.numel() > 1:
            rf.rbf_field.weights.data[idx[1]] = -1e-5
        if idx.numel() > 2:
            rf.rbf_field.weights.data[idx[2]] = 0.0
        zeros_before = (rf.rbf_field.weights.data[active] == 0.0).clone()

    centre = torch.stack(rf._harm_history).squeeze(1).mean(dim=0)
    # Aggressive decay, repeatedly: without the clamp this drives weights to ~0.
    for _ in range(40):
        rf.discharge_domain(centre, factor=0.01, radius=100.0)

    with torch.no_grad():
        w = rf.rbf_field.weights.data[active]
    pos, neg, zero = w > 0, w < 0, w == 0
    assert bool((w[pos] >= MIN_FLOOR).all()), "positive weight fell below +MIN_FLOOR"
    assert bool((w[neg] <= -MIN_FLOOR).all()), "negative weight rose above -MIN_FLOOR"
    # Sign preservation: nothing crossed zero, and only the already-zero weights
    # are zero.
    assert bool((zero == zeros_before).all()), "decay zeroed a non-zero weight"


# ---------------------------------------------------------------- C6

def test_c6_off_branch_is_bit_identical():
    rf = _field(trains=False)
    torch.manual_seed(1234)
    metrics = rf.integrate(num_steps=25)
    assert sorted(metrics) == ["history_size", "integration_loss", "steps"], (
        "OFF branch changed the returned key set -- existing drivers "
        "(v3_exq_214/240/240a/246) read this dict"
    )
    # RNG consumption is identical in both branches: the per-step randn_like is
    # drawn whether or not a step is taken, so a downstream draw lands on the
    # same value.
    rf_on = _field(trains=True)
    torch.manual_seed(1234)
    rf_on.integrate(num_steps=25)
    after_on = torch.randn(3)
    torch.manual_seed(1234)
    rf.integrate(num_steps=25)
    after_off = torch.randn(3)
    assert torch.equal(after_on, after_off), (
        "ON and OFF consume different amounts of RNG -- OFF is no longer "
        "bit-identical for downstream draws"
    )


# ---------------------------------------------------------------- C7

def test_c7_writeback_call_site_and_agent_wiring():
    from ree_core.sleep.phase_manager import SleepLoopManager

    sig = inspect.signature(SleepLoopManager.__init__)
    assert sig.parameters["residue_integration"].default is False
    assert sig.parameters["residue_integration_steps"].default == 10

    mgr = SleepLoopManager(residue_integration=True, residue_integration_steps=3)
    assert mgr.residue_integration is True
    assert mgr.residue_integration_steps == 3
    assert SleepLoopManager().residue_integration is False

    for bad in (0, -1):
        try:
            SleepLoopManager(residue_integration_steps=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("residue_integration_steps=%r was accepted" % bad)

    # The call really is in _run_cycle's WRITEBACK section, not merely a stored
    # attribute (the "structurally present, functionally inert" shape).
    src = inspect.getsource(SleepLoopManager._run_cycle)
    assert "self.residue_integration" in src
    assert "residue_field.integrate(" in src

    # Agent wiring: the REEConfig knobs must reach the manager.
    from ree_core.agent import REEAgent

    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=250,
        action_dim=4,
        use_sleep_loop=True,
        use_sleep_residue_integration=True,
        sleep_residue_integration_steps=5,
        use_offline_integration_gradient_step=True,
    )
    agent = REEAgent(cfg)
    assert agent.sleep_loop.residue_integration is True
    assert agent.sleep_loop.residue_integration_steps == 5
    assert agent.residue_field.config.use_offline_integration_gradient_step is True


# ---------------------------------------------------------------- C8

def test_c8_pairing_trap_is_observable():
    """Call site ON + gradient step OFF is an INERT integration -- the exact
    false-falsification shape MECH-018 is exposed to. It must be identifiable
    from the emitted metrics, not silently scored."""
    src = inspect.getsource(
        __import__("ree_core.sleep.phase_manager", fromlist=["SleepLoopManager"])
        .SleepLoopManager._run_cycle
    )
    assert "mech018_residue_trains" in src, (
        "the call site does not record whether the integration it fired was "
        "actually training -- an inert run would be indistinguishable from a "
        "live one in the manifest"
    )
    assert "mech018_residue_integration_fired" in src


def test_c8b_module_declares_an_optimizer_at_all():
    """Direct regression guard on the original defect: the source of the
    residue module must contain a backward()/step() pair."""
    src = Path(inspect.getsourcefile(ResidueField)).read_text(encoding="utf-8")
    assert "loss.backward()" in src, "integrate() no longer calls backward()"
    assert "torch.optim." in src, "residue module declares no optimizer"


# ---------------------------------------------------------------- C9-C11 (GFLAG-0441)

WORLD_DIM_32 = 32


def _field_dimscaled(world_dim, dim_scaled, seed=21, harm_magnitude=1.0):
    """A single-harm-event field, so the RBF sum has exactly one active
    center and integrate()'s targets carry no cross-center contribution --
    the target/peak ratio measured below is then an exact function of the
    sampling geometry, not an artifact of nearby centers overlapping."""
    torch.manual_seed(seed)
    cfg = ResidueConfig(world_dim=world_dim, use_dim_scaled_integrate_sampling=dim_scaled)
    rf = ResidueField(cfg)
    torch.manual_seed(seed + 1)
    rf.accumulate(torch.randn(1, world_dim) * 0.01, harm_magnitude=harm_magnitude)
    return rf


def _mean_target_over_peak(rf, num_steps=300, seed=99):
    """Mean RBF value at integrate()'s own jittered sample points, over the
    RBF's peak value (its weight -- the value AT the recorded center, where
    the exp() term is exactly 1.0). Captured via a forward hook on the live
    rbf_field so this measures the REAL production call, not a re-derivation
    of the sampling formula."""
    captured = []

    def _hook(_module, _inputs, output):
        captured.append(output.detach().clone())

    handle = rf.rbf_field.register_forward_hook(_hook)
    torch.manual_seed(seed)
    try:
        rf.integrate(num_steps=num_steps, train=False)
    finally:
        handle.remove()

    targets = torch.cat(captured)
    assert targets.numel() == num_steps
    peak = float(rf.rbf_field.weights[rf.rbf_field.active_mask].item())
    return float(targets.mean()) / peak


def test_c9_dim_scaled_sampling_flag_surface_and_from_dims_reachability():
    assert ResidueConfig().use_dim_scaled_integrate_sampling is False

    on = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=250,
        action_dim=4,
        use_dim_scaled_integrate_sampling=True,
    )
    assert on.residue.use_dim_scaled_integrate_sampling is True

    off = REEConfig.from_dims(body_obs_dim=12, world_obs_dim=250, action_dim=4)
    assert off.residue.use_dim_scaled_integrate_sampling is False


def test_c10_world_dim_32_sampling_collapse_and_fix():
    """THE CHIP'S OWN CONTRACT. GFLAG-0441 measured mean target/peak of
    4.5e-06 (bw 1.0) / 8.8e-06 (bw 0.15) at world_dim=32 -- both far below any
    reasonable floor, and matching the analytic 2^(-32/2) ~= 1.5e-05. This
    must still reproduce with the flag OFF (bit-identical-OFF requires it),
    and clear a floor with the flag ON."""
    rf_off = _field_dimscaled(WORLD_DIM_32, dim_scaled=False)
    ratio_off = _mean_target_over_peak(rf_off)
    assert ratio_off < 1e-3, (
        f"OFF-path mean target/peak = {ratio_off} at world_dim=32 -- expected "
        "the documented collapse (~1.5e-05); if this fires, either "
        "bit-identical-OFF was broken by this change or this test's geometry "
        "no longer isolates the defect"
    )

    rf_on = _field_dimscaled(WORLD_DIM_32, dim_scaled=True)
    ratio_on = _mean_target_over_peak(rf_on)
    # Analytic fixed-path expectation ~0.61 (see ResidueConfig field comment
    # for the derivation); 0.3 leaves ample margin for the n=300 Monte Carlo
    # estimate while still being 300x the OFF-path floor above.
    assert ratio_on > 0.3, (
        f"ON-path mean target/peak = {ratio_on} at world_dim=32 -- expected "
        "the fix to clear a floor far above the OFF-path collapse"
    )
    assert ratio_on > ratio_off * 100


def test_c10b_world_dim_8_contract_still_passes_unchanged():
    """Re-confirms the pre-existing world_dim=8 fixture (C3's _field helper,
    unrelated to this flag) still trains normally -- this change must not
    disturb the world_dim=8 case the original MECH-018 contracts pin."""
    rf = _field(trains=True)
    torch.manual_seed(1234)
    metrics = rf.integrate(num_steps=25)
    assert metrics["trained"] == 1.0
    assert metrics["integration_loss_last"] < metrics["integration_loss_first"]


def test_c11_on_off_consume_identical_rng_downstream():
    """Same bit-identical-RNG-consumption proof as C6, for this flag: the
    MAGNITUDE of the randn_like draw differs between ON and OFF, but the
    call's shape/dtype/count is identical, so a downstream unrelated draw
    lands on the same value regardless of the flag."""
    rf_on = _field_dimscaled(WORLD_DIM_32, dim_scaled=True)
    torch.manual_seed(555)
    rf_on.integrate(num_steps=10, train=False)
    after_on = torch.randn(3)

    rf_off = _field_dimscaled(WORLD_DIM_32, dim_scaled=False)
    torch.manual_seed(555)
    rf_off.integrate(num_steps=10, train=False)
    after_off = torch.randn(3)

    assert torch.equal(after_on, after_off), (
        "ON and OFF consume different amounts of RNG -- the sampling-scale "
        "flag is no longer bit-identical for downstream draws when OFF"
    )
