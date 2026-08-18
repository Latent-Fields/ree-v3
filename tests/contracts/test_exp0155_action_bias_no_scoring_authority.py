"""EXP-0155: SD-016/MECH-151 action_bias has NO trajectory-RANKING authority.

[chip_ref: chip-20260818-exp0155-action-bias-scoring-disconnect]

THIS MODULE ENCODES A FINDING, NOT A DESIRED PROPERTY. Every assertion here
pins the substrate AS BUILT on 2026-08-18. A future change that gives
action_bias genuine ranking authority SHOULD break this file loudly -- that
is the point. If you are reading this because a test failed, the correct
response is almost certainly to DELETE or REWRITE the failing assertion and
update the SD-016 EXP-0155 paragraph in ree-v3/CLAUDE.md, not to revert the
substrate change.

WHAT WAS ESTABLISHED (source-traced + measured, 2026-08-18)
----------------------------------------------------------
MECH-151 adds a cue-indexed `action_bias` to E2.action_object():
    o_t = action_object_head([z_world, a_t]) + action_bias
and MECH-151's registered notes assert that this "pre-shapes HippocampalModule
search: action-objects consistent with the cue-activated associations are
ELEVATED, contextually inappropriate ones SUPPRESSED ... E1 biases which
action-objects RANK HIGHLY, but E3 still selects."

That ranking does not happen, because NOTHING ON THE RANKING PATH EVER READS
AN ACTION-OBJECT:
  * `o_t` accumulates only into `Trajectory.action_objects`.
  * The world transition is `z_world = world_forward(z_world, action)` and the
    self transition `predict_next_self(z_self, action)` -- neither reads o_t,
    so `world_states` / `states` are action_bias-independent.
  * `HippocampalModule._score_trajectory` scores `get_world_state_sequence()`
    (residue terrain, + optional wanting / curiosity / MECH-267 mode terms),
    falling back to `get_state_sequence()`. It never touches action_objects.
  * `_trajectory_first_action_class` -- and therefore the SP-CEM
    support-preserving elite path -- reads `trajectory.actions`.
  * The MECH-294 theta packet pushes `candidates[0].actions[:, 0, :]`
    (its prose says "first-action object"; the code takes the ACTION).
  * `ree_core/predictors/e3_selector.py` contains ZERO occurrences of
    `action_object` / `action_bias`; E3 reads .actions / .states /
    .world_states only.
  * Across all of `ree_core/`, `action_objects` appears in exactly two files:
    `predictors/e2_fast.py` (producer) and `hippocampal/module.py` (CEM refit
    + pure storage copies in reverse_replay / record_exploration_trajectory /
    the backward-credit buffer).

So action_bias's ONLY causal channel is the CEM refit, where -- because
scores, and hence softmax weights and elite membership, are bias-blind, and
the bias is the same constant on every candidate and every horizon step -- it
is an exact additive translation of the proposal mean in action-object space:
    ao_mean_biased = ao_mean_base + b        (legacy elite-mean AND
    ao_std_biased  = ao_std_base              differentiable-softmax branches)
That shifted mean seeds the NEXT CEM iteration. At num_cem_iterations=1 the
refit is never consumed and action_bias is bit-identically inert end-to-end.

CONSEQUENCE, and it cuts both ways -- do not quote half of it:
  (1) action_bias has NO ranking authority: it cannot make one candidate
      preferred over another. Measured: candidate scores are BIT-IDENTICAL
      with b=None vs b=1e3, and the elite ordering is unchanged.
  (2) action_bias is NOT behaviourally inert. Through the nonlinear
      `action_object_decoder`, that rigid translation genuinely changes which
      actions later iterations decode -- measured, a large bias flips the
      whole final candidate pool's first-action class. It steers WHERE
      proposals are drawn, never WHICH proposal wins.
This also explains why SD-055's differentiable CEM restored a gradient to the
action sequence (V3-EXQ-568, grad_max=372) without producing the expected
behavioural divergence.

WHY THE REPAIR SITE IS E3, NOT `_score_trajectory` (ARC-007 STRICT)
-------------------------------------------------------------------
Q-020 / ARC-007 STRICT (module header): "HippocampalModule generates
VALUE-FLAT proposals ... No separate value head ... E3 introduces ALL
weighting." Teaching `_score_trajectory` to read action_objects would put
E1-supplied weighting inside the hippocampal scorer -- forbidden. MECH-151's
own formulation ("E1 biases which action-objects rank highly, but E3 still
selects") is satisfiable only where E3 does the selecting, and E3 currently
has NO action-object input channel at all. That is the substrate gap.

WHAT THIS DOES *NOT* SAY -- read before citing it against EXP-0155
-------------------------------------------------------------------
EXP-0155's premise was that "something downstream of cue_action_proj zeroes
the signal before it reaches E3.select", inferred from V3-EXQ-449's C2 arm
training cue_action_proj (grad ~0.013, delta ~0.21) while
action_bias_divergence stayed at EXACTLY 0.0. That premise is REFUTED, and
the disconnect pinned here is NOT its cause: `_action_bias_divergence` (see
experiments/v3_exq_449_sd016_cue_action_proj_wiring_probe.py) is computed
directly on `e1.extract_cue_context(z_world)[0]`, i.e. at the E1 OUTPUT, with
zero calls into HippocampalModule, E2 rollout or E3.select. Nothing
downstream can affect it. The exact 0.0 is an UPSTREAM degeneracy: under the
uniform-softmax ContextMemory attention saddle, `cue_context` is constant in
z_world, so the pre-EXQ-449a `cue_action_proj(cue_context)` returned the same
vector for every context -> cosine similarity exactly 1.0 -> divergence
exactly 0.0. Measured 2026-08-18 at fresh init: slot-selection entropy
2.7726 == ln(16) (exactly uniform), and the divergence of `cue_context`
itself is 0.000e+00. That upstream degeneracy is the subject of
chip-20260816-implsub-contextmemory-writepath-degeneracy, not of this file.
"""

from __future__ import annotations

import pathlib

import torch

from ree_core.hippocampal.module import HippocampalModule
from ree_core.predictors.e2_fast import E2Config, E2FastPredictor
from ree_core.residue.field import ResidueConfig, ResidueField
from ree_core.utils.config import HippocampalConfig

W, A, AO, H = 8, 4, 8, 4
N_CAND = 16
BIG_BIAS = 1e3

# float32 relative tolerance. The additive-shift identity is EXACT in real
# arithmetic; the residual is pure floating-point rounding. Measured relative
# error tracks float32 eps (1.19e-7) across b in {1e-3 .. 1e3} and collapses to
# ~1e-16 when the same computation is run in float64. Do NOT tighten this to
# bit-exactness -- that would be asserting a property of float32, not of the
# substrate, and it fails at large |b| (max|shift-b| = 1.2e-4 at b=1e3).
REL_TOL = 1e-6


def _make(**kw) -> HippocampalModule:
    kw.setdefault("num_cem_iterations", 3)
    cfg = HippocampalConfig(
        world_dim=W,
        action_dim=A,
        action_object_dim=AO,
        hidden_dim=32,
        horizon=H,
        num_candidates=N_CAND,
        elite_fraction=0.25,
        **kw,
    )
    e2 = E2FastPredictor(E2Config(
        self_dim=W, world_dim=W, action_dim=A,
        action_object_dim=AO, hidden_dim=32,
    ))
    residue = ResidueField(ResidueConfig(
        world_dim=W, hidden_dim=32, num_basis_functions=8,
    ))
    return HippocampalModule(cfg, e2=e2, residue_field=residue)


def _state(seed: int = 1234):
    torch.manual_seed(seed)
    return torch.randn(1, W), torch.randn(1, W)


def _bias(mag: float = BIG_BIAS) -> torch.Tensor:
    return torch.full((1, AO), float(mag))


def _pair(module, z_self, z_world, actions, b):
    """Same actions, rolled out with and without the bias."""
    base = module.e2.rollout_with_world(
        z_self, z_world, actions, compute_action_objects=True, action_bias=None)
    biased = module.e2.rollout_with_world(
        z_self, z_world, actions, compute_action_objects=True, action_bias=b)
    return base, biased


# --------------------------------------------------------------------------
# 1. The scoring function itself is invariant to action_bias.
# --------------------------------------------------------------------------

def test_score_trajectory_is_bit_identical_under_action_bias():
    """FINDING: _score_trajectory ignores action_bias entirely (terrain-only)."""
    torch.manual_seed(1234)
    m = _make()
    z_self, z_world = _state()
    actions = torch.randn(1, H, A)
    base, biased = _pair(m, z_self, z_world, actions, _bias())

    s_base = m._score_trajectory(base)
    s_biased = m._score_trajectory(biased)
    # Bit-identical, not merely close: the bias never enters the computation.
    assert torch.equal(s_base, s_biased), (
        f"action_bias changed the trajectory score "
        f"({s_base.item()!r} -> {s_biased.item()!r}). If this is intended, see "
        f"the module docstring: the FINDING has changed and ARC-007 STRICT "
        f"applies to the repair site."
    )


def test_score_invariance_holds_with_every_scoring_extension_enabled():
    """The wanting / curiosity / MECH-267 mode-value terms all key on
    world_states too, so none of them opens a path for action_bias."""
    torch.manual_seed(1234)
    m = _make(
        wanting_weight=0.7,
        curiosity_weight=0.5,
        mode_conditioning_enabled=True,
        mode_value_weight={"forage": [0.3] * W},
    )
    z_self, z_world = _state()
    actions = torch.randn(1, H, A)
    base, biased = _pair(m, z_self, z_world, actions, _bias())

    mode = {"forage": 1.0}
    assert torch.equal(
        m._score_trajectory(base, operating_mode=mode),
        m._score_trajectory(biased, operating_mode=mode),
    )
    # ...and with the MECH-267 horizon window narrowed as well.
    assert torch.equal(
        m._score_trajectory(base, max_horizon=2, operating_mode=mode),
        m._score_trajectory(biased, max_horizon=2, operating_mode=mode),
    )


def test_rollout_state_streams_are_bias_free_but_action_objects_carry_it():
    """Locates the disconnect precisely: the bias lands in action_objects and
    nowhere else. world_forward / predict_next_self never read o_t."""
    torch.manual_seed(1234)
    m = _make()
    z_self, z_world = _state()
    actions = torch.randn(1, H, A)
    b = _bias()
    base, biased = _pair(m, z_self, z_world, actions, b)

    # The two streams _score_trajectory can actually see are untouched.
    assert torch.equal(
        base.get_world_state_sequence(), biased.get_world_state_sequence())
    assert torch.equal(base.get_state_sequence(), biased.get_state_sequence())

    # The one stream that carries it, differs by exactly b at every step.
    delta = biased.get_action_object_sequence() - base.get_action_object_sequence()
    assert torch.equal(delta, b.expand_as(delta)), (
        "action_bias is no longer a uniform per-step additive constant on the "
        "action-object sequence; the refit-translation argument below no "
        "longer follows."
    )


# --------------------------------------------------------------------------
# 2. Elite selection -- the only ranking step -- is bias-blind.
# --------------------------------------------------------------------------

def test_elite_ordering_is_unchanged_by_action_bias():
    """SP-CEM classes come from trajectory.actions, scores from world_states;
    neither sees the bias, so elite membership cannot depend on it."""
    torch.manual_seed(1234)
    m = _make(use_support_preserving_cem=True)
    z_self, z_world = _state()
    acts = [torch.randn(1, H, A) for _ in range(N_CAND)]
    b = _bias()

    base = [m.e2.rollout_with_world(z_self, z_world, a, True, action_bias=None)
            for a in acts]
    biased = [m.e2.rollout_with_world(z_self, z_world, a, True, action_bias=b)
              for a in acts]

    s_base = torch.stack([m._score_trajectory(t) for t in base])
    s_biased = torch.stack([m._score_trajectory(t) for t in biased])
    assert torch.equal(s_base, s_biased)
    assert torch.equal(torch.argsort(s_base), torch.argsort(s_biased))

    # First-action classes -- the SP-CEM stratification key -- also match.
    cls_base = [m._trajectory_first_action_class(t) for t in base]
    cls_biased = [m._trajectory_first_action_class(t) for t in biased]
    assert cls_base == cls_biased

    idx = torch.argsort(s_base)[:4]
    e_base, _ = m._support_preserving_elite_indices(base, s_base, idx)
    e_biased, _ = m._support_preserving_elite_indices(biased, s_biased, idx)
    assert torch.equal(e_base, e_biased)


# --------------------------------------------------------------------------
# 3. The refit -- the ONLY channel -- is an exact additive translation.
# --------------------------------------------------------------------------

def _refit_legacy(trajs, elite_idx):
    stacked = torch.stack(
        [trajs[int(i)].get_action_object_sequence() for i in elite_idx])
    return stacked.mean(dim=0), HippocampalModule._stack_std(stacked) + 1e-6


def _refit_differentiable(trajs, scores, temperature=1.0):
    stacked = torch.stack([t.get_action_object_sequence() for t in trajs])
    w = torch.softmax(-torch.stack(list(scores)) / temperature, dim=0)
    return ((w.view(-1, 1, 1, 1) * stacked).sum(dim=0),
            HippocampalModule._stack_std(stacked) + 1e-6)


def test_cem_refit_shifts_ao_mean_by_exactly_b_and_leaves_ao_std_alone():
    """Both refit branches. ao_mean += b (weights sum to 1 / plain mean);
    ao_std unchanged (std is translation-invariant). This is why the bias
    never compounds across iterations -- always exactly one b."""
    torch.manual_seed(1234)
    m = _make()
    z_self, z_world = _state()
    acts = [torch.randn(1, H, A) for _ in range(N_CAND)]
    b = _bias()
    base = [m.e2.rollout_with_world(z_self, z_world, a, True, action_bias=None)
            for a in acts]
    biased = [m.e2.rollout_with_world(z_self, z_world, a, True, action_bias=b)
              for a in acts]
    scores = [m._score_trajectory(t) for t in base]
    elite = torch.argsort(torch.stack(scores))[:4]

    for label, fn in (
        ("legacy elite-mean", lambda tr: _refit_legacy(tr, elite)),
        ("differentiable softmax", lambda tr: _refit_differentiable(tr, scores)),
    ):
        mean_base, std_base = fn(base)
        mean_biased, std_biased = fn(biased)

        shift = mean_biased - mean_base
        err = (shift - b.expand_as(shift)).abs().max().item()
        assert err <= REL_TOL * BIG_BIAS, (
            f"{label}: ao_mean shift is not the additive constant b "
            f"(max|shift-b| = {err:.3g})"
        )
        std_err = (std_biased - std_base).abs().max().item()
        assert std_err <= REL_TOL * BIG_BIAS, (
            f"{label}: ao_std moved under a pure translation "
            f"(max|delta| = {std_err:.3g})"
        )


# --------------------------------------------------------------------------
# 4. End-to-end: inert at 1 iteration; steers-but-never-ranks beyond that.
# --------------------------------------------------------------------------

def _propose(bias_mag, iters, seed=99, **kw):
    torch.manual_seed(seed)
    m = _make(num_cem_iterations=iters, **kw)
    z_self, z_world = torch.randn(1, W), torch.randn(1, W)
    b = None if bias_mag is None else _bias(bias_mag)
    recorded = []
    original = m._score_trajectory

    def spy(traj, **kwargs):
        s = original(traj, **kwargs)
        recorded.append(float(s.item() if hasattr(s, "item") else s))
        return s

    m._score_trajectory = spy
    torch.manual_seed(seed)  # matched RNG immediately before the pass
    cands = m.propose_trajectories(z_world=z_world, z_self=z_self, action_bias=b)
    return recorded, cands


def test_single_cem_iteration_makes_action_bias_bit_identically_inert():
    """With no second iteration there is no refit to consume, so the bias has
    literally no channel: the proposal pool is bit-identical."""
    base = _propose(None, 1)[1]
    biased = _propose(BIG_BIAS, 1)[1]
    assert len(base) == len(biased)
    assert all(torch.equal(x.actions, y.actions) for x, y in zip(base, biased))


def test_first_cem_iteration_scores_match_then_diverge_only_at_the_refit():
    """The sharp localisation: with matched RNG the first num_candidates
    scoring calls (iteration 0) are identical, and the FIRST call that differs
    is exactly the start of iteration 1 -- i.e. the refit, and nothing else,
    is where action_bias enters."""
    for kw in ({}, {"use_differentiable_cem": True}):
        r_base, _ = _propose(None, 3, **kw)
        r_biased, _ = _propose(BIG_BIAS, 3, **kw)
        assert len(r_base) == len(r_biased)
        assert r_base[:N_CAND] == r_biased[:N_CAND], (
            f"iteration-0 scores differ under action_bias (kw={kw}); the bias "
            f"has acquired within-pass ranking authority"
        )
        first_diff = next(
            (i for i in range(len(r_base)) if r_base[i] != r_biased[i]), None)
        assert first_diff == N_CAND, (
            f"first differing scoring call is {first_diff}, expected {N_CAND} "
            f"(the iteration-1 boundary), kw={kw}"
        )


def test_action_bias_does_steer_the_proposal_pool_across_iterations():
    """NEGATIVE CONTROL for the whole file, and the half that is most often
    misquoted: 'no ranking authority' is NOT 'no effect'. The rigid
    translation passes through the nonlinear action_object_decoder and really
    does move the pool. If this ever starts passing as 'identical', the bias
    has gone fully inert -- a DIFFERENT defect from the one pinned above."""
    base = _propose(None, 3)[1]
    biased = _propose(BIG_BIAS, 3)[1]
    assert not all(
        torch.equal(x.actions, y.actions) for x, y in zip(base, biased))


# --------------------------------------------------------------------------
# 5. The repair site (ARC-007 STRICT): E3 has no action-object channel.
# --------------------------------------------------------------------------

def test_e3_selector_has_no_action_object_channel():
    """MECH-151 says 'E1 biases which action-objects rank highly, but E3 still
    selects'. ARC-007 STRICT ('E3 introduces ALL weighting') means E3 is the
    only place that bias may legitimately be consumed -- and E3 cannot see an
    action-object at all. Pinning the count at zero makes the gap explicit;
    when it is closed, delete this test rather than bumping the number."""
    root = pathlib.Path(__file__).resolve().parents[2]
    src = (root / "ree_core" / "predictors" / "e3_selector.py").read_text()
    assert "action_object" not in src
    assert "action_bias" not in src


def test_action_objects_have_no_consumer_outside_producer_and_hippocampus():
    """Corpus-level pin: only e2_fast.py (producer) and hippocampal/module.py
    (CEM refit + storage copies) mention action_objects anywhere in ree_core.
    A new file appearing here means a consumer was added -- re-adjudicate."""
    root = pathlib.Path(__file__).resolve().parents[2] / "ree_core"
    hits = sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*.py")
        if "action_objects" in p.read_text()
    )
    assert hits == [
        "hippocampal/module.py",
        "predictors/e2_fast.py",
    ], f"unexpected action_objects consumer(s): {hits}"
