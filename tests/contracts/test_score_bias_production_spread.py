"""E3 production score_bias spread assertion.

chip-20260822-scorebias-production-spread-assertion, closing the gap named in
REE_assembly/evidence/planning/authority_to_action_stage_trace_probe_2026-08-22.md
(the "authority-to-action stage trace" probe, 2026-08-22):

    Nothing asserts that the PRODUCTION path -- REEAgent.select_action's
    composed dACC / lateral_pfc / ofc / mech295 / gated_policy score_bias --
    ever produces a non-degenerate score_bias. test_mech451_finer_channel_
    gating.py passes score_bias in BY HAND (a hand-built
    torch.tensor([0.0, -0.3, 0.2, -0.1])) and would keep passing even if the
    production producer were entirely dead. test_flag_inertness.py checks
    that FLAGS change observable behaviour, a different question from
    whether the score_bias CHANNEL itself carries cross-candidate spread.

SPREAD, NOT MAGNITUDE. This reuses e3_selector.authority_spread_ratio /
authority_ratio_is_competitive -- the SAME primitives the CEM elite-stage
readiness gate uses (test_cem_modulatory_authority.py) -- rather than
inventing a second notion of competitiveness. A score_bias can have large
per-candidate MAGNITUDE while being a uniform scalar (zero spread), which is
invariant under argmin/softmax however large it is. That is exactly the
already-catalogued F-C2 finding (design_implementation_audit_2026-07-09):
`dacc_foraging_weight` "adds a uniform scalar to every candidate -> an
argmin/softmax(-cost) selector is invariant to it (dead-by-construction on
the E3 leg)".

THE CONFIGURATION: use_dacc + use_affective_harm_stream is the one conjunction
the probe confirmed is both (a) live in the real experiment corpus (98 of
1361 v3_exq_* drivers set both) and (b) satisfies agent.py's dACC->E3 gate
(`self.dacc is not None and z_harm_a is not None`, agent.py ~line 6779) --
use_dacc ALONE is insufficient because z_harm_a requires
use_affective_harm_stream (SD-011); the adapter is constructed either way so
an ablation on use_dacc alone looks wired while never actually running.

CURRENT VERDICT (measured 2026-08-22, untrained agent, this exact config):
scores is BIT-IDENTICAL to raw_scores on every fresh E3 tick --
modulatory_authority_ratio reads exactly 0.0. xfail(strict=True) below
documents this as a known, already-catalogued defect (F-C2) rather than
silently asserting nothing: the suite stays green now, and the day the dACC
adapter is trained/fixed to emit a genuinely divergent per-candidate bias,
this test XPASSes, the strict marker fails the run, and whoever fixed it is
forced to delete the marker (the regression-latch idiom documented in
tests/test_flag_inertness.py's module docstring).

ADDITIONAL DILIGENCE (not asserted here, recorded so it is not silently
re-discovered): the probe's own caveat is "measured on an UNTRAINED adapter;
a trained head might produce spread, and that has not been shown either
way". Two more avenues were tried before concluding this is the honest
level to assert at:

  * use_gated_policy=True alone: gated_policy's two heads consume a PER-
    CANDIDATE world-summary feature (unlike dACC's scalar bundle), so it
    looked like the more promising untrained channel. Empirically its
    candidate_features carry EXACTLY ZERO cross-candidate spread at the
    default operating point -- traced to agent.py's
    `_candidate_world_summaries()` docstring, which independently confirms
    this as the ALREADY-KNOWN V3-EXQ-614e root cause ("the proposer
    first-step z_world collapses to cross-candidate spread ~0 under
    monostrategy ... every E3-side bias channel sees a class-uniform
    candidate pool").
  * Re-sourcing those summaries via `candidate_summary_source=
    "e2_world_forward"` (the documented GAP-A fix for exactly that root
    cause), PLUS a genuine 200-step Adam/InfoNCE training run of
    `agent.e2` using the substrate's own SD-056 contrastive-loss training
    recipe (tests/contracts/test_sd_056_e2_action_contrastive.py's C7
    idiom, not a synthetic score_bias injection): candidate-feature
    pairwise distance rose 0.15 -> 0.45 (training genuinely took), but the
    downstream gated_policy heads are THEMSELVES untrained random-init
    MLPs, and the resulting score_bias's modulatory_authority_ratio only
    reached ~1.5e-3 -- ~65x short of the 0.1 competitive floor. Training
    the heads too would mean training the full modulatory pipeline via the
    real RL objective across many episodes, which is out of scope for a
    fast, deterministic contract test.

Net: no reachable, untrained-or-lightly-trained configuration was found to
produce a competitive production score_bias. The dacc+affective_harm_stream
case is asserted here because it is the one confirmed LIVE in the corpus;
see WORKSPACE_STATE.md for this chip's closing note on the broader finding.
"""

from __future__ import annotations

import pytest
import torch

from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.predictors.e3_selector import (
    AUTHORITY_COMPETITIVE_RATIO_FLOOR_DEFAULT,
    authority_ratio_is_competitive,
)
from experiments._harness import StepHarness, StepHooks
from experiments._lib.fresh_select import FreshSelectProbe

SEED = 7
# >= 8 fresh E3 ticks at the default heartbeat.e3_steps_per_tick=10 cadence.
STEPS = 80
MIN_FRESH_TICKS = 3


def _build_dacc_affective_harm_agent(seed: int = SEED):
    """The one confirmed-live production conjunction (see module docstring)."""
    torch.manual_seed(seed)
    env = CausalGridWorldV2(
        seed=seed, size=5, num_hazards=1, num_resources=1, use_proxy_fields=True,
    )
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        self_dim=16,
        world_dim=16,
        use_dacc=True,
        use_affective_harm_stream=True,
    )
    # from_dims-swallow-kwarg trap (see authority_trace_probe.py): confirm
    # both flags actually landed before trusting the rollout below.
    # use_affective_harm_stream lives on the nested LatentConfig, not the
    # top-level REEConfig -- from_dims() plumbs it through to cfg.latent.
    assert cfg.use_dacc is True
    assert cfg.latent.use_affective_harm_stream is True
    agent = REEAgent(cfg)
    agent.reset()
    return env, agent, cfg


def _fresh_tick_ratios(env, agent, seed: int, steps: int) -> list:
    """Drive `steps` env ticks through the canonical StepHarness and collect
    the standing `modulatory_authority_ratio` diagnostic
    (authority_spread_ratio(bias_detached, raw_scores), e3_selector.py) on
    every FRESH E3 selection. Gated on experiments._lib.fresh_select so a
    held (non-E3) tick's replicated last_score_diagnostics is never counted
    -- the hold-weighted-readout defect (699 / 689d autopsies)."""
    probe = FreshSelectProbe("scorebias_prod_spread")

    def _on_sense(**kw):
        probe.mark_stale(agent)

    harness = StepHarness(
        agent, env, train_mode=False, hooks=StepHooks(on_sense=_on_sense), seed=seed,
    )
    _flat, obs_dict = env.reset()
    agent.reset()
    harness.reset()
    ratios: list = []
    with torch.no_grad():
        for _ in range(steps):
            result = harness.step(obs_dict)
            obs_dict = result.next_obs_dict
            if probe.is_fresh(agent):
                diag = agent.e3.last_score_diagnostics
                ratio = diag.get("modulatory_authority_ratio")
                if ratio is not None:
                    ratios.append(float(ratio))
            if result.done:
                _flat, obs_dict = env.reset()
                agent.reset()
                harness.reset()
    return ratios


@pytest.mark.xfail(
    strict=True,
    reason=(
        "F-C2 (design_implementation_audit_2026-07-09), extended 2026-08-22: "
        "the dACC->E3 adapter's emitted score_bias is a uniform scalar (zero "
        "cross-candidate spread), so it is invariant under E3's argmin/"
        "softmax regardless of magnitude -- dead-by-construction on the E3 "
        "leg. Measured on an UNTRAINED adapter; see "
        "REE_assembly/evidence/planning/"
        "authority_to_action_stage_trace_probe_2026-08-22.md. If this test "
        "XPASSes, the production path finally emits a competitive "
        "score_bias -- delete this marker, do not loosen the threshold."
    ),
)
def test_production_score_bias_reaches_competitive_spread():
    """THE assertion this chip exists to add: in a configuration that
    satisfies the dACC->E3 producer's own preconditions, the composed
    score_bias reaching E3.select() must have a cross-candidate spread that
    is COMPETITIVE with the dominant (raw_scores) term -- not merely
    nonzero. See module docstring for why SPREAD (not magnitude) is the
    right bar and why neither existing test already covers this."""
    env, agent, cfg = _build_dacc_affective_harm_agent()
    ratios = _fresh_tick_ratios(env, agent, SEED, STEPS)

    assert len(ratios) >= MIN_FRESH_TICKS, (
        f"only {len(ratios)} fresh E3 ticks captured in {STEPS} steps -- "
        "widen STEPS; a near-zero fresh-tick count cannot support a verdict "
        "either way (see experiments/_lib/fresh_select.py)"
    )

    floor = float(cfg.authority_competitive_ratio_floor)
    assert floor == AUTHORITY_COMPETITIVE_RATIO_FLOOR_DEFAULT

    best_ratio = max(ratios)
    assert authority_ratio_is_competitive(best_ratio, floor=floor), (
        f"production score_bias never reached a competitive spread over "
        f"{len(ratios)} fresh E3 ticks (best modulatory_authority_ratio="
        f"{best_ratio!r}, floor={floor}); all ratios={ratios!r}"
    )
