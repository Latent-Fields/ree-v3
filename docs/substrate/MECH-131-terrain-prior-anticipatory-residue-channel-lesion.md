## MECH-131 anticipatory-residue channel lesion knobs, CH1 + CH2 (2026-09-18)

- MECH-131: `vmPFC.residue_activation` -- INSTRUMENT LANDED 2026-09-18.
  `ree_core/hippocampal/module.py` (`HippocampalModule._get_terrain_action_object_mean`).
  Config: `HippocampalConfig.terrain_prior_residue_channel_enabled`
  (default `True` = current behaviour, bit-identical; set `False` to lesion).
  Reachable from drivers via `REEConfig.from_dims(...)` -- all three sites wired
  (dataclass field, `from_dims` signature, post-`cls()` re-apply).
  Data flow: `ResidueField.evaluate(z_world)` -> `residue_val` -> [THIS GATE] ->
  `torch.cat([z_world, e1_prior, residue_val[, benefit_val]])` -> `terrain_prior`
  -> `action_object_mean` -> the initial CEM proposal distribution.
  Backward compatible: default-on preserves behaviour exactly; zeroing is
  out-of-place and keeps `terrain_input_dim` unchanged.
  Phased training required: no (no new learned parameters).
  MECH-094: not applicable (no memory write; a read-path gate only).

**What it is for.** MECH-131 asserts that a vmPFC-analog must ACTIVATE stored
aversive residue as an anticipatory forward-biasing signal BEFORE candidate
generation, and that residue correctly STORED but not so activated fails to
suppress harm-associated trajectory re-selection. Testing that needs a lesion
arm -- activation OFF, storage ON. No such arm existed: `rho_residue`
(`config.py`, default 0.5) zeroes only the POST-HOC scorer
(`E3.compute_residue_cost`), and `terrain_prior`'s input width is structural, so
the channel could not be switched off from config.

**Why out-of-place.** `ResidueField.evaluate` may return a tensor sharing storage
with field state; an in-place `.zero_()` would corrupt the field itself rather
than just this read. `torch.zeros_like` is used instead.

**SCOPE -- this is NOT a complete anticipatory lesion. Read this before
designing an experiment against it.** V3 has **two** live anticipatory
(pre-candidate-list) residue reads, and this knob gates only the first:

| # | Site | Role | Gated by this knob? |
|---|------|------|---------------------|
| 1 | `module.py` `_get_terrain_action_object_mean` -- `residue_field.evaluate` -> `terrain_prior` | biases the INITIAL action-object proposal mean | **yes** |
| 2 | `module.py` `_score_trajectory` -- `residue_field.evaluate_trajectory` | scores every CEM sample; drives `torch.argsort(scores)[:num_elite]` elite selection and the distribution refit | **yes, by `score_trajectory_residue_terrain_enabled`** (added later the same day, user decision OPTION C) |

Read 2 is the DOMINANT one and is *pure residue* in a default `HippocampalConfig`
(`wanting_weight=0.0`, `curiosity_weight=0.0`, `mode_value_weight={}`);
V3-EXQ-931 measured the modulatory cross-candidate spread at ~0.37% of terrain's.
Measured with this knob OFF (contract C6, hub, 2026-09-18): the CEM terrain score
over 32 proposed candidates still ranges 20.83-21.72, spread 0.892 -- i.e. the
elite argsort retains real residue discrimination. So an experiment whose lesion
arm sets only this flag leaves the dominant anticipatory pathway intact.

A field-level gate is ruled out as the fix: read 2 shares
`ResidueField.evaluate_trajectory` with the POST-HOC path
(`E3.compute_residue_cost`), which MECH-131 requires to stay ON. Each read is
therefore gated at its own call site, by its own knob.

## The second knob (CH2), added under user decision OPTION C

    HippocampalConfig.score_trajectory_residue_terrain_enabled   (default True)

Gates `_score_trajectory`'s residue terrain score on **both** of its reads -- the
z_world path and the pre-SD-005 z_self fallback. Gating only the first would leave
residue still driving elite selection for any trajectory carrying no world_states:
a partial lesion with no signature in the config.

Zeroed out-of-place, and applied BEFORE the terrain/modulatory decomposition so
`score == terrain + modulatory` still holds exactly -- the lesion makes the terrain
contribution genuinely zero rather than hiding it from the accounting. The tensor is
on the autograd path the SD-055 differentiable-CEM route reads, so `.zero_()` would
be wrong here for a second, independent reason.

**Why zeroing this score is not a confound.** At default `HippocampalConfig`
(`wanting_weight=0.0`, `curiosity_weight=0.0`, `mode_value_weight={}`)
`_score_trajectory` IS the residue terrain score and nothing else, so zeroing it
removes exactly the residue contribution and no other signal. CEM still refits; it
simply refits toward an unranked elite subset.

### Measured (hub, 2026-09-18)

| Configuration | proposal-mean shift | CEM score spread | post-hoc cost |
|---|---|---|---|
| intact (both True) | -- | 0.892 | live |
| CH1 lesion only | max|delta| 0.449 | 0.892 (**still residue-driven**) | live |
| CH1 + CH2 lesion | max|delta| 0.449 | **exactly 0.0** | 21.62 (**live**) |

The middle row is why one knob was not enough. The bottom row is the
"stored but not activated" arm: no anticipatory read survives, while storage totals
are identical to intact and the post-hoc scorer still discriminates.

**Set BOTH to False for the complete anticipatory lesion.** Either alone is partial.

## Contracts

`tests/contracts/test_mech131_terrain_prior_residue_channel.py` -- C1 CH1 reachability,
C2 CH1 default bit-identical, C3 CH1 liveness, C4 storage untouched, C5 post-hoc scorer
untouched, C6 CH1-alone scope pin, C7 CH2 reachability + spread collapse, C8 the COMPLETE
two-channel lesion (with storage and post-hoc both pinned live), C9 CH2 default
bit-identical. 10 tests, all green on the hub 2026-09-18. Both flags are registered in
`tests/test_flag_inertness.PROBED`.

## Validation experiment

**None queued yet.** The 3-arm design (intact / CH1-lesion / CH1+CH2-lesion) is staged at
`REE_assembly/evidence/planning/mech131_three_arm_lesion_design_staged_20260918.md`, with
arms, DVs, directions, preconditions and recording obligations all fixed from recorded
text. It is blocked on one further decision: MECH-131 has no `what_would_answer`, so the
PASS/FAIL bar -- and the coupled question of whether the run is a `diagnostic` or
governance evidence -- is not derivable from recorded text and was deliberately not
invented.

See MECH-131, ARC-013 (residue as curvature over L-space), INV-038, MECH-126.
