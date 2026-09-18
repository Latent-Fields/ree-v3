## MECH-131 terrain_prior anticipatory-residue channel lesion knob (2026-09-18)

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
| 2 | `module.py` `_score_trajectory` -- `residue_field.evaluate_trajectory` | scores every CEM sample; drives `torch.argsort(scores)[:num_elite]` elite selection and the distribution refit | **no** |

Read 2 is the DOMINANT one and is *pure residue* in a default `HippocampalConfig`
(`wanting_weight=0.0`, `curiosity_weight=0.0`, `mode_value_weight={}`);
V3-EXQ-931 measured the modulatory cross-candidate spread at ~0.37% of terrain's.
Measured with this knob OFF (contract C6, hub, 2026-09-18): the CEM terrain score
over 32 proposed candidates still ranges 20.83-21.72, spread 0.892 -- i.e. the
elite argsort retains real residue discrimination. So an experiment whose lesion
arm sets only this flag leaves the dominant anticipatory pathway intact.

A field-level gate is ruled out as the fix: read 2 shares
`ResidueField.evaluate_trajectory` with the POST-HOC path
(`E3.compute_residue_cost`), which MECH-131 requires to stay ON. Each read must
therefore be gated at its own call site, and a complete lesion needs a SECOND
call-site knob on `_score_trajectory`. That second knob is **not** built: it was
not part of the ratified pre-flight for this landing and is an open design
question (see the decision chip named in the chip ledger for
`chip-proposal-exp-0878-paced`).

**Validation experiment: none queued yet** -- deliberately. The MECH-131
falsifier is blocked on the lesion-completeness decision above, because whether
the lesion arm is one channel or two changes what the experiment measures. The
knob's own liveness and scope are pinned by contracts instead:
`tests/contracts/test_mech131_terrain_prior_residue_channel.py` (C1 reachability,
C2 default bit-identical, C3 liveness, C4 storage untouched, C5 post-hoc scorer
untouched, C6 scope pin). 6/6 passed on the hub 2026-09-18.

See MECH-131, ARC-013 (residue as curvature over L-space), INV-038, MECH-126.
