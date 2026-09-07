## SD-091 / MECH-481: Coalition/Topology Control Substrate -- steps 1-6 of 7 IMPLEMENTED (2026-08-03)
- SD-091: control_plane.coalition_topology_control -- steps 1-3 (module) landed 2026-08-02
  (chip `chip-20260802-sd091-implement-mvp`); steps 4-5 (consumer-site wiring +
  `REEAgent.select_action` integration) and step 6 (live-tick smoke test) landed 2026-08-03
  (chip `chip-20260802-sd091-live-wiring`). Step 7 (`/queue-experiment` the MECH-481 4-arm
  falsifier) is the one remaining step -- see "What remains" below.
  Module (unchanged from 2026-08-02): `ree_core/claustrum/` (`control_demand.py` --
  `ControlDemandType`; `coalition_templates.py` -- `COALITION_TEMPLATES`;
  `coalition_controller.py` -- `CoalitionController` / `CoalitionControllerConfig` /
  `CoalitionState`, plus a new `CoalitionController.reset()` added this pass, mirroring
  `SalienceCoordinator.reset()`/`BetaGate.reset()` -- needed only once the module was wired into
  `REEAgent.reset()`, not while it was self-contained).
  Config (new this pass, 3-site `REEConfig`/`from_dims()` pattern): `use_coalition_controller`
  (default `False`, bit-identical off), `coalition_types_enabled` (default
  `("sensory_resample", "provenance_check")`), `coalition_max_duration_ticks` (default 50),
  `coalition_channel_gain_scale` (default 1.0).
  `REEAgent.__init__`: `self.coalition: Optional[CoalitionController]` instantiated alongside
  (not inside) `self.salience`, gated on `use_coalition_controller`. Reads `operating_mode`
  one-directionally -- never writes into `SalienceCoordinator` (Section 3 requirement; the
  module still does not import `ree_core.cingulate` at all).
  `REEAgent.reset()`: `self.coalition.reset()` clears active coalitions on episode boundary.
  `REEAgent.select_action`: `self.coalition.tick(int(self._step_count))` runs immediately after
  the `self.salience.tick(...)` block closes (independent of whether `SalienceCoordinator` itself
  is enabled), before any consumer site below resolves its effective gate for this tick.
  Consumer sites wired (8 named targets across both MVP templates, `agent.py`):
    - `e1_sensory_encoder` (`sense()`, SD-016/MECH-152 `terrain_weight` -- the literal
      "E1 -> E3 precision modulation" signal) -- `write_gate()` multiplier.
    - `e2_fast_forward_model` (`select_action()`'s `cur_summaries`, gated only on the branch
      actually sourced from `e2.world_forward()`, MECH-314a -- not the proposer-summary
      fallback chain) -- `write_gate()` multiplier.
    - `e3_candidate_count` (`_e3_tick()`'s `_dgpe_num_candidates`, composed on top of the SD-061
      DGPE widening) -- `channel_gain()` multiplier, the literal precedent
      `coalition_templates.py`'s own docstring names ("temporarily raise candidate_count
      ceiling"); unlike `write_gate()` this is NOT capped at 1.0.
    - `hippocampal_anchor_set` (`sense()`'s `sd039_payload.wanting_strength`, scoped to the
      motivational-payload strength this tick's anchor write carries -- does not touch
      `AnchorSet`'s hysteresis/dual-trace internals) -- `write_gate()` multiplier.
    - `hippocampal_persistence_appraisal` (`_e3_tick()`'s `_persistence_appraisal.control_efficacy`,
      MECH-340) -- `write_gate()` multiplier.
    - `e3_commitment_monitor` + `motor_commitment` (both target the SAME `select_action()`
      MECH-090 R-c `_readiness_margin`, composed as a product before
      `should_admit_elevation()`/`is_above_floor()` consume it -- matches
      `CoalitionController.write_gate()`'s own multi-coalition composition idiom, applied here
      across two distinct target names at one site).
    - `hippocampal_write_consolidation` (`sense()`'s per-tick `context_memory.write(obs_state)`
      call, MECH-269b writepath -- attenuates the written state's magnitude rather than
      skipping the write, keeping the gate continuous).
  Every site: read `float(self.coalition.write_gate(name))`/`channel_gain(name)` only when
  `self.coalition is not None`, multiplied into the existing scalar/tensor at that point.
  `write_gate() <= 1.0` always (attenuation-only, `CoalitionState.__post_init__` clamping), so no
  site can ever be RAISED above its uncoalitioned value -- only `channel_gain()` (the
  `e3_candidate_count` axis) can exceed 1.0, per the doc's own parametric-side-effect carve-out.
  Data flow: upstream confidence/coherence signal (currently: test-harness / future
  MECH-481-battery driver) -> `request_coalition(demand_type, tick)` -> `CoalitionState` ->
  `coalition.tick()` each SELECT step -> the 8 named consumer sites above.
  Backward compatible: `use_coalition_controller=False` (default) -> `self.coalition is None` ->
  every new call site's `if self.coalition is not None:` guard is never entered -> byte-for-byte
  identical to pre-2026-08-03. Also verified `use_coalition_controller=True` with no active
  coalition requested is bit-identical to `False` (`write_gate()`/`channel_gain()` return exact
  1.0, and `1.0 * x == x` exactly in IEEE754) -- confirmed via a live 8-tick `act_with_split_obs`
  action-sequence comparison (contract W2), not just by inspecting the gate formula.
  Guardrails (unchanged from 2026-08-02, now additionally exercised through the live wiring):
  attenuation-only composition; sparse-only templates; one-directional mode isolation; Gamma_t
  persistence real/configurable; provisional biological-mapping language.
  Not a learning module (unchanged) -- the new wiring reads existing tensors/floats and
  multiplies by a gate; no new `nn.Module`, no parameters, no phased training. MECH-094 still
  N/A for `CoalitionController` itself (no memory write); the `hippocampal_write_consolidation`
  and `hippocampal_anchor_set` sites it now gates are pre-existing waking-only call sites
  (`sense()`, MECH-094-gated already) and this pass adds an attenuation multiplier to them, not a
  new write path.
  Contracts: `tests/contracts/test_claustrum_coalition_controller.py` (13, unchanged, module-only)
  + `tests/contracts/test_sd091_coalition_controller_wiring.py` (9, new, this pass) --
  default-OFF (W1); bit-identical enabled-but-inactive via live action-sequence comparison (W2);
  `reset()` clears active coalitions (W3); per-template target isolation, i.e. the "no global
  broadcast" guardrail restated at the wiring layer -- every named target reads a genuinely
  distinct gate and off-template targets stay at the 1.0/1.0 baseline (W4/W5); the doc's own
  step-6 live-tick smoke test, codified and passing through the real agent loop for both
  templates (W6); the BetaGate guardrail's required contract test, now exercised via the actual
  `_readiness_margin` composition rather than only `CoalitionState` arithmetic in isolation,
  swept over the full recruit/suppress weight grid (W7); `coalition.tick()` firing on the correct
  (E3-tick-rate, not raw-step-rate) cadence and dissolving on schedule (W8).
  What remains: step 7, `/queue-experiment` the MECH-481 4-arm falsifier (Arm 1 monitoring-only,
  Arm 2 parametric-only, Arm 3 untyped/undifferentiated coalition, Arm 4 typed coalition per
  this landing) -- the doc's own falsification signatures (Section "Falsification signatures")
  are the acceptance criteria. Also register-only, no V3 template yet: the other 8
  `ControlDemandType` taxonomy members (see `control_demand.py` / doc Section 2 triage).
  See SD-091, MECH-481, `REE_assembly/docs/architecture/sd_091_coalition_topology_control.md`,
  ARC-005, MECH-004, MECH-261 (`ree_core/cingulate/salience_coordinator.py`, the pattern this
  wires alongside).
