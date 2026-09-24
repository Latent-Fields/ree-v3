## SD-CM-LIVETAP z_world scale anchor: e1.context_memory.write_tagger_scale_invariant + live_scale_anchor_margin (2026-09-24)

- SD-CM-LIVETAP (scale anchor): e1.context_memory live-tap z_world scale anchor -- IMPLEMENTED 2026-09-24.
  `ContextMemory` / `_BlockUnitNorm` in `ree_core/predictors/e1_deep.py`.
  Config (NOT E1Config fields / from_dims kwargs yet -- same deferral as the tap flag; from_dims
  silently swallows unknown kwargs, so set them on `cfg.e1` directly):
  - `cfg.e1.contextmemory_write_tagger_scale_invariant` (default False) -- option (b): a
    parameter-free per-block unit-norm stage ([z_self | z_world], split at self_dim) prepended to
    `write_addr_tagger`. Requires `contextmemory_write_selection="gumbel_learned"` (raises otherwise).
    Every tagger consumer -- `write()`, `compute_write_addressing_loss[_live]`, and a driver's own
    H3 read coupling that calls `write_addr_tagger` directly -- sees a scale-invariant input.
  - `cfg.e1.contextmemory_write_live_scale_anchor_margin` (default 0.0 = off; 2.0 validated) --
    option (c): hinge `sum_b mean(relu(||x_b|| - margin*ref_b)^2) / ref_b^2` on the tapped
    states, `ref` = per-block mean norm of the FIRST batch `take_live_write_states()` returns,
    frozen. Requires the tap (> 0). The caller adds `cm.live_scale_anchor_loss(states)`;
    `compute_write_addressing_loss_live()` adds it itself. `take_live_write_states()` RAISES if the
    previous batch's anchor term was never computed (`clear_live_write_states()` discharges it).
  Why: with the tap as the encoder's only gradient source the write-side losses were met by
  inflating z_world (pilots 2026-09-24: H3_LIVE 0.37 -> ~140, DIV_LIVE -> 6.3, DETACHED ~0.41;
  SD-070 warmup still 31.8x / 96.6x). Orchestrator decision (orchestrate-20260924-0808, chip
  chip-20260924-ctxmem-livetap-anchor-design-decision): build (b); add (c) only if (b) alone fails.
  Measured (contract fixture, 200 tap updates, Adam 1e-3, DIV objective, seed 42): DETACHED 0.41;
  LIVE unanchored 212 (~500x); LIVE (b) 2.6 end / 5.7 peak -- (b) ALONE FAILS; LIVE (b)+(c) 0.30 end /
  0.59 peak. H3-style objective: unanchored 1.04, (b) 0.78, (b)+(c) 0.39 end / 0.44 peak. (c) alone
  also held in this fixture (0.25 end / 0.64 peak) -- (b) is kept because it removes the radial
  gradient rather than opposing it; (c) absorbs the residual Adam drift.
  Use BOTH flags together for tap-phase validation.
  Data flow: agent.sense() -> record_live_write_state([z_self|z_world], live) -> take_live_write_states()
  -> write_addr_tagger(_BlockUnitNorm(x)) -> write-side loss (+ live_scale_anchor_loss) -> latent_stack.
  Backward compatible: both default off; tagger construction, RNG draws and state_dict keys
  unchanged when off (pinned). When (b) is on, tagger keys shift to write_addr_tagger.1/.3.
  Phased training required: the tap phase is the validation's own design (chip-20260922-ctxmem-livetap-validation-exp).
  Contract: `tests/contracts/test_contextmemory_live_tap_scale_anchor.py` (load-bearing test measures
  LIVE z_world norm within 2x of DETACHED over 200 tap updates, with an unanchored canary arm).
  See SD-016, SD-070, contextmemory_write_content_discrimination, V3-EXQ-972a.
