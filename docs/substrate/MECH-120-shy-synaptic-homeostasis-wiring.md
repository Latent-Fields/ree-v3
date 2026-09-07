## MECH-120: SHY Synaptic Homeostasis Wiring (2026-04-08)
- MECH-120: sleep.sws_denoising_attractor_flattening -- WIRED 2026-04-08.
  E1DeepPredictor.shy_normalise() (e1_deep.py:283-304) was already implemented but
  not called from enter_sws_mode(). Now wired: enter_sws_mode() calls
  self.e1.shy_normalise(decay=self.config.shy_decay_rate) when shy_enabled=True.
  Config: REEConfig.shy_enabled (bool, default False), REEConfig.shy_decay_rate
  (float, default 0.85). Both wired through from_dims().
  Data flow: enter_sws_mode() -> shy_normalise(decay) -> context_memory.memory.data
  modified in-place (slot weights decayed toward slot-mean).
  Backward compatible: shy_enabled=False by default; existing experiments unaffected.
  No trainable parameters. No gradient flow (.data write). No phased training needed.
  Biological basis: Tononi & Cirelli SHY hypothesis (2006). decay=0.85 = ~15%
  reduction per cycle, consistent with SHY literature.
  MECH-094: not applicable (modifies existing weights, does not generate replay content).
  Validation experiment: EXQ-245a queued.
  See MECH-120, MECH-165 (downstream -- replay diversity requires SHY first).
