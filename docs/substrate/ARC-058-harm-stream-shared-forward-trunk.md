## ARC-058: harm_stream.shared_forward_trunk -- REGISTERED (2026-04-19)
- ARC-058: harm_stream.shared_forward_trunk -- REGISTERED 2026-04-19,
  COMPETES WITH ARC-033.
  Module: ree_core/latent/stack.py (HarmForwardTrunk, HarmForwardHead
  -- pre-existing substrate classes). Selection via shared_trunk
  constructor arg on E2HarmSForward / E2HarmAForward (see MECH-258).
  ARC-033 claim: independent per-stream forward models (separate
  ResidualHarmForward per stream). Biological reading: dorsal posterior
  insula (sensory PE) + anterior insula (affective PE) as separate
  learned substrates.
  ARC-058 claim (competing): shared HarmForwardTrunk (unsigned,
  modality-independent PE substrate) + stream-specific HarmForwardHead
  (signed, per-modality readout). Biological reading: Horing & Buchel
  2022 anterior insula encodes modality-independent unsigned PE shared
  across aversive modalities; dorsal posterior insula encodes
  modality-specific signed PE. Trunk ~ unsigned; head ~ signed.
  Same nn.Module topology, different wiring. Constructor switch arbitrates.
  Falsifiable: V3-EXQ-445 three-arm ablation measures per-stream
  forward_r2 for z_harm_s and z_harm_a + downstream dACC bundle
  usefulness under each path. If shared-trunk matches or beats
  independent with fewer parameters AND produces a useful unsigned
  PE signal, ARC-058 wins and ARC-033 is narrowed. If independence
  wins, ARC-058 is retired.
  See ARC-058, ARC-033, MECH-258, MECH-257, SD-032b.
