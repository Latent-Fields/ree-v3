## SD-016 Path 1: ContextMemory Diversification Loss (2026-04-25)
- SD-016 Path 1: e1.context_memory_diversification_loss -- IMPLEMENTED 2026-04-25.
  Module: ree_core/predictors/e1_deep.py (ContextMemory.compute_diversification_loss),
  ree_core/agent.py (REEAgent.compute_prediction_loss), ree_core/utils/config.py.
  EXQ-418d FAILed across all 4 write-path arms with attn_entropy_mean ~2.76 (uniform
  reference 2.7726) and bimodal seed pattern (seed 42 ~0.46 div, seeds 43/44 collapse
  <1e-4). Diagnosis: no gradient pressure for slot diversification -- read-side
  gradient through cue_terrain_loss + cue_action_loss alone cannot differentiate
  slots, and writes-only path is luck-dependent on init symmetry breaking.
  Path 1 substrate: explicit auxiliary diversification loss on ContextMemory.memory:
  mean squared off-diagonal cosine similarity over normalized slot vectors.
  ContextMemory.compute_diversification_loss() method added; weighted loss term added
  in REEAgent.compute_prediction_loss.
  Config: new sd016_diversification_weight float wired through E1Config + REEConfig
  + REEConfig.from_dims (default 0.0; backward compatible).
  Validation: V3-EXQ-418e 4-arm ablation queued (A0_off baseline, A1_writes_only
  replicates 418d, A2_div_only tests div alone, A3_writes_plus_div tests bootstrap;
  supersedes V3-EXQ-418d). Smoke verified: slot_div climbs 0.2->0.5->1.0 across arms;
  wiring confirmed.
  See SD-016, MECH-150, MECH-151, MECH-152, ARC-041, EXP-0155.
  Design doc: REE_assembly/docs/architecture/sd_016_writepath_v3_diversification_loss.md
