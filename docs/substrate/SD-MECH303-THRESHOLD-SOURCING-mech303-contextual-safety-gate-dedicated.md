## SD-MECH303-THRESHOLD-SOURCING: mech303.contextual_safety_gate.dedicated_proximity_signal -- IMPLEMENTED (2026-08-14)
- SD-MECH303-THRESHOLD-SOURCING: mech303.contextual_safety_gate.dedicated_proximity_signal --
  IMPLEMENTED 2026-08-14. ree_core/environment/causal_grid_world.py (dedicated EMA + obs
  channel), ree_core/agent.py (sense() gate selector), ree_core/utils/config.py (selector +
  threshold).
  Gives MECH-303's contextual-safety accumulate gate a DEDICATED anticipatory hazard-proximity
  signal, decoupled from SD-022's damage-sourced z_harm_a. V3-EXQ-917 found the gate cannot
  discriminate safe-vs-unsafe under damage-sourcing (AUC <=0.52, chance) at any of 18 thresholds,
  while a proximity signal reaches AUC 0.84-0.97; a single z_harm_a cannot serve both SD-022
  (wants body/context decoupled) and MECH-303 (wants body/context coupled), so this adds a second
  signal for MECH-303's gate alone -- every other z_harm_a consumer is untouched.
  Env: CausalGridWorldV2(safety_proximity_signal_enabled=False default; True emits
  obs_dict["safety_proximity_harm"], a scalar tau~20 EMA of hazard-proximity-at-agent, updated
  BEFORE the Q-080 effort injection so it is decoupled from both SD-022 and Q-080;
  safety_proximity_ema_alpha=0.05).
  Config: REEConfig.contextual_safety_gate_source ("z_harm_a" default -> unchanged z_harm_a.norm()
  gate; "proximity_signal" -> dedicated signal), contextual_safety_proximity_threshold (0.25;
  "harm absent" gate for the dedicated signal, ~0 safe / ~0.8 dense-hazard). Existing
  contextual_safety_harm_threshold (0.05) untouched.
  Data flow: hazard_at_agent -> env _safety_proximity_ema -> obs_dict["safety_proximity_harm"]
  -> agent.sense(obs_safety_proximity=...) -> MECH-303 accumulate_safety gate (proximity path;
  NO silent z_harm_a fallback when the signal is absent).
  Backward compatible: both switches default no-op; env output bit-identical OFF, gate reads
  z_harm_a by default; new sense() kwarg defaults None. Smoke (2026-08-14): safe(nh=0) mean signal
  0.00 vs unsafe(nh=8) 0.87; end-to-end proximity gate fires 150/150 safe vs 6/150 unsafe.
  Not a learning module -- no encoder, no phased training. MECH-094 N/A (accumulate retains its
  existing hypothesis_tag waking-path guard; signal is a waking env observable).
  Validation experiment: V3-EXQ-930 queued (917-style AUC sweep; acceptance AUC >=~0.84 >> chance
  0.52). See docs/architecture/sd_mech303_threshold_sourcing.md, MECH-303, SD-052, SD-011, SD-022.
