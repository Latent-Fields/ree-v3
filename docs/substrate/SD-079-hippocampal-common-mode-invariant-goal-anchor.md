## SD-079: hippocampal.common_mode_invariant_goal_anchor_match -- IMPLEMENTED (2026-07-22)
- SD-079: hippocampal.common_mode_invariant_goal_anchor_match — IMPLEMENTED 2026-07-22.
  SD-039 Anchor.goal_match + AnchorSet + MECH-292 GhostGoalBank,
  ree_core/hippocampal/anchor_set.py + ree_core/hippocampal/ghost_goal_bank.py.
  Config: AnchorSetConfig.goal_cue_centering (default False; set True to enable) +
  goal_cue_baseline_alpha (default 0.05). REEConfig.from_dims(goal_cue_centering=,
  goal_cue_baseline_alpha=) plumbed to the nested AnchorSetConfig.
  Data flow: z_goal cue -> AnchorSet slow-EMA baseline (advanced on BOTH the write
  path, write_anchor with a goal_payload, and the read paths query_by_goal_match /
  GhostGoalBank.rank) -> Anchor.goal_match(z_goal, baseline=) on centered residuals.
  Snapshots stored RAW, centered at comparison time. baseline=None is the identity,
  so the pre-SD-079 call form is bit-identical.
  Backward compatible: disabled by default; OFF allocates no baseline.
  Why: z_goal is an EMA attractor pulled toward z_world and carries the SD-008 offset
  MORE strongly than its source (pairwise cosine min 0.9878 vs 0.9767). Measured over
  a 24-anchor pool: goal_match spread 0.0111 raw vs 0.9709 centered; MECH-292's
  goal_match_floor excluded 0/24; MECH-339's outshining gate sat at EXACTLY 0.0 for
  every anchor, i.e. its context channel was unconditionally dead whenever enabled.
  ALPHA IS 0.05, NOT SD-066/077/078's 0.02 — do not harmonise it. z_goal is an
  integrator so its common mode DRIFTS and 0.02 lags it (measured spread 0.0942 /
  0.1508 / 0.3319 at 0.02 vs 0.9995+ at 0.05 over seeds 101/202/303); 0.2+ over-tracks
  (14-18 of 20 anchors driven below the floor).
  The WRITE-path advance is load-bearing: advancing on reads alone seeds the baseline
  FROM THE QUERY, zeroing every residual (measured — the first ON arm scored 0.0000
  across all 20 anchors). Contract C6 pins it.
  Phased training required: no.
  Contracts: tests/contracts/test_sd_079_centered_goal_anchor_match.py (C0 fixture
  geometry, C1 default-off bit-identity, C2 centering widens the match range, C3 both
  downstream ABSOLUTE gates unpin, C4 lazy seed + MECH-094, C5 raw-snapshot self-match,
  C6 write path advances the baseline, C7 alpha not in the over-tracking regime — only
  that bound is fixture-assertable, see the test's scope note).
  See SD-066, SD-077, SD-078, SD-008, SD-070, SD-039, MECH-292, MECH-339, MECH-293, MECH-340.
