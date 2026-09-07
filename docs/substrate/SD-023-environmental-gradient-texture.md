## SD-023: Environmental Gradient Texture (2026-04-09)
- SD-023: environment.gradient_texture -- IMPLEMENTED 2026-04-09.
  CausalGridWorldV2 (ree_core/environment/causal_grid_world.py): two new landmark object
  types (Landmark A: navigation anchor, Landmark B: predictive resource cue) each emitting
  a 25-dim gradient field view in world_obs.
  New __init__ params: n_landmarks_a (int, default 0), n_landmarks_b (int, default 0),
  landmark_a_sigma (float, default 1.5), landmark_a_scale (float, default 1.0),
  landmark_b_sigma (float, default 2.5), landmark_b_scale (float, default 0.6),
  landmark_b_resource_bias (float, default 0.7).
  Data flow: reset() -> _place_random_landmarks/_place_biased_near_resources ->
  _compute_landmark_field -> precomputed static Gaussian fields per episode ->
  _get_observation_dict() extracts 5x5 field views -> appended to world_state ->
  world_obs_dim: 250 -> 300 when n_landmarks_a>0 or n_landmarks_b>0.
  obs_dict keys: "landmark_a_field_view" [25], "landmark_b_field_view" [25].
  Backward compatible: n_landmarks_a=0, n_landmarks_b=0 by default; world_obs_dim=250;
  all existing experiments unaffected. Landmarks are gradient-only (no grid entity type).
  Biological basis: all objects in natural environments have a detectable presence
  (olfactory, acoustic, visual texture). Landmark B placed with bias near resources
  provides the predictive co-occurrence structure for MECH-216 (anticipatory wanting).
  No phased training required (env extension only, no new encoder or training target).
  MECH-094: not applicable (waking observation stream).
  Validation experiment: V3-EXQ-263b queued.
  See SD-023, MECH-216, ARC-017, MECH-096, MECH-103.
