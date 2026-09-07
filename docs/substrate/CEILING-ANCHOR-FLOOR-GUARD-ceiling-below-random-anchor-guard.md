## Ceiling-below-random-anchor guard + standing lint (competence-objective autopsy 734/737b/742a) (2026-08-01)

Instrumentation from the CONFIRMED, user-adjudicated competence-objective cluster autopsy
(`REE_assembly/evidence/planning/failure_autopsy_competence-objective-cluster-734-737b-742a_2026-07-22.md`
sec 5.2). A trained learner scoring BELOW its own declared `random_walk` anchor is NOT at a
capacity/representation ceiling -- under a real ceiling it asymptotes toward the anchor from
below; a learner WORSE than random on an oracle-achievable env is optimising a different
objective than the scored DV (the 734 survival-vs-forage inversion: PPO survived 175.0 steps vs
the oracle's 20.4 while foraging 17x less). Emitting a `substrate_ceiling` /
`learner_or_observability_ceiling` verdict on such a run mislabels objective-misspecification as
a substrate limit -- the below-random score was in every manifest and consumed by nothing.

- **Runtime guard:** `experiments/_lib/anchor_floor_guard.py` --
  `refuse_ceiling_below_random(label, learner_scores, random_anchor, ...) -> (safe_label, record)`.
  Downgrades a ceiling-class label resting on a learner strictly below the random anchor to
  `substrate_not_ready_requeue` and returns a manifest-embeddable refusal record (names the
  sub-random arms, deficits, anchor). `strict=True` raises `CeilingBelowRandomAnchorError`. This
  is ORTHOGONAL to `zworld_encoder_guard`: that fires on the INPUT side (was z_world trained);
  this fires on the OUTPUT side (is the measured DV below random), catching the 737b case where
  a genuinely trained encoder (guard GREEN) still scored 0.233 vs random 0.933.
- **Reference consumer:** `v3_exq_734_...` wires it at the D3 (hazard-free) rung after the
  self-route label is computed; the refusal record lands at
  `interpretation.ceiling_route_refusal`. (Same landing also flipped 734's stale manifest
  `diagnostics.zworld_encoder_guard.scope` from `detection_only` to `gating` -- 734 has refused
  and skipped frozen-encoder arms since `26ff282`; 728's gating lives in the canonical 728b,
  pinned by `test_zworld_p0_adoption_reaches_every_driver` C4/C5.)
- **Standing lint:** `validate_experiments.py --checks ceiling_route_anchor_floor` WARNs
  (advisory in both modes; never hardens -- below-random is not statically decidable) when a
  diagnostic/baseline script self-routes to a ceiling label (an EMITTED `label`/grid-key, not a
  prose mention) AND declares a `floor="random_walk"` anchor BUT never wires the guard. SILENT
  on the current corpus (0 fires; 734 wires it, 728/728b only mention ceilings in prose) -- it
  is a forward gate for future drivers. Exempt with `CEILING_ANCHOR_FLOOR_EXEMPT = "<reason>"`
  when the ceiling route cannot rest on a sub-random learner (the 737/742
  recorded_preconditions case, where a readout-side control survives the floor).
- **Contracts:** `tests/contracts/test_ceiling_route_anchor_floor.py` (helper on the autopsy's
  own D3 numbers; lint fires/silent/exempt; corpus-silence; CLI-reachability + WARN-only).
