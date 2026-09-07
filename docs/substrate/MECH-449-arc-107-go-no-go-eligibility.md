## MECH-449 / ARC-107: Go/No-Go eligibility constitution (the OPPONENCY leg of the basal-ganglia E3-selector constitution; generalises MECH-260) (2026-06-21)
- MECH-449: selection.go_nogo_eligibility_constitution -- IMPLEMENTED 2026-06-21
  (substrate; MECH-449 stays candidate / substrate_conditional -- PROMOTES NOTHING;
  the ablation falsifier gates promotion, NOT the build). Component 1 (the core
  Go/No-Go opponency) of the ARC-107 BG constitution; the missing core complementing
  the landed MECH-448 (component 3, pallidal permission gate). Build gate cleared on
  BOTH faces (gate INVERTED 2026-06-21, anti-partial-instantiation): selection-face
  V3-EXQ-689f (No-Go-necessity falsifier: the live _f_eligibility_envelope admits
  high-F undesirable candidates only an active No-Go can suppress) + behavioural-face
  the 485e->485k six-autopsy lineage (demotion-alone converts the passive OFC
  discrimination signature but cannot express the active No-Go WITHDRAWAL devaluation
  requires). Second major worked application of ARC-106 (grounding synthesis 2.1:
  Kravitz 2010 D1/D2 opponency; Mink 1996 focal-go + surround-no-go; Maia & Frank 2011).
  PROBLEM: MECH-448's rank-preserving F->eligibility demotion is order-preserving over
  F, so it structurally CANNOT exclude an F-eligible-but-undesirable candidate on a
  non-F axis (safety/staleness/perseveration/low-viability) -- only an active No-Go can
  (exactly what 689f demonstrates).
  THE FIX (no-op default; bit-identical OFF): a BOUNDED Go/No-Go pressure SET housed as
  METHODS on E3TrajectorySelector (_go_nogo_eligibility_gate), the same pattern as
  _f_eligibility_envelope / _gap_scaled_commit_pick -- NO parallel module (ARC-106 G2).
  The gate runs inside the shortlist-then-modulate block AFTER the F-built eligible set
  (eligible_idx) is computed and BEFORE the within-eligible _modulatory_accum
  arbitration -- governing WHICH candidates may compete for the pallidal-like
  permission-to-commit gate:
    No-Go (suppress): drop a candidate from the eligible set when ANY bounded axis
      crosses its floor -- safety (>= gng_safety_floor), staleness (>= gng_staleness_floor),
      perseveration (recency-share >= gng_perseveration_floor), viability
      (< gng_viability_floor). ORTHOGONAL to F-rank; a No-Go'd candidate is removed
      from the eligible set so the within-eligible argmin can NEVER select it regardless
      of its modulatory pull (the SAFETY guarantee).
    Go (promote): re-admit (bounded by gng_go_max_promote) a candidate F demoted OUT of
      the envelope whose go-evidence >= gng_go_threshold (and not itself No-Go'd).
    FAIL-OPEN (gng_protect_min_eligible): No-Go never drops the eligible set below this
      many survivors UNLESS they are SAFETY-No-Go'd (safety never overridden) -- guards
      the No-Go-over-pressure -> catatonia/avolition pole from deadlocking the gate.
  REUSE-BEFORE-DUPLICATE (ARC-106 G2): the PERSEVERATION No-Go axis CONSUMES MECH-260's
  existing dACC anti-recency suppression vector (agent routes
  _dacc_last_bundle["suppression"] in as the perseveration signal) -- generalising
  MECH-260 from a drowned score-bias into an eligibility-access gate; NO duplicate
  recency buffer. The other axes are genuinely new functions MECH-260 lacks.
  Modules: ree_core/predictors/e3_selector.py (_go_nogo_eligibility_gate + invocation
  in the shortlist block + go_nogo_signals kwarg on select() + 6 diagnostics), and
  ree_core/utils/config.py (E3Config 8 fields + from_dims passthrough),
  ree_core/agent.py (build go_nogo_signals with the MECH-260 perseveration reuse +
  set_injected_go_nogo_signals() falsifier seam; passed ONLY when the constitution is
  engaged -- version-layering guard mirroring DR-12).
  Per-candidate signals via select(go_nogo_signals=...) (optional [K] tensors keyed
  safety/staleness/perseveration/viability/go; missing axis inert). Default loop wires
  only the MECH-260 perseveration reuse; the falsifier supplies the constructed-bank
  axes via REEAgent.set_injected_go_nogo_signals().
  Config (E3Config + from_dims, all no-op default -> bit-identical OFF):
  use_go_nogo_constitution (False) + gng_safety_floor (0.5) + gng_staleness_floor (0.5)
  + gng_perseveration_floor (0.5) + gng_viability_floor (0.1) + gng_go_threshold (0.5)
  + gng_go_max_promote (2) + gng_protect_min_eligible (1).
  PRECONDITION (same as MECH-448): the gate runs only when a modulatory channel is
  present (_modulatory_accum not None) -- with no modulatory arbitration there is
  nothing to govern, legacy F-argmin runs (gate inert). This is the non-vacuity
  precondition the falsifier enforces (SD-056-trained e2.world_forward + ARC-065 GAP-A
  candidate_summary_source=e2_world_forward divergent pool + a modulatory channel).
  Backward compatible: use_go_nogo_constitution=False by default -> the gate block is
  skipped, eligible_idx passes through unchanged -> bit-identical to the MECH-448
  selector. 6/6 new contracts in tests/contracts/test_mech_449_go_nogo_constitution.py
  (C1 bit-identical OFF even with signals passed / C2 No-Go suppresses within the
  eligible set / C3 SAFETY holds under overwhelming modulatory pull / C4 bounded Go
  re-admits a demoted candidate / C5 composes with the MECH-448 f_demotion envelope /
  C6 fail-open never empties the eligible set) + 8/8 preflight + 148 E3-cluster
  contracts (e3/selector/score-bias/mech_448/modulatory/mech_341/conflict-grade/dr12)
  PASS. Full-agent boot smoke: default == explicit-OFF action stream bit-identical;
  gate-ON path runs without error.
  Phased training: N/A (pure-arithmetic gate; no learned parameters; no gradient flow).
  MECH-094: waking committed-selection path only; no replay/memory write surface;
  call-site-scoped to select_action like the sibling levers.
  Evidence-staleness (Step 8.5): NOT triggered -- no-op-default flag; every existing
  experiment uses the default (constitution off), so no dependent claim's measured
  mechanism changed. KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. MECH-449 stays candidate / substrate_conditional;
  ARC-107 / MECH-448 / MECH-260 / MECH-439 untouched. claims.yaml carries only an
  implementation_note.
  Validation experiment: V3-EXQ-689g (the MECH-449 ablation falsifier; claim_ids reflect
  MECH-449 + ARC-107) queued via /queue-experiment -- on the GAP-A-ready foraging
  substrate, the built Go/No-Go constitution must CONVERT >=1 previously-gated downstream
  channel beyond what MECH-448 achieves (over-specification if not). Pre-registered: a
  built-but-no-conversion result is non_contributory / does-not-promote, NOT an ARC-107
  falsification; self-route substrate_not_ready_requeue if the candidate pool is not
  divergent or Go/No-Go variables do not vary.
  Design doc: REE_assembly/docs/architecture/mech_449_go_nogo_constitution.md
  Design note: REE_assembly/evidence/planning/arc_107_selector_constitution_design_2026-06-20.md
  (s3.2 + s6b completeness ledger). Grounding: REE_assembly/evidence/literature/
  targeted_review_connectome_mech_439/ARC107_GROUNDING_SYNTHESIS.md (s2.1).
  See MECH-449 (this claim), ARC-107 (architecture umbrella), MECH-448 (the eligibility
  envelope this governs; pallidal gate), MECH-260 (dACC No-Go generalised; ree_core/
  cingulate/dacc.py), MECH-439 (F-dominance root), ARC-106 (grounding framework; second
  worked application), modulatory-bias-selection-authority (the shortlist-then-modulate
  arbitration the gate composes with), V3-EXQ-689f (No-Go-necessity falsifier; positive
  build trigger), V3-EXQ-689g (validation), Q-078 (umbrella), MECH-094 (N/A).
