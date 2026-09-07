## MECH-090 R-c continuation: nav_competence axis (2026-05-29)
- MECH-090 R-c continuation: control_plane.beta_gate.commit_entry_readiness_
  conjunction.nav_competence -- IMPLEMENTED 2026-05-29 (commitment_closure:GAP-4
  substrate landing pass 2 of 2; behavioural validation still pending V3-EXQ-592b
  PASS). The 2026-05-28 landing implemented the WITHIN-TICK DECISIVENESS axis
  (per-candidate score margin -- Hanes & Schall 1996 reading). This pass adds
  the ACROSS-TICK MOTOR-PROGRAM READINESS axis (Cisek & Kalaska 2010 affordance-
  preparation + Roesch / Calu / Schoenbaum 2007 dopaminergic readiness). Both
  axes are R-c readings; both can be enabled/disabled independently; they
  AND-compose at both elevate sites.
  Module: ree_core/policy/commit_readiness.py (CommitReadiness +
  CommitReadinessConfig). Pure-arithmetic regulator (no nn.Module, no learned
  params), sibling pattern to MECH-313 NoiseFloor / MECH-320 TonicVigor.
  Maintains a [0, 1] readiness EMA over per-tick outcome signals plus an
  explicit notify_outcome(value) harness-push seam. Initial value 1.0
  (fail-open). MECH-094 standard simulation_mode pattern.
  Wiring (ree_core/agent.py): REEAgent.__init__ instantiates self.commit_
  readiness when config.use_commit_readiness=True (auto-armed by __post_init__ /
  from_dims OR-only resolver when use_mech090_readiness_conjunction=True).
  REEAgent.select_action computes _readiness_admits =
  commit_readiness.is_above_floor(mech090_readiness_floor) once at the top
  of the beta-gate block and AND-composes with the existing
  should_admit_elevation(score_margin, K) at BOTH call sites (bistable +
  legacy). Block diagnostics advance via commit_readiness.notify_block() at
  the source. REEAgent.reset calls commit_readiness.reset() per-episode.
  Per-tick outcome-signal source (Phase 1): the experiment harness pushes via
  commit_readiness.notify_outcome(value). The substrate-side seam is wired;
  the harness is responsible for the per-tick update. committed_mode_curriculum.py
  pushes its probe-derived nav_competence via this seam. Phase 2 follow-on
  (separate /implement-substrate pass): wire an env-emitted
  "mech090_readiness_outcome" key reading in agent.sense() so the substrate
  advances readiness automatically without harness involvement.
  Config (REEConfig + from_dims, in contrast with the prior session's
  HeartbeatConfig-resident score_margin gate flags):
  use_mech090_readiness_conjunction (bool, default False; bit-identical OFF),
  mech090_readiness_floor (float, default 0.3 -- mid-low floor that V3-EXQ-
  592 seed 42's nav_competence=0.0 clearly fails to clear; calibratable),
  use_commit_readiness (bool, default False; auto-armed True via the
  OR-only resolver when the conjunction flag is on),
  commit_readiness_window (int, default 20; informational, alpha is the
  load-bearing knob), commit_readiness_ema_alpha (float, default 0.1;
  ~10-tick half-life), commit_readiness_initial (float, default 1.0;
  fail-open).
  Backward compatible: 523/523 contracts PASS (506 prior + 17 new MECH-090
  R-c-nav-competence contracts) with both R-c master flags OFF. Master-OFF
  construction produces agent.commit_readiness=None and the agent runs
  bit-identical to pre-amendment. Master-ON with default
  commit_readiness_initial=1.0 produces readiness == 1.0 on first tick, so
  the conjunction admits while the EMA has no real outcome data (fail-open).
  The conjunction begins blocking only once notify_outcome (harness) pushes
  a low value or update drives the EMA below the floor via real outcome
  signals.
  Composition with the score_margin gate (both at both elevate sites):
    _readiness_margin = sorted(scores)[1] - sorted(scores)[0]    (existing)
    _readiness_admits = commit_readiness.is_above_floor(floor)   (NEW)
                        when use_mech090_readiness_conjunction
                        else True (legacy bit-identical)
    elevation admitted iff:
        result.committed
        AND BetaGate.should_admit_elevation(margin, K)   (existing)
        AND _readiness_admits                            (NEW)
  Phased training: N/A (pure-arithmetic regulator; no learned parameters;
  no gradient flow; no encoder head).
  MECH-094: standard simulation_mode pattern. update(simulation_mode=True)
  and notify_outcome(value, simulation_mode=True) return without advancing
  the readiness EMA. Gate decisions at waking action-selection only; the
  substrate is read-only over commit_readiness state at the elevate sites
  and writes only a control-state transition.
  Validation continuation: V3-EXQ-592b grid extended to 4 arms (ARM_2
  GATED_NAV_COMP_ON: nav_competence gate alone; ARM_3 GATED_BOTH_ON: both
  R-c gates active; ARM_4 BOTH_GATES_OFF_HARNESS_FORCES_READY: rv-only
  baseline with harness pushing notify_outcome(1.0) each tick). Falsifier
  grid: see design doc "R-c amendment continued / Falsifiability" section
  for the four orthogonal outcomes (which-axis-carries-the-load discrimination).
  Design doc: REE_assembly/docs/architecture/mech_090_commit_entry_predicate.md
  (R-c continuation section appended 2026-05-29).
  See MECH-090 (this claim's predecessor pass: within-tick decisiveness axis
  landed 2026-05-28 via BetaGate.should_admit_elevation + HeartbeatConfig
  flags), MECH-091 (urgency interrupt; orthogonal release-side override),
  ARC-028 + MECH-105 (hippocampal-BetaGate completion coupling; release side),
  SD-034 / MECH-266 / MECH-267 / MECH-268 (downstream behavioural arms;
  transitively unblocked via GAP-4), commitment_closure:GAP-4 (the closure-
  plan gap this two-pass amendment resolves), Cisek & Kalaska 2010 (across-
  tick affordance-preparation anchor), Roesch / Calu / Schoenbaum 2007
  (dopaminergic readiness anchor), MECH-313 NoiseFloor / MECH-320 TonicVigor
  (sibling pure-arithmetic regulators in ree_core/policy/), MECH-094
  (simulation_mode argument standard pattern).
