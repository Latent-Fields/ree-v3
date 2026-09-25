"""
MECH-279: PAG (periaqueductal gray) freeze-gate.

Architectural commitment (see REE_assembly/docs/architecture/sd_036_gabaergic_decay_regulator.md
section 3 "PAG freeze-gating (MECH-279)"):

  Freeze is a *committed* behavioural state -- sustained motor immobility plus
  elevated autonomic arousal. Not just "no movement"; an active commitment to
  not-move with its own duration and exit criterion. Biologically gated by PAG,
  where descending inputs from amygdala / hypothalamus / medial PFC converge on
  freeze-promoting cells that are themselves GABAergic. Freeze termination
  requires GABAergic inhibition to wane.

Logic:

  freeze_commit(t) = (z_harm_a(t) * duration_above_threshold(t)) > theta_freeze
  freeze_active(t) = freeze_commit OR (freeze_active(t-1) AND z_harm_a(t) > exit_threshold)
  exit_threshold   = theta_freeze * gaba_tone(t)

  duration_above_threshold(t) = ticks since z_harm_a first crossed
                                duration_input_threshold (resets when z_harm_a
                                falls back below that threshold).

  When freeze is active, the action selector is constrained to no-op /
  minimal-movement actions. Exit requires z_harm_a to fall below
  exit_threshold, which depends on both SD-036 decay (z_harm_a returns toward
  baseline) and gaba_tone (GABA agonists raise the exit_threshold, making exit
  easier). The same neurotransmitter system gates BOTH entry (PAG freeze-cell
  commitment) and exit (SD-036 decay returning z_harm_a below threshold).

Non-trainable: pure arithmetic over scalars and a small counter. No gradient
flow. Reset per episode.

MECH-287 instrument note (2026-09-25): the three long-standing counters
_n_ticks / _n_commits / _n_releases are CUMULATIVE across every episode --
reset() has never cleared them -- and an episode ending while frozen consumes
a commit with no matching release. The ratio n_commits / n_releases is
therefore an episode-count artifact, not "re-commits per release". A
phase-scoped per-episode readout (episode_diagnostics / episode_records /
set_phase / reset_diagnostics) is provided ALONGSIDE; the cumulative counters
keep their existing semantics so no prior experiment's numbers move. Analysis:
REE_assembly/evidence/planning/mech287_dv_instrument_confound_20260924.md.

Master switch: REEConfig.use_pag_freeze_gate (default False) gates instantiation
and wiring. With the flag off, agents behave bit-identically to legacy.

MECH-094: simulation_mode=True ticks return a zeroed-output (freeze inactive)
without updating internal state. Replay / DMN content must not commit the
agent into a behavioural freeze state.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional


@dataclass
class PAGFreezeGateConfig:
    """MECH-279 freeze-gate configuration.

    Defaults are conservative -- freeze fires on sustained, substantial
    z_harm_a load. With `theta_freeze=2.0` and `duration_input_threshold=0.4`,
    a z_harm_a magnitude of 0.5 needs at least 4 sustained ticks above 0.4 to
    accumulate (z_harm_a * duration) > theta_freeze. Tune `theta_freeze` per
    experiment to set the catatonia entry sensitivity.
    """

    # Master flag, mirrored on REEConfig. Held here too so the gate is
    # independently testable from the agent.
    enabled: bool = True

    # Commit threshold for entering freeze. Compared against
    # z_harm_a_norm * duration_above_threshold.
    theta_freeze: float = 2.0

    # Threshold above which the duration-above counter increments. When
    # z_harm_a_norm falls back below this value, the counter resets to zero.
    duration_input_threshold: float = 0.4

    # Optional minimum committed-freeze duration (in ticks). Once freeze
    # commits, the gate stays active for at least this many ticks regardless
    # of z_harm_a / exit_threshold dynamics. 0 = no minimum (exit on first
    # tick where z_harm_a drops below exit_threshold).
    min_freeze_duration: int = 0

    # Optional maximum freeze duration (in ticks). 0 = no cap. Useful to
    # prevent permanent locks during smoke / debug; in normal operation,
    # SD-036 decay should reliably bring z_harm_a below exit_threshold.
    max_freeze_duration: int = 0

    # SD-037: scaling on theta_freeze for the broadcast override regulator.
    # When override_signal is supplied to tick(),
    #   exit_threshold = theta_freeze * (1 + alpha_override * override_signal)
    #                                 * gaba_tone
    # so a recruited override (orexin-analog) raises the effective exit
    # threshold and shortens the committed-freeze state. Default 0.0 is a
    # no-op (override has no effect).
    alpha_override: float = 0.0


@dataclass
class PAGFreezeGateOutput:
    """Per-tick PAGFreezeGate output."""

    freeze_active: bool = False
    # True on the tick where freeze commits this step (newly entered).
    freeze_commit: bool = False
    # True on the tick where freeze releases this step.
    freeze_release: bool = False
    # Accumulated duration counter at the end of this tick.
    duration_above_threshold: int = 0
    # The exit threshold actually used this tick (theta_freeze * gaba_tone).
    exit_threshold: float = 0.0
    # The z_harm_a magnitude observed this tick.
    z_harm_a_norm: float = 0.0
    # How many ticks freeze has been active (0 when inactive).
    ticks_in_freeze: int = 0


@dataclass
class PAGEpisodeRecord:
    """MECH-287 instrument: one finalised episode's freeze summary.

    `recommits` is the quantity MECH-287's DV text means by "freeze
    re-commit count per PAG release" -- a commit that happened AFTER a
    release WITHIN THE SAME EPISODE. A commit that opens an episode is
    not a re-commit, and an episode that ends while still frozen records
    `ended_frozen=True` and contributes NO release.
    """

    index: int = 0
    phase: str = ""
    ticks: int = 0
    commits: int = 0
    releases: int = 0
    recommits: int = 0
    ended_frozen: bool = False

    def as_dict(self) -> dict:
        return {
            "index": int(self.index),
            "phase": str(self.phase),
            "ticks": int(self.ticks),
            "commits": int(self.commits),
            "releases": int(self.releases),
            "recommits": int(self.recommits),
            "ended_frozen": bool(self.ended_frozen),
        }


def _empty_agg() -> Dict[str, int]:
    return {
        "n_episodes": 0,
        "ticks": 0,
        "commits": 0,
        "releases": 0,
        "recommits": 0,
        "episodes_ending_frozen": 0,
    }


class PAGFreezeGate:
    """MECH-279 PAG-analog committed-freeze gate.

    Public API:
      tick(z_harm_a_norm, gaba_tone=1.0, simulation_mode=False) -> PAGFreezeGateOutput
        Compute the freeze state for this step.

      reset()
        Clear per-episode state.

      is_active        Convenience property mirroring last freeze_active.

      diagnostics      Dict of running counters (cumulative; see its docstring
                       for the MECH-287 confound warning). Also carries the
                       additive episode_* keys from episode_diagnostics().

      episode_diagnostics(phase=None, include_current=True)
                       MECH-287 phase-scoped per-episode readout. This is the
                       instrument for "freeze re-commit count per PAG release";
                       the cumulative counters are NOT.
      episode_records(phase=None)   Per-episode detail rows.
      set_phase(label)              Label subsequent episodes ("warmup"/"eval").
      reset_diagnostics()           Explicit eval-phase reset of ALL counters.

    State (per episode):
      _freeze_active                   bool, current freeze status
      _duration_above_threshold        int, sustained-input counter
      _ticks_in_freeze                 int, how long freeze has been active
      _last_output                     PAGFreezeGateOutput, last tick's output
      _n_ticks, _n_commits, _n_releases  diagnostic counters
    """

    def __init__(self, config: Optional[PAGFreezeGateConfig] = None):
        self.config = config or PAGFreezeGateConfig()

        self._freeze_active: bool = False
        self._duration_above_threshold: int = 0
        self._ticks_in_freeze: int = 0
        self._last_output: PAGFreezeGateOutput = PAGFreezeGateOutput()

        self._n_ticks: int = 0
        self._n_commits: int = 0
        self._n_releases: int = 0

        # -- MECH-287 phase-scoped per-episode readout (purely additive) --
        #
        # WHY THIS EXISTS. The three counters above are CUMULATIVE: reset()
        # deliberately does not clear them, so across a run they span every
        # warmup and eval episode. Worse, an episode that ends while frozen
        # consumes a commit with NO matching release (reset() clears the
        # freeze without incrementing _n_releases), which makes
        #     n_commits - n_releases  <=  n_episodes
        # and turns the naive ratio n_commits / n_releases into a function of
        # EPISODE COUNT rather than of re-commit behaviour after a release.
        # V3-EXQ-475's headline "~12.9 re-commits per release" (71/6, 70/5,
        # 64/5 over 65 episodes) is that artifact. See
        # REE_assembly/evidence/planning/mech287_dv_instrument_confound_20260924.md.
        #
        # The fields below are read-only bookkeeping accumulated alongside the
        # cumulative counters. They consume no RNG and are never branched on,
        # so the agent's behaviour is bit-identical whether or not anything
        # reads them. The cumulative counters' semantics are UNCHANGED.
        self._episode_index: int = 0
        self._episode_ticks: int = 0
        self._episode_commits: int = 0
        self._episode_releases: int = 0
        self._episode_recommits: int = 0
        self._episode_released_yet: bool = False
        self._phase: str = ""
        # Exact per-phase aggregates. Independent of the bounded record list
        # below, so a long run's totals stay exact even once records roll off.
        self._episode_agg: Dict[str, Dict[str, int]] = {}
        self._episode_records: Deque[PAGEpisodeRecord] = deque(maxlen=8192)

    # -- State management --

    def set_phase(self, phase: str) -> None:
        """MECH-287: label episodes finalised from now on (e.g. "warmup" / "eval").

        KNOWN DEFECT (red-team 2026-09-25, re-measured). The label is stamped
        at FINALIZE time, not at episode start, so under a RESET-AT-START
        driver (`for ep: agent.reset(); rollout()`) the last episode of the
        old phase is still in progress here and is finalised under the NEW
        label. Measured on a 60-warmup/5-eval shape: warmup 59, eval 6, with
        the contaminating episode being an ends-frozen one. Safe only under a
        RESET-AT-END driver. Not fixed here because the fix is bundled with
        the owed DV decision (see episode_diagnostics).

        Applies to the CURRENTLY in-progress episode and every later one, so
        call it at a phase boundary. Purely a readout label: no behavioural
        effect.
        """
        self._phase = str(phase)

    def _finalize_episode(self) -> None:
        """MECH-287: roll the in-progress episode into the per-episode readout.

        Called by reset() (the per-episode boundary) and folded in read-only
        by the readout accessors so the LAST episode of a run -- which nothing
        ever calls reset() after -- is not silently dropped.
        """
        if self._episode_ticks == 0:
            # Nothing ticked: a double reset(), or a reset() before the first
            # episode. Recording it would inflate n_episodes with phantoms.
            return
        rec = PAGEpisodeRecord(
            index=int(self._episode_index),
            phase=str(self._phase),
            ticks=int(self._episode_ticks),
            commits=int(self._episode_commits),
            releases=int(self._episode_releases),
            recommits=int(self._episode_recommits),
            ended_frozen=bool(self._freeze_active),
        )
        self._episode_records.append(rec)
        agg = self._episode_agg.setdefault(rec.phase, _empty_agg())
        agg["n_episodes"] += 1
        agg["ticks"] += rec.ticks
        agg["commits"] += rec.commits
        agg["releases"] += rec.releases
        agg["recommits"] += rec.recommits
        agg["episodes_ending_frozen"] += int(rec.ended_frozen)
        self._episode_index += 1
        self._episode_ticks = 0
        self._episode_commits = 0
        self._episode_releases = 0
        self._episode_recommits = 0
        self._episode_released_yet = False

    def reset(self) -> None:
        """Clear per-episode state.

        MECH-287 note: this finalises the per-episode readout record FIRST
        (capturing whether the episode ended still frozen), then clears the
        per-episode state exactly as before. The cumulative counters
        _n_ticks / _n_commits / _n_releases are deliberately NOT cleared --
        that is long-standing behaviour other experiments depend on. Use
        reset_diagnostics() for an explicit phase-scoped counter reset.
        """
        self._finalize_episode()
        self._freeze_active = False
        self._duration_above_threshold = 0
        self._ticks_in_freeze = 0
        self._last_output = PAGFreezeGateOutput()

    def reset_diagnostics(self) -> None:
        """MECH-287: explicitly zero ALL counters, cumulative ones included.

        The "explicit eval-phase reset" option: call this at eval entry so the
        cumulative counters describe the eval phase alone rather than
        warmup+eval. Not called by reset() and not called by the agent -- a
        driver must opt in, so no existing experiment's numbers change.
        """
        self._finalize_episode()
        self._n_ticks = 0
        self._n_commits = 0
        self._n_releases = 0
        self._episode_index = 0
        self._episode_agg = {}
        self._episode_records.clear()

    @property
    def is_active(self) -> bool:
        return bool(self._freeze_active)

    # -- Tick: main per-step computation --

    def tick(
        self,
        z_harm_a_norm: float,
        gaba_tone: float = 1.0,
        simulation_mode: bool = False,
        override_signal: float = 0.0,
    ) -> PAGFreezeGateOutput:
        """Compute the freeze gate state for this step.

        Args:
            z_harm_a_norm: Magnitude (L2 norm) of the SD-011 affective harm
                stream this tick.
            gaba_tone: Global GABAergic tonic multiplier (matches the SD-036
                regulator's gaba_tone). Used to compute exit_threshold =
                theta_freeze * gaba_tone. Higher tone -> higher exit threshold
                -> easier exit (benzo-analog accelerates termination).
            simulation_mode: MECH-094 hypothesis-tag equivalent. True -> return
                a zeroed output and do not update internal state. Replay / DMN
                content must not commit the agent into freeze.

        Returns:
            PAGFreezeGateOutput with freeze_active, commit / release edges,
            duration counter, and diagnostic fields.
        """
        if not self.config.enabled:
            # Master-off path returns a zeroed output without touching state.
            out = PAGFreezeGateOutput(
                freeze_active=False,
                freeze_commit=False,
                freeze_release=False,
                duration_above_threshold=int(self._duration_above_threshold),
                exit_threshold=0.0,
                z_harm_a_norm=float(z_harm_a_norm),
                ticks_in_freeze=0,
            )
            self._last_output = out
            return out

        if simulation_mode:
            # MECH-094 gate: replay / simulation -- return zeroed and do NOT
            # update entry / exit counters.
            out = PAGFreezeGateOutput(
                freeze_active=False,
                freeze_commit=False,
                freeze_release=False,
                duration_above_threshold=int(self._duration_above_threshold),
                exit_threshold=0.0,
                z_harm_a_norm=float(z_harm_a_norm),
                ticks_in_freeze=int(self._ticks_in_freeze),
            )
            self._last_output = out
            return out

        self._n_ticks += 1
        self._episode_ticks += 1

        z = float(z_harm_a_norm)
        # Clamp gaba_tone to non-negative; values <0 are not biologically
        # meaningful for the exit-threshold computation.
        tone = max(0.0, float(gaba_tone))

        # 1. Update the sustained-input duration counter. The counter only
        #    accumulates while the gate is INACTIVE -- once committed, the
        #    accumulator stops advancing (and is reset to zero on release).
        #    This implements the per-cycle "fresh accumulation" semantic:
        #    each commit requires a new run-up of sustained input above
        #    duration_input_threshold rather than re-firing on the same
        #    accumulator immediately after release.
        if not self._freeze_active:
            if z > float(self.config.duration_input_threshold):
                self._duration_above_threshold += 1
            else:
                self._duration_above_threshold = 0

        # 2. Compute exit threshold for this tick.
        # SD-037: an active broadcast override raises the exit threshold via
        # alpha_override (no-op when alpha_override=0.0 or override_signal=0.0).
        override = max(0.0, min(1.0, float(override_signal)))
        override_factor = 1.0 + float(self.config.alpha_override) * override
        exit_threshold = float(self.config.theta_freeze) * override_factor * tone

        # 3. Edge detection.
        commit_this_tick = False
        release_this_tick = False

        # 4. Entry check (only when not already in freeze).
        if not self._freeze_active:
            commit_value = z * float(self._duration_above_threshold)
            if commit_value > float(self.config.theta_freeze):
                self._freeze_active = True
                self._ticks_in_freeze = 0  # will increment to 1 below
                commit_this_tick = True
                self._n_commits += 1
                # MECH-287: a commit that follows a release WITHIN THIS
                # EPISODE is a genuine re-commit. The episode's opening
                # commit is not.
                self._episode_commits += 1
                if self._episode_released_yet:
                    self._episode_recommits += 1

        # 5. Exit check (only when active and not just committed).
        if self._freeze_active and not commit_this_tick:
            below_exit = z < exit_threshold
            min_dur_met = self._ticks_in_freeze >= int(self.config.min_freeze_duration)
            max_dur_cap = (
                int(self.config.max_freeze_duration) > 0
                and self._ticks_in_freeze >= int(self.config.max_freeze_duration)
            )
            if (below_exit and min_dur_met) or max_dur_cap:
                self._freeze_active = False
                release_this_tick = True
                self._n_releases += 1
                self._episode_releases += 1
                self._episode_released_yet = True
                # Reset both the freeze-duration counter and the sustained-
                # input accumulator on release so the next commit requires a
                # fresh run-up.
                self._ticks_in_freeze = 0
                self._duration_above_threshold = 0

        # 6. Tick the in-freeze counter when active.
        if self._freeze_active:
            self._ticks_in_freeze += 1

        out = PAGFreezeGateOutput(
            freeze_active=bool(self._freeze_active),
            freeze_commit=bool(commit_this_tick),
            freeze_release=bool(release_this_tick),
            duration_above_threshold=int(self._duration_above_threshold),
            exit_threshold=float(exit_threshold),
            z_harm_a_norm=float(z),
            ticks_in_freeze=int(self._ticks_in_freeze),
        )
        self._last_output = out
        return out

    # -- Read-only accessors --

    @property
    def last_output(self) -> PAGFreezeGateOutput:
        return self._last_output

    @property
    def diagnostics(self) -> dict:
        """Cumulative counters, UNCHANGED semantics, plus the MECH-287 readout.

        WARNING -- n_commits / n_releases / n_ticks span EVERY episode of the
        run (reset() does not clear them), and an episode ending frozen
        consumes a commit with no release. So `n_commits / n_releases` is NOT
        "re-commits per release": it is dominated by episode count. For that
        DV read `episode_recommits_per_release` below, or episode_diagnostics.
        """
        d = {
            "n_ticks": int(self._n_ticks),
            "n_commits": int(self._n_commits),
            "n_releases": int(self._n_releases),
            "freeze_active": bool(self._freeze_active),
            "duration_above_threshold": int(self._duration_above_threshold),
            "ticks_in_freeze": int(self._ticks_in_freeze),
        }
        # Additive MECH-287 keys. NOTE (red-team 2026-09-25): these are
        # ALL-PHASE aggregates -- episode_diagnostics() is called with no
        # phase=, so at a 60-warmup/5-eval shape warmup dominates 12:1. Use
        # episode_diagnostics(phase="eval") for a phase-scoped read; do not
        # treat these keys as one. They also omit dv_measurable.
        ep = self.episode_diagnostics()
        for k in (
            "n_episodes",
            "episodes_ending_frozen",
            "commits",
            "releases",
            "recommits",
            "recommits_per_release",
        ):
            d["episode_" + k] = ep[k]
        return d

    # -- MECH-287 phase-scoped per-episode readout --

    def episode_diagnostics(
        self,
        phase: Optional[str] = None,
        include_current: bool = True,
    ) -> dict:
        """The repaired MECH-287 DV: per-episode freeze commit/release deltas.

        Args:
            phase: restrict to episodes labelled with this phase (see
                set_phase). None = every phase.
            include_current: fold the in-progress, not-yet-reset episode in.
                Default True because nothing calls reset() after a run's LAST
                episode, so a post-loop read would otherwise silently drop it.

        !! REFUSED AS A DV -- READ THIS BEFORE USING recommits_per_release !!

        An adversarial red-team returned BLOCKING on this field on 2026-09-25
        and the finding was independently re-measured. Do NOT put it in an
        experiment's criteria until the owed decision is made (decision chip
        chip-20260925-mech287-dv-choice; findings:
        REE_assembly/evidence/planning/mech287_readout_redteam_findings_20260925.md).

        Why: a commit fires only from the inactive state and a release only
        from the active state, so within an episode every re-commit is
        preceded by exactly one release. Hence recommits <= releases ALWAYS
        and this ratio is bounded in [0, 1] -- it cannot be compared with
        V3-EXQ-475's "~12.9". Worse, it reduces to

            recommits_per_release = 1 - (episodes with >=1 release that ended
                                         NOT frozen) / (total releases)

        so it is PINNED AT EXACTLY 1.0, with zero variance, whenever every
        episode ends frozen -- which is precisely V3-EXQ-475's phenotype
        (1000/1000 freeze-active steps) and the comparator regime MECH-287's
        non-degeneracy precondition requires. Measured: 1.0000 at 1, 2, 5 and
        10 re-commit cycles per episode. And where episodes DO end released it
        moves with episode LENGTH (1 - 1/cycles: 0.0 / 0.5 / 0.75 / 0.875 /
        0.95 at 1/2/4/8/20 cycles). So the episode-count artifact was
        re-parameterised, not removed.

        `recommits_per_episode` is unbounded and does not saturate, but
        whether IT (or something else) is MECH-287's DV is a registered-
        falsifier question for governance, not a choice this module may make.

        Returns a dict whose fields are per-episode deltas.
        `episodes_ending_frozen` is reported alongside because it is exactly
        the quantity the old cumulative ratio was confounded WITH.
        """
        agg = _empty_agg()
        for ph, a in self._episode_agg.items():
            if phase is not None and ph != phase:
                continue
            for k in agg:
                agg[k] += a[k]

        if include_current and self._episode_ticks > 0:
            if phase is None or self._phase == phase:
                agg["n_episodes"] += 1
                agg["ticks"] += int(self._episode_ticks)
                agg["commits"] += int(self._episode_commits)
                agg["releases"] += int(self._episode_releases)
                agg["recommits"] += int(self._episode_recommits)
                agg["episodes_ending_frozen"] += int(self._freeze_active)

        n_ep = max(int(agg["n_episodes"]), 1)
        out = dict(agg)
        out["phase"] = phase if phase is not None else "*"
        out["recommits_per_release"] = (
            float(agg["recommits"]) / float(agg["releases"])
            if agg["releases"] > 0
            else 0.0
        )
        out["commits_per_episode"] = float(agg["commits"]) / float(n_ep)
        out["releases_per_episode"] = float(agg["releases"]) / float(n_ep)
        out["recommits_per_episode"] = float(agg["recommits"]) / float(n_ep)
        out["frac_episodes_ending_frozen"] = (
            float(agg["episodes_ending_frozen"]) / float(n_ep)
        )
        # Explicit cannot-determine category: a zero DV with no releases at
        # all means the instrument never had an occasion to measure, which is
        # NOT the same reading as "re-commits did not happen after releases".
        out["dv_measurable"] = bool(agg["releases"] > 0)
        return out

    @property
    def episode_phases(self) -> List[str]:
        """Phase labels seen so far, in first-finalised order."""
        return list(self._episode_agg.keys())

    def episode_records(self, phase: Optional[str] = None) -> List[dict]:
        """Finalised per-episode records (bounded to the most recent 8192)."""
        return [
            r.as_dict()
            for r in self._episode_records
            if phase is None or r.phase == phase
        ]
