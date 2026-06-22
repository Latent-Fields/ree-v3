"""ARC-108 JOB-2 (c): rho_t maintenance ramp -- the proximity-scaled driver that
REPLACES the flat-hold maintenance DRIVER of the MECH-090 bistable commit latch.

WHAT THIS IS (unified_dopamine_substrate_design_2026-06-22.md sec 3c / 4 / 6):
    REE built the commit/maintain/de-commit MACHINERY (the MECH-090 beta-gate
    latch, the natural-commit latch-hold re-assertion, the SD-034 closure
    operator, the refractory) and ran maintenance off a FLAT hold: while a
    natural commit is armed the beta latch is re-asserted UNCONDITIONALLY each
    tick (ree_core/agent.py latch-hold re-assertion). A flat hold has no
    intrinsic decay term -- nothing stops it running ~2400 steps (the 460h
    monolithic-hold / deviation B6), so it SWAMPS every other channel.

    In the brain the DA maintenance signal is a RAMP scaled to goal-proximity x
    value (Howe 2013; Mohebi 2019) -- it peaks-then-declines, so it *cannot*
    monopolise. rho_t renders that ramp: maintenance ramps up while approaching
    the goal and DECLINES past the proximity peak, so the hold SELF-LIMITS
    instead of running monolithically. This is the structural B6 fix -- "the
    campaign's lesson is structural bounding works, parametric tuning does not"
    (assembly map C1): a proximity-scaled ramp bounds the hold by construction,
    where ten iterations of refractory/latch-hold engineering hand-emulated,
    badly, what one ramp does natively.

COMPOSE, DON'T REPLACE THE MACHINERY (ARC-106 G2, no parallel module):
    This regulator does NOT own a latch. It REUSES the existing natural-commit
    latch-hold (ree_core/agent.py) -- the gate, the closure operator, and the
    refractory remain the safety-bearing plumbing that decide WHETHER a hold is
    permitted. rho_t decides HOW LONG it is held: it REPLACES only the flat
    (unconditional) re-assertion DRIVER with a ramp-gated one. All the existing
    latch-hold YIELD conditions (refractory active / MECH-091 genuine-threat
    interrupt / rung-6 duration release / max-ticks) are KEPT -- the ramp ADDS a
    proximity-decline self-limit on top.

rho_t (formed from quantities REE already has; no new substrate):
    rho_t = goal_proximity(z_world) x value
      goal_proximity in [0, 1] = GoalState.goal_proximity (rises approaching the
        goal; peaks at the goal; declines past it).
      value = the benefit valuation already feeding F (E3.benefit_eval_head,
        clamped >= 0) -- the "x value" the literature pairs proximity with.
    REEAgent._compute_rho_t builds rho_t; this regulator only tracks its peak.

ACCUMULATOR DYNAMICS (per maintenance tick, only while the latch-hold is armed):
    peak = max(peak, rho_t)                                   # running proximity peak
    if ticks <= onset_grace:        hold  (let the ramp rise to its peak first)
    elif rho_t < hold_floor:        RELEASE  (no value left to maintain)
    elif peak > 0 and (peak - rho_t) >= release_margin * peak:
                                    RELEASE  (declined past the proximity peak)
    else:                           hold

    The peaks-then-declines self-limit is the load-bearing piece: a FLAT hold
    never crosses the decline test, so it never self-limits (the 460h monopoly);
    a proximity-scaled rho_t peaks at the goal and then declines, crossing the
    test by construction. release_margin is the fraction of the peak the ramp
    must fall before releasing.

INTEGRATION (REEAgent latch-hold re-assertion block):
    note_commit_entry() at a fresh natural/closure-coupled commit entry (arm +
    reset peak). Each maintenance tick, BEFORE the unconditional re-assert: when
    the ramp says RELEASE, the agent disarms the hold (self-limit); otherwise the
    re-assert proceeds as before. reset() per episode.

MECH-094
    tick(simulation_mode=True) returns False (hold) without advancing the peak. A
    replay/DMN tick must not self-limit a committed motor program. Matches the
    SD-035 / MECH-279 / MECH-313 / MECH-320 / MECH-342 / NaturalCommitUrgencyRelease
    simulation_mode pattern.

See REE_assembly/docs/architecture/arc_108_job2_control_plane.md and
ree_core/policy/natural_commit_urgency.py (the rung-6 duration-release sibling
that shortens the held occupancy from the URGENCY side; this is the MAINTENANCE
side -- the two are the "dopamine-driven temporal dynamics replacing stateless
arithmetic" pair, design sec 5).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RhoMaintenanceRampConfig:
    """Configuration for the rho_t maintenance ramp (the B6-fix maintenance driver).

    Attributes:
        use_rho_maintenance_ramp : master switch. False = disabled (default,
            backward-compatible). REEAgent does not instantiate the ramp when
            False, and the latch-hold's flat re-assertion is unchanged.
        hold_floor : below this rho_t the ramp releases (no value left to
            maintain).
        release_margin : release once rho_t has declined from its running peak by
            >= release_margin * peak (the peaks-then-declines self-limit). In
            (0, 1]; smaller = releases sooner after the peak.
        onset_grace_ticks : grace ticks at commit entry before the ramp may
            self-limit (let rho_t rise to its proximity peak first; guards a
            spurious early-tick release while the ramp is still climbing).
    """

    use_rho_maintenance_ramp: bool = False
    hold_floor: float = 0.05
    release_margin: float = 0.5
    onset_grace_ticks: int = 3


class RhoMaintenanceRamp:
    """rho_t maintenance-ramp regulator (waking-only).

    Pure-arithmetic, no learned parameters, no nn.Module inheritance. Owns only
    the running proximity peak + diagnostics; it REUSES the existing
    natural-commit latch-hold (ARC-106 guardrail G2: reuse, do not duplicate the
    latch). The agent feeds rho_t (goal_proximity x value) and acts on the
    returned release bool.

    Diagnostics tracked:
        _peak                      : float (running rho_t peak this commit)
        _ticks                     : int   (maintenance ticks since the arm)
        _active                    : bool  (a committed run is being tracked)
        _last_rho                  : float
        _last_decline_frac         : float ((peak - rho_t) / peak at the last tick)
        _n_ticks                   : int
        _n_releases                : int   (proximity-decline self-limit events)
        _last_ticks_at_release     : int   (ramp ticks at the last self-limit)
        _n_simulation_skips        : int   (MECH-094 skip count)
    """

    def __init__(
        self, config: "RhoMaintenanceRampConfig | None" = None
    ) -> None:
        self.config = (
            config if config is not None else RhoMaintenanceRampConfig()
        )
        c = self.config
        if not (0.0 < c.release_margin <= 1.0):
            raise ValueError(
                f"release_margin must be in (0, 1]. Got {c.release_margin}."
            )
        if c.hold_floor < 0.0:
            raise ValueError(f"hold_floor must be >= 0. Got {c.hold_floor}.")
        if c.onset_grace_ticks < 0:
            raise ValueError(
                f"onset_grace_ticks must be >= 0. Got {c.onset_grace_ticks}."
            )
        self._peak: float = 0.0
        self._ticks: int = 0
        self._active: bool = False
        self._last_rho: float = 0.0
        self._last_decline_frac: float = 0.0
        self._n_ticks: int = 0
        self._n_releases: int = 0
        self._last_ticks_at_release: int = 0
        self._n_simulation_skips: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def note_commit_entry(self) -> None:
        """Arm the ramp at a fresh commit entry.

        Resets the running peak + tick counter (each committed program ramps
        independently). Called by REEAgent when the natural-commit latch-hold
        arms.
        """
        self._peak = 0.0
        self._ticks = 0
        self._active = True

    @property
    def is_active(self) -> bool:
        """True when a committed run is being tracked by the ramp."""
        return self._active

    # ------------------------------------------------------------------
    # Forward path
    # ------------------------------------------------------------------
    def tick(self, rho_t: float, simulation_mode: bool = False) -> bool:
        """Advance the ramp one maintenance tick; return True iff it self-limits.

        Call only while the latch-hold is armed (the caller guards on
        self._active via note_commit_entry / _ncl_hold_active). No-op (returns
        False = hold) when not armed.

        Args:
            rho_t : the proximity-scaled maintenance drive = goal_proximity x
                value (built by the agent). Higher = stronger maintenance.
            simulation_mode : MECH-094 gate. When True, no peak advance and only
                the simulation-skip counter increments; returns False (hold).

        Returns:
            True iff this tick fires a proximity-decline self-limit (the agent
            should disarm the hold). False = keep holding.
        """
        if simulation_mode:
            self._n_simulation_skips += 1
            return False
        if not self._active:
            return False

        self._ticks += 1
        self._n_ticks += 1
        rho = float(rho_t)
        self._last_rho = rho
        if rho > self._peak:
            self._peak = rho
        decline = self._peak - rho
        self._last_decline_frac = (
            decline / self._peak if self._peak > 0.0 else 0.0
        )

        c = self.config
        # Grace window: let the ramp climb to its proximity peak before it can
        # self-limit (a still-rising ramp has not peaked yet).
        if self._ticks <= c.onset_grace_ticks:
            return False

        release = (rho < c.hold_floor) or (
            self._peak > 0.0 and decline >= c.release_margin * self._peak
        )
        if release:
            self._n_releases += 1
            self._last_ticks_at_release = self._ticks
            self._active = False
        return release

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset per-episode state and diagnostic counters."""
        self._peak = 0.0
        self._ticks = 0
        self._active = False
        self._last_rho = 0.0
        self._last_decline_frac = 0.0
        self._n_ticks = 0
        self._n_releases = 0
        self._last_ticks_at_release = 0
        self._n_simulation_skips = 0

    def get_state(self) -> dict:
        """Diagnostic snapshot for experiment manifests."""
        return {
            "rho_active": self._active,
            "rho_last": self._last_rho,
            "rho_peak": self._peak,
            "rho_last_decline_frac": self._last_decline_frac,
            "rho_n_ticks": self._n_ticks,
            # Proximity-decline self-limit events (the B6-fix readout: the hold
            # ended itself instead of running monolithically).
            "rho_n_releases": self._n_releases,
            "rho_last_ticks_at_release": self._last_ticks_at_release,
            "rho_n_simulation_skips": self._n_simulation_skips,
        }
