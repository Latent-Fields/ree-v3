"""MECH-353: blocked-agency / control-failure affect stream (z_block).

A derived affect readout -- NOT an encoder output. z_block rises when an
intended action repeatedly fails to produce its forward-model-predicted
outcome, the mismatch is attributed to an EXTERNAL constraint (not the
agent's own motor error), AND the goal and the agent's capacity-belief are
RETAINED. It is the energised "assert / restore" pole the existing REE
substrate lacks (REE encodes only the capacity-collapsed withdraw pole via
SD-019b / z_harm_a + Q-036).

DISTINCTNESS (feedback_biology_before_formal_definitions; see
docs/architecture/affect_primitives.md blocked_agency subsection +
docs/architecture/mech_353_blocked_agency_zblock.md):
  vs harm (SD-011 z_harm_*): harm needs noxious contact; z_block fires with
    ZERO noxious input (a merely-blocked / omitted outcome). Distinct biology
    (RAGE circuit, Davis & Montag 2019; frustrative non-reward, Papini 2024).
  vs suffering (SD-019b z_harm_a + Q-036): SAME controllability axis, OPPOSITE
    pole. capacity-belief collapsed -> withdraw (suffering, already in REE);
    capacity-belief retained -> assert (z_block). z_block hands off to z_harm_a
    only when capacity-belief collapses.
  vs residue (MECH-056): residue is an action TAKEN at a value-cost; z_block
    is an action PREVENTED. Opposite causal structure.
  vs commitment-hold (MECH-090 beta-gate): MECH-090 is a licit, self-imposed
    output gate; z_block is an externally-imposed block against a live goal.

COMPUTATIONAL FORM (smallest, V3; this module owns step 3 of the verdict):
  1. Detector (exists, supplied by the agent loop): the SD-029 agency
     comparator applied to the ACTION-OUTCOME / z_world channel --
     outcome_mismatch = ||E2.world_forward(z_world_prev, a) - z_world_now||
     normalised. High = the intended action effect did not happen.
  2. Expectation (exists): z_goal / wanting (MECH-112) supplies "there is an
     intended outcome" -- the agent caches goal_active and passes it in.
  3. New work (here): integrate the comparator mismatch over a window into
     z_block, behind two gates --
       - ATTRIBUTION gate: count only when motor_agency >= floor (the motor
         command executed as predicted on the z_self channel) so the block is
         external, not the agent's own motor error;
       - CAPACITY gate: z_block expresses as ASSERT only while capacity-belief
         is retained; as capacity-belief falls the assert share decays and a
         withdraw-handoff share rises (routed to the suffering pole).

This is a pure-arithmetic regulator: no nn.Module, no learned parameters, no
gradient flow. Mirrors the MECH-313 NoiseFloor / MECH-320 TonicVigor /
MECH-342 CommitMaintenanceRelease pattern.

MECH-094: update() is a no-op under simulation_mode=True (replay / DMN must
not accumulate blocked-agency on imagined outcomes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class BlockedAgencyConfig:
    """MECH-353 blocked-agency regulator configuration.

    Attributes:
        use_blocked_agency: master switch. False = disabled (default,
            backward-compatible); REEAgent does not instantiate BlockedAgency.
        accumulation_rate: per-tick EMA-style rise of z_block when an external
            block is detected (alpha on the outcome_mismatch antecedent).
        leak_rate: per-tick decay of z_block when the action succeeds (the
            intended outcome occurred) -- frustration dissipates on success.
        outcome_mismatch_floor: minimum normalised action-outcome comparator
            mismatch for a tick to count as a block (below this the action
            produced roughly its predicted effect; not blocked).
        attribution_motor_floor: minimum motor_agency (z_self efference-copy
            agency signal in (0,1]) for the mismatch to be attributed to an
            EXTERNAL constraint rather than the agent's own motor error. Low
            motor_agency means the body itself did not do what was predicted
            (motor error / world-acted-on-self), which is NOT blocked-agency.
        capacity_collapse_weight: maps z_harm_a magnitude to a reduction in
            capacity-belief: capacity = clip(1 - w * z_harm_a_norm, 0, 1).
            High suffering -> low capacity-belief -> assert yields to withdraw.
        require_goal_active: when True (default) z_block accumulates only while
            a goal is retained (there is an intended outcome to be blocked
            from). Set False only for substrate probes that drive the
            comparator without the goal pipeline.
        z_block_cap: hard clamp on the integrated z_block scalar.
        assert_action_weight: ASSERT consumer -- weight of the negative bias on
            action-trajectories (REE lower-is-better, so negative favours
            action; the "escalate effort / raise vigor" pole), scaled by
            z_block_assert.
        assert_passive_weight: ASSERT consumer -- weight of the positive bias
            on no-op trajectories (penalise passivity under a live block),
            scaled by z_block_assert.
        assert_alt_action_weight: ASSERT consumer -- positive bias on the
            just-blocked first-action class (alternative-action search: try a
            DIFFERENT action that restores the intended outcome), scaled by
            z_block_assert.
        assert_bias_scale: clamp on the absolute per-candidate assert bias
            (mirrors lateral_pfc / curiosity / tonic_vigor bias_scale so the
            assert pole cannot dominate the score-bias chain).
        decommit_bound: when z_block_assert sustains above this bound for
            decommit_consecutive_ticks while the goal is still not achieved,
            emit a decommit signal (release the blocked commitment rather than
            escalate unboundedly). Realises the MECH-342 decommit consumer.
        decommit_consecutive_ticks: consecutive above-bound ticks required
            before the decommit signal fires.
        noop_class: action class index treated as no-op (matches MECH-279 /
            MECH-320 convention).
    """

    use_blocked_agency: bool = False
    accumulation_rate: float = 0.2
    leak_rate: float = 0.1
    outcome_mismatch_floor: float = 0.1
    attribution_motor_floor: float = 0.5
    capacity_collapse_weight: float = 1.0
    require_goal_active: bool = True
    z_block_cap: float = 1.5
    assert_action_weight: float = 0.1
    assert_passive_weight: float = 0.1
    assert_alt_action_weight: float = 0.1
    assert_bias_scale: float = 0.1
    decommit_bound: float = 1.0
    decommit_consecutive_ticks: int = 5
    noop_class: int = 0


@dataclass
class BlockedAgencyOutput:
    """Diagnostic snapshot for one BlockedAgency.update() call."""

    z_block: float = 0.0
    z_block_assert: float = 0.0
    withdraw_handoff: float = 0.0
    outcome_mismatch: float = 0.0
    motor_agency: float = 1.0
    capacity_belief: float = 1.0
    external_block_this_tick: bool = False
    consecutive_block_ticks: int = 0
    decommit_signal: bool = False


class BlockedAgency:
    """MECH-353 blocked-agency / control-failure regulator (z_block).

    Pure-arithmetic, no learned parameters, no nn.Module inheritance.
    Maintains the integrated z_block scalar across waking ticks, applies the
    external-attribution + capacity gates, and exposes the assert / decommit /
    handoff consumer signals.
    """

    def __init__(self, config: Optional[BlockedAgencyConfig] = None) -> None:
        self.config = config if config is not None else BlockedAgencyConfig()
        c = self.config
        # Validate (loud, not silent).
        if not (0.0 <= c.accumulation_rate <= 1.0):
            raise ValueError(
                "accumulation_rate must be in [0, 1]. "
                f"Got {c.accumulation_rate}."
            )
        if not (0.0 <= c.leak_rate <= 1.0):
            raise ValueError(
                f"leak_rate must be in [0, 1]. Got {c.leak_rate}."
            )
        if c.z_block_cap <= 0.0:
            raise ValueError(
                f"z_block_cap must be > 0. Got {c.z_block_cap}."
            )
        if c.assert_bias_scale <= 0.0:
            raise ValueError(
                f"assert_bias_scale must be > 0. Got {c.assert_bias_scale}."
            )
        if c.decommit_consecutive_ticks < 1:
            raise ValueError(
                "decommit_consecutive_ticks must be >= 1. "
                f"Got {c.decommit_consecutive_ticks}."
            )
        # State.
        self._z_block: float = 0.0
        self._consecutive_block_ticks: int = 0
        self._consecutive_assert_above_bound: int = 0
        self._last_blocked_action_class: int = -1
        self._last_output: BlockedAgencyOutput = BlockedAgencyOutput()
        # Diagnostics.
        self._n_waking_updates: int = 0
        self._n_simulation_skips: int = 0
        self._n_external_blocks: int = 0
        self._n_decommit_signals: int = 0

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------
    def update(
        self,
        outcome_mismatch: float,
        motor_agency: float,
        goal_active: bool,
        capacity_belief: float,
        blocked_action_class: int = -1,
        simulation_mode: bool = False,
    ) -> BlockedAgencyOutput:
        """Advance z_block for one waking tick.

        Args:
            outcome_mismatch: normalised SD-029 action-outcome comparator
                mismatch on the z_world channel (>= 0). High = the action's
                forward-model-predicted effect did not occur.
            motor_agency: z_self efference-copy agency signal in (0, 1].
                High = motor command executed as predicted (external block);
                low = own motor error (NOT blocked-agency).
            goal_active: whether a goal is currently retained.
            capacity_belief: belief that the agent retains the capacity to
                alter its situation, in [0, 1] (1 = fully retained).
            blocked_action_class: first-action class of the action that was
                blocked this tick (for alternative-action search). -1 = unknown.
            simulation_mode: MECH-094 gate. When True, no state advances and
                the prior output is returned with decommit_signal forced False.
        """
        if simulation_mode:
            self._n_simulation_skips += 1
            out = BlockedAgencyOutput(
                z_block=self._z_block,
                z_block_assert=self._last_output.z_block_assert,
                withdraw_handoff=self._last_output.withdraw_handoff,
                outcome_mismatch=0.0,
                motor_agency=float(motor_agency),
                capacity_belief=float(capacity_belief),
                external_block_this_tick=False,
                consecutive_block_ticks=self._consecutive_block_ticks,
                decommit_signal=False,
            )
            return out

        c = self.config
        cap = max(0.0, min(1.0, float(capacity_belief)))
        mism = max(0.0, float(outcome_mismatch))
        motor = float(motor_agency)

        goal_ok = goal_active or (not c.require_goal_active)
        external_block = (
            goal_ok
            and mism >= c.outcome_mismatch_floor
            and motor >= c.attribution_motor_floor
        )

        if external_block:
            # Rise toward the (capped) accumulation target driven by mismatch.
            self._z_block = self._z_block + c.accumulation_rate * mism
            self._consecutive_block_ticks += 1
            self._n_external_blocks += 1
            if blocked_action_class >= 0:
                self._last_blocked_action_class = int(blocked_action_class)
        else:
            # Action succeeded (or no live goal / motor error): frustration leaks.
            self._z_block = max(0.0, self._z_block - c.leak_rate)
            self._consecutive_block_ticks = 0
            if mism < c.outcome_mismatch_floor:
                # Genuine success clears the alternative-action target.
                self._last_blocked_action_class = -1

        self._z_block = max(0.0, min(c.z_block_cap, self._z_block))

        # Capacity-gated split: assert while capacity retained; hand off the
        # remainder to the withdraw / suffering pole as capacity collapses.
        z_block_assert = self._z_block * cap
        withdraw_handoff = self._z_block * (1.0 - cap)

        # Decommit accounting: sustained failed assert (asserting hard but the
        # block persists) -> emit a decommit signal so the ARC-016-gated
        # release can fire rather than escalate unboundedly.
        decommit_signal = False
        if external_block and z_block_assert >= c.decommit_bound:
            self._consecutive_assert_above_bound += 1
            if self._consecutive_assert_above_bound >= c.decommit_consecutive_ticks:
                decommit_signal = True
                self._consecutive_assert_above_bound = 0
                self._n_decommit_signals += 1
        elif not external_block:
            self._consecutive_assert_above_bound = 0

        self._n_waking_updates += 1
        out = BlockedAgencyOutput(
            z_block=self._z_block,
            z_block_assert=z_block_assert,
            withdraw_handoff=withdraw_handoff,
            outcome_mismatch=mism,
            motor_agency=motor,
            capacity_belief=cap,
            external_block_this_tick=external_block,
            consecutive_block_ticks=self._consecutive_block_ticks,
            decommit_signal=decommit_signal,
        )
        self._last_output = out
        return out

    # ------------------------------------------------------------------
    # ASSERT consumer
    # ------------------------------------------------------------------
    def compute_assert_score_bias(
        self,
        action_classes,
        device,
        dtype,
    ):
        """ASSERT consumer -- escalate-effort + alternative-action search.

        Returns a per-candidate additive score bias [K] (REE lower-is-better):
          - NEGATIVE on action-trajectories (favour acting -> raise vigor /
            escalate effort; the new behavioural pole REE lacks);
          - POSITIVE on no-op trajectories (penalise passivity under a block);
          - POSITIVE on the just-blocked first-action class (push the search
            toward a DIFFERENT action that restores the intended outcome).
        Magnitude scales with the last-tick z_block_assert; zero when there is
        no asserting block this tick (so OFF / no-block ticks add nothing).

        Args:
            action_classes: sequence/tensor of first-action class indices [K].
            device, dtype: target torch device / dtype for the returned bias.
        """
        import torch

        K = len(action_classes)
        z_assert = float(self._last_output.z_block_assert)
        if K == 0 or z_assert <= 0.0:
            return torch.zeros(K, dtype=dtype, device=device)

        c = self.config
        action_term = -c.assert_action_weight * z_assert
        passive_term = c.assert_passive_weight * z_assert
        alt_term = c.assert_alt_action_weight * z_assert
        blocked_cls = self._last_blocked_action_class

        vals = []
        for cls in action_classes:
            cls_i = int(cls)
            if cls_i == c.noop_class:
                vals.append(passive_term)
            else:
                b = action_term
                if blocked_cls >= 0 and cls_i == blocked_cls:
                    # Penalise re-selecting the blocked action (alt-action search).
                    b = b + alt_term
                vals.append(b)
        bias = torch.tensor(vals, dtype=dtype, device=device)
        scale = c.assert_bias_scale
        return torch.clamp(bias, min=-scale, max=scale)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_z_block(self) -> float:
        return self._z_block

    def last_output(self) -> BlockedAgencyOutput:
        return self._last_output

    def reset(self) -> None:
        """Reset per-episode state + diagnostic counters."""
        self._z_block = 0.0
        self._consecutive_block_ticks = 0
        self._consecutive_assert_above_bound = 0
        self._last_blocked_action_class = -1
        self._last_output = BlockedAgencyOutput()
        self._n_waking_updates = 0
        self._n_simulation_skips = 0
        self._n_external_blocks = 0
        self._n_decommit_signals = 0

    def get_state(self) -> dict:
        """Diagnostic snapshot for experiment manifests."""
        o = self._last_output
        return {
            "z_block": self._z_block,
            "last_z_block_assert": o.z_block_assert,
            "last_withdraw_handoff": o.withdraw_handoff,
            "last_outcome_mismatch": o.outcome_mismatch,
            "last_motor_agency": o.motor_agency,
            "last_capacity_belief": o.capacity_belief,
            "last_external_block": o.external_block_this_tick,
            "consecutive_block_ticks": self._consecutive_block_ticks,
            "n_waking_updates": self._n_waking_updates,
            "n_simulation_skips": self._n_simulation_skips,
            "n_external_blocks": self._n_external_blocks,
            "n_decommit_signals": self._n_decommit_signals,
        }
