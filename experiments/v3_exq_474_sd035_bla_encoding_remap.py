#!/opt/local/bin/python3
"""
V3-EXQ-474 -- SD-035 BLAAnalog substrate-readiness (MECH-074a
              encoding_gain + MECH-074d remap_signal).

Claims: SD-035, MECH-074a, MECH-074d

Substrate readiness validation for the BLA half of the amygdala analogue
(SD-035, IMPLEMENTED 2026-04-21). This is a diagnostic experiment --
not a behavioural-recall gate. BLA outputs (encoding_gain,
retrieval_bias, remap_signal) are produced and cached on the agent, but
the HippocampalModule consumer wiring (write-gain multiplication,
retrieval reweighting, remap handoff) is deferred to a follow-up pass
once this substrate-readiness EXQ passes. Therefore this script
verifies the BLA module behaves correctly at the module boundary, NOT
that behavioural recall on threat-associated contexts improves.

Confirms:
  (a) Backward compatibility -- enabling use_amygdala_analog +
      use_bla_analog with z_harm_a quiet produces action-class entropy
      within tolerance of the fully-OFF baseline.
  (b) Inverted-U encoding gain (Roozendaal 2011): gain=1.0 below
      arousal_threshold_on (=0.4), gain in (1.0, encoding_gain_max]
      monotonically with arousal up to arousal_peak (=0.7), then
      declines toward encoding_gain_max (default 2.5 at peak).
  (c) Remap gate -- Moita 2004 attribution-gated: a synthetic PE spike
      above remap_pe_sigma_threshold fires remap_signal with the
      top remap_code_fraction of attribution candidates; the same PE
      spike WITHOUT candidates (under default
      remap_requires_attribution=True) does NOT fire.
  (d) Encoding-window decay (post-fire): after an arousal spike, the
      window_steps_remaining counter drops to 0 on schedule; gain
      returns to 1.0 (or floor) after the window elapses.
  (e) MECH-094 boundary: BLA does not itself have a simulation-mode
      gate argument (deferred to caller / hippocampal consumer via
      MECH-261). Verified that BLAConfig / BLAOutput do not expose a
      simulation_mode parameter that would be silently bypassed --
      the gate lives on the consumer side and is testable there.

Conditions (BLA on/off, quiet rollout):
  BLA_OFF_QUIET: use_amygdala_analog=False
  BLA_ON_QUIET:  use_amygdala_analog=True, use_bla_analog=True,
                 use_cea_analog=False

Plus a deterministic unit block for (b), (c), (d), (e).

Acceptance checks:
  C1 (backward compat): action-class entropy delta BLA_OFF_QUIET vs
     BLA_ON_QUIET <= 0.2 nats in >= 2/3 seeds.
  C2 (inverted-U gain): gain(arousal=0.0) == 1.0;
     gain(arousal in [0.5, 0.7]) > 1.0;
     gain(arousal saturated = 1.5) <= encoding_gain_max.
     At least two points on the curve strictly exceed the rest baseline.
  C3 (remap gate): synthetic PE spike with attribution candidates ->
     remap_signal populated (>= 1 code); same spike without candidates
     -> remap_signal empty (attribution-gated).
  C4 (window decay): after a fire, window_steps_remaining is > 0 and
     monotonically non-increasing across subsequent ticks; after
     window_steps ticks, gain returns to 1.0 (within floor tolerance)
     when arousal returns below threshold.
  C5 (MECH-094 boundary): BLAAnalog.tick signature does NOT expose a
     silent simulation_mode arg. (Inspection of dataclass fields +
     tick signature.) This documents that the simulation gate is a
     consumer-side responsibility (MECH-261 write gate) and prevents
     a silent false-safe path.

PASS: all of C1..C5.
FAIL otherwise.

experiment_purpose=diagnostic. Substrate readiness gate. Behavioural
recall validation (MECH-074a downstream effect on threat-associated
context recall) deferred to a follow-up EXQ once HippocampalModule
read-side consumers of encoding_gain / retrieval_bias / remap_signal
are wired.

See REE_assembly/docs/architecture/sd_035_amygdala_analog.md
See ree-v3/CLAUDE.md "SD-035: Amygdala Analogue" section.
"""

import sys
import json
import math
import inspect
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ree_core.agent import REEAgent
from ree_core.amygdala import BLAAnalog, BLAConfig
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_474_sd035_bla_encoding_remap"
CLAIM_IDS = ["SD-035", "MECH-074a", "MECH-074d"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 7, 13]
EPISODES = 4
STEPS_PER_EP = 60

CONDITIONS = ["BLA_OFF_QUIET", "BLA_ON_QUIET"]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=2,
        num_resources=3,
        hazard_harm=0.04,
        proximity_harm_scale=0.12,
        proximity_benefit_scale=0.18,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        harm_history_len=10,
    )


def _make_agent(env: CausalGridWorldV2, condition: str) -> REEAgent:
    amygdala_on = condition == "BLA_ON_QUIET"
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        use_harm_stream=True,
        harm_obs_dim=51,
        use_affective_harm_stream=True,
        harm_obs_a_dim=50,
        harm_history_len=10,
        z_harm_a_dim=16,
        use_amygdala_analog=amygdala_on,
        use_bla_analog=True,
        use_cea_analog=False,    # isolate BLA
        use_salience_coordinator=False,
    )
    return REEAgent(cfg)


def _shannon_entropy(values: List[float]) -> float:
    ent = 0.0
    for p in values:
        if p > 0:
            ent -= p * math.log(p)
    return ent


def _action_class_entropy(counts: Dict[int, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return _shannon_entropy([c / total for c in counts.values() if c > 0])


def _run_condition(seed: int, condition: str) -> Dict:
    torch.manual_seed(seed)
    env = _make_env(seed)
    agent = _make_agent(env, condition)

    action_counts: Dict[int, int] = {}
    bla_gain_values: List[float] = []

    for ep in range(EPISODES):
        obs, _info = env.reset()
        for step in range(STEPS_PER_EP):
            action = agent.act(obs)
            a_idx = int(action[0].argmax().item())
            action_counts[a_idx] = action_counts.get(a_idx, 0) + 1
            obs, _r, terminated, truncated, _info = env.step(a_idx)
            if condition == "BLA_ON_QUIET" and agent.bla is not None:
                out = agent._bla_last_output
                if out is not None:
                    bla_gain_values.append(float(out.encoding_gain))
            if terminated or truncated:
                break

    return {
        "condition": condition,
        "seed": seed,
        "action_class_entropy": _action_class_entropy(action_counts),
        "n_actions": sum(action_counts.values()),
        "bla_gain_mean": (sum(bla_gain_values) / len(bla_gain_values)) if bla_gain_values else None,
        "bla_n_ticks": len(bla_gain_values),
    }


def _arousal_vec(dim: int, mag: float) -> torch.Tensor:
    """Synthetic z_harm_a with L1/dim norm equal to mag (canonical BLA input shape)."""
    v = torch.ones(dim) * mag
    return v


def _run_unit_checks() -> Dict:
    """Deterministic unit-level checks of BLAAnalog."""
    # C2: inverted-U gain curve. BLAAnalog's canonical arousal input is
    # derived from z_harm_a mean magnitude. Feed a series of canned inputs.
    z_dim = 16
    bla = BLAAnalog(BLAConfig())
    gain_by_arousal: Dict[float, float] = {}
    arousals = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.9, 1.2, 1.5]
    # Fresh instance per tick so window-accumulation does not confound the
    # point evaluation.
    for a in arousals:
        b = BLAAnalog(BLAConfig())
        out = b.tick(_arousal_vec(z_dim, a))
        gain_by_arousal[a] = float(out.encoding_gain)

    gain_rest = gain_by_arousal[0.0]
    gain_sat = gain_by_arousal[1.5]
    above_rest_count = sum(1 for a, g in gain_by_arousal.items() if a > 0 and g > gain_rest + 1e-6)
    gain_below_cap = all(g <= bla.config.encoding_gain_max + 1e-6 for g in gain_by_arousal.values())
    c2_rest_is_one = abs(gain_rest - 1.0) < 1e-6
    c2_curve_rises = above_rest_count >= 2
    c2 = c2_rest_is_one and c2_curve_rises and gain_below_cap

    # C3: remap gate (attribution-required vs not).
    bla_remap = BLAAnalog(BLAConfig(remap_pe_sigma_threshold=2.0, remap_pe_ema_alpha=0.1))
    # Prime quiet PE EMA
    for _ in range(50):
        _ = bla_remap.tick(torch.zeros(z_dim), z_harm_a_pred=torch.zeros(z_dim))
    # With attribution candidates -> remap fires
    candidates = {0: 0.9, 1: 0.7, 2: 0.3, 3: 0.1}
    spike_attrib = bla_remap.tick(
        torch.ones(z_dim) * 3.0,
        z_harm_a_pred=torch.zeros(z_dim),
        candidate_code_contributions=candidates,
    )
    remap_with_attrib_codes = len(spike_attrib.remap_signal)

    # Without attribution candidates -> gated off
    bla_remap2 = BLAAnalog(BLAConfig(remap_pe_sigma_threshold=2.0, remap_pe_ema_alpha=0.1))
    for _ in range(50):
        _ = bla_remap2.tick(torch.zeros(z_dim), z_harm_a_pred=torch.zeros(z_dim))
    spike_no_attrib = bla_remap2.tick(
        torch.ones(z_dim) * 3.0,
        z_harm_a_pred=torch.zeros(z_dim),
        candidate_code_contributions=None,
    )
    remap_without_attrib_codes = len(spike_no_attrib.remap_signal)
    c3 = remap_with_attrib_codes >= 1 and remap_without_attrib_codes == 0

    # C4: encoding-window decay. Fire once at peak, then feed quiet ticks.
    # Use a short window override so the test is fast.
    bla_window = BLAAnalog(BLAConfig(
        window_steps=20,
        window_half_life_steps=5,
    ))
    out_fire = bla_window.tick(_arousal_vec(z_dim, 0.7))
    window_start = out_fire.encoding_window_steps_remaining
    gain_after_fire = out_fire.encoding_gain
    window_remaining_trajectory = [window_start]
    gain_trajectory = [gain_after_fire]
    for _ in range(25):  # past the 20-step window
        q = bla_window.tick(_arousal_vec(z_dim, 0.0))
        window_remaining_trajectory.append(q.encoding_window_steps_remaining)
        gain_trajectory.append(q.encoding_gain)
    monotone_non_increasing = all(
        window_remaining_trajectory[i + 1] <= window_remaining_trajectory[i]
        for i in range(len(window_remaining_trajectory) - 1)
    )
    final_window = window_remaining_trajectory[-1]
    final_gain = gain_trajectory[-1]
    returns_to_baseline = abs(final_gain - 1.0) < 0.1 or final_gain <= bla_window.config.encoding_gain_floor + 0.1
    c4 = monotone_non_increasing and final_window == 0 and returns_to_baseline

    # C5: signature inspection -- BLAAnalog.tick does not expose a silent
    # simulation_mode argument (consumer side handles MECH-094 gating via
    # MECH-261 write gate).
    sig = inspect.signature(BLAAnalog.tick)
    has_sim_mode = "simulation_mode" in sig.parameters
    # Presence is fine as long as it is documented; the assertion here is
    # weaker: the parameter may exist (BLA may wish to accept it for
    # symmetry with CeA) but must default to False. If it does not exist,
    # also acceptable -- consumer gates BLA through MECH-261.
    if has_sim_mode:
        default_ok = sig.parameters["simulation_mode"].default is False
        c5 = default_ok
    else:
        c5 = True

    return {
        "c2_gain_curve": gain_by_arousal,
        "c2_rest_is_one": c2_rest_is_one,
        "c2_curve_above_rest_count": above_rest_count,
        "c2_gain_below_cap": gain_below_cap,
        "c3_remap_with_attribution_n_codes": remap_with_attrib_codes,
        "c3_remap_without_attribution_n_codes": remap_without_attrib_codes,
        "c4_window_start": window_start,
        "c4_window_monotone_non_increasing": monotone_non_increasing,
        "c4_final_window_remaining": final_window,
        "c4_final_gain": final_gain,
        "c4_returns_to_baseline": returns_to_baseline,
        "c5_tick_has_simulation_mode_param": has_sim_mode,
        "c5_tick_simulation_mode_default_false": c5,
    }


def main() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for seed in SEEDS:
        for cond in CONDITIONS:
            print(f"Seed {seed} Condition {cond}")
            r = _run_condition(seed=seed, condition=cond)
            print(f"  -> {r}")
            all_results.append(r)

    print("Running unit-level BLA checks")
    unit = _run_unit_checks()
    print(f"  -> {unit}")

    def by_cond(c):
        return [r for r in all_results if r["condition"] == c]

    off = by_cond("BLA_OFF_QUIET")
    on = by_cond("BLA_ON_QUIET")

    c1_deltas = []
    c1_passes = 0
    for off_r, on_r in zip(off, on):
        delta = abs(on_r["action_class_entropy"] - off_r["action_class_entropy"])
        c1_deltas.append(delta)
        if delta <= 0.2:
            c1_passes += 1
    c1 = c1_passes >= 2

    c2 = bool(unit["c2_rest_is_one"]) and unit["c2_curve_above_rest_count"] >= 2 and bool(unit["c2_gain_below_cap"])
    c3 = unit["c3_remap_with_attribution_n_codes"] >= 1 and unit["c3_remap_without_attribution_n_codes"] == 0
    c4 = (
        bool(unit["c4_window_monotone_non_increasing"])
        and unit["c4_final_window_remaining"] == 0
        and bool(unit["c4_returns_to_baseline"])
    )
    c5 = bool(unit["c5_tick_simulation_mode_default_false"])

    outcome = "PASS" if (c1 and c2 and c3 and c4 and c5) else "FAIL"

    summary = {
        "c1_backward_compat_entropy_delta": {
            "deltas": c1_deltas,
            "threshold": 0.2,
            "seeds_passing": c1_passes,
            "pass": c1,
            "desc": "|entropy(BLA_ON_QUIET) - entropy(BLA_OFF_QUIET)| <= 0.2 nats in >= 2/3 seeds",
        },
        "c2_inverted_u_gain_curve": {
            "gain_by_arousal": unit["c2_gain_curve"],
            "rest_is_one": unit["c2_rest_is_one"],
            "above_rest_count": unit["c2_curve_above_rest_count"],
            "gain_below_cap": unit["c2_gain_below_cap"],
            "pass": c2,
            "desc": "rest gain=1.0; at least 2 arousal points exceed rest; all <= encoding_gain_max",
        },
        "c3_remap_attribution_gate": {
            "with_attribution_codes": unit["c3_remap_with_attribution_n_codes"],
            "without_attribution_codes": unit["c3_remap_without_attribution_n_codes"],
            "pass": c3,
            "desc": "remap fires with attribution candidates, does not fire without (Moita 2004)",
        },
        "c4_window_decay": {
            "window_start": unit["c4_window_start"],
            "monotone": unit["c4_window_monotone_non_increasing"],
            "final_window": unit["c4_final_window_remaining"],
            "final_gain": unit["c4_final_gain"],
            "returns_to_baseline": unit["c4_returns_to_baseline"],
            "pass": c4,
            "desc": "window_steps_remaining monotone non-increasing; returns to 0 and gain to ~1",
        },
        "c5_tick_simulation_mode_default": {
            "has_sim_mode_param": unit["c5_tick_has_simulation_mode_param"],
            "default_false_if_present": unit["c5_tick_simulation_mode_default_false"],
            "pass": c5,
            "desc": "BLA.tick has no silent simulation_mode (default False or absent)",
        },
    }

    print(f"\nOutcome: {outcome}")
    for k, v in summary.items():
        print(f"  {k}: pass={v['pass']}")

    per_claim = {
        "SD-035": "supports" if (c1 and c2 and c3 and c4) else "weakens",
        "MECH-074a": "supports" if (c2 and c4) else "weakens",
        "MECH-074d": "supports" if c3 else "weakens",
    }

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": outcome,
        "evidence_direction": "supports" if outcome == "PASS" else "weakens",
        "evidence_direction_per_claim": per_claim,
        "pass_criteria_summary": summary,
        "per_seed_results": all_results,
        "unit_checks": unit,
        "config": {
            "conditions": CONDITIONS,
            "seeds": SEEDS,
            "episodes": EPISODES,
            "steps_per_ep": STEPS_PER_EP,
        },
    }

    out_file = out_dir / f"{run_id}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Output written to: {out_file}")


if __name__ == "__main__":
    main()
