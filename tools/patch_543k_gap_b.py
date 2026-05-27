#!/opt/local/bin/python3
"""One-shot patch: v3_exq_543i copy -> V3-EXQ-543k GAP-B falsifier."""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "experiments" / "v3_exq_543i_arc062_differential_heads_falsifier.py"
DST = Path(__file__).resolve().parents[1] / "experiments" / "v3_exq_543k_arc062_mode_separation_gap_b_falsifier.py"

text = SRC.read_text(encoding="utf-8")

# Truncate docstring through SLEEP DRIVER line.
text = re.sub(
    r"^#!/opt/local/bin/python3\n.*?SLEEP DRIVER: not applicable.*?\n\"\"\"\n\n",
    '''#!/opt/local/bin/python3
"""V3-EXQ-543k: ARC-062 GAP-B mode-separation floor falsifier. Supersedes 543i.

EXPERIMENT_PURPOSE = evidence. Post-543i substrate retest: mode_separation_floor
on composed gated bias plus optional P1 w-deviation aux so discriminator w~0.5
does not cancel differential-head contrast under outcome-coupled REINFORCE.

WHY (failure_autopsy_V3-EXQ-543i_2026-05-19): use_differential_heads alone did NOT
stabilize basin selection. GAP-B adds floor*(h0-h1) on composed output and P1 aux.

Design: same 12-arm grid as 543i. Gated arms: floor=0.25, p1_w_deviation_aux=0.1.
K_IDENTICAL_RUNS=3 per (arm, seed) basin-stability gate; hostname in manifest.

PASS = basin_stable AND diff_on_escape AND diff_off_reproduced_collapse AND c2c3_on_pass.
No per-claim direction unless basin_stable.

SLEEP DRIVER: not applicable (no sleep loop in this experiment).
"""

''',
    text,
    count=1,
    flags=re.DOTALL,
)

text = text.replace(
    "import json\nimport os\nimport random",
    "import json\nimport os\nimport random\nimport socket",
    1,
)
text = text.replace(
    'EXPERIMENT_TYPE = "v3_exq_543i_arc062_differential_heads_falsifier"',
    'EXPERIMENT_TYPE = "v3_exq_543k_arc062_mode_separation_gap_b_falsifier"',
    1,
)
text = text.replace('QUEUE_ID = "V3-EXQ-543i"', 'QUEUE_ID = "V3-EXQ-543k"', 1)
text = text.replace('SUPERSEDES = "V3-EXQ-543h"', 'SUPERSEDES = "V3-EXQ-543i"', 1)
text = text.replace(
    'SUPERSEDES_CHAIN = ["V3-EXQ-543h", "V3-EXQ-543g"]',
    'SUPERSEDES_CHAIN = ["V3-EXQ-543i", "V3-EXQ-543h", "V3-EXQ-543g"]',
    1,
)

if "K_IDENTICAL_RUNS" not in text:
    text = text.replace(
        "DIFF_OFF_REPRO_MIN_INERT_SEEDS = 2\n",
        "DIFF_OFF_REPRO_MIN_INERT_SEEDS = 2\n\n"
        "MODE_SEPARATION_FLOOR = 0.25\n"
        "P1_W_DEVIATION_AUX_WEIGHT = 0.1\n"
        "K_IDENTICAL_RUNS = 3\n",
        1,
    )

old_cfg = """    config.gated_policy_use_differential_heads = bool(differential_heads)
    config.gated_policy_differential_bias_scale = 0.1

    agent = REEAgent(config)"""
new_cfg = """    config.gated_policy_use_differential_heads = bool(differential_heads)
    config.gated_policy_differential_bias_scale = 0.1
    if use_gated_policy:
        config.gated_policy_mode_separation_floor = MODE_SEPARATION_FLOOR
        config.gated_policy_p1_w_deviation_aux_weight = P1_W_DEVIATION_AUX_WEIGHT

    agent = REEAgent(config)"""
assert old_cfg in text, "make_agent block"
text = text.replace(old_cfg, new_cfg, 1)

old_loss = """    loss = reinforce_loss - LAMBDA_DISC_VAR * disc_var_term
    return loss, float(reinforce_loss.detach().item()), float(disc_var_term.detach().item())"""
new_loss = """    loss = reinforce_loss - LAMBDA_DISC_VAR * disc_var_term
    if (
        float(getattr(agent.gated_policy.config, "p1_w_deviation_aux_weight", 0.0)) > 0.0
        and len(disc_w_values) > 0
    ):
        loss = loss + agent.gated_policy.p1_training_auxiliary_loss(disc_w_values[-1])
    return loss, float(reinforce_loss.detach().item()), float(disc_var_term.detach().item())"""
assert old_loss in text, "loss block"
text = text.replace(old_loss, new_loss, 1)

marker = "    del agent_arm1, agent_arm2, _env1, _env2, agent_xtal, _env_x"
if "mode_separation_floor not wired" not in text:
    floor_check = """
    agent_floor, _env_fl = _make_agent_and_env(
        0, use_gated_policy=True, use_dacc=False,
        dacc_suppression_weight=0.0, differential_heads=True,
    )
    assert abs(agent_floor.gated_policy.config.mode_separation_floor - MODE_SEPARATION_FLOOR) < 1e-9
    assert abs(agent_floor.gated_policy.config.p1_w_deviation_aux_weight - P1_W_DEVIATION_AUX_WEIGHT) < 1e-9
    del agent_floor, _env_fl
"""
    text = text.replace(marker, floor_check + marker, 1)

if "_consensus_seed_result" not in text:
    helper = '''

def _consensus_seed_result(k_runs: List[Dict]) -> Dict:
    inert_flags = [bool(r.get("p1_inert_gating_detected", False)) for r in k_runs]
    consensus_inert = inert_flags[0] if len(set(inert_flags)) == 1 else None
    out = dict(k_runs[-1])
    if consensus_inert is not None:
        out["p1_inert_gating_detected"] = consensus_inert
    out["k_identical_runs"] = int(len(k_runs))
    out["k_inert_flags"] = inert_flags
    out["k_basin_unanimous"] = bool(len(set(inert_flags)) == 1)
    return out


def _basin_stable_all_gated(seed_results_by_arm: Dict[str, List[Dict]]) -> bool:
    gated = (
        "ARM_2_gated_only", "ARM_3_both", "ARM_6_gated_only_xtal", "ARM_7_both_xtal",
        "ARM_8_gated_only_diff", "ARM_9_both_diff",
        "ARM_10_gated_only_xtal_diff", "ARM_11_both_xtal_diff",
    )
    return all(
        all(bool(r.get("k_basin_unanimous", True)) for r in seed_results_by_arm.get(lbl, []))
        for lbl in gated
    )

'''
    text = text.replace("\ndef _aggregate_arm(seed_results: List[Dict]) -> Dict:", helper + "\ndef _aggregate_arm(seed_results: List[Dict]) -> Dict:", 1)

old_loop = """            seed_results_by_arm[arm_label].append(r)"""
# Only replace inside run() - use fuller context
old_loop = """            r = run_arm_seed(
                arm_label,
                use_gated_policy=use_gated,
                use_dacc=use_dacc,
                seed=seed,
                dry_run=dry_run,
                crystallize=use_xtal,
                differential_heads=use_diff,
            )
            seed_results_by_arm[arm_label].append(r)"""
new_loop = """            k_runs: List[Dict] = []
            for _k in range(K_IDENTICAL_RUNS):
                k_runs.append(
                    run_arm_seed(
                        arm_label,
                        use_gated_policy=use_gated,
                        use_dacc=use_dacc,
                        seed=seed,
                        dry_run=dry_run,
                        crystallize=use_xtal,
                        differential_heads=use_diff,
                    )
                )
            seed_results_by_arm[arm_label].append(_consensus_seed_result(k_runs))"""
assert old_loop in text, "run loop"
text = text.replace(old_loop, new_loop, 1)

old_primary = """    diff_primary_pass = bool(
        diff_on_escape and diff_off_reproduced_collapse and c2c3_on_pass
    )"""
new_primary = """    diff_primary_pass = bool(
        diff_on_escape and diff_off_reproduced_collapse and c2c3_on_pass
    )"""
# basin_stable computed in run(), not _compute_acceptance
assert old_primary in text

old_return = """    return {
        "arm_summaries": arm_summaries,
        "seed_results_by_arm": seed_results_by_arm,
        "acceptance": acceptance,
    }"""
new_return = """    acceptance["basin_stable"] = _basin_stable_all_gated(seed_results_by_arm)
    acceptance["overall_pass"] = bool(
        acceptance.get("diff_primary_pass")
        and acceptance["basin_stable"]
    )
    acceptance["diff_primary_pass"] = acceptance["overall_pass"]

    return {
        "arm_summaries": arm_summaries,
        "seed_results_by_arm": seed_results_by_arm,
        "acceptance": acceptance,
    }"""
assert old_return in text
text = text.replace(old_return, new_return, 1)

# per-claim direction: gate on basin_stable
old_pc = """    if not repro:
        outcome = "FAIL"
        per_claim = {
            "ARC-062": "non_contributory", "MECH-309": "non_contributory",
            "INV-074": "non_contributory", "MECH-334": "non_contributory",
        }
        branch = "c_diff_off_collapse_not_reproduced_substrate_drift"
    elif escape and c2c3:"""
new_pc = """    basin_stable = bool(acceptance.get("basin_stable", False))
    if not basin_stable:
        outcome = "FAIL"
        per_claim = {
            "ARC-062": "non_contributory", "MECH-309": "non_contributory",
            "INV-074": "non_contributory", "MECH-334": "non_contributory",
        }
        branch = "b_basin_unstable_nondeterministic_no_directional_read"
    elif not repro:
        outcome = "FAIL"
        per_claim = {
            "ARC-062": "non_contributory", "MECH-309": "non_contributory",
            "INV-074": "non_contributory", "MECH-334": "non_contributory",
        }
        branch = "c_diff_off_collapse_not_reproduced_substrate_drift"
    elif escape and c2c3:"""
assert old_pc in text
text = text.replace(old_pc, new_pc, 1)

# manifest hostname + substrate fields
text = text.replace(
    '"timestamp_utc": datetime.utcnow().isoformat() + "Z",',
    '"timestamp_utc": datetime.utcnow().isoformat() + "Z",\n'
    '        "hostname": socket.gethostname(),\n'
    '        "mode_separation_floor": MODE_SEPARATION_FLOOR,\n'
    '        "p1_w_deviation_aux_weight": P1_W_DEVIATION_AUX_WEIGHT,\n'
    '        "k_identical_runs": K_IDENTICAL_RUNS,',
    1,
)

text = text.replace("[V3-EXQ-543i]", "[V3-EXQ-543k]", 1)
text = text.replace("=== V3-EXQ-543i SUMMARY ===", "=== V3-EXQ-543k SUMMARY ===", 1)
text = text.replace(
    '        f"  [PRIMARY] diff_primary_pass={acc[\'diff_primary_pass\']}"',
    '        f"  basin_stable={acc.get(\'basin_stable\')}"\n'
    '        f"  [PRIMARY] diff_primary_pass={acc[\'diff_primary_pass\']}"',
    1,
)

DST.write_text(text, encoding="utf-8")
print("Wrote", DST, "bytes", DST.stat().st_size)
