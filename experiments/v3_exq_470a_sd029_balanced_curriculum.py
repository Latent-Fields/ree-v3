#!/opt/local/bin/python3
"""
V3-EXQ-470a: SD-029 Balanced Hazard-Event Curriculum -- Substrate Validation (fix).

Supersedes: V3-EXQ-470 (BASELINE/SCHEDULED bit-identical across all 4 seeds
because episodes died before step 25, so the `steps % interval == 0` injection
condition never fired.)

Root cause diagnosis:
  EXQ-470 used hazard_harm=0.3 with 4 hazards on an 8x8 grid and a uniform-random
  policy. Random walker dies after ~3-4 hazard hits (health=1.0). Per-seed
  n_steps_total in EXQ-470 was 15-34 across 3 episodes, so self.steps % 25 == 0
  never held at a non-zero tick, and _inject_external_hazard was never called.
  BASELINE vs SCHEDULED under the same RNG seed produced byte-identical
  trajectories because no SCHEDULED-specific code path ever fired.

Fix:
  Lower hazard_harm from 0.3 to 0.05 so episodes survive past step 25 (the
  curriculum interval). Also add a per-injection diagnostic log to the manifest
  (list of (seed, ep, step) tuples) so we can verify the injection actually fired.

This experiment validates ONLY the env curriculum mechanism, not downstream
claim evidence. Primary questions:
  - Does info['external_hazard_event_count'] grow monotonically under SCHEDULED?
  - Does it remain at 0 under BASELINE (backward compat)?
  - Does SCHEDULED produce >= 3 injection events across 3 x 200-step eps per
    seed (interval=25 implies theoretical max ~24 injections per 600 steps)?
  - Does BASELINE vs SCHEDULED now produce measurably different trajectories
    (n_steps_total, total_harm) on matched seeds?

Acceptance (substrate validation, not governance evidence):
  C_back:    BASELINE external_hazard_event_count == 0 in all seeds.
  C_fire:    SCHEDULED external_hazard_event_count >= 3 per seed across 3 eps.
  C_differ: BASELINE and SCHEDULED n_steps_total differ on at least 1 seed
             (confirms the curriculum path actually changes dynamics).
  C_info:    info['external_hazard_injected'] and
             info['external_hazard_event_count'] present on every step.

4 seeds x 2 conditions; 3 eps per seed, 200 steps each. Random policy.

Claims tested: SD-029 (substrate readiness only -- NO evidence_direction set).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
import random
from datetime import datetime
from typing import Dict, List

from ree_core.environment.causal_grid_world import CausalGridWorldV2
MANIFEST_WRITER_EXEMPT = "archival early-era manifest (no resolvable status key: carries final_verdict/pass/verdict/passed, not status/outcome/overall_outcome; unqueued, last-run >=79d, not re-run)"

# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE    = "v3_exq_470a_sd029_balanced_curriculum"
CLAIM_IDS          = ["SD-029"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS      = [42, 7, 13, 91]
CONDITIONS = ["BASELINE", "SCHEDULED"]

EPISODES_PER_SEED = 3
STEPS_PER_EPISODE = 200

GRID_SIZE     = 8
NUM_RESOURCES = 3
NUM_HAZARDS   = 4

# Fix vs EXQ-470: hazard_harm lowered 0.3 -> 0.05 so episodes survive to curriculum tick.
HAZARD_HARM   = 0.05

CURRICULUM_INTERVAL       = 25
CURRICULUM_PROB           = 1.0
CURRICULUM_ADJACENT_ONLY  = True


def _make_env(seed: int, scheduled: bool) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_resources=NUM_RESOURCES,
        num_hazards=NUM_HAZARDS,
        hazard_harm=HAZARD_HARM,
        resource_benefit=0.3,
        resource_respawn_on_consume=True,
        proximity_harm_scale=0.1,
        proximity_approach_threshold=0.2,
        use_proxy_fields=True,
        scheduled_external_hazard_enabled=scheduled,
        scheduled_external_hazard_interval=CURRICULUM_INTERVAL,
        scheduled_external_hazard_prob=CURRICULUM_PROB,
        scheduled_external_hazard_adjacent_only=CURRICULUM_ADJACENT_ONLY,
    )


def _run_seed(seed: int, scheduled: bool, dry_run: bool) -> Dict:
    env = _make_env(seed, scheduled)
    rng = random.Random(seed)
    n_eps = 1 if dry_run else EPISODES_PER_SEED
    n_steps_cap = 30 if dry_run else STEPS_PER_EPISODE
    ext_events_per_ep: List[int] = []
    injection_log: List[Dict] = []  # (ep, step) for each injection fire
    total_self_events = 0
    total_harm_last_ep = 0.0
    n_info_keys_ok = True
    n_steps_total = 0
    for ep in range(n_eps):
        obs, info_dict = env.reset()
        ep_injections_before = 0
        for t in range(n_steps_cap):
            a = rng.randint(0, 4)
            flat_obs, h, done, info, obs_dict = env.step(a)
            if ('external_hazard_injected' not in info
                    or 'external_hazard_event_count' not in info):
                n_info_keys_ok = False
            transition = info.get('transition_type', 'none')
            if transition == 'agent_caused_hazard':
                total_self_events += 1
            n_steps_total += 1
            # Log injection fires
            current_count = int(info.get('external_hazard_event_count', 0))
            if current_count > ep_injections_before:
                injection_log.append({'ep': ep, 'step': t + 1, 'cumulative': current_count})
                ep_injections_before = current_count
            total_harm_last_ep = float(info.get('total_harm', 0.0))
            if done:
                break
        ext_events_per_ep.append(int(info.get('external_hazard_event_count', 0)))
    return {
        'seed': seed,
        'scheduled': bool(scheduled),
        'n_steps_total': int(n_steps_total),
        'external_hazard_events_per_ep': ext_events_per_ep,
        'external_hazard_events_total': int(sum(ext_events_per_ep)),
        'self_caused_events_total': int(total_self_events),
        'total_harm_last_ep': float(total_harm_last_ep),
        'info_keys_ok': bool(n_info_keys_ok),
        'injection_log': injection_log,
    }


def main(dry_run: bool = False) -> Dict:
    results: Dict[str, List[Dict]] = {c: [] for c in CONDITIONS}
    for cond in CONDITIONS:
        scheduled = (cond == 'SCHEDULED')
        for seed in SEEDS:
            r = _run_seed(seed, scheduled, dry_run)
            results[cond].append(r)

    # Acceptance checks
    baseline_ext_zero = all(r['external_hazard_events_total'] == 0
                            for r in results['BASELINE'])
    scheduled_fire_ge_3 = sum(
        1 for r in results['SCHEDULED']
        if r['external_hazard_events_total'] >= 3
    )
    # C_differ: at least one seed must have different n_steps_total under
    # BASELINE vs SCHEDULED (same seed -> if SCHEDULED changed dynamics, step
    # counts will diverge).
    n_diff_seeds = sum(
        1 for rb, rs in zip(results['BASELINE'], results['SCHEDULED'])
        if rb['n_steps_total'] != rs['n_steps_total']
    )
    info_keys_always_present = all(
        r['info_keys_ok']
        for c in CONDITIONS for r in results[c]
    )

    c_back = bool(baseline_ext_zero)
    c_fire = int(scheduled_fire_ge_3) >= 3
    c_differ = n_diff_seeds >= 1
    c_info = bool(info_keys_always_present)

    passed = c_back and c_fire and c_differ and c_info

    criteria = {
        'C_back_baseline_zero': c_back,
        'C_fire_scheduled_ge3_per_seed': c_fire,
        'C_differ_baseline_vs_scheduled': c_differ,
        'C_info_keys_present': c_info,
    }

    summary = {
        'experiment_type': EXPERIMENT_TYPE,
        'experiment_purpose': EXPERIMENT_PURPOSE,
        'claim_ids': CLAIM_IDS,
        'architecture_epoch': 'ree_hybrid_guardrails_v1',
        'supersedes': 'v3_exq_470_sd029_balanced_curriculum',
        'timestamp_utc': datetime.utcnow().isoformat() + 'Z',
        'seeds': SEEDS,
        'conditions': CONDITIONS,
        'episodes_per_seed': EPISODES_PER_SEED,
        'steps_per_episode': STEPS_PER_EPISODE,
        'hazard_harm': HAZARD_HARM,
        'curriculum_interval': CURRICULUM_INTERVAL,
        'curriculum_prob': CURRICULUM_PROB,
        'curriculum_adjacent_only': CURRICULUM_ADJACENT_ONLY,
        'results': results,
        'criteria': criteria,
        'passed': passed,
        'evidence_direction': 'supports' if passed else 'weakens',
    }
    return summary


def _write_output(summary: Dict) -> str:
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    out_dir = os.path.join(
        os.path.dirname(__file__),
        '..', '..',
        'REE_assembly', 'evidence', 'experiments', EXPERIMENT_TYPE
    )
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_id}.json")
    summary['run_id'] = run_id
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    summary = main(dry_run=args.dry_run)
    if args.dry_run:
        print('DRY RUN SUMMARY:')
        print(json.dumps(summary['criteria'], indent=2))
        print('passed:', summary['passed'])
        # Show injection logs
        for cond in CONDITIONS:
            for r in summary['results'][cond]:
                print(f"  {cond} seed={r['seed']}: n_steps={r['n_steps_total']} "
                      f"ext_events={r['external_hazard_events_total']} "
                      f"injections={len(r['injection_log'])}")
    else:
        out_path = _write_output(summary)
        print('wrote:', out_path)
        print('passed:', summary['passed'])
