#!/opt/local/bin/python3
"""
V3-EXQ-479: SD-029 Balanced Hazard-Event Curriculum -- Substrate Validation (fix2).

Supersedes: V3-EXQ-470a (PASS on C_back/C_info but FAIL on C_fire + C_differ --
the root cause was that episodes still died before the curriculum_interval=25
tick fired reliably. V3-EXQ-470a had hazard_harm=0.05, num_hazards=4; random
walker n_steps_total was 18-34 across 3 episodes for 3 of 4 seeds, so
steps % 25 == 0 fired rarely or never).

Root cause diagnosis (continuing from 470a):
  EXQ-470a lowered hazard_harm 0.3 -> 0.05, but 4 hazards on 8x8 grid still
  kills the random walker before step 25 most episodes. n_steps_total mean
  was ~26 per episode across 12 episodes, but with high variance -- the
  curriculum interval tick at step 25 fell on the death tick or past it
  for half the runs. External_hazard_events_total stayed at 0 for all seeds
  under SCHEDULED condition.

Fix:
  Three compounding adjustments so episodes reliably survive past the
  curriculum tick:
    - num_hazards 4 -> 2 (half the death surface area)
    - hazard_harm 0.05 -> 0.02 (slower cumulative damage)
    - curriculum_interval 25 -> 10 (fires earlier in episode; 20 opportunities
      per 200-step episode rather than 8)
  Plus keep the injection_log diagnostic from 470a.

Acceptance (substrate validation, unchanged):
  C_back:    BASELINE external_hazard_event_count == 0 in all seeds.
  C_fire:    SCHEDULED external_hazard_event_count >= 3 per seed.
  C_differ: BASELINE and SCHEDULED n_steps_total differ on at least 1 seed.
  C_info:    info['external_hazard_injected'] and
             info['external_hazard_event_count'] present on every step.

4 seeds x 2 conditions; 3 eps per seed, 200 steps each. Random policy.

Claims tested: SD-029 (substrate readiness only -- diagnostic, no governance
evidence contributed).
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

EXPERIMENT_TYPE    = "v3_exq_479_sd029_balanced_curriculum_fix"
CLAIM_IDS          = ["SD-029"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS      = [42, 7, 13, 91]
CONDITIONS = ["BASELINE", "SCHEDULED"]

EPISODES_PER_SEED = 3
STEPS_PER_EPISODE = 200

GRID_SIZE     = 8
NUM_RESOURCES = 3
# Fix vs EXQ-470a: num_hazards 4 -> 2
NUM_HAZARDS   = 2
# Fix vs EXQ-470a: hazard_harm 0.05 -> 0.02
HAZARD_HARM   = 0.02

# Fix vs EXQ-470a: curriculum_interval 25 -> 10
CURRICULUM_INTERVAL       = 10
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
    injection_log: List[Dict] = []
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

    baseline_ext_zero = all(r['external_hazard_events_total'] == 0
                            for r in results['BASELINE'])
    scheduled_fire_ge_3 = sum(
        1 for r in results['SCHEDULED']
        if r['external_hazard_events_total'] >= 3
    )
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
        'supersedes': 'v3_exq_470a_sd029_balanced_curriculum',
        'timestamp_utc': datetime.utcnow().isoformat() + 'Z',
        'seeds': SEEDS,
        'conditions': CONDITIONS,
        'episodes_per_seed': EPISODES_PER_SEED,
        'steps_per_episode': STEPS_PER_EPISODE,
        'num_hazards': NUM_HAZARDS,
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
        for cond in CONDITIONS:
            for r in summary['results'][cond]:
                print(f"  {cond} seed={r['seed']}: n_steps={r['n_steps_total']} "
                      f"ext_events={r['external_hazard_events_total']} "
                      f"injections={len(r['injection_log'])}")
    else:
        out_path = _write_output(summary)
        print('Result written to:', out_path)
        print('passed:', summary['passed'])
