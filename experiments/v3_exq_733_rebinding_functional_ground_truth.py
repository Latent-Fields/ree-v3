#!/opt/local/bin/python3
"""
V3-EXQ-733 -- FUNCTIONAL rebinding-under-perturbation test (MECH-456).

Gates promotion of MECH-456 (entities.rebinding_under_perturbation;
candidate / substrate_conditional / v3_pending, registered 2026-07-10 via
/claim-synthesis, salvaged from failure_autopsy_V3-EXQ-725a_2026-07-10 sec 8).
Substrate ALREADY BUILT + CONVERGED: the learned cross_stream_binder
(ree-v3/ree_core/latent/cross_stream_binder.py) + binding_score/rebinding_probe.
NO ree_core change -- harness-level measurement over the trained-then-frozen binder.

WHAT THIS TESTS (and what it deliberately does NOT)
---------------------------------------------------
MECH-456 asserts an established self-anchor <-> world-configuration binding is
NOT fixed: when a competing configuration overtakes the currently-bound one, the
binder re-binds to the competitor. V3-EXQ-725a observed n_rebind_total=1676
(0 in all predecessors) -- but that is EXERCISABILITY (a trained bilinear scorer's
argmax CAN flip under anchor perturbation), NOT a demonstration that rebinding
does behavioural work. The sibling MECH-269(b) anchor-reset precedent is
load-bearing: V3-EXQ-478/480 showed resets fire abundantly but
freeze_recommit_count was bit-identical OFF=ON (inert). So a claim-tagged
FUNCTIONAL test is owed, and this experiment is deliberately NOT passable by
n_rebind>0 alone.

This test is DECOUPLED from the coherence-SPECIFICITY question (candidate Q
entities/selection.coherence_nonreducibility), which is settled NO-CLAIM by
V3-EXQ-725a (SPEC 1/6; divergence reproduced by a contrast-matched shuffle).
There is NO C(tau) coherence term, no shuffle-of-coherence, no gap-relative
coherence authority here. This is purely about whether the binder's argmax
binding-affinity tracks the TRUE current world-configuration (ground-truth) and
whether re-binding vs a frozen binding produces a graded behavioural consequence.

MECH-456 what_would_answer (the two mandatory conditions this satisfies)
------------------------------------------------------------------------
(1) The binder re-binds to the newly-correct configuration -- rebinding tracks
    the TRUE competitor (not arbitrary anchor noise), measured by
    rebinding-ALIGNMENT-WITH-GROUND-TRUTH above a SHUFFLE control.
(2) A behavioural / competence consequence -- rebinding-ON vs rebinding-FROZEN
    produces a downstream outcome difference (faster re-acquisition, fewer
    stale-binding errors) on a GRADED, NON-SATURATING metric (the 725a
    frac_state_div saturated on 3/6 seeds -- explicitly NOT reused here).
UNSUPPORTED if rebinding events fire but ON==FROZEN on the behavioural DV
(the MECH-269(b) V3-EXQ-478 inert signature), OR if rebinding does not track the
true competitor above the shuffle control (arbitrary anchor-sensitivity).

DESIGN -- genuine ground-truthed competing-configuration overtake
-----------------------------------------------------------------
GROUND TRUTH (binder-independent). The agent moves through a size x size grid.
Partition it into a G x G coarse lattice -> K = G*G spatial REGIONS. The
ground-truth current world-configuration g(t) is the region the agent's position
(env.agent_x, env.agent_y) falls in -- read directly off the env, NOTHING the
binder produces. Because the world_state channels (local view, hazard/resource
gradient fields, reef scent) and the self-state (harm/benefit exposure) vary with
position, each region has a distinct z_world signature AND a distinct z_self
anchor. As the agent crosses a region boundary the ground-truth config CHANGES
mid-episode -- a genuine competing-configuration OVERTAKE.

PROTOTYPE CONFIG BANK. During P0 warmup, accumulate proto_k = running mean of the
OBSERVED z_world (latent.z_world from agent.sense, the SAME distribution the
binder's contrastive curriculum trains on) while the agent is in region k. Frozen
at P1 start. The candidate pool at every P1 step is the K prototypes; the
ground-truth-correct index is the current region g(t).

BINDER READOUT. b_hat(anchor) = argmax_k binding_score(anchor, proto_k). The
learned binder scores cos(phi_self(anchor), phi_world(proto_k)); the argmax is a
genuine function of the anchor's direction, so an anchor that carries region
information picks the correct prototype and an anchor that does not cannot.

TWO REBINDING RULES (the "arms" -- eval-time readouts over ONE shared trajectory
per seed, so the ground-truth stream g(t) is IDENTICAL across arms; any DV
difference is PURELY the rebinding rule, maximally controlled):
  - REBIND-ON  : b_on(t) = b_hat(z_self_t)   -- re-evaluated with the LIVE anchor
                 every step (re-binds whenever a competitor overtakes).
  - REBIND-FROZEN: b_frozen(t) = b_hat(z_self_episode_entry) -- the binding is set
                 once at episode entry and NEVER re-evaluated (rebinding disabled;
                 the MECH-269(b) freeze_recommit analogue at the entity locus).

MEASUREMENTS
------------
DV1 (rebinding tracks the TRUE competitor -- condition (1)):
  A_real  = frac of P1 steps where b_on(t) == g(t).
  A_shuf  = frac where b_on(t) == sigma(g(t)), averaged over N_SHUFFLE_PERMS random
            NON-identity relabelings sigma of the region labels (the chance
            baseline; the direct analogue of the 725a shuffle-of-real control:
            same argmax computation, ground-truth correspondence destroyed).
  A_anchor_shuf = frac where b_hat(z_self_pi(t)) == g(t) over random timestep
            permutations pi (report only; isolates "the CORRECT anchor matters",
            not just any anchor -- the "not arbitrary anchor noise" phrasing).
  Post-overtake alignment (report): A_real restricted to the REACQ_WINDOW steps
            immediately following each overtake -- does ON specifically FOLLOW the
            new region above shuffle.
  DV1 per seed = (A_real >= A_shuf + ALIGN_MARGIN).

DV2 (behavioural / competence consequence -- condition (2)):
  Overtake event = a P1 step where g(t) != g(t-1) within an episode.
  Re-acquisition latency after an overtake at t0 = min{d>=0 : b(t0+d) == g(t0+d)},
  CENSORED at min(REACQ_WINDOW, steps-left-in-episode). Graded (real-valued step
  counts), NON-saturating (a latency, not a bounded fraction; the 725a saturation
  note is respected -- frac_state_div is NOT used).
    mean_lat_on     = mean re-acq latency (REBIND-ON) over all overtakes.
    mean_lat_frozen = mean re-acq latency (REBIND-FROZEN) over all overtakes.
  Stale-binding-error rate (context / corroboration): frac of P1 steps where the
  committed binding != g(t), for each arm.
  DV2 per seed = (mean_lat_frozen - mean_lat_on >= LATENCY_MARGIN)  -- ON
  re-acquires the correct binding meaningfully FASTER than a frozen binding.

RUN PASS = DV1 on >= MIN_SEEDS_FOR_PASS seeds AND DV2 on >= MIN_SEEDS_FOR_PASS
seeds (both conditions of what_would_answer met), with the substrate READINESS
gate satisfied (see below).

READINESS (P0 gate -- an unconverged binder is the 725 untrained-substrate
artifact, NOT a rebinding verdict; inadequate region coverage / too few overtakes
starves the measurement):
  - binder_converged (loss_ema < conv_frac*log(batch)) on EVERY seed.
  - every region visited >= MIN_REGION_VISITS_P0 times in P0 (stable prototype).
  - >= MIN_OVERTAKES_P1 overtake events per seed in P1.
If not ready on any seed -> self-route substrate_not_ready_requeue (evidence_
direction non_contributory), NEVER a rebinding verdict.

INTERPRETATION GRID (one row per outcome -> next action)
  functional_rebinding_supported        (DV1 AND DV2 on >= MIN_SEEDS, ready)
      -> Rebinding tracks the true competitor above shuffle AND re-binding
         produces a graded behavioural advantage over a frozen binding.
         evidence: SUPPORTS MECH-456. Route /governance to move MECH-456 candidate
         -> shown (clear v3_pending). Next: /implement-substrate wiring the
         binding-tracking readout into the agent's live config identification.
  rebinding_inert_off_equals_on          (DV1 holds, DV2 fails -- MECH-269(b) signature)
      -> The binder tracks ground truth, but re-binding confers NO behavioural
         advantage over a frozen binding (ON==FROZEN on the graded DV). This is the
         V3-EXQ-478 inert-reset signature at the entity locus. evidence: WEAKENS
         (does not support promotion). Route /failure-autopsy.
  rebinding_not_tracking_truth           (DV1 fails)
      -> Alignment is not above the shuffle control -> arbitrary anchor-sensitivity,
         NOT functional rebinding (n_rebind>0 was exercisability only). evidence:
         WEAKENS. Route /failure-autopsy.
  substrate_not_ready_requeue            (readiness gate unmet)
      -> Unconverged binder / thin region coverage / too few overtakes. NOT a
         verdict; re-queue at adequate P0 / more P1 steps.

PHASED TRAINING (mandatory, satisfied). The learned binder requires a P0
contrastive curriculum: during P0 warmup ONLY, each step calls
agent.update_cross_stream_binder(z_self, z_world) (one InfoNCE step on DETACHED
observed pairs -- no gradient into E1/E2 encoders). In P1 the binder is FROZEN
(the update is not called; agent.eval() is set) and the measurement reads the
trained-then-frozen binder. Locomotion is the natural main-path policy
(agent.select_action) in BOTH phases and is IDENTICAL across the two rebinding
readouts within a seed, so the readouts differ only in the rebinding rule.

Claims: [MECH-456] (experiment_purpose=evidence; claim-tagged functional test).
Bears on (cited, NOT tagged): MECH-269 (sibling anchor-reset, different locus),
MECH-270 (ephaptic binder substrate), ARC-006 (bindable entities), MECH-045
(object-file persistence), INV-002 (temporal binding).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from experiment_protocol import emit_outcome
from experiments._metrics import check_degeneracy
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_733_rebinding_functional_ground_truth"
QUEUE_ID = "V3-EXQ-733"
CLAIM_IDS: List[str] = ["MECH-456"]
EXPERIMENT_PURPOSE = "evidence"

# cross_stream_binding_substrate activation -- the CONVERGED learned binder
# (cosine InfoNCE, ree-v3 b86fa3e). strength/temperature/conv_frac carried from
# the converged V3-EXQ-725a config (NOT tuned to a pass here).
CROSS_STREAM_BINDING_ENABLED = True
CROSS_STREAM_BINDING_LEARNED = True
CROSS_STREAM_BINDING_STRENGTH = 0.5
CROSS_STREAM_BINDING_TEMPERATURE = 0.2
CROSS_STREAM_BINDING_CONV_FRAC = 0.85

# Ground-truth regime lattice: G x G partition of the size x size grid -> K regions.
G_PARTITION = 2
K_REGIONS = G_PARTITION * G_PARTITION

SEEDS = [42, 43, 44, 45, 46, 47]
P0_WARMUP_EPISODES = 40
P1_MEASUREMENT_EPISODES = 25
TOTAL_TRAIN_EPISODES = P0_WARMUP_EPISODES + P1_MEASUREMENT_EPISODES
STEPS_PER_EPISODE = 120

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 3
DRY_RUN_P1 = 3
DRY_RUN_STEPS = 40

# Acceptance thresholds (PRE-REGISTERED; never derived from this run's own stats).
ALIGN_MARGIN = 0.10          # DV1: real alignment must beat the shuffle chance
                             # baseline by >= 10 percentage points (per seed).
LATENCY_MARGIN = 2.0         # DV2: frozen re-acq latency must exceed ON by >= 2
                             # steps (per seed) -- ON re-binds meaningfully faster.
REACQ_WINDOW = 20            # censor cap (steps) for re-acq latency + the
                             # post-overtake alignment window.
N_SHUFFLE_PERMS = 200        # random region relabelings for the chance baseline.

MIN_SEEDS_FOR_PASS = 4       # of 6 seeds must satisfy DV1 (and DV2) -- 2/3 ratio.
MIN_SEEDS_COMPLETED = 4      # of 6 runs must reach P1 without error.

# Readiness floors.
MIN_REGION_VISITS_P0 = 10    # each region must be visited >= this in P0 for a
                             # stable prototype.
MIN_OVERTAKES_P1 = 20        # per seed: enough overtake events for latency stats.

# Reef-bipartite SD-054 env wiring (identical to V3-EXQ-725a) -- gives spatial
# structure (reef half vs food half + gradient fields) so regions carry distinct
# z_world/z_self signatures and the agent traverses region boundaries.
ENV_KWARGS = dict(
    size=12,
    num_hazards=4,
    num_resources=5,
    hazard_harm=0.05,
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    toroidal=False,
    harm_history_len=10,
    reef_enabled=True,
    n_reef_patches=3,
    reef_patch_radius=2,
    hazard_food_attraction=0.7,
    reef_bipartite_layout=True,
    reef_bipartite_axis="horizontal",
    reef_bipartite_agent_band_radius=1,
)


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    """Main-path SP-CEM + MECH-313 + V_s + SD-054 stack with the CONVERGED
    learned cross_stream_binder engaged. No substrate change (identical config
    surface to V3-EXQ-725a apart from carrying only the binder activation)."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        use_harm_stream=True,
        z_harm_dim=32,
        use_affective_harm_stream=True,
        z_harm_a_dim=16,
        harm_history_len=10,
        z_goal_enabled=True,
        goal_weight=0.5,
        drive_weight=2.0,
        e1_goal_conditioned=True,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        benefit_eval_enabled=True,
        benefit_weight=1.0,
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        use_noise_floor=True,
        noise_floor_alpha=0.1,
        use_per_stream_vs=True,
        use_per_region_vs=True,
        use_event_segmenter=True,
        use_invalidation_trigger=True,
        use_anchor_sets=True,
        cross_stream_binding_enabled=CROSS_STREAM_BINDING_ENABLED,
        cross_stream_binding_learned=CROSS_STREAM_BINDING_LEARNED,
        cross_stream_binding_strength=CROSS_STREAM_BINDING_STRENGTH,
        cross_stream_binding_temperature=CROSS_STREAM_BINDING_TEMPERATURE,
        cross_stream_binding_conv_frac=CROSS_STREAM_BINDING_CONV_FRAC,
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# obs helpers (mirror V3-EXQ-725a / 641)
# ---------------------------------------------------------------------------


def _obs_harm(obs_dict):
    h = obs_dict.get("harm_obs")
    return h.float().unsqueeze(0) if h is not None else None


def _obs_harm_a(obs_dict):
    h = obs_dict.get("harm_obs_a")
    return h.float().unsqueeze(0) if h is not None else None


def _obs_harm_history(obs_dict):
    h = obs_dict.get("harm_history")
    return h.float().unsqueeze(0) if h is not None else None


# ---------------------------------------------------------------------------
# Ground-truth region (binder-independent) + prototype bank
# ---------------------------------------------------------------------------


def _region_of(env: CausalGridWorldV2) -> int:
    """Ground-truth world-configuration label = the G x G lattice region the
    agent's position falls in. Read directly off the env (agent_x/agent_y);
    NOTHING the binder produces enters this."""
    s = int(env.size)
    g = G_PARTITION
    rx = min(g - 1, (int(env.agent_x) * g) // s)
    ry = min(g - 1, (int(env.agent_y) * g) // s)
    return rx * g + ry


def _argmax_binding(binder, anchor: torch.Tensor, protos: List[torch.Tensor]) -> Tuple[int, List[float]]:
    """b_hat = argmax_k binding_score(anchor, proto_k). anchor [1,self_dim];
    protos: list of [1,world_dim]. Returns (index, per-proto scores)."""
    with torch.no_grad():
        scores = [float(binder.binding_score(anchor, p).item()) for p in protos]
    return max(range(len(scores)), key=lambda i: scores[i]), scores


# ---------------------------------------------------------------------------
# One natural-locomotion decision, exposing the observed latent
# ---------------------------------------------------------------------------


def _step_decision(
    agent: REEAgent,
    obs_dict,
    zself_prev: Optional[torch.Tensor],
    act_prev: Optional[torch.Tensor],
    is_p1: bool,
):
    """Natural main-path decision (sense -> generate -> select_action). Returns
    (action[1,A], z_self_now[1,self_dim], z_world_now[1,world_dim]). During P0
    (is_p1=False) runs one binder curriculum step on the observed joint state;
    in P1 the binder is FROZEN (no update). Online E2 self-model + z_goal updates
    fire in both phases (identical to the V3-EXQ-725a discipline)."""
    body = obs_dict["body_state"].float()
    world = obs_dict["world_state"].float()
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)

    latent = agent.sense(
        obs_body=body, obs_world=world,
        obs_harm=_obs_harm(obs_dict),
        obs_harm_a=_obs_harm_a(obs_dict),
        obs_harm_history=_obs_harm_history(obs_dict),
    )
    z_self_now = latent.z_self.detach()
    z_world_now = latent.z_world.detach()
    if zself_prev is not None and act_prev is not None:
        agent.record_transition(zself_prev, act_prev, z_self_now)

    # P0 binder curriculum only; FROZEN in P1.
    if not is_p1:
        agent.update_cross_stream_binder(latent.z_self, latent.z_world)

    ticks = agent.clock.advance()
    wdim = latent.z_world.shape[-1]
    e1_prior = (
        agent._e1_tick(latent) if ticks.get("e1_tick", False)
        else torch.zeros(1, wdim, device=agent.device)
    )
    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
    action = agent.select_action(candidates, ticks)
    return action, z_self_now, z_world_now


# ---------------------------------------------------------------------------
# Alignment + latency helpers (over ONE seed's recorded P1 stream)
# ---------------------------------------------------------------------------


def _label_shuffle_alignment(
    b_on: List[int], gts: List[int], gen: torch.Generator
) -> float:
    """Chance baseline: mean over N_SHUFFLE_PERMS random NON-identity relabelings
    sigma of frac_t 1[b_on[t] == sigma(gts[t])]. Same argmax outputs, ground-truth
    correspondence destroyed (analogue of the 725a shuffle-of-real control)."""
    n = len(gts)
    if n == 0:
        return 0.0
    total = 0.0
    perms = 0
    for _ in range(N_SHUFFLE_PERMS):
        perm = torch.randperm(K_REGIONS, generator=gen).tolist()
        if perm == list(range(K_REGIONS)):
            continue  # skip identity so the baseline is a genuine shuffle
        agree = sum(1 for t in range(n) if b_on[t] == perm[gts[t]])
        total += agree / n
        perms += 1
    return (total / perms) if perms else 0.0


def _anchor_shuffle_alignment(
    binder, anchors: List[torch.Tensor], protos: List[torch.Tensor],
    gts: List[int], gen: torch.Generator,
) -> float:
    """Report-only: b_hat(z_self_pi(t)) vs g(t) over one random timestep
    permutation pi -- isolates whether the CORRECT anchor matters (functional)
    vs any anchor (arbitrary anchor-sensitivity)."""
    n = len(gts)
    if n < 2:
        return 0.0
    perm = torch.randperm(n, generator=gen).tolist()
    agree = 0
    for t in range(n):
        b, _ = _argmax_binding(binder, anchors[perm[t]], protos)
        if b == gts[t]:
            agree += 1
    return agree / n


def _reacq_latencies(
    committed: List[int], gts: List[int], ep_ids: List[int]
) -> List[int]:
    """Re-acquisition latency after each overtake. committed[t] = the arm's
    committed binding at step t; gts[t] = ground truth; ep_ids[t] = episode id
    (latency search does not cross episode boundaries). An overtake at t is a
    within-episode ground-truth change (gts[t] != gts[t-1]); latency = steps until
    committed==gts again, censored at REACQ_WINDOW or episode end."""
    lats: List[int] = []
    n = len(gts)
    for t in range(1, n):
        if ep_ids[t] != ep_ids[t - 1]:
            continue  # new episode -- not an in-episode overtake
        if gts[t] == gts[t - 1]:
            continue  # no overtake
        # Overtake at t: find first d>=0 with committed[t+d]==gts[t+d], same episode.
        cap = REACQ_WINDOW
        d = 0
        lat = cap
        while d < cap and (t + d) < n and ep_ids[t + d] == ep_ids[t]:
            if committed[t + d] == gts[t + d]:
                lat = d
                break
            d += 1
        lats.append(lat)
    return lats


# ---------------------------------------------------------------------------
# One seed: P0 warmup (train binder + build prototypes) then P1 measurement
# ---------------------------------------------------------------------------


def _run_seed(
    seed: int,
    p0_episodes: int,
    p1_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    total_train_eps = p0_episodes + p1_episodes
    error_note: Optional[str] = None

    torch.manual_seed(seed)
    env = _make_env(seed)
    agent = _make_agent(env)
    agent.eval()

    world_dim = int(agent.config.latent.world_dim)
    # Prototype accumulators (running sum + count per region), P0 only.
    proto_sum = [torch.zeros(1, world_dim) for _ in range(K_REGIONS)]
    proto_cnt = [0 for _ in range(K_REGIONS)]
    global_zworld_sum = torch.zeros(1, world_dim)
    global_zworld_cnt = 0

    # P1 recorded stream.
    p1_anchors: List[torch.Tensor] = []   # z_self at each P1 step (live anchor)
    p1_gts: List[int] = []                # ground-truth region at each P1 step
    p1_ep_ids: List[int] = []             # episode id per P1 step
    p1_b_on: List[int] = []               # REBIND-ON committed binding
    p1_b_frozen: List[int] = []           # REBIND-FROZEN committed binding
    region_visits_p0 = [0 for _ in range(K_REGIONS)]

    protos: Optional[List[torch.Tensor]] = None
    binder = None

    gen_align = torch.Generator()
    gen_align.manual_seed(seed * 104729 + 7)

    for ep in range(total_train_eps):
        is_p1 = ep >= p0_episodes
        _, obs = env.reset()
        agent.reset()
        zp = ap = None
        frozen_binding: Optional[int] = None  # set at episode entry in P1

        for _step in range(steps_per_episode):
            region_now = _region_of(env)  # region the CURRENT obs corresponds to

            action, zself_now, zworld_now = _step_decision(
                agent, obs, zp, ap, is_p1
            )
            if not torch.isfinite(action).all():
                error_note = f"non-finite action seed={seed} ep={ep} step={_step}"
                break

            if not is_p1:
                # Accumulate prototypes + region visits from OBSERVED z_world.
                proto_sum[region_now] = proto_sum[region_now] + zworld_now.reshape(1, -1)
                proto_cnt[region_now] += 1
                region_visits_p0[region_now] += 1
                global_zworld_sum = global_zworld_sum + zworld_now.reshape(1, -1)
                global_zworld_cnt += 1
            else:
                if protos is None:
                    # Freeze prototypes at P1 entry (fallback = global mean for any
                    # region never visited in P0 -- readiness gate flags that).
                    gmean = (
                        (global_zworld_sum / global_zworld_cnt)
                        if global_zworld_cnt > 0 else torch.zeros(1, world_dim)
                    )
                    protos = [
                        (proto_sum[k] / proto_cnt[k]) if proto_cnt[k] > 0 else gmean.clone()
                        for k in range(K_REGIONS)
                    ]
                    binder = agent.cross_stream_binder
                if binder is None:
                    error_note = f"binder absent seed={seed} (substrate OFF/fixed)"
                    break
                anchor = zself_now.reshape(1, -1)
                b_on, _ = _argmax_binding(binder, anchor, protos)
                if frozen_binding is None:
                    frozen_binding, _ = _argmax_binding(binder, anchor, protos)
                p1_anchors.append(anchor.clone())
                p1_gts.append(int(region_now))
                p1_ep_ids.append(int(ep))
                p1_b_on.append(int(b_on))
                p1_b_frozen.append(int(frozen_binding))

            # Advance env + online z_goal update (keeps locomotion natural).
            _, _hs, done, info, obs = env.step(action)
            if agent.goal_state is not None:
                be = float(info.get("benefit_exposure", 0.0))
                en = float(obs["body_state"].float().reshape(-1)[3].item())
                agent.update_z_goal(benefit_exposure=be, drive_level=max(0.0, 1.0 - en))

            zp, ap = zself_now, action.detach()
            if done:
                break

        if error_note is not None:
            break
        if (ep + 1) % 10 == 0 or ep == total_train_eps - 1:
            print(
                f"  [train] seed={seed} ep {ep + 1}/{total_train_eps} "
                f"phase={'P1' if is_p1 else 'P0'} "
                f"p1_steps={len(p1_gts)} "
                f"regions_covered={sum(1 for c in proto_cnt if c > 0)}/{K_REGIONS}",
                flush=True,
            )

    # Binder convergence telemetry (readiness).
    binder_b = agent.cross_stream_binder
    binder_learned = bool(binder_b is not None and getattr(binder_b, "learned", False))
    binder_loss_ema = getattr(binder_b, "loss_ema", None) if binder_b else None
    binder_chance_floor = getattr(binder_b, "chance_floor", None) if binder_b else None
    binder_converged = bool(getattr(binder_b, "binder_converged", False)) if binder_b else False

    # --- Compute DV1 / DV2 over the recorded P1 stream --------------------
    n_p1 = len(p1_gts)
    n_overtakes = sum(
        1 for t in range(1, n_p1)
        if p1_ep_ids[t] == p1_ep_ids[t - 1] and p1_gts[t] != p1_gts[t - 1]
    )

    a_real = (sum(1 for t in range(n_p1) if p1_b_on[t] == p1_gts[t]) / n_p1) if n_p1 else 0.0
    a_shuf = _label_shuffle_alignment(p1_b_on, p1_gts, gen_align) if n_p1 else 0.0
    a_anchor_shuf = (
        _anchor_shuffle_alignment(binder_b, p1_anchors, protos, p1_gts, gen_align)
        if (n_p1 and protos is not None and binder_b is not None) else 0.0
    )

    # Post-overtake alignment (ON), within REACQ_WINDOW after each overtake.
    post_hits = 0
    post_tot = 0
    for t in range(1, n_p1):
        if p1_ep_ids[t] == p1_ep_ids[t - 1] and p1_gts[t] != p1_gts[t - 1]:
            d = 0
            while d < REACQ_WINDOW and (t + d) < n_p1 and p1_ep_ids[t + d] == p1_ep_ids[t]:
                post_tot += 1
                if p1_b_on[t + d] == p1_gts[t + d]:
                    post_hits += 1
                d += 1
    post_overtake_align_on = (post_hits / post_tot) if post_tot else 0.0

    lats_on = _reacq_latencies(p1_b_on, p1_gts, p1_ep_ids)
    lats_frozen = _reacq_latencies(p1_b_frozen, p1_gts, p1_ep_ids)
    mean_lat_on = (sum(lats_on) / len(lats_on)) if lats_on else float(REACQ_WINDOW)
    mean_lat_frozen = (sum(lats_frozen) / len(lats_frozen)) if lats_frozen else float(REACQ_WINDOW)

    stale_rate_on = (
        (sum(1 for t in range(n_p1) if p1_b_on[t] != p1_gts[t]) / n_p1) if n_p1 else 1.0
    )
    stale_rate_frozen = (
        (sum(1 for t in range(n_p1) if p1_b_frozen[t] != p1_gts[t]) / n_p1) if n_p1 else 1.0
    )

    dv1 = bool(a_real >= a_shuf + ALIGN_MARGIN)
    dv2 = bool((mean_lat_frozen - mean_lat_on) >= LATENCY_MARGIN)

    min_region_visits = min(region_visits_p0) if region_visits_p0 else 0
    regions_covered = sum(1 for c in proto_cnt if c > 0)
    seed_ready = bool(
        binder_learned and binder_converged
        and regions_covered == K_REGIONS
        and min_region_visits >= MIN_REGION_VISITS_P0
        and n_overtakes >= MIN_OVERTAKES_P1
    )

    return {
        "seed": int(seed),
        "error_note": error_note,
        "n_p1_steps": int(n_p1),
        "n_overtakes": int(n_overtakes),
        "regions_covered": int(regions_covered),
        "min_region_visits_p0": int(min_region_visits),
        "region_visits_p0": [int(v) for v in region_visits_p0],
        # DV1
        "alignment_real": float(a_real),
        "alignment_shuffle": float(a_shuf),
        "alignment_margin_obs": float(a_real - a_shuf),
        "alignment_anchor_shuffle": float(a_anchor_shuf),
        "post_overtake_alignment_on": float(post_overtake_align_on),
        "DV1_tracks_truth_above_shuffle": dv1,
        # DV2
        "mean_reacq_latency_on": float(mean_lat_on),
        "mean_reacq_latency_frozen": float(mean_lat_frozen),
        "latency_gap_obs": float(mean_lat_frozen - mean_lat_on),
        "n_overtake_events_scored": int(len(lats_on)),
        "stale_rate_on": float(stale_rate_on),
        "stale_rate_frozen": float(stale_rate_frozen),
        "stale_rate_gap_obs": float(stale_rate_frozen - stale_rate_on),
        "DV2_behavioural_consequence": dv2,
        # readiness
        "binder_learned": binder_learned,
        "binder_loss_ema": binder_loss_ema,
        "binder_chance_floor": binder_chance_floor,
        "binder_converged": binder_converged,
        "seed_ready": seed_ready,
    }


# ---------------------------------------------------------------------------
# Interpret / run / manifest
# ---------------------------------------------------------------------------


def _interpret(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in rows if r["error_note"] is None]

    all_ready = bool(ok) and all(r["seed_ready"] for r in ok)
    n_dv1 = sum(1 for r in ok if r["DV1_tracks_truth_above_shuffle"])
    n_dv2 = sum(1 for r in ok if r["DV2_behavioural_consequence"])
    n_both = sum(
        1 for r in ok
        if r["DV1_tracks_truth_above_shuffle"] and r["DV2_behavioural_consequence"]
    )

    dv1_pass = bool(n_dv1 >= MIN_SEEDS_FOR_PASS)
    dv2_pass = bool(n_dv2 >= MIN_SEEDS_FOR_PASS)
    both_pass = bool(n_both >= MIN_SEEDS_FOR_PASS)

    if not all_ready:
        label = "substrate_not_ready_requeue"
    elif dv1_pass and dv2_pass and both_pass:
        label = "functional_rebinding_supported"
    elif dv1_pass and not dv2_pass:
        label = "rebinding_inert_off_equals_on"
    elif not dv1_pass:
        label = "rebinding_not_tracking_truth"
    else:
        # DV1 by-seed-count passes but not jointly on >= MIN_SEEDS (mixed) -- treat
        # as inert (behavioural consequence not established jointly).
        label = "rebinding_inert_off_equals_on"

    # Readiness preconditions (hygiene; the indexer reads these for adjudication).
    min_loss_gap = None
    worst_ema = None
    worst_chance = None
    _emas = [r.get("binder_loss_ema") for r in ok if r.get("binder_loss_ema") is not None]
    _ch = [r.get("binder_chance_floor") for r in ok if r.get("binder_chance_floor") is not None]
    if _emas:
        worst_ema = float(max(_emas))
    if _ch:
        worst_chance = float(min(_ch))
    conv_threshold = (
        float(CROSS_STREAM_BINDING_CONV_FRAC) * worst_chance
        if worst_chance is not None else None
    )
    min_overtakes = min((r["n_overtakes"] for r in ok), default=0)
    min_region_visits = min((r["min_region_visits_p0"] for r in ok), default=0)

    preconditions = [
        {
            "name": "learned_binder_converged",
            "kind": "readiness",
            "description": (
                "Every completed seed's binder must converge (loss_ema < "
                "conv_frac*log(batch)); an unconverged binder is the 725 "
                "untrained-substrate artifact, not a rebinding verdict. Routed "
                "statistic: WORST (max) loss_ema vs the convergence threshold "
                "(upper-bound: met when measured <= threshold)."
            ),
            "measured": (worst_ema if worst_ema is not None else 1e9),
            "threshold": (conv_threshold if conv_threshold is not None else 0.0),
            "direction": "upper",
            "control": "substrate binder_converged on the trained-then-frozen P0 binder",
            "met": bool(ok and all(r["binder_converged"] for r in ok)),
        },
        {
            "name": "region_coverage_adequate",
            "kind": "readiness",
            "description": (
                "min region visit count in P0 across completed seeds clears the "
                "prototype-stability floor (every region seen enough to mean-pool a "
                "prototype); below floor -> thin/ill-defined prototypes."
            ),
            "measured": int(min_region_visits),
            "threshold": int(MIN_REGION_VISITS_P0),
            "control": "per-region P0 visit counts",
            "met": bool(ok and min_region_visits >= MIN_REGION_VISITS_P0),
        },
        {
            "name": "overtake_events_adequate",
            "kind": "readiness",
            "description": (
                "min overtake (ground-truth region change) count in P1 across "
                "completed seeds clears the floor so the latency DV is not a mean "
                "over a near-empty set of events."
            ),
            "measured": int(min_overtakes),
            "threshold": int(MIN_OVERTAKES_P1),
            "control": "per-seed in-episode ground-truth region-change count",
            "met": bool(ok and min_overtakes >= MIN_OVERTAKES_P1),
        },
    ]

    criteria = [
        {"name": "learned_binder_converged", "load_bearing": True,
         "passed": bool(ok and all(r["binder_converged"] for r in ok))},
        {"name": "DV1_tracks_truth_above_shuffle", "load_bearing": True,
         "passed": dv1_pass},
        {"name": "DV2_behavioural_consequence", "load_bearing": True,
         "passed": dv2_pass},
    ]

    return {
        "label": label,
        "preconditions": preconditions,
        "criteria": criteria,
        "all_ready": all_ready,
        "n_completed": len(ok),
        "n_DV1": int(n_dv1),
        "n_DV2": int(n_dv2),
        "n_both": int(n_both),
        "min_seeds_for_pass": int(MIN_SEEDS_FOR_PASS),
        "dv1_pass": dv1_pass,
        "dv2_pass": dv2_pass,
        "both_pass": both_pass,
        "worst_binder_loss_ema": worst_ema,
        "binder_conv_threshold": conv_threshold,
        "min_overtakes": int(min_overtakes),
        "min_region_visits_p0": int(min_region_visits),
    }


def run_experiment(
    seeds: List[int],
    p0_episodes: int,
    p1_episodes: int,
    steps_per_episode: int,
    dry_run: bool,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    first = True
    for s in seeds:
        print(f"Seed {s} Condition rebinding_functional", flush=True)
        if first:
            print(
                f"  (P0={p0_episodes} ep, P1={p1_episodes} ep, "
                f"steps_per_episode={steps_per_episode}, K_regions={K_REGIONS}, "
                f"align_margin={ALIGN_MARGIN}, latency_margin={LATENCY_MARGIN}, "
                f"dry_run={dry_run})",
                flush=True,
            )
            first = False
        row = _run_seed(s, p0_episodes, p1_episodes, steps_per_episode)
        rows.append(row)
        verdict = "PASS" if row["error_note"] is None else "FAIL"
        print(f"verdict: {verdict}", flush=True)

    interp = _interpret(rows)
    ok = [r for r in rows if r["error_note"] is None]
    n_completed = len(ok)

    passed = bool(
        n_completed >= MIN_SEEDS_COMPLETED
        and interp["label"] == "functional_rebinding_supported"
    )

    # Non-degeneracy net (applies to evidence runs): load-bearing DV spreads.
    degen = check_degeneracy({
        "alignment_margin": [r["alignment_margin_obs"] for r in ok] or [0.0, 0.0],
        "latency_gap": [r["latency_gap_obs"] for r in ok] or [0.0, 0.0],
    })

    return {
        "outcome": "PASS" if passed else "FAIL",
        "seeds": list(seeds),
        "n_completed": int(n_completed),
        "n_total_runs": int(len(seeds)),
        "min_seeds_completed": int(MIN_SEEDS_COMPLETED),
        "p0_episodes": int(p0_episodes),
        "p1_episodes": int(p1_episodes),
        "steps_per_episode": int(steps_per_episode),
        "k_regions": int(K_REGIONS),
        "g_partition": int(G_PARTITION),
        "acceptance_thresholds": {
            "align_margin": float(ALIGN_MARGIN),
            "latency_margin": float(LATENCY_MARGIN),
            "reacq_window": int(REACQ_WINDOW),
            "n_shuffle_perms": int(N_SHUFFLE_PERMS),
            "min_seeds_for_pass": int(MIN_SEEDS_FOR_PASS),
            "min_region_visits_p0": int(MIN_REGION_VISITS_P0),
            "min_overtakes_p1": int(MIN_OVERTAKES_P1),
        },
        "binder_config": {
            "cross_stream_binding_enabled": bool(CROSS_STREAM_BINDING_ENABLED),
            "cross_stream_binding_learned": bool(CROSS_STREAM_BINDING_LEARNED),
            "cross_stream_binding_strength": float(CROSS_STREAM_BINDING_STRENGTH),
            "cross_stream_binding_temperature": float(CROSS_STREAM_BINDING_TEMPERATURE),
            "cross_stream_binding_conv_frac": float(CROSS_STREAM_BINDING_CONV_FRAC),
        },
        "per_seed_results": rows,
        "interpretation": interp,
        "non_degenerate": degen["non_degenerate"],
        "degeneracy_reason": degen["degeneracy_reason"],
        "degenerate_metrics": degen["degenerate_metrics"],
    }


def _evidence_direction(label: str) -> Tuple[str, str]:
    """Map the interpretation label to a (direction, note) for MECH-456."""
    if label == "functional_rebinding_supported":
        return "supports", (
            "DV1 (rebinding-alignment-with-ground-truth above the shuffle control) "
            "AND DV2 (rebinding-ON re-acquires the correct binding faster than a "
            "rebinding-frozen arm on a graded, non-saturating latency metric) both "
            "clear on >= MIN_SEEDS_FOR_PASS seeds -> functional rebinding demonstrated; "
            "SUPPORTS MECH-456 promotion (candidate -> shown; clear v3_pending)."
        )
    if label == "rebinding_inert_off_equals_on":
        return "weakens", (
            "The binder tracks ground truth (DV1) but re-binding confers no graded "
            "behavioural advantage over a frozen binding (DV2 fails: ON==FROZEN on "
            "re-acquisition latency) -- the MECH-269(b)/V3-EXQ-478 inert-reset "
            "signature at the entity locus. Rebinding is exercisable but not "
            "functional; does NOT support promotion."
        )
    if label == "rebinding_not_tracking_truth":
        return "weakens", (
            "Rebinding alignment is not above the shuffle control -- arbitrary "
            "anchor-sensitivity, not functional tracking of the true competitor "
            "(the n_rebind>0 of 725a was exercisability only). Does NOT support "
            "promotion."
        )
    # substrate_not_ready_requeue
    return "non_contributory", (
        "Readiness gate unmet (unconverged binder / thin region coverage / too few "
        "overtakes). NOT a rebinding verdict; re-queue at adequate P0 / more P1 "
        "steps. Not weighted in confidence/conflict scoring."
    )


def _build_manifest(result: Dict[str, Any], timestamp_utc: str, dry_run: bool) -> Dict[str, Any]:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    interp = result.get("interpretation", {})
    label = interp.get("label", "")
    direction, direction_note = _evidence_direction(label)

    review_caveats = [
        "GROUND TRUTH is the agent's G x G lattice region read directly off "
        "env.agent_x/agent_y -- binder-independent. DV1 (alignment_real vs "
        "alignment_shuffle) tests whether the LIVE-anchor argmax binding-affinity "
        "tracks the true current region above a region-label-shuffle chance "
        "baseline; alignment_anchor_shuffle (report) isolates that the CORRECT "
        "anchor matters, not any anchor.",
        "DV2 is a GRADED, NON-SATURATING re-acquisition latency (steps until the "
        "committed binding re-matches ground truth after an overtake, censored at "
        "reacq_window) -- NOT the 725a frac_state_div (which saturated 3/6). "
        "REBIND-ON (live anchor, re-binds every step) vs REBIND-FROZEN (binding set "
        "at episode entry, never re-evaluated) are two eval-time readouts over ONE "
        "shared trajectory per seed, so g(t) is identical across arms and the "
        "latency gap is a maximally-controlled contrast.",
        "UNSUPPORTED routing: rebinding_inert_off_equals_on (DV1 holds, DV2 fails = "
        "MECH-269(b) inert signature) and rebinding_not_tracking_truth (DV1 fails = "
        "arbitrary anchor-sensitivity) both WEAKEN MECH-456; substrate_not_ready_"
        "requeue is non_contributory (not a verdict).",
    ]
    if not interp.get("all_ready", False):
        review_caveats.insert(
            0,
            "WARNING readiness gate UNMET on >=1 completed seed "
            f"(min_overtakes={interp.get('min_overtakes')} vs "
            f"{MIN_OVERTAKES_P1}; min_region_visits_p0="
            f"{interp.get('min_region_visits_p0')} vs {MIN_REGION_VISITS_P0}; "
            "binder convergence per-seed in per_seed_results). Label self-routes "
            "substrate_not_ready_requeue (non_contributory); do NOT read as a "
            "rebinding verdict.",
        )

    return {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp_utc,
        "outcome": result["outcome"],
        "evidence_direction": direction,
        "evidence_direction_note": direction_note,
        "evidence_direction_per_claim": {"MECH-456": direction},
        "non_degenerate": result.get("non_degenerate", True),
        "degeneracy_reason": result.get("degeneracy_reason", ""),
        "bears_on_not_tagged": [
            "MECH-269", "MECH-270", "ARC-006", "MECH-045", "INV-002",
        ],
        "review_caveats": review_caveats,
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "alpha_world": 0.9,
            "use_support_preserving_cem": True,
            "use_per_stream_vs": True,
            "reef_enabled": True,
            "harness_level_measurement": True,
            "ree_core_modified": False,
            "cross_stream_binding_enabled": bool(CROSS_STREAM_BINDING_ENABLED),
            "cross_stream_binding_learned": bool(CROSS_STREAM_BINDING_LEARNED),
            "cross_stream_binding_strength": float(CROSS_STREAM_BINDING_STRENGTH),
        },
        "result": result,
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Short smoke-test run (1 seed, 3+3 ep, 40 steps).",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Override output dir (default: REE_assembly evidence/experiments).",
    )
    args = parser.parse_args()

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        p0, p1, steps = DRY_RUN_P0, DRY_RUN_P1, DRY_RUN_STEPS
    else:
        seeds = list(SEEDS)
        p0, p1, steps = P0_WARMUP_EPISODES, P1_MEASUREMENT_EPISODES, STEPS_PER_EPISODE

    result = run_experiment(
        seeds=seeds, p0_episodes=p0, p1_episodes=p1,
        steps_per_episode=steps, dry_run=bool(args.dry_run),
    )

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run))

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = (
            Path(__file__).resolve().parents[2]
            / "REE_assembly" / "evidence" / "experiments"
        )
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config") or manifest.get("config_summary"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )

    print(f"manifest: {out_path}", flush=True)
    if not args.dry_run:
        print(f"Result written to: {out_path}", flush=True)
    print(
        f"outcome: {result['outcome']} label={result['interpretation']['label']} "
        f"DV1={result['interpretation']['n_DV1']}/{result['interpretation']['n_completed']} "
        f"DV2={result['interpretation']['n_DV2']}/{result['interpretation']['n_completed']} "
        f"(need {MIN_SEEDS_FOR_PASS}) direction={manifest['evidence_direction']}",
        flush=True,
    )

    outcome_norm = result["outcome"].upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    return outcome_emit, str(out_path), bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _manifest_path, _dry = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path, dry_run=_dry)
    sys.exit(0)
