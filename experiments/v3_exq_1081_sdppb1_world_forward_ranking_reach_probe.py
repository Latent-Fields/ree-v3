#!/opt/local/bin/python3
"""
V3-EXQ-1081 -- SD-PP-B1 probe: does a post-sleep change in e2.world_forward move
E3's candidate RANKING, and through WHICH route?

SLEEP DRIVER: manual-cycle-loop (force_cycle() called once, between a pre-sleep
              waking window and the yoked probe; sleep_loop_episodes_K=1e6 so the
              automatic cadence never fires)
red-team (fable): CONTESTED -> 4 fixed (twin E3-schedule desync now a
          per-twin determinacy gate + undetermined label; C3 three-state and
          route tokens explicit; NOISE_MATCHED norm-matched control + near-tie
          and skill-delta recorded; C4 demoted to recorded, non-finite ticks
          excluded)

WHY THIS RUN
------------
Commissioned by /governance (GFLAG-0413 option A, user-accepted 2026-09-23,
rec-20260923-e1e14f35): "arm the MECH-314a curiosity_candidate_source=
'e2_world_forward' source in all arms and measure whether a post-sleep
world-forward head change moves E3 ranking" -- a RANKING-SHIFT DV, not a
behaviour DV. SD-PP-B1 (substrate_queue) is node_class complex (probe-gated)
until this reads out; B1 caps MECH-572/573/574 and INV-063 leg B at the
mechanistic level.

PREMISE CORRECTION (GFLAG-0437, raised by this session before any code)
-----------------------------------------------------------------------
B1's registered premise -- "no default-on behavioural consumer of
e2.world_forward; the head has no native route to E3" -- is FALSE against
ree-v3 4b32e9d. HippocampalModule proposes every E3 candidate through
E2FastPredictor.rollout_with_world (hippocampal/module.py:1113, 1342, 1384,
2284, 2895, 3031), and rollout_with_world calls self.world_forward(z_world,
action) at every rollout step (e2_fast.py ~:822). E3.score_trajectory then
computes F / M / Phi / benefit / goal over exactly those world_states
(e3_selector.py _get_world_states :1301 and the cost terms :1334-1444). A
scratch probe at defaults (curiosity OFF, source "proposer", alpha_world 0.9,
one seeded E3 tick, K=32) confirmed it: perturbing only e2.world_transition
changed the candidate world_states, changed the CEM-selected candidate action
set, and moved the E3 argmin 26 -> 19, while an unperturbed rebuild was
bit-identical. That was STRUCTURAL reach only (a large random perturbation, an
untrained agent). It does not say whether a SLEEP-SIZED change reaches ranking.

So the commissioned question is re-scoped, not replaced: 314a stays armed in
ALL twins exactly as commissioned, and the probe ATTRIBUTES whatever ranking
shift a post-sleep head change produces to its two routes --
  (R) the default proposer-rollout route (candidate world_states), and
  (C) the MECH-314a curiosity route (per-candidate novelty summaries).

DESIGN -- a yoked-twin counterfactual (no deep copy: REEAgent is not
deep-copyable, so twins are rebuilt from state_dicts, as V3-EXQ-1073 does)
-------------------------------------------------------------------------
Per seed:
  P0     world_forward reconstruction training on random-action transitions
         (the RECON-only 1073 recipe), at alpha_world 0.9 (SD-008 floor --
         NOT the from_dims default 0.3 that made 1073/1075/1079 provisional).
         done IS read: an episode boundary resets the env and drops the
         crossing transition (the 1073/1075/1079 battery collectors did not,
         and collected post-death -- chip-20260923-battery-postdeath-done-flag).
  WAKE   a short pre-sleep waking window on the agent's own selection, so the
         world experience buffer holds replayable transitions.
  S_pre  = agent.state_dict()   (snapshot)
  SLEEP  one force_cycle() with use_sleep_world_forward_consolidation=True.
  S_post = agent.state_dict()
  PROBE  N twins are rebuilt from those snapshots under an identical RNG reset
         and driven through ONE shared env in lock-step. The env is driven by
         twin A's action; every twin is told A's executed action and harm, so
         all twins see the same observation stream. Before EACH twin's tick the
         global RNG is reset to the same per-step seed, so twins that differ in
         nothing make bit-identical choices (the NULL twin asserts this from
         output rather than trusting it).

  twin                  weights                          314a summaries read
  --------------------  -------------------------------  -------------------
  A (reference)         S_pre                            (own head = pre)
  NULL                  S_pre                            (own head = pre)
  SLEEP_BOTH            S_pre + world head from S_post   (own head = post)
  SLEEP_ROLLOUT_ONLY    S_pre + world head from S_post   PRE head (swapped in)
  SLEEP_CURIOSITY_ONLY  S_pre                            POST head (swapped in)
  SLEEP_X4_BOTH         S_pre + head pre + 4*(post-pre)  (own head)
  FULL_SLEEP            S_post (every module post-sleep) (own head = post)
  REINIT_BOTH           S_pre + a freshly-initialised    (own head)
                        world head   [POSITIVE CONTROL, both routes]
  REINIT_CURIOSITY_ONLY S_pre                            the random head
                                     [POSITIVE CONTROL, curiosity route only]
  NOISE_MATCHED         S_pre + head pre + random dir    (own head)
                        with per-tensor norm = sleep's   [CONTEXT: magnitude]

  Heads are written into the LIVE parameters after load_state_dict, never by
  editing state_dict keys: the world head is registered twice (e2.* and
  hippocampal.e2.* alias one module), so an e2.*-only key edit is silently
  undone by the alias on load. The first smoke of this driver hit exactly that
  -- REINIT_BOTH read bit-identical to A -- and the build now asserts from
  output that each twin carries its intended head.

  "world head" = e2.world_transition + e2.world_action_encoder, the only
  parameters e2.world_forward reads (e2_fast.py world_forward).

DV (a ranking-shift DV, computed at every E3 tick where the twin ran a FRESH
select -- e3.last_scores is cleared before each select_action and read after,
so a latched tick records nothing):
  top1_diff_rate   fraction of ticks where the first-action class of the
                   argmin-score candidate (E3 is lower-is-better) differs
                   from A's. Compared by first-action CLASS, not candidate
                   index, because the rollout route changes the candidate SET.
  sel_diff_rate    same for the action select_action actually returned
                   (sampled; identical RNG, so NULL is exact by construction).
  tv_mean          mean total-variation distance between A's and the twin's
                   pre-commit softmax mass aggregated by first-action class.
  cand_identical_frac  fraction of ticks where the twin's candidate action set
                   is identical to A's (1.0 expected for CURIOSITY_ONLY: that
                   route cannot change the candidate set; <1.0 for rollout).
  rank_discordance_mean  on identical-candidate-set ticks only: fraction of
                   candidate PAIRS whose E3 score order flips (Kendall
                   discordance). The continuous ranking DV. It matters most for
                   the curiosity route: the E3 softmax is sharp, so top1/TV move
                   only when the argmin moves, while the 314a novelty deviation
                   is ~5% of the E3 score range (smoke) -- it can reorder
                   mid-ranked candidates without touching the top.
                   RECORDED, not gating (no pre-registered bar).

PRE-REGISTERED THRESHOLDS
  REACH_BAR = 0.05   excess top1_diff_rate over the NULL twin that counts as
                     "moves E3 ranking" (5% of E3 ticks re-ranked at the top).
  SEEDS_REQUIRED = 2 of 3.
  POS_CONTROL_FLOOR = 0.10  REINIT_BOTH must re-rank >= 10% of ticks, or the
                     instrument cannot tell "no reach" from "insensitive".
  MIN_TICKS = 20     E3 ticks per seed (sample floor).
  CURIOSITY_LIVE_FRAC = 0.5  fraction of A's ticks with a non-zero 314a
                     novelty deviation range; scoped ONLY to the C3
                     curiosity-route criterion (it certifies that channel and
                     nothing else).

PRECONDITIONS (per seed, numeric, recomputable)
  yoke_exact           max(NULL top1/sel/tv, ran-mismatch count) <= 0 (upper)
  sleep_moved_head     relative L2 change of the world head > 1e-6
  positive_control     REINIT_BOTH top1_diff_rate >= 0.10
  n_ticks              A-vs-NULL paired ticks >= 20
  curiosity_route_live (C3 only)
  curiosity_route_positive_control  REINIT_CURIOSITY_ONLY top1 >= 0.10
                       (C3 only; unmet = the 314a route has no top-1 authority
                       at curiosity_novelty_weight 0.05, so C3 cannot
                       discriminate and is reported as such)

CRITERIA
  Every twin criterion is evaluated only on seeds where that twin is
  DETERMINATE (>= MIN_TICKS paired finite ticks). A head change can move a
  twin's commit timing, and commitment calls clock.phase_reset(), which fires
  E3 on a different env step than A (heartbeat/clock.py:149-154) -- so a twin
  can lose pairs. It is then pass/fail/UNDETERMINED, never a silent "no reach".
  C1 (LOAD-BEARING) SLEEP_BOTH - NULL top1_diff_rate >= REACH_BAR on >= 2
     ready seeds on which SLEEP_BOTH is determinate.
  C2 route R: SLEEP_ROLLOUT_ONLY, same bar.   C3 route C: SLEEP_CURIOSITY_ONLY,
     same bar, over curiosity-live seeds whose REINIT_CURIOSITY_ONLY control
     passes.   C4 SLEEP_X4_BOTH (extrapolation, recorded only -- an MLP head is
     not linear, so x4 is not a dose).   C5 FULL_SLEEP context.   C6
     NOISE_MATCHED context: if C6 reaches as often as C1, the reach is generic
     sensitivity to a head change of that size, not something specific to the
     sleep direction.  C2-C6 are recorded, not load-bearing.
  combination_rule: outcome PASS iff >= 2 ready seeds AND C1 passes.
  Also recorded, per seed: skill_delta_post_minus_pre (sign of the sleep
  change's effect on held-out world_forward skill -- a C1 pass on a DEGRADING
  change bears on MECH-572 differently from one on an improving change) and
  a_near_tie_frac (A's ticks whose top-2 score gap is < 1% of the range;
  argmin flips concentrate there).

INTERPRETATION GRID (self-route is a hypothesis; /failure-autopsy adjudicates)
  < 2 ready seeds                  -> substrate_not_ready_requeue
  C1 undetermined (desync)         -> sleep_reach_undetermined_twin_desync
  C1 pass                          -> sleep_head_change_reaches_e3_ranking__
                                      rollout_<yes|no|undetermined>__
                                      curiosity_<yes|no|undetermined>
  C1 fail                          -> sleep_sized_head_change_below_reach

DV-SYMMETRY (Step 3.5): the DV is an argmin/first-action-class statistic,
invariant under a uniform additive shift of all candidate scores. The
manipulation (a head weight change) is not such a shift: it changes each
candidate's rollout world_states and each candidate's 314a summary
separately, so the per-candidate cost terms and novelty distances move
non-uniformly. REINIT_BOTH is the empirical check that the manipulation class
can move the DV at all.

DECLARED LIMITS
  - E3's evaluators (harm_eval_head, benefit, residue) are at their P0 state:
    P0 trains only e2. Reach is measured against those evaluators; a trained
    evaluator could be more or less sensitive to a world-state change. This
    measures reach of the head through the CURRENT default consumer, not
    through a trained organism.
  - The z_world ENCODER is untrained (P0 trains e2 only), as in 1073/1075/1079
    -- the SD-PP-B10 encoder leg is out of scope here.
  - Twins that re-rank drift apart in internal state (commitment, running
    variance) over the probe window; that drift IS downstream reach, but it
    means late ticks are not independent. top1_diff_rate_early (first 20
    paired ticks) is recorded alongside the full-window rate.
  - E3 composition at defaults: normalize_score_bias_to_e3_range,
    use_modulatory_selection_authority and use_finer_channel_gating are OFF
    (asserted at build).
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell, reset_all_rng
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiments.pack_writer import write_flat_manifest
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------
QUEUE_ID = "V3-EXQ-1081"
EXPERIMENT_TYPE = "v3_exq_1081_sdppb1_world_forward_ranking_reach_probe"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS: List[str] = []
VALIDATES_SUBSTRATE = "SD-PP-B1-world-forward-behavioural-consumer"
BEARS_ON = ["MECH-572", "MECH-573", "MECH-574", "INV-063"]
GOVERNANCE_REFS = ["GFLAG-0413", "GFLAG-0437", "rec-20260923-e1e14f35"]

# ---------------------------------------------------------------------------
# configuration (env / dims shared with the V3-EXQ-1073 lineage so a reader
# can place the head; alpha_world deliberately NOT inherited)
# ---------------------------------------------------------------------------
SEEDS = [42, 123, 456]
GRID_SIZE = 5
N_HAZARDS = 1
N_RESOURCES = 1
SELF_DIM = 16
WORLD_DIM = 16
ALPHA_WORLD = 0.9                 # SD-008 floor; from_dims default is 0.3
STEPS_PER_EPISODE = 90
P0_STEPS = 3600
WAKE_STEPS = 180
PROBE_STEPS = 900
E2_LR = 1e-3
BATCH_K = 8
BUF_MAX = 256
MIN_BUF_BEFORE_TRAIN = 16
MAX_GRAD_NORM = 1.0
CMC_STEPS = 8
CMC_LR = 1e-3
CMC_BATCH = 16
HELDOUT_N = 64
X_DOSE = 4.0

# pre-registered thresholds
REACH_BAR = 0.05
SEEDS_REQUIRED = 2
POS_CONTROL_FLOOR = 0.10
MIN_TICKS = 20
CURIOSITY_LIVE_FRAC = 0.5
HEAD_MOVED_FLOOR = 1e-6
EARLY_WINDOW = 20
NEAR_TIE_REL = 0.01               # top-2 gap / score range below this = near-tie tick

HEAD_MODULES = ("world_transition", "world_action_encoder")
TWINS = ["A", "NULL", "SLEEP_BOTH", "SLEEP_ROLLOUT_ONLY", "SLEEP_CURIOSITY_ONLY",
         "SLEEP_X4_BOTH", "FULL_SLEEP", "REINIT_BOTH", "REINIT_CURIOSITY_ONLY",
         "NOISE_MATCHED"]
COMPARED = [t for t in TWINS if t != "A"]

_ZG = ZGoalStreamAccumulator()


def _set_schedule(dry_run: bool) -> None:
    global SEEDS, P0_STEPS, WAKE_STEPS, PROBE_STEPS
    if dry_run:
        SEEDS = [42]
        P0_STEPS = 180
        WAKE_STEPS = 90
        PROBE_STEPS = 270


def _episodes_per_run() -> int:
    return (P0_STEPS + WAKE_STEPS + PROBE_STEPS) // STEPS_PER_EPISODE


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------
def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, size=GRID_SIZE, num_hazards=N_HAZARDS,
                             num_resources=N_RESOURCES, use_proxy_fields=True)


def _config_slice() -> Dict[str, Any]:
    return {
        "env": {"size": GRID_SIZE, "num_hazards": N_HAZARDS,
                "num_resources": N_RESOURCES, "use_proxy_fields": True},
        "dims": {"self": SELF_DIM, "world": WORLD_DIM}, "alpha_world": ALPHA_WORLD,
        "schedule": {"p0": P0_STEPS, "wake": WAKE_STEPS, "probe": PROBE_STEPS,
                     "steps_per_episode": STEPS_PER_EPISODE},
        "p0_opt": {"lr": E2_LR, "batch": BATCH_K, "buf": BUF_MAX,
                   "min_buf": MIN_BUF_BEFORE_TRAIN, "max_grad_norm": MAX_GRAD_NORM},
        "heldout_n": HELDOUT_N,
        "thresholds": {"reach_bar": REACH_BAR, "seeds_required": SEEDS_REQUIRED,
                       "pos_control_floor": POS_CONTROL_FLOOR, "min_ticks": MIN_TICKS,
                       "curiosity_live_frac": CURIOSITY_LIVE_FRAC,
                       "head_moved_floor": HEAD_MOVED_FLOOR, "early_window": EARLY_WINDOW,
                       "near_tie_rel": NEAR_TIE_REL},
        "sleep": {"cmc_steps": CMC_STEPS, "cmc_lr": CMC_LR, "cmc_batch": CMC_BATCH,
                  "world_forward_consolidation": True},
        "curiosity": {"structured": True, "novelty": True, "uncertainty": False,
                      "learning_progress": False, "novelty_source": "visitation",
                      "candidate_source": "e2_world_forward"},
        "twins": TWINS, "x_dose": X_DOSE,
    }


def _make_config(env: CausalGridWorldV2) -> REEConfig:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=ALPHA_WORLD,
        # MECH-314a armed in ALL twins (the commission). 314b/314c OFF so the
        # curiosity bias is the per-candidate novelty alone.
        use_structured_curiosity=True,
        use_curiosity_novelty=True,
        use_curiosity_uncertainty=False,
        use_curiosity_learning_progress=False,
        curiosity_novelty_source="visitation",
        curiosity_candidate_source="e2_world_forward",
        # sleep: one deliberate force_cycle, world head consolidated.
        sws_enabled=True,
        rem_enabled=True,
        use_sleep_loop=True,
        sleep_loop_episodes_K=1_000_000,
        use_cross_module_consolidation=True,
        cross_module_consolidation_schedule="interleaved",
        cross_module_consolidation_steps=CMC_STEPS,
        cross_module_consolidation_lr=CMC_LR,
        cross_module_consolidation_batch=CMC_BATCH,
        use_sleep_world_forward_consolidation=True,
        use_within_life_sleep_trigger=False,
    )
    return cfg


def _assert_default_composition(agent: REEAgent) -> Dict[str, Any]:
    e3c = agent.e3.config
    flags = {
        "normalize_score_bias_to_e3_range": bool(getattr(e3c, "normalize_score_bias_to_e3_range", False)),
        "use_modulatory_selection_authority": bool(getattr(e3c, "use_modulatory_selection_authority", False)),
        "use_finer_channel_gating": bool(getattr(e3c, "use_finer_channel_gating", False)),
        "curiosity_candidate_source": str(getattr(agent.config, "curiosity_candidate_source", "")),
        "curiosity_present": agent.curiosity is not None,
        "visitation_buffer_present": agent._zworld_visitation_buffer is not None,
        "alpha_world": float(getattr(agent.config.latent, "alpha_world", float("nan"))),
    }
    assert not flags["normalize_score_bias_to_e3_range"]
    assert not flags["use_modulatory_selection_authority"]
    assert not flags["use_finer_channel_gating"]
    assert flags["curiosity_candidate_source"] == "e2_world_forward", flags
    assert flags["curiosity_present"] and flags["visitation_buffer_present"], flags
    assert abs(flags["alpha_world"] - ALPHA_WORLD) < 1e-9, flags
    return flags


def _to_b(x: Any, device: Any) -> torch.Tensor:
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)
    return (t.unsqueeze(0) if t.ndim == 1 else t).to(device)


def _sense(agent: REEAgent, obs_dict: Dict[str, Any]):
    harm = obs_dict.get("harm_obs", None)
    return agent.sense(_to_b(obs_dict["body_state"], agent.device),
                       _to_b(obs_dict["world_state"], agent.device),
                       obs_harm=_to_b(harm, agent.device) if harm is not None else None)


def _clone_state(agent: REEAgent) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in agent.state_dict().items()}


def _head_live(agent: REEAgent) -> Dict[str, torch.Tensor]:
    """name -> live parameter tensor of the world head, keyed like state_dict."""
    out: Dict[str, torch.Tensor] = {}
    for m in HEAD_MODULES:
        mod = getattr(agent.e2, m)
        for n, p in mod.named_parameters():
            out[f"e2.{m}.{n}"] = p
    return out


@contextmanager
def _head_swapped(agent: REEAgent, head: Optional[Dict[str, torch.Tensor]]):
    """Temporarily copy `head` into the agent's world-head parameters (no_grad),
    restoring the originals on exit. head=None is a no-op."""
    if head is None:
        yield
        return
    live = _head_live(agent)
    saved = {k: p.detach().clone() for k, p in live.items()}
    with torch.no_grad():
        for k, p in live.items():
            p.copy_(head[k])
    try:
        yield
    finally:
        with torch.no_grad():
            for k, p in live.items():
                p.copy_(saved[k])


def _install_curiosity_head(agent: REEAgent,
                            head: Optional[Dict[str, torch.Tensor]]) -> None:
    """Route split: make ONLY the MECH-314a summary read (agent.py
    _curiosity_candidate_summaries, the sole 314a world_forward call) see
    `head`, while rollouts keep the agent's own head."""
    if head is None:
        return
    orig = agent._curiosity_candidate_summaries

    def _patched(candidates):
        with _head_swapped(agent, head):
            return orig(candidates)

    agent._curiosity_candidate_summaries = _patched  # type: ignore[assignment]


def _apply_head(agent: REEAgent, head: Dict[str, torch.Tensor]) -> None:
    """Write `head` into the LIVE world-head parameters. Must run AFTER
    load_state_dict: the world head is registered twice in the state_dict
    (e2.* and hippocampal.e2.* alias the same module), so a state_dict edit
    of only the e2.* keys is silently overwritten by the hippocampal.e2.* alias
    on load (caught by this driver's own REINIT positive control at smoke)."""
    live = _head_live(agent)
    assert set(live) == set(head), (sorted(live), sorted(head))
    with torch.no_grad():
        for k, p in live.items():
            p.copy_(head[k])


def _build_twin(cfg: REEConfig, state: Dict[str, torch.Tensor],
                build_seed: int,
                head: Optional[Dict[str, torch.Tensor]] = None,
                curiosity_head: Optional[Dict[str, torch.Tensor]] = None) -> REEAgent:
    reset_all_rng(build_seed)
    agent = REEAgent(cfg)
    agent.load_state_dict(state, strict=True)
    if head is not None:
        _apply_head(agent, head)
    agent.eval()
    _install_curiosity_head(agent, curiosity_head)
    return agent


def _head_snapshot(agent: REEAgent) -> Dict[str, torch.Tensor]:
    return {k: p.detach().clone() for k, p in _head_live(agent).items()}


# ---------------------------------------------------------------------------
# P0 (world_forward reconstruction, 1073 RECON-only recipe) -- done IS read
# ---------------------------------------------------------------------------
def _run_p0(agent: REEAgent, seed: int, ep_offset: int) -> Dict[str, Any]:
    env = _make_env(seed)
    _, obs = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()
    opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_LR)
    buf: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = deque(maxlen=BUF_MAX)
    rng = random.Random(seed)
    prev: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    losses: List[float] = []
    n_done = 0
    for step in range(P0_STEPS):
        if step % STEPS_PER_EPISODE == 0:
            print(f"  [train] seed={seed} ep {ep_offset + step // STEPS_PER_EPISODE + 1}"
                  f"/{_episodes_per_run()} phase=P0", flush=True)
        with torch.no_grad():
            z = _sense(agent, obs).z_world.detach().reshape(-1).clone()
        if prev is not None and bool(torch.isfinite(z).all()):
            buf.append((prev[0], prev[1], z))
        idx = rng.randrange(env.action_dim)
        a = torch.zeros(env.action_dim, dtype=torch.float32)
        a[idx] = 1.0
        prev = (z, a)
        _, _, done, _, obs = env.step(a.unsqueeze(0).to(agent.device))
        if len(buf) >= MIN_BUF_BEFORE_TRAIN:
            pool = list(buf)
            batch = pool if len(pool) <= BATCH_K else rng.sample(pool, BATCH_K)
            z0 = torch.stack([t[0] for t in batch]).to(agent.device)
            a0 = torch.stack([t[1] for t in batch]).to(agent.device)
            z1 = torch.stack([t[2] for t in batch]).to(agent.device)
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(agent.e2.world_forward(z0, a0), z1)
            lv = float(loss.detach().item())
            if math.isfinite(lv):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.e2.parameters(), MAX_GRAD_NORM)
                opt.step()
                losses.append(lv)
        if done:
            n_done += 1
            _, obs = env.reset()
            agent.e1.reset_hidden_state()
            prev = None   # never train across an episode boundary
    q = max(1, len(losses) // 10)
    return {"p0_loss_first_decile": _mean(losses[:q]), "p0_loss_last_decile": _mean(losses[-q:]),
            "p0_n_train_steps": len(losses), "p0_n_done": n_done}


def _heldout(agent: REEAgent, seed: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Held-out transitions from a DISTINCT env instance, random actions, done
    read (a boundary drops the crossing pair). Sensing only; no training."""
    env = _make_env(seed + 9973)
    _, obs = env.reset()
    rng = random.Random(seed + 9973)
    rows: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    prev = None
    guard = 0
    with torch.no_grad():
        while len(rows) < HELDOUT_N and guard < HELDOUT_N * 8:
            guard += 1
            z = _sense(agent, obs).z_world.detach().reshape(-1).clone()
            if prev is not None:
                rows.append((prev[0], prev[1], z))
            a = torch.zeros(env.action_dim)
            a[rng.randrange(env.action_dim)] = 1.0
            prev = (z, a)
            _, _, done, _, obs = env.step(a.unsqueeze(0).to(agent.device))
            if done:
                _, obs = env.reset()
                prev = None
    return (torch.stack([r[0] for r in rows]), torch.stack([r[1] for r in rows]),
            torch.stack([r[2] for r in rows]))


def _skill(agent: REEAgent, batt, head: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
    z0, a0, z1 = batt
    with torch.no_grad(), _head_swapped(agent, head):
        m = float(F.mse_loss(agent.e2.world_forward(z0, a0), z1).item())
    i = float(F.mse_loss(z0, z1).item())
    return {"mse_model": m, "mse_identity": i,
            "persistence_relative_skill": (i - m) / (i + m) if (i + m) > 0 else float("nan")}


# ---------------------------------------------------------------------------
# waking tick (shared by WAKE and every twin in PROBE)
# ---------------------------------------------------------------------------
def _tick(agent: REEAgent, obs: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, Any]]]:
    latent = _sense(agent, obs)
    ticks = agent.clock.advance()
    wdim = latent.z_world.shape[-1]
    e1_prior = (agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device))
    cands = agent.generate_trajectories(latent, e1_prior, ticks)
    agent.e3.last_scores = None          # latch-clear idiom: record only a FRESH select
    agent.e3.last_precommit_probs = None
    action = agent.select_action(cands, ticks)
    scores = agent.e3.last_scores
    if scores is None or not ticks.get("e3_tick", False):
        return action, None
    scores = scores.detach().reshape(-1)
    fa = [int(c.actions[0, 0, :].argmax().item()) for c in cands]
    if len(fa) != scores.numel():
        return action, None
    adim = int(cands[0].actions.shape[-1])
    finite = bool(torch.isfinite(scores).all().item())
    k = int(torch.argmin(scores).item())
    srt = torch.sort(scores).values
    rng_ = float((scores.max() - scores.min()).item())
    top2_rel = (float((srt[1] - srt[0]).item()) / rng_) if (scores.numel() > 1 and rng_ > 0) else float("nan")
    pdist = torch.zeros(adim)
    probs = agent.e3.last_precommit_probs
    if probs is not None and probs.numel() == len(fa):
        for i, p in enumerate(probs.detach().reshape(-1).tolist()):
            pdist[fa[i]] += float(p)
    cur = agent.curiosity
    rec = {
        "top1": fa[k],
        "sel": int(action.reshape(-1).argmax().item()) if action is not None else -1,
        "pdist": pdist,
        "cand_actions": torch.stack([c.actions.detach()[0] for c in cands]),
        "scores": scores.clone(),
        "finite": finite,
        "top2_gap_rel": top2_rel,
        "score_range": float((scores.max() - scores.min()).item()),
        "novelty_dev_range": float(getattr(cur, "_last_novelty_dev_range", 0.0)) if cur else 0.0,
        "novelty_source_used": str(getattr(cur, "_last_novelty_source_used", "none")) if cur else "none",
    }
    return action, rec


def _run_wake(agent: REEAgent, seed: int, ep_offset: int) -> Dict[str, Any]:
    env = _make_env(seed + 101)
    _, obs = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()
    gen = torch.Generator().manual_seed(seed + 101)
    n_e3 = 0
    for step in range(WAKE_STEPS):
        if step % STEPS_PER_EPISODE == 0:
            print(f"  [train] seed={seed} ep {ep_offset + step // STEPS_PER_EPISODE + 1}"
                  f"/{_episodes_per_run()} phase=WAKE", flush=True)
        action, rec = _tick(agent, obs)
        n_e3 += int(rec is not None)
        if action is None or not bool(torch.isfinite(action).all()):
            action = torch.zeros(1, env.action_dim, device=agent.device)
            action[0, int(torch.randint(0, env.action_dim, (1,), generator=gen).item())] = 1.0
        agent.record_executed_action(action)
        _, harm, done, _, obs = env.step(action)
        with torch.no_grad():
            agent.update_residue(harm_signal=float(harm), world_delta=None,
                                 hypothesis_tag=False, owned=True)
        if done:
            _, obs = env.reset()
            agent.e1.reset_hidden_state()
            agent.notify_env_reset()
    return {"wake_n_e3_ticks": n_e3}


# ---------------------------------------------------------------------------
# the yoked probe
# ---------------------------------------------------------------------------
def _run_probe(twins: Dict[str, REEAgent], seed: int, ep_offset: int) -> Dict[str, List[Optional[Dict[str, Any]]]]:
    env = _make_env(seed + 777)
    _, obs = env.reset()
    for ag in twins.values():
        ag.reset()
        ag.e1.reset_hidden_state()
    gen = torch.Generator().manual_seed(seed + 777)
    recs: Dict[str, List[Optional[Dict[str, Any]]]] = {t: [] for t in twins}
    for step in range(PROBE_STEPS):
        if step % STEPS_PER_EPISODE == 0:
            print(f"  [train] seed={seed} ep {ep_offset + step // STEPS_PER_EPISODE + 1}"
                  f"/{_episodes_per_run()} phase=PROBE", flush=True)
        tick_seed = seed * 1_000_003 + step
        a_exec = None
        for name, ag in twins.items():
            reset_all_rng(tick_seed)
            act, rec = _tick(ag, obs)
            recs[name].append(rec)
            if name == "A":
                a_exec = act
        if a_exec is None or not bool(torch.isfinite(a_exec).all()):
            a_exec = torch.zeros(1, env.action_dim)
            a_exec[0, int(torch.randint(0, env.action_dim, (1,), generator=gen).item())] = 1.0
        a_exec = a_exec.detach()
        for ag in twins.values():
            ag.record_executed_action(a_exec.to(ag.device))   # yoke
        _, harm, done, _, obs = env.step(a_exec)
        for ag in twins.values():
            with torch.no_grad():
                ag.update_residue(harm_signal=float(harm), world_delta=None,
                                  hypothesis_tag=False, owned=True)
        if done:
            _, obs = env.reset()
            for ag in twins.values():
                ag.e1.reset_hidden_state()
                ag.notify_env_reset()
    return recs


def _compare(ref: List[Optional[Dict[str, Any]]],
             other: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    raw = [(a, b) for a, b in zip(ref, other) if a is not None and b is not None]
    both = [(a, b) for a, b in raw if a["finite"] and b["finite"]]
    n_nonfinite = len(raw) - len(both)
    mismatch = sum(1 for a, b in zip(ref, other) if (a is None) != (b is None))
    n = len(both)
    desync = mismatch / (n + mismatch) if (n + mismatch) > 0 else 0.0
    if n == 0:
        return {"n_paired_ticks": 0, "ran_mismatch": mismatch, "desync_frac": desync,
                "n_nonfinite_ticks": n_nonfinite, "determinate": False}
    top = [int(a["top1"] != b["top1"]) for a, b in both]
    sel = [int(a["sel"] != b["sel"]) for a, b in both]
    tv = [0.5 * float((a["pdist"] - b["pdist"]).abs().sum().item()) for a, b in both]
    ident = [int(a["cand_actions"].shape == b["cand_actions"].shape
                 and torch.equal(a["cand_actions"], b["cand_actions"])) for a, b in both]
    first = next((i for i, d in enumerate(top) if d), None)
    # Continuous ranking DV on ticks whose candidate SET is identical (the
    # curiosity route cannot change the set, so for it this is every tick):
    # fraction of candidate PAIRS whose score order flips (Kendall discordance).
    disc: List[float] = []
    for (a, b), same in zip(both, ident):
        if not same:
            continue
        sa, sb = a["scores"], b["scores"]
        da = torch.sign(sa.unsqueeze(0) - sa.unsqueeze(1))
        db = torch.sign(sb.unsqueeze(0) - sb.unsqueeze(1))
        iu = torch.triu_indices(sa.numel(), sa.numel(), offset=1)
        pa, pb = da[iu[0], iu[1]], db[iu[0], iu[1]]
        mask = (pa != 0) | (pb != 0)
        disc.append(float((pa[mask] != pb[mask]).float().mean().item()) if bool(mask.any()) else 0.0)
    return {
        "n_paired_ticks": n, "ran_mismatch": mismatch, "desync_frac": desync,
        "n_nonfinite_ticks": n_nonfinite, "determinate": n >= MIN_TICKS,
        "top1_diff_rate": sum(top) / n,
        "top1_diff_rate_early": sum(top[:EARLY_WINDOW]) / max(1, min(n, EARLY_WINDOW)),
        "sel_diff_rate": sum(sel) / n,
        "tv_mean": sum(tv) / n, "tv_max": max(tv),
        "cand_identical_frac": sum(ident) / n,
        "first_divergence_tick": first,
        "rank_discordance_mean": (sum(disc) / len(disc)) if disc else None,
        "rank_discordance_n_ticks": len(disc),
    }


def _rel_l2(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor], keys: List[str]) -> float:
    num = sum(float(((b[k].float() - a[k].float()) ** 2).sum().item()) for k in keys)
    den = sum(float((a[k].float() ** 2).sum().item()) for k in keys)
    return math.sqrt(num) / max(math.sqrt(den), 1e-12)


def _run_cell(seed: int) -> Dict[str, Any]:
    print(f"Seed {seed} Condition YOKED_TWINS", flush=True)
    with arm_cell(seed, config_slice=_config_slice(), script_path=Path(__file__),
                  config_slice_declared=True, include_driver_script_in_hash=False) as cell:
        env0 = _make_env(seed)
        cfg = _make_config(env0)
        agent = REEAgent(cfg)
        comp = _assert_default_composition(agent)

        p0 = _run_p0(agent, seed, 0)
        batt = _heldout(agent, seed)
        wake = _run_wake(agent, seed, P0_STEPS // STEPS_PER_EPISODE)
        s_pre = _clone_state(agent)
        head_pre = _head_snapshot(agent)
        world_buf_len = len(getattr(agent, "_world_experience_buffer", []) or [])
        sk_pre = _skill(agent, batt)
        sleep_metrics = agent.sleep_loop.force_cycle(agent) or {}
        s_post = _clone_state(agent)
        head_post = _head_snapshot(agent)
        hk = sorted(head_pre)
        sk_post = _skill(agent, batt, head_post)
        head_rel = _rel_l2(head_pre, head_post, hk)
        head_max_abs = max(float((head_post[k] - head_pre[k]).abs().max().item()) for k in hk)
        alias = {k for k in s_pre if any(f"{m}." in k for m in HEAD_MODULES)}
        other = [k for k in s_pre if k not in alias and s_pre[k].is_floating_point()]
        other_rel = _rel_l2(s_pre, s_post, other)

        head_x = {k: head_pre[k] + X_DOSE * (head_post[k] - head_pre[k]) for k in hk}
        reset_all_rng(seed + 31337)
        head_reinit = _head_snapshot(REEAgent(cfg))
        head_reinit_rel = _rel_l2(head_pre, head_reinit, hk)
        # NOISE_MATCHED: a random direction whose per-tensor L2 norm equals the
        # sleep change's, so a C1 pass can be read against generic magnitude
        # sensitivity (red-team finding 3).
        _g = torch.Generator().manual_seed(seed + 4242)
        head_noise = {}
        for k in hk:
            d = head_post[k] - head_pre[k]
            r_ = torch.randn(d.shape, generator=_g, dtype=d.dtype)
            head_noise[k] = head_pre[k] + r_ * (d.norm() / r_.norm().clamp(min=1e-12))
        head_noise_rel = _rel_l2(head_pre, head_noise, hk)

        bseed = seed + 500
        twins = {
            "A": _build_twin(cfg, s_pre, bseed),
            "NULL": _build_twin(cfg, s_pre, bseed),
            "SLEEP_BOTH": _build_twin(cfg, s_pre, bseed, head=head_post),
            "SLEEP_ROLLOUT_ONLY": _build_twin(cfg, s_pre, bseed, head=head_post,
                                              curiosity_head=head_pre),
            "SLEEP_CURIOSITY_ONLY": _build_twin(cfg, s_pre, bseed, curiosity_head=head_post),
            "SLEEP_X4_BOTH": _build_twin(cfg, s_pre, bseed, head=head_x),
            "FULL_SLEEP": _build_twin(cfg, s_post, bseed),
            "REINIT_BOTH": _build_twin(cfg, s_pre, bseed, head=head_reinit),
            "REINIT_CURIOSITY_ONLY": _build_twin(cfg, s_pre, bseed, curiosity_head=head_reinit),
            "NOISE_MATCHED": _build_twin(cfg, s_pre, bseed, head=head_noise),
        }
        # Assert from OUTPUT that each twin carries the head it is meant to.
        for _nm, _want in (("A", head_pre), ("SLEEP_BOTH", head_post),
                           ("SLEEP_X4_BOTH", head_x), ("FULL_SLEEP", head_post),
                           ("REINIT_BOTH", head_reinit), ("NOISE_MATCHED", head_noise)):
            _got = _head_snapshot(twins[_nm])
            assert all(torch.equal(_got[k], _want[k]) for k in hk), _nm
        recs = _run_probe(twins, seed, (P0_STEPS + WAKE_STEPS) // STEPS_PER_EPISODE)
        _ZG.observe(twins["A"])

        cmp = {t: _compare(recs["A"], recs[t]) for t in COMPARED}
        a_ticks = [r for r in recs["A"] if r is not None]
        live = [int(r["novelty_dev_range"] > 1e-9) for r in a_ticks]
        cur_live_frac = sum(live) / len(live) if live else 0.0
        srange = [r["score_range"] for r in a_ticks]
        nrange = [r["novelty_dev_range"] for r in a_ticks]
        null = cmp["NULL"]
        yoke = max(null.get("top1_diff_rate", 1.0), null.get("sel_diff_rate", 1.0),
                   null.get("tv_max", 1.0), float(null.get("ran_mismatch", 1)))

        row: Dict[str, Any] = {
            "seed": seed, "arm": "YOKED_TWINS", "composition_flags": comp,
            **p0, **wake,
            "world_experience_buffer_len_pre_sleep": world_buf_len,
            "skill_pre": sk_pre, "skill_post_head": sk_post,
            "head_rel_l2_change": head_rel, "head_max_abs_change": head_max_abs,
            "reinit_head_rel_l2_from_pre": head_reinit_rel,
            "noise_matched_head_rel_l2_from_pre": head_noise_rel,
            "skill_delta_post_minus_pre": (sk_post["persistence_relative_skill"]
                                           - sk_pre["persistence_relative_skill"]),
            "a_near_tie_frac": (sum(1 for r in a_ticks if math.isfinite(r["top2_gap_rel"])
                                    and r["top2_gap_rel"] < NEAR_TIE_REL) / len(a_ticks)
                                if a_ticks else float("nan")),
            "non_head_rel_l2_change": other_rel,
            "sleep_metrics": {k: (float(v) if isinstance(v, (int, float)) else str(v))
                              for k, v in sleep_metrics.items()},
            "n_a_e3_ticks": len(a_ticks),
            "curiosity_live_frac": cur_live_frac,
            "a_score_range_mean": _mean(srange),
            "a_novelty_dev_range_mean": _mean(nrange),
            "a_novelty_over_score_range_mean": _mean(
                [n / s for n, s in zip(nrange, srange) if s > 0]),
            "yoke_exact_measured": yoke,
            "comparisons": cmp,
        }
        excess = {t: (cmp[t].get("top1_diff_rate", float("nan"))
                      - null.get("top1_diff_rate", float("nan"))) for t in COMPARED}
        row["top1_excess_over_null"] = excess
        ready = (yoke <= 0.0 and head_rel > HEAD_MOVED_FLOOR
                 and cmp["REINIT_BOTH"].get("determinate", False)
                 and cmp["REINIT_BOTH"].get("top1_diff_rate", 0.0) >= POS_CONTROL_FLOOR
                 and null.get("n_paired_ticks", 0) >= MIN_TICKS)
        row["seed_ready"] = bool(ready)
        reach = bool(ready and cmp["SLEEP_BOTH"].get("determinate", False)
                     and excess["SLEEP_BOTH"] >= REACH_BAR)
        row["seed_sleep_reach"] = reach
        cell.stamp(row)
    print(f"verdict: {'PASS' if reach else 'FAIL'}", flush=True)
    return row


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def _mean(xs: List[float]) -> float:
    v = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return sum(v) / len(v) if v else float("nan")


def _f(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def run_experiment(dry_run: bool = False) -> Tuple[Dict[str, Any], float]:
    _set_schedule(dry_run)
    t0 = time.perf_counter()
    rows = [_run_cell(s) for s in SEEDS]
    ready = [r for r in rows if r["seed_ready"]]
    n_ready = len(ready)

    def _det(twin: str, rows_: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Seeds on which `twin` is DETERMINATE (>= MIN_TICKS paired, finite
        ticks). A twin that desynchronises its E3 schedule from A's (commit-
        triggered clock.phase_reset) loses pairs; it is then undetermined on
        that seed, never silently counted as 'no reach' (red-team finding 1)."""
        return [r for r in rows_ if r["comparisons"][twin].get("determinate", False)]

    def _count(twin: str, rows_: List[Dict[str, Any]]) -> int:
        return sum(1 for r in _det(twin, rows_)
                   if r["top1_excess_over_null"][twin] >= REACH_BAR)

    def _state(twin: str, rows_: List[Dict[str, Any]]) -> Tuple[str, int, int]:
        """pass / fail / undetermined over `rows_`. Undetermined when fewer than
        SEEDS_REQUIRED seeds are determinate for this twin."""
        n_det = len(_det(twin, rows_))
        c = _count(twin, rows_)
        if n_det < SEEDS_REQUIRED:
            return "undetermined", c, n_det
        return ("pass" if c >= SEEDS_REQUIRED else "fail"), c, n_det

    cur_ready = [r for r in ready if r["curiosity_live_frac"] >= CURIOSITY_LIVE_FRAC
                 and r["comparisons"]["REINIT_CURIOSITY_ONLY"].get("determinate", False)
                 and r["comparisons"]["REINIT_CURIOSITY_ONLY"].get("top1_diff_rate", 0.0)
                 >= POS_CONTROL_FLOOR]
    enough = n_ready >= SEEDS_REQUIRED
    st1, c1, d1 = _state("SLEEP_BOTH", ready)
    st2, c2, d2 = _state("SLEEP_ROLLOUT_ONLY", ready)
    st3, c3, d3 = _state("SLEEP_CURIOSITY_ONLY", cur_ready)
    st4, c4, d4 = _state("SLEEP_X4_BOTH", ready)
    st5, c5, d5 = _state("FULL_SLEEP", ready)
    st6, c6, d6 = _state("NOISE_MATCHED", ready)
    c1_pass, c2_pass, c3_pass, c4_pass = (st1 == "pass", st2 == "pass",
                                          st3 == "pass", st4 == "pass")

    # Route tokens are EXPLICIT per route; "undetermined" is never read as
    # "excluded" (red-team finding 2). C4 (x4 extrapolation) is recorded only,
    # not in the grid: linear extrapolation of an MLP head is not a dose
    # (red-team finding 4).
    if not enough:
        label = "substrate_not_ready_requeue"
    elif st1 == "undetermined":
        label = "sleep_reach_undetermined_twin_desync"
    elif c1_pass:
        label = ("sleep_head_change_reaches_e3_ranking__rollout_%s__curiosity_%s"
                 % ({"pass": "yes", "fail": "no"}.get(st2, "undetermined"),
                    {"pass": "yes", "fail": "no"}.get(st3, "undetermined")))
    else:
        label = "sleep_sized_head_change_below_reach"
    outcome = "PASS" if c1_pass else "FAIL"

    preconditions: List[Dict[str, Any]] = []
    for r in rows:
        s = r["seed"]
        null = r["comparisons"]["NULL"]
        preconditions += [
            {"name": f"seed{s}::yoke_exact", "measured": r["yoke_exact_measured"],
             "threshold": 0.0, "direction": "upper",
             "description": "NULL twin (identical weights) must reproduce A bit-exactly: "
                            "max of top1/sel diff rate, TV max and ran-mismatch count",
             "control": "NULL twin -- identical state_dict, identical per-step RNG",
             "met": r["yoke_exact_measured"] <= 0.0},
            {"name": f"seed{s}::sleep_moved_head", "measured": r["head_rel_l2_change"],
             "threshold": HEAD_MOVED_FLOOR, "comparator": ">",
             "description": "relative L2 change of the world head across the sleep cycle",
             "control": "S_pre vs S_post world-head parameters",
             "met": r["head_rel_l2_change"] > HEAD_MOVED_FLOOR},
            {"name": f"seed{s}::positive_control_reach",
             "measured": r["comparisons"]["REINIT_BOTH"].get("top1_diff_rate", 0.0),
             "threshold": POS_CONTROL_FLOOR,
             "description": "a freshly-initialised world head must re-rank >= 10% of E3 "
                            "ticks, else the instrument cannot register reach",
             "control": "REINIT_BOTH twin (both routes carry a random head); certifies the "
                        "top1_diff_rate instrument for BOTH routes jointly",
             "met": r["comparisons"]["REINIT_BOTH"].get("top1_diff_rate", 0.0) >= POS_CONTROL_FLOOR},
            {"name": f"seed{s}::n_ticks", "measured": float(null.get("n_paired_ticks", 0)),
             "threshold": float(MIN_TICKS),
             "description": "paired fresh-select E3 ticks (A vs NULL)",
             "control": "sample floor", "met": null.get("n_paired_ticks", 0) >= MIN_TICKS},
            {"name": f"seed{s}::curiosity_route_live", "measured": r["curiosity_live_frac"],
             "threshold": CURIOSITY_LIVE_FRAC,
             "description": "fraction of A's E3 ticks with non-zero 314a novelty deviation "
                            "range; scoped to C3 ONLY",
             "control": "A twin, live 314a channel; certifies the curiosity route readout only",
             "applies_to": "C3_curiosity_route",
             "met": r["curiosity_live_frac"] >= CURIOSITY_LIVE_FRAC},
            {"name": f"seed{s}::curiosity_route_positive_control",
             "measured": r["comparisons"]["REINIT_CURIOSITY_ONLY"].get("top1_diff_rate", 0.0),
             "threshold": POS_CONTROL_FLOOR,
             "description": "a random head swapped into ONLY the 314a summary read must "
                            "re-rank >= 10% of ticks; scoped to C3 ONLY. If unmet, the "
                            "314a route has no top-1 authority at this curiosity weight "
                            "and C3 cannot discriminate (rank_discordance still recorded)",
             "control": "REINIT_CURIOSITY_ONLY twin; certifies the curiosity route readout only",
             "applies_to": "C3_curiosity_route",
             "met": r["comparisons"]["REINIT_CURIOSITY_ONLY"].get("top1_diff_rate", 0.0)
                    >= POS_CONTROL_FLOOR},
        ]

    ready_seeds = [r["seed"] for r in ready]
    criteria = [
        {"name": "C1_sleep_both_reaches_ranking", "load_bearing": True, "passed": c1_pass,
         "state": st1, "measured": float(c1), "threshold": float(SEEDS_REQUIRED),
         "per_seed_bar": REACH_BAR, "n_ready_seeds": n_ready, "ready_seeds": ready_seeds,
         "n_determinate_seeds": d1},
        {"name": "C2_rollout_route_reaches", "load_bearing": False, "passed": c2_pass,
         "state": st2, "measured": float(c2), "threshold": float(SEEDS_REQUIRED),
         "per_seed_bar": REACH_BAR, "n_determinate_seeds": d2},
        {"name": "C3_curiosity_route_reaches", "load_bearing": False, "passed": c3_pass,
         "state": st3, "measured": float(c3), "threshold": float(SEEDS_REQUIRED),
         "per_seed_bar": REACH_BAR, "n_curiosity_live_ready_seeds": len(cur_ready),
         "n_determinate_seeds": d3},
        {"name": "C4_x4_extrapolation_RECORDED", "load_bearing": False, "passed": c4_pass,
         "state": st4, "measured": float(c4), "threshold": float(SEEDS_REQUIRED),
         "per_seed_bar": REACH_BAR, "n_determinate_seeds": d4,
         "note": "not a dose (MLP head, linear extrapolation); recorded, not in the grid"},
        {"name": "C5_full_sleep_reaches_CONTEXT", "load_bearing": False,
         "passed": st5 == "pass", "state": st5, "measured": float(c5),
         "threshold": float(SEEDS_REQUIRED), "per_seed_bar": REACH_BAR, "n_determinate_seeds": d5},
        {"name": "C6_noise_matched_reaches_CONTEXT", "load_bearing": False,
         "passed": st6 == "pass", "state": st6, "measured": float(c6),
         "threshold": float(SEEDS_REQUIRED), "per_seed_bar": REACH_BAR, "n_determinate_seeds": d6,
         "note": "random direction, per-tensor norm matched to the sleep change: reads C1 "
                 "against generic magnitude sensitivity; recorded, not in the grid"},
    ]
    combination_rule = ("outcome PASS iff >= %d ready seeds AND C1 (SLEEP_BOTH top1_diff_rate "
                        "minus NULL >= %.2f on >= %d ready seeds on which SLEEP_BOTH is "
                        "determinate, i.e. >= MIN_TICKS paired finite ticks). C2/C3 attribute "
                        "the route, each pass/fail/undetermined and spelled out in the label; "
                        "C4 (x4 extrapolation), C5 (whole-organism sleep) and C6 (norm-matched "
                        "random direction) are recorded context. Ready = yoke_exact AND "
                        "sleep_moved_head AND REINIT_BOTH determinate and >= positive-control "
                        "floor AND n_ticks." % (SEEDS_REQUIRED, REACH_BAR, SEEDS_REQUIRED))

    pos_ok = all(r["comparisons"]["REINIT_BOTH"].get("top1_diff_rate", 0.0) >= POS_CONTROL_FLOOR
                 for r in ready) and n_ready > 0
    non_deg = bool(enough and pos_ok)

    flat: Dict[str, Any] = {
        "n_seeds": len(rows), "n_ready_seeds": n_ready,
        "c1_count": c1, "c2_count": c2, "c3_count": c3, "c4_count": c4, "c5_count": c5,
        "c6_count": c6,
        "c1_pass": int(c1_pass), "c2_pass": int(c2_pass), "c3_pass": int(c3_pass),
        "c4_pass": int(c4_pass), "reach_bar": REACH_BAR,
        "c1_undetermined": int(st1 == "undetermined"), "c2_undetermined": int(st2 == "undetermined"),
        "c3_undetermined": int(st3 == "undetermined"),
    }
    for r in rows:
        s = r["seed"]
        flat[f"head_rel_l2_change_seed{s}"] = _f(r["head_rel_l2_change"])
        flat[f"non_head_rel_l2_change_seed{s}"] = _f(r["non_head_rel_l2_change"])
        flat[f"skill_pre_seed{s}"] = _f(r["skill_pre"]["persistence_relative_skill"])
        flat[f"skill_post_seed{s}"] = _f(r["skill_post_head"]["persistence_relative_skill"])
        flat[f"curiosity_live_frac_seed{s}"] = _f(r["curiosity_live_frac"])
        flat[f"novelty_over_score_range_seed{s}"] = _f(r["a_novelty_over_score_range_mean"])
        flat[f"seed_ready_seed{s}"] = int(r["seed_ready"])
        flat[f"yoke_exact_measured_seed{s}"] = _f(r["yoke_exact_measured"])
        flat[f"skill_delta_post_minus_pre_seed{s}"] = _f(r["skill_delta_post_minus_pre"])
        flat[f"a_near_tie_frac_seed{s}"] = _f(r["a_near_tie_frac"])
        for t in COMPARED:
            c = r["comparisons"][t]
            tl = t.lower()
            for k in ("top1_diff_rate", "top1_diff_rate_early", "sel_diff_rate",
                      "tv_mean", "cand_identical_frac", "rank_discordance_mean",
                      "desync_frac", "n_paired_ticks", "n_nonfinite_ticks"):
                flat[f"{tl}_{k}_seed{s}"] = _f(c.get(k))
            flat[f"{tl}_top1_excess_seed{s}"] = _f(r["top1_excess_over_null"][t])
    flat = {k: v for k, v in flat.items() if v is not None}

    manifest: Dict[str, Any] = {
        "queue_id": QUEUE_ID,
        "validates_substrate": VALIDATES_SUBSTRATE,
        "bears_on": BEARS_ON,
        "governance_refs": GOVERNANCE_REFS,
        "run_id": f"{EXPERIMENT_TYPE}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "non_contributory",
        "outcome": outcome,
        "timestamp_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "sleep_driver_pattern": "manual-cycle-loop (one force_cycle between WAKE and PROBE)",
        "non_degenerate": non_deg,
        "degeneracy_reason": (None if non_deg else
                              "fewer than %d ready seeds, or the REINIT positive control "
                              "did not register reach" % SEEDS_REQUIRED),
        "ethics_preflight": {
            "involves_negative_valence": False, "involves_suffering_like_state": False,
            "involves_self_model": False, "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False, "decision": "allow"},
        "readout": flat,
        "arm_results": rows,
        "custom_information": {
            "premise_correction": "GFLAG-0437: e2.world_forward reaches E3 by default "
                                  "through HippocampalModule -> E2.rollout_with_world",
            "route_split_mechanism": "SLEEP_ROLLOUT_ONLY / SLEEP_CURIOSITY_ONLY swap the "
                                     "world head only around agent._curiosity_candidate_"
                                     "summaries (the sole 314a world_forward read)",
        },
        "interpretation": {
            "label": label,
            "combination_rule": combination_rule,
            "preconditions": preconditions,
            "criteria": criteria,
            "criteria_non_degenerate": {
                "C1_sleep_both_reaches_ranking": non_deg,
                "C2_rollout_route_reaches": non_deg,
                "C3_curiosity_route_reaches": bool(non_deg and st3 != "undetermined"),
                "C4_x4_extrapolation_RECORDED": bool(non_deg and st4 != "undetermined"),
                "C5_full_sleep_reaches_CONTEXT": bool(non_deg and st5 != "undetermined"),
                "C6_noise_matched_reaches_CONTEXT": bool(non_deg and st6 != "undetermined")},
        },
    }
    return manifest, t0


if __name__ == "__main__":
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--dry-run", action="store_true")
    _args = _ap.parse_args()

    _manifest, _t0 = run_experiment(dry_run=_args.dry_run)
    _out_path = write_flat_manifest(
        _manifest, dry_run=_args.dry_run, config=_config_slice(),
        seeds=SEEDS, script_path=Path(__file__), started_at=_t0,
        z_goal_stream_stats=_ZG.stats())

    print(f"[{EXPERIMENT_TYPE}] outcome={_manifest['outcome']} "
          f"label={_manifest['interpretation']['label']}", flush=True)
    _o = str(_manifest["outcome"]).upper()
    emit_outcome(outcome=_o if _o in ("PASS", "FAIL") else "FAIL",
                 manifest_path=_out_path, dry_run=_args.dry_run)
