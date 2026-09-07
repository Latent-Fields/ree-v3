"""DV-RANGE PROBE for campaign W4-S2 item A1, leg H2 (waypoint_field_consumer_reach /
H-wpfield-zworld-interface). NOT an experiment; no manifest, no queue entry.

Question it answers, and ONLY this: what is the achievable RANGE of the H2 dependent
variable -- linear decodability of the pending waypoint's DIRECTION -- at (a) the raw
observation, which is H2's declared upstream reference, and (b) z_world, which is where
H2 says the signal is lost. Without this number a pre-registered H2 threshold is a guess,
and V3-EXQ-1005 was REFUSED at red-team for exactly that omission (its design floor sat
above the achievable ceiling).

DESIGN NOTE carried forward to the driver: the field is ABLATED BY ZEROING the trailing
25 dims, NOT by turning the env flag off. Turning the flag off shrinks world_obs_dim
275 -> 250, which changes the encoder's first Linear and therefore its init RNG draw, so
the ON/OFF pair would not be matched. Zeroing keeps the encoder byte-identical at init
and makes the manipulation purely the CONTENT of the trailing 25 dims -- the same repair
V3-EXQ-1004 applied to its readers (its F7 zero-padding note).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.latent.stack import SplitEncoder

GRID_SIZE, N_WAYPOINTS = 12, 3
WAYPOINT_VISIT_REWARD, WAYPOINT_FIELD_DECAY = 0.2, 0.25
NUM_HAZARDS = NUM_RESOURCES = 0
ENERGY_DECAY = 0.0
HAZARD_FREE_CONTAMINATION_GATE = True
WAYPOINT_COMPLETION_REWARD, SEQUENCE_COMMITMENT_TIMEOUT = 0.8, 20
FIELD_DIMS = 25
WORLD_DIM = 32          # z_world width, matches the 1002/1008 lineage
HIDDEN = 128


def build_env(seed):
    """1004 bench geometry, field flag ALWAYS ON -- see the module docstring."""
    return CausalGridWorldV2(
        seed=seed, size=GRID_SIZE, use_proxy_fields=True, subgoal_mode=True,
        num_waypoints=N_WAYPOINTS, waypoint_visit_reward=WAYPOINT_VISIT_REWARD,
        subgoal_arrival_position_check=True, num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES, energy_decay=ENERGY_DECAY,
        hazard_free_contamination_gate=HAZARD_FREE_CONTAMINATION_GATE,
        waypoint_completion_reward=WAYPOINT_COMPLETION_REWARD,
        sequence_commitment_timeout=SEQUENCE_COMMITMENT_TIMEOUT,
        waypoint_proximity_field_enabled=True,
        waypoint_field_decay=WAYPOINT_FIELD_DECAY,
    )


def direction_label(env):
    """4-class direction to the PENDING waypoint from env ground truth (torus-aware),
    the same quantity the 1004 oracle acts on. -1 when no pending target."""
    idx = int(getattr(env, "_next_waypoint_idx", 0))
    wps = getattr(env, "waypoints", []) or []
    if not wps or idx >= len(wps):
        return -1
    ax, ay = int(env.agent_x), int(env.agent_y)
    wx, wy = int(wps[idx][0]), int(wps[idx][1])
    dx, dy = wx - ax, wy - ay
    if getattr(env, "toroidal", False):
        n = env.size
        if abs(dx) > n - abs(dx): dx = -np.sign(dx) * (n - abs(dx))
        if abs(dy) > n - abs(dy): dy = -np.sign(dy) * (n - abs(dy))
    if dx == 0 and dy == 0:
        return -1
    return (0 if dx < 0 else 1) if abs(dx) >= abs(dy) else (2 if dy < 0 else 3)


def collect(seed, n_episodes, steps, rng):
    """Random-policy rollouts. A random walk is the honest state distribution for a
    DECODABILITY probe: it asks what the representation CARRIES, not what a policy that
    already navigates visits (which would be circular)."""
    env = build_env(seed)
    X, Y = [], []
    for _ in range(n_episodes):
        obs = env.reset()
        obs_dict = obs[1] if isinstance(obs, tuple) else obs
        for _ in range(steps):
            lab = direction_label(env)
            if lab >= 0 and isinstance(obs_dict, dict) and "world_state" in obs_dict:
                X.append(np.asarray(obs_dict["world_state"], np.float32).reshape(-1))
                Y.append(lab)
            a = int(rng.integers(0, int(env.action_dim)))
            # CausalGridWorldV2.step -> (obs_tensor, reward, done, info, obs_dict);
            # the dict is the LAST element, not the first. Reading index 0 silently
            # yields a tensor and drops every state (4 usable of 240 in the first smoke).
            step = env.step(a)
            obs_dict = step[-1] if isinstance(step[-1], dict) else {}
            if bool(step[2]):
                break
    return np.asarray(X, np.float32), np.asarray(Y, np.int64)


def linear_probe(Xtr, Ytr, Xte, Yte, epochs=300, lr=0.05, seed=0):
    """Multinomial logistic regression, full-batch LBFGS-free SGD. Deliberately LINEAR:
    H2's claim is about what z_world LINEARLY carries, so a nonlinear head would answer
    a different question."""
    g = torch.Generator().manual_seed(seed)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    xt = torch.as_tensor((Xtr - mu) / sd); xe = torch.as_tensor((Xte - mu) / sd)
    yt = torch.as_tensor(Ytr); ye = torch.as_tensor(Yte)
    W = torch.zeros(xt.shape[1], 4, requires_grad=True)
    b = torch.zeros(4, requires_grad=True)
    with torch.no_grad():
        W += 0.01 * torch.randn(W.shape, generator=g)
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(xt @ W + b, yt).backward()
        opt.step()
    with torch.no_grad():
        return float(((xe @ W + b).argmax(1) == ye).float().mean())


def encode(X, world_obs_dim, seed):
    """UNTRAINED SplitEncoder -- the architectural floor. If the field does not survive
    even an untrained random compression, the loss is structural (width/scale), not
    learned; if it does survive here, H2's live question is what TRAINING does to it."""
    torch.manual_seed(seed)
    enc = SplitEncoder(body_obs_dim=12, world_obs_dim=world_obs_dim, self_dim=32,
                       world_dim=WORLD_DIM, topdown_dim=32, hidden_dim=HIDDEN)
    # The world pathway ONLY. z_world's other inputs (self/topdown) carry no waypoint
    # information, so including them would add matched noise to both arms and dilute
    # the very lift being measured.
    with torch.no_grad():
        h = enc.world_encoder(torch.as_tensor(X))
    return h.numpy()


def randproj(X, seed, out_dim=WORLD_DIM):
    """DECISIVE CONTROL. A random orthonormal-ish LINEAR projection 275 -> 32, no ReLU,
    no encoder. Johnson-Lindenstrauss says such a map is a near-isometry for linearly
    decodable structure, so whatever decodability survives HERE survives for reasons that
    have nothing to do with REE. This is the floor an H2 verdict arm must beat DOWNWARD
    to mean anything -- see the note in the probe's output."""
    g = torch.Generator().manual_seed(10_000 + seed)
    P = torch.randn(X.shape[1], out_dim, generator=g) / np.sqrt(X.shape[1])
    with torch.no_grad():
        return (torch.as_tensor(X) @ P).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--json", action="store_true")
    a = p.parse_args()

    rows = []
    for seed in a.seeds:
        rng = np.random.default_rng(seed)
        X, Y = collect(seed, a.episodes, a.steps, rng)
        if len(X) < 200:
            print(f"seed {seed}: only {len(X)} usable states -- SKIP"); continue
        n = len(X); perm = np.random.default_rng(seed).permutation(n)
        ntr = int(0.7 * n); tr, te = perm[:ntr], perm[ntr:]
        wdim = X.shape[1]
        X_off = X.copy(); X_off[:, -FIELD_DIMS:] = 0.0   # zero-ablation, see docstring

        maj = float(np.bincount(Y[te], minlength=4).max() / len(te))
        raw_on = linear_probe(X[tr], Y[tr], X[te], Y[te], seed=seed)
        raw_off = linear_probe(X_off[tr], Y[tr], X_off[te], Y[te], seed=seed)

        Z_on = encode(X, wdim, seed); Z_off = encode(X_off, wdim, seed)
        if Z_on is None:
            zw_on = zw_off = float("nan")
        else:
            zw_on = linear_probe(Z_on[tr], Y[tr], Z_on[te], Y[te], seed=seed)
            zw_off = linear_probe(Z_off[tr], Y[tr], Z_off[te], Y[te], seed=seed)

        R_on = randproj(X, seed); R_off = randproj(X_off, seed)
        rp_on = linear_probe(R_on[tr], Y[tr], R_on[te], Y[te], seed=seed)
        rp_off = linear_probe(R_off[tr], Y[tr], R_off[te], Y[te], seed=seed)

        rows.append(dict(seed=seed, n_states=n, world_obs_dim=int(wdim), majority=maj,
                         raw_on=raw_on, raw_off=raw_off, raw_lift=raw_on - raw_off,
                         zworld_on=zw_on, zworld_off=zw_off, zworld_lift=zw_on - zw_off,
                         randproj_on=rp_on, randproj_off=rp_off,
                         randproj_lift=rp_on - rp_off))
        print(f"seed {seed}: n={n} dim={wdim} majority={maj:.3f} | "
              f"RAW on={raw_on:.3f} off={raw_off:.3f} lift={raw_on-raw_off:+.3f} | "
              f"ZWORLD(untrained) lift={zw_on-zw_off:+.3f} | "
              f"RANDPROJ lift={rp_on-rp_off:+.3f}")

    if rows:
        print("\n=== DV RANGE (mean over seeds) ===")
        for k in ("raw_lift", "zworld_lift", "randproj_lift", "raw_on", "zworld_on",
                  "majority"):
            print(f"  {k:12s} {np.mean([r[k] for r in rows]):+.4f}")
    if a.json:
        print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
