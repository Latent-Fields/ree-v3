"""Independent confirmation of red-team F1: is T biased positive with NO content signal?

Built from scratch (not the reviewer's script), using only the arithmetic the
experiment uses. NO encoder, NO env: reps are one shared unit direction plus
iid gaussian noise, so the attained/non-attained LABELS carry ZERO information
by construction. If T still 'separates', T is measuring EMA self-overlap.
"""
import torch

ALPHA, DECAY, DIM = 0.05, 0.005, 32
N_STEPS, N_ATT, N_PERM = 400, 60, 400

def cos(a, b):
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-8))

def replay(reprs, positions, n_steps):
    parent = torch.zeros(DIM)
    cset, k = set(positions), 0
    a = min(1.0, ALPHA)
    for t in range(1, n_steps + 1):
        parent = parent * (1.0 - DECAY)
        if t in cset and k < len(reprs):
            parent = (1.0 - a) * parent + a * reprs[k]
            k += 1
    return parent

def meanv(g):
    return torch.stack(g).mean(dim=0)

def T(parent, A, B):
    return cos(parent, meanv(A)) - cos(parent, meanv(B))

print("F1 CHECK: T on data whose labels carry ZERO signal by construction")
print("reps = shared unit direction + iid N(0, sigma^2); n_att=%d n_tot=%d" % (N_ATT, N_STEPS))
print()
print(" sigma | T_obs      null_p95    pct   | C1 would say")
for sigma in (0.02, 0.05, 0.10, 0.20):
    fires = 0
    for trial in range(12):
        g = torch.Generator().manual_seed(1000 * trial + int(sigma * 1000))
        mu = torch.zeros(DIM); mu[0] = 1.0
        reps = [mu + sigma * torch.randn(DIM, generator=g) for _ in range(N_STEPS)]
        idx = torch.randperm(N_STEPS, generator=g).tolist()
        att_i = sorted(idx[:N_ATT])
        att = [reps[i] for i in att_i]
        non = [reps[i] for i in idx[N_ATT:]]
        pos = [i + 1 for i in att_i]
        parent = replay(att, pos, N_STEPS)
        t_obs = T(parent, att, non)
        pool = att + non
        null = []
        for _ in range(N_PERM):
            pm = torch.randperm(len(pool), generator=g).tolist()
            null.append(T(parent, [pool[i] for i in pm[:N_ATT]], [pool[i] for i in pm[N_ATT:]]))
        null.sort()
        p95 = null[int(0.95 * len(null))]
        if t_obs > p95 and t_obs >= 0.0002:
            fires += 1
        if trial == 0:
            first = (t_obs, p95, 100.0 * sum(1 for v in null if v < t_obs) / len(null))
    print(" %.2f  | %.6f  %.6f  %5.1f | SEPARATES on %d/12 trials"
          % (sigma, first[0], first[1], first[2], fires))
print()
print("Any 'SEPARATES' above is a FALSE POSITIVE: the labels are random.")
