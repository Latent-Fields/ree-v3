# Probe archived with the V3-EXQ-1005 refusal record (written by session hopeful-solomon-01a60c, 2026-09-07).
# See REE_assembly/evidence/planning/exq1005_mech267_location_dv_redteam_blocking_20260907.md
# Throwaway substrate probe: run from ree-v3 root; writes nothing under evidence/.
"""Own confirmer: per-iteration ao_std min/max and E2 action-object action-dependence, seed 0."""
import sys, statistics
sys.path.insert(0, "/Users/dgolden/REE_Working/ree-v3")
sys.path.insert(0, "/Users/dgolden/REE_Working/ree-v3/experiments")
import torch
import importlib.util
spec = importlib.util.spec_from_file_location("drv", "/Users/dgolden/REE_Working/ree-v3/experiments/v3_exq_1005_mech267_mode_content_location_separation.py")
drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)
for arm in ("CTRL_OFF", "H2_ONLY", "NOISE_H3_REF"):
    seed = 0
    torch.manual_seed(seed); hip = drv._make_hippocampal(arm)
    torch.manual_seed(seed + 900_000); zw = torch.randn(1, drv.WORLD_DIM); zs = torch.randn(1, drv.SELF_DIM)
    for mode in ("internal_planning", "offline_consolidation"):
        torch.manual_seed(drv._cell_sampling_seed(seed, arm, mode))
        tr = hip.propose_trajectories(zw, zs, num_candidates=drv.NUM_CANDIDATES, operating_mode={mode: 1.0})
        diag = hip.get_last_propose_diagnostics()
        its = diag.get("cem_iteration_diagnostics") or diag.get("cem_iterations") or []
        print(f"arm={arm} mode={mode} diag_keys={sorted(k for k in diag.keys())[:12]}")
        for i, it in enumerate(its):
            if isinstance(it, dict):
                print(f"   iter{i}: " + ", ".join(f"{k}={it[k]}" for k in sorted(it) if 'std' in k or 'elite' in k)[:400])
        # action-object action-dependence: recomputed ao across candidates
        try:
            aos = [t.get_action_object_sequence() for t in tr]
            aos = torch.stack([a if torch.is_tensor(a) else torch.as_tensor(a) for a in aos])
            print(f"   recomputed ao across-candidate std: mean={aos.float().std(dim=0).mean():.5f} max={aos.float().std(dim=0).max():.5f}")
        except Exception as e:
            print("   ao recompute read failed:", e)
