"""V3-EXQ-1105a pre-flight Mac smoke (NOT a validation run): calibration seed 45 (trapped; excluded from
admission), full preamble budgets, arms NULL0, NULL1 (M2 sign-shuffled) and NOISYHACK at 1,500 steps.
Measures: null theta_harm_final, own-sd D_W z, increment tau_int / lag-1, kappa (vote-event coupling),
and NOISYHACK's h/g split and its D_W z. Thresholds in the driver were fixed before this ran. ASCII only."""
import json, math, sys, time
from pathlib import Path
WT = Path("/Users/dgolden/REE_Working/.scratch/breakthrough-20260924/w1105a/ree-v3-wt")
sys.path.insert(0, str(WT)); sys.path.insert(0, str(WT / "experiments/_probes/nulldet3"))
sys.path.insert(0, str(WT / "experiments/_probes/nulldet4"))
import nulldet_core as C  # noqa
import rules_1105a as NR  # noqa
import importlib.util
spec = importlib.util.spec_from_file_location("drv", str(WT / "experiments/v3_exq_1105a_grounded_valuation_null_detector_v4.py"))
drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)
seed = 45
out = Path(__file__).resolve().parent / "PREFLIGHT_s45.json"
t0 = time.time()
enc_state, headD, evT, gate_n, agent, p0 = C.preamble(seed, C.PREAMBLE_BUDGETS_FULL)
del agent
res = {"seed": seed, "gate_n": int(gate_n), "preamble_s": round(time.time() - t0, 1), "arms": {}}
print("PREAMBLE gate_n=%d t=%.0f" % (gate_n, time.time() - t0), flush=True)
for name in ["NULL0", "NOISYHACK", "NULL1"]:
    ta = time.time()
    rule = NR.NoisyHacker(seed, drv.NULL_COIN_BASE) if name == "NOISYHACK" else NR.M2SignShuffled(seed, drv._null_coin_offset(int(name[4:])))
    arm = C.run_arm(seed, enc_state, headD, evT, gate_n, 1500, rule)
    s = drv._summarise_arm(arm); s["rule_diagnostics"] = rule.diagnostics(); s["dw"] = drv._dw(s)
    s["wall_s"] = round(time.time() - ta, 1)
    s.pop("theta_harm_trajectory", None); s.pop("ends", None)
    res["arms"][name] = s
    json.dump(res, open(out, "w"), indent=1)
    print("%s theta_end=%s n=%d sd_h=%.4f z_W=%.2f tau=%s lag1=%s maxjump=%.3f diag=%s wall=%.0f" % (
        name, [round(x, 3) for x in s["theta_end"]], s["n_ticks"], s["dtheta_sd"][1], s["dw"]["z"],
        None if s["harm_increment_tau_int"] is None else round(s["harm_increment_tau_int"], 2),
        None if s["harm_increment_lag1_autocorr"] is None else round(s["harm_increment_lag1_autocorr"], 2),
        s["max_abs_harm_increment"], json.dumps(s["rule_diagnostics"])[:400], s["wall_s"]), flush=True)
res["t_total_s"] = round(time.time() - t0, 1)
json.dump(res, open(out, "w"), indent=1)
print("DONE t=%.0f" % (time.time() - t0))
