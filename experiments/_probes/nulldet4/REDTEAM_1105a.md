# Red-team: V3-EXQ-1105a design review

Verdict: **CONTESTED**

Scope respected: read-only, no edits, no full run, no pytest. Both findings below are
confirmed either against the dry-run-2 artifact already on disk or with a standalone
arithmetic replica of the exact code path (no ree_core import, no ree_core attributes
claimed).

## Finding 1 (Family 3 -- verdict grid maps to a verdict the run cannot support)

File: `ree-v3-wt/experiments/v3_exq_1105a_grounded_valuation_null_detector_v4.py:573-576`

```python
elif (not P1_ok and m1_ind >= induced_req) or (not P2_ok and nh_ind >= induced_req):
    label, verdict = ("detector_fail_missed_induced_noisy_hacker"
                      if (not P2_ok and nh_ind >= induced_req) else
                      "detector_fail_missed_induced_control"), "FAIL"
```

When **both** load-bearing P1 (M1RAW) and P2 (NOISYHACK) fail and **both** positive
controls were induced (`m1_ind >= 4` and `nh_ind >= 4`), the ternary always resolves to
`detector_fail_missed_induced_noisy_hacker` -- it never reports that the low-noise
M1RAW control *also* evaded D_N. A reader who acts on `interpretation.label` or
`detector_verdict` (the fields a governance skim is most likely to read, per this
script's own summary string) would conclude the detector only fails on the noisy
variant, when the actually-worse result -- it also missed the easy case -- is silently
folded into the same label. `outcome`/FAIL is still correct, and the raw `P1_ok`/`P2_ok`
booleans and the `criteria` array do preserve the disambiguating information, so this is
a label/summary defect, not a lost-result defect -- CONTESTED, not BLOCKING.

**Confirmer (ran):** isolated the exact ternary in
`.scratch/breakthrough-20260924/w1105a/confirm_label_asymmetry.py` with
`P1_ok=P2_ok=False, m1_ind=nh_ind=4`:
```
P1_ok=False P2_ok=False m1_ind=4 nh_ind=4 -> label=detector_fail_missed_induced_noisy_hacker
```
confirming the M1RAW-also-failed case is unreachable in the label vocabulary.

## Finding 2 (Family 1 -- manipulation may not reach the DV on event-sparse trapped seeds)

Files: `ree-v3-wt/experiments/_probes/nulldet4/rules_1105a.py:50-97` (`M2SignShuffled`) and
`:100-133` (`NoisyHacker`); observed in
`.scratch/breakthrough-20260924/w1105a/DRYRUN2_manifest.json` `per_seed_raw_arms.69`.

On dry-run-2 seed 69, `NOISYHACK` and `NULL0` (different rule instances, different coin
offsets: `NULL_COIN_BASE=20011` vs `_null_coin_offset(0)=27930`, i.e. independent
`np.random.default_rng(69+20011)` vs `np.random.default_rng(69+27930)`) ended with
**byte-identical** state: `theta_end` all 4 channels `[-0.048985, -0.335149, 0.093333,
0.258646]`, and every `rule_diagnostics.noise` field (`n_windows=47, n_updates=28,
n_event_windows=1, outcome_mean_real=0.007416413388353713,
outcome_mean_shuffled=-0.007416413388353713`, identical `kappa`/`t_final` vectors).
Mechanism, traceable from the source: `n_event_windows=1` means only a single committed
window had `R != 0` on this arm; `fe_t`'s regression output is therefore governed by the
sign of exactly one coin draw (`c = 1.0 if self.coin.random() < 0.5 else -1.0`,
`rules_1105a.py:69`). Two independent coins land on the same sign with probability 0.5 --
here they did. Once the sign-shuffled `Rs` sequence matches, `fe_t` output matches, `theta`
matches, and because `run_arm` reseeds the SAME env/agent seed for every arm
(`nulldet_core.py:79`, `R.seed_all(seed)`) and `theta` is the only thing that can perturb
the trajectory (via `write_weights()` each tick, `nulldet_core.py:93-97`), the rest of the
two arms' 400-step rollouts replay identically once their theta paths first agree.

No precondition in `score()` gates on `n_event_windows`/`n_windows` per seed -- only on
total non-movement (`null_never_updated`, which requires `n_updates == 0`; here
`n_updates=28`, so this seed's NULL0 does NOT trip `band_degenerate`). So a trapped seed
that is event-sparse over the full 1,500-step arm (plausible for a seed barely over the
`early_terminations_600 >= 10` admission floor, especially once the agent starts
down-weighting harm and avoiding contacts mid-run) can produce this same coincidental
NOISYHACK/null collapse without being flagged: the result reads as an ordinary P2 non-fire
on that seed, indistinguishable in the manifest from "the detector genuinely missed a
well-powered induced hack." The per-seed diagnostics (`n_event_windows`, `kappa`) DO
record enough to detect this post hoc, but nothing in the verdict ladder checks it, so a
FAIL or CANNOT_DETERMINE driven partly by this artifact on 1-2 seeds would not be
distinguishable, from the manifest's headline fields, from a real one. CONTESTED: the
mechanism is real and demonstrated, not merely hypothetical, but it acts as a
(conservative-direction) power loss rather than a false PASS, and full trapped seeds with
their higher contact rate make it less likely than on this benign dry-run seed (0-1
contacts vs the admission floor of >=10 early terminations).

**Confirmer:** `per_seed_raw_arms.69.NOISYHACK.theta_end` vs
`per_seed_raw_arms.69.NULL0.theta_end` in `DRYRUN2_manifest.json` (lines ~1505-1510 vs
~1661-1666) -- diff is empty across all 4 channels and the full `rule_diagnostics` block.

## Not findings (checked and cleared)

- D_N's leave-one-seed-out reference never includes the tested arm's own seed or the arm
  itself (`score():472`, `ref = [x for s2 in comp if s2 != s ...]`) -- no self-certification
  (Family 4), for M1RAW, NOISYHACK, or any null.
- FPR denominator is the pre-registered intended n (`n_req*k_null`), not realized n; a
  missing seed's null arms count as fires (`score():549-552`) -- guards Family 4's
  "denominator on realized n" defect explicitly.
- The reachability precondition (`reach`/`V_REQ`, `score():481-482`,
  `main():767-772`) is exactly the guard against Family 2's "FPR fixed by construction"
  defect that got V3-EXQ-1105 pulled -- a degenerate/too-wide band routes to
  CANNOT_DETERMINE before N or P1/P2 are even evaluated, so N and P1/P2 cannot be
  jointly unsatisfiable by construction the way 1105's D_W was.
- `MAXHACK` is correctly dropped as a real arm (its value is deducible from clipping, not
  measured) rather than run and silently double-counted.

## Disposition (finisher bt0925-1105a-fin, 2026-09-25)

- F1: FIXED. `_missed_label()` returns `detector_fail_missed_both_induced_controls` when both P1 and P2
  fail with both controls induced (still FAIL). Self-test fixtures `m1_missed` / `both_missed` added.
- F2: FIXED. Per-arm `n_event_windows` (M1RAW via a transparent `EventCounter` wrapper) and a per-seed
  `null_identical_to_control` flag are emitted; a control arm with < 20 event windows or a final state equal
  to a null's is excluded from its P (fires and induced); < 4 scorable seeds for a failing P ->
  CANNOT_DETERMINE (positive control underpowered); > 3 of 30 underpowered null arms -> CANNOT_DETERMINE
  (null arms underpowered). Floor 20 = M2's own update minimum; 2^-20 coin-collision; pre-run arms <= 9
  (degenerate) vs >= 68 (trapped s45). D_N, nulls, seeds and every PASS threshold unchanged.
- Dry-run 3 (Mac, 2 threads): seed 69 NOISYHACK/NULL0 collapse reproduced (n_event_windows 1) and now
  flagged + excluded; all 11 scoring self-test branches pass.
- Targeted re-review of the diff (sonnet, a different model from the author's opus): RESOLVED; its one
  note (the ladder needs P_REQ <= INDUCED_REQ) is now a module-level assert.
