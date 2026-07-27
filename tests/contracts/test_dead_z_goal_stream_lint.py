"""Contracts for the dead-z_goal-stream lint.

Surfaces under test:
  (1) validate_experiments.dead_z_goal_stream_lint -- flags a driver that enables a
      z_goal-dependent config but never writes z_goal, so every goal-gated branch
      silently no-ops for the whole run.
  (2) validate_experiments.py --checks dead_z_goal_stream -- the selector, and the
      invariant that this gate is WARN-ONLY IN BOTH MODES (never hardens under --paths,
      never affects the exit code even under --strict).
  (3) The SUBSTRATE FACTS the gate rests on, asserted against ree_core rather than
      trusted: `update_z_goal` is the sole z_goal writer, and `REEAgent.reset()` does
      NOT reset `goal_state`.

WHY THIS GATE EXISTS. `z_goal` has exactly ONE writer: the explicit
`REEAgent.update_z_goal(...)`. Nothing in sense() / generate_trajectories() /
select_action() / update_residue() touches it. A driver that hand-rolls its inner loop
and omits the call therefore runs with z_goal pinned at zero-init for the whole run;
`GoalState.is_active()` returns False, `agent.py` passes `current_z_goal=None` to every
consumer, and the E3 goal term / MECH-293 ghost probes / MECH-288 slow BOCPD scale /
MECH-189 super-ordinal anchors / SD-057 incentive bank / MECH-295 liking->approach
bridge / frontopolar counterfactual read all go quiet. Nothing raises, and no manifest
field shows it.

Confirmed twice, in opposite orders:
  - V3-EXQ-626 (2026-06-01): a bespoke episode loop omitted the call, so z_goal sat at
    zero across every arm of a diagnostic whose C1-C5 were ALL keyed on z_goal norm.
    Superseded by 626a (wired the call) and 626b (forced-seed positive control).
  - V3-EXQ-830 (2026-07-27): reused the V3-EXQ-816 policy-decomposition harness -- which
    omits the call, harmless for 816, which never reads z_goal -- to measure MECH-288's
    slow scale, which does read it. Its readiness gate refused the dry-run smoke TWICE on
    `zgoal_present_frac = 0.0`. Without that gate the run would have burned ~5h of cloud
    time and reported a wiring artefact as a finding that CLOSED a design question.

SCOPE. This gates NEW scripts. The landed carriers' runs are complete and are NOT
retro-edited, hence WARN-only and hence the fire-rate pin is a BACKLOG SIZE, not a
target of zero. It is knob-gated rather than shape-gated on purpose: ~500 corpus scripts
hand-roll a loop without the call and are all CORRECT, because they never enable z_goal
and `update_z_goal` would early-return anyway.
"""
import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import validate_experiments as V  # noqa: E402

EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def _run(*args):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "validate_experiments.py"), *args],
        capture_output=True, text=True, cwd=str(REPO_ROOT))


def _lint_src(src: str):
    """Lint a synthetic script written into experiments/ (so relative scoping holds)."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(src)
        name = f.name
    try:
        return V.dead_z_goal_stream_lint(Path(name))
    finally:
        os.unlink(name)


_DEFECTIVE = '''
"""A driver that enables z_goal and never writes it."""
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig


def run(env):
    cfg = REEConfig.from_dims(world_dim=32, z_goal_enabled=True, goal_weight=0.5)
    agent = REEAgent(cfg)
    obs = env.reset()
    for _ in range(200):
        latent = agent.sense(obs)
        action = agent.select_action(latent)
        obs, r, done, info = env.step(action)
    return agent
'''


def _variant(**subs):
    src = _DEFECTIVE
    for old, new in subs.items():
        src = src.replace(old.replace("__", " "), new)
    return src


# ---- (1) the defect fires -------------------------------------------------------------

def test_dzg_fires_on_the_830_shape():
    out = _lint_src(_DEFECTIVE)
    assert out is not None
    assert "DEAD Z_GOAL STREAM" in out
    assert "z_goal_enabled" in out


def test_dzg_fires_on_attribute_assignment_form():
    src = _DEFECTIVE.replace(
        "cfg = REEConfig.from_dims(world_dim=32, z_goal_enabled=True, goal_weight=0.5)",
        "cfg = REEConfig.from_dims(world_dim=32)\n    cfg.goal.z_goal_enabled = True")
    assert _lint_src(src) is not None


def test_dzg_fires_on_the_sd024_producer_knob_without_z_goal_enabled():
    """SD-024 dies with the call even when z_goal is never enabled.

    `accumulate_benefit` is invoked from inside update_z_goal but AHEAD of the
    `goal_state is None` guard, so `benefit_terrain_live_producer` is reachable only
    through the call -- independently of z_goal_enabled.
    """
    src = _DEFECTIVE.replace("z_goal_enabled=True, goal_weight=0.5",
                             "benefit_terrain_live_producer=True")
    out = _lint_src(src)
    assert out is not None
    assert "benefit_terrain_live_producer" in out


# ---- (2) the discharges -------------------------------------------------------------

def test_dzg_update_z_goal_call_is_exempt():
    src = _DEFECTIVE.replace(
        "        obs, r, done, info = env.step(action)",
        "        obs, r, done, info = env.step(action)\n"
        "        agent.update_z_goal(benefit_exposure=1.0, drive_level=0.9)")
    assert _lint_src(src) is None


def test_dzg_step_harness_is_exempt():
    """StepHarness pins the call as invariant 2 -- a user of it cannot carry the defect."""
    src = _DEFECTIVE.replace("from ree_core.agent import REEAgent",
                             "from _harness import StepHarness\n"
                             "from ree_core.agent import REEAgent")
    assert _lint_src(src) is None


def test_dzg_direct_goal_state_update_is_exempt():
    """The V3-EXQ-085h family constructs its own GoalState and drives it by hand."""
    src = _DEFECTIVE.replace(
        "        obs, r, done, info = env.step(action)",
        "        obs, r, done, info = env.step(action)\n"
        "        goal_state_resource.update(rf, benefit_exposure=1.0)")
    assert _lint_src(src) is None


def test_dzg_cue_pull_is_exempt():
    src = _DEFECTIVE.replace(
        "        obs, r, done, info = env.step(action)",
        "        obs, r, done, info = env.step(action)\n"
        "        agent.goal_state.cue_pull(z_obj, strength=0.3)")
    assert _lint_src(src) is None


def test_dzg_direct_z_goal_assignment_is_exempt():
    """The V3-EXQ-104/105/108/642 idiom: poke the attractor to stage a fixed goal."""
    src = _DEFECTIVE.replace(
        "    obs = env.reset()",
        "    agent.goal_state._z_goal = staged_goal\n    obs = env.reset()")
    assert _lint_src(src) is None


def test_dzg_unrelated_dot_update_does_not_exempt():
    """`.update()` discharges only on a receiver whose NAME mentions goal.

    Otherwise every `metrics.update(...)` / `d.update(...)` in the corpus would
    silence the gate.
    """
    src = _DEFECTIVE.replace(
        "        obs, r, done, info = env.step(action)",
        "        obs, r, done, info = env.step(action)\n"
        "        metrics.update({'r': r})")
    assert _lint_src(src) is not None


# ---- (3) the non-triggers: why ~500 correct scripts stay silent ----------------------

def test_dzg_hand_rolled_loop_without_the_knob_is_silent():
    """The majority case, and the reason the gate is knob-gated rather than shape-gated.

    A script that never enables z_goal loses NOTHING by omitting the call: goal_state is
    None and update_z_goal early-returns. Firing on the ~500 corpus scripts of this shape
    would make the gate pure noise.
    """
    src = _DEFECTIVE.replace(", z_goal_enabled=True, goal_weight=0.5", "")
    assert _lint_src(src) is None


def test_dzg_knob_set_false_is_silent():
    src = _DEFECTIVE.replace("z_goal_enabled=True", "z_goal_enabled=False")
    assert _lint_src(src) is None


def test_dzg_docstring_prose_is_not_a_setting():
    """Regression pin for the V3-EXQ-551 false positive.

    551's docstring DISCUSSES `z_goal_enabled=True` while the config sets it False. A
    name-scan fires on it; the AST keyword/attribute scan must not.
    """
    src = _DEFECTIVE.replace(
        '"""A driver that enables z_goal and never writes it."""',
        '"""Prior work compared z_goal_enabled=False against z_goal_enabled=True."""'
    ).replace("z_goal_enabled=True, goal_weight=0.5", "goal_weight=0.5")
    assert _lint_src(src) is None


def test_dzg_manifest_dict_echo_is_not_a_setting():
    """A `"z_goal_enabled": True` dict entry is a manifest echo, not the config."""
    src = _DEFECTIVE.replace(
        "z_goal_enabled=True, goal_weight=0.5", "goal_weight=0.5").replace(
        "    return agent",
        '    manifest = {"config": {"z_goal_enabled": True}}\n    return agent')
    assert _lint_src(src) is None


def test_dzg_non_driver_is_exempt():
    """A config-only unit probe has no stream to kill."""
    src = ('from ree_core.utils.config import REEConfig\n'
           'def build():\n'
           '    return REEConfig.from_dims(world_dim=32, z_goal_enabled=True)\n')
    assert _lint_src(src) is None


def test_dzg_driver_without_a_loop_is_exempt():
    src = _DEFECTIVE.replace("    for _ in range(200):\n", "    if True:\n")
    assert _lint_src(src) is None


# ---- (4) the helper discharge, and the hole it must NOT open ------------------------

def test_dzg_called_scaffold_helper_discharges():
    """The V3-EXQ-460/797/799 family warms up through the SD-054 scaffold.

    ScaffoldedSD054OnboardingScheduler calls update_z_goal every Stage-0/P1/P2 step, so
    those scripts DO seed z_goal. Their measurement loop then stops calling it and z_goal
    FREEZES at its post-warmup value -- a different (and milder) defect than the
    dead-zero stream this gate names. Discharged here so the warning text stays true.
    """
    src = _DEFECTIVE.replace(
        "from ree_core.agent import REEAgent",
        "from scaffolded_sd054_onboarding import ScaffoldedSD054OnboardingScheduler\n"
        "from ree_core.agent import REEAgent").replace(
        "    obs = env.reset()",
        "    ScaffoldedSD054OnboardingScheduler(cfg).run(agent, env)\n    obs = env.reset()")
    assert _lint_src(src) is None


def test_dzg_merely_importing_a_helper_does_not_discharge():
    """An unused import is not a z_goal write."""
    src = _DEFECTIVE.replace(
        "from ree_core.agent import REEAgent",
        "from scaffolded_sd054_onboarding import ScaffoldedSD054OnboardingScheduler\n"
        "from ree_core.agent import REEAgent")
    assert _lint_src(src) is not None


def test_dzg_arm_fingerprint_import_does_not_blanket_exempt():
    """THE HOLE THIS GATE ALMOST SHIPPED WITH.

    `_lib/arm_fingerprint.py` imports `_harness`, and the fingerprint is mandatory for
    multi-arm scripts, so nearly every modern script reaches `_harness` transitively. A
    discharge that followed the imported module's OWN imports therefore exempted most of
    the corpus -- including the V3-EXQ-830 shape the gate exists for. Measured at the
    time: 5 fires with the narrowing, 0 without it. `_uses_a_z_goal_driving_helper`
    checks `_writes_z_goal_directly` on the imported module and does not recurse.
    """
    src = _DEFECTIVE.replace(
        "from ree_core.agent import REEAgent",
        "from experiments._lib.arm_fingerprint import arm_cell\n"
        "from ree_core.agent import REEAgent").replace(
        "    obs = env.reset()",
        "    cell = arm_cell('ARM_0', 42)\n    obs = env.reset()")
    assert _lint_src(src) is not None
    fp = EXPERIMENTS_DIR / "_lib" / "arm_fingerprint.py"
    assert fp.is_file()
    assert not V._writes_z_goal_directly(ast.parse(fp.read_text(encoding="utf-8"))), (
        "arm_fingerprint.py must not itself write z_goal -- if it starts to, this "
        "discharge silently exempts most of the corpus.")


# ---- (5) the opt-out ----------------------------------------------------------------

def test_dzg_explicit_opt_out_is_honoured():
    src = _DEFECTIVE.replace(
        "from ree_core.agent import REEAgent",
        'DEAD_Z_GOAL_STREAM_EXEMPT = "goal-OFF parity arm; zero z_goal is the point"\n'
        "from ree_core.agent import REEAgent")
    assert _lint_src(src) is None


# ---- (6) the message carries what a triager needs -----------------------------------

def test_dzg_message_names_the_sole_writer_and_the_silent_consumers():
    out = _lint_src(_DEFECTIVE)
    assert "SOLE writer" in out
    for consumer in ("MECH-293", "MECH-288", "MECH-189", "SD-057", "MECH-295"):
        assert consumer in out, f"message should name {consumer} as a silenced consumer"


def test_dzg_message_declares_the_sd024_retrofit_side_effect():
    """A retrofit is NOT free and the message must say so.

    update_z_goal is also the SD-024 benefit-attractor producer, so adding it to an
    existing script populates benefit_rbf_field and un-zeroes the SD-025 curiosity bonus
    in HippocampalModule._curiosity_bonus. That is a behaviour change; a script retrofit
    this way is not comparable to the runs that preceded it.
    """
    out = _lint_src(_DEFECTIVE)
    assert "SD-024" in out and "SD-025" in out
    assert "behaviour change" in out


def test_dzg_message_names_the_kwargs_only_trap():
    """A POSITIONAL update_z_goal call collides with `latent` and raises every tick --
    that is how the EXQ-471/475/483/490/524 cohort failed."""
    out = _lint_src(_DEFECTIVE)
    assert "POSITIONAL" in out
    assert "StepHarness" in out


# ---- (7) the substrate facts the gate rests on --------------------------------------

def test_dzg_update_z_goal_is_the_sole_z_goal_writer_in_the_substrate():
    """Assert the premise against ree_core rather than trusting the docstring.

    Both GoalState mutators -- `.update(...)` and `.cue_pull(...)` -- must be called ONLY
    from inside `REEAgent.update_z_goal`. If a second writer ever appears, this gate's
    whole rationale changes and the assertion should fail loudly rather than the lint
    quietly becoming wrong.
    """
    agent_py = REPO_ROOT / "ree_core" / "agent.py"
    tree = ast.parse(agent_py.read_text(encoding="utf-8"))
    uzg_spans = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "update_z_goal":
            last = max((getattr(x, "lineno", node.lineno) for x in ast.walk(node)),
                       default=node.lineno)
            uzg_spans.append((node.lineno, last))
    assert uzg_spans, "REEAgent.update_z_goal not found in ree_core/agent.py"

    stray = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("update", "cue_pull")):
            recv = node.func.value
            name = (recv.id if isinstance(recv, ast.Name)
                    else recv.attr if isinstance(recv, ast.Attribute) else "")
            if name != "goal_state":
                continue
            if not any(lo <= node.lineno <= hi for lo, hi in uzg_spans):
                stray.append(node.lineno)
    assert not stray, (
        f"goal_state.update/cue_pull called OUTSIDE update_z_goal at line(s) {stray}. "
        "z_goal now has a second writer -- re-read dead_z_goal_stream_lint()'s premise.")


def test_dzg_agent_reset_does_not_reset_goal_state():
    """The fact that separates 'frozen goal' from 'dead goal'.

    `REEAgent.reset()` resets ~20 subsystems but NOT `goal_state`, and GoalState's decay
    lives inside `update()`. So a scaffold-warmed script that stops calling update_z_goal
    freezes z_goal at its post-warmup value rather than returning it to zero -- which is
    why that family is discharged by `_uses_a_z_goal_driving_helper` instead of being
    reported under a message that says "pinned at zero-init for the whole run".
    """
    agent_py = REPO_ROOT / "ree_core" / "agent.py"
    src = agent_py.read_text(encoding="utf-8")
    tree = ast.parse(src)
    reset_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "REEAgent":
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == "reset":
                    reset_fn = f
    assert reset_fn is not None, "REEAgent.reset not found"
    resets_goal = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "reset"
        and isinstance(n.func.value, ast.Attribute)
        and n.func.value.attr == "goal_state"
        for n in ast.walk(reset_fn))
    assert not resets_goal, (
        "REEAgent.reset() now resets goal_state. That flips the scaffold-warmed family "
        "from a FROZEN goal to a genuinely DEAD one, and they should no longer be "
        "discharged by _uses_a_z_goal_driving_helper -- re-read that helper's docstring.")


# ---- (8) real corpus witnesses ------------------------------------------------------

def test_dzg_real_626_is_the_detection_witness():
    """V3-EXQ-626's bespoke loop is the confirmed carrier; its successors are clean."""
    assert V.dead_z_goal_stream_lint(
        EXPERIMENTS_DIR / "v3_exq_626_goal_pipeline_developmental_window_diagnostic.py"
    ) is not None


def test_dzg_real_626a_and_626b_fixes_are_clean():
    for name in ("v3_exq_626a_goal_pipeline_developmental_window_diagnostic.py",
                 "v3_exq_626b_goal_pipeline_forced_seed_positive_control.py"):
        assert V.dead_z_goal_stream_lint(EXPERIMENTS_DIR / name) is None, name


def test_dzg_real_830_fix_is_clean():
    """V3-EXQ-830 wired the call kwargs-only after its readiness gate refused twice."""
    assert V.dead_z_goal_stream_lint(
        EXPERIMENTS_DIR / "v3_exq_830_mech321_scale_resolved_rollout_boundary.py") is None


def test_dzg_real_816_harness_family_is_clean():
    """816 omits the call too, but never enables z_goal -- correctly silent.

    This is the asymmetry the gate is built on: the SAME harness is fine for 816 and
    fatal for 830, and only the config knob tells them apart.
    """
    assert V.dead_z_goal_stream_lint(
        EXPERIMENTS_DIR
        / "v3_exq_816_mech321_policy_decomposition_discriminative.py") is None


# ---- (9) invariants: WARN-only, and the backlog does not silently grow ---------------

def test_dzg_is_warn_only_under_strict_and_paths():
    """INVARIANT: never blocks, like every other branch of this gate family."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(_DEFECTIVE)
        name = f.name
    try:
        r = _run("--checks", "dead_z_goal_stream", "--quiet", "--strict", "--paths", name)
        assert r.returncode == 0, r.stdout[-2000:]
        assert "DEAD Z_GOAL STREAM" in r.stdout
    finally:
        os.unlink(name)


# Pinned 2026-07-27 against the v3_exq_*.py corpus, at the commit that introduced this
# gate. This is a BACKLOG SIZE, not a target -- all 12 have run and are deliberately NOT
# retro-edited (a completed run's pre-registered emission is not rewritten). The pin
# exists so a NEW script carrying the defect shows up as a rise, and so a later widening
# or narrowing of the rule announces its own blast radius instead of drifting silently.
#
# The 12 reconcile exactly with the independent hand audit that commissioned the gate
# (13 carriers, minus `v3_mech124_zgoal_salience_longrun.py`, which is outside the
# `v3_exq_*` glob -- it never ran, and separately its z_goal readout was doubly dead
# because it read `latent.z_goal`, an attribute LatentState does not have).
#
# ADJUDICATION of the 12, against the landed manifests (2026-07-27; no manifest was
# edited). NINE of the 12 are already `non_contributory`; NONE of the 12 has a live
# conclusion resting on the dead stream:
#   - 626        FAIL / superseded  -- the defect was found and fixed in June (626a/626b).
#   - 618        PASS / non_contributory -- its C3/C4 probe the AIC and MECH-295 routers
#                by DIRECT call with synthetic per-axis inputs (`per_input_l1`), not
#                through the live `goal_state.is_active()`-gated bridge, so the criteria
#                never depended on the stream the docstring assumed was live.
#   - 701/a/b/c, 718/718a, 798  FAIL / non_contributory -- all seven set
#                `z_goal_enabled=True, goal_weight=0.5` and then read nothing goal-related;
#                the knobs were inert, identically in every arm.
#   - 615        PASS / supports / ARC-065 -- the ONLY landed `supports` carrier. Its
#                C1/C2/C3 are `selected_action_entropy` and `n_unique_classes` contrasts
#                between arms; none reads z_goal, and `goal_weight=0.5` was set
#                identically in all three arms, so the inert goal term is arm-SYMMETRIC
#                and cannot have produced the between-arm difference the criteria test.
#   - 263, 593   never landed a manifest at all (263's lineage continued as 263a/263b,
#                both of which DO call update_z_goal).
_PINNED_CORPUS_FIRE_COUNT = 12


def test_dzg_corpus_fire_rate_is_pinned():
    fired = [p for p in sorted(EXPERIMENTS_DIR.glob("v3_exq_*.py"))
             if V.dead_z_goal_stream_lint(p) is not None]
    assert len(fired) == _PINNED_CORPUS_FIRE_COUNT, (
        f"dead-z_goal-stream fire count moved: {len(fired)} vs pinned "
        f"{_PINNED_CORPUS_FIRE_COUNT}. If a NEW script is in this list, fix the script "
        f"(drive the loop through StepHarness) rather than re-pinning -- and note the "
        f"SD-024 side effect if you are retrofitting an existing one. If you deliberately "
        f"widened or narrowed the rule, re-pin and say so in the commit message. "
        f"Fired: {[p.name for p in fired]}")
