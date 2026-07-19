"""Contracts for the shared untrained-world-encoder guard (experiments/_lib/zworld_encoder_guard.py).

CONTEXT. V3-EXQ-780's z_world arm ran on a frozen random projection: the P0 warmup left every
one of the 61 latent_stack tensors bit-identical. The guard was built inside
experiments/_lib/mech457_fanout.py, then LIFTED here (2026-07-19) after a fanout audit showed
the exposure reaches every _train_all_on_agent caller and is NOT gated on p1_episodes.

These contracts pin DETECTION, not a fix -- the training path is deliberately unchanged. The
load-bearing test is C3: it pins the ROOT CAUSE structurally (no optimizer parameter group
inside _train_all_on_agent intersects latent_stack), which is why p1_episodes cannot rescue
the encoder. That check is milliseconds; the behavioural equivalent would need a real warmup.

Diagnosis: REE_assembly/evidence/planning/zworld_bc_install_failure_V3-EXQ-780_2026-07-19.md
"""

from pathlib import Path

import pytest
import torch

import experiments._lib.mech457_fanout as fan
import experiments._lib.zworld_encoder_guard as guard
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734


def _all_on_agent():
    env = x734._make_env(42, x734._env_kwargs_for_rung(x734.DIFFICULTY_RUNGS[-1]))
    return x734._make_all_on_agent(env)


# --------------------------------------------------------------------------- C1
def test_c1_fanout_re_exports_the_shared_guard_identically():
    """The MECH-457 contracts import the guard through mech457_fanout. The lift must not
    fork it -- same class object, same functions, or C18* would be testing a different guard
    from the one the drivers use."""
    assert fan.ZWorldEncoderUntrainedError is guard.ZWorldEncoderUntrainedError
    assert fan.latent_stack_weight_delta is guard.latent_stack_weight_delta
    assert fan._latent_stack_snapshot is guard.latent_stack_snapshot
    assert fan._untrained_encoder_message is guard.untrained_encoder_message
    assert fan.WORLD_ENCODER_PREFIX == guard.WORLD_ENCODER_PREFIX
    assert fan.WORLD_PATH_PREFIXES == guard.WORLD_PATH_PREFIXES


# --------------------------------------------------------------------------- C2
def test_c2_guard_source_is_ascii():
    src = Path(guard.__file__).read_text(encoding="utf-8")
    non_ascii = [(i, ch) for i, ch in enumerate(src) if ord(ch) > 127]
    assert not non_ascii, f"non-ASCII in zworld_encoder_guard source: {non_ascii[:5]}"


# --------------------------------------------------------------------------- C3
def test_c3_no_train_all_on_optimizer_group_touches_latent_stack():
    """THE ROOT CAUSE, pinned structurally.

    _train_all_on_agent builds exactly three optimizers:
        P0/P1  e2_opt        <- agent.e2.parameters()
        P1     bias_opt      <- agent.lateral_pfc.bias_head_parameters()
        P1     ofc_deval_opt <- agent.ofc.devaluation_bias_head_parameters()

    None of them intersects latent_stack, so split_encoder.world_encoder receives no gradient
    in EITHER phase. This is why `p1_episodes > 0` does not rescue the encoder -- P1 trains
    two heads DOWNSTREAM of it. If a future change routes the encoder into one of these
    groups this test flips, which is the signal to revisit the guard's default.
    """
    agent = _all_on_agent()
    stack_ids = {id(p) for _, p in agent.latent_stack.named_parameters()}
    assert len(stack_ids) > 0

    groups = {
        "e2": list(agent.e2.parameters()),
        "lateral_pfc.bias_head": list(agent.lateral_pfc.bias_head_parameters()),
        "ofc.devaluation_bias_head": list(agent.ofc.devaluation_bias_head_parameters()),
    }
    for label, params in groups.items():
        assert params, f"{label} parameter group is empty -- the check would be vacuous"
        overlap = [p for p in params if id(p) in stack_ids]
        assert not overlap, (
            f"{label} now overlaps latent_stack ({len(overlap)} tensor(s)). The "
            "untrained-encoder guard's premise has changed -- re-audit before relaxing it."
        )


# --------------------------------------------------------------------------- C4
def test_c4_world_encoder_is_inside_latent_stack_and_outside_every_group():
    """Complement to C3: the encoder really is a latent_stack member (so the snapshot covers
    it) and really is absent from every optimizer group (so it cannot move)."""
    agent = _all_on_agent()
    enc = [n for n, _ in agent.latent_stack.named_parameters()
           if n.startswith(guard.WORLD_ENCODER_PREFIX)]
    assert enc, "no split_encoder.world_encoder tensors -- guard would be vacuous"

    enc_ids = {id(p) for n, p in agent.latent_stack.named_parameters()
               if n.startswith(guard.WORLD_ENCODER_PREFIX)}
    covered = [p for p in (list(agent.e2.parameters())
                           + list(agent.lateral_pfc.bias_head_parameters())
                           + list(agent.ofc.devaluation_bias_head_parameters()))
               if id(p) in enc_ids]
    assert not covered, f"world encoder is now in an optimizer group ({len(covered)} tensor(s))"


# --------------------------------------------------------------------------- C5
def test_c5_unchanged_stack_raises_by_default():
    """A no-op 'warmup' is exactly the 780 signature: strict default must raise."""
    agent = _all_on_agent()
    with pytest.raises(guard.ZWorldEncoderUntrainedError, match="world_encoder"):
        guard.guarded_warmup(agent, lambda: None, p0=5, context="unit")


# --------------------------------------------------------------------------- C6
def test_c6_escape_hatch_warns_and_records_instead_of_raising(capsys):
    """strict=False is the explicit, auditable opt-out for a deliberate frozen-encoder
    ablation -- it must WARN unmissably and still hand back the record."""
    agent = _all_on_agent()
    rec = guard.guarded_warmup(agent, lambda: None, p0=5, strict=False, context="unit")
    assert rec["guard_checked"] is True
    assert rec["zworld_encoder_trained"] is False
    assert rec["n_world_encoder_changed"] == 0
    assert rec["n_world_encoder_tensors"] > 0
    assert rec["world_encoder_max_abs_delta"] == 0.0
    assert rec["n_latent_stack_changed"] == 0
    assert "GUARD-WARNING" in capsys.readouterr().out


# --------------------------------------------------------------------------- C7
def test_c7_no_warmup_requested_is_not_a_guard_failure():
    """p0 <= 0 requests no warmup at all -- a caller's explicit choice, not the silent
    failure being detected."""
    agent = _all_on_agent()
    rec = guard.guarded_warmup(agent, lambda: None, p0=0)   # strict default: no raise
    assert rec["guard_checked"] is False
    assert rec["p0_episodes"] == 0


# --------------------------------------------------------------------------- C8
def test_c8_guard_passes_when_the_encoder_actually_moves():
    """Positive control: the delta check is not vacuously failing."""
    agent = _all_on_agent()

    def _one_step():
        enc = agent.latent_stack.split_encoder.world_encoder
        sum(p.sum() for p in enc.parameters()).backward()
        with torch.no_grad():
            for p in enc.parameters():
                p -= 1e-3 * p.grad

    rec = guard.guarded_warmup(agent, _one_step, p0=5, context="unit")
    assert rec["zworld_encoder_trained"] is True
    assert rec["n_world_encoder_changed"] == rec["n_world_encoder_tensors"]
    assert rec["world_encoder_max_abs_delta"] > 0.0


# --------------------------------------------------------------------------- C9
def test_c9_message_names_the_context_and_the_diagnosis_doc():
    """A fanout of many cells must say WHICH one tripped, and point at the diagnosis."""
    agent = _all_on_agent()
    before = guard.latent_stack_snapshot(agent)
    report = guard.latent_stack_weight_delta(agent, before)
    msg = guard.untrained_encoder_message(report, 200, context="x742:ac_on/seed42")
    assert "x742:ac_on/seed42" in msg
    assert guard.DIAGNOSIS_DOC in msg
    assert "FROZEN RANDOM PROJECTION" in msg
    assert all(ord(c) < 128 for c in msg), "guard message must be ASCII-safe (repo rule)"
