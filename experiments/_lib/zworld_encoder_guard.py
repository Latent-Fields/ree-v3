"""Untrained-world-encoder detection for any `_train_all_on_agent`-style warmup.

WHY THIS MODULE EXISTS (lifted from experiments/_lib/mech457_fanout.py, 2026-07-19).
The guard was built for the MECH-457 fanout path when V3-EXQ-780's z_world arm was found to
have run on a frozen random projection. Auditing the other `_train_all_on_agent` callers
showed the exposure is NOT specific to that path and NOT gated on `p1_episodes`:

    NONE of the three optimizers inside x734._train_all_on_agent touches ANY latent_stack
    parameter. Measured on the D3_hazard_free rung, all-ON agent:
        e2.parameters()                        18 params, 0 overlap with latent_stack (0/61)
        lateral_pfc.bias_head_parameters()      4 params, 0 overlap
        ofc.devaluation_bias_head_parameters()  4 params, 0 overlap

P0 trains the SD-056 e2 forward model; P1 trains the lateral-PFC bias head and the OFC
devaluation head. `split_encoder.world_encoder` is in none of those parameter groups, so it
receives no gradient in EITHER phase. `p1_episodes > 0` therefore does not rescue the encoder
-- it only adds training to two heads that are downstream of it. The P0 loop additionally
buffers `latent.z_world.detach()`, which is the mechanism the V3-EXQ-780 diagnosis identified.

CONSEQUENCE: every arm built on this warmup whose scientific premise requires a LEARNED
world representation is reading a frozen random projection at initialisation, silently.

SCOPE: DETECTION ONLY. This module does not attempt to make the encoder train. That fix has a
second part -- the SD-018 head supervises max(resource_field_view), a scalar magnitude that
decodes the demonstrator's action at chance (0.2552 vs a 0.2534 majority-class floor) -- and
the naive "just enable the prescribed P0 training" is already REFUTED in-corpus (SD-070:
prescribed P0 at dim=128 collapses z_world, participation ratio 9.21 -> 1.06). The fix is
downstream of the V3-EXQ-783 adjudication and belongs to governance.

References:
  REE_assembly/evidence/planning/zworld_bc_install_failure_V3-EXQ-780_2026-07-19.md
  REE_assembly/evidence/experiments/zworld_near_static_characterisation_2026-07-18.md
  REE_assembly/evidence/planning/sd009_event_contrastive_channel_mismatch_2026-07-18.md

METHOD (unchanged from the landed MECH-457 guard, so every consumer agrees with the diagnosis
by construction): snapshot every latent_stack named parameter before the warmup, compare
after, and count a tensor as changed only when max|delta| > 0. Bit-identity is the signature
being detected, so the comparison is exact rather than tolerance-based; realised magnitudes
are reported alongside so a "changed but negligibly" case is visible rather than silently
passing.

ASCII-only in all runtime strings (repo rule).
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Dict, List, Tuple

import torch

WORLD_ENCODER_PREFIX = "split_encoder.world_encoder."
WORLD_PATH_PREFIXES: Tuple[str, ...] = (
    "split_encoder.world_encoder.",              # the encoder itself -- sets z_world content
    "split_encoder.event_classifier",            # SD-009 head reading z_world
    "split_encoder.resource_proximity_head.",    # SD-018 head reading z_world
)

DIAGNOSIS_DOC = (
    "REE_assembly/evidence/planning/"
    "zworld_bc_install_failure_V3-EXQ-780_2026-07-19.md"
)


class ZWorldEncoderUntrainedError(RuntimeError):
    """Raised when a warmup leaves the world encoder bit-identical to its random
    initialisation, i.e. the arm would run on a frozen random projection."""


def latent_stack_snapshot(agent: Any) -> Dict[str, torch.Tensor]:
    """Detached clones of every latent_stack parameter, keyed by name. Empty dict when the
    agent has no latent_stack (nothing to guard)."""
    stack = getattr(agent, "latent_stack", None)
    if stack is None:
        return {}
    return {name: p.detach().clone() for name, p in stack.named_parameters()}


def latent_stack_weight_delta(agent: Any, before: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Compare the agent's current latent_stack parameters against a `latent_stack_snapshot`.

    Returns an audit record (JSON-safe, manifest-embeddable). `zworld_encoder_trained` is the
    load-bearing bit: True only when at least one split_encoder.world_encoder tensor moved."""
    stack = getattr(agent, "latent_stack", None)
    after = {} if stack is None else dict(stack.named_parameters())
    n_total = 0
    n_changed = 0
    world_path: List[str] = []
    world_path_changed: List[str] = []
    enc_names: List[str] = []
    enc_changed: List[str] = []
    enc_unchanged: List[str] = []
    enc_max_delta = 0.0
    for name, prev in before.items():
        cur = after.get(name)
        if cur is None:
            continue
        n_total += 1
        delta = float((cur.detach() - prev).abs().max().item())
        moved = delta > 0.0
        if moved:
            n_changed += 1
        if name.startswith(WORLD_PATH_PREFIXES):
            world_path.append(name)
            if moved:
                world_path_changed.append(name)
        if name.startswith(WORLD_ENCODER_PREFIX):
            enc_names.append(name)
            enc_max_delta = max(enc_max_delta, delta)
            (enc_changed if moved else enc_unchanged).append(name)
    return {
        "n_latent_stack_tensors": n_total,
        "n_latent_stack_changed": n_changed,
        "n_world_path_tensors": len(world_path),
        "n_world_path_changed": len(world_path_changed),
        "n_world_encoder_tensors": len(enc_names),
        "n_world_encoder_changed": len(enc_changed),
        "world_encoder_max_abs_delta": enc_max_delta,
        "unchanged_world_encoder_tensors": sorted(enc_unchanged),
        "zworld_encoder_trained": bool(enc_changed),
    }


def untrained_encoder_message(
    report: Dict[str, Any], p0: int, *, context: str = "", escape_hint: str = "",
) -> str:
    """The unmissable failure message. `context` names the calling arm/experiment so a fanout
    of many cells says WHICH one tripped; `escape_hint` names the caller's own opt-out flag."""
    frozen = ", ".join(report.get("unchanged_world_encoder_tensors") or []) or "(none enumerated)"
    where = f" [{context}]" if context else ""
    hatch = escape_hint or (
        "pass require_trained_encoder=False so the choice is explicit and auditable in the "
        "manifest"
    )
    return (
        f"z_world UNTRAINED-ENCODER GUARD TRIPPED{where}: the P0 warmup changed 0 of "
        f"{report.get('n_world_encoder_tensors', 0)} split_encoder.world_encoder tensors "
        f"across {int(p0)} episode(s) -- max|delta| = "
        f"{report.get('world_encoder_max_abs_delta', 0.0):.3e} (bit-identical). "
        f"latent_stack: {report.get('n_latent_stack_changed', 0)}/"
        f"{report.get('n_latent_stack_tensors', 0)} tensors changed. "
        f"Frozen: {frozen}. "
        "This arm's z_world is a FROZEN RANDOM PROJECTION at initialisation, NOT a "
        "prediction-trained encoder, so any result it produces is uninterpretable as "
        "evidence about z_world. Known cause: no optimizer in _train_all_on_agent covers a "
        "latent_stack parameter (P0 trains e2; P1 trains the lPFC bias + OFC devaluation "
        "heads), and the P0 loop buffers latent.z_world.detach(), so no gradient reaches the "
        f"world encoder. See {DIAGNOSIS_DOC}. "
        f"If a frozen random projection is the DELIBERATE condition under test, {hatch}."
    )


def assert_world_encoder_trained(
    agent: Any,
    before: Dict[str, torch.Tensor],
    *,
    p0: int,
    strict: bool = True,
    context: str = "",
    escape_hint: str = "",
) -> Dict[str, Any]:
    """Verify that a warmup moved the world encoder, given a pre-warmup snapshot.

    With `strict=True` (the default) a zero-change result RAISES
    ZWorldEncoderUntrainedError rather than silently handing back a frozen random projection.
    `strict=False` downgrades it to an unmissable warning -- for deliberate frozen-encoder
    ablations only.

    Returns the audit record (plus `p0_episodes` and `guard_checked`) so a driver can embed it
    in the run manifest. `p0 <= 0` requests no warmup at all, so the guard records
    `guard_checked: False` and never fires -- an unwarmed rep is a caller's explicit choice,
    not the silent failure being detected."""
    report = latent_stack_weight_delta(agent, before)
    report["p0_episodes"] = int(p0)
    report["guard_checked"] = bool(int(p0) > 0 and before)
    if not report["guard_checked"] or report["zworld_encoder_trained"]:
        return report
    message = untrained_encoder_message(report, p0, context=context, escape_hint=escape_hint)
    if strict:
        raise ZWorldEncoderUntrainedError(message)
    print(f"[GUARD-WARNING] {message}", file=sys.stderr, flush=True)
    print(f"[GUARD-WARNING] {message}", flush=True)
    return report


def guarded_warmup(
    agent: Any,
    warmup: Callable[[], Any],
    *,
    p0: int,
    strict: bool = True,
    context: str = "",
    escape_hint: str = "",
) -> Dict[str, Any]:
    """Snapshot -> run `warmup()` -> assert the world encoder moved. The one-call form for a
    driver that already has its own warmup closure; returns the manifest-embeddable record."""
    before = latent_stack_snapshot(agent)
    warmup()
    return assert_world_encoder_trained(
        agent, before, p0=p0, strict=strict, context=context, escape_hint=escape_hint,
    )


# Back-compat alias: the MECH-457 contracts (tests/contracts/test_mech457_bootstrap_explorer.py
# C18d) call the private-named snapshot helper through mech457_fanout.
_latent_stack_snapshot = latent_stack_snapshot

__all__ = [
    "DIAGNOSIS_DOC",
    "WORLD_ENCODER_PREFIX",
    "WORLD_PATH_PREFIXES",
    "ZWorldEncoderUntrainedError",
    "assert_world_encoder_trained",
    "guarded_warmup",
    "latent_stack_snapshot",
    "latent_stack_weight_delta",
    "untrained_encoder_message",
]
