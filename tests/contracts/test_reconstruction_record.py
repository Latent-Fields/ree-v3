"""Contract: a ReconstructionRecord preserves the possibility of future reconstruction.

Thought: REE_assembly/docs/thoughts/
2026-08-14_preserving_the_possibility_of_future_reconstruction.md
Design:  REE_assembly/evidence/planning/preservation_snapshot_plan.md

A REE life is a pure function of (REEConfig, seed) on pinned code, so preserving
those inputs immutably keeps the whole organism reconstructable by re-deriving it
from birth. This contract pins the properties that make that guarantee real rather
than hopeful:

  1. FAITHFUL CONFIG: REEConfig round-trips through the record's plain-JSON form
     EXACTLY -- including tuple-typed fields and bare-`list` fields holding nested
     dataclasses (e.g. EventSegmenterConfig.scales), which annotation-guided
     rebuild silently got wrong. A config that did not round-trip would mean a
     reconstructed organism that is quietly not the one we preserved.
  2. SUFFICIENCY: the record ALONE (round-tripped through JSON text, with no access
     to the original live objects) rebuilds a BIT-IDENTICAL birth agent.
  3. INTEGRITY: capture() seals a sha256; verify/load detect any later mutation.
  4. IMMUTABILITY: write is append-only -- it refuses to overwrite a record.
  5. SAFETY: reconstruction only imports dataclasses from ree_core, never an
     arbitrary dotted path named in a file.

FIDELITY BOUNDARY (why this asserts CONSTRUCTION, not multi-step replay): weight
init from a seed uses randn/normal, which is bit-identical across machine classes.
Multi-step replay routes through torch.multinomial in the E3 selector, which
diverges across machine classes (see the memory
`reference-cross-machine-class-contract-divergence`). So equivalence is asserted
on birth construction, which is machine-robust; the record stamps machine_class so
a future reconstructor knows the boundary for replay.
"""

import pytest
import torch

from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
from ree_core.preservation import (
    capture,
    config_to_dict,
    config_from_dict,
    reconstruct_config,
    to_json,
    from_json,
    write_record,
    load_record,
    verify_integrity,
    compute_integrity,
    ReconstructionRecord,
    IntegrityError,
    RecordExistsError,
)
from ree_core.preservation import reconstruction_record as rr
from experiments._lib.arm_fingerprint import seeded_construct

_DIMS = dict(body_obs_dim=12, world_obs_dim=250, action_dim=4)


def _plain_cfg():
    return REEConfig.from_dims(**_DIMS)


def _rich_cfg():
    # Exercises the fields annotation-guided rebuild got wrong: the segmenter's
    # bare-`list` of nested dataclasses, plus harm/sleep sub-configs.
    return REEConfig.from_dims(
        use_event_classifier=True,
        use_harm_stream=True,
        use_affective_harm_stream=True,
        sws_enabled=True,
        rem_enabled=True,
        use_sleep_aggregation_cluster=True,
        sleep_loop_episodes_K=1,
        **_DIMS,
    )


def _sealed(cfg, **over):
    kw = dict(
        record_id="test_life_v3",
        seed=777,
        config=cfg,
        environment={
            "class": "ree_core.environment.causal_grid_world.CausalGridWorld",
            "params": {"seed": 777, "size": 9},
        },
        substrate_hash="sha256:deadbeef",
        machine_class="test-machine-class",
        substrate_commit="abc123",
        architecture_epoch="ree_hybrid_guardrails_v1",
        understanding={"claims": ["ARC-000"], "note": "contract"},
    )
    kw.update(over)
    return capture(**kw)


# --------------------------------------------------------------------------- #
# 1. faithful config round-trip                                               #
# --------------------------------------------------------------------------- #

def test_plain_config_roundtrips_exactly():
    cfg = _plain_cfg()
    assert config_from_dict(config_to_dict(cfg)) == cfg


def test_feature_rich_config_roundtrips_exactly():
    cfg = _rich_cfg()
    rt = config_from_dict(config_to_dict(cfg))
    assert rt == cfg


def test_segmenter_scales_reconstruct_as_dataclasses_not_dicts():
    # The regression that motivated tagged serialization: a bare-`list` of nested
    # dataclasses must come back as dataclasses, or downstream `scale.streams`
    # attribute access would break for any config that uses the segmenter.
    cfg = _rich_cfg()
    rt = config_from_dict(config_to_dict(cfg))
    scales = rt.hippocampal.event_segmenter.scales
    assert scales, "expected the feature-rich config to define segmenter scales"
    for s in scales:
        assert type(s).__name__ == "EventSegmenterScaleConfig"
        assert isinstance(s.streams, tuple)  # tuple-ness restored, not left a list


# --------------------------------------------------------------------------- #
# 2. sufficiency: the record alone rebuilds a bit-identical birth agent        #
# --------------------------------------------------------------------------- #

def _birth_state_dict(cfg):
    return seeded_construct(777, lambda: REEAgent(cfg)).state_dict()


def test_reconstructed_config_builds_bit_identical_birth_agent():
    cfg = _plain_cfg()
    rec = _sealed(cfg)
    orig = _birth_state_dict(_plain_cfg())
    recon = _birth_state_dict(reconstruct_config(rec))
    assert orig.keys() == recon.keys()
    mism = [k for k in orig if not torch.equal(orig[k], recon[k])]
    assert not mism, "reconstructed birth agent diverged in %d tensors" % len(mism)


def test_record_is_self_sufficient_after_json_text_roundtrip():
    # Reconstruct from JSON TEXT only -- no original live objects in scope.
    cfg = _rich_cfg()
    text = to_json(_sealed(cfg))
    rec = from_json(text)
    recon = _birth_state_dict(reconstruct_config(rec))
    orig = _birth_state_dict(_rich_cfg())
    assert orig.keys() == recon.keys()
    assert not [k for k in orig if not torch.equal(orig[k], recon[k])]


# --------------------------------------------------------------------------- #
# 3. integrity                                                                 #
# --------------------------------------------------------------------------- #

def test_capture_seals_a_verifiable_integrity_hash():
    rec = _sealed(_plain_cfg())
    assert rec.integrity and rec.integrity.startswith("sha256:")
    assert verify_integrity(rec)


def test_mutation_breaks_integrity():
    rec = _sealed(_plain_cfg())
    rec.seed = 778
    assert not verify_integrity(rec)
    assert rec.integrity != compute_integrity(rec)


def test_json_roundtrip_preserves_integrity():
    rec = _sealed(_plain_cfg())
    rec2 = from_json(to_json(rec))
    assert rec2.integrity == rec.integrity
    assert verify_integrity(rec2)


# --------------------------------------------------------------------------- #
# 4. immutable, integrity-checked storage                                     #
# --------------------------------------------------------------------------- #

def test_write_then_load_roundtrip(tmp_path):
    rec = _sealed(_plain_cfg())
    path = write_record(rec, str(tmp_path))
    loaded = load_record(path)
    assert loaded.integrity == rec.integrity
    assert loaded.seed == rec.seed


def test_write_is_append_only(tmp_path):
    rec = _sealed(_plain_cfg())
    write_record(rec, str(tmp_path))
    with pytest.raises(RecordExistsError):
        write_record(rec, str(tmp_path))


def test_load_detects_tampering(tmp_path):
    rec = _sealed(_plain_cfg())
    path = write_record(rec, str(tmp_path))
    with open(path, "r", encoding="utf-8") as fh:
        txt = fh.read()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(txt.replace('"seed": 777', '"seed": 778'))
    with pytest.raises(IntegrityError):
        load_record(path)


def test_write_refuses_unsealed_record(tmp_path):
    # A hand-built record with no integrity hash must not be writable.
    rec = ReconstructionRecord(
        record_id="unsealed_v3",
        seed=1,
        config=config_to_dict(_plain_cfg()),
        environment={},
        provenance={},
    )
    assert rec.integrity is None
    with pytest.raises(IntegrityError):
        write_record(rec, str(tmp_path))


# --------------------------------------------------------------------------- #
# 5. safety: no arbitrary imports from a record                               #
# --------------------------------------------------------------------------- #

def test_deserializer_refuses_non_ree_core_class():
    d = config_to_dict(_plain_cfg())
    d[rr._DC_TAG] = "collections.OrderedDict"  # not ree_core, not a dataclass
    with pytest.raises(ValueError):
        config_from_dict(d)
