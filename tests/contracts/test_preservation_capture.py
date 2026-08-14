"""Contract: the experiment-side emitter writes a reconstructable, provenanced record.

Companion to test_reconstruction_record.py. That one pins the record primitive;
this one pins experiments/_lib/preservation.py -- the glue that fills provenance
from manifest_core/arm_fingerprint and writes a record at the point a life ends.

Pinned:
  1. preserve_life writes an immutable record that reconstructs a BIT-IDENTICAL
     birth agent from disk alone.
  2. provenance is populated: non-empty substrate_hash and machine_class,
     architecture_epoch stamped. substrate_commit is None-tolerant on purpose --
     manifest_core.substrate_commit shells to git and returns None in a git-less
     checkout (the remote-pytest staged tree excludes .git), so asserting a sha
     would be a phantom failure there.
  3. emission is append-only: a second preserve_life with the same record_id
     refuses (records are immutable).
"""

import pytest
import torch

from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
from ree_core.preservation import load_record, reconstruct_config, RecordExistsError
from experiments._lib.arm_fingerprint import seeded_construct
from experiments._lib.preservation import (
    capture_life,
    preserve_life,
    DEFAULT_ARCHITECTURE_EPOCH,
)

_DIMS = dict(body_obs_dim=12, world_obs_dim=250, action_dim=4)
_SEED = 4242


def _cfg():
    return REEConfig.from_dims(**_DIMS)


def _env_spec():
    return {
        "class": "ree_core.environment.causal_grid_world.CausalGridWorld",
        "params": {"seed": _SEED, "size": 9},
    }


def _birth(cfg):
    return seeded_construct(_SEED, lambda: REEAgent(cfg)).state_dict()


def test_preserve_life_writes_reconstructable_record(tmp_path):
    path = preserve_life(
        archive_dir=str(tmp_path),
        record_id="emit_life_v3",
        seed=_SEED,
        config=_cfg(),
        environment=_env_spec(),
        understanding={"claims": ["GOV-PRESERVE-1"], "note": "contract"},
        reason_for_ending="end of contract run",
        lifetime={"age_steps": 0, "phase": 1},
    )
    rec = load_record(path)  # re-verifies integrity
    orig = _birth(_cfg())
    recon = _birth(reconstruct_config(rec))
    assert orig.keys() == recon.keys()
    assert not [k for k in orig if not torch.equal(orig[k], recon[k])]


def test_provenance_is_populated():
    rec = capture_life(
        record_id="prov_life_v3",
        seed=_SEED,
        config=_cfg(),
        environment=_env_spec(),
    )
    prov = rec.provenance
    assert isinstance(prov["substrate_hash"], str) and prov["substrate_hash"]
    assert isinstance(prov["machine_class"], str) and prov["machine_class"]
    assert prov["architecture_epoch"] == DEFAULT_ARCHITECTURE_EPOCH
    # None-tolerant: git-less staged tree yields no commit sha.
    assert prov["substrate_commit"] is None or isinstance(prov["substrate_commit"], str)


def test_emission_is_append_only(tmp_path):
    kw = dict(
        record_id="dup_life_v3",
        seed=_SEED,
        config=_cfg(),
        environment=_env_spec(),
    )
    preserve_life(archive_dir=str(tmp_path), **kw)
    with pytest.raises(RecordExistsError):
        preserve_life(archive_dir=str(tmp_path), **kw)
