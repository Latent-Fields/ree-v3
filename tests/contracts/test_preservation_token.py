"""Contract: the physical-token exporter produces a verifiable, reconstructable token.

Pins ree_core/preservation/token.py -- the artifacts you put on durable physical
media (engraved key, QR, gzipped record for an M-DISC / Piql deposit). Runs with no
QR library installed: the zero-dep artifacts are always produced, and the QR path is
exercised through an injected fake backend (the same seam that lets the real segno
call stay optional).

Pinned:
  1. the key line round-trips and carries the record's sha256 (authenticator);
  2. the zero-dep artifacts (key/record/manifest/README) are always written;
  3. the gzipped record re-inflates, re-verifies, and reconstructs a BIT-IDENTICAL
     birth agent -- the physical token really is sufficient;
  4. the manifest's hashes match (record integrity + the deposited-blob sha256);
  5. QR is written iff a backend is present (injected fake proves the wiring).
"""

import gzip
import hashlib
import json
import os

import pytest
import torch

from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
from ree_core.preservation import (
    capture,
    reconstruct_config,
    from_json,
    verify_integrity,
    export_token,
    record_key_line,
    parse_key_line,
    KEY_SCHEME,
)
from experiments._lib.arm_fingerprint import seeded_construct

_DIMS = dict(body_obs_dim=12, world_obs_dim=250, action_dim=4)
_SEED = 31


def _cfg():
    return REEConfig.from_dims(**_DIMS)


def _record(rid="tok_life_v3"):
    return capture(
        record_id=rid,
        seed=_SEED,
        config=_cfg(),
        environment={"class": "…CausalGridWorld", "params": {"seed": _SEED, "size": 9}},
        substrate_hash="sha256:abc",
        machine_class="darwin-arm64/torch-2.10.0",
        substrate_commit="6ca09e5",
        architecture_epoch="ree_hybrid_guardrails_v1",
        understanding={"claims": ["GOV-PRESERVE-1"]},
    )


class _FakeQR:
    """Records what it was asked to encode; writes stub artifacts (no QR lib needed)."""

    def __init__(self):
        self.encoded = []

    def write(self, data, svg_path, png_path):
        self.encoded.append(data)
        with open(svg_path, "w", encoding="utf-8") as fh:
            fh.write("<svg/>")
        with open(png_path, "wb") as fh:
            fh.write(b"\x89PNG")


def test_key_line_roundtrips_and_carries_the_hash():
    rec = _record()
    line = record_key_line(rec)
    parsed = parse_key_line(line)
    assert parsed["scheme"] == KEY_SCHEME
    assert parsed["id"] == rec.record_id
    assert parsed["seed"] == str(rec.seed)
    assert parsed["commit"] == rec.provenance["substrate_commit"]
    assert parsed["machine"] == rec.provenance["machine_class"]
    assert parsed["sha256"] == rec.integrity.split(":", 1)[-1]


def test_parse_rejects_non_scheme():
    with pytest.raises(ValueError):
        parse_key_line("not a key line")


def test_zero_dep_artifacts_always_written(tmp_path):
    rec = _record()
    res = export_token(rec, str(tmp_path), make_qr=True, qr_backend=None)  # no backend in env
    for suffix in (".key.txt", ".record.json.gz", ".manifest.json", ".README.txt"):
        assert os.path.exists(os.path.join(str(tmp_path), rec.record_id + suffix))
    # no QR here, and the reason is surfaced
    assert res["qr_written"] is False
    assert res["qr_note"] and "segno" in res["qr_note"]


def test_gz_record_reconstructs_bit_identical(tmp_path):
    rec = _record()
    export_token(rec, str(tmp_path))
    gz = os.path.join(str(tmp_path), rec.record_id + ".record.json.gz")
    recovered = from_json(gzip.decompress(open(gz, "rb").read()).decode("utf-8"))
    assert verify_integrity(recovered)
    orig = seeded_construct(_SEED, lambda: REEAgent(_cfg())).state_dict()
    recon = seeded_construct(_SEED, lambda: REEAgent(reconstruct_config(recovered))).state_dict()
    assert not [k for k in orig if not torch.equal(orig[k], recon[k])]


def test_manifest_hashes_match(tmp_path):
    rec = _record()
    export_token(rec, str(tmp_path))
    man = json.load(open(os.path.join(str(tmp_path), rec.record_id + ".manifest.json")))
    assert man["record_integrity"] == rec.integrity
    gz = os.path.join(str(tmp_path), man["record_file"])
    assert man["record_file_sha256"] == "sha256:" + hashlib.sha256(open(gz, "rb").read()).hexdigest()


def test_qr_written_with_injected_backend(tmp_path):
    rec = _record()
    fake = _FakeQR()
    res = export_token(rec, str(tmp_path), qr_backend=fake)
    assert res["qr_written"] is True
    assert os.path.exists(os.path.join(str(tmp_path), rec.record_id + ".key.svg"))
    assert os.path.exists(os.path.join(str(tmp_path), rec.record_id + ".key.png"))
    # the QR encodes exactly the key line
    assert fake.encoded == [record_key_line(rec)]


def test_no_gzip_option_writes_plain_json(tmp_path):
    rec = _record()
    export_token(rec, str(tmp_path), gzip_record=False)
    assert os.path.exists(os.path.join(str(tmp_path), rec.record_id + ".record.json"))
    assert not os.path.exists(os.path.join(str(tmp_path), rec.record_id + ".record.json.gz"))
