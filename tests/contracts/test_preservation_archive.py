"""Contract: the archive backends store records content-addressed, append-only, encrypted.

Companion to test_reconstruction_record.py / test_preservation_capture.py. Pins
ree_core/preservation/archive.py: the pluggable object-store backends behind the
tiered European storage plan (LocalArchive; S3Archive whose intended node 1 is
Hetzner Object Storage). Everything here runs with NO network, NO credentials, and
NO boto3 -- the S3 backend is exercised through an injected fake client, exactly
the seam that lets the real Hetzner call stay a thin, lazy-imported shell.

Pinned:
  1. content-addressing -- the object key embeds the record's sha256, so a changed
     record cannot overwrite the original (append-only by construction);
  2. round-trip -- store then load re-verifies integrity and yields the record back;
  3. client-side encryption -- with AesGcmEncryptor the stored bytes are ciphertext,
     the right key round-trips, the wrong key cannot decrypt;
  4. S3 backend logic -- key, append-only head-check, and metadata are correct
     against a fake client, without importing boto3.
"""

import io

import pytest
import torch

from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
from ree_core.preservation import (
    capture,
    reconstruct_config,
    to_json,
    LocalArchive,
    S3Archive,
    NoEncryption,
    AesGcmEncryptor,
    RecordExistsError,
    IntegrityError,
)
from experiments._lib.arm_fingerprint import seeded_construct
from experiments._lib.preservation import preserve_life

_DIMS = dict(body_obs_dim=12, world_obs_dim=250, action_dim=4)
_SEED = 909


def _cfg():
    return REEConfig.from_dims(**_DIMS)


def _record(record_id="arch_life_v3"):
    return capture(
        record_id=record_id,
        seed=_SEED,
        config=_cfg(),
        environment={"class": "…CausalGridWorld", "params": {"seed": _SEED, "size": 9}},
        substrate_hash="sha256:deadbeef",
        machine_class="test-machine-class",
        substrate_commit="abc123",
        architecture_epoch="ree_hybrid_guardrails_v1",
        understanding={"claims": ["GOV-PRESERVE-1"]},
    )


# --- a fake S3 client: enough of boto3's surface to drive S3Archive, no network --- #

class _FakeS3Error(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeS3Client:
    def __init__(self):
        self.store = {}          # (bucket, key) -> {"Body": bytes, "Metadata": {...}, ...}

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.store:
            raise _FakeS3Error("404")
        return {}

    def put_object(self, Bucket, Key, Body, **kw):
        self.store[(Bucket, Key)] = {"Body": Body, **kw}

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.store:
            raise _FakeS3Error("NoSuchKey")
        return {"Body": io.BytesIO(self.store[(Bucket, Key)]["Body"])}


# --------------------------------------------------------------------------- #
# 1-2. LocalArchive: content-address, round-trip, append-only                 #
# --------------------------------------------------------------------------- #

def test_local_archive_content_addressed_key(tmp_path):
    rec = _record()
    arch = LocalArchive(str(tmp_path))
    key = arch.put(rec)
    sha = rec.integrity.split(":", 1)[-1]
    assert sha in key and key.endswith(".json")  # content hash is in the key


def test_local_archive_roundtrip_reconstructs_bit_identical(tmp_path):
    rec = _record()
    arch = LocalArchive(str(tmp_path))
    key = arch.put(rec)
    loaded = arch.load(key)               # re-verifies integrity
    orig = seeded_construct(_SEED, lambda: REEAgent(_cfg())).state_dict()
    recon = seeded_construct(_SEED, lambda: REEAgent(reconstruct_config(loaded))).state_dict()
    assert not [k for k in orig if not torch.equal(orig[k], recon[k])]


def test_local_archive_is_append_only(tmp_path):
    rec = _record()
    arch = LocalArchive(str(tmp_path))
    arch.put(rec)
    with pytest.raises(RecordExistsError):
        arch.put(rec)


def test_preserve_life_via_archive_backend(tmp_path):
    key = preserve_life(
        archive=LocalArchive(str(tmp_path)),
        record_id="emit_arch_v3",
        seed=_SEED,
        config=_cfg(),
        environment={"class": "…", "params": {}},
    )
    assert key.endswith(".json")


# --------------------------------------------------------------------------- #
# 3. client-side encryption (real cryptography)                               #
# --------------------------------------------------------------------------- #

def test_aesgcm_encryptor_roundtrip_and_key_len():
    pytest.importorskip("cryptography")
    key = AesGcmEncryptor.generate_key()
    enc = AesGcmEncryptor(key)
    pt = b'{"hello":"world"}' * 50
    ct = enc.encrypt(pt)
    assert ct != pt and enc.decrypt(ct) == pt
    # wrong key cannot decrypt
    with pytest.raises(Exception):
        AesGcmEncryptor(AesGcmEncryptor.generate_key()).decrypt(ct)
    # bad key length rejected
    with pytest.raises(ValueError):
        AesGcmEncryptor(b"tooshort")


def test_encrypted_archive_stores_ciphertext_and_roundtrips(tmp_path):
    pytest.importorskip("cryptography")
    key_bytes = AesGcmEncryptor.generate_key()
    rec = _record()
    arch = LocalArchive(str(tmp_path), encryptor=AesGcmEncryptor(key_bytes))
    obj_key = arch.put(rec)
    assert obj_key.endswith(".json.enc")
    # on-disk bytes are NOT the plaintext JSON
    on_disk = arch._read(obj_key)
    assert b'"record_id"' not in on_disk
    assert on_disk != to_json(rec).encode("utf-8")
    # same key round-trips; wrong key cannot open the archive object
    assert arch.load(obj_key).integrity == rec.integrity
    wrong = LocalArchive(str(tmp_path), encryptor=AesGcmEncryptor(AesGcmEncryptor.generate_key()))
    with pytest.raises(Exception):
        wrong.load(obj_key)


# --------------------------------------------------------------------------- #
# 4. S3Archive against a fake client (no boto3, no network, no creds)         #
# --------------------------------------------------------------------------- #

def test_s3archive_roundtrip_key_and_metadata():
    fake = FakeS3Client()
    arch = S3Archive("ree-preserved", client=fake, prefix="preserved/")
    rec = _record("s3_life_v3")
    key = arch.put(rec)
    sha = rec.integrity.split(":", 1)[-1]
    assert key == "preserved/s3_life_v3/%s.json" % sha
    stored = fake.store[("ree-preserved", key)]
    assert stored["Metadata"]["record-id"] == "s3_life_v3"
    assert stored["Metadata"]["integrity"] == rec.integrity
    assert stored["Metadata"]["enc-alg"] == "none"
    assert arch.load(key).integrity == rec.integrity


def test_s3archive_append_only_via_head_check():
    fake = FakeS3Client()
    arch = S3Archive("ree-preserved", client=fake)
    rec = _record("s3_dup_v3")
    arch.put(rec)
    with pytest.raises(RecordExistsError):
        arch.put(rec)


def test_s3archive_encrypted_metadata_flag():
    pytest.importorskip("cryptography")
    fake = FakeS3Client()
    arch = S3Archive("ree-preserved", client=fake,
                     encryptor=AesGcmEncryptor(AesGcmEncryptor.generate_key()))
    rec = _record("s3_enc_v3")
    key = arch.put(rec)
    assert key.endswith(".json.enc")
    stored = fake.store[("ree-preserved", key)]
    assert stored["Metadata"]["enc-alg"] == "aes-256-gcm"
    assert stored["ContentType"] == "application/octet-stream"
    assert b'"record_id"' not in stored["Body"]   # ciphertext at rest
