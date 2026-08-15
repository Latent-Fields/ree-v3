"""Contract: multi-backend deposit (the >1-copy rule) and env-driven construction.

Pins the operational layer over the archive backends: MultiArchive fans a record
out to every backend (so no single provider holds the only copy) and is idempotent
(safe to re-run); s3_archive_from_env builds a provider backend from a group of env
vars with INDEPENDENT credentials. No boto3, no network, no real credentials -- the
S3 client is injected via a fake factory.
"""

import io

import pytest

from ree_core.utils.config import REEConfig
from ree_core.preservation import (
    capture,
    MultiArchive,
    LocalArchive,
    S3Archive,
    s3_archive_from_env,
    NoEncryption,
)

_DIMS = dict(body_obs_dim=12, world_obs_dim=250, action_dim=4)


def _record(rid="multi_life_v3"):
    return capture(
        record_id=rid,
        seed=5,
        config=REEConfig.from_dims(**_DIMS),
        environment={"class": "…", "params": {}},
        substrate_hash="sha256:abc",
        machine_class="test-machine-class",
        substrate_commit="abc123",
        architecture_epoch="ree_hybrid_guardrails_v1",
    )


class _FakeS3Error(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeS3Client:
    def __init__(self):
        self.store = {}

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
# MultiArchive fan-out                                                        #
# --------------------------------------------------------------------------- #

def test_multiarchive_writes_to_every_backend(tmp_path):
    rec = _record()
    a = LocalArchive(str(tmp_path / "a"))
    b = S3Archive("bucket-b", client=FakeS3Client())
    multi = MultiArchive([a, b], names=["local", "s3"])
    res = multi.put(rec)
    assert res["local"]["status"] == "written"
    assert res["s3"]["status"] == "written"
    # both hold it, and verify passes for both
    assert multi.verify(rec) == {"local": True, "s3": True}


def test_multiarchive_is_idempotent(tmp_path):
    rec = _record()
    multi = MultiArchive(
        [LocalArchive(str(tmp_path / "a")), S3Archive("bucket-b", client=FakeS3Client())],
        names=["local", "s3"],
    )
    multi.put(rec)
    again = multi.put(rec)  # must not raise; reports "exists"
    assert again["local"]["status"] == "exists"
    assert again["s3"]["status"] == "exists"


def test_multiarchive_requires_a_backend():
    with pytest.raises(ValueError):
        MultiArchive([])


# --------------------------------------------------------------------------- #
# s3_archive_from_env                                                         #
# --------------------------------------------------------------------------- #

def test_s3_archive_from_env_reads_group_vars_and_deposits(monkeypatch):
    monkeypatch.setenv("REE_PRESERVE_HETZNER_ENDPOINT", "https://fsn1.your-objectstorage.com")
    monkeypatch.setenv("REE_PRESERVE_HETZNER_BUCKET", "ree-preserved")
    monkeypatch.setenv("REE_PRESERVE_HETZNER_KEY_ID", "AKIAtest")
    monkeypatch.setenv("REE_PRESERVE_HETZNER_SECRET", "s3cret")
    monkeypatch.setenv("REE_PRESERVE_HETZNER_PREFIX", "preserved/")

    captured = {}

    def fake_factory(**kw):
        captured.update(kw)
        return FakeS3Client()

    arch = s3_archive_from_env("HETZNER", client_factory=fake_factory)
    # credentials + endpoint were read from the group's env and passed to the client
    assert captured["endpoint_url"] == "https://fsn1.your-objectstorage.com"
    assert captured["aws_access_key_id"] == "AKIAtest"
    assert captured["aws_secret_access_key"] == "s3cret"
    assert arch.bucket == "ree-preserved"
    assert arch.prefix == "preserved/"
    assert arch.name == "hetzner"
    # and it actually works end to end
    key = arch.put(_record("env_life_v3"))
    assert key.startswith("preserved/env_life_v3/")


def test_s3_archive_from_env_requires_endpoint(monkeypatch):
    monkeypatch.delenv("REE_PRESERVE_MISSING_ENDPOINT", raising=False)
    with pytest.raises(ValueError):
        s3_archive_from_env("MISSING", client_factory=lambda **kw: FakeS3Client())
