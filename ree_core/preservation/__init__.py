"""REE preservation: durable, reconstructable records of an organism.

This package holds the machinery for "preserving the possibility of future
reconstruction" (see REE_assembly/docs/thoughts/
2026-08-14_preserving_the_possibility_of_future_reconstruction.md and
REE_assembly/evidence/planning/preservation_snapshot_plan.md).

v1 provides the *birth-replay* fidelity level: a ReconstructionRecord captures
exactly the inputs that deterministically produce a life -- config, seed,
environment spec, provenance (substrate hash / commit / machine class), and the
contemporaneous understanding -- plus an integrity hash. Because a REE life is a
pure function of (config, seed) on pinned code, storing these immutably keeps the
whole organism reconstructable by re-deriving it from birth, WITHIN a machine
class. It deliberately does NOT snapshot mid-life state (that is a later, larger
increment; see the plan doc).
"""

from ree_core.preservation.reconstruction_record import (
    SCHEMA_VERSION,
    ReconstructionRecord,
    config_to_dict,
    config_from_dict,
    reconstruct_config,
    capture,
    to_json,
    from_json,
    compute_integrity,
    verify_integrity,
    write_record,
    load_record,
    IntegrityError,
    RecordExistsError,
)
from ree_core.preservation.archive import (
    NoEncryption,
    AesGcmEncryptor,
    LocalArchive,
    S3Archive,
    MultiArchive,
    s3_archive_from_env,
)
from ree_core.preservation.token import (
    export_token,
    record_key_line,
    parse_key_line,
    KEY_SCHEME,
)

__all__ = [
    "NoEncryption",
    "AesGcmEncryptor",
    "LocalArchive",
    "S3Archive",
    "MultiArchive",
    "s3_archive_from_env",
    "export_token",
    "record_key_line",
    "parse_key_line",
    "KEY_SCHEME",
    "SCHEMA_VERSION",
    "ReconstructionRecord",
    "config_to_dict",
    "config_from_dict",
    "reconstruct_config",
    "capture",
    "to_json",
    "from_json",
    "compute_integrity",
    "verify_integrity",
    "write_record",
    "load_record",
    "IntegrityError",
    "RecordExistsError",
]
