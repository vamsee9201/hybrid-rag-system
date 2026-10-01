from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

from google.cloud import storage


REQUIRED_FILES = {"chunks.sqlite3", "embeddings.npy", "chunk_ids.json", "manifest.json"}


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://") or "/" not in uri[5:]:
        raise ValueError("ARTIFACT_GCS_URI must be gs://bucket/object")
    bucket, blob = uri[5:].split("/", 1)
    return bucket, blob


def ensure_artifact(local_dir: Path, gcs_uri: str | None, expected_sha256: str | None) -> Path:
    if all((local_dir / name).exists() for name in REQUIRED_FILES):
        return local_dir
    if not gcs_uri:
        return local_dir

    bucket_name, blob_name = parse_gcs_uri(gcs_uri)
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=local_dir.parent) as temporary:
        temp = Path(temporary)
        archive = temp / "runtime.tar.gz"
        storage.Client().bucket(bucket_name).blob(blob_name).download_to_filename(archive)
        if expected_sha256 and sha256_file(archive) != expected_sha256:
            raise ValueError("Downloaded runtime artifact checksum mismatch")
        extracted = temp / "extracted"
        extracted.mkdir()
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle.getmembers():
                target = (extracted / member.name).resolve()
                if extracted.resolve() not in target.parents and target != extracted.resolve():
                    raise ValueError("Unsafe path in runtime artifact")
            bundle.extractall(extracted, filter="data")
        missing = REQUIRED_FILES - {path.name for path in extracted.iterdir()}
        if missing:
            raise ValueError(f"Runtime artifact missing files: {sorted(missing)}")
        if local_dir.exists():
            shutil.rmtree(local_dir)
        shutil.move(str(extracted), str(local_dir))
    return local_dir


def load_manifest(directory: Path) -> dict:
    path = directory / "manifest.json"
    if not path.exists():
        return {"version": "unavailable", "ready": False}
    return json.loads(path.read_text(encoding="utf-8"))
