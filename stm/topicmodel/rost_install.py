# stm, Apache-2.0 license
# Filename: topicmodel/rost_install.py
# Description: Download and cache prebuilt rost-cli binaries
from __future__ import annotations

import io
import os
import tarfile
import urllib.request
from pathlib import Path

# Public HTTPS URL of the hosted bin/ tarball. Override with STM_ROST_CLI_URL.
ROST_CLI_URL = ""

ROST_BINARIES = ("topics.refine.t", "words.bincount")
_DEFAULT_DEST = Path.home() / ".cache" / "stm" / "rost-cli"


def ensure_rost_cli(url: str | None = None, dest: Path | str | None = None) -> Path:
    """Return the directory that contains the rost-cli binaries.

    Downloads and extracts the hosted ``bin/`` archive on first use.
    Subsequent calls with the same *dest* return that directory when both
    binaries are already present.
    """
    resolved_url = url or os.environ.get("STM_ROST_CLI_URL") or ROST_CLI_URL
    if not resolved_url:
        raise RuntimeError(
            "ROST CLI URL is unset. Set STM_ROST_CLI_URL or pass url= to ensure_rost_cli()."
        )
    dest_path = Path(dest) if dest is not None else _DEFAULT_DEST
    bin_dir = dest_path / "bin"
    if all((bin_dir / name).is_file() for name in ROST_BINARIES):
        return bin_dir

    dest_path.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(resolved_url) as response:
        payload = response.read()
    _extract_archive(payload, dest_path)
    missing = [name for name in ROST_BINARIES if not (bin_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            "rost-cli archive is missing "
            + ", ".join(missing)
            + ". Expected bin/topics.refine.t and bin/words.bincount at the archive root."
        )
    return bin_dir


def _extract_archive(payload: bytes, dest: Path) -> None:
    """Extract a gzip tarball into *dest*, rejecting member paths that escape it."""
    try:
        archive = tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz")
    except tarfile.TarError as exc:
        raise RuntimeError("rost-cli download is not a gzip tarball.") from exc
    with archive:
        members = _safe_members(archive, dest)
        extract_kwargs: dict[str, object] = {"members": members}
        if hasattr(tarfile, "data_filter"):
            extract_kwargs["filter"] = "data"
        archive.extractall(dest, **extract_kwargs)


def _safe_members(archive: tarfile.TarFile, dest: Path) -> list[tarfile.TarInfo]:
    """Return archive members, including links such as ``bin/topics.render``.

    Member names that are absolute or contain ``..`` are refused. Symlinks and
    hard links are kept so the rost-cli ``bin/`` tree extracts intact.
    """
    del dest
    members: list[tarfile.TarInfo] = []
    for member in archive.getmembers():
        parts = Path(member.name).parts
        if member.name.startswith("/") or ".." in parts:
            raise RuntimeError(f"refusing suspect path in rost-cli archive: {member.name}")
        members.append(member)
    return members
