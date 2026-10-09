# stm, Apache-2.0 license
# Filename: test_rost_install.py
# Description: Tests for downloading prebuilt rost-cli binaries
from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from stm.topicmodel.rost_install import ensure_rost_cli


class _Response:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def _tar_bytes(*names: str, links: dict[str, str] | None = None) -> bytes:
    """Build a gzip tarball whose root contains bin/<name> files and symlinks."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in names:
            data = f"{name}\n".encode()
            info = tarfile.TarInfo(name=f"bin/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        for name, target in (links or {}).items():
            info = tarfile.TarInfo(name=f"bin/{name}")
            info.type = tarfile.SYMTYPE
            info.linkname = target
            archive.addfile(info)
    return buffer.getvalue()


def test_ensure_rost_cli_extracts_and_skips_second_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that a valid archive is extracted once and reused."""
    calls = {"n": 0}
    payload = _tar_bytes("topics.refine.t", "words.bincount")

    def urlopen(url: str) -> _Response:
        calls["n"] += 1
        assert url == "https://example.test/rost.tar.gz"
        return _Response(payload)

    monkeypatch.setattr("stm.topicmodel.rost_install.urllib.request.urlopen", urlopen)
    first = ensure_rost_cli(url="https://example.test/rost.tar.gz", dest=tmp_path)
    second = ensure_rost_cli(url="https://example.test/rost.tar.gz", dest=tmp_path)
    assert first == second == tmp_path / "bin"
    assert (first / "topics.refine.t").is_file()
    assert (first / "words.bincount").is_file()
    assert calls["n"] == 1


def test_ensure_rost_cli_rejects_archive_missing_a_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that an archive without words.bincount is refused."""
    payload = _tar_bytes("topics.refine.t")
    monkeypatch.setattr(
        "stm.topicmodel.rost_install.urllib.request.urlopen",
        lambda url: _Response(payload),
    )
    with pytest.raises(RuntimeError, match="words.bincount"):
        ensure_rost_cli(url="https://example.test/rost.tar.gz", dest=tmp_path)


def test_ensure_rost_cli_extracts_internal_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that a symlink such as bin/topics.render is extracted."""
    payload = _tar_bytes(
        "topics.refine.t",
        "words.bincount",
        links={"topics.render": "topics.refine.t"},
    )
    monkeypatch.setattr(
        "stm.topicmodel.rost_install.urllib.request.urlopen",
        lambda url: _Response(payload),
    )
    bin_dir = ensure_rost_cli(url="https://example.test/rost.tar.gz", dest=tmp_path)
    rendered = bin_dir / "topics.render"
    assert rendered.is_symlink()
    assert rendered.read_text() == "topics.refine.t\n"


def test_ensure_rost_cli_rejects_parent_path_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that a member name containing .. is refused."""
    payload = _tar_bytes("topics.refine.t", "words.bincount", links={"../outside": "topics.refine.t"})
    monkeypatch.setattr(
        "stm.topicmodel.rost_install.urllib.request.urlopen",
        lambda url: _Response(payload),
    )
    with pytest.raises(RuntimeError, match="suspect path"):
        ensure_rost_cli(url="https://example.test/rost.tar.gz", dest=tmp_path)


def test_ensure_rost_cli_requires_a_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that download is not attempted when no URL is configured."""
    monkeypatch.delenv("STM_ROST_CLI_URL", raising=False)
    monkeypatch.setattr("stm.topicmodel.rost_install.ROST_CLI_URL", "")

    def urlopen(url: str) -> _Response:
        raise AssertionError(f"unexpected download of {url}")

    monkeypatch.setattr("stm.topicmodel.rost_install.urllib.request.urlopen", urlopen)
    with pytest.raises(RuntimeError, match="URL is unset"):
        ensure_rost_cli(dest=tmp_path)
