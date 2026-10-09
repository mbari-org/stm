# stm, Apache-2.0 license
# Filename: test_rost_install.py
# Description: Tests for rost-cli download and S3 deploy
from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from stm.topicmodel.rost_deploy import deploy_rost_cli, parse_bucket
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


class _FakeS3:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        self.calls.append((filename, bucket, key))


def _tar_bytes(*names: str) -> bytes:
    """Build a gzip tarball whose root contains bin/<name> files."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in names:
            data = f"{name}\n".encode()
            info = tarfile.TarInfo(name=f"bin/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
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


def test_parse_bucket_accepts_name_and_uri() -> None:
    """Test that a bare name and an s3:// URI both parse."""
    assert parse_bucket("my-bucket") == ("my-bucket", "")
    assert parse_bucket("s3://my-bucket/releases/linux") == (
        "my-bucket",
        "releases/linux",
    )


def test_deploy_rost_cli_uploads_to_bucket_and_prefix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that the tar is uploaded under the URI prefix and the URL is printed."""
    tar_path = tmp_path / "rost-cli-linux-x86_64.tar.gz"
    tar_path.write_bytes(b"archive")
    client = _FakeS3()
    url = deploy_rost_cli("s3://my-bucket/releases", tar_path, client=client)
    assert client.calls == [
        (str(tar_path), "my-bucket", "releases/rost-cli-linux-x86_64.tar.gz")
    ]
    assert url == (
        "https://my-bucket.s3.amazonaws.com/releases/rost-cli-linux-x86_64.tar.gz"
    )
    assert capsys.readouterr().out.strip() == url


def test_deploy_rost_cli_uses_explicit_key(tmp_path: Path) -> None:
    """Test that --key replaces the object name and still takes the URI prefix."""
    tar_path = tmp_path / "local.tar.gz"
    tar_path.write_bytes(b"archive")
    client = _FakeS3()
    deploy_rost_cli("my-bucket", tar_path, key="custom/rost.tar.gz", client=client)
    assert client.calls == [(str(tar_path), "my-bucket", "custom/rost.tar.gz")]


def test_deploy_requires_boto3_when_no_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a missing boto3 install exits with the extra name."""
    tar_path = tmp_path / "rost.tar.gz"
    tar_path.write_bytes(b"archive")
    real_import = __import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "boto3":
            raise ImportError("no boto3")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(SystemExit, match="stm\\[deploy\\]"):
        deploy_rost_cli("my-bucket", tar_path)
