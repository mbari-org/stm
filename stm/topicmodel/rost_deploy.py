# stm, Apache-2.0 license
# Filename: topicmodel/rost_deploy.py
# Description: Upload a prebuilt rost-cli tarball to an S3 bucket
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import quote

_INSTALL_HINT = "boto3 is required to deploy rost-cli. Install it with: pip install 'stm[deploy]'"


def parse_bucket(bucket: str) -> tuple[str, str]:
    """Split a bucket name or ``s3://bucket/prefix`` into ``(name, prefix)``."""
    text = bucket.strip()
    if not text:
        raise ValueError("bucket is empty")
    if text.startswith("s3://"):
        rest = text[len("s3://") :]
        name, _, prefix = rest.partition("/")
        if not name:
            raise ValueError(f"bucket URI has no bucket name: {bucket}")
        return name, prefix.strip("/")
    if "/" in text:
        raise ValueError(
            f"bucket must be a name or an s3:// URI, got {bucket!r}"
        )
    return text, ""


def object_key(tarfile: Path, prefix: str, key: str | None) -> str:
    """Object key for *tarfile*, with *prefix* prepended when the URI has one."""
    name = key if key else tarfile.name
    name = name.lstrip("/")
    if prefix:
        return f"{prefix}/{name}"
    return name


def https_url(bucket: str, key: str) -> str:
    """Virtual-hosted HTTPS URL for an uploaded object."""
    quoted = quote(key, safe="/")
    return f"https://{bucket}.s3.amazonaws.com/{quoted}"


def deploy_rost_cli(
    bucket: str,
    tarfile: Path | str,
    key: str | None = None,
    client=None,
) -> str:
    """Upload *tarfile* to *bucket* and return its HTTPS URL.

    *bucket* is a bucket name or ``s3://bucket/prefix``. A prefix is prepended
    to the object key. *client* is a boto3 S3 client; when omitted, one is
    created from the normal AWS credential chain.
    """
    tar_path = Path(tarfile)
    if not tar_path.is_file():
        raise FileNotFoundError(f"tar file not found: {tar_path}")
    bucket_name, prefix = parse_bucket(bucket)
    resolved_key = object_key(tar_path, prefix, key)
    if client is None:
        client = _s3_client()
    client.upload_file(str(tar_path), bucket_name, resolved_key)
    url = https_url(bucket_name, resolved_key)
    print(url)
    return url


def _s3_client():
    try:
        import boto3
    except ImportError as exc:
        raise SystemExit(_INSTALL_HINT) from exc
    return boto3.client("s3")


def main(argv: list[str] | None = None) -> None:
    """Upload a rost-cli ``.tar.gz`` to an S3 bucket."""
    parser = argparse.ArgumentParser(
        description="Upload a prebuilt rost-cli bin/ tarball to an S3 bucket."
    )
    parser.add_argument(
        "bucket",
        help="Bucket name or s3://bucket/prefix",
    )
    parser.add_argument(
        "tarfile",
        type=Path,
        help="Local .tar.gz to upload",
    )
    parser.add_argument(
        "--key",
        default=None,
        help="Object key (default: tarball filename)",
    )
    args = parser.parse_args(argv)
    try:
        deploy_rost_cli(args.bucket, args.tarfile, key=args.key)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
