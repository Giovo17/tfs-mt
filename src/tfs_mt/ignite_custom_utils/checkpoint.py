# Ignite custom checkpointer
#
# Author: Giovanni Spadaro - https://giovannispadaro.it
# Project: https://github.com/Giovo17/tfs-mt
# Documentation: https://giovo17.github.io/tfs-mt
#
# Copyright (c) Giovanni Spadaro.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import boto3
import botocore
import ignite.distributed as idist
import torch
from botocore.client import BaseClient
from dotenv import load_dotenv
from ignite.handlers.checkpoint import BaseSaveHandler


class BucketNotFoundError(Exception):
    def __init__(self, bucket):
        msg = f"S3 bucket '{bucket}' does not exist."
        super().__init__(msg)


class BucketNotEmptyError(Exception):
    def __init__(self, bucket, prefix, existing):
        msg = f"""Bucket '{bucket}' already contains .pt or .pth objects under prefix "
        '{prefix}': {existing}. Set `require_empty=False` to ignore."""
        super().__init__(msg)


class S3Saver(BaseSaveHandler):
    """
    Handler that saves a checkpoint directly to an S3 bucket.

    Args:
        bucket:      Name of the target S3 bucket.
        prefix:      Optional key prefix (e.g. ``"checkpoints/"``).  The final
                     object key will be ``prefix + filename``.
        atomic:      If ``True`` the checkpoint is first written to a temporary
                     local file and then uploaded, guaranteeing that a partially
                     uploaded object never appears in the bucket.
        create_bucket: If ``True`` the bucket will be created when it does not
                     already exist (requires appropriate IAM permissions).
        require_empty: If ``True`` raises an error when the bucket already
                     contains objects with the same ``prefix`` and ``.pt`` suffix.
        save_on_rank: Rank on which the checkpoint will be saved (useful for
                     distributed training).
        aws_kwargs:  Additional keyword arguments passed to ``boto3.client``.
        **kwargs:    Keyword arguments forwarded to ``torch.save`` (or ``xm.save``
                     for XLA devices).
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        atomic: bool = True,
        require_empty: bool = True,
        save_on_rank: int = 0,
        **kwargs: Any,
    ):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + "/" if prefix and not prefix.endswith("/") else prefix
        self._atomic = atomic
        self.save_on_rank = save_on_rank
        self.kwargs = kwargs

        # Check if any required env vars are missing, load .env if needed
        required_vars = ["AWS_ACCESS_KEY", "AWS_SECRET_KEY", "AWS_REGION", "CUSTOM_BASE_URL"]
        if any(os.getenv(var) is None for var in required_vars):
            load_dotenv()
        self.s3 = self._make_s3_client()

        if idist.get_rank() == save_on_rank:
            self._check_and_setup(require_empty)

    @staticmethod
    def _make_s3_client() -> BaseClient:
        """Create a boto3 S3 client using env vars and optional custom endpoint."""
        access_key = os.getenv("AWS_ACCESS_KEY")
        secret_key = os.getenv("AWS_SECRET_KEY")
        region = os.getenv("AWS_REGION")
        custom_base = os.getenv("CUSTOM_BASE_URL")

        session_kwargs = {}
        if access_key and secret_key:
            session_kwargs["aws_access_key_id"] = access_key
            session_kwargs["aws_secret_access_key"] = secret_key
        if region:
            session_kwargs["region_name"] = region

        session = boto3.Session(**session_kwargs)

        client_kwargs: dict = {}
        if custom_base:
            # e.g. CUSTOM_BASE_URL = "example.com" → endpoint https://s3.example.com
            client_kwargs["endpoint_url"] = f"https://s3.{custom_base}"

        return session.client("s3", **client_kwargs)

    def _check_and_setup(self, require_empty: bool) -> None:
        """Validate bucket existence and optionally create it."""

        # Verify bucket existence
        try:
            self.s3.head_bucket(Bucket=self.bucket)
        except botocore.exceptions.ClientError as exc:
            if exc.response["Error"]["Code"] == "404":
                raise BucketNotFoundError(self.bucket) from exec
            else:
                raise

        if require_empty:
            resp = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=self.prefix)
            existing = [obj["Key"] for obj in resp.get("Contents", []) if obj["Key"].endswith((".pt", ".pth"))]
            if existing:
                raise BucketNotEmptyError(self.bucket, self.prefix, existing)

    def __call__(
        self,
        checkpoint: Mapping,
        filename: str,
        metadata: Mapping | None = None,
    ) -> None:
        """Save ``checkpoint`` to S3 under ``self.prefix + filename``."""
        s3_key = f"{self.prefix}{filename}"

        if idist.has_xla_support:
            import torch_xla.core.xla_model as xm

            self._save_func(checkpoint, s3_key, xm.save)
        elif self.save_on_rank == idist.get_rank():
            self._save_func(checkpoint, s3_key, torch.save)

    def _save_func(self, checkpoint: Mapping, s3_key: str, func: Callable) -> None:
        """Write checkpoint to a temporary file (if atomic) and upload to S3."""
        if not self._atomic:
            # Direct upload from an inmemory buffer (no temp file)
            with tempfile.SpooledTemporaryFile() as buf:
                func(checkpoint, buf, **self.kwargs)
                buf.seek(0)
                self.s3.upload_fileobj(buf, self.bucket, s3_key)
        else:
            # Write to a local temp file first, guarantees that a partially written object never appears in the bucket.
            with tempfile.NamedTemporaryFile(delete=False, dir="/tmp") as tmp:
                tmp_path = Path(tmp.name)
                try:
                    func(checkpoint, tmp_path, **self.kwargs)
                except BaseException:
                    tmp.close()
                    os.remove(tmp_path)
                    raise

            # Upload the completed file
            self.s3.upload_file(str(tmp_path), self.bucket, s3_key)

            # Clean up the temporary file
            os.remove(tmp_path)

            # Optional: make the object readable by others (mirrors chmod)
            # Users can control this via an env var if desired.
            if os.getenv("S3_PUBLIC_READ") == "1":
                self.s3.put_object_acl(Bucket=self.bucket, Key=s3_key, ACL="public-read")

    def remove(self, filename: str) -> None:
        """Delete ``filename`` from the bucket."""
        if idist.get_rank() == self.save_on_rank:
            s3_key = f"{self.prefix}{filename}"
            self.s3.delete_object(Bucket=self.bucket, Key=s3_key)
