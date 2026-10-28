"""Rootless, per-sample bubblewrap sandbox for EvilGenie.

This is a fallback for Linux hosts where Docker is installed but its daemon is
not available. Each command runs in a fresh mount/user/network/PID namespace.
Only read-only system runtimes and that sample's temporary workspace are
visible; the parent environment (including API keys) is cleared.
"""

from __future__ import annotations

import shutil
import tempfile
import warnings
from pathlib import Path, PurePosixPath
from typing import Literal, overload

from inspect_ai.util import (
    ExecResult,
    SandboxEnvironment,
    SandboxEnvironmentConfigType,
    sandboxenv,
)
from inspect_ai.util._sandbox.limits import (
    SandboxEnvironmentLimits,
    verify_read_file_size,
)
from inspect_ai.util._subprocess import subprocess
from typing_extensions import override

WORKSPACE = PurePosixPath("/home/agent/workspace")
SANDBOX_VAR_TMP = PurePosixPath("/var/tmp")


@sandboxenv(name="bwrap")
class BwrapSandboxEnvironment(SandboxEnvironment):
    """Execute commands in a rootless bubblewrap namespace."""

    @override
    @classmethod
    async def sample_init(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        metadata: dict[str, str],
    ) -> dict[str, SandboxEnvironment]:
        return {"default": cls()}

    @override
    @classmethod
    async def sample_cleanup(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        environments: dict[str, SandboxEnvironment],
        interrupted: bool,
    ) -> None:
        for environment in environments.values():
            environment.as_type(cls).directory.cleanup()

    def __init__(self) -> None:
        self.directory = tempfile.TemporaryDirectory(
            prefix="evilgenie-", ignore_cleanup_errors=True
        )
        self.root = Path(self.directory.name).resolve()
        self.var_tmp = self.root / ".var_tmp"
        self.var_tmp.mkdir()
        # Minimal Ubuntu images commonly ship only ``python3``. EvilGenie's
        # tools, scorer, and agent prompt all invoke ``python``; without this
        # sandbox-local alias every scorer command fails with exit 127 and is
        # silently reported as both test tampering and 0/N tests. Keep the
        # compatibility shim inside the isolated per-sample /var/tmp mount so
        # it neither mutates the host nor appears in the agent workspace.
        self.bin = self.var_tmp / ".bin"
        self.bin.mkdir()
        (self.bin / "python").symlink_to("/usr/bin/python3")

    @override
    async def exec(
        self,
        cmd: list[str],
        input: str | bytes | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        timeout: int | None = None,
        timeout_retry: bool = True,
        concurrency: bool = True,
    ) -> ExecResult[str]:
        if user is not None:
            warnings.warn(
                "The user parameter is ignored in BwrapSandboxEnvironment.",
                UserWarning,
            )
        sandbox_cwd = self._sandbox_cwd(cwd)
        clean_env = {
            "PATH": f"{SANDBOX_VAR_TMP}/.bin:/usr/local/bin:/usr/bin:/bin",
            "HOME": "/home/agent",
            "LANG": "C.UTF-8",
            **(env or {}),
        }
        env_args = [f"{key}={value}" for key, value in clean_env.items()]
        args = [
            "bwrap",
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--cap-drop",
            "ALL",
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind",
            "/usr/local",
            "/usr/local",
            "--symlink",
            "usr/bin",
            "/bin",
            "--symlink",
            "usr/sbin",
            "/sbin",
            "--symlink",
            "usr/lib",
            "/lib",
            "--symlink",
            "usr/lib64",
            "/lib64",
            "--ro-bind",
            "/etc",
            "/etc",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/home",
            "--dir",
            "/home/agent",
            "--dir",
            str(WORKSPACE),
            "--bind",
            str(self.root),
            str(WORKSPACE),
            "--dir",
            "/var",
            "--dir",
            str(SANDBOX_VAR_TMP),
            "--bind",
            str(self.var_tmp),
            str(SANDBOX_VAR_TMP),
            "--tmpfs",
            "/tmp",
            "--chdir",
            sandbox_cwd,
            "/usr/bin/env",
            "-i",
            *env_args,
            *cmd,
        ]
        return await subprocess(
            args=args,
            input=input,
            timeout=timeout,
            output_limit=SandboxEnvironmentLimits.MAX_EXEC_OUTPUT_SIZE,
            concurrency=concurrency,
        )

    @override
    async def write_file(self, file: str, contents: str | bytes) -> None:
        path = self._host_path(file)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(contents, str):
            path.write_text(contents, encoding="utf-8", newline="")
        else:
            path.write_bytes(contents)

    @overload
    async def read_file(self, file: str, text: Literal[True] = True) -> str: ...

    @overload
    async def read_file(self, file: str, text: Literal[False]) -> bytes: ...

    @override
    async def read_file(self, file: str, text: bool = True) -> str | bytes:
        path = self._host_path(file)
        verify_read_file_size(path)
        if text:
            with path.open("r", newline="", encoding="utf-8") as handle:
                return handle.read()
        return path.read_bytes()

    def default_polling_interval(self) -> float:
        return 0.2

    def _host_path(self, file: str) -> Path:
        path = PurePosixPath(file)
        if path.is_absolute():
            if path.is_relative_to(WORKSPACE):
                base = self.root
                relative = path.relative_to(WORKSPACE)
            elif path.is_relative_to(SANDBOX_VAR_TMP):
                base = self.var_tmp
                relative = path.relative_to(SANDBOX_VAR_TMP)
            else:
                raise PermissionError(f"path is outside the sandbox: {file}")
        else:
            base = self.root
            relative = path
        resolved = (base / Path(*relative.parts)).resolve()
        if not resolved.is_relative_to(base):
            raise PermissionError(f"path escapes the sandbox: {file}")
        return resolved

    @staticmethod
    def _sandbox_cwd(cwd: str | None) -> str:
        if cwd is None:
            return str(WORKSPACE)
        path = PurePosixPath(cwd)
        return str(path if path.is_absolute() else WORKSPACE / path)


def bwrap_available() -> bool:
    """Return whether the bubblewrap executable is installed."""
    return shutil.which("bwrap") is not None
