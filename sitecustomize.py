"""Project-wide Python startup defaults."""

from __future__ import annotations

import os
import sys
from pathlib import Path


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def _argument_value(name: str) -> str | None:
    try:
        index = sys.argv.index(name)
    except ValueError:
        return None
    return sys.argv[index + 1] if index + 1 < len(sys.argv) else None


def _should_reserve_cuda_memory() -> bool:
    raw_gib = os.environ.get("GMNER_CUDA_RESERVE_GB", "").strip()
    if not raw_gib or float(raw_gib) <= 0:
        return False

    expected_script = os.environ.get(
        "GMNER_CUDA_RESERVE_SCRIPT",
        "train.py",
    ).strip()
    if expected_script and Path(sys.argv[0]).name != expected_script:
        return False

    expected_config = os.environ.get(
        "GMNER_CUDA_RESERVE_CONFIG_BASENAME",
        "",
    ).strip()
    config_value = _argument_value("--config")
    if expected_config and (
        config_value is None or Path(config_value).name != expected_config
    ):
        return False
    return True


def _reserve_cuda_memory() -> None:
    """Prime the process-local CUDA cache so later tensors can reuse it."""

    if not _should_reserve_cuda_memory():
        return

    strict = os.environ.get("GMNER_CUDA_RESERVE_STRICT", "1") != "0"
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        reserve_gib = float(os.environ["GMNER_CUDA_RESERVE_GB"])
        if reserve_gib <= 0:
            return
        chunk_mib = max(
            16,
            int(os.environ.get("GMNER_CUDA_RESERVE_CHUNK_MB", "256")),
        )
        target_bytes = int(reserve_gib * 1024**3)
        chunk_bytes = chunk_mib * 1024**2
        device_name = os.environ.get("GMNER_CUDA_RESERVE_DEVICE", "cuda")
        requested_device = torch.device(device_name)
        device = torch.device(
            "cuda",
            (
                requested_device.index
                if requested_device.index is not None
                else torch.cuda.current_device()
            ),
        )

        torch.cuda.set_device(device)
        reserved_before = int(torch.cuda.memory_reserved(device))
        blocks = []
        remaining = target_bytes
        while remaining > 0:
            allocation_bytes = min(chunk_bytes, remaining)
            blocks.append(
                torch.empty(
                    allocation_bytes,
                    dtype=torch.uint8,
                    device=device,
                )
            )
            remaining -= allocation_bytes
        torch.cuda.synchronize(device)
        blocks.clear()

        reserved_after = int(torch.cuda.memory_reserved(device))
        reserved_delta = max(0, reserved_after - reserved_before)
        minimum_expected = int(target_bytes * 0.90)
        if reserved_delta < minimum_expected:
            raise RuntimeError(
                "PyTorch retained only "
                f"{reserved_delta / 1024**3:.2f} GiB of the requested "
                f"{reserve_gib:.2f} GiB"
            )
        print(
            "[gmner.cuda_reserve] "
            f"cached={reserved_delta / 1024**3:.2f} GiB "
            f"total_reserved={reserved_after / 1024**3:.2f} GiB "
            f"device={device}",
            file=sys.stderr,
            flush=True,
        )
    except Exception as exc:
        print(
            "[gmner.cuda_reserve] failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        if strict:
            raise SystemExit(86) from exc


_reserve_cuda_memory()
