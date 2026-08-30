"""
tests/e2e/_harness.py

Shared infrastructure for end-to-end tests:

    ONNX model -> tiny-os compiler -> gcc shared library -> ctypes execution
    -> numeric comparison against onnxruntime.

The build uses unique library names so parallel/serial reruns never hit the
Windows "file in use" linker lock.
"""

import ctypes
import os
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent


def compile_model(onnx_path: Path, out_dir: Path) -> None:
    r = subprocess.run(
        [sys.executable, "-m", "compiler.main", str(onnx_path), str(out_dir)],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"tiny-os compiler failed for {onnx_path}:\n{r.stdout}\n{r.stderr}"
        )


def build_runner(gen_dir: Path, workdir: Path, tag: str = "runner") -> Path:
    """Compile kernels + generated image into a fresh uniquely-named DLL."""
    workdir.mkdir(parents=True, exist_ok=True)
    ext = ".dll" if os.name == "nt" else ".so"
    lib_path = workdir / f"tinyos_{tag}_{uuid.uuid4().hex[:8]}{ext}"
    cmd = [
        "gcc", "-shared", "-o", str(lib_path), "-O2",
        "-Wall", "-Wextra", "-Werror",
        "-I", str(ROOT / "runtime" / "include"),
        "-I", str(ROOT / "runtime" / "dispatch"),
        "-I", str(gen_dir),
        str(ROOT / "runtime" / "kernels" / "cpu_ref" / "kernels_ref.c"),
        str(ROOT / "runtime" / "dispatch" / "dma_engine.c"),
        str(ROOT / "runtime" / "power" / "power_mgr.c"),
        str(gen_dir / "model_exec.c"),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"gcc build failed:\n{r.stderr}")
    return lib_path


class TinyOSRunner:
    """ctypes wrapper around the built image."""

    def __init__(self, lib_path: Path):
        self.lib = ctypes.CDLL(str(lib_path))
        self.lib.model_exec_run.argtypes = []
        self.lib.model_exec_run.restype = ctypes.c_int   # tinyos_status_t
        self.lib.model_exec_run_ctx.argtypes = [ctypes.c_uint32, ctypes.c_uint64]
        self.lib.model_exec_run_ctx.restype = ctypes.c_int
        self.lib.model_get_tensor_ptr.argtypes = [ctypes.c_char_p]
        self.lib.model_get_tensor_ptr.restype = ctypes.c_void_p
        self.lib.model_get_tensor_size.argtypes = [ctypes.c_char_p]
        self.lib.model_get_tensor_size.restype = ctypes.c_uint32

    def set_input(self, name: str, arr: np.ndarray) -> None:
        ptr = self.lib.model_get_tensor_ptr(name.encode())
        assert ptr, f"tensor '{name}' has no address"
        size = self.lib.model_get_tensor_size(name.encode())
        assert size == arr.nbytes, (
            f"'{name}': runtime size {size}B != host {arr.nbytes}B"
        )
        ctypes.memmove(ptr, arr.ctypes.data, size)

    def run(self) -> int:
        return self.lib.model_exec_run()

    def run_ctx(self, caps: int, deadline_cycles: int = 0) -> int:
        return self.lib.model_exec_run_ctx(caps, deadline_cycles)

    # Simulated machine telemetry (dma_engine.h exports)
    def telemetry(self) -> dict:
        def u64(name):
            return ctypes.c_uint64.in_dll(self.lib, name).value
        def u32(name):
            return ctypes.c_uint32.in_dll(self.lib, name).value
        return {
            "cycle_count": u64("g_cycle_count"),
            "dma_submits": u32("g_dma_submits"),
            "dma_copies": u32("g_dma_copies"),
            "dma_transfer_cycles": u64("g_dma_transfer_cycles"),
        }

    def get_output_raw(self, name: str, dtype, shape) -> np.ndarray:
        ptr = self.lib.model_get_tensor_ptr(name.encode())
        assert ptr, f"tensor '{name}' has no address"
        size = self.lib.model_get_tensor_size(name.encode())
        buf = np.empty(size, dtype=np.uint8)
        ctypes.memmove(buf.ctypes.data, ptr, size)
        return buf.view(dtype).reshape(shape)


def run_case(tmp_root: Path, tag: str, onnx_bytes_builder, feeds: dict,
             atol: float = 1e-4, rtol: float = 1e-4):
    """Full pipeline check for one model.

    onnx_bytes_builder() -> (onnx.ModelProto, input_feeds_spec) where
    feeds maps input name -> np.ndarray.
    Returns list of (output_name, ref, got).
    """
    import onnx
    import onnxruntime as ort

    tmp_root.mkdir(parents=True, exist_ok=True)
    gen_dir = tmp_root / f"gen_{tag}"
    onnx_path = tmp_root / f"{tag}.onnx"

    model = onnx_bytes_builder()
    onnx.save(model, str(onnx_path))

    compile_model(onnx_path, gen_dir)
    lib = build_runner(gen_dir, tmp_root / f"build_{tag}", tag)

    sess = ort.InferenceSession(str(onnx_path))
    ort_outs = sess.run(None, feeds)

    runner = TinyOSRunner(lib)
    for name, arr in feeds.items():
        runner.set_input(name, arr)
    status = runner.run()
    assert status == 0, f"model_exec_run faulted with status {status}"

    results = []
    for out_meta, ref in zip(sess.get_outputs(), ort_outs):
        name = out_meta.name
        got = runner.get_output_raw(name, ref.dtype, ref.shape)
        results.append((name, ref, got))
    return results


def assert_matches(results, atol=1e-4, rtol=1e-4):
    for name, ref, got in results:
        np.testing.assert_allclose(
            ref, got, rtol=rtol, atol=atol,
            err_msg=f"Output '{name}' diverges from onnxruntime",
        )
