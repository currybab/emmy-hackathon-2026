"""Read a Kernel IR json and its sibling meta.json into one immutable KernelSpec.

The IR gives the kernel's name and body. meta.json is the launch contract: the parameter
order (`arg_order`), the buffer shapes, and the grid/block the kernel will run under.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from ptxc.errors import CompileError

SUPPORTED_DTYPES = {"f32"}


@dataclass(frozen=True)
class Buffer:
    name: str
    shape: tuple[int, ...]
    dtype: str
    is_output: bool


@dataclass(frozen=True)
class KernelSpec:
    name: str
    tile: object  # emmy's Tile statement: the whole kernel body
    params: tuple[Buffer, ...]  # in meta.json arg_order
    grid: tuple[int, int, int]
    block: tuple[int, int, int]

    @property
    def total_threads(self) -> int:
        return math.prod(self.grid) * math.prod(self.block)

    @property
    def block_threads(self) -> int:
        return math.prod(self.block)


def _read_json(path: Path, what: str) -> dict:
    try:
        return json.loads(path.read_text())
    except OSError as exc:
        raise CompileError(f"cannot read {what} {path}: {exc.strerror}") from exc
    except json.JSONDecodeError as exc:
        raise CompileError(f"{what} {path} is not valid json: {exc}") from exc


def _kernel_op(ir: dict, path: Path) -> object:
    from emmy.compiler.graph import Graph

    try:
        graph = Graph.from_dict(ir)
    except Exception as exc:  # emmy raises many kinds; all mean "not a loadable IR"
        raise CompileError(f"emmy could not load the IR {path}: {exc}") from exc
    kernels = [node.op for node in graph.nodes.values() if type(node.op).__name__ == "KernelOp"]
    if len(kernels) != 1:
        raise CompileError(f"expected exactly one KernelOp in {path}, found {len(kernels)}")
    return kernels[0]


def _launch(meta: dict, entry: str, meta_path: Path) -> dict:
    for kernel in meta.get("kernels", []):
        if kernel.get("entry") == entry:
            return kernel
    raise CompileError(f"kernel {entry!r} is not listed in {meta_path}")


def _buffer(arg: dict, entry: str) -> Buffer:
    if arg.get("scalar"):
        raise CompileError(f"{entry}: scalar parameter {arg['name']!r} is not supported")
    if arg["dtype"] not in SUPPORTED_DTYPES:
        raise CompileError(f"{entry}: buffer {arg['name']!r} has unsupported dtype {arg['dtype']!r}")
    return Buffer(
        name=arg["name"],
        shape=tuple(int(d) for d in arg["shape"]),
        dtype=arg["dtype"],
        is_output=bool(arg.get("output")),
    )


def _dims3(values: list, what: str) -> tuple[int, int, int]:
    dims = tuple(int(v) for v in values)
    if len(dims) != 3:
        raise CompileError(f"meta.json {what} must have 3 dimensions, got {values!r}")
    return dims


def load_kernel(ir_path: Path) -> KernelSpec:
    ir_path = Path(ir_path)
    op = _kernel_op(_read_json(ir_path, "IR"), ir_path)

    meta_path = ir_path.parent / "meta.json"
    if not meta_path.exists():
        raise CompileError(f"no meta.json next to {ir_path}; it carries arg_order and the launch")
    launch = _launch(_read_json(meta_path, "meta"), op.name, meta_path)

    args = {arg["name"]: arg for arg in launch["args"]}
    missing = [name for name in launch["arg_order"] if name not in args]
    if missing:
        raise CompileError(f"{op.name}: arg_order names {missing} without an args entry")

    if len(op.body) != 1 or type(op.body[0]).__name__ != "Tile":
        raise CompileError(f"{op.name}: expected the body to be a single Tile")

    return KernelSpec(
        name=op.name,
        tile=op.body[0],
        params=tuple(_buffer(args[name], op.name) for name in launch["arg_order"]),
        grid=_dims3(launch["grid"], "grid"),
        block=_dims3(launch["block"], "block"),
    )
