"""A grid of cells: one arm over worlds × seeds, run in parallel and resumable (blueprint §6).

An arm is a proposer kind and an `MDAConfig`, read from a JSON file in `configs/`:

    {"description": "...", "proposer": "null", "config": {"n_starts": 4}}

`config` holds the fields that differ from the defaults, and `description` is optional. The
proposer kind is a key of `PROPOSERS`, which maps it to a factory.

`run_grid` runs `run_cell` for every (world, seed) cell and writes its log to
`<out>/<world>/s<seed>.json`, exactly the text of `RunLog.to_json`, with a manifest of the runs in
`<out>/manifest.json`:

- Cells run in parallel, each from start to end in one worker process. The proposer is built in
  that worker, by the arm's factory, because an LLM client cannot be pickled, and so every LLM
  call ID of a cell belongs to one process. With `jobs=1` the cells run in this process.
- A log is written atomically, so a run that is killed leaves no partial log.
- A rerun skips each cell whose log exists and was produced under the arm's config. A log or a
  manifest that disagrees with the arm stops the run before any cell starts: the out dir holds
  one arm, never a mix.
- A cell that raises is recorded as failed, and the others go on. It leaves no log, so the next
  run retries it.
- The manifest records the arm, its hash and the model ID once, and every invocation as an entry
  of `runs`: git SHA, package versions, command line, worlds, seeds, and the outcome of each
  cell it ran. A resumed run adds an entry, so cells run under another commit stay visible.
"""

import fcntl
import hashlib
import json
import os
import pickle
import platform
import re
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from mda.bench import canonical_worlds, list_worlds
from mda.config import MDAConfig
from mda.propose import NullProposer, Proposer
from mda.run import RunLog, run_cell

MANIFEST = "manifest.json"
MANIFEST_FORMAT = 1  # version of the manifest format; raise it when a field changes meaning
_LOCK = ".lock"  # held by the run using an out dir
# An arm name labels its out dir and starts its LLM call IDs, so it is one call-ID segment
_ARM_NAME = re.compile(r"[A-Za-z0-9_.-]+")
_ARM_KEYS = {"description", "proposer", "config"}
# BLAS thread pools, set to one thread in the workers. A cell's matrices are too small for BLAS
# to use more: 24 cells in parallel took the same time either way. It saves each worker the 94
# idle threads of numpy's and scipy's two OpenBLAS pools on a 48-core machine
_BLAS_THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

type Cell = tuple[str, int]  # (world, seed)


class GridError(RuntimeError):
    """The out dir cannot be used for this run: it holds another arm's results, or is in use."""


@dataclass(frozen=True)
class CellContext:
    """What a proposer factory knows about the cell it builds a proposer for."""

    arm: str  # the arm's name
    world: str
    seed: int
    config: MDAConfig
    out_dir: Path  # the arm's out dir, which holds this cell's log
    cache_dir: Path  # the LLM reply cache, shared by every cell and arm (D17)

    @property
    def call_prefix(self) -> str:
        """The start of the cell's LLM call IDs: `<arm>/<world>/s<seed>` (D17)."""
        return f"{self.arm}/{self.world}/s{self.seed}"

    @property
    def log_path(self) -> Path:
        return cell_path(self.out_dir, self.world, self.seed)

    @property
    def transcript(self) -> Path:
        """The cell's LLM transcript, next to its log. A retried cell appends to it."""
        return self.out_dir / self.world / f"s{self.seed}.transcript.jsonl"


type ProposerFactory = Callable[[CellContext], Proposer]


def _null(ctx: CellContext) -> Proposer:
    return NullProposer()


PROPOSERS: dict[str, ProposerFactory] = {"null": _null}
"""The proposer kinds an arm file can name, each with the factory that builds it for one cell.

The factory runs in the worker process that runs the cell. It is sent there by reference, so for
`jobs > 1` it must be a module-level function of an importable module: anything else raises
`GridError` before a cell starts.
"""


@dataclass(frozen=True)
class Arm:
    """One arm of the experiment: a proposer kind and the config of every cell."""

    name: str  # labels the out dir and starts the arm's LLM call IDs
    proposer: str  # a key of `PROPOSERS`
    config: MDAConfig
    description: str = ""

    def __post_init__(self) -> None:
        if not _ARM_NAME.fullmatch(self.name) or self.name in (".", ".."):
            raise ValueError(
                f"invalid arm name {self.name!r}: use letters, digits, `_`, `-` and `.`"
            )
        if self.proposer not in PROPOSERS:
            raise ValueError(
                f"unknown proposer kind {self.proposer!r}; known kinds: {sorted(PROPOSERS)}"
            )

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "Arm":
        """The arm in a JSON file, named after the file's stem."""
        path = Path(path)
        try:
            d = json.loads(path.read_text())
            if not isinstance(d, dict):
                raise ValueError("an arm file holds a JSON object")
            unknown = set(d) - _ARM_KEYS
            if unknown:
                raise ValueError(f"unknown keys {sorted(unknown)}; an arm has {sorted(_ARM_KEYS)}")
            if not isinstance(d.get("proposer"), str):
                raise ValueError('"proposer" must name a proposer kind')
            overrides = d.get("config", {})
            if not isinstance(overrides, dict):
                raise ValueError('"config" must be an object of MDAConfig fields')
            try:
                config = MDAConfig.from_dict(overrides)
            except TypeError as e:  # an unknown field
                raise ValueError(str(e)) from e
            description = d.get("description", "")
            if not isinstance(description, str):
                raise ValueError('"description" must be a string')
            return cls(path.stem, d["proposer"], config, description)
        except ValueError as e:  # including invalid JSON and invalid config values
            raise ValueError(f"{path}: {e}") from e

    def identity(self) -> dict[str, Any]:
        """What makes two arms the same: the proposer kind and the full config."""
        return {"proposer": self.proposer, "config": self.config.to_dict()}

    @property
    def digest(self) -> str:
        """The SHA-256 of `identity()` as canonical JSON. The name and description are not in it."""
        text = json.dumps(self.identity(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class GridResult:
    """What one `run_grid` call did with each cell it was given."""

    done: tuple[Cell, ...]  # run, and the log written
    skipped: tuple[Cell, ...]  # already done by an earlier run
    failed: dict[Cell, str]  # each failed cell with its traceback

    @property
    def ok(self) -> bool:
        return not self.failed


def cell_path(out_dir: str | os.PathLike[str], world: str, seed: int) -> Path:
    """Where the log of a cell is written: `<out>/<world>/s<seed>.json`."""
    return Path(out_dir) / world / f"s{seed}.json"


def select_worlds(spec: str) -> list[str]:
    """The worlds named by `spec`, in benchmark order for a group and in the given order otherwise.

    `spec` is `all` (the 57 benchmark worlds), `canonical` (the 9 whose forms are the seed library),
    `compound` (the other 48), or a comma-separated list of world IDs. An ID can be shortened to
    its number, such as `c0` for `c0_michaelis_menten`. Repeats are dropped.
    """
    groups = {
        "all": list_worlds(),
        "canonical": canonical_worlds(),
        "compound": [w for w in list_worlds() if w not in canonical_worlds()],
    }
    if spec in groups:
        return groups[spec]
    worlds: list[str] = []
    for token in (t.strip() for t in spec.split(",")):
        match = [w for w in list_worlds() if token in (w, w.split("_")[0])]
        if not match:
            raise ValueError(f"unknown world {token!r}: use all, canonical, compound or world IDs")
        if match[0] not in worlds:
            worlds.append(match[0])
    return worlds


def run_grid(
    arm: Arm,
    worlds: Sequence[str],
    seeds: Sequence[int],
    *,
    out_dir: str | os.PathLike[str],
    cache_dir: str | os.PathLike[str] = "cache",
    jobs: int = 1,
    command: Sequence[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> GridResult:
    """Run every (world, seed) cell of `arm` that `out_dir` does not hold yet.

    Cells run on `jobs` worker processes, or in this process when `jobs` is 1. Each log is
    written to `cell_path(out_dir, world, seed)` as `RunLog.to_json` returns it, so it is the same
    at any `jobs`. A cell that raises is recorded in the result and the manifest, and leaves no
    log. `command` is recorded in the manifest (default `sys.argv`), and `progress` is called with
    a line of text as each cell ends.

    The workers are spawned, and a spawned worker imports the main module. A script that calls
    this with `jobs > 1` must therefore do so under `if __name__ == "__main__":`, or every worker
    starts the grid again and fails on the out dir's lock.

    Raises `ValueError` for an unknown world, a negative seed or `jobs < 1`, and `GridError` when
    the out dir holds a manifest or a log of another arm, a log that cannot be read, or is in use
    by another run. Either is raised before any cell starts.
    """
    unknown = [w for w in worlds if w not in list_worlds()]
    if unknown:
        raise ValueError(f"unknown worlds {unknown}")
    if any(s < 0 for s in seeds):
        raise ValueError(f"seeds must be non-negative, got {list(seeds)}")
    if jobs < 1:
        raise ValueError(f"jobs must be at least 1, got {jobs}")
    factory = PROPOSERS[arm.proposer]
    if jobs > 1:
        try:
            pickle.dumps(factory)
        except Exception as e:
            raise GridError(
                f"the factory of proposer kind {arm.proposer!r} cannot be sent to a worker "
                f"process; make it a module-level function ({e})"
            ) from e
    out, cache = Path(out_dir), Path(cache_dir)
    say = progress if progress is not None else (lambda _: None)
    cells = [(w, s) for w in dict.fromkeys(worlds) for s in dict.fromkeys(seeds)]

    out.mkdir(parents=True, exist_ok=True)
    with _locked(out):
        manifest = _read_manifest(out, arm)
        todo, skipped = _plan(out, arm, cells)
        run: dict[str, Any] = {
            "arm": arm.name,
            "started": _now(),
            "finished": None,
            "git": _git_state(),
            "versions": _versions(),
            "command": list(sys.argv if command is None else command),
            "worlds": list(dict.fromkeys(worlds)),
            "seeds": list(dict.fromkeys(seeds)),
            "jobs": jobs,
            "cache_dir": str(cache),
            "skipped": len(skipped),
            "cells": {},
        }
        manifest["runs"].append(run)
        _write_atomic(out / MANIFEST, json.dumps(manifest, indent=2) + "\n")

        contexts = [CellContext(arm.name, w, s, arm.config, out, cache) for w, s in todo]
        done: list[Cell] = []
        failed: dict[Cell, str] = {}
        say(
            f"arm {arm.name}: {len(cells)} cells, {len(skipped)} already done, {len(todo)} to run"
            + (f" on {min(jobs, len(todo))} workers" if jobs > 1 and todo else "")
        )
        try:
            with closing(_outcomes(factory, contexts, jobs)) as outcomes:
                for outcome in outcomes:
                    cell = (outcome.world, outcome.seed)
                    entry: dict[str, Any] = {"seconds": round(outcome.seconds, 2)}
                    if outcome.error is None:
                        done.append(cell)
                        entry["status"] = "done"
                        status = f"done in {outcome.seconds:.1f} s"
                    else:
                        failed[cell] = outcome.error
                        entry |= {"status": "failed", "error": outcome.error}
                        status = "FAILED: " + outcome.error.strip().splitlines()[-1]
                    run["cells"][f"{cell[0]}/s{cell[1]}"] = entry
                    _write_atomic(out / MANIFEST, json.dumps(manifest, indent=2) + "\n")
                    n = len(done) + len(failed)
                    say(f"[{n}/{len(todo)}] {cell[0]} s{cell[1]}: {status}")
        finally:
            run["finished"] = _now()
            _write_atomic(out / MANIFEST, json.dumps(manifest, indent=2) + "\n")
    return GridResult(tuple(done), tuple(skipped), failed)


def _plan(out: Path, arm: Arm, cells: list[Cell]) -> tuple[list[Cell], list[Cell]]:
    """The cells to run and the cells already done. Raises `GridError` if a log disagrees."""
    todo, skipped, problems = [], [], []
    expected = arm.config.to_dict()
    for world, seed in cells:
        path = cell_path(out, world, seed)
        if not path.exists():
            todo.append((world, seed))
            continue
        try:
            log = RunLog.from_json(path.read_text())
        except (ValueError, KeyError, TypeError) as e:
            problems.append(f"{path}: cannot be read ({e!r})")
            continue
        if (log.world, log.seed) != (world, seed):
            problems.append(f"{path}: holds the cell ({log.world}, {log.seed})")
        elif log.config != expected:
            problems.append(
                f"{path}: written under a config that differs in {_diff(log.config, expected)}"
            )
        else:
            skipped.append((world, seed))
    if problems:
        raise GridError(
            f"{out} holds logs that are not of arm {arm.name!r} as configured now; move or delete "
            "them, or choose another out dir:\n  " + "\n  ".join(problems)
        )
    return todo, skipped


@dataclass(frozen=True)
class _Outcome:
    world: str
    seed: int
    seconds: float
    error: str | None  # the traceback of a failed cell


def _run_task(factory: ProposerFactory, ctx: CellContext) -> _Outcome:
    """Run one cell and write its log; in a worker process, or in this one when `jobs` is 1."""
    start = time.perf_counter()
    try:
        log = run_cell(ctx.world, ctx.seed, ctx.config, factory(ctx))
        _write_atomic(ctx.log_path, log.to_json())
    except Exception:  # anything but an interrupt: the other cells go on
        return _Outcome(ctx.world, ctx.seed, time.perf_counter() - start, traceback.format_exc())
    return _Outcome(ctx.world, ctx.seed, time.perf_counter() - start, None)


def _outcomes(
    factory: ProposerFactory, contexts: list[CellContext], jobs: int
) -> Generator[_Outcome]:
    """Run the cells, yielding each outcome as the cell ends."""
    if jobs == 1 or not contexts:
        for ctx in contexts:
            yield _run_task(factory, ctx)
        return
    # Spawned workers start from a fresh interpreter rather than a copy of this one, whatever its
    # threads and imports; they inherit the environment in force when they start, which they all
    # do as the cells are submitted
    with (
        _single_threaded_blas(),
        ProcessPoolExecutor(min(jobs, len(contexts)), mp_context=get_context("spawn")) as pool,
    ):
        futures = {pool.submit(_run_task, factory, ctx): ctx for ctx in contexts}
        try:
            for future in as_completed(futures):
                ctx = futures[future]
                try:
                    yield future.result()
                except Exception:  # the worker died, taking the cells it had not finished
                    yield _Outcome(ctx.world, ctx.seed, 0.0, traceback.format_exc())
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise


@contextmanager
def _single_threaded_blas() -> Generator[None]:
    """Set the BLAS thread counts to 1 for processes started in the block, unless already set."""
    saved = {k: os.environ.get(k) for k in _BLAS_THREADS}
    for k in _BLAS_THREADS:
        os.environ.setdefault(k, "1")
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextmanager
def _locked(out: Path) -> Generator[None]:
    """Hold the out dir's lock: two runs on one out dir would run, and call the LLM for, the same
    cells twice."""
    with (out / _LOCK).open("w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise GridError(f"{out} is in use by another run") from e
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _read_manifest(out: Path, arm: Arm) -> dict[str, Any]:
    """The out dir's manifest, or a new one for `arm`. Raises `GridError` if it is another arm's."""
    path = out / MANIFEST
    if not path.exists():
        if any(out.glob("*/s*.json")):
            # A log records its config but not the proposer kind, and two arms can share a
            # config (fixed and hybrid do): only the manifest tells their logs apart
            raise GridError(
                f"{out} holds run logs but no {MANIFEST}, so the arm that wrote them is unknown; "
                "choose another out dir, or move or delete them"
            )
        return {
            "format": MANIFEST_FORMAT,
            "arm_hash": arm.digest,
            "proposer": arm.proposer,
            "model": arm.config.llm_model,
            "config": arm.config.to_dict(),
            "runs": [],
        }
    try:
        manifest = json.loads(path.read_text())
        digest = manifest["arm_hash"]
        if manifest["format"] != MANIFEST_FORMAT or not isinstance(manifest["runs"], list):
            raise ValueError(f"not a manifest of format {MANIFEST_FORMAT}")
    except (ValueError, KeyError, TypeError) as e:
        raise GridError(f"{path} cannot be read ({e!r}); move or delete it") from e
    if digest != arm.digest:
        theirs = manifest.get("config")
        raise GridError(
            f"{out} holds the results of another arm than {arm.name!r}: proposer "
            f"{manifest.get('proposer')!r} against {arm.proposer!r}, config differing in "
            f"{_diff(theirs if isinstance(theirs, dict) else {}, arm.config.to_dict())}; "
            "choose another out dir, or move or delete it"
        )
    return manifest


def _diff(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """The keys whose values differ between two configs."""
    return sorted(k for k in a.keys() | b.keys() if a.get(k, _MISSING) != b.get(k, _MISSING))


_MISSING = object()


def _write_atomic(path: Path, text: str) -> None:
    """Write `text` to `path` by renaming a complete temporary file, so a killed run never leaves
    half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _git_state() -> dict[str, Any]:
    """The commit of the code being run, and whether the working tree differs from it."""
    root = Path(__file__).resolve().parent

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {"sha": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):  # no git, or not a checkout
        return {"sha": None, "dirty": None}


def _versions() -> dict[str, str | None]:
    """The versions of Python, this package and its runtime dependencies."""
    names = ["mda-impl"]
    for requirement in metadata.requires("mda-impl") or []:
        name = re.match(r"[A-Za-z0-9_.-]+", requirement)
        if name is not None and "extra ==" not in requirement:
            names.append(name.group())
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
