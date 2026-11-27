import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import replace
from functools import cache
from pathlib import Path

import numpy
import pytest
import scipy

import mda
import mda.grid
from mda.bench import canonical_worlds, list_worlds
from mda.config import DEFAULTS, MDAConfig
from mda.grid import (
    MANIFEST,
    PROPOSERS,
    Arm,
    CellContext,
    GridError,
    GridResult,
    cell_path,
    run_grid,
    select_worlds,
)
from mda.llm import LLMError
from mda.propose import CallRecord, Proposal
from mda.run import run_cell

ROOT = Path(__file__).resolve().parents[1]
C0, C1 = "c0_michaelis_menten", "c1_competitive_inhibition"
# The grid only moves logs around, so its cells stop after the initialisation, with one random
# start per fit: a quarter of a second each under the runtime type checks. The seed still
# changes the fits and the query set, so every cell's log is its own
QUICK_FIELDS = {"n_active": 0, "n_starts": 1}
QUICK = replace(DEFAULTS, **QUICK_FIELDS)
ARM = Arm("quick", "null", QUICK)
FAIL = "MDA_TEST_FAIL_CELL"  # names the cell, as "<world>/s<seed>", whose probe raises
# A number of cells: each probe waits until that many probes have been built, so the cells can
# only finish if that many run at once
RENDEZVOUS = "MDA_TEST_RENDEZVOUS"


@cache
def expected(world: str, seed: int) -> str:
    """The log of a cell of `ARM`, as `run_cell` writes it outside the grid."""
    return run_cell(world, seed, QUICK).to_json()


class Probe:
    """A proposer that names its initial call after the cell's call prefix, and raises an
    `LLMError`, as after a network failure, when it is told to fail."""

    def __init__(self, call_prefix: str, fail: bool = False) -> None:
        self.call_prefix = call_prefix
        self.fail = fail

    def propose(self, kind, pool, X, y, *, weights, budget_left) -> Proposal:
        if self.fail:
            raise LLMError("connection reset (raised by a test)")
        if kind != "init":
            return Proposal()
        return Proposal((), (CallRecord(f"{self.call_prefix}/init"),))


def probe(ctx: CellContext) -> Probe:
    """The factory of the proposer kind "probe", which a spawned worker imports from this module.

    It records where it ran, and the context it was given, in `<out>/probes/<world>-s<seed>.json`.
    The cell that `FAIL` names fails, and with `RENDEZVOUS` set, it waits for the other cells."""
    info = {
        "pid": os.getpid(),
        "blas": os.environ.get("OPENBLAS_NUM_THREADS"),
        "context": [ctx.arm, ctx.world, ctx.seed, str(ctx.out_dir), str(ctx.cache_dir)],
        "paths": [str(ctx.log_path), str(ctx.transcript), ctx.call_prefix],
        "config": ctx.config.to_dict(),
    }
    path = ctx.out_dir / "probes" / f"{ctx.world}-s{ctx.seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info))
    deadline = time.monotonic() + 60
    while len(list(path.parent.glob("*.json"))) < int(os.environ.get(RENDEZVOUS, "0")):
        if time.monotonic() > deadline:
            raise TimeoutError(f"no other cell ran while {ctx.world} s{ctx.seed} waited")
        time.sleep(0.02)
    return Probe(ctx.call_prefix, fail=os.environ.get(FAIL) == f"{ctx.world}/s{ctx.seed}")


def probe_info(out: Path, world: str, seed: int) -> dict:
    return json.loads((out / "probes" / f"{world}-s{seed}.json").read_text())


@cache
def expected_probe(world: str, seed: int, arm: str = "probe-arm") -> str:
    """The log of a cell of the probe arm, as `run_cell` writes it outside the grid."""
    return run_cell(world, seed, QUICK, Probe(f"{arm}/{world}/s{seed}")).to_json()


@pytest.fixture
def with_probe(monkeypatch):
    monkeypatch.setitem(PROPOSERS, "probe", probe)


def write_arm(path: Path, proposer: str = "null", **config) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"proposer": proposer, "config": {**QUICK_FIELDS, **config}}))
    return path


def manifest(out: Path) -> dict:
    return json.loads((out / MANIFEST).read_text())


def snapshot(root: Path) -> dict[str, bytes]:
    """Every file under `root` with its contents."""
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


# Arms


def test_every_arm_config_in_configs_loads():
    paths = sorted((ROOT / "configs").glob("*.json"))
    assert "fixed.json" in [p.name for p in paths]
    for path in paths:
        arm = Arm.load(path)
        assert arm.name == path.stem and arm.proposer in PROPOSERS
        assert arm.description
    fixed = Arm.load(ROOT / "configs" / "fixed.json")
    assert (fixed.proposer, fixed.config) == ("null", DEFAULTS)


def test_arm_file_holds_the_fields_that_differ_from_the_defaults(tmp_path):
    path = tmp_path / "small.json"
    path.write_text(json.dumps({"proposer": "null", "config": {"n_starts": 2, "n_e_max": None}}))
    assert Arm.load(path) == Arm("small", "null", replace(DEFAULTS, n_starts=2))
    path.write_text(json.dumps({"proposer": "null", "description": "seeds only"}))
    assert Arm.load(path) == Arm("small", "null", DEFAULTS, "seeds only")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{", "Expecting"),
        ("[]", "JSON object"),
        ('{"config": {}}', "proposer"),
        ('{"proposer": "llm-of-tomorrow"}', "unknown proposer kind"),
        ('{"proposer": "null", "configs": {}}', "unknown keys"),
        ('{"proposer": "null", "config": {"n_start": 2}}', "n_start"),
        ('{"proposer": "null", "config": {"bic": "aic"}}', "bic"),
        ('{"proposer": "null", "config": [1]}', "config"),
        ('{"proposer": "null", "description": 1}', "description"),
    ],
)
def test_invalid_arm_file_is_rejected_with_its_path(tmp_path, content, message):
    path = tmp_path / "bad.json"
    path.write_text(content)
    with pytest.raises(ValueError, match=message) as e:
        Arm.load(path)
    assert str(path) in str(e.value)


@pytest.mark.parametrize("name", ["a/b", "", ".", "..", "x y"])
def test_arm_name_must_be_one_call_id_segment(name):
    with pytest.raises(ValueError, match="arm name"):
        Arm(name, "null", DEFAULTS)


def test_arm_digest_is_the_hash_of_the_proposer_kind_and_the_full_config(with_probe):
    identity = {"proposer": "null", "config": QUICK.to_dict()}
    text = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    assert ARM.identity() == identity
    assert ARM.digest == hashlib.sha256(text.encode()).hexdigest()
    # The name and the description label the arm; they are not part of it
    assert Arm("other", "null", QUICK, "a description").digest == ARM.digest
    for other in (
        Arm("quick", "probe", QUICK),
        Arm("quick", "null", replace(QUICK, n_starts=2)),
        Arm("quick", "null", replace(QUICK, llm_model="claude-sonnet-5-5")),
        Arm("quick", "null", replace(QUICK, n_e_max=0)),
    ):
        assert other.digest != ARM.digest


# Worlds


def test_world_groups():
    assert select_worlds("all") == list_worlds() and len(list_worlds()) == 57
    assert select_worlds("canonical") == canonical_worlds() and len(canonical_worlds()) == 9
    compound = select_worlds("compound")
    assert len(compound) == 48 and not set(compound) & set(canonical_worlds())
    assert sorted(compound + canonical_worlds()) == sorted(list_worlds())


def test_world_list_keeps_its_order_and_accepts_short_ids():
    assert select_worlds(f"c1,{C0}") == [C1, C0]
    assert select_worlds("c93, c0,c0") == ["c93_fractal_competitive", C0]
    assert select_worlds("c10") == ["c10_mm_competitive_arrhenius"]  # not c1


@pytest.mark.parametrize("spec", ["c4", "c4_ph_activity", "michaelis", "c0,", "Canonical"])
def test_unknown_world_is_rejected(spec):
    with pytest.raises(ValueError, match="unknown world"):
        select_worlds(spec)


# The grid


@pytest.fixture(scope="module")
def base(tmp_path_factory) -> Path:
    """An out dir in which `ARM` has run (C0, 0) and (C0, 1). Tests change copies of it."""
    out = tmp_path_factory.mktemp("base") / "quick"
    assert run_grid(ARM, [C0], [0, 1], out_dir=out).done == ((C0, 0), (C0, 1))
    return out


@pytest.fixture
def out(base, tmp_path) -> Path:
    """A copy of the `base` out dir."""
    return Path(shutil.copytree(base, tmp_path / "quick"))


@pytest.fixture(scope="module")
def parallel(tmp_path_factory) -> tuple[Path, GridResult]:
    """The probe arm on 4 cells and 2 worker processes, in which the cell (C1, 0) raises.

    Each probe waits until two have been built, so a cell fails unless two run at once."""
    out = tmp_path_factory.mktemp("parallel") / "probe-arm"
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(PROPOSERS, "probe", probe)
        mp.setenv(FAIL, f"{C1}/s0")
        mp.setenv(RENDEZVOUS, "2")  # the two workers run two cells at once
        mp.delenv("OPENBLAS_NUM_THREADS", raising=False)
        arm = Arm("probe-arm", "probe", QUICK)
        result = run_grid(arm, [C0, C1], [0, 2], out_dir=out, cache_dir=out / "cache", jobs=2)
        assert "OPENBLAS_NUM_THREADS" not in os.environ  # restored once the workers started
    return out, result


def test_grid_logs_equal_run_cell_logs_byte_for_byte_in_this_process(base):
    for world, seed in ((C0, 0), (C0, 1)):
        assert cell_path(base, world, seed).read_text() == expected(world, seed)
    assert expected(C0, 0) != expected(C0, 1)


def test_grid_logs_equal_run_cell_logs_byte_for_byte_in_worker_processes(parallel):
    out, result = parallel
    assert sorted(result.done) == [(C0, 0), (C0, 2), (C1, 2)]
    for world, seed in result.done:
        path = out / world / f"s{seed}.json"
        assert path == cell_path(out, world, seed)
        assert path.read_text() == expected_probe(world, seed)
    # The logs, the probes' records and the manifest, and no temporary file
    assert sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()) == [
        ".lock",
        f"{C0}/s0.json",
        f"{C0}/s2.json",
        f"{C1}/s2.json",
        MANIFEST,
        f"probes/{C0}-s0.json",
        f"probes/{C0}-s2.json",
        f"probes/{C1}-s0.json",
        f"probes/{C1}-s2.json",
    ]


def test_proposer_is_built_in_a_worker_from_the_cells_context(parallel):
    out, _ = parallel
    for world, seed in ((C0, 0), (C0, 2), (C1, 0), (C1, 2)):
        info = probe_info(out, world, seed)
        assert info["pid"] != os.getpid()
        assert info["context"] == ["probe-arm", world, seed, str(out), str(out / "cache")]
        assert info["paths"] == [
            str(out / world / f"s{seed}.json"),
            str(out / world / f"s{seed}.transcript.jsonl"),
            f"probe-arm/{world}/s{seed}",
        ]
        assert info["config"] == QUICK.to_dict()
        assert info["blas"] == "1"  # single-threaded BLAS in the workers
    pids = {probe_info(out, w, s)["pid"] for w, s in ((C0, 0), (C0, 2), (C1, 0), (C1, 2))}
    assert 1 <= len(pids) <= 2  # at most `jobs` workers


def test_proposer_is_built_for_each_cell_in_this_process_with_one_job(tmp_path, with_probe):
    arm = Arm("probe-arm", "probe", QUICK)
    run_grid(arm, [C0], [0, 1], out_dir=tmp_path, cache_dir=tmp_path / "c", jobs=1)
    for seed in (0, 1):
        assert cell_path(tmp_path, C0, seed).read_text() == expected_probe(C0, seed)
        info = probe_info(tmp_path, C0, seed)
        assert info["pid"] == os.getpid()
        assert info["context"] == ["probe-arm", C0, seed, str(tmp_path), str(tmp_path / "c")]


def test_failing_cell_is_isolated_recorded_and_retried(parallel, with_probe, tmp_path):
    out, result = parallel
    assert list(result.failed) == [(C1, 0)] and not result.ok
    assert "Traceback" in result.failed[(C1, 0)]
    assert "LLMError: connection reset" in result.failed[(C1, 0)]
    assert not cell_path(out, C1, 0).exists()
    run = manifest(out)["runs"][-1]
    assert run["cells"][f"{C1}/s0"]["status"] == "failed"
    assert run["cells"][f"{C1}/s0"]["error"] == result.failed[(C1, 0)]
    assert {k: v["status"] for k, v in run["cells"].items() if k != f"{C1}/s0"} == {
        f"{C0}/s0": "done",
        f"{C0}/s2": "done",
        f"{C1}/s2": "done",
    }
    # The next run retries the failed cell, and only it, here on a copy of the out dir
    copy = Path(shutil.copytree(out, tmp_path / "probe-arm"))
    before = snapshot(copy)
    retry = run_grid(Arm("probe-arm", "probe", QUICK), [C0, C1], [0, 2], out_dir=copy)
    assert retry == GridResult(((C1, 0),), ((C0, 0), (C0, 2), (C1, 2)), {})
    assert cell_path(copy, C1, 0).read_text() == expected_probe(C1, 0)
    after = snapshot(copy)
    for name in (f"{C0}/s0.json", f"{C0}/s2.json", f"{C1}/s2.json"):
        assert after[name] == before[name]


def test_failure_in_this_process_does_not_stop_the_other_cells(tmp_path, with_probe, monkeypatch):
    monkeypatch.setenv(FAIL, f"{C0}/s0")
    result = run_grid(Arm("p", "probe", QUICK), [C0], [0, 1], out_dir=tmp_path)
    assert (result.done, list(result.failed)) == (((C0, 1),), [(C0, 0)])
    assert not cell_path(tmp_path, C0, 0).exists() and cell_path(tmp_path, C0, 1).exists()


def test_resume_skips_completed_cells(out, monkeypatch):
    before = snapshot(out)
    ran = []

    def spy(world, seed, config, proposer):
        ran.append((world, seed))
        return run_cell(world, seed, config, proposer)

    monkeypatch.setattr(mda.grid, "run_cell", spy)
    result = run_grid(ARM, [C0, C1], [0, 1], out_dir=out)
    assert ran == [(C1, 0), (C1, 1)]
    assert result == GridResult(((C1, 0), (C1, 1)), ((C0, 0), (C0, 1)), {})
    after = snapshot(out)
    assert all(after[f"{C0}/s{s}.json"] == before[f"{C0}/s{s}.json"] for s in (0, 1))
    assert after[f"{C1}/s1.json"].decode() == expected(C1, 1)
    # Nothing left to run
    ran.clear()
    again = run_grid(ARM, [C1, C0], [1, 0], out_dir=out)
    assert ran == [] and again == GridResult((), ((C1, 1), (C1, 0), (C0, 1), (C0, 0)), {})


def test_resume_under_another_arm_name_with_the_same_identity(out):
    renamed = Arm("renamed", "null", QUICK, "the same arm")
    assert run_grid(renamed, [C0], [0, 1], out_dir=out).skipped == ((C0, 0), (C0, 1))
    assert [r["arm"] for r in manifest(out)["runs"]] == ["quick", "renamed"]


def corrupt_other_cell(out: Path) -> None:
    (out / C0 / "s1.json").write_text((out / C0 / "s0.json").read_text())


def corrupt_truncate(out: Path) -> None:
    text = (out / C0 / "s0.json").read_text()
    (out / C0 / "s0.json").write_text(text[: len(text) // 2])


def corrupt_schema(out: Path) -> None:
    d = json.loads((out / C0 / "s0.json").read_text())
    d["schema"] = 2
    (out / C0 / "s0.json").write_text(json.dumps(d))


def corrupt_config(out: Path) -> None:
    d = json.loads((out / C0 / "s0.json").read_text())
    d["config"]["n_evals"] = 10
    (out / C0 / "s0.json").write_text(json.dumps(d))


def corrupt_manifest(out: Path) -> None:
    (out / MANIFEST).write_text("{")


def lose_manifest(out: Path) -> None:
    # The logs alone cannot say which proposer wrote them
    (out / MANIFEST).unlink()


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (corrupt_other_cell, f"holds the cell \\({C0}, 0\\)"),
        (corrupt_truncate, "cannot be read"),
        (corrupt_schema, "cannot be read"),
        (corrupt_config, r"differs in \['n_evals'\]"),
        (corrupt_manifest, "manifest.json cannot be read"),
        (lose_manifest, "holds run logs but no manifest.json"),
    ],
)
def test_out_dir_that_disagrees_is_refused_before_any_cell_runs(out, corrupt, message):
    corrupt(out)
    before = snapshot(out)
    with pytest.raises(GridError, match=message):
        run_grid(ARM, [C0, C1], [0, 1], out_dir=out)
    assert snapshot(out) == before


@pytest.mark.parametrize(
    ("proposer", "config", "message"),
    [
        ("null", replace(QUICK, n_starts=2), r"config differing in \['n_starts'\]"),
        ("probe", QUICK, r"proposer 'null' against 'probe', config differing in \[\]"),
    ],
)
def test_another_arm_is_refused_in_the_out_dir(out, with_probe, proposer, config, message):
    before = snapshot(out)
    with pytest.raises(GridError, match=message):
        run_grid(Arm("quick", proposer, config), [C1], [0], out_dir=out)
    assert snapshot(out) == before


def test_logs_are_written_atomically(out, monkeypatch):
    # A run killed between writing a log and publishing it leaves no log, and no temporary file
    replace_ = os.replace
    published = []

    def killed(src, dst):
        if Path(dst).name == "s1.json":
            # The complete text is in a temporary file in the log's directory
            assert Path(src).parent == Path(dst).parent and not Path(dst).exists()
            assert Path(src).read_text() == expected(C1, 1)
            raise KeyboardInterrupt
        published.append(str(Path(dst).relative_to(out)))
        replace_(src, dst)

    monkeypatch.setattr(os, "replace", killed)
    with pytest.raises(KeyboardInterrupt):
        run_grid(ARM, [C1], [0, 1], out_dir=out)
    assert f"{C1}/s0.json" in published
    assert sorted(p.name for p in (out / C1).iterdir()) == ["s0.json"]
    run = manifest(out)["runs"][-1]
    assert list(run["cells"]) == [f"{C1}/s0"] and run["finished"] is not None
    # The next run runs it
    monkeypatch.setattr(os, "replace", replace_)
    assert run_grid(ARM, [C1], [0, 1], out_dir=out).done == ((C1, 1),)


def test_out_dir_in_use_is_refused(out):
    with (out / ".lock").open("w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        with pytest.raises(GridError, match="in use"):
            run_grid(ARM, [C1], [0], out_dir=out)
        assert not cell_path(out, C1, 0).exists()
    assert run_grid(ARM, [C1], [0], out_dir=out).done == ((C1, 0),)


def test_factory_that_cannot_reach_a_worker_is_refused(tmp_path, monkeypatch):
    monkeypatch.setitem(PROPOSERS, "local", lambda ctx: Probe(ctx.call_prefix))
    arm = Arm("local", "local", QUICK)
    with pytest.raises(GridError, match="module-level function"):
        run_grid(arm, [C0], [0, 1], out_dir=tmp_path, jobs=2)
    assert not tmp_path.joinpath(MANIFEST).exists()
    assert run_grid(arm, [C0], [0], out_dir=tmp_path, jobs=1).ok  # in this process it is fine


@pytest.mark.parametrize(
    ("worlds", "seeds", "jobs", "message"),
    [
        (["c4_ph_activity"], [0], 1, "unknown worlds"),
        ([C0], [-1], 1, "non-negative"),
        ([C0], [0], 0, "at least 1"),
    ],
)
def test_invalid_grid_is_rejected_before_anything_is_written(
    tmp_path, worlds, seeds, jobs, message
):
    with pytest.raises(ValueError, match=message):
        run_grid(ARM, worlds, seeds, out_dir=tmp_path / "out", jobs=jobs)
    assert not (tmp_path / "out").exists()


# The manifest


def test_manifest_records_the_arm_and_each_run(tmp_path, monkeypatch):
    run_grid(ARM, [C1, C0], [1], out_dir=tmp_path, cache_dir="somewhere", command=["a", "b"])
    first = manifest(tmp_path)
    assert {k: v for k, v in first.items() if k != "runs"} == {
        "format": 1,
        "arm_hash": ARM.digest,
        "proposer": "null",
        "model": "claude-haiku-5-5",
        "config": QUICK.to_dict(),
    }
    (run,) = first["runs"]
    assert run["git"] == {
        "sha": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }
    versions = run["versions"]
    assert (versions["numpy"], versions["scipy"]) == (numpy.__version__, scipy.__version__)
    assert versions["mda-impl"] == "0.1.0" and versions["python"].startswith("3.")
    assert {"anthropic", "cma", "jaxtyping", "python-dotenv"} <= versions.keys()
    assert (run["arm"], run["command"]) == ("quick", ["a", "b"])
    assert (run["worlds"], run["seeds"]) == ([C1, C0], [1])
    assert (run["jobs"], run["cache_dir"], run["skipped"]) == (1, "somewhere", 0)
    assert run["started"] <= run["finished"]
    statuses = {k: v["status"] for k, v in run["cells"].items()}
    assert statuses == {f"{C1}/s1": "done", f"{C0}/s1": "done"}
    assert all(v["seconds"] > 0 for v in run["cells"].values())

    # A resumed run under another commit adds its own entry and keeps the first
    monkeypatch.setattr(mda.grid, "_git_state", lambda: {"sha": "f" * 40, "dirty": False})
    run_grid(ARM, [C0], [1, 2], out_dir=tmp_path, command=["c"])
    second = manifest(tmp_path)
    assert second["runs"][0] == run and len(second["runs"]) == 2
    assert {k: v for k, v in second.items() if k != "runs"} == {
        k: v for k, v in first.items() if k != "runs"
    }
    later = second["runs"][1]
    assert later["git"] == {"sha": "f" * 40, "dirty": False}
    assert (later["command"], later["seeds"], later["skipped"]) == (["c"], [1, 2], 1)
    assert list(later["cells"]) == [f"{C0}/s2"]


def test_manifest_records_the_model_and_the_jobs(tmp_path):
    arm = Arm("m", "null", replace(QUICK, llm_model="claude-sonnet-5-5"))
    run_grid(arm, [C0], [0], out_dir=tmp_path, jobs=3)
    assert manifest(tmp_path)["model"] == "claude-sonnet-5-5"
    assert manifest(tmp_path)["runs"][0]["jobs"] == 3


def test_manifest_command_defaults_to_the_process_arguments(out, monkeypatch):
    monkeypatch.setattr("sys.argv", ["mda-impl", "run", "--arm", "quick"])
    run_grid(ARM, [C0], [0], out_dir=out)
    assert manifest(out)["runs"][-1]["command"] == ["mda-impl", "run", "--arm", "quick"]


# The command line


def test_cli_runs_a_two_cell_grid(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    write_arm(tmp_path / "configs" / "quick.json")
    argv = ["run", "--arm", "quick", "--worlds", "c0,c1", "--seeds", "1"]
    assert mda.main(argv) == 0
    out = tmp_path / "runs" / "quick"  # the default out dir
    for world in (C0, C1):
        assert cell_path(out, world, 1).read_text() == expected(world, 1)
    run = manifest(out)["runs"][0]
    assert (run["command"], run["worlds"], run["seeds"]) == (["mda-impl", *argv], [C0, C1], [1])
    assert (run["jobs"], run["cache_dir"]) == (1, "cache")
    assert "2 done, 0 skipped, 0 failed" in capsys.readouterr().err
    # Run again, it resumes: nothing left to do
    assert mda.main(argv) == 0
    assert "0 done, 2 skipped, 0 failed" in capsys.readouterr().err


def test_cli_takes_an_arm_file_and_the_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    arm = write_arm(tmp_path / "elsewhere" / "mine.json")
    argv = ["run", "--arm", str(arm), "--worlds", "c0", "--seeds", "0"]
    assert mda.main([*argv, "--out", "o", "--cache-dir", "k"]) == 0
    assert cell_path(tmp_path / "o", C0, 0).read_text() == expected(C0, 0)
    run = manifest(tmp_path / "o")["runs"][0]
    assert (run["arm"], run["cache_dir"]) == ("mine", "k")


def test_cli_exits_1_while_a_cell_fails(tmp_path, monkeypatch, capsys, with_probe):
    monkeypatch.chdir(tmp_path)
    write_arm(tmp_path / "configs" / "p.json", proposer="probe")
    argv = ["run", "--arm", "p", "--worlds", "c0", "--seeds", "0,1"]
    monkeypatch.setenv(FAIL, f"{C0}/s1")
    assert mda.main(argv) == 1
    err = capsys.readouterr().err
    assert f"{C0} s1 failed" in err and "LLMError" in err and "1 done, 0 skipped, 1 failed" in err
    monkeypatch.delenv(FAIL)
    assert mda.main(argv) == 0
    assert "1 done, 1 skipped, 0 failed" in capsys.readouterr().err


def test_cli_refuses_an_out_dir_of_another_arm(base, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    shutil.copytree(base, tmp_path / "runs" / "quick")
    write_arm(tmp_path / "configs" / "quick.json", n_starts=3)
    assert mda.main(["run", "--arm", "quick", "--worlds", "c1", "--seeds", "0"]) == 2
    assert "holds the results of another arm" in capsys.readouterr().err
    assert not cell_path(tmp_path / "runs" / "quick", C1, 0).exists()


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["run", "--arm", "nowhere"], "no arm file configs/nowhere.json"),
        (["run", "--arm", "quick", "--worlds", "c4"], "unknown world 'c4'"),
        (["run", "--arm", "quick", "--jobs", "0"], "at least 1"),
    ],
)
def test_cli_reports_errors_with_status_2(tmp_path, monkeypatch, capsys, argv, message):
    monkeypatch.chdir(tmp_path)
    write_arm(tmp_path / "configs" / "quick.json")
    assert mda.main(argv) == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["run"],
        ["run", "--arm", "quick", "--seeds", "0,a"],
        ["run", "--arm", "x", "--seeds", "-1"],
    ],
)
def test_cli_usage_errors_exit_2(argv):
    with pytest.raises(SystemExit) as e:
        mda.main(argv)
    assert e.value.code == 2


def test_cli_parses_seeds_and_worlds(tmp_path, monkeypatch):
    seen = {}

    def fake_run_grid(arm, worlds, seeds, **kwargs):
        seen.update(arm=arm, worlds=worlds, seeds=seeds, **kwargs)
        return GridResult((), (), {})

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("mda.cli.run_grid", fake_run_grid)
    write_arm(tmp_path / "configs" / "quick.json")
    assert mda.main(["run", "--arm", "quick", "--worlds", "canonical", "--seeds", "2,0,2"]) == 0
    assert (seen["worlds"], seen["seeds"], seen["jobs"]) == (canonical_worlds(), [2, 0], 1)
    assert (seen["out_dir"], seen["cache_dir"]) == (Path("runs/quick"), Path("cache"))
    assert seen["arm"] == Arm("quick", "null", QUICK)
    assert mda.main(["run", "--arm", "quick", "--jobs", "5"]) == 0
    assert (seen["worlds"], seen["seeds"], seen["jobs"]) == (list_worlds(), [0, 1, 2], 5)


def test_cli_loads_the_env_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MDA_TEST_FROM_ENV_FILE", raising=False)
    monkeypatch.setattr("mda.cli.run_grid", lambda *a, **k: GridResult((), (), {}))
    (tmp_path / ".env").write_text('MDA_TEST_FROM_ENV_FILE="yes"\n')
    write_arm(tmp_path / "configs" / "quick.json")
    try:
        assert mda.main(["run", "--arm", "quick"]) == 0
        assert os.environ["MDA_TEST_FROM_ENV_FILE"] == "yes"
    finally:
        os.environ.pop("MDA_TEST_FROM_ENV_FILE", None)


def test_config_round_trips_through_an_arm_file(tmp_path):
    # An arm file can also hold a whole config, as `to_dict` writes it
    path = tmp_path / "frozen.json"
    config = replace(QUICK, passive_c_a=(0.1, 1.0), n_e_max=4)
    path.write_text(json.dumps({"proposer": "null", "config": config.to_dict()}))
    assert Arm.load(path).config == config == MDAConfig.from_dict(config.to_dict())
