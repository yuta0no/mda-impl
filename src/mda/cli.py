"""The `mda-impl` command.

    mda-impl run --arm fixed --worlds all --seeds 0,1,2 --jobs 24

`run` runs one arm over a grid of worlds × seeds with `mda.grid.run_grid`, and resumes where an
earlier run with the same arm stopped. Paths are relative to the current directory, which is
meant to be the repository root: arms are read from `configs/<arm>.json`, logs go to
`runs/<arm>/`, and LLM replies are cached in `cache/`. The API key is read from `.env`.

The exit status is 0 when every cell is done, 1 when a cell failed (rerun to retry it), and 2 for
a usage error or an out dir that holds another arm's results.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from mda.grid import Arm, GridError, run_grid, select_worlds

CONFIGS_DIR = Path("configs")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line `argv` (default `sys.argv[1:]`) and return the exit status."""
    parser = _parser()
    args = parser.parse_args(argv)
    # The variables already set win over those of the file
    load_dotenv(find_dotenv(usecwd=True))
    command = ["mda-impl", *(sys.argv[1:] if argv is None else argv)]  # as typed, for the manifest
    try:
        return args.run(args, command)
    except (ValueError, GridError) as e:
        print(f"mda-impl {args.command}: error: {e}", file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mda-impl", description="Reproduce the MDA ChemBench experiment."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run",
        help="run one arm over worlds x seeds, resuming an earlier run",
        description="Run one arm over worlds x seeds. Cells whose log exists are skipped, so "
        "rerunning the same command resumes the grid and retries the cells that failed.",
    )
    run.add_argument(
        "--arm",
        required=True,
        help="an arm name, read from configs/<arm>.json, or the path of an arm file (*.json)",
    )
    run.add_argument(
        "--worlds",
        default="all",
        help="all (57), canonical (9), compound (48), or comma-separated world IDs, which can be "
        "shortened to their number, such as c0 (default: all)",
    )
    run.add_argument(
        "--seeds", default="0,1,2", type=_seeds, help="comma-separated seeds (default: 0,1,2)"
    )
    run.add_argument("--out", type=Path, help="the out dir (default: runs/<arm>)")
    run.add_argument(
        "--cache-dir", type=Path, default=Path("cache"), help="the LLM reply cache (default: cache)"
    )
    run.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="worker processes; each runs whole cells. 1 runs the cells in this process "
        "(default: 1)",
    )
    run.set_defaults(run=_run)
    return parser


def _run(args: argparse.Namespace, command: list[str]) -> int:
    path = Path(args.arm) if args.arm.endswith(".json") else CONFIGS_DIR / f"{args.arm}.json"
    if not path.is_file():
        raise ValueError(f"no arm file {path}")
    arm = Arm.load(path)
    result = run_grid(
        arm,
        select_worlds(args.worlds),
        args.seeds,
        out_dir=args.out if args.out is not None else Path("runs") / arm.name,
        cache_dir=args.cache_dir,
        jobs=args.jobs,
        command=command,
        progress=lambda line: print(line, file=sys.stderr, flush=True),
    )
    summary = f"{len(result.done)} done, {len(result.skipped)} skipped, {len(result.failed)} failed"
    if result.ok:
        print(summary, file=sys.stderr)
        return 0
    for (world, seed), error in result.failed.items():
        print(f"\n{world} s{seed} failed:\n{error}", file=sys.stderr)
    print(f"{summary}; rerun the same command to retry the failed cells", file=sys.stderr)
    return 1


def _seeds(text: str) -> list[int]:
    try:
        seeds = [int(s) for s in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a list of integers: {text!r}") from None
    if any(s < 0 for s in seeds):
        raise argparse.ArgumentTypeError(f"seeds must be non-negative: {text!r}")
    return list(dict.fromkeys(seeds))
