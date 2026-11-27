from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """The `mda-impl` command (`mda.cli`). Imported when called, so `import mda` stays light."""
    from mda.cli import main as cli_main

    return cli_main(argv)
