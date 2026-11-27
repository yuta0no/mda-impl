"""`tools/llm_check.py` without the API: its two prompts, and its commands on a fake client."""

import importlib.util
import json
from functools import cache
from pathlib import Path
from types import ModuleType

import pytest

from mda.hypothesis import HYPOTHESES_SCHEMA
from mda.llm import FakeLLM, JSONSchema, Reply, Request, StopReason, Usage
from mda.propose import SYSTEM_PROMPT

TOOL = Path(__file__).resolve().parents[1] / "tools" / "llm_check.py"


@cache
def tool() -> ModuleType:
    """The tool, loaded from its file: `tools/` is not a package."""
    spec = importlib.util.spec_from_file_location("llm_check", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@cache
def prompts() -> tuple[str, str]:
    return tool().init_prompt(), tool().expand_prompt()


class CountingLLM(FakeLLM):
    """A fake client whose token count is a function of the request, and which records them."""

    def __init__(self) -> None:
        super().__init__()
        self.counted: list[Request] = []

    def count_tokens(self, system: str, user: str, *, schema: JSONSchema | None = None) -> int:
        self.counted.append(Request(system, user, schema))
        return len(user) // 4 + (0 if schema is None else 425)


def test_prompts_are_the_smallest_and_the_largest_of_a_cell():
    init, expand = prompts()
    # The init call: D0 alone, no refining section, 12 laws
    assert init.count(" -> r0=") == 3
    assert "Forms ALREADY TRIED" not in init and "Experiment budget remaining" not in init
    assert "Propose 12 distinct" in init
    # An expansion after the last experiment: 25 observations, D0 first, and a full pool
    parts = expand.split("\n\n")
    assert len(parts) == 5
    assert parts[1].split("\n")[:4] == init.split("\n\n")[1].split("\n")
    assert expand.count(" -> r0=") == 25
    forms = parts[2].split("\n")[1:-1]
    assert len(forms) == 24 and len({line.split(" -- ")[0] for line in forms}) == 24
    assert [line.endswith("(current best form)") for line in forms] == [True] + [False] * 23
    assert "Experiment budget remaining: 0. " in expand and "Propose 6 distinct" in expand
    # The prompts are the same at every call, so that `call` finds its reply in the cache
    assert (tool().init_prompt(), tool().expand_prompt()) == (init, expand)


def test_count_measures_both_prompts_with_and_without_the_schema(
    capsys: pytest.CaptureFixture[str],
):
    init, expand = prompts()
    llm = CountingLLM()
    tool().count(llm)
    assert llm.counted == [
        Request(SYSTEM_PROMPT, init),
        Request(SYSTEM_PROMPT, init, HYPOTHESES_SCHEMA),
        Request(SYSTEM_PROMPT, expand),
        Request(SYSTEM_PROMPT, expand, HYPOTHESES_SCHEMA),
    ]
    assert llm.requests == []  # counting sends no request
    rows = [line.split()[-3:] for line in capsys.readouterr().out.splitlines()[-2:]]
    assert rows == [
        [str(len(SYSTEM_PROMPT) + len(user)), str(len(user) // 4), str(len(user) // 4 + 425)]
        for user in (init, expand)
    ]


def test_call_sends_one_request_and_replays_it_from_the_cache(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    k = [{"name": "k", "low": 0.001, "high": 1000.0}]
    reply = {
        "hypotheses": [
            {"name": "linear", "expr": "k * C_A", "params": k},
            {"name": "bad", "expr": "k * unknown_input", "params": k},
        ]
    }
    llm = FakeLLM([json.dumps(reply)], cache_dir=tmp_path)
    tool().call(llm)
    assert llm.requests == [Request(SYSTEM_PROMPT, prompts()[0], HYPOTHESES_SCHEMA)]
    assert (llm.n_sent, llm.n_cached) == (1, 1)
    out = capsys.readouterr().out
    assert "first call: cached=False" in out and "second call: cached=True" in out
    assert "1 valid hypotheses, 1 rejected" in out and "  linear: k * C_A" in out


def test_call_stops_on_an_unusable_reply(tmp_path: Path):
    cut = Reply('{"hypotheses": [', StopReason.MAX_TOKENS, Usage(), "fake")
    with pytest.raises(SystemExit, match="unusable reply: stop_reason=max_tokens"):
        tool().call(FakeLLM([cut], cache_dir=tmp_path))
