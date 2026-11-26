import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.types import Message

from mda.config import DEFAULTS
from mda.hypothesis import HYPOTHESES_SCHEMA, parse_hypotheses
from mda.llm import (
    FREE,
    PRICING,
    ClaudeLLM,
    FakeLLM,
    LLMClient,
    LLMError,
    Pricing,
    Rates,
    Reply,
    ReplyCache,
    Request,
    StopReason,
    Usage,
)

HAIKU = PRICING["claude-haiku-5-5"]
REPLY_JSON = (
    '{"hypotheses": [{"name": "mm", "expr": "Vmax * C_A / (Km + C_A)", '
    '"params": [{"name": "Vmax", "low": 0.001, "high": 1000}, '
    '{"name": "Km", "low": 0.001, "high": 1000}]}]}'
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


# --- Fake Anthropic API ---------------------------------------------------------------------------


class FakeAPI:
    """Stands in for the Anthropic API behind an `httpx2.MockTransport`, recording each request.

    Each request takes the next of `responses`; an exception among them is raised, as a network
    failure would be.
    """

    def __init__(self, *responses: httpx2.Response | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append({"path": request.url.path, "body": json.loads(request.content)})
        if not self.responses:
            raise AssertionError("unexpected request to the API")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def client(self) -> anthropic.Anthropic:
        return anthropic.Anthropic(
            api_key="test-key",
            # Nothing is sent there: the mock transport answers every request. The SDK still builds
            # each request's URL from the base URL, and the tests read the path from it, so it is
            # fixed here and an ANTHROPIC_BASE_URL in the environment cannot change it
            base_url="https://api.anthropic.com",
            max_retries=0,
            http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(self)),
        )


# The model ID in the fake API's replies. It differs from the alias requested, to show which of the
# two a `Reply` reports
REPLY_MODEL = "claude-haiku-5-5-20261001"


def message(
    content: list[dict[str, Any]] | None = None,
    stop_reason: str | None = "end_turn",
    stop_details: dict[str, Any] | None = None,
    usage: dict[str, Any] | None = None,
    request_id: str | None = "req_test",
) -> httpx2.Response:
    body = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": REPLY_MODEL,
        "content": content if content is not None else [{"type": "text", "text": REPLY_JSON}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "stop_details": stop_details,
        "usage": usage or {"input_tokens": 1000, "output_tokens": 2000},
    }
    # These replies are written by hand, so check them against the SDK's own response type: a
    # field of the wrong type or an unknown stop reason fails here, not silently in a test
    Message.model_validate(body, strict=True)
    headers = {"request-id": request_id} if request_id is not None else {}
    return httpx2.Response(200, json=body, headers=headers)


def claude(api: FakeAPI, **kwargs: Any) -> ClaudeLLM:
    return ClaudeLLM(client=api.client(), **kwargs)


# --- ClaudeLLM ---------------------------------------------------------------------------------


def test_claude_request_and_reply():
    usage = {
        "input_tokens": 1000,
        "output_tokens": 3000,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    content = [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "text", "text": "part one, "},
        {"type": "redacted_thinking", "data": "opaque"},
        {"type": "text", "text": "part two"},
    ]
    api = FakeAPI(message(content=content, usage=usage))
    llm = claude(api)
    reply = llm.complete("system prompt", "user prompt", schema=HYPOTHESES_SCHEMA)

    [sent] = api.requests
    assert sent["path"] == "/v1/messages"
    # The whole body, so that nothing else is sent: no sampling parameters, which Haiku 5.5 rejects
    # (spec §9), and no `thinking`, which leaves adaptive thinking on
    assert sent["body"] == {
        "model": "claude-haiku-5-5",
        "max_tokens": 16_000,
        "system": "system prompt",
        "messages": [{"role": "user", "content": "user prompt"}],
        "output_config": {
            "effort": "medium",
            "format": {"type": "json_schema", "schema": HYPOTHESES_SCHEMA},
        },
    }

    # Every field of the reply
    assert reply == Reply(
        text="part one, part two",  # only the text blocks, joined; thinking blocks are skipped
        stop_reason=StopReason.END_TURN,
        usage=Usage(input_tokens=1000, output_tokens=3000),
        model=REPLY_MODEL,  # as the API reports it, not the alias requested
        request_id="req_test",  # from the response header, not the message ID
        stop_detail=None,
        cost=HAIKU.cost(Usage(1000, 3000)),
        cached=False,
    )
    assert reply.ok
    assert reply.cost == pytest.approx((1000 * 0.10 + 3000 * 0.50) / 1e6)


def test_claude_without_schema_sends_no_format():
    api = FakeAPI(message())
    claude(api, effort="low", max_tokens=8_000).complete("s", "u")
    assert api.requests[0]["body"]["output_config"] == {"effort": "low"}
    assert api.requests[0]["body"]["max_tokens"] == 8_000


def test_claude_usage_keeps_prompt_cache_tokens_apart():
    usage = {
        "input_tokens": 10,
        "output_tokens": 20,
        "cache_creation_input_tokens": 30,
        "cache_read_input_tokens": 40,
    }
    reply = claude(FakeAPI(message(usage=usage))).complete("s", "u")
    assert reply.usage == Usage(10, 20, cache_write_tokens=30, cache_read_tokens=40)
    assert reply.usage.prompt_tokens == 80


def test_claude_max_tokens_is_surfaced():
    api = FakeAPI(message(content=[{"type": "text", "text": '{"hypo'}], stop_reason="max_tokens"))
    reply = claude(api).complete("s", "u")
    assert reply.stop_reason is StopReason.MAX_TOKENS
    assert not reply.ok
    assert reply.text == '{"hypo'


@pytest.mark.parametrize(
    ("details", "expected"),
    [
        ({"type": "refusal", "category": "bio", "explanation": "declined"}, "bio: declined"),
        # The API can leave out the category, the explanation or the details altogether
        ({"type": "refusal", "category": "bio", "explanation": None}, "bio"),
        ({"type": "refusal", "category": None, "explanation": "declined"}, "declined"),
        ({"type": "refusal", "category": None, "explanation": None}, None),
        (None, None),
    ],
)
def test_claude_refusal_is_surfaced(details: dict[str, Any] | None, expected: str | None):
    api = FakeAPI(message(content=[], stop_reason="refusal", stop_details=details))
    reply = claude(api).complete("s", "u")
    assert reply.stop_reason is StopReason.REFUSAL
    assert not reply.ok
    assert reply.text == ""
    assert reply.stop_detail == expected


def test_claude_other_stop_reason_keeps_the_raw_value():
    # The second reply carries refusal details that do not belong to its stop reason. The API
    # should never send that, but if it did they are not read: details are only read for a refusal
    stray = {"type": "refusal", "category": "bio", "explanation": "declined"}
    api = FakeAPI(
        message(stop_reason="model_context_window_exceeded"),
        message(stop_reason="pause_turn", stop_details=stray),
    )
    llm = claude(api)
    for raw in ("model_context_window_exceeded", "pause_turn"):
        reply = llm.complete("s", "u")
        assert reply.stop_reason is StopReason.OTHER
        assert reply.stop_detail == raw
        assert not reply.ok


def test_claude_reply_without_a_request_id_or_stop_reason():
    # A proxy can drop the request-id header, and the API types allow a null stop reason
    api = FakeAPI(message(stop_reason=None, request_id=None))
    reply = claude(api).complete("s", "u")
    assert reply.request_id is None
    assert (reply.stop_reason, reply.stop_detail, reply.ok) == (StopReason.OTHER, None, False)
    assert reply.text == REPLY_JSON


def test_claude_failed_request_raises_llm_error(tmp_path: Path):
    error = {"type": "error", "error": {"type": "api_error", "message": "boom"}}
    api = FakeAPI(
        httpx2.Response(500, json=error),
        httpx2.ConnectError("unreachable"),
        httpx2.Response(500, json=error),
    )
    transcript = tmp_path / "transcript.jsonl"
    llm = claude(api, transcript=transcript)
    # The error keeps the API's own message and the SDK's exception, for the run log
    with pytest.raises(LLMError, match="request failed: .*500.*boom") as failure:
        llm.complete("s", "u")
    assert isinstance(failure.value.__cause__, anthropic.InternalServerError)
    with pytest.raises(LLMError, match="request failed: Connection error") as failure:
        llm.complete("s", "u")
    assert isinstance(failure.value.__cause__, anthropic.APIConnectionError)
    with pytest.raises(LLMError, match="token count failed: .*500.*boom") as failure:
        llm.count_tokens("s", "u")
    assert isinstance(failure.value.__cause__, anthropic.InternalServerError)
    assert len(api.requests) == 3
    # A failed request is not counted, costs nothing and leaves no transcript line
    assert (llm.n_sent, llm.cost, llm.usage) == (0, 0.0, Usage())
    assert not transcript.exists()


def test_claude_cache_hit_makes_no_network_call(tmp_path: Path):
    api = FakeAPI(message())  # serves exactly one response; a second request fails the test
    llm = claude(api, cache_dir=tmp_path)
    first = llm.complete("s", "u", call_id="hybrid/c0/s0/call00/try1")
    second = llm.complete("s", "u", call_id="hybrid/c0/s0/call00/try1")
    assert len(api.requests) == 1
    assert second.cached and not first.cached
    assert second.text == first.text
    assert second.request_id == first.request_id == "req_test"
    # A cached reply is the original in every field but `cached`: it keeps its usage and cost
    assert second == replace(first, cached=True)
    assert first.cost > 0

    # A new client, as in a rerun, replays from disk
    rerun = claude(FakeAPI(), cache_dir=tmp_path)
    assert rerun.complete("s", "u", call_id="hybrid/c0/s0/call00/try1").cached
    assert rerun.n_sent == 0 and rerun.cost == 0.0


def test_claude_settings_are_part_of_the_cache_key(tmp_path: Path):
    llm = claude(FakeAPI(message()), cache_dir=tmp_path)
    # Exactly these: a key added or renamed here orphans every reply cached so far
    assert llm.settings() == {
        "provider": "anthropic",
        "model": "claude-haiku-5-5",
        "effort": "medium",
        "max_tokens": 16_000,
    }
    llm.complete("s", "u", call_id="a")
    for kwargs in ({"effort": "high"}, {"max_tokens": 8000}):
        other = claude(FakeAPI(message()), cache_dir=tmp_path, **kwargs)
        assert not other.complete("s", "u", call_id="a").cached


def test_claude_count_tokens():
    api = FakeAPI(
        httpx2.Response(200, json={"input_tokens": 1234}),
        httpx2.Response(200, json={"input_tokens": 99}),
    )
    llm = claude(api)
    assert llm.count_tokens("system prompt", "user prompt", schema=HYPOTHESES_SCHEMA) == 1234
    assert llm.count_tokens("system prompt", "user prompt") == 99

    with_schema, plain = api.requests
    assert with_schema["path"] == plain["path"] == "/v1/messages/count_tokens"
    # What `complete` sends, less `max_tokens`, so that the count is of the same prompt
    assert with_schema["body"] == {
        "model": "claude-haiku-5-5",
        "system": "system prompt",
        "messages": [{"role": "user", "content": "user prompt"}],
        "output_config": {
            "effort": "medium",
            "format": {"type": "json_schema", "schema": HYPOTHESES_SCHEMA},
        },
    }
    assert plain["body"]["output_config"] == {"effort": "medium"}
    assert (llm.n_sent, llm.cost) == (0, 0.0)  # counting is free and is not a call


@pytest.mark.parametrize("status", [400, 401, 429, 529])
def test_claude_any_error_status_raises_llm_error(status: int):
    error = {"type": "error", "error": {"type": "some_error", "message": "boom"}}
    api = FakeAPI(httpx2.Response(status, json=error))
    with pytest.raises(LLMError, match=f"request failed: .*{status}.*boom"):
        claude(api).complete("s", "u")
    assert len(api.requests) == 1  # these clients do not retry; the default client does


def test_claude_model_and_pricing_can_be_set():
    api = FakeAPI(message(), httpx2.Response(200, json={"input_tokens": 7}))
    rates = Pricing(Rates(1.0, 2.0, 0.0, 0.0))
    llm = ClaudeLLM("claude-other", client=api.client(), pricing=rates)
    reply = llm.complete("s", "u")
    assert llm.count_tokens("s", "u") == 7
    assert [sent["body"]["model"] for sent in api.requests] == ["claude-other", "claude-other"]
    assert llm.model == llm.settings()["model"] == "claude-other"
    assert reply.cost == pytest.approx((1000 * 1.0 + 2000 * 2.0) / 1e6)

    # An explicit rate card also wins for a model that has an entry in PRICING
    assert claude(FakeAPI(), pricing=FREE).pricing == FREE
    assert claude(FakeAPI()).pricing == HAIKU


def test_claude_max_tokens_over_the_sdk_limit_is_rejected_before_sending():
    # Requests are not streamed, and the SDK refuses a non-streaming request whose `max_tokens`
    # could take over 10 minutes (about 21,300 tokens). The default `llm_max_tokens` stays under that
    api = FakeAPI(message())
    with pytest.raises(ValueError, match="Streaming is required"):
        claude(api, max_tokens=22_000).complete("s", "u")
    assert not api.requests
    assert DEFAULTS.llm_max_tokens <= 21_000
    assert claude(api, max_tokens=21_000).complete("s", "u").ok


def test_claude_default_client_takes_max_retries(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert ClaudeLLM()._client.max_retries == DEFAULTS.llm_max_retries
    assert ClaudeLLM(max_retries=1)._client.max_retries == 1


def test_claude_unknown_model_needs_pricing():
    with pytest.raises(ValueError, match="no pricing"):
        ClaudeLLM("claude-unknown", client=FakeAPI().client())
    llm = ClaudeLLM("claude-unknown", client=FakeAPI().client(), pricing=FREE)
    assert llm.pricing == FREE


# --- LLMClient as a new backend sees it -----------------------------------------------------------


class EchoLLM(LLMClient):
    """A third backend with the documented minimum: `provider`, `model`, `_send`, and a setting."""

    provider = "echo"

    def __init__(self, *, temperature: float = 0.0, **kwargs: Any) -> None:
        super().__init__(pricing=Pricing(Rates(1.0, 2.0, 0.0, 0.0)), **kwargs)
        self.temperature = temperature
        self.sent = 0

    @property
    def model(self) -> str:
        return "echo-1"

    def settings(self) -> dict[str, Any]:
        return {**super().settings(), "temperature": self.temperature}

    def _send(self, request: Request) -> Reply:
        self.sent += 1
        # The `cost` and `cached` a backend sets are overwritten by `complete`
        return Reply(
            request.user.upper(),
            StopReason.END_TURN,
            Usage(10, 20),
            "echo-1-2026",
            cost=99.0,
            cached=True,
        )


def test_new_backend_gets_cache_transcript_and_totals(tmp_path: Path):
    transcript = tmp_path / "transcript.jsonl"
    llm = EchoLLM(cache_dir=tmp_path / "cache", transcript=transcript)
    first = llm.complete("s", "hello", call_id="c/0")
    assert (first.text, first.model, first.cached) == ("HELLO", "echo-1-2026", False)
    assert first.cost == pytest.approx((10 * 1.0 + 20 * 2.0) / 1e6)  # from the pricing, not 99.0

    assert llm.complete("s", "hello", call_id="c/0") == replace(first, cached=True)
    assert llm.sent == 1
    assert (llm.n_sent, llm.n_cached, llm.usage) == (1, 1, Usage(10, 20))
    assert llm.cost == first.cost

    # The backend's own setting is part of the cache key and is logged
    warmer = EchoLLM(temperature=0.7, cache_dir=tmp_path / "cache", transcript=transcript)
    assert not warmer.complete("s", "hello", call_id="c/0").cached
    assert read_jsonl(transcript)[-1]["settings"] == {
        "provider": "echo",
        "model": "echo-1",
        "temperature": 0.7,
    }

    with pytest.raises(NotImplementedError, match="EchoLLM cannot count tokens"):
        llm.count_tokens("s", "hello")


def test_backend_must_define_provider_model_and_send():
    with pytest.raises(TypeError, match="abstract"):
        LLMClient(pricing=FREE)  # ty: ignore[abstract-class-instantiation]

    class NoSend(LLMClient):
        provider = "x"

        @property
        def model(self) -> str:
            return "m"

    with pytest.raises(TypeError, match="_send"):
        NoSend(pricing=FREE)  # ty: ignore[abstract-class-instantiation]

    class NoProvider(LLMClient):
        @property
        def model(self) -> str:
            return "m"

        def _send(self, request: Request) -> Reply:
            raise AssertionError("never sent")

    with pytest.raises(TypeError, match="must set the class attribute `provider`"):
        NoProvider(pricing=FREE)


# --- Reply cache and transcripts (provider-independent, through FakeLLM) --------------------------


def test_cache_hit_makes_no_provider_call(tmp_path: Path):
    llm = FakeLLM(["first", "second"], cache_dir=tmp_path)
    a = llm.complete("s", "u", call_id="arm/w/s0/0")
    b = llm.complete("s", "u", call_id="arm/w/s0/0")
    assert len(llm.requests) == 1
    assert (a.text, a.cached) == ("first", False)
    assert (b.text, b.cached) == ("first", True)
    assert (llm.n_sent, llm.n_cached) == (1, 1)
    assert (tmp_path / "arm/w/s0/0").is_dir()


def test_cache_misses_on_a_new_prompt_call_id_schema_or_model(tmp_path: Path):
    llm = FakeLLM(lambda r: f"reply to {r.user}", cache_dir=tmp_path)
    llm.complete("s", "u", call_id="c/0")
    llm.complete("s", "changed", call_id="c/0")  # same call, new prompt (D17: prompt hash)
    llm.complete("changed", "u", call_id="c/0")  # same call, new system prompt
    llm.complete("s", "u", call_id="c/1")  # same prompt, next call index
    llm.complete("s", "u", schema=HYPOTHESES_SCHEMA, call_id="c/0")
    assert len(llm.requests) == 5
    assert llm.n_cached == 0
    other_model = FakeLLM(["x"], model="other", cache_dir=tmp_path)
    assert not other_model.complete("s", "u", call_id="c/0").cached


def test_caching_needs_both_a_cache_dir_and_a_call_id(tmp_path: Path):
    llm = FakeLLM(["a", "b"], cache_dir=tmp_path)  # no call_id
    assert llm.complete("s", "u").text == "a"
    assert llm.complete("s", "u").text == "b"
    assert not any(tmp_path.iterdir())

    llm = FakeLLM(["a", "b"])  # no cache_dir
    assert llm.complete("s", "u", call_id="c").text == "a"
    assert llm.complete("s", "u", call_id="c").text == "b"
    assert (llm.n_sent, llm.n_cached) == (2, 0)


def test_cache_entry_records_the_call(tmp_path: Path):
    text = "Vmax·C_A/(Kₘ + C_A) → mM/min"  # replies are not always ASCII
    reply = Reply(text, StopReason.OTHER, Usage(1, 2, 3, 4), "fake-2026", "req_1", "pause_turn")
    llm = FakeLLM([reply], pricing=HAIKU, cache_dir=tmp_path)
    schema = {"type": "object"}
    sent = llm.complete("sys", "usr µ", schema=schema, call_id="arm/w/s0/call00/try1")

    [path] = (tmp_path / "arm/w/s0/call00/try1").iterdir()
    assert re.fullmatch(r"[0-9a-f]{64}\.json", path.name)
    assert path.read_bytes().isascii()  # so reading it back never depends on the locale
    entry = json.loads(path.read_text())
    assert datetime.fromisoformat(entry.pop("created")).tzinfo is not None
    assert entry == {
        "call_id": "arm/w/s0/call00/try1",
        "settings": {"provider": "fake", "model": "fake"},
        "request": {"system": "sys", "user": "usr µ", "schema": schema},
        "reply": {
            "text": text,
            "stop_reason": "other",
            "usage": {
                "input_tokens": 1,
                "output_tokens": 2,
                "cache_write_tokens": 3,
                "cache_read_tokens": 4,
            },
            "model": "fake-2026",
            "request_id": "req_1",
            "stop_detail": "pause_turn",
            "cost": sent.cost,
            "cached": False,
        },
    }

    # Every field of the reply survives the round trip
    rerun = FakeLLM([], pricing=HAIKU, cache_dir=tmp_path)
    again = rerun.complete("sys", "usr µ", schema=schema, call_id="arm/w/s0/call00/try1")
    assert again == replace(sent, cached=True)


def test_cache_key_is_stable(tmp_path: Path):
    llm = FakeLLM(["r", "with a schema"], cache_dir=tmp_path)
    llm.complete("s", "u", call_id="c")
    [path] = (tmp_path / "c").iterdir()
    # A fixed value on purpose. If this has to change, every reply cached so far is orphaned and
    # a rerun queries the provider again instead of replaying (D17)
    assert path.stem == "b81687f1f19d14f4eb127f7f32feb41f7b0651476ef0dfd6aa04f0fd86818b4b"

    # The key does not depend on the order in which a schema's keys were written
    llm.complete("s", "u", schema={"type": "object", "properties": {}}, call_id="c")
    assert llm.complete("s", "u", schema={"properties": {}, "type": "object"}, call_id="c").cached


def test_unusable_replies_are_cached_with_their_stop_reason(tmp_path: Path):
    truncated = Reply('{"hy', StopReason.MAX_TOKENS, Usage(10, 16000), "fake", "req_1")
    refused = Reply("", StopReason.REFUSAL, Usage(10, 5), "fake", stop_detail="bio: no")
    llm = FakeLLM([truncated, refused], cache_dir=tmp_path)
    llm.complete("s", "u", call_id="c/0/try1")
    llm.complete("s", "u", call_id="c/0/try2")

    rerun = FakeLLM([], cache_dir=tmp_path)
    r1 = rerun.complete("s", "u", call_id="c/0/try1")
    r2 = rerun.complete("s", "u", call_id="c/0/try2")
    assert not rerun.requests
    assert (r1.stop_reason, r1.text, r1.request_id, r1.ok) == (
        StopReason.MAX_TOKENS,
        '{"hy',
        "req_1",
        False,
    )
    assert (r2.stop_reason, r2.stop_detail) == (StopReason.REFUSAL, "bio: no")
    assert r2.usage == Usage(10, 5)


def test_errors_are_not_cached(tmp_path: Path):
    llm = FakeLLM([LLMError("down"), "ok"], cache_dir=tmp_path)
    with pytest.raises(LLMError):
        llm.complete("s", "u", call_id="c/0")
    assert llm.complete("s", "u", call_id="c/0").text == "ok"
    assert llm.n_sent == 1


# Not JSON, an empty file (as an interrupted write could leave), and JSON of the wrong shape
@pytest.mark.parametrize("content", ["{not json", "", "[]", '{"reply": {"text": "x"}}'])
def test_corrupt_cache_entry_is_a_miss(tmp_path: Path, content: str):
    llm = FakeLLM(["a", "b"], cache_dir=tmp_path)
    llm.complete("s", "u", call_id="c")
    [entry] = (tmp_path / "c").iterdir()  # one file: the write left no temporary file behind
    entry.write_text(content)
    with pytest.warns(UserWarning, match="unreadable cache entry"):
        assert llm.complete("s", "u", call_id="c").text == "b"
    # The fresh reply replaced the corrupt entry
    assert FakeLLM([], cache_dir=tmp_path).complete("s", "u", call_id="c").text == "b"


def test_failed_cache_write_leaves_no_file(tmp_path: Path):
    cache = ReplyCache(tmp_path)
    with pytest.raises(TypeError):
        cache.put("c", "key", {"reply": object()})  # not JSON-serialisable
    assert not list((tmp_path / "c").iterdir())
    assert cache.get("c", "key") is None

    # Over an existing entry, a failed write leaves that entry as it was
    FakeLLM(["kept"], cache_dir=tmp_path).complete("s", "u", call_id="d")
    [path] = (tmp_path / "d").iterdir()
    with pytest.raises(TypeError):
        cache.put("d", path.stem, {"reply": object()})
    assert [p.name for p in (tmp_path / "d").iterdir()] == [path.name]
    kept = cache.get("d", path.stem)
    assert kept is not None and kept.text == "kept"


@pytest.mark.parametrize(
    "call_id", ["", "../escape", "a//b", "a/./b", "a/../b", "/abs", "a/", "a b", "a\n"]
)
def test_invalid_call_ids_are_rejected(tmp_path: Path, call_id: str):
    with pytest.raises(ValueError, match="invalid call_id"):
        ReplyCache(tmp_path).path(call_id, "key")
    # Also without a cache, so that a test with a bare FakeLLM rejects what a cached run would
    llm = FakeLLM(["never sent"])
    with pytest.raises(ValueError, match="invalid call_id"):
        llm.complete("s", "u", call_id=call_id)
    assert not llm.requests


def test_transcript_has_one_line_per_call(tmp_path: Path):
    transcript = tmp_path / "runs" / "transcript.jsonl"
    reply = Reply("r", StopReason.END_TURN, Usage(100, 200), "fake", "req_9")

    def slow(request: Request) -> Reply:
        time.sleep(0.02)
        return reply

    llm = FakeLLM(slow, pricing=HAIKU, cache_dir=tmp_path / "cache", transcript=transcript)
    llm.complete("sys", "usr", call_id="c/0")
    llm.complete("sys", "usr", call_id="c/0")

    first, second = read_jsonl(transcript)
    assert set(first) == {"time", "call_id", "settings", "request", "reply", "latency_s"}
    assert datetime.fromisoformat(first["time"]).tzinfo is not None
    assert first["call_id"] == "c/0"
    assert first["settings"] == {"provider": "fake", "model": "fake"}
    assert first["request"] == {"system": "sys", "user": "usr", "schema": None}
    assert first["reply"]["text"] == "r"
    assert first["reply"]["stop_reason"] == "end_turn"
    assert first["reply"]["request_id"] == "req_9"
    assert first["reply"]["usage"] == {
        "input_tokens": 100,
        "output_tokens": 200,
        "cache_write_tokens": 0,
        "cache_read_tokens": 0,
    }
    assert first["reply"]["cost"] == pytest.approx((100 * 0.10 + 200 * 0.50) / 1e6)
    assert first["reply"]["cached"] is False
    assert second["reply"]["cached"] is True
    assert first["latency_s"] >= 0.02  # the time the provider took
    assert second["latency_s"] == 0  # a cache hit
    assert transcript.read_bytes().isascii()

    # A later client appends to the file; a call without a `call_id` is logged too
    FakeLLM(["x"], transcript=transcript).complete("sys", "usr")
    assert [line["call_id"] for line in read_jsonl(transcript)] == ["c/0", "c/0", None]


# --- FakeLLM --------------------------------------------------------------------------------------


def test_fake_llm_scripts():
    llm = FakeLLM(["a"])
    assert llm.complete("s", "u").text == "a"
    with pytest.raises(RuntimeError, match="ran out"):
        llm.complete("s", "u")

    echo = FakeLLM(lambda r: r.user.upper())
    assert echo.complete("s", "hello").text == "HELLO"
    assert echo.requests == [Request("s", "hello")]

    with pytest.raises(NotImplementedError):
        echo.count_tokens("s", "u")

    # A bare string is iterable, and would be consumed one character per reply
    with pytest.raises(TypeError, match="list of replies"):
        FakeLLM("one reply")


# --- Cost accounting ------------------------------------------------------------------------------


def test_haiku_cost():
    # $0.10 / $0.50 per million input / output tokens, for prompts of up to 100K tokens
    assert HAIKU.cost(Usage(100_000, 0)) == pytest.approx(0.01)
    assert HAIKU.cost(Usage(0, 1_000_000)) == pytest.approx(0.50)
    assert HAIKU.cost(Usage(2_000, 10_000)) == pytest.approx(0.0002 + 0.005)
    # Cache writes at 1.25x and reads at 0.1x the input rate
    assert HAIKU.cost(Usage(0, 0, cache_write_tokens=80_000)) == pytest.approx(0.01)
    assert HAIKU.cost(Usage(0, 0, cache_read_tokens=80_000)) == pytest.approx(0.0008)


def test_haiku_long_prompt_tier():
    # A prompt of more than 100K tokens, counting cached tokens, is billed at $0.50 / $2.50
    assert HAIKU.cost(Usage(100_000, 1_000)) == pytest.approx(0.01 + 0.0005)
    assert HAIKU.cost(Usage(100_001, 1_000)) == pytest.approx(100_001 * 0.5e-6 + 0.0025)
    assert HAIKU.cost(Usage(50_000, 1_000, cache_read_tokens=50_001)) == pytest.approx(
        50_000 * 0.5e-6 + 0.0025 + 50_001 * 0.05e-6
    )


def test_pricing_needs_both_long_fields():
    with pytest.raises(ValueError, match="both"):
        Pricing(Rates.standard(1.0, 2.0), long_above=10)


def test_usage_adds_field_by_field():
    assert Usage(1, 2, 3, 4) + Usage(10, 20, 30, 40) == Usage(11, 22, 33, 44)
    assert Usage() + Usage(1, 2, 3, 4) == Usage(1, 2, 3, 4)


def test_client_totals_count_sent_calls_only(tmp_path: Path):
    replies = [
        Reply("a", StopReason.END_TURN, Usage(2_000, 10_000), "fake"),
        Reply("b", StopReason.MAX_TOKENS, Usage(2_000, 16_000), "fake"),
    ]
    llm = FakeLLM(replies, pricing=HAIKU, cache_dir=tmp_path)
    a = llm.complete("s", "u", call_id="c/0")
    b = llm.complete("s", "u", call_id="c/1")
    llm.complete("s", "u", call_id="c/0")  # cache hit: free
    assert a.cost == pytest.approx(0.0052)
    assert b.cost == pytest.approx(0.0082)
    assert llm.cost == pytest.approx(a.cost + b.cost)
    assert llm.usage == Usage(4_000, 26_000)
    assert (llm.n_sent, llm.n_cached) == (2, 1)


def test_client_can_be_shared_between_threads(tmp_path: Path):
    transcript = tmp_path / "transcript.jsonl"
    llm = FakeLLM(
        lambda r: Reply("x", StopReason.END_TURN, Usage(1, 2), "fake"),
        pricing=HAIKU,
        cache_dir=tmp_path / "cache",
        transcript=transcript,
    )
    n_threads, n_calls = 8, 40

    def work(thread: int) -> None:
        for k in range(n_calls):
            llm.complete("s", "u", call_id=f"t{thread}/{k}")

    # Switch threads as often as possible, so that an unguarded total would lose updates
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        with ThreadPoolExecutor(n_threads) as pool:
            list(pool.map(work, range(n_threads)))
    finally:
        sys.setswitchinterval(interval)

    total = n_threads * n_calls
    assert (llm.n_sent, llm.n_cached) == (total, 0)
    assert llm.usage == Usage(total, 2 * total)
    assert llm.cost == pytest.approx(total * (0.10 + 2 * 0.50) / 1e6)
    assert len(read_jsonl(transcript)) == total  # every line is whole


def test_identical_concurrent_calls_send_once(tmp_path: Path):
    def slow(request: Request) -> str:
        time.sleep(0.05)  # long enough for the other threads to arrive while this one is sending
        return f"reply {len(llm.requests)}"

    llm = FakeLLM(slow, pricing=HAIKU, cache_dir=tmp_path)
    with ThreadPoolExecutor(8) as pool:
        replies = list(pool.map(lambda _: llm.complete("s", "u", call_id="c"), range(8)))

    # One thread sent; the others waited for it and took its reply from the cache
    assert len(llm.requests) == 1
    assert {r.text for r in replies} == {"reply 1"}
    assert sorted(r.cached for r in replies) == [False] + [True] * 7
    assert (llm.n_sent, llm.n_cached) == (1, 7)
    assert not llm._entry_locks  # no lock is kept once the calls are over


def test_different_calls_are_sent_in_parallel(tmp_path: Path):
    # Each send waits until all four are in flight, so this passes only if calls for different
    # cache entries do not wait for one another. Otherwise the barrier times out and raises
    barrier = threading.Barrier(4, timeout=5)

    def together(request: Request) -> str:
        barrier.wait()
        return "r"

    llm = FakeLLM(together, cache_dir=tmp_path)
    with ThreadPoolExecutor(4) as pool:
        replies = list(pool.map(lambda k: llm.complete("s", "u", call_id=f"c/{k}"), range(4)))
    assert [r.cached for r in replies] == [False] * 4
    assert not llm._entry_locks


def test_waiting_call_sends_itself_when_the_first_one_fails(tmp_path: Path):
    started = threading.Event()

    def flaky(request: Request) -> str:
        if len(llm.requests) == 1:
            started.set()
            time.sleep(0.05)  # the second call arrives and waits meanwhile
            raise LLMError("down")
        return "ok"

    llm = FakeLLM(flaky, cache_dir=tmp_path)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(llm.complete, "s", "u", call_id="c")
        assert started.wait(timeout=5)
        second = pool.submit(llm.complete, "s", "u", call_id="c")
        with pytest.raises(LLMError, match="down"):
            first.result(timeout=5)
        # The error was not cached and released the entry, so the waiting call sent its own
        assert second.result(timeout=5).text == "ok"
    assert (len(llm.requests), llm.n_sent, llm.n_cached) == (2, 1, 0)
    assert not llm._entry_locks


# --- Hypotheses schema (D15) ----------------------------------------------------------------------


def test_hypotheses_schema_meets_structured_output_rules():
    def objects(node: Any):
        if isinstance(node, dict):
            if node.get("type") == "object":
                yield node
            for v in node.values():
                yield from objects(v)

    found = list(objects(HYPOTHESES_SCHEMA))
    assert len(found) == 3
    for obj in found:
        assert obj["additionalProperties"] is False
        assert sorted(obj["required"]) == sorted(obj["properties"])


def conforms(value: Any, schema: dict[str, Any]) -> bool:
    """Whether `value` follows `schema`, for the part of JSON Schema that `HYPOTHESES_SCHEMA` uses."""
    match schema["type"]:
        case "object":
            return (
                isinstance(value, dict)
                and set(value) == set(schema["properties"])
                and all(conforms(value[k], sub) for k, sub in schema["properties"].items())
            )
        case "array":
            return isinstance(value, list) and all(conforms(v, schema["items"]) for v in value)
        case "string":
            return isinstance(value, str)
        case "number":
            return isinstance(value, int | float) and not isinstance(value, bool)
    raise AssertionError(f"unsupported schema type {schema['type']!r}")


def test_reply_that_follows_the_schema_parses():
    # Ties the schema to the parser: a reply in the shape the API is asked for is one that
    # `parse_hypotheses` accepts, so changing a key on one side fails here
    reply = json.loads(REPLY_JSON)
    assert conforms(reply, HYPOTHESES_SCHEMA)
    assert not conforms({"hypotheses": [{"name": "mm", "expr": "C_A"}]}, HYPOTHESES_SCHEMA)
    hypotheses, rejected = parse_hypotheses(REPLY_JSON)
    assert [h.name for h in hypotheses] == ["mm"] and not rejected
