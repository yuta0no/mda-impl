"""LLM clients: one interface for every provider, with a reply cache, transcripts and cost accounting.

`LLMClient` is the provider-neutral base class. A provider subclass implements `_send`, which makes
one request and returns a `Reply`; the base class adds what every provider shares:

- the on-disk reply cache (D17), so a rerun replays earlier replies instead of querying again;
- a JSONL transcript with one line per call, cache hits included;
- running totals of token usage and cost.

`ClaudeLLM` is the backend used in the experiments. `FakeLLM` returns scripted replies and
is what tests use: no test calls a real API.
"""

import hashlib
import json
import os
import re
import tempfile
import threading
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Literal

import anthropic
from anthropic.types import OutputConfigParam

DEFAULT_MODEL = "claude-haiku-5-5"  # spec §9
DEFAULT_EFFORT = "medium"  # spec §9: the paper's model ran at medium reasoning effort
# Room for adaptive thinking plus a reply of up to 12 hypotheses. Thinking tokens count against
# this cap, and a non-streaming request must stay under the SDK's ~21K-token limit
DEFAULT_MAX_TOKENS = 16_000
# The SDK retries 408, 409, 429, 5xx and connection errors with exponential backoff. Its default of
# 2 retries is short for an unattended run of several hours
DEFAULT_MAX_RETRIES = 5

# Claude's effort levels (`output_config.effort`). Other providers name and scale theirs differently,
# so a new backend defines its own
type ClaudeEffort = Literal["low", "medium", "high", "xhigh", "max"]
type JSONSchema = dict[str, Any]


class LLMError(RuntimeError):
    """Raised when a provider request fails, after the provider's own retries."""


class StopReason(StrEnum):
    """Why generation stopped, mapped to the same values for every provider."""

    END_TURN = "end_turn"  # the reply is complete
    MAX_TOKENS = "max_tokens"  # truncated at the output-token cap
    REFUSAL = "refusal"  # declined by the provider's safety systems
    OTHER = "other"  # anything else; `Reply.stop_detail` holds the provider's own value


@dataclass(frozen=True)
class Usage:
    """Token counts of one call, or a sum of calls. Output tokens include thinking tokens.

    The three input counts are disjoint: `input_tokens` leaves out the tokens written to or read
    from the prompt cache. A provider that reports cached tokens inside its prompt total needs
    them subtracted in `_send`, or `Pricing.cost` bills them twice.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def prompt_tokens(self) -> int:
        """Every input token, whether written to, read from or not touching the prompt cache."""
        return self.input_tokens + self.cache_write_tokens + self.cache_read_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
        )


@dataclass(frozen=True)
class Rates:
    """Prices in USD per million tokens."""

    input: float
    output: float
    cache_write: float
    cache_read: float

    @classmethod
    def standard(cls, input: float, output: float) -> "Rates":
        """Rates with Anthropic's usual prompt-cache multipliers: 1.25x input for 5-minute cache
        writes and 0.1x input for cache reads."""
        return cls(input, output, cache_write=1.25 * input, cache_read=0.1 * input)


@dataclass(frozen=True)
class Pricing:
    """A model's rate card. Prompts of more than `long_above` tokens are billed entirely at `long`."""

    base: Rates
    long: Rates | None = None
    long_above: int | None = None

    def __post_init__(self) -> None:
        if (self.long is None) != (self.long_above is None):
            raise ValueError("set both `long` and `long_above`, or neither")

    def rates(self, usage: Usage) -> Rates:
        if (
            self.long is not None
            and self.long_above is not None
            and usage.prompt_tokens > self.long_above
        ):
            return self.long
        return self.base

    def cost(self, usage: Usage) -> float:
        """The cost of one call in USD. Apply it per call: the long-prompt tier depends on the
        prompt length of each call, so it is wrong for a `Usage` summed over calls."""
        r = self.rates(usage)
        return (
            usage.input_tokens * r.input
            + usage.output_tokens * r.output
            + usage.cache_write_tokens * r.cache_write
            + usage.cache_read_tokens * r.cache_read
        ) / 1e6


FREE = Pricing(Rates(0.0, 0.0, 0.0, 0.0))

# https://platform.claude.com/docs/en/models/haiku-5-5/overview, checked 2026-11-26
PRICING: dict[str, Pricing] = {
    "claude-haiku-5-5": Pricing(
        base=Rates.standard(input=0.10, output=0.50),
        long=Rates.standard(input=0.50, output=2.50),
        long_above=100_000,
    ),
}


@dataclass(frozen=True)
class Request:
    system: str
    user: str
    # JSON schema the reply must follow (structured outputs, D15), or None for free text
    schema: JSONSchema | None = None


@dataclass(frozen=True)
class Reply:
    text: str
    stop_reason: StopReason
    usage: Usage
    model: str  # the model that answered, as the provider reports it
    request_id: str | None = None  # the provider's ID for the request, when it gives one
    # Refusal category and explanation, or the provider's stop reason when `stop_reason` is OTHER
    stop_detail: str | None = None
    cost: float = 0.0  # USD, of the call that produced the reply; set by `LLMClient.complete`
    cached: bool = False  # replayed from the reply cache, at no cost

    @property
    def ok(self) -> bool:
        """Whether generation finished normally. A reply that is not ok is unusable (D14)."""
        return self.stop_reason is StopReason.END_TURN


# Call IDs become cache paths: slash-separated segments of letters, digits, `_`, `-` and `.`
_CALL_ID = re.compile(r"[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*")


def _check_call_id(call_id: str) -> None:
    if not _CALL_ID.fullmatch(call_id) or any(s in (".", "..") for s in call_id.split("/")):
        raise ValueError(
            f"invalid call_id {call_id!r}: use `/`-separated segments of letters, digits, "
            "`_`, `-` and `.`, and no `.` or `..` segment"
        )


class ReplyCache:
    """Replies stored as `<root>/<call_id>/<request hash>.json` (D17).

    The call ID names the logical call, such as `hybrid/c0_michaelis_menten/s0/call03/try1`, and
    the hash covers everything else that determines the reply: the provider settings (model,
    effort, ...) and the request. Changing the prompt or the model therefore misses the cache,
    while a rerun with the same inputs replays the stored reply. Retrying an unusable reply needs
    a new call ID, because the same ID and request return the stored reply.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    def path(self, call_id: str, key: str) -> Path:
        _check_call_id(call_id)
        return self.root / call_id / f"{key}.json"

    def get(self, call_id: str, key: str) -> Reply | None:
        path = self.path(call_id, key)
        try:
            entry = json.loads(path.read_text())
            return _reply_from_json(entry["reply"])
        except FileNotFoundError:
            return None
        except (ValueError, KeyError, TypeError) as e:
            warnings.warn(f"ignoring unreadable cache entry {path}: {e!r}", stacklevel=3)
            return None

    def put(self, call_id: str, key: str, entry: dict[str, Any]) -> None:
        path = self.path(call_id, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary file and rename it, so an interrupted run never leaves half an entry
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(entry, f, indent=1)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


class LLMClient(ABC):
    """Base class of every LLM backend.

    Subclasses set `provider`, implement `model` and `_send`, and add their own generation settings
    to `settings()`. They can also implement `count_tokens`.

    A client can be shared between threads: the totals and the transcript stay consistent, and
    concurrent calls for the same cache entry send only once, the others waiting for that reply.
    A client cannot be pickled, so with multiprocessing each worker builds its own: workers can
    share a `cache_dir`, and each should write its own transcript. Nothing stops two processes
    from sending the same call at the same time, so each `call_id` must belong to one worker.
    """

    provider: ClassVar[str]

    def __init__(
        self,
        *,
        pricing: Pricing,
        cache_dir: str | os.PathLike[str] | None = None,
        transcript: str | os.PathLike[str] | None = None,
    ) -> None:
        """`cache_dir` enables the reply cache, for calls made with a `call_id`. `transcript` is a
        JSONL file that gets one line per call; it is appended to, never truncated."""
        if not isinstance(getattr(self, "provider", None), str):
            # Otherwise the first `settings()` call fails, and only once a cache or transcript is on
            raise TypeError(f"{type(self).__name__} must set the class attribute `provider`")
        self.pricing = pricing
        self.cache = ReplyCache(cache_dir) if cache_dir is not None else None
        self.transcript = Path(transcript) if transcript is not None else None
        self.usage = Usage()  # summed over calls sent to the provider; cache hits add nothing
        self.cost = 0.0  # USD, likewise
        self.n_sent = 0  # calls the provider answered; a request that fails is not counted
        self.n_cached = 0  # calls answered from the cache
        self._lock = threading.Lock()  # guards the totals, the transcript and `_entry_locks`
        # One lock per cache entry in use, with the number of calls holding or waiting for it
        self._entry_locks: dict[Path, tuple[threading.Lock, int]] = {}

    @property
    @abstractmethod
    def model(self) -> str:
        """The model requested."""

    def settings(self) -> dict[str, Any]:
        """Everything besides the request that determines the reply. Part of the cache key, so
        subclasses add their generation settings."""
        return {"provider": self.provider, "model": self.model}

    @abstractmethod
    def _send(self, request: Request) -> Reply:
        """Make one request to the provider. Raise `LLMError` if it fails. `cost` and `cached` of
        the returned reply are ignored."""

    def count_tokens(self, system: str, user: str, *, schema: JSONSchema | None = None) -> int:
        """The number of input tokens the request would use."""
        raise NotImplementedError(f"{type(self).__name__} cannot count tokens")

    def complete(
        self,
        system: str,
        user: str,
        *,
        schema: JSONSchema | None = None,
        call_id: str | None = None,
    ) -> Reply:
        """Send one single-turn request, or replay it from the cache.

        `schema` asks for structured output following that JSON schema (D15), where the provider
        supports it. Replies are cached only when the client has a `cache_dir` and the call has a
        `call_id` (see `ReplyCache`). A cached reply comes back with `cached=True` and the usage
        and cost of the original call. Every reply is returned, including ones that are not `ok`;
        handling them is the caller's job. Raises `LLMError` when the request fails, and
        `ValueError` for an invalid `call_id`.
        """
        request = Request(system, user, schema)
        if call_id is None:
            return self._send_and_record(None, request, None)
        # Checked even without a cache, so that a test rejects the IDs a cached run would
        _check_call_id(call_id)
        if self.cache is None:
            return self._send_and_record(call_id, request, None)

        key = _hash({"settings": self.settings(), "request": asdict(request)})
        # One call at a time per cache entry. Otherwise two threads making the same call would
        # both miss the cache and both send, paying twice and getting two different replies
        with self._exclusive(self.cache.path(call_id, key)):
            cached = self.cache.get(call_id, key)
            if cached is not None:
                reply = replace(cached, cached=True)
                with self._lock:
                    self.n_cached += 1
                    self._log(call_id, request, reply, latency=0.0)
                return reply
            return self._send_and_record(call_id, request, key)

    @contextmanager
    def _exclusive(self, entry: Path) -> Generator[None, None, None]:
        """Hold the lock of one cache entry. Calls for other entries are not held up, and a lock
        is dropped once no call holds or waits for it."""
        with self._lock:
            lock, users = self._entry_locks.get(entry, (threading.Lock(), 0))
            self._entry_locks[entry] = (lock, users + 1)
        try:
            with lock:
                yield
        finally:
            with self._lock:
                lock, users = self._entry_locks[entry]
                if users > 1:
                    self._entry_locks[entry] = (lock, users - 1)
                else:
                    del self._entry_locks[entry]

    def _send_and_record(self, call_id: str | None, request: Request, key: str | None) -> Reply:
        """Send the request, add it to the totals and the transcript, and cache the reply under
        `key` when there is one."""
        start = time.perf_counter()
        reply = self._send(request)
        latency = time.perf_counter() - start
        reply = replace(reply, cost=self.pricing.cost(reply.usage), cached=False)
        with self._lock:
            self.n_sent += 1
            self.usage += reply.usage
            self.cost += reply.cost
            if key is not None and call_id is not None and self.cache is not None:
                self.cache.put(
                    call_id,
                    key,
                    {
                        "call_id": call_id,
                        "created": _now(),
                        "settings": self.settings(),
                        "request": asdict(request),
                        "reply": _reply_to_json(reply),
                    },
                )
            self._log(call_id, request, reply, latency)
        return reply

    def _log(self, call_id: str | None, request: Request, reply: Reply, latency: float) -> None:
        if self.transcript is None:
            return
        line = {
            "time": _now(),
            "call_id": call_id,
            "settings": self.settings(),
            "request": asdict(request),
            "reply": _reply_to_json(reply),
            "latency_s": round(latency, 3),
        }
        self.transcript.parent.mkdir(parents=True, exist_ok=True)
        with self.transcript.open("a") as f:
            f.write(json.dumps(line) + "\n")


class ClaudeLLM(LLMClient):
    """Claude through the Anthropic Messages API, with adaptive thinking at a set effort."""

    provider = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        effort: ClaudeEffort = DEFAULT_EFFORT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        pricing: Pricing | None = None,
        client: anthropic.Anthropic | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        cache_dir: str | os.PathLike[str] | None = None,
        transcript: str | os.PathLike[str] | None = None,
    ) -> None:
        """`client` defaults to `anthropic.Anthropic(max_retries=max_retries)`, which reads the
        API key from the environment. `pricing` defaults to the entry for `model` in `PRICING`.
        Requests are not streamed, so the SDK rejects a `max_tokens` above about 21,000 with a
        `ValueError` when the request is made."""
        if pricing is None:
            if model not in PRICING:
                raise ValueError(f"no pricing known for {model!r}; pass `pricing`")
            pricing = PRICING[model]
        super().__init__(pricing=pricing, cache_dir=cache_dir, transcript=transcript)
        self._model = model
        self.effort: ClaudeEffort = effort
        self.max_tokens = max_tokens
        self._client = (
            client if client is not None else anthropic.Anthropic(max_retries=max_retries)
        )

    @property
    def model(self) -> str:
        return self._model

    def settings(self) -> dict[str, Any]:
        return {**super().settings(), "effort": self.effort, "max_tokens": self.max_tokens}

    def _output_config(self, schema: JSONSchema | None) -> OutputConfigParam:
        config: OutputConfigParam = {"effort": self.effort}
        if schema is not None:
            config["format"] = {"type": "json_schema", "schema": schema}
        return config

    def _send(self, request: Request) -> Reply:
        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=self.max_tokens,
                system=request.system,
                messages=[{"role": "user", "content": request.user}],
                output_config=self._output_config(request.schema),
            )
        except anthropic.APIError as e:
            raise LLMError(f"Anthropic request failed: {e}") from e

        stop_reason, stop_detail = StopReason.OTHER, message.stop_reason
        if message.stop_reason in ("end_turn", "max_tokens", "refusal"):
            stop_reason, stop_detail = StopReason(message.stop_reason), None
        if message.stop_reason == "refusal" and message.stop_details is not None:
            # The API can leave out the category, the explanation or both
            details = (message.stop_details.category, message.stop_details.explanation)
            stop_detail = ": ".join(d for d in details if d) or None
        u = message.usage
        return Reply(
            # Thinking blocks are skipped: their text is omitted by default on this model family
            text="".join(b.text for b in message.content if b.type == "text"),
            stop_reason=stop_reason,
            usage=Usage(
                input_tokens=u.input_tokens,
                output_tokens=u.output_tokens,
                cache_write_tokens=u.cache_creation_input_tokens or 0,
                cache_read_tokens=u.cache_read_input_tokens or 0,
            ),
            model=message.model,
            request_id=message._request_id,
            stop_detail=stop_detail,
        )

    def count_tokens(self, system: str, user: str, *, schema: JSONSchema | None = None) -> int:
        """Input tokens of the request, from the free `count_tokens` endpoint."""
        try:
            result = self._client.messages.count_tokens(
                model=self._model,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config=self._output_config(schema),
            )
        except anthropic.APIError as e:
            raise LLMError(f"Anthropic token count failed: {e}") from e
        return result.input_tokens


# One item of a `FakeLLM` script: a reply text, a full `Reply`, or an exception to raise
type Scripted = str | Reply | Exception


class FakeLLM(LLMClient):
    """A backend that returns scripted replies, for tests.

    `script` is either a sequence, consumed one item per sent request, or a function of the request.
    An item is a reply text (a complete reply with zero usage), a full `Reply`, or an exception to
    raise. `requests` records every request that reached the "provider", so a cache hit leaves it
    unchanged.
    """

    provider = "fake"

    def __init__(
        self,
        script: Iterable[Scripted] | Callable[[Request], Scripted] = (),
        *,
        model: str = "fake",
        pricing: Pricing = FREE,
        cache_dir: str | os.PathLike[str] | None = None,
        transcript: str | os.PathLike[str] | None = None,
    ) -> None:
        super().__init__(pricing=pricing, cache_dir=cache_dir, transcript=transcript)
        self._model = model
        self._respond: Callable[[Request], Scripted]
        if isinstance(script, str):
            # The script itself must not be one string. A string is an iterable of strings, so
            # it passes as `Iterable[Scripted]` and would be consumed one character per reply
            raise TypeError("`script` takes a list of replies; wrap a single reply in a list")
        if isinstance(script, Iterable):
            items: Iterator[Scripted] = iter(script)
            self._respond = lambda _request: next(items)
        else:
            self._respond = script
        self.requests: list[Request] = []

    @property
    def model(self) -> str:
        return self._model

    def _send(self, request: Request) -> Reply:
        self.requests.append(request)
        try:
            item = self._respond(request)
        except StopIteration:
            raise RuntimeError("FakeLLM ran out of scripted replies") from None
        if isinstance(item, Exception):
            raise item
        if isinstance(item, str):
            return Reply(
                text=item, stop_reason=StopReason.END_TURN, usage=Usage(), model=self.model
            )
        return item


def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _reply_to_json(reply: Reply) -> dict[str, Any]:
    return {**asdict(reply), "stop_reason": reply.stop_reason.value}


def _reply_from_json(d: dict[str, Any]) -> Reply:
    return Reply(
        text=d["text"],
        stop_reason=StopReason(d["stop_reason"]),
        usage=Usage(**d["usage"]),
        model=d["model"],
        request_id=d.get("request_id"),
        stop_detail=d.get("stop_detail"),
        cost=d.get("cost", 0.0),
    )
