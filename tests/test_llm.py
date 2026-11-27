import pytest
from stubs import chat_server

from napoleon.llm import (
    LLMError,
    LLMResponse,
    OpenAICompatClient,
    ScriptedClient,
    clean_reply,
    extract_json,
    make_client,
)
from napoleon.usage import Pricing, Usage

MSGS = [{"role": "system", "content": "s"}, {"role": "user", "content": "hello there"}]


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('<think>hmm {"x": 0}</think>\n{"a": 1}', {"a": 1}),
        ('reasoning...</think>{"a": 1}', {"a": 1}),
        ('Sure!\n```json\n{"a": {"b": 2}}\n```\nDone.', {"a": {"b": 2}}),
        ('I will do this: {"a": 1} hope it helps', {"a": 1}),
        ('{"broken": ', None),
        ('<think>never finished {"a": 1}', None),
        ("no json at all", None),
        ("[1, 2]", None),
    ],
)
def test_extract_json_is_lenient(text, expected):
    assert extract_json(text) == expected


def test_clean_reply_strips_think():
    assert clean_reply("<think>a\nb</think>  answer ") == "answer"


def test_usage_adds_and_pricing_costs():
    u = Usage(calls=1, input_tokens=10, output_tokens=5) + Usage(calls=2, input_tokens=1)
    assert (u.calls, u.input_tokens, u.output_tokens) == (3, 11, 5)
    assert Pricing(input_per_mtok=2, output_per_mtok=10).cost(1_000_000, 500_000) == 7.0


def test_scripted_client_replays_then_repeats_last():
    client = ScriptedClient(
        ["one", "two"], pricing=Pricing(input_per_mtok=1e6, output_per_mtok=1e6)
    )
    first = client.complete(MSGS)
    assert [client.complete(MSGS).text for _ in range(2)] == ["two", "two"]
    assert first.text == "one"
    assert first.usage.calls == 1 and first.usage.input_tokens > 0
    assert first.usage.cost_usd > 0
    assert len(client.calls) == 3


def test_scripted_client_callable_and_explicit_usage():
    client = ScriptedClient(lambda msgs: f"{len(msgs)} messages")
    assert client.complete(MSGS).text == "2 messages"
    fixed = LLMResponse("x", Usage(calls=1, input_tokens=7))
    assert ScriptedClient([fixed]).complete(MSGS) is fixed


def test_http_client_talks_chat_completions():
    with chat_server(lambda msgs: "<think>x</think>hi") as (url, seen):
        client = OpenAICompatClient(
            "qwen3:8b", url, "sekret", pricing=Pricing(input_per_mtok=1, output_per_mtok=2)
        )
        out = client.complete(MSGS, max_tokens=64)
    req = seen[0]
    assert req["path"] == "/v1/chat/completions"
    assert req["auth"] == "Bearer sekret"
    assert req["model"] == "qwen3:8b" and req["max_tokens"] == 64 and req["messages"] == MSGS
    assert out.text == "<think>x</think>hi"  # cleaning is the parser's job, not the client's
    assert (out.usage.input_tokens, out.usage.output_tokens) == (100, 20)
    assert out.usage.cost_usd == pytest.approx(140 / 1e6)


def test_http_client_options():
    with chat_server(lambda msgs: "ok") as (url, seen):
        OpenAICompatClient("m", url, no_think=True, json_mode=True).complete(MSGS)
    assert seen[0]["auth"] is None
    assert seen[0]["messages"][-1]["content"].endswith(" /no_think")
    assert seen[0]["response_format"] == {"type": "json_object"}


def test_http_client_errors_are_llm_errors():
    with chat_server(lambda msgs: "", status=500) as (url, _), pytest.raises(LLMError, match="500"):
        OpenAICompatClient("m", url).complete(MSGS)
    with pytest.raises(LLMError, match="could not get a reply"):
        OpenAICompatClient("m", "http://127.0.0.1:1/v1", timeout=2).complete(MSGS)


def test_make_client_reads_env(monkeypatch):
    monkeypatch.setenv("NAPOLEON_LLM_BASE_URL", "http://example.test/v1/")
    monkeypatch.setenv("NAPOLEON_LLM_MODEL", "qwen-plus")
    monkeypatch.setenv("NAPOLEON_LLM_API_KEY", "k")
    c = make_client()
    assert (c.model, c.base_url, c.api_key) == ("qwen-plus", "http://example.test/v1", "k")
    assert make_client("other", "http://x/v1").model == "other"
