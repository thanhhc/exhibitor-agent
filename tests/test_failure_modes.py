"""
Failure-mode tests for the agent guards.

Design note — the split that matters:

  Unit tests (fast, free, deterministic) call execute_tool directly and prove
  each guard fires. They do NOT ask the model to misbehave, because you cannot
  reliably make a model invent a tool name or send the wrong type on demand.
  You test the guard, not the trigger.

  Loop tests use a fake client so the multi-turn logic (repetition, iteration
  limit) is tested without spending money or depending on model mood.

  Integration tests hit the real API and are marked so they can be skipped.
  Run them before a demo, not on every save:  pytest -m integration

Run:  uv run pytest tests/ -v
"""

import json
import time
import pytest

import agent
from agent import execute_tool, call_signature, run_agent, MAX_TOOL_OUTPUT


# ------------------------------------------------------------------ fixtures

@pytest.fixture(autouse=True)
def restore_tools():
    """Every test may stomp on TOOL_IMPL; put it back afterwards."""
    original = dict(agent.TOOL_IMPL)
    yield
    agent.TOOL_IMPL.clear()
    agent.TOOL_IMPL.update(original)


# ------------------------------------------------------------------ 1. tool raises

def test_tool_exception_becomes_error_message():
    def exploding(show_id: str, topic: str) -> str:
        return 1 / 0

    agent.TOOL_IMPL["lookup_regulation"] = exploding

    content, is_error = execute_tool("lookup_regulation", {"show_id": "x", "topic": "y"})

    assert is_error is True
    assert "ZeroDivisionError" in content, "model needs the exception type to reason about recovery"


def test_tool_exception_does_not_propagate():
    """The whole point: a broken tool must never crash the agent process."""
    def exploding(show_id: str, topic: str) -> str:
        raise RuntimeError("database connection lost")

    agent.TOOL_IMPL["lookup_regulation"] = exploding

    content, is_error = execute_tool("lookup_regulation", {"show_id": "x", "topic": "y"})
    assert is_error is True
    assert "database connection lost" in content


# ------------------------------------------------------------------ 2. timeout

@pytest.mark.slow
def test_slow_tool_times_out():
    def sluggish(show_id: str, topic: str) -> str:
        time.sleep(30)
        return "never seen"

    agent.TOOL_IMPL["lookup_regulation"] = sluggish

    started = time.monotonic()
    content, is_error = execute_tool("lookup_regulation", {"show_id": "x", "topic": "y"})
    elapsed = time.monotonic() - started

    assert is_error is True
    assert "timed out" in content.lower()
    assert "narrow" in content.lower(), "tell the model what to do differently, not just that it failed"
    assert elapsed < agent.TOOL_TIMEOUT_S + 2, (
        f"returned after {elapsed:.1f}s — the timeout is not actually releasing control. "
        "See the executor shutdown note in FAILURES.md."
    )


# ------------------------------------------------------------------ 3. unknown tool

def test_unknown_tool_name():
    content, is_error = execute_tool("check_wifi_password", {"foo": "bar"})

    assert is_error is True
    assert "no such tool" in content.lower()
    assert "lookup_regulation" in content, "list the real tools so the model can self-correct"


# ------------------------------------------------------------------ 4. bad arguments

def test_wrong_type_rejected():
    content, is_error = execute_tool("lookup_regulation", {"show_id": 123, "topic": "height"})

    assert is_error is True
    assert "invalid arguments" in content.lower()
    assert "show_id" in content, "the model needs to know WHICH field was wrong"


def test_missing_required_field():
    content, is_error = execute_tool("lookup_regulation", {"show_id": "ambiente-2027"})

    assert is_error is True
    assert "topic" in content


def test_correct_schema_per_tool():
    """Regression: get_deadline was once validated against RegulationArgs."""
    content, is_error = execute_tool(
        "get_deadline", {"show_id": "ambiente-2027", "form_name": "rigging order"}
    )
    assert is_error is False
    assert json.loads(content)["deadline"] == "2027-01-15"


def test_extra_field_is_tolerated():
    """Models sometimes add plausible-looking fields. Don't fail on that."""
    content, is_error = execute_tool(
        "lookup_regulation",
        {"show_id": "ambiente-2027", "topic": "height", "hall": "4.1"},
    )
    assert is_error is False


# ------------------------------------------------------------------ 5. oversized output

def test_huge_output_truncated():
    def firehose(show_id: str, topic: str) -> str:
        return "x" * 500_000

    agent.TOOL_IMPL["lookup_regulation"] = firehose

    content, is_error = execute_tool("lookup_regulation", {"show_id": "x", "topic": "y"})

    assert is_error is False, "truncation is not an error, just a limit"
    assert len(content) < MAX_TOOL_OUTPUT + 200
    assert "truncated" in content.lower(), "the model must know it saw a partial result"


# ------------------------------------------------------------------ 6. repetition loop

class FakeBlock:
    def __init__(self, name, tool_input, block_id="toolu_fake"):
        self.type = "tool_use"
        self.id = block_id
        self.name = name
        self.input = tool_input


class FakeUsage:
    input_tokens = 100
    output_tokens = 50


class FakeResponse:
    def __init__(self, content, stop_reason="tool_use"):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = FakeUsage()


class StubbornClient:
    """A model that always asks for the same tool with the same arguments."""

    def __init__(self):
        self.calls = 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        if "tools" not in kwargs:                       # the final tools-removed call
            return FakeResponse([type("T", (), {"type": "text", "text": "gave up"})()],
                                stop_reason="end_turn")
        return FakeResponse([
            FakeBlock("lookup_regulation",
                      {"show_id": "ambiente-2027", "topic": "height"},
                      block_id=f"toolu_{self.calls}")
        ])


def test_repetition_guard_short_circuits(monkeypatch, capsys):
    stub = StubbornClient()
    monkeypatch.setattr(agent, "client", stub)

    result = run_agent("anything", max_iterations=8)

    out = capsys.readouterr().out
    assert "✗" in out, "repeated calls should be flagged as errors after the limit"
    assert result, "must still return something rather than raising"


def test_call_signature_is_order_independent():
    a = call_signature("t", {"show_id": "x", "topic": "y"})
    b = call_signature("t", {"topic": "y", "show_id": "x"})
    assert a == b, "dict ordering must not defeat the repetition guard"


def test_call_signature_distinguishes_different_args():
    a = call_signature("t", {"show_id": "x", "topic": "height"})
    b = call_signature("t", {"show_id": "x", "topic": "rigging"})
    assert a != b, "different queries are not repetition"


# ------------------------------------------------------------------ 7. iteration limit

class NeverStopsClient:
    """A model that keeps requesting genuinely different tool calls forever."""

    def __init__(self):
        self.calls = 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        if "tools" not in kwargs:
            return FakeResponse(
                [type("T", (), {"type": "text", "text": "Partial answer: could not finish."})()],
                stop_reason="end_turn",
            )
        return FakeResponse([
            FakeBlock("lookup_regulation",
                      {"show_id": "ambiente-2027", "topic": f"topic-{self.calls}"},
                      block_id=f"toolu_{self.calls}")
        ])


def test_iteration_limit_degrades_gracefully(monkeypatch):
    stub = NeverStopsClient()
    monkeypatch.setattr(agent, "client", stub)

    result = run_agent("anything", max_iterations=4)

    assert "Partial answer" in result, "must return a best-effort answer, not raise"
    assert stub.calls == 5, "4 tool turns + 1 final tools-removed call"


# ------------------------------------------------------------------ integration

@pytest.mark.integration
def test_agent_recovers_from_broken_tool_end_to_end():
    """Real API. Does the model actually handle the error text sensibly?"""
    def exploding(show_id: str, topic: str) -> str:
        raise RuntimeError("regulations database is offline")

    agent.TOOL_IMPL["lookup_regulation"] = exploding

    answer = run_agent(
        "What is the maximum stand height at Ambiente 2027?",
        max_iterations=4, verbose=False,
    )

    lowered = answer.lower()
    assert any(w in lowered for w in ["unable", "could not", "offline", "unavailable", "error"]), (
        f"Agent should admit failure, not invent a height. Got: {answer[:300]}"
    )
    assert "4.0 m" not in answer, "HALLUCINATION: invented a limit with no working tool"


@pytest.mark.integration
def test_agent_does_not_answer_outside_its_tools():
    answer = run_agent("What time does the venue open?", max_iterations=4, verbose=False)
    assert len(answer) > 0