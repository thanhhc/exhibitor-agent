from dotenv import load_dotenv; load_dotenv()
import json
import time
import concurrent.futures
from dataclasses import dataclass, field
from typing import Callable, Type
from anthropic import Anthropic
from pydantic import BaseModel, ValidationError

client = Anthropic()
MODEL = "claude-sonnet-4-5"
TOOL_TIMEOUT_S = 10
MAX_TOOL_OUTPUT = 50_000
REPEAT_LIMIT = 3
MAX_DEPTH = 3
COST_PER_MTOK = {"input": 3.00, "output": 15.00}


# ================================================================ tool plumbing

class RegulationArgs(BaseModel):
    show_id: str
    topic: str


class DeadlineArgs(BaseModel):
    show_id: str
    form_name: str


def anthropic_schema(model: Type[BaseModel]) -> dict:
    """Pydantic model -> Anthropic input_schema. One source of truth per tool."""
    schema = model.model_json_schema()
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    return schema


@dataclass
class Tool:
    name: str
    description: str
    args: Type[BaseModel]
    fn: Callable[..., str]

    def spec(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": anthropic_schema(self.args),
        }


@dataclass
class Budget:
    """Shared across a whole agent tree so nested specialists can't blow the bank."""
    limit_usd: float = 0.50
    spent: float = 0.0
    calls: int = 0

    def charge(self, resp) -> None:
        self.spent += (resp.usage.input_tokens * COST_PER_MTOK["input"]
                       + resp.usage.output_tokens * COST_PER_MTOK["output"]) / 1_000_000
        self.calls += 1

    @property
    def exhausted(self) -> bool:
        return self.spent >= self.limit_usd


@dataclass
class Trace:
    turns: int = 0
    tool_calls: int = 0
    errors: int = 0
    lines: list[str] = field(default_factory=list)


# ================================================================ leaf tools

def lookup_regulation(show_id: str, topic: str) -> str:
    """Stub. Saturday this becomes hybrid retrieval over real exhibitor manuals."""
    return json.dumps({
        "clause": "3.2.1",
        "rule": "Island stands may not exceed 4.0 m. Structures above 2.5 m require written approval.",
        "source": f"{show_id} Technical Guidelines, s.3.2",
    })

KNOWN_FORMS = {"stand design approval", "rigging order", "electrical order"}

def get_deadline(show_id: str, form_name: str) -> str:
    if form_name.lower().strip() not in KNOWN_FORMS:
        return json.dumps({"error": f"No form '{form_name}' for {show_id}. Known: {sorted(KNOWN_FORMS)}"})
    return json.dumps({"form": form_name, "deadline": "2027-01-15", "late_fee_pct": 25})

REGULATION_TOOL = Tool(
    name="lookup_regulation",
    description=(
        "Look up a booth construction or safety regulation for a specific show. "
        "Use for questions about height limits, rigging, electrical, fire safety, "
        "flooring, or structural rules."
    ),
    args=RegulationArgs,
    fn=lambda **kw: TOOL_IMPL["lookup_regulation"](**kw),   # late lookup: tests can patch
)

DEADLINE_TOOL = Tool(
    name="get_deadline",
    description="Get the submission deadline for a required exhibitor form.",
    args=DeadlineArgs,
    fn=lambda **kw: TOOL_IMPL["get_deadline"](**kw),
)


# --- module-level registry, kept so the existing test suite still passes -------

TOOL_IMPL: dict[str, Callable] = {
    "lookup_regulation": lookup_regulation,
    "get_deadline": get_deadline,
}

SCHEMAS: dict[str, Type[BaseModel]] = {
    "lookup_regulation": RegulationArgs,
    "get_deadline": DeadlineArgs,
}

TOOLS = [REGULATION_TOOL.spec(), DEADLINE_TOOL.spec()]


# ================================================================ guards

def _run_guarded(name: str, fn: Callable, args_model: Type[BaseModel], raw_input: dict) -> tuple[str, bool]:
    try:
        args = args_model(**raw_input)
    except ValidationError as e:
        return f"Invalid arguments for '{name}': {e}", True

    ex = concurrent.futures.ThreadPoolExecutor(1)
    try:
        fut = ex.submit(fn, **args.model_dump())
        out = fut.result(timeout=TOOL_TIMEOUT_S)
        ex.shutdown(wait=False)
    except concurrent.futures.TimeoutError:
        ex.shutdown(wait=False, cancel_futures=True)
        return f"Tool '{name}' timed out after {TOOL_TIMEOUT_S}s. Try a narrower query.", True
    except Exception as e:
        ex.shutdown(wait=False)
        return f"Tool '{name}' failed: {type(e).__name__}: {e}", True

    if len(out) > MAX_TOOL_OUTPUT:
        out = out[:MAX_TOOL_OUTPUT] + "\n...[truncated, narrow your query]"
    return out, False


def execute_tool(name: str, raw_input: dict) -> tuple[str, bool]:
    """Default-registry entry point. Reads TOOL_IMPL at call time so tests can patch it."""
    if name not in TOOL_IMPL:
        return f"No such tool '{name}'. Available: {list(TOOL_IMPL)}", True
    return _run_guarded(name, TOOL_IMPL[name], SCHEMAS[name], raw_input)


def call_signature(name: str, raw_input: dict) -> str:
    return f"{name}:{json.dumps(raw_input, sort_keys=True)}"


# ================================================================ the loop

def run_loop(
    user_msg: str,
    tools: list[Tool] | None = None,
    system: str | None = None,
    max_iterations: int = 8,
    budget: Budget | None = None,
    trace: Trace | None = None,
    depth: int = 0,
    label: str = "agent",
    verbose: bool = True,
) -> str:
    """One agent. A multi-agent system is this function composed with itself."""
    tools = tools or []
    budget = budget or Budget()
    trace = trace or Trace()
    by_name = {t.name: t for t in tools}
    messages = [{"role": "user", "content": user_msg}]
    seen: dict[str, int] = {}
    indent = "  " * depth

    if depth > MAX_DEPTH:
        return f"Delegation depth limit ({MAX_DEPTH}) reached; refusing to nest further."

    def create():
        kw = {}
        if system:
            kw["system"] = system
        if tools:
            kw["tools"] = [t.spec() for t in tools]
        resp = client.messages.create(model=MODEL, max_tokens=2000, messages=messages, **kw)
        budget.charge(resp)
        trace.turns += 1
        return resp

    def wrap_up(reason: str) -> str:
        messages.append({"role": "user", "content": reason})
        final = client.messages.create(model=MODEL, max_tokens=2000, messages=messages)
        budget.charge(final)
        return "".join(b.text for b in final.content if b.type == "text")

    for i in range(max_iterations):
        if budget.exhausted:
            return wrap_up("Cost budget exhausted. Summarise what you have and stop.")

        resp = create()
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")

        results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            trace.tool_calls += 1

            sig = call_signature(block.name, block.input)
            seen[sig] = seen.get(sig, 0) + 1

            if seen[sig] > REPEAT_LIMIT:
                out, is_error = (
                    f"You have already called '{block.name}' with these exact arguments "
                    f"{REPEAT_LIMIT} times. Use the previous result or try something different.",
                    True,
                )
            elif block.name in by_name:
                t = by_name[block.name]
                out, is_error = _run_guarded(block.name, t.fn, t.args, block.input)
            else:
                out, is_error = f"No such tool '{block.name}'. Available: {list(by_name)}", True

            if is_error:
                trace.errors += 1
            line = f"{indent}[{label}:{i}] → {block.name}({block.input}){' ✗' if is_error else ''}"
            trace.lines.append(line)
            if verbose:
                print(line)

            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": out,
                "is_error": is_error,
            })

        messages.append({"role": "user", "content": results})

    return wrap_up(
        f"You have reached the {max_iterations}-step limit. Answer as best you can from "
        "what you already have, and state clearly what you could not determine."
    )


def run_agent(user_msg: str, max_iterations: int = 8, verbose: bool = True) -> str:
    """Single-agent entry point. The existing test suite uses this."""
    budget = Budget()
    out = run_loop(user_msg, tools=[REGULATION_TOOL, DEADLINE_TOOL],
                   max_iterations=max_iterations, budget=budget,
                   label="single", verbose=verbose)
    if verbose:
        print(f"\n[${budget.spent:.4f} · {budget.calls} model calls]")
    return out


# ================================================================ specialists

def as_tool(fn: Callable, name: str, description: str, args: Type[BaseModel]) -> Tool:
    """Wrap a specialist agent so the supervisor can call it like any other tool."""
    return Tool(name=name, description=description, args=args, fn=fn)


class ComplianceArgs(BaseModel):
    show_id: str
    stand_spec: str


class DeadlineTaskArgs(BaseModel):
    show_id: str
    requirements: str


class DraftArgs(BaseModel):
    findings: str
    tone: str = "professional and direct"


def make_specialists(budget: Budget, trace: Trace, depth: int = 0, verbose: bool = True) -> list[Tool]:
    """Specialists close over the shared budget and trace so cost rolls up across the tree."""

    def compliance_agent(show_id: str, stand_spec: str) -> str:
        return run_loop(
            user_msg=f"Show: {show_id}\nStand spec:\n{stand_spec}",
            tools=[REGULATION_TOOL],
            system=("You verify stand designs against show technical guidelines. "
                    "Cite the clause number for every finding. If a rule is not found, "
                    "say so explicitly — never infer a rule that was not returned."),
            max_iterations=6, budget=budget, trace=trace, depth=depth + 1,
            label="compliance", verbose=verbose,
        )

    def deadline_agent(show_id: str, requirements: str) -> str:
        return run_loop(
            user_msg=f"Show: {show_id}\nRequirements identified:\n{requirements}",
            tools=[DEADLINE_TOOL],
            system=("You determine which exhibitor forms are required and when they are due. "
                    "Return a compact list: form, deadline, late penalty."),
            max_iterations=6, budget=budget, trace=trace, depth=depth + 1,
            label="deadline", verbose=verbose,
        )

    def drafting_agent(findings: str, tone: str = "professional and direct") -> str:
        return run_loop(
            user_msg=f"Tone: {tone}\n\nFindings:\n{findings}",
            tools=[],                                    # no tools: writing only
            system=("You write replies to exhibitors. Be concrete about actions and dates. "
                    "Preserve every clause citation exactly as given. Add nothing that is "
                    "not in the findings."),
            max_iterations=2, budget=budget, trace=trace, depth=depth + 1,
            label="drafting", verbose=verbose,
        )

    return [
        as_tool(compliance_agent, "check_compliance",
                "Check a stand design against show regulations. Returns findings with clause citations.",
                ComplianceArgs),
        as_tool(deadline_agent, "determine_deadlines",
                "Determine which forms are required and when they are due.",
                DeadlineTaskArgs),
        as_tool(drafting_agent, "draft_reply",
                "Draft an exhibitor-facing reply from a set of findings. Call this last.",
                DraftArgs),
    ]


SUPERVISOR_SYSTEM = (
    "You coordinate specialists to answer exhibitor questions. "
    "Check compliance first, then determine deadlines for anything that requires a form, "
    "then draft the reply. Do not answer regulation questions yourself — you have no "
    "access to the guidelines. Return the drafted reply as your final answer."
)


def run_supervisor(user_msg: str, verbose: bool = True) -> tuple[str, Budget, Trace]:
    budget, trace = Budget(limit_usd=1.00), Trace()
    specialists = make_specialists(budget, trace, depth=0, verbose=verbose)
    out = run_loop(user_msg, tools=specialists, system=SUPERVISOR_SYSTEM,
                   max_iterations=8, budget=budget, trace=trace,
                   label="supervisor", verbose=verbose)
    return out, budget, trace


# ================================================================ the measurement

QUESTION = ("I'm exhibiting at Ambiente 2027 with a 6x6 island stand and a 3.5m hanging sign. "
            "What do I need to do and by when?")


def compare(question: str = QUESTION) -> None:
    """Friday's actual deliverable: does splitting beat a single agent?"""
    print("=" * 70 + "\nSINGLE AGENT\n" + "=" * 70)
    b1, t1 = Budget(), Trace()
    start = time.monotonic()
    single = run_loop(question, tools=[REGULATION_TOOL, DEADLINE_TOOL],
                      budget=b1, trace=t1, label="single")
    single_s = time.monotonic() - start

    print("\n" + "=" * 70 + "\nSUPERVISOR + 3 SPECIALISTS\n" + "=" * 70)
    start = time.monotonic()
    multi, b2, t2 = run_supervisor(question)
    multi_s = time.monotonic() - start

    print(f"""
{'=' * 70}
                    single      multi     delta
model calls        {b1.calls:>7}    {b2.calls:>7}   {b2.calls - b1.calls:+}
tool calls         {t1.tool_calls:>7}    {t2.tool_calls:>7}   {t2.tool_calls - t1.tool_calls:+}
errors             {t1.errors:>7}    {t2.errors:>7}   {t2.errors - t1.errors:+}
cost (USD)         {b1.spent:>7.4f}    {b2.spent:>7.4f}   {b2.spent - b1.spent:+.4f}
wall time (s)      {single_s:>7.1f}    {multi_s:>7.1f}   {multi_s - single_s:+.1f}
answer length      {len(single):>7}    {len(multi):>7}   {len(multi) - len(single):+}
{'=' * 70}

Judge correctness yourself. Record the verdict in ARCHITECTURE.md — including
if the single agent wins.
""")
    print("--- SINGLE ---\n" + single + "\n\n--- MULTI ---\n" + multi)


if __name__ == "__main__":
    compare()