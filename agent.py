from dotenv import load_dotenv; load_dotenv()
import json
import concurrent.futures
from anthropic import Anthropic
from pydantic import BaseModel, ValidationError

client = Anthropic()
MODEL = "claude-sonnet-4-5"
TOOL_TIMEOUT_S = 10
MAX_TOOL_OUTPUT = 50_000          # chars, before truncation
REPEAT_LIMIT = 3
COST_PER_MTOK = {"input": 3.00, "output": 15.00}   # check current pricing


# ---------------------------------------------------------------- schemas

class RegulationArgs(BaseModel):
    show_id: str
    topic: str


class DeadlineArgs(BaseModel):
    show_id: str
    form_name: str


SCHEMAS = {
    "lookup_regulation": RegulationArgs,
    "get_deadline": DeadlineArgs,
}


# ---------------------------------------------------------------- tools

TOOLS = [
    {
        "name": "lookup_regulation",
        "description": (
            "Look up a booth construction or safety regulation for a specific show. "
            "Use for questions about height limits, rigging, electrical, fire safety, "
            "flooring, or structural rules."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "show_id": {"type": "string", "description": "Show identifier, e.g. 'ambiente-2027'"},
                "topic": {"type": "string", "description": "e.g. 'booth height', 'rigging', 'electrical load'"},
            },
            "required": ["show_id", "topic"],
        },
    },
    {
        "name": "get_deadline",
        "description": "Get the submission deadline for a required exhibitor form.",
        "input_schema": {
            "type": "object",
            "properties": {
                "show_id": {"type": "string"},
                "form_name": {"type": "string", "description": "e.g. 'stand design approval', 'rigging order'"},
            },
            "required": ["show_id", "form_name"],
        },
    },
]


# Stubs for now. Saturday these become real retrieval.
def lookup_regulation(show_id: str, topic: str) -> str:
    return json.dumps({
        "clause": "3.2.1",
        "rule": "Island stands may not exceed 4.0 m. Structures above 2.5 m require written approval.",
        "source": f"{show_id} Technical Guidelines, s.3.2",
    })


def get_deadline(show_id: str, form_name: str) -> str:
    return json.dumps({"form": form_name, "deadline": "2027-01-15", "late_fee_pct": 25})


TOOL_IMPL = {
    "lookup_regulation": lookup_regulation,
    "get_deadline": get_deadline,
}


# ---------------------------------------------------------------- guards

def execute_tool(name: str, raw_input: dict) -> tuple[str, bool]:
    """Run a tool behind every guard. Returns (content, is_error). Never raises."""

    # 1. unknown tool
    if name not in TOOL_IMPL:
        return f"No such tool '{name}'. Available: {list(TOOL_IMPL)}", True

    # 2. validate args before executing
    try:
        args = SCHEMAS[name](**raw_input)
    except ValidationError as e:
        return f"Invalid arguments for '{name}': {e}", True

    # 3. timeout, 4. never crash
    ex = concurrent.futures.ThreadPoolExecutor(1)
    try:
        fut = ex.submit(TOOL_IMPL[name], **args.model_dump())
        out = fut.result(timeout=TOOL_TIMEOUT_S)
        ex.shutdown(wait=False)
    except concurrent.futures.TimeoutError:
        ex.shutdown(wait=False, cancel_futures=True)
        return f"Tool '{name}' timed out after {TOOL_TIMEOUT_S}s. Try a narrower query.", True
    except Exception as e:
        ex.shutdown(wait=False)
        return f"Tool '{name}' failed: {type(e).__name__}: {e}", True

    # 5. truncate oversized output so it can't blow the context
    if len(out) > MAX_TOOL_OUTPUT:
        out = out[:MAX_TOOL_OUTPUT] + "\n...[truncated, narrow your query]"
    return out, False


def call_signature(name: str, raw_input: dict) -> str:
    return f"{name}:{json.dumps(raw_input, sort_keys=True)}"


# ---------------------------------------------------------------- loop

def run_agent(user_msg: str, max_iterations: int = 8, verbose: bool = True) -> str:
    messages = [{"role": "user", "content": user_msg}]
    cost = 0.0
    seen: dict[str, int] = {}

    def add_cost(resp):
        nonlocal cost
        cost += (resp.usage.input_tokens * COST_PER_MTOK["input"]
                 + resp.usage.output_tokens * COST_PER_MTOK["output"]) / 1_000_000

    for i in range(max_iterations):
        resp = client.messages.create(
            model=MODEL, max_tokens=2000, tools=TOOLS, messages=messages,
        )
        add_cost(resp)
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            if verbose:
                print(f"\n[done in {i + 1} turns · ${cost:.4f}]")
            return "".join(b.text for b in resp.content if b.type == "text")

        results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue

            # 6. repetition guard
            sig = call_signature(block.name, block.input)
            seen[sig] = seen.get(sig, 0) + 1
            if seen[sig] > REPEAT_LIMIT:
                out, is_error = (
                    f"You have already called '{block.name}' with these exact arguments "
                    f"{REPEAT_LIMIT} times. Use the previous result or try something different.",
                    True,
                )
            else:
                out, is_error = execute_tool(block.name, block.input)

            if verbose:
                flag = " ✗" if is_error else ""
                print(f"  [{i}] → {block.name}({block.input}){flag}")

            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": out,
                "is_error": is_error,
            })

        messages.append({"role": "user", "content": results})

    # graceful degradation: one final call with tools removed
    messages.append({
        "role": "user",
        "content": (
            f"You have reached the {max_iterations}-step limit. "
            "Answer as best you can from what you already have, and state clearly "
            "what you could not determine."
        ),
    })
    final = client.messages.create(model=MODEL, max_tokens=2000, messages=messages)
    add_cost(final)
    if verbose:
        print(f"\n[hit iteration limit · ${cost:.4f}]")
    return "".join(b.text for b in final.content if b.type == "text")


if __name__ == "__main__":
    print(run_agent(
        "I'm exhibiting at Ambiente 2027 with a 6x6 island stand and a 3.5m hanging sign. "
        "What do I need to do and by when?"
    ))