
from dotenv import load_dotenv; load_dotenv()
import json
from anthropic import Anthropic

client = Anthropic()
MODEL = "claude-sonnet-5"

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

# Day 1: stubs. Saturday these become real retrieval.
def lookup_regulation(show_id: str, topic: str) -> str:
    return json.dumps({
        "clause": "3.2.1",
        "rule": "Island stands may not exceed 4.0 m. Structures above 2.5 m require written approval.",
        "source": f"{show_id} Technical Guidelines, s.3.2",
    })

def get_deadline(show_id: str, form_name: str) -> str:
    return json.dumps({"form": form_name, "deadline": "2027-01-15", "late_fee_pct": 25})

TOOL_IMPL = {"lookup_regulation": lookup_regulation, "get_deadline": get_deadline}


def run_agent(user_msg: str, max_iterations: int = 8) -> str:
    messages = [{"role": "user", "content": user_msg}]

    for i in range(max_iterations):
        resp = client.messages.create(
            model=MODEL, max_tokens=2000, tools=TOOLS, messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")

        results = []
        for block in resp.content:
            if block.type == "tool_use":
                print(f"  [{i}] → {block.name}({block.input})")
                out = TOOL_IMPL[block.name](**block.input)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": out,
                })
        messages.append({"role": "user", "content": results})

    raise RuntimeError(f"Exceeded {max_iterations} iterations without a final answer")


if __name__ == "__main__":
    print(run_agent(
        "I'm exhibiting at Ambiente 2027 with a 6x6 island stand and a 3.5m hanging sign. "
        "What do I need to do and by when?"
    ))