from dotenv import load_dotenv; load_dotenv()
from anthropic import Anthropic

client = Anthropic()

resp = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1000,
    tools=[{
        "name": "lookup_regulation",
        "description": "Look up a booth construction or safety regulation for a specific show.",
        "input_schema": {
            "type": "object",
            "properties": {
                "show_id": {"type": "string"},
                "topic": {"type": "string"},
            },
            "required": ["show_id", "topic"],
        },
    }],
    messages=[{
        "role": "user",
        "content": "What's time does the venue open?",
    }],
)

print("stop_reason:", resp.stop_reason)
print("usage:", resp.usage)
print()
for block in resp.content:
    print("--- block type:", block.type)
    print(block)
