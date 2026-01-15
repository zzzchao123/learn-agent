import sys

from anthropic import Anthropic
from dotenv import load_dotenv
import os
import subprocess

load_dotenv()

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL")
)
MODEL = os.getenv("MODEL_NAME", "claude-sonnet-4-20250514")

TOOL = [{
    "name": "bash",
    "description": """Execute shell command. Common patterns:
- Read: dir,type, cd
- Subagent: python v0_bash_agent.py 'task description' (spawns isolated agent, returns summary)""",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"]
    }
}]

SYSTEM = f"""You are a CLI agent at {os.getcwd()}. Solve problems using bash commands.

Rules:
- Prefer tools over prose. Act first, explain briefly after.
- Read files: dir,type,cd
- Subagent: For complex subtasks, spawn a subagent to keep context clean:
  python v0_agent.py "explore src/ and summarize the architecture"

When to use subagent:
- Task requires reading many files (isolate the exploration)
- Task is independent and self-contained
- You want to avoid polluting current conversation with intermediate details

The subagent runs in isolation and returns only its final summary."""


def chat(prompt, history=None):
    if history is None:
        history = []

    history.append({"role": "user", "content": prompt})

    while True:
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=history,
            tools=TOOL,
            max_tokens=8000
        )

        content = []
        for block in response.content:
            if hasattr(block, "text"):
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                })
        history.append({"role": "assistant", "content": content})

        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if hasattr(b, "text"))

        results = []
        for block in response.content:
            if block.type == "tool_use":
                cmd = block.input["command"]
                print(f"\033[33m$ {cmd}\033[0m")

                try:
                    out = subprocess.run(
                        cmd,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=300,
                        cwd=os.getcwd()
                    )
                    output = out.stdout + out.stderr
                except subprocess.TimeoutExpired:
                    output = "(timeout after 300s)"

                print(output or "(empty)")
                results.append({
                    "type": "tool_result",
                    "tool_use_id": str(block.id),
                    "content":[{"type": "text", "text": output[:50000]}]  # Truncate very long outputs
                })

        history.append({"role": "user", "content": results})


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(chat(sys.argv[1]))
    else:
        history = []
        while True:
            try:
                query = input("\033[36m>> \033[0m")
            except (EOFError, KeyboardInterrupt):
                break

            if query in ("q", "exit", ""):
                break
            print(chat(query, history))

