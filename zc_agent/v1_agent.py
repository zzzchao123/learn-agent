import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load configuration from .env file
load_dotenv()

try:
    from anthropic import Anthropic
except ImportError:
    sys.exit("Please install: pip install anthropic python-dotenv")

API_KEY = os.getenv("ANTHROPIC_API_KEY")
BASE_URL = os.getenv("ANTHROPIC_BASE_URL")
MODEL = os.getenv("MODEL_NAME", "claude-sonnet-4-20250514")
WORKDIR = Path.cwd()

client = Anthropic(api_key=API_KEY, base_url=BASE_URL) if BASE_URL else Anthropic(api_key=API_KEY)

SYSTEM = f"""You are a coding agent at {WORKDIR}.

Loop: think briefly -> use tools -> report results.

Rules:
- Prefer tools over prose. Act, don't just explain.
- Never invent file paths. Use bash ls/find first if unsure.
- Make minimal changes. Don't over-engineer.
- After finishing, summarize what changed."""

TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command. Use for: cd, git, npm, python, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                }
            },
            "required": ["command"],
        },
    },

    {
        "name": "read_file",
        "description": "Read file contents. Returns UTF-8 text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max lines to read (default: all)"
                },
            },
            "required": ["path"],
        },
    },

    {
        "name": "write_file",
        "description": "Write content to a file. Creates parent directories if needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path for the file"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write"
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in a file. Use for surgical edits.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file"
                },
                "old_text": {
                    "type": "string",
                    "description": "Exact text to find (must match precisely)"
                },
                "new_text": {
                    "type": "string",
                    "description": "Replacement text"
                },
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
]


def safe_path(p: str) -> Path:
    """
    Ensure path stays within workspace (security measure).

    Prevents the model from accessing files outside the project directory.
    Resolves relative paths and checks they don't escape via '../'.
    """
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    """
    Execute shell command with safety checks.

    Security: Blocks obviously dangerous commands.
    Timeout: 60 seconds to prevent hanging.
    Output: Truncated to 50KB to prevent context overflow.
    """
    dangerous = [
        "del /f /s /q ",
        "rd /s /q ",
        "Remove-Item -Recurse -Force",

        "runas /user:Administrator",
        "Start-Process -Verb RunAs",

        "shutdown /s",
        "Stop-Computer",

        # 对应 reboot（系统重启）
        "shutdown /r",
        "Restart-Computer",

        # 对应 > /dev/（写入/覆盖系统核心、破坏性输出重定向/格式化）
        "> C:\\Windows\\",
        "> %SystemRoot%\\",
        "format ",
        "format /q /y"
    ]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=60
        )
        output = (result.stdout + result.stderr).strip()
        return output[:5000] if output else "(no output)"

    except subprocess.TimeoutExpired:
        return "Error: Command timed out (60s)"
    except Exception as e:
        return f"Error: {e}"


def run_read(path: str, limit: int = None) -> str:
    """
    Read file contents with optional line limit.

    For large files, use limit to read just the first N lines.
    Output truncated to 50KB to prevent context overflow.
    """
    try:
        text = safe_path(path).read_text(encoding="utf-8")
        lines = text.splitlines()

        if limit and limit < len(lines):
            lines = lines[:limit]
            lines.append(f"... ({len(text.splitlines()) - limit} more lines)")

        return "\n".join(lines)[:50000]

    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """
    Write content to file, creating parent directories if needed.

    This is for complete file creation/overwrite.
    For partial edits, use edit_file instead.
    """
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """
    Replace exact text in a file (surgical edit).

    Uses exact string matching - the old_text must appear verbatim.
    Only replaces the first occurrence to prevent accidental mass changes.
    """
    try:
        fp = safe_path(path)
        content = fp.read_text()

        if old_text not in content:
            return f"Error: Text not found in {path}"

        new_content = content.replace(old_text, new_text, 1)
        fp.write_text(new_content, encoding="utf-8")
        return f"Edited {path}"

    except Exception as e:
        return f"Error: {e}"


def execute_tool(name: str, args: dict) -> str:
    """
    Dispatch tool call to the appropriate implementation.

    This is the bridge between the model's tool calls and actual execution.
    Each tool returns a string result that goes back to the model.
    """
    if name == "bash":
        return run_bash(args["command"])
    if name == "read_file":
        return run_read(args["path"], args.get("limit"))
    if name == "write_file":
        return run_write(args["path"], args["content"])
    if name == "edit_file":
        return run_edit(args["path"], args["old_text"], args["new_text"])
    return f"Unknown tool: {name}"


def agent_loop(messages: list) -> list:
    """
    The complete agent in one function.

    This is the pattern that ALL coding agents share:

        while True:
            response = model(messages, tools)
            if no tool calls: return
            execute tools, append results, continue

    The model controls the loop:
      - Keeps calling tools until stop_reason != "tool_use"
      - Results become context (fed back as "user" messages)
      - Memory is automatic (messages list accumulates history)

    Why this works:
      1. Model decides which tools, in what order, when to stop
      2. Tool results provide feedback for next decision
      3. Conversation history maintains context across turns
    """
    while True:
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )
        tool_calls = []
        for block in response.content:
            if hasattr(block, "text"):
                print(block.text)
            if block.type == "tool_use":
                tool_calls.append(block)

        if response.stop_reason != "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            return messages

        results = []
        for tc in tool_calls:
            print(f"\n> {tc.name}: {tc.input}")

            output = execute_tool(tc.name, tc.input)
            preview = output[:200] + "..." if len(output) > 200 else output
            print(f"  {preview}")

            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": output,
                }
            )
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": results})


def main():
    """
    Simple Read-Eval-Print Loop for interactive use.

    The history list maintains conversation context across turns,
    allowing multi-turn conversations with memory.
    """
    print(f"Mini Claude Code v1 - {WORKDIR}")
    print("Type 'exit' to quit.\n")

    history = []

    while True:
        try:
            user_input = input("You:").strip()
        except:
            break

        if not user_input or user_input.lower() in ("exit", "quit", "q"):
            break

        history.append({"role": "user", "content": user_input})

        try:
            agent_loop(history)
        except Exception as e:
            print(f"Error: {e}")

        print()


if __name__ == '__main__':
    main()
