"""
core/code_validator.py

Extracts code blocks from agent responses and runs them if they are Python.
Non-Python languages (Java, C++, JS etc.) are detected and skipped gracefully
— the orchestrator treats them as has_code=False and skips LLM escalation.
"""

import re
import subprocess
import sys
import textwrap

# Captures optional language tag + code body from fenced blocks
_CODE_BLOCK_RE = re.compile(r"```(\w*)\s*\n(.*?)```", re.DOTALL)

# Only these tags are executed — everything else is skipped
_EXECUTABLE_LANGS = {"python", "py", ""}

TIMEOUT_SECONDS = 10


def extract_code(text: str) -> tuple[str | None, str | None]:
    """
    Returns (language, code) for the first code block found.
    language is lowercased tag (e.g. 'java', 'python', '').
    Returns (None, None) if no code block found.
    """
    match = _CODE_BLOCK_RE.search(text)
    if not match:
        return None, None
    lang = match.group(1).strip().lower()
    code = textwrap.dedent(match.group(2)).strip()
    return lang, code


def run_code(code: str) -> dict:
    """Execute Python code in a subprocess. Returns {passed, output}."""
    # Strip __main__ guards so code actually executes
    cleaned = re.sub(
        r'if\s+__name__\s*==\s*["\']__main__["\'].*', "", code, flags=re.DOTALL
    ).strip()

    if not cleaned:
        return {"passed": True, "output": "(no executable statements outside __main__ guard)"}

    try:
        result = subprocess.run(
            [sys.executable, "-c", cleaned],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
        if result.returncode == 0:
            return {"passed": True, "output": result.stdout.strip() or "(no output)"}
        else:
            error = result.stderr.strip() or result.stdout.strip()
            return {"passed": False, "output": error}

    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "output": f"Code timed out after {TIMEOUT_SECONDS}s (possible infinite loop)",
        }
    except Exception as e:
        return {"passed": False, "output": f"Execution error: {e}"}


def validate_code_response(response: str) -> dict:
    """
    Full validation pipeline for a coding agent response.
    Returns:
        {
            "has_code":   bool  — True only if Python code was found and executed
            "passed":     bool  — True if execution succeeded
            "output":     str   — stdout or error message
            "language":   str   — detected language tag
        }
    """
    lang, code = extract_code(response)

    if lang is None:
        return {"has_code": False, "passed": False, "output": "No code block found.", "language": ""}

    if lang not in _EXECUTABLE_LANGS:
        # Non-Python language — don't attempt execution, just acknowledge
        return {
            "has_code": False,
            "passed":   True,
            "output":   f"Non-Python language detected ({lang}), skipping execution.",
            "language": lang,
        }

    run_result = run_code(code)
    return {
        "has_code": True,
        "passed":   run_result["passed"],
        "output":   run_result["output"],
        "language": lang,
    }