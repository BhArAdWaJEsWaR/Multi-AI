#!/usr/bin/env python3
"""Benchmark three response paths for local LLM chat latency and quality.

Modes:
- baseline_7b: direct calls to 7B-class models for each task
- lite_1p5b: direct calls to 1.5B models for each task
- dynamic_loader: the repository's current orchestration + model registry design

The script intentionally does not modify the existing project files. It is a
standalone benchmarking harness for collecting metric numbers you can paste into
reports or slide decks.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import sys
import time
from typing import Any

import requests

from config import TIERED_MODELS

OLLAMA_URL = "http://localhost:11434"

TASKS = [
    {
        "name": "memory_name",
        "prompt": "my name is Bharadwaj",
        "expected_task": "general",
    },
    {
        "name": "memory_preference",
        "prompt": "I prefer recursion over iteration",
        "expected_task": "general",
    },
    {
        "name": "coding_sum",
        "prompt": "Write a Python function that returns the sum of the first n natural numbers using recursion.",
        "expected_task": "coding",
    },
    {
        "name": "coding_factorial",
        "prompt": "Give me a recursive factorial function in Python.",
        "expected_task": "coding",
    },
    {
        "name": "math_basic",
        "prompt": "What is 27 * 13?",
        "expected_task": "math",
    },
    {
        "name": "math_derive",
        "prompt": "Solve the derivative of x^3 + 4x^2 - 7x + 9.",
        "expected_task": "math",
    },
    {
        "name": "writing_email",
        "prompt": "Write a short professional email asking for a deadline extension.",
        "expected_task": "writing",
    },
    {
        "name": "general_fact",
        "prompt": "Explain what a hash table is in simple terms.",
        "expected_task": "general",
    },
]


def _call_ollama(model: str, user_prompt: str, system_prompt: str | None = None) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.2},
    }
    response = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=300)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = response.text.strip()
        raise RuntimeError(f"Ollama failed for model '{model}': {detail or exc}") from exc
    data = response.json()
    return data["message"]["content"].strip()


def _loaded_models() -> set[str]:
    try:
        response = requests.get(f"{OLLAMA_URL}/api/ps", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        return {m.get("name") for m in models if isinstance(m, dict) and m.get("name")}
    except Exception:
        return set()


def _ollama_ram_mb() -> float:
    """Return resident memory reported by Ollama for currently loaded models.

    `size` is the model's total resident allocation and includes GPU/CPU
    placement as reported by Ollama. `size_vram` is retained separately in the
    detailed JSON output.
    """
    try:
        response = requests.get(f"{OLLAMA_URL}/api/ps", timeout=5)
        response.raise_for_status()
        return sum(
            float(model.get("size", 0) or 0)
            for model in response.json().get("models", [])
            if isinstance(model, dict)
        ) / (1024 * 1024)
    except (OSError, requests.RequestException, ValueError, TypeError):
        return 0.0


def _ollama_memory_snapshot() -> dict[str, Any]:
    """Capture Ollama's model allocation details for the JSON report."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/ps", timeout=5)
        response.raise_for_status()
        models = []
        for model in response.json().get("models", []):
            if isinstance(model, dict):
                models.append({
                    "name": model.get("name"),
                    "size_mb": round(float(model.get("size", 0) or 0) / (1024 * 1024), 2),
                    "size_vram_mb": round(float(model.get("size_vram", 0) or 0) / (1024 * 1024), 2),
                    "processor": model.get("processor"),
                })
        return {"total_mb": round(sum(m["size_mb"] for m in models), 2), "models": models}
    except (OSError, requests.RequestException, ValueError, TypeError):
        return {"total_mb": 0.0, "models": []}


def _available_models() -> set[str]:
    response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
    response.raise_for_status()
    return {
        item.get("name")
        for item in response.json().get("models", [])
        if isinstance(item, dict) and item.get("name")
    }


def _peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024.0


def _route_model_for_task(task: str, size: str) -> str:
    if size == "7b":
        mapping = {
            "coding": "qwen2.5-coder:7b",
            "math": "deepseek-r1:7b",
            "writing": "gemma3:4b",
            "general": "qwen3:4b",
        }
    elif size == "1.5b":
        mapping = {
            "coding": TIERED_MODELS["coding"]["tier1"],
            "math": TIERED_MODELS["math"]["tier1"],
            "writing": TIERED_MODELS["writing"]["tier1"],
            "general": TIERED_MODELS["general"]["tier1"],
        }
    else:
        raise ValueError(f"Unsupported size: {size}")
    return mapping.get(task, mapping["general"])


def _validate_response(task: str, prompt: str, response: str) -> bool:
    try:
        from core.code_validator import validate_code_response
    except Exception:
        validate_code_response = None

    try:
        from core.agents import AGENTS
    except Exception:
        AGENTS = None

    if task == "coding" and validate_code_response is not None:
        result = validate_code_response(response)
        return bool(result.get("passed", False))

    if AGENTS is not None:
        validator = AGENTS["cheap_validator"]
        verdict = validator.run(
            f"Task: {task}\nPrompt: {prompt}\nResponse: {response}\n\nGive only CONFIDENCE: HIGH or LOW and ISSUES: ...",
            stream=False,
        )
        return "CONFIDENCE: HIGH" in verdict.upper()

    # Last-resort heuristic: reject empty or obviously broken responses.
    return bool(response.strip()) and (len(response) > 20 or task == "math")


def run_baseline_direct(mode: str, iterations: int = 1) -> dict[str, Any]:
    """Run the direct model baseline for the chosen model tier."""
    latencies: list[float] = []
    route_hits = 0
    validation_hits = 0
    total_model_loads = 0
    peak_total_ram = 0.0
    peak_python_rss = 0.0
    peak_ollama_ram = 0.0
    last_ollama_memory: dict[str, Any] = {}

    for task in TASKS:
        for _ in range(iterations):
            print(f"[{mode}] {task['name']} ({task['expected_task']})...", flush=True)
            before = _loaded_models()
            start = time.perf_counter()
            model = _route_model_for_task(task["expected_task"], mode)
            response = _call_ollama(
                model,
                task["prompt"],
                system_prompt="You are a helpful assistant.",
            )
            latency = time.perf_counter() - start
            after = _loaded_models()
            total_model_loads += max(0, len(after - before))
            latencies.append(latency)
            python_rss = _peak_rss_mb()
            ollama_ram = _ollama_ram_mb()
            peak_python_rss = max(peak_python_rss, python_rss)
            peak_ollama_ram = max(peak_ollama_ram, ollama_ram)
            peak_total_ram = max(peak_total_ram, python_rss + ollama_ram)
            last_ollama_memory = _ollama_memory_snapshot()

            route_hits += 1 if model.startswith(_route_model_for_task(task["expected_task"], mode).split(":")[0]) else 0
            validation_hits += 1 if _validate_response(task["expected_task"], task["prompt"], response) else 0

    return {
        "avg_latency": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "peak_ram_mb": round(peak_total_ram, 2),
        "peak_python_rss_mb": round(peak_python_rss, 2),
        "peak_ollama_ram_mb": round(peak_ollama_ram, 2),
        "last_ollama_memory": last_ollama_memory,
        "model_loads_per_query": round(total_model_loads / max(1, len(TASKS) * iterations), 3),
        "routing_accuracy": round(route_hits / max(1, len(TASKS) * iterations), 3),
        "validation_success_rate": round(validation_hits / max(1, len(TASKS) * iterations), 3),
        "escalation_rate": 0.0,
    }


def run_dynamic_loader(iterations: int = 1) -> dict[str, Any]:
    try:
        from core.orchestrator import Orchestrator
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"dynamic_loader benchmark requires project dependencies: {exc}") from exc

    latencies: list[float] = []
    route_hits = 0
    validation_hits = 0
    total_model_loads = 0
    escalations = 0
    peak_total_ram = 0.0
    peak_python_rss = 0.0
    peak_ollama_ram = 0.0
    last_ollama_memory: dict[str, Any] = {}

    orch = Orchestrator(use_memory=True, use_validator=False)

    for task in TASKS:
        for _ in range(iterations):
            print(f"[dynamic] {task['name']} ({task['expected_task']})...", flush=True)
            before = _loaded_models()
            start = time.perf_counter()
            result = orch.handle(task["prompt"], verbose=False)
            latency = time.perf_counter() - start
            after = _loaded_models()
            total_model_loads += max(0, len(after - before))
            latencies.append(latency)
            python_rss = _peak_rss_mb()
            ollama_ram = _ollama_ram_mb()
            peak_python_rss = max(peak_python_rss, python_rss)
            peak_ollama_ram = max(peak_ollama_ram, ollama_ram)
            peak_total_ram = max(peak_total_ram, python_rss + ollama_ram)
            last_ollama_memory = _ollama_memory_snapshot()

            response = result.get("response", "")
            actual_task = result.get("task_type")
            route_hits += 1 if actual_task == task["expected_task"] else 0
            validation_hits += 1 if _validate_response(task["expected_task"], task["prompt"], response) else 0
            escalations += 1 if result.get("escalated") else 0

    return {
        "avg_latency": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "peak_ram_mb": round(peak_total_ram, 2),
        "peak_python_rss_mb": round(peak_python_rss, 2),
        "peak_ollama_ram_mb": round(peak_ollama_ram, 2),
        "last_ollama_memory": last_ollama_memory,
        "model_loads_per_query": round(total_model_loads / max(1, len(TASKS) * iterations), 3),
        "routing_accuracy": round(route_hits / max(1, len(TASKS) * iterations), 3),
        "validation_success_rate": round(validation_hits / max(1, len(TASKS) * iterations), 3),
        "escalation_rate": round(escalations / max(1, len(TASKS) * iterations), 3),
    }


def print_metrics_table(results: dict[str, dict[str, Any]]) -> None:
    print("\n| Metric                   | Baseline 7B direct | 1.5B direct | Dynamic loader |")
    print("| ------------------------ | ----------------: | ----------: | -------------: |")
    print(f"| Average response latency | {results['baseline_7b']['avg_latency']} s | {results['lite_1p5b']['avg_latency']} s | {results['dynamic_loader']['avg_latency']} s |")
    print(f"| Peak total resident RAM  | {results['baseline_7b']['peak_ram_mb']} MB | {results['lite_1p5b']['peak_ram_mb']} MB | {results['dynamic_loader']['peak_ram_mb']} MB |")
    print(f"| └─ Python RSS            | {results['baseline_7b']['peak_python_rss_mb']} MB | {results['lite_1p5b']['peak_python_rss_mb']} MB | {results['dynamic_loader']['peak_python_rss_mb']} MB |")
    print(f"| └─ Ollama models         | {results['baseline_7b']['peak_ollama_ram_mb']} MB | {results['lite_1p5b']['peak_ollama_ram_mb']} MB | {results['dynamic_loader']['peak_ollama_ram_mb']} MB |")
    print(f"| Model loads per query    | {results['baseline_7b']['model_loads_per_query']} | {results['lite_1p5b']['model_loads_per_query']} | {results['dynamic_loader']['model_loads_per_query']} |")
    print(f"| Correct task routing     | {results['baseline_7b']['routing_accuracy']:.2%} | {results['lite_1p5b']['routing_accuracy']:.2%} | {results['dynamic_loader']['routing_accuracy']:.2%} |")
    print(f"| Validation success rate  | {results['baseline_7b']['validation_success_rate']:.2%} | {results['lite_1p5b']['validation_success_rate']:.2%} | {results['dynamic_loader']['validation_success_rate']:.2%} |")
    print(f"| Escalation rate          | {results['baseline_7b']['escalation_rate']:.2%} | {results['lite_1p5b']['escalation_rate']:.2%} | {results['dynamic_loader']['escalation_rate']:.2%} |")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark direct 7B vs 1.5B vs dynamic model loading on local Ollama.")
    parser.add_argument("--iterations", type=int, default=1, help="Number of rounds per task.")
    parser.add_argument("--skip-dynamic", action="store_true", help="Skip the dynamic-loader benchmark.")
    args = parser.parse_args()

    try:
        available = _available_models()
        required = {
            _route_model_for_task(task["expected_task"], size)
            for size in ("7b", "1.5b")
            for task in TASKS
        }
        missing = sorted(required - available)
        if missing:
            print("The benchmark cannot start because these Ollama models are missing:")
            for model in missing:
                print(f"  - {model}")
            print("\nInstall the missing models with, for example:")
            print("  ollama pull <model-name>")
            print("\nAvailable models:")
            for model in sorted(available):
                print(f"  - {model}")
            return 2
    except Exception as exc:
        print(f"Ollama is not reachable at {OLLAMA_URL}: {exc}")
        print("Start Ollama first: ollama serve")
        return 2

    results: dict[str, dict[str, Any]] = {}
    try:
        results["baseline_7b"] = run_baseline_direct("7b", iterations=args.iterations)
        results["lite_1p5b"] = run_baseline_direct("1.5b", iterations=args.iterations)
        if not args.skip_dynamic:
            results["dynamic_loader"] = run_dynamic_loader(iterations=args.iterations)
        else:
            results["dynamic_loader"] = {
                "avg_latency": 0.0,
                "peak_ram_mb": 0.0,
                "peak_python_rss_mb": 0.0,
                "peak_ollama_ram_mb": 0.0,
                "last_ollama_memory": {"total_mb": 0.0, "models": []},
                "model_loads_per_query": 0.0,
                "routing_accuracy": 0.0,
                "validation_success_rate": 0.0,
                "escalation_rate": 0.0,
            }
    except (RuntimeError, requests.RequestException, KeyError) as exc:
        print(f"\nBenchmark stopped: {exc}")
        return 1

    print_metrics_table(results)
    print("\nJSON snapshot:")
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
