"""Testes unitários das métricas e da latência de ponta a ponta."""
import pytest

import candidate_starter.harness as harness
from candidate_starter.harness import (
    compute_precision_at_k,
    compute_router_metrics,
    compute_savings,
    run_harness,
)
from common.schemas import RetrievalResult, RouteResult, ToolMatch


LABELS = ["FAST_PATH", "AGENT"]


def test_router_metrics_and_confusion_matrix() -> None:
    result = compute_router_metrics(
        ["FAST_PATH", "FAST_PATH", "AGENT", "AGENT"],
        ["FAST_PATH", "AGENT", "FAST_PATH", "AGENT"],
        LABELS,
    )

    assert result == {
        "accuracy": 0.5,
        "confusion_matrix": {
            "FAST_PATH": {"FAST_PATH": 1, "AGENT": 1},
            "AGENT": {"FAST_PATH": 1, "AGENT": 1},
        },
    }


def test_router_metrics_reject_invalid_input() -> None:
    with pytest.raises(ValueError, match="mesma quantidade"):
        compute_router_metrics(["FAST_PATH"], [], LABELS)
    with pytest.raises(ValueError, match="desconhecidos"):
        compute_router_metrics(["UNKNOWN"], ["FAST_PATH"], LABELS)


def test_precision_at_k() -> None:
    assert compute_precision_at_k([1, 0, 1, 1]) == pytest.approx(0.75)
    assert compute_precision_at_k([]) == 0.0
    with pytest.raises(ValueError, match="0 ou 1"):
        compute_precision_at_k([2])


def test_savings_and_zero_baseline() -> None:
    assert compute_savings(25, 60, 100, 100) == {
        "cost_savings_pct": 75.0,
        "latency_savings_pct": 40.0,
    }
    assert compute_savings(10, 20, 0, 0) == {
        "cost_savings_pct": 0.0,
        "latency_savings_pct": 0.0,
    }


def test_run_harness_counts_agent_llm_latency(monkeypatch) -> None:
    class AgentRouter:
        def predict(self, query: str) -> RouteResult:
            return RouteResult(route="AGENT", latency_ms=2.0, confidence=1.0)

    class Retriever:
        def search(self, query: str, k: int = 2) -> RetrievalResult:
            return RetrievalResult(
                matches=[ToolMatch(name="consultar_saldo", score=0.9)],
                latency_ms=3.0,
            )

    clock = iter([10.0, 10.007, 20.0, 20.011])
    monkeypatch.setattr(harness.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        harness,
        "simulate_agent_llm_call",
        lambda query, tool_name: {"cost_usd": 0.01},
    )
    monkeypatch.setattr(
        harness,
        "simulate_baseline_llm_call",
        lambda query: {"cost_usd": 0.03},
    )

    report = run_harness(
        AgentRouter(),
        Retriever(),
        tools=[],
        eval_dataset=[
            {
                "query": "Quero saber meu saldo",
                "expected_route": "AGENT",
                "expected_tool": "consultar_saldo",
            }
        ],
    )

    assert report["smart_pipeline"]["total_latency_ms"] == pytest.approx(12.0)
    assert report["baseline_always_llm"]["total_latency_ms"] == pytest.approx(11.0)
    assert report["latency_breakdown"]["agent_llm_total_ms"] == pytest.approx(7.0)
