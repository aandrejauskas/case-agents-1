"""Testes de contrato e validações defensivas do Router local."""
import pytest

from candidate_starter.router import QueryRouter
from common.data_loader import load_router_training_data


@pytest.fixture(scope="module")
def router() -> QueryRouter:
    texts, labels = load_router_training_data()
    return QueryRouter().fit(texts, labels)


def test_predict_returns_route_confidence_and_latency(router: QueryRouter) -> None:
    result = router.predict("Bom dia, qual é o horário de atendimento?")

    assert result.route in {"FAST_PATH", "AGENT"}
    assert result.confidence is not None
    assert 0.0 <= result.confidence <= 1.0
    assert result.latency_ms >= 0.0


def test_predict_requires_fit() -> None:
    with pytest.raises(RuntimeError, match=r"fit\(\)"):
        QueryRouter().predict("Bom dia")


def test_fit_validates_training_data() -> None:
    with pytest.raises(ValueError, match="pelo menos um texto"):
        QueryRouter().fit([], [])
    with pytest.raises(ValueError, match="mesma quantidade"):
        QueryRouter().fit(["texto"], [])
    with pytest.raises(ValueError, match="FAST_PATH e AGENT"):
        QueryRouter().fit(["a", "b"], ["FAST_PATH", "FAST_PATH"])


def test_router_handles_agent_and_fast_path_examples(router: QueryRouter) -> None:
    assert router.predict("Quero consultar meu saldo agora").route == "AGENT"
    assert router.predict("Oi, bom dia!").route == "FAST_PATH"
