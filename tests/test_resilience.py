import time

import pytest

from src.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitRegistry,
    CircuitState,
)


def test_circuit_starts_closed():
    breaker = CircuitBreaker(
        provider="groq",
        failure_threshold=3,
        cooldown_s=30,
    )

    assert breaker.state == CircuitState.CLOSED
    assert breaker.allow_request() is True


def test_circuit_opens_after_failure_threshold():
    breaker = CircuitBreaker(
        provider="groq",
        failure_threshold=3,
        cooldown_s=30,
    )

    breaker.record_failure()
    breaker.record_failure()

    assert breaker.state == CircuitState.CLOSED

    breaker.record_failure()

    assert breaker.state == CircuitState.OPEN
    assert breaker.allow_request() is False


def test_open_circuit_blocks_requests():
    breaker = CircuitBreaker(
        provider="groq",
        failure_threshold=1,
        cooldown_s=30,
    )

    breaker.record_failure()

    with pytest.raises(CircuitOpenError):
        breaker.ensure_request_allowed()


def test_success_resets_failure_count():
    breaker = CircuitBreaker(
        provider="groq",
        failure_threshold=3,
        cooldown_s=30,
    )

    breaker.record_failure()
    breaker.record_failure()

    breaker.record_success()

    assert breaker.state == CircuitState.CLOSED
    assert breaker.snapshot().consecutive_failures == 0


def test_open_circuit_moves_to_half_open_after_cooldown():
    breaker = CircuitBreaker(
        provider="groq",
        failure_threshold=1,
        cooldown_s=0.05,
    )

    breaker.record_failure()

    assert breaker.state == CircuitState.OPEN

    time.sleep(0.07)

    assert breaker.state == CircuitState.HALF_OPEN
    assert breaker.allow_request() is True


def test_only_one_half_open_probe_is_allowed():
    breaker = CircuitBreaker(
        provider="groq",
        failure_threshold=1,
        cooldown_s=0.01,
    )

    breaker.record_failure()

    time.sleep(0.02)

    assert breaker.allow_request() is True
    assert breaker.allow_request() is False


def test_half_open_success_closes_circuit():
    breaker = CircuitBreaker(
        provider="groq",
        failure_threshold=1,
        cooldown_s=0.01,
    )

    breaker.record_failure()

    time.sleep(0.02)

    assert breaker.allow_request() is True

    breaker.record_success()

    assert breaker.state == CircuitState.CLOSED
    assert breaker.allow_request() is True


def test_half_open_failure_reopens_circuit():
    breaker = CircuitBreaker(
        provider="groq",
        failure_threshold=1,
        cooldown_s=0.01,
    )

    breaker.record_failure()

    time.sleep(0.02)

    assert breaker.allow_request() is True

    breaker.record_failure()

    assert breaker.state == CircuitState.OPEN


def test_registry_keeps_provider_breakers_independent():
    registry = CircuitRegistry(
        failure_threshold=1,
        cooldown_s=30,
    )

    groq = registry.get("groq")
    mistral = registry.get("mistral")

    groq.record_failure()

    assert groq.state == CircuitState.OPEN
    assert mistral.state == CircuitState.CLOSED


def test_registry_returns_same_breaker():
    registry = CircuitRegistry()

    first = registry.get("groq")
    second = registry.get("groq")

    assert first is second