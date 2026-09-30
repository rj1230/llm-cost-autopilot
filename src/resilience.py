"""Provider resilience primitives for LLM Cost Autopilot."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from enum import Enum
from typing import Callable, TypeVar


T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised when a provider circuit is currently open."""


class ProviderTimeoutError(TimeoutError):
    """Raised when a provider call exceeds its configured timeout."""


@dataclass
class CircuitSnapshot:
    provider: str
    state: CircuitState
    consecutive_failures: int
    opened_at: float | None
    cooldown_remaining_s: float


class CircuitBreaker:
    """
    Thread-safe circuit breaker for one provider.

    CLOSED:
        Requests are allowed.

    OPEN:
        Requests are blocked until the cooldown expires.

    HALF_OPEN:
        One recovery probe is allowed.
    """

    def __init__(
        self,
        provider: str,
        failure_threshold: int = 3,
        cooldown_s: float = 30.0,
    ) -> None:
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be greater than zero.")

        if cooldown_s < 0:
            raise ValueError("cooldown_s cannot be negative.")

        self.provider = provider
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s

        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._half_open_probe_in_flight = False

        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Return the current circuit state."""

        with self._lock:
            self._refresh_state()
            return self._state

    def _refresh_state(self) -> None:
        """Move OPEN to HALF_OPEN after cooldown."""

        if self._state != CircuitState.OPEN:
            return

        if self._opened_at is None:
            return

        elapsed = time.monotonic() - self._opened_at

        if elapsed >= self.cooldown_s:
            self._state = CircuitState.HALF_OPEN
            self._half_open_probe_in_flight = False

    def allow_request(self) -> bool:
        """
        Determine whether a provider request may execute.

        Only one recovery probe is allowed while HALF_OPEN.
        """

        with self._lock:
            self._refresh_state()

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                return False

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_probe_in_flight:
                    return False

                self._half_open_probe_in_flight = True
                return True

            return False

    def record_success(self) -> None:
        """Record a successful provider request."""

        with self._lock:
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._opened_at = None
            self._half_open_probe_in_flight = False

    def record_failure(self) -> None:
        """Record a failed provider request."""

        with self._lock:
            self._refresh_state()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                self._half_open_probe_in_flight = False
                return

            self._consecutive_failures += 1

            if self._consecutive_failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()

    def snapshot(self) -> CircuitSnapshot:
        """Return operational state for diagnostics and dashboards."""

        with self._lock:
            self._refresh_state()

            if self._state == CircuitState.OPEN:
                if self._opened_at is None:
                    remaining = self.cooldown_s
                else:
                    elapsed = time.monotonic() - self._opened_at
                    remaining = max(
                        0.0,
                        self.cooldown_s - elapsed,
                    )
            else:
                remaining = 0.0

            return CircuitSnapshot(
                provider=self.provider,
                state=self._state,
                consecutive_failures=self._consecutive_failures,
                opened_at=self._opened_at,
                cooldown_remaining_s=remaining,
            )

    def ensure_request_allowed(self) -> None:
        """Raise if the provider circuit is unavailable."""

        if not self.allow_request():
            snapshot = self.snapshot()

            raise CircuitOpenError(
                f"Circuit for provider '{self.provider}' is {snapshot.state.value}."
            )


class CircuitRegistry:
    """Manage independent circuit breakers for each provider."""

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_s: float = 30.0,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s

        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get(self, provider: str) -> CircuitBreaker:
        """Return the circuit breaker for a provider."""

        with self._lock:
            breaker = self._breakers.get(provider)

            if breaker is None:
                breaker = CircuitBreaker(
                    provider=provider,
                    failure_threshold=self.failure_threshold,
                    cooldown_s=self.cooldown_s,
                )

                self._breakers[provider] = breaker

            return breaker

    def snapshots(self) -> dict[str, CircuitSnapshot]:
        """Return snapshots for all registered providers."""

        with self._lock:
            providers = list(self._breakers)

        return {provider: self.get(provider).snapshot() for provider in providers}


def call_with_timeout(
    function: Callable[[], T],
    timeout_s: float,
) -> T:
    """
    Execute a provider operation with a bounded wait.

    The provider operation runs in a worker thread. The caller receives
    ProviderTimeoutError when the configured deadline is exceeded.

    The worker thread itself cannot be forcefully killed by Python.
    Provider SDK-level timeout settings should therefore also be used
    when available.
    """

    if timeout_s <= 0:
        raise ValueError("timeout_s must be greater than zero.")

    executor = ThreadPoolExecutor(max_workers=1)

    future = executor.submit(function)

    try:
        return future.result(timeout=timeout_s)

    except FutureTimeoutError as exc:
        future.cancel()

        raise ProviderTimeoutError(
            f"Provider call exceeded timeout of {timeout_s:.2f}s."
        ) from exc

    finally:
        executor.shutdown(
            wait=False,
            cancel_futures=True,
        )
