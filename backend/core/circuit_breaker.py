"""
Circuit breaker implementation for external service resilience.
Prevents cascading failures by temporarily disabling failed services.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class CircuitBreakerState(Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Circuit breaker is open, calls are failing
    HALF_OPEN = "half_open" # Testing if service has recovered

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior"""
    failure_threshold: int = 5          # Number of failures before opening
    timeout: int = 60                   # Seconds to wait before trying again
    half_open_max_calls: int = 3        # Max calls to test in half-open state
    success_threshold: int = 2          # Successes needed to close from half-open
    
class CircuitBreakerException(Exception):
    """Exception raised when circuit breaker is open"""
    
    def __init__(self, message: str, retry_after: int):
        super().__init__(message)
        self.retry_after = retry_after

class CircuitBreaker:
    """
    Circuit breaker implementation that prevents cascading failures
    by temporarily disabling failed external services.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        half_open_max_calls: int = 3,
        success_threshold: int = 2,
        name: str = "unknown"
    ):
        self.config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            timeout=timeout,
            half_open_max_calls=half_open_max_calls,
            success_threshold=success_threshold
        )
        self.name = name
        
        # State tracking
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0
        
        # Statistics
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
        self.state_changes: Dict[str, int] = {
            "closed_to_open": 0,
            "open_to_half_open": 0,
            "half_open_to_closed": 0,
            "half_open_to_open": 0
        }
        
        self._lock = asyncio.Lock()
    
    async def __aenter__(self):
        """Context manager entry - check if call is allowed"""
        async with self._lock:
            self.total_calls += 1
            
            if self.state == CircuitBreakerState.OPEN:
                # Check if timeout has passed
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    remaining_timeout = self._get_remaining_timeout()
                    raise CircuitBreakerException(
                        f"Circuit breaker '{self.name}' is open",
                        retry_after=remaining_timeout
                    )
            
            elif self.state == CircuitBreakerState.HALF_OPEN:
                # Check if we've exceeded half-open call limit
                if self.half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitBreakerException(
                        f"Circuit breaker '{self.name}' is half-open with max calls reached",
                        retry_after=self.config.timeout
                    )
                
                self.half_open_calls += 1
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - record success or failure"""
        async with self._lock:
            if exc_type is None:
                # Success
                await self._record_success()
            else:
                # Failure (unless it's our own CircuitBreakerException)
                if not isinstance(exc_val, CircuitBreakerException):
                    await self._record_failure()
    
    async def _record_success(self):
        """Record a successful operation"""
        self.total_successes += 1
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
        
        elif self.state == CircuitBreakerState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
        
        logger.debug(f"Circuit breaker '{self.name}' recorded success. State: {self.state.value}")
    
    async def _record_failure(self):
        """Record a failed operation"""
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self._transition_to_open()
        
        elif self.state == CircuitBreakerState.HALF_OPEN:
            # Any failure in half-open state reopens the circuit
            self._transition_to_open()
        
        logger.warning(f"Circuit breaker '{self.name}' recorded failure. State: {self.state.value}, Failures: {self.failure_count}")
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = datetime.utcnow() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.timeout
    
    def _get_remaining_timeout(self) -> int:
        """Get remaining timeout in seconds"""
        if self.last_failure_time is None:
            return 0
        
        time_since_failure = datetime.utcnow() - self.last_failure_time
        remaining = self.config.timeout - time_since_failure.total_seconds()
        return max(0, int(remaining))
    
    def _transition_to_open(self):
        """Transition to open state"""
        old_state = self.state
        self.state = CircuitBreakerState.OPEN
        self.success_count = 0
        self.half_open_calls = 0
        
        if old_state == CircuitBreakerState.CLOSED:
            self.state_changes["closed_to_open"] += 1
        elif old_state == CircuitBreakerState.HALF_OPEN:
            self.state_changes["half_open_to_open"] += 1
        
        logger.warning(f"Circuit breaker '{self.name}' opened after {self.failure_count} failures")
    
    def _transition_to_half_open(self):
        """Transition to half-open state"""
        self.state = CircuitBreakerState.HALF_OPEN
        self.half_open_calls = 0
        self.success_count = 0
        self.state_changes["open_to_half_open"] += 1
        
        logger.info(f"Circuit breaker '{self.name}' transitioned to half-open")
    
    def _transition_to_closed(self):
        """Transition to closed state"""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        self.state_changes["half_open_to_closed"] += 1
        
        logger.info(f"Circuit breaker '{self.name}' closed after successful recovery")
    
    def is_open(self) -> bool:
        """Check if circuit breaker is open"""
        return self.state == CircuitBreakerState.OPEN
    
    def is_half_open(self) -> bool:
        """Check if circuit breaker is half-open"""
        return self.state == CircuitBreakerState.HALF_OPEN
    
    def is_closed(self) -> bool:
        """Check if circuit breaker is closed"""
        return self.state == CircuitBreakerState.CLOSED
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        success_rate = 0.0
        if self.total_calls > 0:
            success_rate = self.total_successes / self.total_calls
        
        return {
            "name": self.name,
            "state": self.state.value,
            "total_calls": self.total_calls,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "success_rate": success_rate,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "state_changes": self.state_changes.copy(),
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "timeout": self.config.timeout,
                "half_open_max_calls": self.config.half_open_max_calls,
                "success_threshold": self.config.success_threshold
            }
        }
    
    def reset(self):
        """Manually reset the circuit breaker to closed state"""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        self.last_failure_time = None
        
        logger.info(f"Circuit breaker '{self.name}' manually reset")

class CircuitBreakerManager:
    """
    Manager for multiple circuit breakers across different services
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    def get_circuit_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout: int = 60,
        half_open_max_calls: int = 3,
        success_threshold: int = 2
    ) -> CircuitBreaker:
        """Get or create a circuit breaker for a service"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(
                failure_threshold=failure_threshold,
                timeout=timeout,
                half_open_max_calls=half_open_max_calls,
                success_threshold=success_threshold,
                name=name
            )
        
        return self.circuit_breakers[name]
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers"""
        return {
            name: cb.get_stats()
            for name, cb in self.circuit_breakers.items()
        }
    
    def reset_all(self):
        """Reset all circuit breakers"""
        for cb in self.circuit_breakers.values():
            cb.reset()
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary"""
        total_breakers = len(self.circuit_breakers)
        open_breakers = sum(1 for cb in self.circuit_breakers.values() if cb.is_open())
        half_open_breakers = sum(1 for cb in self.circuit_breakers.values() if cb.is_half_open())
        
        return {
            "total_circuit_breakers": total_breakers,
            "open_circuit_breakers": open_breakers,
            "half_open_circuit_breakers": half_open_breakers,
            "healthy_services": total_breakers - open_breakers - half_open_breakers,
            "overall_health": "healthy" if open_breakers == 0 else "degraded" if open_breakers < total_breakers / 2 else "unhealthy"
        }

# Global circuit breaker manager
circuit_breaker_manager = CircuitBreakerManager()

# Decorator for easy circuit breaker usage
def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    timeout: int = 60,
    half_open_max_calls: int = 3,
    success_threshold: int = 2
):
    """
    Decorator to add circuit breaker protection to a function
    
    Usage:
        @circuit_breaker("external_api", failure_threshold=3, timeout=30)
        async def call_external_api():
            # Your API call here
            pass
    """
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            cb = circuit_breaker_manager.get_circuit_breaker(
                name=name,
                failure_threshold=failure_threshold,
                timeout=timeout,
                half_open_max_calls=half_open_max_calls,
                success_threshold=success_threshold
            )
            
            async with cb:
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator 