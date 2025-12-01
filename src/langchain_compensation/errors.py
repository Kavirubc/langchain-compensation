"""Pluggable error detection strategies for compensation middleware.

This module implements the Strategy pattern for error detection, allowing
developers to customize how tool failures are identified. Each strategy
returns True (error), False (success), or None (defer to next strategy).
"""

import json
from abc import ABC, abstractmethod
from typing import Any, List

from langchain_core.messages import ToolMessage


class ErrorStrategy(ABC):
    """Abstract base class for error detection strategies.

    Each strategy examines a ToolMessage and determines if it represents
    an error condition. Strategies can:
    - Return True: Definitively an error
    - Return False: Definitively NOT an error
    - Return None: Cannot determine, defer to next strategy

    This allows strategies to be chained in priority order.
    """

    @abstractmethod
    def is_error(self, result: ToolMessage) -> bool | None:
        """Determine if the result indicates an error.

        Args:
            result: The ToolMessage returned from tool execution

        Returns:
            True if error detected, False if success detected, None to defer
        """
        pass

    @property
    def name(self) -> str:
        """Human-readable name for logging and debugging."""
        return self.__class__.__name__


class ExplicitStatusStrategy(ErrorStrategy):
    """Check ToolMessage.status attribute for explicit error indication.

    This is the most reliable strategy as it uses LangChain's built-in
    error signaling mechanism. Tools that properly set status='error'
    will be caught here.

    Example:
        ToolMessage(content="Failed", status="error")  # Detected as error
    """

    def is_error(self, result: ToolMessage) -> bool | None:
        if hasattr(result, "status"):
            if result.status == "error":
                return True
            elif result.status == "success":
                return False
        return None


class ContentDictStrategy(ErrorStrategy):
    """Check content dict for common error indicators.

    Many tools return structured JSON/dict responses with explicit
    success/failure fields. This strategy checks for common patterns:

    - {"status": "error"} or {"status": "failed"}
    - {"error": "..."} or {"error": {...}}
    - {"success": false}
    - {"ok": false}
    - {"failed": true}

    Example:
        ToolMessage(content='{"status": "error", "message": "Not found"}')
    """

    # Common error indicator fields and their error values
    ERROR_PATTERNS = [
        ("status", ["error", "failed", "failure"]),
        ("success", [False, "false", "False"]),
        ("ok", [False, "false", "False"]),
        ("failed", [True, "true", "True"]),
    ]

    def is_error(self, result: ToolMessage) -> bool | None:
        content = result.content

        # Parse JSON string to dict if needed
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                return None  # Not a JSON dict, can't determine

        if not isinstance(content, dict):
            return None  # Not a dict, can't determine

        # Check for explicit error key
        if "error" in content and content["error"]:
            return True

        # Check common error patterns
        for field, error_values in self.ERROR_PATTERNS:
            if field in content:
                if content[field] in error_values:
                    return True
                # If field exists with non-error value, likely success
                if field in ("status", "success", "ok"):
                    return False

        return None  # No definitive indicators found


class ExceptionContentStrategy(ErrorStrategy):
    """Detect error messages that look like Python exceptions.

    When tools catch exceptions and convert them to strings, the content
    often contains exception-like patterns. This strategy detects these
    without being overly aggressive with keyword matching.

    Patterns detected:
    - "Error: ..." at start of content
    - "Exception: ..." at start of content
    - "Traceback (most recent call last)" anywhere
    - "failed: ..." at start of content

    Example:
        ToolMessage(content="Error: Connection refused")
        ToolMessage(content="ValueError: Invalid input")
    """

    # Prefixes that strongly indicate an error
    ERROR_PREFIXES = [
        "error:",
        "exception:",
        "failed:",
        "failure:",
        "valueerror:",
        "typeerror:",
        "keyerror:",
        "attributeerror:",
        "runtimeerror:",
        "connectionerror:",
        "timeouterror:",
    ]

    def is_error(self, result: ToolMessage) -> bool | None:
        content = result.content

        if not isinstance(content, str):
            return None

        content_lower = content.lower().strip()

        # Check for exception-like prefixes
        for prefix in self.ERROR_PREFIXES:
            if content_lower.startswith(prefix):
                return True

        # Check for Python traceback
        if "traceback (most recent call last)" in content_lower:
            return True

        return None


class CompositeErrorStrategy(ErrorStrategy):
    """Chains multiple strategies together for comprehensive error detection.

    Strategies are evaluated in order until one returns a definitive
    True or False. If all strategies return None, the default behavior
    (configurable) is assumed.

    Example:
        strategy = CompositeErrorStrategy([
            ExplicitStatusStrategy(),
            ContentDictStrategy(),
            ExceptionContentStrategy(),
        ], default_is_error=False)

        is_error = strategy.is_error(tool_message)
    """

    def __init__(
        self,
        strategies: List[ErrorStrategy] | None = None,
        default_is_error: bool = False,
    ):
        """Initialize composite strategy.

        Args:
            strategies: Ordered list of strategies to apply. If None, uses
                default strategies: ExplicitStatus, ContentDict, ExceptionContent
            default_is_error: What to return if no strategy is definitive.
                False means assume success (recommended for production).
                True means assume error (conservative, may cause false positives).
        """
        self.strategies = strategies or [
            ExplicitStatusStrategy(),
            ContentDictStrategy(),
            ExceptionContentStrategy(),
        ]
        self.default_is_error = default_is_error

    def is_error(self, result: ToolMessage) -> bool | None:
        for strategy in self.strategies:
            decision = strategy.is_error(result)
            if decision is not None:
                return decision
        return self.default_is_error

    @property
    def name(self) -> str:
        strategy_names = [s.name for s in self.strategies]
        return f"CompositeErrorStrategy({', '.join(strategy_names)})"


# Default strategy chain - used by CompensationMiddleware if none specified
DEFAULT_ERROR_STRATEGIES = [
    ExplicitStatusStrategy(),
    ContentDictStrategy(),
    ExceptionContentStrategy(),
]


def create_error_detector(
    strategies: List[ErrorStrategy] | None = None,
    default_is_error: bool = False,
) -> CompositeErrorStrategy:
    """Factory function to create a configured error detector.

    Args:
        strategies: Custom strategy list, or None for defaults
        default_is_error: Fallback behavior when uncertain

    Returns:
        Configured CompositeErrorStrategy instance

    Example:
        # Use defaults
        detector = create_error_detector()

        # Custom strategies
        detector = create_error_detector([
            ExplicitStatusStrategy(),
            MyCustomStrategy(),
        ], default_is_error=True)
    """
    return CompositeErrorStrategy(
        strategies=strategies,
        default_is_error=default_is_error,
    )
