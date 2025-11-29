"""Pluggable parameter extraction strategies for compensation middleware.

This module implements the Strategy pattern for extracting compensation
parameters from tool results. When a tool like book_flight() succeeds,
we need to extract the booking_id to pass to cancel_flight().

Extraction Priority (configurable):
1. state_mapper (explicit function from developer)
2. CompensationSchema (declarative field mapping)
3. Heuristic extraction (common ID field names)
4. Recursive search (deep nested structures)
5. LLM extraction (if enabled - Phase 3)
6. Raise error

This allows developers to use auto-detection for simple cases while
providing explicit control when needed.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from langchain_core.tools import BaseTool


@dataclass
class CompensationSchema:
    """Declarative schema for compensation parameter extraction.

    Allows developers to explicitly declare how to map result fields
    to compensation tool parameters without writing a full state_mapper.

    Example:
        schema = CompensationSchema(
            param_mapping={
                "booking_id": "result.id",           # result["id"] -> booking_id
                "confirmation": "result.conf_code",  # result["conf_code"] -> confirmation
            }
        )

    Path Syntax:
        - "result.field" - Access result["field"]
        - "result.nested.field" - Access result["nested"]["field"]
        - "params.field" - Access original params["field"]
        - "result[0]" - Access result[0] (list index)
        - "result.items[0].id" - Nested with list access

    Optional Fields:
        - If a field path ends with "?", it's optional (won't error if missing)
        - Example: "result.optional_field?"
    """

    param_mapping: Dict[str, str] = field(default_factory=dict)
    """Maps compensation param name -> path expression to extract value."""

    static_params: Dict[str, Any] = field(default_factory=dict)
    """Static parameters to always include (e.g., {"reason": "Auto rollback"})."""

    def extract(self, result: Any, original_params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract compensation parameters using the schema.

        Args:
            result: The result from the original tool call
            original_params: The original parameters passed to the tool

        Returns:
            Dict of extracted parameters for the compensation tool

        Raises:
            ValueError: If a required field path cannot be resolved
        """
        extracted = dict(self.static_params)  # Start with static params
        context = {"result": result, "params": original_params}

        for param_name, path_expr in self.param_mapping.items():
            optional = path_expr.endswith("?")
            if optional:
                path_expr = path_expr[:-1]

            try:
                value = self._resolve_path(path_expr, context)
                extracted[param_name] = value
            except (KeyError, IndexError, TypeError) as e:
                if not optional:
                    raise ValueError(
                        f"Cannot extract '{param_name}' from path '{path_expr}': {e}"
                    )

        return extracted

    def _resolve_path(self, path: str, context: Dict[str, Any]) -> Any:
        """Resolve a dot-notation path to a value.

        Supports:
            - Dot notation: "result.nested.field"
            - Array indices: "result.items[0]"
            - Mixed: "result.items[0].name"
        """
        # Split on dots, but handle array indices
        parts = re.split(r"\.(?![^\[]*\])", path)
        current = context

        for part in parts:
            # Check for array index: "items[0]"
            match = re.match(r"(\w+)\[(\d+)\]", part)
            if match:
                key, index = match.groups()
                current = current[key][int(index)]
            else:
                current = current[part]

        return current


class ExtractionStrategy(ABC):
    """Abstract base class for parameter extraction strategies.

    Each strategy attempts to extract compensation parameters from a
    tool result. Strategies can:
    - Return dict: Successfully extracted parameters
    - Return None: Cannot extract, defer to next strategy

    This allows strategies to be chained in priority order.
    """

    @abstractmethod
    def extract(
        self,
        result: Any,
        original_params: Dict[str, Any],
        compensation_tool: BaseTool | None = None,
        tool_name: str | None = None,
    ) -> Dict[str, Any] | None:
        """Attempt to extract compensation parameters.

        Args:
            result: The result from the original tool call
            original_params: The original parameters passed to the tool
            compensation_tool: The compensation tool (for schema inspection)
            tool_name: Name of the original tool (for lookup)

        Returns:
            Dict of extracted parameters, or None to defer
        """
        pass

    @property
    def name(self) -> str:
        """Human-readable name for logging and debugging."""
        return self.__class__.__name__


class StateMappersStrategy(ExtractionStrategy):
    """Use developer-provided state_mapper functions.

    This is the highest priority strategy - if a developer provides
    an explicit mapping function, use it.

    Example:
        state_mappers = {
            "book_flight": lambda result, params: {
                "booking_id": result["booking"]["id"],
                "reason": "Automatic rollback",
            }
        }
    """

    def __init__(
        self,
        state_mappers: Dict[str, Callable[[Any, Dict[str, Any]], Dict[str, Any]]] | None = None,
    ):
        self.state_mappers = state_mappers or {}

    def extract(
        self,
        result: Any,
        original_params: Dict[str, Any],
        compensation_tool: BaseTool | None = None,
        tool_name: str | None = None,
    ) -> Dict[str, Any] | None:
        if tool_name and tool_name in self.state_mappers:
            try:
                return self.state_mappers[tool_name](result, original_params)
            except Exception:
                return None  # Mapper failed, defer to next strategy
        return None


class SchemaExtractionStrategy(ExtractionStrategy):
    """Use CompensationSchema for declarative extraction.

    Second priority - allows developers to declare field mappings
    without writing code.

    Example:
        schemas = {
            "book_flight": CompensationSchema(
                param_mapping={"booking_id": "result.id"}
            )
        }
    """

    def __init__(self, schemas: Dict[str, CompensationSchema] | None = None):
        self.schemas = schemas or {}

    def extract(
        self,
        result: Any,
        original_params: Dict[str, Any],
        compensation_tool: BaseTool | None = None,
        tool_name: str | None = None,
    ) -> Dict[str, Any] | None:
        if tool_name and tool_name in self.schemas:
            try:
                return self.schemas[tool_name].extract(result, original_params)
            except ValueError:
                return None  # Schema extraction failed, defer
        return None


class HeuristicExtractionStrategy(ExtractionStrategy):
    """Extract common ID fields using heuristics.

    This is the default strategy for most tools - looks for common
    field names that typically contain identifiers needed for
    compensation.

    Priority of fields checked:
    1. id
    2. booking_id
    3. resource_id
    4. transaction_id
    5. order_id
    6. reservation_id

    Example:
        result = {"id": "12345", "status": "confirmed"}
        # Returns: {"id": "12345"}
    """

    # Common ID field names in order of priority
    ID_FIELDS = [
        "id",
        "booking_id",
        "resource_id",
        "transaction_id",
        "order_id",
        "reservation_id",
        "confirmation_id",
        "reference_id",
        "request_id",
    ]

    def extract(
        self,
        result: Any,
        original_params: Dict[str, Any],
        compensation_tool: BaseTool | None = None,
        tool_name: str | None = None,
    ) -> Dict[str, Any] | None:
        if not isinstance(result, dict):
            # For string results, assume it's the ID itself
            if isinstance(result, str) and result:
                return {"id": result}
            return None

        # Look for common ID fields
        for id_field in self.ID_FIELDS:
            if id_field in result:
                return {id_field: result[id_field]}

        return None


class RecursiveSearchStrategy(ExtractionStrategy):
    """Deep search through nested structures for ID fields.

    When heuristics fail on flat structures, this strategy searches
    recursively through nested dicts and lists to find ID fields.

    Example:
        result = {
            "data": {
                "booking": {
                    "id": "12345",
                    "details": {...}
                }
            }
        }
        # Returns: {"id": "12345"}
    """

    ID_FIELDS = [
        "id",
        "booking_id",
        "resource_id",
        "transaction_id",
        "order_id",
        "reservation_id",
    ]

    def __init__(self, max_depth: int = 5):
        """Initialize with maximum search depth.

        Args:
            max_depth: Maximum nesting depth to search (prevents infinite loops)
        """
        self.max_depth = max_depth

    def extract(
        self,
        result: Any,
        original_params: Dict[str, Any],
        compensation_tool: BaseTool | None = None,
        tool_name: str | None = None,
    ) -> Dict[str, Any] | None:
        if not isinstance(result, (dict, list)):
            return None

        found = self._search(result, depth=0)
        return found if found else None

    def _search(self, data: Any, depth: int) -> Dict[str, Any] | None:
        """Recursively search for ID fields."""
        if depth > self.max_depth:
            return None

        if isinstance(data, dict):
            # First check this level for ID fields
            for id_field in self.ID_FIELDS:
                if id_field in data and data[id_field]:
                    return {id_field: data[id_field]}

            # Then search nested structures
            for value in data.values():
                if isinstance(value, (dict, list)):
                    found = self._search(value, depth + 1)
                    if found:
                        return found

        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    found = self._search(item, depth + 1)
                    if found:
                        return found

        return None


class LLMExtractionStrategy(ExtractionStrategy):
    """Use LLM to extract compensation parameters when heuristics fail.

    This is the novel research contribution - using an LLM to intelligently
    extract compensation parameters from complex tool results when simpler
    strategies cannot determine the correct mapping.

    The LLM is given:
    - The tool result (what the original tool returned)
    - The compensation tool's expected schema (what parameters it needs)
    - Context about the task (tool names, original params)

    It returns a JSON object mapping parameter names to extracted values.

    Example:
        # Complex nested result that heuristics can't handle
        result = {
            "response": {
                "data": {
                    "flight": {
                        "confirmation": "ABC123",
                        "segments": [...]
                    }
                }
            }
        }

        # Compensation tool expects: {"booking_reference": str}
        # LLM extracts: {"booking_reference": "ABC123"}

    Research Value:
    - First use of LLM for runtime saga compensation parameter inference
    - Enables robust compensation for arbitrary tool schemas
    - Benchmark: Extraction accuracy vs. heuristic-only approaches
    """

    # Prompt template for parameter extraction
    EXTRACTION_PROMPT = '''You are a parameter extraction assistant. Your task is to extract the correct parameters needed to call a compensation (rollback) tool.

Original Tool: {tool_name}
Original Parameters: {original_params}
Tool Result: {result}

Compensation Tool: {comp_tool_name}
Expected Parameters Schema: {comp_schema}

Extract the parameters needed to call the compensation tool from the result above.
Return ONLY a valid JSON object with the extracted parameters.
Do not include any explanation or markdown formatting.

Example response format:
{{"booking_id": "ABC123", "reason": "automatic rollback"}}

Your response:'''

    def __init__(
        self,
        model: str | Any = "gpt-4o-mini",
        cache_extractions: bool = True,
        max_retries: int = 2,
        temperature: float = 0.0,
    ):
        """Initialize LLM extraction strategy.

        Args:
            model: Model identifier string (e.g., "gpt-4o-mini", "claude-3-haiku")
                or a pre-configured LLM instance. Uses cheap/fast models by default.
            cache_extractions: Cache extraction results to avoid repeated LLM calls
                for identical (tool_name, result) pairs.
            max_retries: Number of retries on parse failure.
            temperature: LLM temperature (0.0 for deterministic extractions).
        """
        self._model_config = model
        self._llm = None  # Lazy initialization
        self.cache_extractions = cache_extractions
        self.max_retries = max_retries
        self.temperature = temperature
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _get_llm(self):
        """Lazy-load the LLM instance."""
        if self._llm is not None:
            return self._llm

        # If already an LLM instance, use directly
        if not isinstance(self._model_config, str):
            self._llm = self._model_config
            return self._llm

        # Otherwise, initialize from model string
        model_name = self._model_config

        try:
            # Try OpenAI first
            if "gpt" in model_name.lower():
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(
                    model=model_name,
                    temperature=self.temperature,
                )
            # Try Anthropic
            elif "claude" in model_name.lower():
                from langchain_anthropic import ChatAnthropic
                self._llm = ChatAnthropic(
                    model=model_name,
                    temperature=self.temperature,
                )
            # Fallback to generic chat model
            else:
                from langchain_core.language_models import BaseChatModel
                # Try to import from common providers
                try:
                    from langchain_openai import ChatOpenAI
                    self._llm = ChatOpenAI(model=model_name, temperature=self.temperature)
                except ImportError:
                    raise ImportError(
                        f"Could not load LLM for model '{model_name}'. "
                        "Install langchain-openai or langchain-anthropic."
                    )
        except ImportError as e:
            raise ImportError(
                f"LLM extraction requires additional dependencies. "
                f"Install langchain-openai for GPT models or langchain-anthropic for Claude. "
                f"Error: {e}"
            )

        return self._llm

    def _get_tool_schema(self, tool: BaseTool | None) -> str:
        """Extract schema description from a tool."""
        if tool is None:
            return "Unknown - extract common ID fields like 'id', 'booking_id', etc."

        schema_parts = []

        # Try to get args_schema
        if hasattr(tool, "args_schema") and tool.args_schema:
            try:
                schema = tool.args_schema.schema()
                if "properties" in schema:
                    for name, prop in schema["properties"].items():
                        prop_type = prop.get("type", "any")
                        description = prop.get("description", "")
                        required = name in schema.get("required", [])
                        schema_parts.append(
                            f"- {name} ({prop_type}{'*' if required else ''}): {description}"
                        )
            except Exception:
                pass

        # Fallback to tool description
        if not schema_parts and hasattr(tool, "description"):
            return f"Tool expects parameters for: {tool.description}"

        return "\n".join(schema_parts) if schema_parts else "Unknown schema"

    def _cache_key(self, tool_name: str, result: Any) -> str:
        """Generate cache key for extraction results."""
        import json
        import hashlib
        result_str = json.dumps(result, sort_keys=True, default=str)
        return hashlib.md5(f"{tool_name}:{result_str}".encode()).hexdigest()

    def _parse_llm_response(self, response: str) -> Dict[str, Any] | None:
        """Parse LLM response to extract JSON parameters."""
        import json

        # Clean up response
        response = response.strip()

        # Remove markdown code blocks if present
        if response.startswith("```"):
            lines = response.split("\n")
            # Remove first and last lines (``` markers)
            lines = [l for l in lines if not l.strip().startswith("```")]
            response = "\n".join(lines).strip()

        # Try direct JSON parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in response
        import re
        json_match = re.search(r'\{[^{}]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return None

    def extract(
        self,
        result: Any,
        original_params: Dict[str, Any],
        compensation_tool: BaseTool | None = None,
        tool_name: str | None = None,
    ) -> Dict[str, Any] | None:
        """Extract parameters using LLM.

        Args:
            result: Tool result to extract from
            original_params: Original parameters passed to the tool
            compensation_tool: The compensation tool (for schema)
            tool_name: Name of the original tool

        Returns:
            Extracted parameters dict, or None if extraction fails
        """
        import json

        # Check cache first
        if self.cache_extractions and tool_name:
            cache_key = self._cache_key(tool_name, result)
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Get LLM and tool schema
        try:
            llm = self._get_llm()
        except ImportError as e:
            # LLM not available, defer to next strategy
            return None

        comp_tool_name = compensation_tool.name if compensation_tool else "unknown"
        comp_schema = self._get_tool_schema(compensation_tool)

        # Build prompt
        prompt = self.EXTRACTION_PROMPT.format(
            tool_name=tool_name or "unknown",
            original_params=json.dumps(original_params, default=str),
            result=json.dumps(result, default=str) if not isinstance(result, str) else result,
            comp_tool_name=comp_tool_name,
            comp_schema=comp_schema,
        )

        # Call LLM with retries
        for attempt in range(self.max_retries + 1):
            try:
                response = llm.invoke(prompt)

                # Extract content from response
                if hasattr(response, "content"):
                    response_text = response.content
                else:
                    response_text = str(response)

                # Parse the response
                extracted = self._parse_llm_response(response_text)

                if extracted and isinstance(extracted, dict):
                    # Cache successful extraction
                    if self.cache_extractions and tool_name:
                        self._cache[cache_key] = extracted
                    return extracted

            except Exception as e:
                if attempt == self.max_retries:
                    # Final attempt failed, defer to next strategy
                    return None
                continue

        return None

    @property
    def name(self) -> str:
        model_name = self._model_config if isinstance(self._model_config, str) else "custom"
        return f"LLMExtractionStrategy({model_name})"


class PassthroughStrategy(ExtractionStrategy):
    """Pass through the entire result as compensation params.

    Last resort strategy - if no other extraction works, pass the
    entire result to the compensation tool. This works when the
    compensation tool accepts the same structure as the result.

    Example:
        result = {"booking_id": "123", "amount": 500}
        # Returns: {"booking_id": "123", "amount": 500}
    """

    def extract(
        self,
        result: Any,
        original_params: Dict[str, Any],
        compensation_tool: BaseTool | None = None,
        tool_name: str | None = None,
    ) -> Dict[str, Any] | None:
        if isinstance(result, dict):
            return dict(result)
        return None


class CompositeExtractionStrategy(ExtractionStrategy):
    """Chains multiple extraction strategies together.

    Strategies are evaluated in order until one returns a non-None
    result. This implements the priority chain described in the module
    docstring.

    Example:
        strategy = CompositeExtractionStrategy([
            StateMappersStrategy(state_mappers={...}),
            SchemaExtractionStrategy(schemas={...}),
            HeuristicExtractionStrategy(),
            RecursiveSearchStrategy(),
        ])

        params = strategy.extract(result, original_params, comp_tool, tool_name)
    """

    def __init__(
        self,
        strategies: List[ExtractionStrategy] | None = None,
        raise_on_failure: bool = True,
    ):
        """Initialize composite strategy.

        Args:
            strategies: Ordered list of strategies. If None, uses defaults:
                [Heuristic, RecursiveSearch, Passthrough]
            raise_on_failure: If True, raises ValueError when no strategy
                succeeds. If False, returns None.
        """
        self.strategies = strategies or [
            HeuristicExtractionStrategy(),
            RecursiveSearchStrategy(),
            PassthroughStrategy(),
        ]
        self.raise_on_failure = raise_on_failure

    def extract(
        self,
        result: Any,
        original_params: Dict[str, Any],
        compensation_tool: BaseTool | None = None,
        tool_name: str | None = None,
    ) -> Dict[str, Any] | None:
        for strategy in self.strategies:
            extracted = strategy.extract(
                result, original_params, compensation_tool, tool_name
            )
            if extracted is not None:
                return extracted

        if self.raise_on_failure:
            raise ValueError(
                f"No extraction strategy could extract parameters for tool '{tool_name}' "
                f"from result: {result}"
            )
        return None

    @property
    def name(self) -> str:
        strategy_names = [s.name for s in self.strategies]
        return f"CompositeExtractionStrategy({', '.join(strategy_names)})"


# Factory function for creating configured extractors
def create_extraction_strategy(
    state_mappers: Dict[str, Callable[[Any, Dict[str, Any]], Dict[str, Any]]] | None = None,
    compensation_schemas: Dict[str, CompensationSchema] | None = None,
    include_llm: bool = False,
    llm_model: str | None = None,
    raise_on_failure: bool = True,
) -> CompositeExtractionStrategy:
    """Factory function to create a configured extraction strategy chain.

    Args:
        state_mappers: Custom mapping functions by tool name
        compensation_schemas: CompensationSchema instances by tool name
        include_llm: If True, adds LLM extraction as final fallback (Phase 3)
        llm_model: Model to use for LLM extraction (e.g., "gpt-4o-mini")
        raise_on_failure: Whether to raise error if extraction fails

    Returns:
        Configured CompositeExtractionStrategy instance

    Example:
        # Basic usage with defaults
        extractor = create_extraction_strategy()

        # With custom mappers and schemas
        extractor = create_extraction_strategy(
            state_mappers={"book_flight": lambda r, p: {"id": r["booking_id"]}},
            compensation_schemas={"book_hotel": CompensationSchema(...)},
        )

        # With LLM fallback (Phase 3)
        extractor = create_extraction_strategy(include_llm=True, llm_model="gpt-4o-mini")
    """
    strategies: List[ExtractionStrategy] = []

    # Priority 1: Developer-provided state mappers
    if state_mappers:
        strategies.append(StateMappersStrategy(state_mappers))

    # Priority 2: Declarative schemas
    if compensation_schemas:
        strategies.append(SchemaExtractionStrategy(compensation_schemas))

    # Priority 3: Heuristic extraction (always included)
    strategies.append(HeuristicExtractionStrategy())

    # Priority 4: Recursive search
    strategies.append(RecursiveSearchStrategy())

    # Priority 5: LLM extraction
    if include_llm:
        strategies.append(LLMExtractionStrategy(model=llm_model or "gpt-4o-mini"))

    # Priority 6: Passthrough (last resort)
    strategies.append(PassthroughStrategy())

    return CompositeExtractionStrategy(
        strategies=strategies,
        raise_on_failure=raise_on_failure,
    )
