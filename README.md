# LangChain Compensation

**v0.4.0** — Composable compensation middleware for LangChain agents with automatic rollback, pluggable strategies, multi-agent support, and LLM-driven parameter extraction. Implements the Saga pattern for distributed transactions in agent workflows.

[![PyPI version](https://badge.fury.io/py/langchain-compensation.svg)](https://badge.fury.io/py/langchain-compensation)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Automatic Rollback**: Compensates completed actions when failures occur using the Saga pattern
- **DAG-Based Ordering**: Rolls back actions in correct dependency order (not just LIFO)
- **Pluggable Error Detection**: Extensible strategies for detecting tool failures
- **Pluggable Parameter Extraction**: Multiple strategies including LLM-based extraction
- **Multi-Agent Support**: Shared compensation log for coordinated rollback across agents
- **Parallel Execution Safe**: Thread-safe with atomic batch operations
- **Fault Tolerance**: Checkpoint integration with LangGraph for crash recovery
- **Simple API**: Drop-in replacement for `create_agent()` with one additional parameter

## Installation

```bash
pip install langchain-compensation
```

For LLM-based parameter extraction (optional):
```bash
pip install langchain-compensation langchain-openai
# or
pip install langchain-compensation langchain-anthropic
```

## Quick Start

```python
from langchain_compensation import create_comp_agent
from langchain_core.tools import tool

@tool
def book_flight(destination: str) -> dict:
    """Books a flight to the destination."""
    return {"id": f"FL-{destination.upper()}-123", "status": "confirmed"}

@tool
def cancel_flight(id: str) -> str:
    """Cancels a flight booking."""
    return f"Flight {id} cancelled"

@tool
def book_hotel(location: str) -> dict:
    """Books a hotel at the location."""
    if "fail" in location.lower():
        return {"status": "error", "message": "No rooms available"}
    return {"id": f"HT-{location.upper()}-456", "status": "confirmed"}

@tool
def cancel_hotel(id: str) -> str:
    """Cancels a hotel booking."""
    return f"Hotel {id} cancelled"

# Create agent with automatic compensation
agent = create_comp_agent(
    model="gpt-4o",
    tools=[book_flight, cancel_flight, book_hotel, cancel_hotel],
    compensation_mapping={
        "book_flight": "cancel_flight",
        "book_hotel": "cancel_hotel",
    },
)

# When hotel booking fails, flight is automatically cancelled!
result = agent.invoke({
    "messages": [("user", "Book a flight to Paris and a hotel in FailCity")]
})
```

## How It Works

```
1. Agent calls book_flight("Paris")     → Success, records compensation action
2. Agent calls book_hotel("FailCity")   → Fails with error status
3. Middleware detects failure           → Triggers automatic rollback
4. Middleware calls cancel_flight()     → Compensates the successful booking
5. Agent continues with clean state
```

The compensation log tracks all compensatable actions with their results, enabling intelligent parameter extraction for compensation calls.

---

## Core Concepts

### Compensation Mapping

Maps each tool to its compensation (rollback) tool:

```python
compensation_mapping = {
    "book_flight": "cancel_flight",      # book_flight → cancel_flight
    "reserve_inventory": "release_inventory",
    "charge_payment": "refund_payment",
}
```

### Parameter Extraction Priority

When calling compensation tools, parameters are extracted in this order:

1. **state_mappers** - Custom functions (highest priority)
2. **CompensationSchema** - Declarative field mapping
3. **Heuristic** - Common ID fields (id, booking_id, etc.)
4. **Recursive Search** - Deep nested structure search
5. **LLM Extraction** - AI-powered extraction (if enabled)
6. **Passthrough** - Return entire result (last resort)

---

## Parameter Extraction Options

### Option 1: Automatic (Default)

Works out of the box for tools returning common ID patterns:

```python
# Tool returns: {"id": "123", "status": "confirmed"}
# Compensation receives: {"id": "123"}

agent = create_comp_agent(
    model="gpt-4o",
    tools=[book_flight, cancel_flight],
    compensation_mapping={"book_flight": "cancel_flight"},
    # No additional configuration needed!
)
```

### Option 2: Custom State Mappers

For complex extraction logic:

```python
def extract_flight_params(result, original_params):
    """Custom extraction from nested result."""
    return {
        "booking_id": result["booking"]["confirmation_code"],
        "reason": f"Rollback for {original_params.get('destination')}",
    }

agent = create_comp_agent(
    model="gpt-4o",
    tools=[book_flight, cancel_flight],
    compensation_mapping={"book_flight": "cancel_flight"},
    state_mappers={"book_flight": extract_flight_params},
)
```

### Option 3: Declarative Schema

Use `CompensationSchema` for path-based extraction without code:

```python
from langchain_compensation import CompensationSchema, create_comp_agent

agent = create_comp_agent(
    model="gpt-4o",
    tools=[book_flight, cancel_flight],
    compensation_mapping={"book_flight": "cancel_flight"},
    compensation_schemas={
        "book_flight": CompensationSchema(
            param_mapping={
                "booking_id": "result.id",              # result["id"]
                "confirmation": "result.conf_code",     # result["conf_code"]
                "origin": "params.origin",              # original params
            },
            static_params={
                "reason": "Automatic rollback",
            },
        ),
    },
)
```

**Path Syntax:**
- `result.field` → `result["field"]`
- `result.nested.field` → `result["nested"]["field"]`
- `params.field` → `original_params["field"]`
- `result.items[0].id` → `result["items"][0]["id"]`
- `result.optional?` → Optional field (no error if missing)

### Option 4: LLM-Based Extraction

For complex results where heuristics fail:

```python
agent = create_comp_agent(
    model="gpt-4o",
    tools=[book_flight, cancel_flight],
    compensation_mapping={"book_flight": "cancel_flight"},
    use_llm_extraction=True,           # Enable LLM extraction
    extraction_model="gpt-4o-mini",    # Use cheap model for extraction
)
```

The LLM receives the tool result and compensation tool schema, then intelligently extracts the required parameters.

---

## Multi-Agent Support

Multiple agents can share a compensation log for coordinated rollback:

```python
from langchain_compensation import create_comp_agent, CompensationLog

# Create shared log
shared_log = CompensationLog()

# Flight booking agent
flight_agent = create_comp_agent(
    model="gpt-4o",
    tools=[book_flight, cancel_flight],
    compensation_mapping={"book_flight": "cancel_flight"},
    shared_log=shared_log,
    agent_id="flight-agent",
)

# Hotel booking agent
hotel_agent = create_comp_agent(
    model="gpt-4o",
    tools=[book_hotel, cancel_hotel],
    compensation_mapping={"book_hotel": "cancel_hotel"},
    shared_log=shared_log,
    agent_id="hotel-agent",
)

# If hotel_agent fails, flight_agent's bookings are also rolled back!
```

### Multi-Agent Flow

```
Flight Agent: book_flight()   → Success → Recorded in shared_log
Hotel Agent:  book_hotel()    → Success → Recorded in shared_log
Hotel Agent:  book_rental()   → FAILS!
                ↓
Shared log triggers coordinated rollback:
  1. cancel_rental() (if it was recorded)
  2. cancel_hotel()  (Hotel Agent's action)
  3. cancel_flight() (Flight Agent's action)
                ↓
Both agents' actions compensated atomically
```

---

## Fault Tolerance with Checkpointing

Integrate with LangGraph checkpointers for crash recovery:

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver

# Development: In-memory
agent = create_comp_agent(
    model="gpt-4o",
    tools=[...],
    compensation_mapping={...},
    checkpointer=MemorySaver(),
    use_checkpointing=True,
)

# Production: PostgreSQL
agent = create_comp_agent(
    model="gpt-4o",
    tools=[...],
    compensation_mapping={...},
    checkpointer=PostgresSaver(conn_string="postgresql://..."),
    use_checkpointing=True,
)
```

### Recovery After Crash

```python
from langchain_compensation import CheckpointMiddleware, CompensationMiddleware

# After process restart, restore state
checkpoint_middleware = CheckpointMiddleware(checkpointer=saved_checkpointer)
restored_log = checkpoint_middleware.restore_from_failure({"thread_id": "abc123"})

if restored_log:
    # Resume with restored compensation state
    middleware = CompensationMiddleware(
        compensation_mapping={...},
        shared_log=restored_log,
    )
```

---

## Pluggable Strategies

### Custom Error Detection

```python
from langchain_compensation import (
    ErrorStrategy,
    ExplicitStatusStrategy,
    ContentDictStrategy,
    create_comp_agent,
)

class CustomAPIErrorStrategy(ErrorStrategy):
    """Detect errors from custom API responses."""

    def is_error(self, result) -> bool | None:
        content = result.content
        if isinstance(content, dict):
            if content.get("api_status") == "failed":
                return True
            if content.get("error_code") is not None:
                return True
        return None  # Defer to next strategy

agent = create_comp_agent(
    model="gpt-4o",
    tools=[...],
    compensation_mapping={...},
    error_strategies=[
        ExplicitStatusStrategy(),    # Check ToolMessage.status
        ContentDictStrategy(),       # Check {"error": ...}
        CustomAPIErrorStrategy(),    # Your custom logic
    ],
)
```

### Custom Parameter Extraction

```python
from langchain_compensation import (
    ExtractionStrategy,
    HeuristicExtractionStrategy,
    create_comp_agent,
)

class GraphQLExtractionStrategy(ExtractionStrategy):
    """Extract IDs from GraphQL-style responses."""

    def extract(self, result, original_params, comp_tool=None, tool_name=None):
        if isinstance(result, dict) and "data" in result:
            data = result["data"]
            # Find the first mutation result with an ID
            for key, value in data.items():
                if isinstance(value, dict) and "id" in value:
                    return {"id": value["id"]}
        return None  # Defer to next strategy

agent = create_comp_agent(
    model="gpt-4o",
    tools=[...],
    compensation_mapping={...},
    extraction_strategies=[
        GraphQLExtractionStrategy(),
        HeuristicExtractionStrategy(),
    ],
)
```

---

## API Reference

### `create_comp_agent`

```python
def create_comp_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable],
    *,
    compensation_mapping: dict[str, str],

    # Parameter Extraction
    state_mappers: dict[str, Callable] = None,
    compensation_schemas: dict[str, CompensationSchema] = None,
    use_llm_extraction: bool = False,
    extraction_model: str = "gpt-4o-mini",

    # Multi-Agent
    shared_log: CompensationLog = None,
    agent_id: str = None,

    # Pluggable Strategies
    error_strategies: list[ErrorStrategy] = None,
    extraction_strategies: list[ExtractionStrategy] = None,

    # Fault Tolerance
    checkpointer: Checkpointer = None,
    use_checkpointing: bool = False,

    # Standard Agent Parameters
    system_prompt: str = None,
    middleware: Sequence[AgentMiddleware] = (),
    ...
) -> CompiledStateGraph
```

### `CompensationSchema`

```python
@dataclass
class CompensationSchema:
    param_mapping: Dict[str, str]   # Maps param name → path expression
    static_params: Dict[str, Any]   # Static values to always include

    def extract(self, result, original_params) -> Dict[str, Any]
```

### `CompensationLog`

```python
class CompensationLog:
    def add(self, record: CompensationRecord) -> None
    def update(self, record_id: str, **kwargs) -> None
    def get(self, record_id: str) -> CompensationRecord | None
    def snapshot(self) -> Dict[str, CompensationRecord]  # Thread-safe copy
    def atomic_batch(self, operations: List[tuple]) -> None  # Atomic ops
    def filter_by_agent(self, agent_id: str) -> List[CompensationRecord]
    def get_rollback_plan(self, agent_id: str = None) -> List[CompensationRecord]
    def mark_compensated(self, record_id: str) -> None
    def clear(self, agent_id: str = None) -> None
```

### `CompensationMiddleware`

```python
class CompensationMiddleware(AgentMiddleware):
    def __init__(
        self,
        compensation_mapping: Dict[str, str],
        tools: List[BaseTool] = None,
        state_mappers: Dict[str, Callable] = None,
        shared_log: CompensationLog = None,
        agent_id: str = None,
        compensation_schemas: Dict[str, CompensationSchema] = None,
        error_strategies: List[ErrorStrategy] = None,
        extraction_strategies: List[ExtractionStrategy] = None,
    )
```

### Error Strategies

| Strategy | Description |
|----------|-------------|
| `ExplicitStatusStrategy` | Checks `ToolMessage.status == "error"` |
| `ContentDictStrategy` | Checks for `{"error": ...}`, `{"success": false}`, `{"ok": false}` |
| `ExceptionContentStrategy` | Detects exception-like patterns in content |
| `CompositeErrorStrategy` | Chains multiple strategies together |

### Extraction Strategies

| Strategy | Description |
|----------|-------------|
| `StateMappersStrategy` | Uses developer-provided functions |
| `SchemaExtractionStrategy` | Uses `CompensationSchema` declarations |
| `HeuristicExtractionStrategy` | Looks for common ID fields |
| `RecursiveSearchStrategy` | Deep searches nested structures |
| `LLMExtractionStrategy` | Uses LLM for intelligent extraction |
| `PassthroughStrategy` | Returns entire result as-is |
| `CompositeExtractionStrategy` | Chains multiple strategies |

---

## Use Cases

- **Travel Booking**: Cancel flights/hotels/cars if any booking fails
- **E-commerce**: Reverse inventory, refund payments on order failure
- **Financial**: Rollback transfers, cancel holds on transaction failure
- **Infrastructure**: Delete created resources if provisioning fails
- **Multi-Service**: Coordinate rollback across microservices
- **AI Agents**: Clean up side effects when agent workflows fail

---

## Migration from v0.3.x

v0.4.0 is fully backwards compatible. New features are opt-in:

```python
# v0.3.x code still works
agent = create_comp_agent(
    model="gpt-4o",
    tools=[...],
    compensation_mapping={...},
    state_mappers={...},  # Still supported
)

# v0.4.0 additions (all optional)
agent = create_comp_agent(
    model="gpt-4o",
    tools=[...],
    compensation_mapping={...},
    # New in v0.4.0:
    compensation_schemas={...},      # Declarative extraction
    shared_log=CompensationLog(),    # Multi-agent support
    agent_id="my-agent",             # Agent identification
    use_llm_extraction=True,         # LLM-based extraction
    use_checkpointing=True,          # Fault tolerance
)
```

---

## Requirements

- Python 3.9+
- langchain >= 0.3.0
- langchain-core >= 0.3.0
- langgraph >= 0.2.0

Optional:
- langchain-openai (for LLM extraction with GPT models)
- langchain-anthropic (for LLM extraction with Claude models)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.

## Credits

Inspired by the Saga pattern for distributed transactions and built on LangChain's middleware system.
