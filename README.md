# LangChain Compensation

Automatic compensation middleware for LangChain agents with LIFO (Last-In-First-Out) rollback capabilities. Inspired by the Saga pattern, this package provides automatic rollback of completed actions when a failure occurs in a multi-step agent workflow.

## Features

- 🔄 **Automatic Rollback**: Automatically compensates completed actions when failures occur
- 📚 **LIFO Order**: Rolls back actions in reverse chronological order (like a transaction stack)
- 🎯 **Simple API**: Easy-to-use agent factory with compensation mapping
- 🔧 **Flexible**: Support for custom state mappers to extract compensation parameters
- 🧵 **Thread-Safe**: Built with thread safety in mind
- 📦 **Zero Config**: Works out of the box with sensible defaults

## Installation

```bash
pip install langchain-compensation
```

## Quick Start

```python
from langchain_compensation import create_comp_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

# Define your tools
@tool
def book_flight(destination: str) -> str:
    """Books a flight to the destination."""
    return f"flight_id_for_{destination}"

@tool
def cancel_flight(booking_id: str) -> str:
    """Cancels a flight booking."""
    return "Cancellation successful"

@tool
def book_hotel(location: str) -> str:
    """Books a hotel at the location."""
    if "fail" in location.lower():
        return "Error: Hotel booking failed!"
    return f"hotel_id_for_{location}"

@tool
def cancel_hotel(booking_id: str) -> str:
    """Cancels a hotel booking."""
    return "Cancellation successful"

# Create agent with compensation
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
agent = create_comp_agent(
    model=model,
    tools=[book_flight, cancel_flight, book_hotel, cancel_hotel],
    compensation_mapping={
        "book_flight": "cancel_flight",
        "book_hotel": "cancel_hotel"
    }
)

# Run the agent
result = agent.invoke({
    "messages": [("user", "Book a flight to London and a hotel in FailCity")]
})
# When hotel booking fails, the flight booking is automatically cancelled!
```

## How It Works

1. **Track Actions**: Compensatable actions are tracked in the agent's state
2. **Detect Failures**: When a tool returns an error, the middleware detects it
3. **Automatic Rollback**: All completed compensatable actions are rolled back in LIFO order
4. **Continue**: The agent continues with the rolled-back state

## Advanced Usage

### Custom State Mappers

Sometimes you need custom logic to extract compensation parameters from the original result:

```python
def extract_flight_id(result, original_params):
    """Extract booking ID from the result."""
    return {"booking_id": result["id"]}

agent = create_comp_agent(
    model=model,
    tools=tools,
    compensation_mapping={"book_flight": "cancel_flight"},
    state_mappers={"book_flight": extract_flight_id}
)
```

### With Checkpointer

Use with LangGraph's checkpointing for persistent compensation logs:

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
agent = create_comp_agent(
    model=model,
    tools=tools,
    compensation_mapping={"book_flight": "cancel_flight"},
    checkpointer=memory
)
```

## API Reference

### `create_comp_agent`

Creates a LangChain agent with automatic compensation capabilities.

**Parameters:**
- `model` (BaseChatModel | str | None): The LLM to use for the agent
- `tools` (Sequence[BaseTool | Callable]): List of tools available to the agent
- `compensation_mapping` (dict[str, str]): Maps tool names to their compensation tools
- `state_mappers` (dict[str, Callable] | None): Custom parameter extraction functions
- `system_prompt` (str | None): Additional system prompt for the agent
- `checkpointer` (Checkpointer | None): Optional checkpointer for state persistence
- Other standard LangChain agent parameters

**Returns:** `CompiledStateGraph` - A configured agent with compensation middleware

### `CompensationMiddleware`

The core middleware that handles compensation logic.

**Parameters:**
- `compensation_mapping` (dict[str, str]): Maps tool names to compensation tools
- `tools` (list | None): Tools to cache for compensation execution
- `state_mappers` (dict[str, Callable] | None): Custom state mappers

## Use Cases

- **Travel Booking Systems**: Cancel flights/hotels if any part of the trip fails
- **E-commerce**: Reverse inventory reservations if payment fails
- **Multi-step Workflows**: Undo completed steps when later steps fail
- **Database Operations**: Rollback related operations in distributed systems
- **API Integrations**: Clean up created resources when workflows fail

## Requirements

- Python 3.9+
- langchain >= 0.3.0
- langchain-core >= 0.3.0
- langgraph >= 0.2.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Credits

Inspired by the Saga pattern and built on top of LangChain's middleware system.
