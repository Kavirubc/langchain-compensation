"""Example: Travel booking with automatic compensation."""

import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_compensation import create_comp_agent

# Setup API Key
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    print("Please set your GOOGLE_API_KEY environment variable.")
    exit(1)


# Define Tools
@tool
def book_flight(destination: str) -> str:
    """Books a flight to the given destination. Returns a booking ID."""
    print(f"\n[Tool] Booking flight to {destination}...")
    return f"flight_id_for_{destination}"


@tool
def cancel_flight(booking_id: str) -> str:
    """Cancels a flight with the given booking ID."""
    print(f"\n[Tool] Cancelling flight {booking_id}...")
    return "Cancellation successful"


@tool
def book_hotel(location: str) -> str:
    """Books a hotel in the given location. Returns a booking ID."""
    print(f"\n[Tool] Booking hotel in {location}...")
    # Simulate a failure for specific locations
    if "fail" in location.lower():
        return "Error: Hotel booking system is down! Cannot complete booking."
    return f"hotel_id_for_{location}"


@tool
def cancel_hotel(booking_id: str) -> str:
    """Cancels a hotel booking."""
    print(f"\n[Tool] Cancelling hotel {booking_id}...")
    return "Cancellation successful"


def main():
    # Create the Agent with compensation
    print("Initializing Agent with compensation middleware...")
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

    agent = create_comp_agent(
        model=model,
        tools=[book_flight, cancel_flight, book_hotel, cancel_hotel],
        compensation_mapping={"book_flight": "cancel_flight", "book_hotel": "cancel_hotel"},
    )

    # Scenario 1: Success Path
    print("\n" + "=" * 50)
    print("SCENARIO 1: Success Path")
    print("User: Book a flight to Paris and a hotel in Paris")
    print("=" * 50)

    inputs = {"messages": [("user", "Book a flight to Paris and a hotel in Paris")]}
    for step in agent.stream(inputs, stream_mode="values"):
        message = step["messages"][-1]
        if message.type == "ai" and not message.tool_calls:
            print(f"\nAgent: {message.content}")

    # Scenario 2: Failure with automatic compensation
    print("\n" + "=" * 50)
    print("SCENARIO 2: Failure & Compensation")
    print("User: Book a flight to London and a hotel in FailCity")
    print("=" * 50)

    inputs = {"messages": [("user", "Book a flight to London and a hotel in FailCity")]}
    for step in agent.stream(inputs, stream_mode="values"):
        message = step["messages"][-1]
        if message.type == "ai" and not message.tool_calls:
            print(f"\nAgent: {message.content}")


if __name__ == "__main__":
    main()
