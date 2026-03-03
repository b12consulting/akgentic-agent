"""
Simple BaseAgent Example - Creating an Orchestrator and BaseAgent
==================================================================

This example demonstrates the basic setup:
- Creating an ActorSystem
- Creating an Orchestrator to manage team
- Creating a BaseAgent with LLM capabilities
- Sending a simple UserMessage and getting a ResultMessage response

Run with: python examples/00_example.py
Or with:  uv run python examples/00_example.py (from project root)
Or with:  cd packages/akgentic-agent && uv run python examples/00_example.py
"""

from __future__ import annotations

import time

from akgentic.core import ActorSystem, BaseConfig, Orchestrator
from akgentic.core.messages import UserMessage
from akgentic.llm.config import ModelConfig
from akgentic.llm.prompts import PromptTemplate

from akgentic.agent import AgentConfig, BaseAgent

# =============================================================================
# STEP 1: Configure the BaseAgent
# =============================================================================
# BaseAgent requires configuration for:
# - prompt: Agent's backstory/system prompt
# - model_cfg: LLM provider and model
# - runtime_cfg: Temperature, max tokens, etc.
# - usage_limits: Token/request constraints


def create_agent_config() -> AgentConfig:
    """Create a simple agent configuration."""
    return AgentConfig(
        name="@SimpleAssistant",
        role="Assistant",
        prompt=PromptTemplate(
            template="You are a helpful assistant. Answer questions concisely and accurately."
        ),
        model_cfg=ModelConfig(
            provider="openai",
            model="gpt-4o-mini",  # Using mini model for this example
        ),
        max_help_requests=3,
    )


# =============================================================================
# STEP 2: Main execution
# =============================================================================


def main() -> None:
    """Create orchestrator, agent, and test basic interaction."""
    print("=" * 70)
    print("Simple BaseAgent Example")
    print("=" * 70)

    # Create actor system - the foundation for all agents
    system = ActorSystem()

    try:
        # Create orchestrator - manages the team
        orchestrator_addr = system.createActor(
            Orchestrator, config=BaseConfig(name="@Orchestrator", role="Orchestrator")
        )
        print(f"\n✓ Orchestrator created at: {orchestrator_addr}")

        # Create agent config
        agent_config = create_agent_config()
        print(f"\n✓ Agent config created for: {agent_config.name}")

        # Create BaseAgent with orchestrator reference

        orchestrator_proxy = system.proxy_ask(orchestrator_addr, Orchestrator)
        agent_addr = orchestrator_proxy.createActor(BaseAgent, config=agent_config)
        print(f"✓ BaseAgent created at: {agent_addr}")

        # Send a simple user message
        print("\n" + "-" * 70)
        print("Sending UserMessage to agent...")
        print("-" * 70)

        user_message = UserMessage(content="What is 2 + 2?")
        system.ask(agent_addr, user_message, timeout=30)

        print("\n✓ Message sent! Waiting for response...")

        # Wait a moment for async processing
        time.sleep(2)

        print("messages :")
        print([message.__class__.__name__ for message in orchestrator_proxy.get_messages()])

        print("\n" + "=" * 70)
        print("Example completed successfully!")
        print("=" * 70)
        print("\nNote: In a real application, you would:")
        print("  - Handle ResultMessage responses asynchronously")
        print("  - Use proper message handlers or callbacks")
        print("  - Check orchestrator telemetry for message flow")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        print("\n\nShutting down actor system...")
        system.shutdown()
        print("✓ Actor system shutdown complete")


if __name__ == "__main__":
    main()
