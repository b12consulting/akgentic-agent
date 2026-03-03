"""Example 04: Adapting V1 Simple Team to V2 APIs

This example adapts the v1 simple_team.py pattern to v2 manual setup,
demonstrating migration from akgentic-framework v1 to v2.

V1 → V2 Migration Changes:
- MemberConfig → AgentCard (from akgentic-core)
- LLMConfig → ModelConfig (from akgentic-llm)
- team_config dict → Manual AgentCard setup
- Controller+PrintModule → EventSubscriber
- WebSearchTool → (removed, not in Phase 2 scope)

Team Structure:
- Manager: Coordinates team, can hire Assistant and Expert
- Assistant: Provides support and research
- Expert: Provides specialized knowledge

Interactive Features:
- @ mention routing (e.g., @Manager hello)
- Direct human-to-manager messaging
- Dynamic team member hiring (via Manager's hire_members tool)

Run: uv run packages/akgentic-agent/examples/04_simple_team_v1_adapted.py
"""

import logging
import time

from akgentic.core import (
    ActorSystem,
    AgentCard,
    BaseConfig,
    EventSubscriber,
    Orchestrator,
)
from akgentic.core.messages import Message
from akgentic.core.messages.orchestrator import SentMessage
from akgentic.llm import ModelConfig, PromptTemplate

from akgentic.agent import AgentConfig, BaseAgent, HumanProxy
from akgentic.agent.messages import AgentMessage

logging.basicConfig(level=logging.WARNING, format="%(name)s - %(levelname)s - %(message)s")


class MessagePrinter(EventSubscriber):
    """Prints messages as they flow through the orchestrator.

    Subscribes to orchestrator events and prints relevant message traffic
    to provide visibility into agent interactions.

    V1 equivalent: Controller with PrintModule
    """

    def on_stop(self) -> None:
        pass

    def on_message(self, message: Message) -> None:

        if isinstance(message, SentMessage):
            self.handle_sent_message(message)

    def handle_sent_message(self, message: SentMessage) -> None:
        """Handle sentMessage events from orchestrator.

        Args:
            sender: Address of the message sender
            recipient: Address of the message recipient
            message: The message being sent
        """
        msg = message.message
        assert msg.sender is not None
        sender = msg.sender.name
        recipient = message.recipient.name

        # Filter to relevant message types
        if hasattr(msg, "content"):
            content = getattr(msg, "content")
            print("-" * 100)
            print(f"[{sender}] -> AgentMessage ({recipient}): \n{content[:5000]}...\n")


def main() -> None:
    """Run the simple team example with v2 APIs."""
    # 1. Create ActorSystem
    actor_system = ActorSystem()

    # 2. Create Orchestrator
    orchestrator_addr = actor_system.createActor(
        Orchestrator, config=BaseConfig(name="@Orchestrator", role="Orchestrator")
    )
    orchestrator_proxy = actor_system.proxy_ask(orchestrator_addr, Orchestrator)

    # 8. Subscribe to orchestrator events
    printer = MessagePrinter()
    orchestrator_proxy.subscribe(printer)

    # 3. Define AgentCards for each role
    # AgentCards define available roles in the team that can be hired dynamically
    manager_card = AgentCard(
        role="Manager",
        description="Helpful manager coordinating team work",
        skills=["coordination", "delegation"],
        agent_class="akgentic.agent.BaseAgent",
        config=AgentConfig(
            name="@Manager",
            role="Manager",
            prompt=PromptTemplate(
                template="You are a helpful manager. Coordinate the team effectively."
            ),
            model_cfg=ModelConfig(provider="openai", model="gpt-4.1", temperature=0.3),
        ),
        routes_to=["Assistant", "Expert"],  # Can hire these roles
    )

    assistant_card = AgentCard(
        role="Assistant",
        description="Helpful assistant providing support",
        skills=["research", "writing"],
        agent_class="akgentic.agent.BaseAgent",
        config=AgentConfig(
            name="@Assistant",
            role="Assistant",
            prompt=PromptTemplate(
                template="You are a helpful assistant. Provide clear and accurate information."
            ),
            model_cfg=ModelConfig(provider="openai", model="gpt-4.1", temperature=0.3),
        ),
    )

    expert_card = AgentCard(
        role="Expert",
        description="Helpful expert providing specialized knowledge",
        skills=["analysis", "problem-solving"],
        agent_class="akgentic.agent.BaseAgent",
        config=AgentConfig(
            name="@Expert",
            role="Expert",
            prompt=PromptTemplate(
                template="You are a helpful expert. Provide deep specialized knowledge."
            ),
            model_cfg=ModelConfig(provider="openai", model="gpt-4.1", temperature=0.3),
        ),
    )

    # 4. Register cards with orchestrator
    agent_cards = [manager_card, assistant_card, expert_card]
    orchestrator_proxy.register_agent_profiles(agent_cards)

    # 5. Create HumanProxy
    human_config = BaseConfig(name="@Human", role="Human")
    human_addr = orchestrator_proxy.createActor(HumanProxy, config=human_config)
    human_proxy = actor_system.proxy_tell(human_addr, HumanProxy)

    # 6. Create Manager agent
    # Build AgentConfig from the registered card's configuration
    manager_config = AgentConfig(
        name="@Manager",
        role="Manager",
        prompt=PromptTemplate(
            template="You are a helpful manager. Coordinate the team effectively."
        ),
        model_cfg=ModelConfig(provider="openai", model="gpt-4.1", temperature=0.3),
    )
    manager_addr = orchestrator_proxy.createActor(BaseAgent, config=manager_config)
    manager_proxy = actor_system.proxy_ask(manager_addr, BaseAgent)

    # 7. Create the team from the manager
    manager_proxy.createActor(BaseAgent, config=assistant_card.get_config_copy())
    manager_proxy.createActor(BaseAgent, config=expert_card.get_config_copy())

    # 9. Wait for actors to initialize
    time.sleep(0.3)

    # 10. Show team roster
    team = orchestrator_proxy.get_team()
    print("\nTeam members:")
    for member in team:
        print(f"  - {member.name} ({member.role})")

    # 11. Interactive chat loop
    print(
        "\nType your message (start the message with @{agent_name} to route to specific agent, "
        "'exit' to quit):"
    )

    print("-" * 100)
    while True:
        user_input = input("")
        print()
        if user_input.strip() == "":
            continue
        if user_input.strip().lower() == "exit":
            print("Exiting chat loop.")
            break

        if user_input.startswith("@"):
            # Route to specific agent
            target_name = user_input.split(" ")[0]
            try:
                actor_addr = orchestrator_proxy.get_team_member(target_name)
                if actor_addr is None:
                    print(f"Error: Agent {target_name} not found")
                    continue
                human_proxy.send(actor_addr, AgentMessage(content=user_input))
            except Exception as e:
                print(f"Error routing message: {e}")
        else:
            human_proxy.send(manager_addr, AgentMessage(content=user_input))

    # 12. Shutdown
    actor_system.shutdown()


if __name__ == "__main__":
    main()
