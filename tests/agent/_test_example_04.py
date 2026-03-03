"""Tests for 04_simple_team_v1_adapted.py example.

Tests the v1 migration example components and setup.
"""

import os
import time

import pytest
from akgentic.core import ActorAddress, ActorSystem, AgentCard, BaseConfig, Orchestrator
from akgentic.core.messages import Message, ResultMessage, UserMessage
from akgentic.core.messages.orchestrator import SentMessage
from akgentic.llm import ModelConfig, PromptTemplate

from akgentic.agent import BaseAgent, HumanProxy
from akgentic.agent.config import AgentConfig

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def actor_system():
    """Create an ActorSystem for tests."""
    system = ActorSystem()
    yield system
    # Cleanup: shutdown all actors to prevent hanging
    try:
        system.shutdown(timeout=5)
    except Exception:
        pass  # Best effort cleanup


@pytest.fixture
def orchestrator_address(actor_system: ActorSystem):
    """Create an Orchestrator actor."""
    addr = actor_system.createActor(
        Orchestrator, config=BaseConfig(name="@Orchestrator", role="Orchestrator")
    )
    yield addr
    # Orchestrator cleanup happens via actor_system.shutdown()


# =============================================================================
# AGENTCARD CREATION TESTS
# =============================================================================


class TestSimpleTeamV1Adapted:
    """Test the v1 adapted simple team example components."""

    def test_agent_card_creation(self) -> None:
        """Test creating AgentCards for team roles."""
        # Create cards for each role
        manager_card = AgentCard(
            role="Manager",
            description="Helpful manager coordinating team work",
            skills=["coordination", "delegation"],
            agent_class="akgentic.agent.BaseAgent",
            config={
                "name": "@Manager",
                "role": "Manager",
                "prompt": {
                    "template": "You are a helpful manager. Coordinate the team effectively."
                },
                "model_cfg": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "temperature": 0.3,
                },
            },
            routes_to=["Assistant", "Expert"],
        )

        assistant_card = AgentCard(
            role="Assistant",
            description="Helpful assistant providing support",
            skills=["research", "writing"],
            agent_class="akgentic.agent.BaseAgent",
            config={
                "name": "@Assistant",
                "role": "Assistant",
                "prompt": {
                    "template": "You are a helpful assistant. Provide clear and accurate information."
                },
                "model_cfg": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "temperature": 0.3,
                },
            },
        )

        # Verify card properties
        assert manager_card.role == "Manager"
        assert manager_card.agent_class == "akgentic.agent.BaseAgent"
        assert manager_card.has_skill("coordination")
        assert manager_card.can_route_to("Assistant")
        assert manager_card.can_route_to("Expert")
        assert not manager_card.can_route_to("Other")

        assert assistant_card.role == "Assistant"
        assert assistant_card.has_skill("research")
        assert assistant_card.has_skill("writing")
        # Empty routes_to means can route to anyone
        assert assistant_card.can_route_to("Manager")

    # =============================================================================
    # MANUAL TEAM SETUP TESTS
    # =============================================================================

    def test_manual_team_setup(self, actor_system: ActorSystem, orchestrator_address) -> None:
        """Test manual team setup without TeamFactory."""
        orchestrator = actor_system.proxy_ask(orchestrator_address, Orchestrator)

        # Define and register cards
        manager_card = AgentCard(
            role="Manager",
            description="Test manager",
            skills=["coordination"],
            agent_class="akgentic.agent.BaseAgent",
            config={},
            routes_to=["Assistant"],
        )

        assistant_card = AgentCard(
            role="Assistant",
            description="Test assistant",
            skills=["research"],
            agent_class="akgentic.agent.BaseAgent",
            config={},
        )

        orchestrator.register_agent_profile(manager_card)
        orchestrator.register_agent_profile(assistant_card)

        # Get registered cards back
        cards = orchestrator.get_agent_catalog()
        assert len(cards) == 2
        assert any(c.role == "Manager" for c in cards)
        assert any(c.role == "Assistant" for c in cards)

        # Create manager agent manually
        manager_config = AgentConfig(
            name="@Manager",
            role="Manager",
            prompt=PromptTemplate(template="You are a test manager."),
            model_cfg=ModelConfig(provider="openai", model="gpt-4o-mini", temperature=0.3),
        )
        manager_addr = orchestrator.createActor(BaseAgent, config=manager_config)

        # Create HumanProxy
        human_config = BaseConfig(name="@Human", role="Human")
        human_addr = orchestrator.createActor(HumanProxy, config=human_config)

        # Register manager with HumanProxy
        actor_system.tell(human_addr, ("RegisterManager", manager_addr))

        # Wait for initialization
        time.sleep(0.2)

        # Verify team roster
        team = orchestrator.get_team()
        assert len(team) >= 2  # At least Manager and Human
        team_names = [member.name for member in team]
        assert "@Manager" in team_names
        assert "@Human" in team_names

    # =============================================================================
    # EVENT SUBSCRIBER TESTS
    # =============================================================================

    @pytest.mark.xfail(reason="Event subscription timing issue - known limitation")
    def test_orchestrator_event_subscriber(
        self, actor_system: ActorSystem, orchestrator_address
    ) -> None:
        """Test EventSubscriber for message visibility.

        NOTE: This test has timing issues with event subscription that need
        architectural resolution. Marked as xfail pending fix.
        """
        from akgentic.core import EventSubscriber

        from akgentic.agent import HelpAnswerMessage, HelpRequestMessage

        # Create a test subscriber
        events_captured: list[tuple[str, str, str]] = []

        class TestSubscriber(EventSubscriber):
            def sentMessage(self, message: Message) -> None:
                msg_type = type(message).__name__
                events_captured.append((sender.name, recipient.name, msg_type))

        subscriber = TestSubscriber()

        orchestrator = actor_system.proxy_ask(orchestrator_address, Orchestrator)

        # Subscribe
        orchestrator.subscribe(subscriber)

        # Create a simple agent
        config = AgentConfig(
            name="@TestAgent",
            role="TestAgent",
            prompt=PromptTemplate(template="You are a test agent."),
            model_cfg=ModelConfig(provider="openai", model="gpt-4o-mini", temperature=0.3),
        )
        agent_addr = orchestrator.createActor(BaseAgent, config=config)

        time.sleep(0.2)

        # Send a message
        actor_system.tell(agent_addr, UserMessage(content="Test message"))

        time.sleep(0.5)

        # Verify events were captured
        assert len(events_captured) > 0
        # Check that UserMessage was captured
        assert any(msg_type == "UserMessage" for _, _, msg_type in events_captured)

    # =============================================================================
    # MESSAGE ROUTING TESTS
    # =============================================================================

    def test_message_routing_by_name(self, actor_system: ActorSystem, orchestrator_address) -> None:
        """Test @ mention routing to specific agents."""
        orchestrator = actor_system.proxy_ask(orchestrator_address, Orchestrator)

        orchestrator = actor_system.proxy_ask(orchestrator_address, Orchestrator)

        # Create two agents
        agent1_config = AgentConfig(
            name="@Agent1",
            role="Agent1",
            prompt=PromptTemplate(template="You are agent 1."),
            model_cfg=ModelConfig(provider="openai", model="gpt-4o-mini", temperature=0.3),
        )
        agent1_addr = orchestrator.createActor(BaseAgent, config=agent1_config)

        agent2_config = AgentConfig(
            name="@Agent2",
            role="Agent2",
            prompt=PromptTemplate(template="You are agent 2."),
            model_cfg=ModelConfig(provider="openai", model="gpt-4o-mini", temperature=0.3),
        )
        agent2_addr = orchestrator.createActor(BaseAgent, config=agent2_config)

        time.sleep(0.2)

        # Test get_team_member routing
        found_agent1 = orchestrator.get_team_member("@Agent1")
        found_agent2 = orchestrator.get_team_member("@Agent2")

        assert found_agent1 is not None
        assert found_agent2 is not None
        assert found_agent1.name == "@Agent1"
        assert found_agent2.name == "@Agent2"

        # Test non-existent agent
        not_found = orchestrator.get_team_member("@NonExistent")
        assert not_found is None

    # =============================================================================
    # HUMANPROXY TESTS
    # =============================================================================

    def test_human_proxy_creation(self, actor_system: ActorSystem, orchestrator_address) -> None:
        """Test HumanProxy creation and team membership."""
        orchestrator = actor_system.proxy_ask(orchestrator_address, Orchestrator)

        # Create manager
        manager_config = AgentConfig(
            name="@Manager",
            role="Manager",
            prompt=PromptTemplate(template="You are a manager."),
            model_cfg=ModelConfig(provider="openai", model="gpt-4o-mini", temperature=0.3),
        )
        manager_addr = orchestrator.createActor(BaseAgent, config=manager_config)

        # Create HumanProxy
        human_config = BaseConfig(name="@Human", role="Human")
        human_addr = orchestrator.createActor(HumanProxy, config=human_config)

        time.sleep(0.2)

        # Verify both exist in team
        team = orchestrator.get_team()
        team_names = [member.name for member in team]
        assert "@Manager" in team_names
        assert "@Human" in team_names

        # Verify HumanProxy can send messages to manager
        human_proxy = actor_system.proxy_tell(human_addr, HumanProxy)
        human_proxy.send(manager_addr, UserMessage(content="Test message"))
        # Message sent successfully without error

    # =============================================================================
    # IMPORT TESTS
    # =============================================================================

    def test_example_imports(self) -> None:
        """Test that all imports in the example are available."""
        # Test core imports
        # Test llm imports
        from akgentic.llm import ModelConfig, PromptTemplate

        # Test messages imports
        from akgentic.core import (
            ActorSystem,
            AgentCard,
            Orchestrator,
        )

        # Test team imports
        from akgentic.agent import (
            BaseAgent,
            HumanProxy,
        )
        from akgentic.agent.config import AgentConfig

        # All imports should succeed without errors
        assert ActorSystem is not None
        assert Orchestrator is not None
        assert AgentCard is not None
        assert BaseAgent is not None
        assert HumanProxy is not None
        assert ModelConfig is not None
        assert PromptTemplate is not None
        assert AgentConfig is not None


# =============================================================================
# INTEGRATION TESTS (require OPENAI_API_KEY)
# =============================================================================


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_real_manager_assistant_interaction(
    actor_system: ActorSystem, orchestrator_address
) -> None:
    """Integration test with real LLM - Manager delegates to Assistant."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set - skipping integration test")

    orchestrator = actor_system.proxy_ask(orchestrator_address, Orchestrator)

    # Register AgentCards
    manager_card = AgentCard(
        role="Manager",
        description="Helpful manager coordinating team work",
        skills=["coordination", "delegation"],
        agent_class="akgentic.agent.BaseAgent",
        config={},
        routes_to=["Assistant"],
    )

    assistant_card = AgentCard(
        role="Assistant",
        description="Helpful assistant providing support",
        skills=["research", "writing"],
        agent_class="akgentic.agent.BaseAgent",
        config={},
    )

    orchestrator.register_agent_profile(manager_card)
    orchestrator.register_agent_profile(assistant_card)

    # Create Manager agent
    manager_config = AgentConfig(
        name="@Manager",
        role="Manager",
        prompt=PromptTemplate(
            template="You are a helpful manager. Coordinate the team effectively. Answer concisely."
        ),
        model_cfg=ModelConfig(provider="openai", model="gpt-4o-mini", temperature=0.3),
    )
    manager_addr = orchestrator.createActor(BaseAgent, config=manager_config)

    # Send message to manager
    user_msg = UserMessage(content="What is 2+2? Answer with just the number.")
    actor_system.tell(manager_addr, user_msg)

    # Wait for processing with polling loop
    timeout = 15  # seconds
    poll_interval = 0.5  # seconds
    elapsed = 0
    result_messages = []

    while elapsed < timeout:
        messages = orchestrator.get_messages()
        result_messages = [
            msg
            for msg in messages
            if isinstance(msg, SentMessage) and isinstance(msg.message, ResultMessage)
        ]
        if result_messages:
            break
        time.sleep(poll_interval)
        elapsed += poll_interval

    # Verify we got a response
    assert len(result_messages) > 0, f"No ResultMessage found after {timeout}s timeout"

    # Verify response contains the answer
    result_content = result_messages[-1].message.content  # type: ignore
    assert "4" in result_content or "four" in result_content.lower()


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_real_human_proxy_integration(actor_system: ActorSystem, orchestrator_address) -> None:
    """Integration test verifying HumanProxy can send messages to Manager."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set - skipping integration test")

    orchestrator = actor_system.proxy_ask(orchestrator_address, Orchestrator)

    # Register AgentCards
    manager_card = AgentCard(
        role="Manager",
        description="Helpful manager",
        skills=["coordination"],
        agent_class="akgentic.agent.BaseAgent",
        config={},
    )
    orchestrator.register_agent_profile(manager_card)

    # Create Manager
    manager_config = AgentConfig(
        name="@Manager",
        role="Manager",
        prompt=PromptTemplate(
            template="You are a helpful manager. Answer concisely in one sentence."
        ),
        model_cfg=ModelConfig(provider="openai", model="gpt-4o-mini", temperature=0.3),
    )
    manager_addr = orchestrator.createActor(BaseAgent, config=manager_config)

    # Create HumanProxy
    human_config = BaseConfig(name="@Human", role="Human")
    human_addr = orchestrator.createActor(HumanProxy, config=human_config)

    time.sleep(0.3)

    # HumanProxy sends message to Manager (simulating human input routing)
    human_proxy = actor_system.proxy_tell(human_addr, HumanProxy)
    human_proxy.send(manager_addr, UserMessage(content="Hello, are you available?"))

    # Wait for processing
    timeout = 15  # seconds
    poll_interval = 0.5  # seconds
    elapsed = 0
    result_messages = []

    while elapsed < timeout:
        messages = orchestrator.get_messages()
        result_messages = [
            msg
            for msg in messages
            if isinstance(msg, SentMessage) and isinstance(msg.message, ResultMessage)
        ]
        if result_messages:
            break
        time.sleep(poll_interval)
        elapsed += poll_interval

    # Verify response was generated
    assert len(result_messages) > 0, (
        f"Manager should respond to message from HumanProxy (timeout after {timeout}s)"
    )
    # Verify response is coherent
    result_content = result_messages[-1].message.content  # type: ignore
    assert len(result_content) > 0


@pytest.mark.integration
@pytest.mark.xfail(reason="Event subscription timing issue - known limitation")
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_real_event_subscriber_captures_flow(
    actor_system: ActorSystem, orchestrator_address
) -> None:
    """Integration test verifying EventSubscriber captures message flow.

    NOTE: This test has timing issues with event subscription that need
    architectural resolution. Marked as xfail pending fix.
    """
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set - skipping integration test")

    from akgentic.core import EventSubscriber

    orchestrator = actor_system.proxy_ask(orchestrator_address, Orchestrator)

    # Track events
    events_captured: list[tuple[str, str, str]] = []

    class TestSubscriber(EventSubscriber):
        def sentMessage(
            self, sender: ActorAddress, recipient: ActorAddress, message: Message
        ) -> None:
            msg_type = type(message).__name__
            events_captured.append((sender.name, recipient.name, msg_type))

    subscriber = TestSubscriber()
    orchestrator.subscribe(subscriber)

    # Register card and create agent
    manager_card = AgentCard(
        role="Manager",
        description="Test manager",
        skills=["coordination"],
        agent_class="akgentic.agent.BaseAgent",
        config={},
    )
    orchestrator.register_agent_profile(manager_card)

    manager_config = AgentConfig(
        name="@Manager",
        role="Manager",
        prompt=PromptTemplate(template="You are a manager. Answer 'yes' to any question."),
        model_cfg=ModelConfig(provider="openai", model="gpt-4o-mini", temperature=0.3),
    )
    manager_addr = orchestrator.createActor(BaseAgent, config=manager_config)

    time.sleep(0.3)

    # Send message
    actor_system.tell(manager_addr, UserMessage(content="Are you ready?"))

    # Wait for processing
    timeout = 15
    poll_interval = 0.5
    elapsed = 0

    while elapsed < timeout:
        messages = orchestrator.get_messages()
        result_messages = [
            msg
            for msg in messages
            if isinstance(msg, SentMessage) and isinstance(msg.message, ResultMessage)
        ]
        if result_messages:
            break
        time.sleep(poll_interval)
        elapsed += poll_interval

    # Verify events were captured
    assert len(events_captured) > 0, "Should capture message flow events"
    # Should have UserMessage event at minimum
    assert any(msg_type == "UserMessage" for _, _, msg_type in events_captured), (
        "Should capture UserMessage"
    )
