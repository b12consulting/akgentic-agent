"""HumanProxy: team-aware human-in-the-loop proxy agent.

Extends akgentic-core's UserProxy so a team agent can address an AgentMessage to
the human and get the human's reply back. ``process_human_input`` addresses that
reply to the sender of the message being answered — one hop, back to the agent
that asked.
"""

from __future__ import annotations

import logging

from akgentic.agent.messages import AgentMessage
from akgentic.core import ActorAddress, UserProxy
from akgentic.core.messages.message import Message

logger = logging.getLogger(__name__)


class HumanProxy(UserProxy):
    """UserProxy that receives team AgentMessages and relays the human's reply.

    HumanProxy extends UserProxy with:
    - receiveMsg_AgentMessage: receives any AgentMessage addressed to the human
      and logs it — the request is not yet queued or forwarded anywhere
    - process_human_input: sends the human's reply to the agent that sent the
      message being answered

    Usage:
        >>> from akgentic.core import ActorSystem
        >>> from akgentic.core.agent_config import BaseConfig
        >>> from akgentic.agent import AgentMessage, HumanProxy
        >>>
        >>> system = ActorSystem()
        >>> config = BaseConfig(name="human", role="HumanProxy")
        >>> proxy_addr = system.createActor(HumanProxy, config=config)
        >>>
        >>> # An agent asks the human something; `incoming` is what the proxy got
        >>> incoming = AgentMessage(content="Proceed?", recipient=proxy_addr)
        >>>
        >>> # Human responds; the reply is addressed to incoming.sender
        >>> system.proxy_ask(proxy_addr, HumanProxy).process_human_input(
        ...     "My answer", incoming
        ... ).get()

    Attributes:
        Inherits config and state from UserProxy (BaseConfig, BaseState).
    """

    def receiveMsg_AgentMessage(  # noqa: N802
        self, message: AgentMessage, sender: ActorAddress
    ) -> None:
        """Handle an AgentMessage from an agent — currently log-only.

        When an agent needs human input it sends an AgentMessage. This method
        logs it and nothing else: the message is not stored, queued or forwarded
        (see the comment below for what a real implementation would add). The
        human's response is sent separately, via process_human_input().

        Args:
            message: The AgentMessage from an agent.
            sender: The ActorAddress of the requesting agent.
        """
        logger.info(
            f"[{self.config.name}-{self.team_id}] Received '{message.type}' AgentMessage "
            f"from {sender} ({len(message.content)} chars)"
        )
        # In a real implementation, this would:
        # - Queue the request for human review
        # - Send to UI for human response
        # For now, just log it - human responds via process_human_input()

    def process_human_input(self, content: str, message: Message) -> None:
        """Process human input and send it back to the agent that asked.

        Takes the human's response to an AgentMessage and sends a new AgentMessage
        carrying it. The recipient (destinataire) is derived from
        ``message.sender`` -- i.e. the agent that originally sent the incoming
        message is the one that receives the human's answer. This is a single hop;
        any further routing is whatever that agent decides to do on its own turn.

        This method simulates the ``on_receive()`` lifecycle by setting
        ``_current_message`` before calling ``send()`` (so that ``parent_id``
        tracking works correctly) and clearing it afterwards.

        Args:
            content: The human's response text.
            message: The original AgentMessage being answered. Its ``sender``
                field determines the recipient of the outgoing response.

        Example:
            >>> # Agent A asks the human for input
            >>> incoming = AgentMessage(content="Should we proceed?", ...)
            >>> # Human responds
            >>> human_proxy.process_human_input("Yes, proceed", incoming)
            >>> # The reply is sent to Agent A (incoming.sender)
        """

        # Simulate the on_receive() lifecycle so that send() can read
        # _current_message for parent_id tracking.
        self._current_message = message
        try:
            # The sender of the AgentMessage is who we route the answer back to
            # This is the agent who directly asked for human input
            destinataire = message.sender
            assert destinataire is not None, "AgentMessage must have a sender"

            # Send the answer back to the requesting agent
            self.send(
                destinataire,
                AgentMessage(
                    content=content,
                    type="response",
                    recipient=destinataire,
                ),
            )
        finally:
            self._current_message = None
