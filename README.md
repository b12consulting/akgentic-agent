# akgentic-agent

[![CI](https://github.com/b12consulting/akgentic-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/b12consulting/akgentic-agent/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/gpiroux/69ad301e9b6491972aa7324eb8953f8a/raw/coverage.json)](https://github.com/b12consulting/akgentic-agent/actions/workflows/ci.yml)

LLM-driven collaborative agents for the
[Akgentic](https://github.com/b12consulting/akgentic-framework) multi-agent framework
(open-source bundle). `BaseAgent` composes the actor runtime, LLM integration, and tool infrastructure into a
single unit where agents communicate through a typed message protocol and route messages
to each other via structured LLM output.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Communication Model](#communication-model)
- [Message Protocol](#message-protocol)
- [Team Composition](#team-composition)
- [Configuration](#configuration)
- [Tool Channels](#tool-channels)
- [Runtime Model Switching](#runtime-model-switching)
- [Run Cancellation](#run-cancellation)
- [Examples](#examples)
- [Documentation](#documentation)
- [Development](#development)
- [License](#license)

## Overview

Each agent is an Akgent actor. When it receives an `AgentMessage`, it runs a REACT loop
(`ReactAgent.run_sync`) and returns a `StructuredOutput` — a list of `Request` objects
that each name a recipient and a message type. The framework resolves the recipients and
delivers the messages; the LLM navigates the conversation graph.

```
Human
  │  AgentMessage(content, type="request")
  ▼
HumanProxy ──send()──► BaseAgent (Manager)
                             │
                    receiveMsg_AgentMessage()   ← no decorator; just the work
                             │  hands the message over unchanged — it composes
                             │  no prompt of its own
                             │
                    act(message, StructuredOutput)   ← @guard_usage_limits()
                      │
                      ├─ append **Context update N** block (if shared state changed)
                      ├─ message.rendering() — the message frames itself:
                      │  "You received a request from @Human. A reply is
                      │   expected: respond to @Human with the result."
                      ├─ expand !!glob_pattern refs in the rendered string
                      │  (if WorkspaceTool present)
                      └─ ReactAgent.run_sync(prompt, output_type=StructuredOutput)
                           (a queued /stop or CancelMessage cancels the run at the
                            next step boundary — see Run Cancellation)
                             │
                    StructuredOutput.messages = [
                        Request(recipient="@Assistant", message_type="instruction", message="..."),
                        Request(recipient="Developer",  message_type="request",     message="..."),
                    ]
                             │
                    for each Request:
                      ├─ "@Name" → resolve to existing actor
                      └─ "Role"  → hire_member(role) → create actor
                             │
                    send AgentMessage(content=request.message,
                                      type=request.message_type)
                      └─ the RAW message — the sender does not enrich it.
                         The delivered message frames itself with its own
                         reply protocol when the receiver runs the loop above.
```

## Installation

Published on PyPI. Python 3.12 or newer.

```bash
uv add akgentic-agent
# or
pip install akgentic-agent
```

That is the whole install. `akgentic-core`, `akgentic-llm`, `akgentic-tool` and
`pydantic-ai` come with it as ordinary dependencies — no workspace checkout, no
submodules.

### As part of the framework bundle

`akgentic-framework` is the meta-distribution that pins every akgentic package
at versions built and tested together. Install `akgentic-agent` through it when
you want the release-wide pin rather than a single package:

```bash
pip install "akgentic-framework[agent]"   # this package + its closure, release-pinned
pip install "akgentic-framework[all]"     # the whole framework
```

### Working on the package itself

To develop `akgentic-agent` rather than use it, clone the open-source bundle
[akgentic-framework](https://github.com/b12consulting/akgentic-framework), which
carries every package together as submodules:

```bash
git clone git@github.com:b12consulting/akgentic-framework.git
cd akgentic-framework
git submodule update --init
# uncomment the two "SOURCE MODE" blocks in pyproject.toml
uv sync
```

Source mode resolves `akgentic-*` to the local checkouts, editable.

## Quick Start

```python
import time
from akgentic.agent import AgentConfig, AgentMessage, BaseAgent, HumanProxy
from akgentic.core import ActorSystem, AgentCard, BaseConfig, Orchestrator
from akgentic.llm import ModelConfig, PromptTemplate

# Actor runtime + Orchestrator
actor_system = ActorSystem()
orchestrator_addr = actor_system.createActor(
    Orchestrator, config=BaseConfig(name="@Orchestrator", role="Orchestrator")
)
orchestrator_proxy = actor_system.proxy_ask(orchestrator_addr, Orchestrator)

# Define and register a role blueprint.
# NOTE: there is no `role=` keyword on AgentCard — `card.role` is a read-only
# property reading `config.role`, which is the single source of truth.
manager_card = AgentCard(
    description="Project manager who coordinates specialists",
    skills=["coordination", "delegation"],
    agent_class="akgentic.agent.BaseAgent",
    config=AgentConfig(
        name="@Manager",
        role="Manager",
        prompt=PromptTemplate(template="You are a project manager. Delegate to specialists."),
        model_cfg=ModelConfig(provider="openai", model="gpt-4.1"),
    ),
    routes_to=["Developer", "QA"],   # roles this agent can hire on demand
)
# Register role blueprints — accepts a list of AgentCard
orchestrator_proxy.register_agent_profiles([manager_card])

# Human entry point
human_addr = orchestrator_proxy.createActor(
    HumanProxy, config=BaseConfig(name="@Human", role="Human")
)
human_proxy = actor_system.proxy_tell(human_addr, HumanProxy)

# Instantiate Manager and send the first message
manager_addr = orchestrator_proxy.createActor(
    BaseAgent, config=manager_card.get_config_copy()
)

time.sleep(0.3)

# Send a message from the human to the manager
human_proxy.send(manager_addr, AgentMessage(content="Plan the next sprint."))
```

## Communication Model

Every message in the system carries an **intent** — a declaration of what the sender
expects from the recipient. Intent is the core abstraction that drives conversation flow
between agents.

### Intent: the driving concept

When an agent sends a message, it declares its intent via a `message_type`:

| Intent | Meaning | Expected reply |
|---|---|---|
| `request` | "Do this and **bring me the result**" | `response` |
| `instruction` | "Do this (possibly for a third party)" | `acknowledgment` |
| `response` | "Here is what you asked for" | Optional |
| `notification` | "FYI — no action needed" | None |
| `acknowledgment` | "Got it" | None |

The key distinction is **who needs the result**: a `request` means "bring it back to
me", an `instruction` means "go do this on my behalf".

Intent flows through the system in two complementary ways:

1. **When sending** — The LLM chooses an intent for each outbound `Request`. `_route_output()`
   delivers the **raw** `request.message` as an `AgentMessage` whose `type` field carries that
   intent unchanged. The sender does not rewrite the content.

2. **When receiving** — `AgentMessage.rendering()` puts a one-line reply protocol, keyed on
   the message's own `type` via `REPLY_PROTOCOLS`, in front of its content. The guidance is
   therefore always the one matching the intent *that* agent received, and it reaches the LLM
   through the **prompt** — not through the output schema. The handler composes nothing.

### AgentMessage

All inter-agent communication uses a single `AgentMessage` type:

```python
class AgentMessage(Message):
    type: Literal["request", "response", "notification", "instruction", "acknowledgment"] = "request"
    content: str
```

The `type` field carries the sender's intent through the system. The first message is
typically sent by an external system (e.g., `HumanProxy`) as a `request` with plain
content. From there, each agent's LLM decides the intent it attaches to every outbound
message.

### StructuredOutput and Request

Each LLM call produces a `StructuredOutput` with a list of outbound `Request` objects:

```python
class Request(BaseModel):
    message_type: Literal[
        "request",        # ask recipient to perform a task and reply to you with the result
        "instruction",    # direct recipient to perform a task, you may ask for acknowledgement
        "response",       # respond to a previous request
        "notification",   # send information to the recipient, no reply is expected
        "acknowledgment", # confirm receipt of an instruction, no reply is expected
    ]
    message: str
    recipient: str   # "@MemberName" (existing actor) or "RoleName" (triggers hiring)

class StructuredOutput(BaseModel):
    messages: list[Request] = []
```

The LLM chooses both the **recipient** and the **intent** for every outbound message.
`Request.message_type` flows directly into the delivered `AgentMessage.type`, so every
receiver sees the sender's intent as first-class data.

An empty list means the agent has nothing more to send — but the LLM still runs. A
`notification` or `acknowledgment` means the output list should be empty, **not** that
the LLM call is skipped. The message is still processed and added to the agent's context
for future interactions.

### One Output Object per Turn

`_route_output()` consumes **one** `StructuredOutput` per turn. That is deliberate — routing N
outputs would mean every consumer downstream handles N — but until 2026-09 it was a contract
enforced only in Python, stated nowhere the model could read it.

**What went wrong.** `gpt-5.6-terra` on the OpenAI **Responses API** routinely answered with several
complete `StructuredOutput` objects, one per message it wanted to send, as separate text parts of a
single model response. The Responses API carries a *list* of output items, so this is expressible
there in a way a single-text-slot API cannot express. pydantic-ai has no representation for "several
outputs" and collapses them, two ways, both silent:

| The response also carries… | What happens |
|---|---|
| a tool call | the text is **discarded**, and still written to history — so the model reads back a delegation it never sent, concludes it already delegated, and never re-derives it. The recipient is simply never contacted |
| no tool call | the parts are **string-concatenated** into `{...}{...}`, which is not JSON. The run burns its output retries recovering — 13,758 tokens, 21% of a 64,054-token turn, measured |

**Why the model did it.** Not a provider defect, and not the model ignoring an instruction — the
instruction did not exist. The `messages` field description told it *"you may send several messages
in one turn — they are dispatched in parallel"*, and never said that several messages means several
**entries in this one list**. On a channel that can carry several output items, one item per message
is a reasonable reading of what it was handed.

**The fix is one sentence in the schema the model fills:**

```python
"CRITICAL: your entire reply for this turn is ONE output object. Several "
"messages means several entries in this list, never a second output object."
```

It lives in the field description rather than a system prompt because it is a property of the output
format: it travels with the JSON schema on every run, for every role, in every team — a prompt
sentence has to be repeated in each one and a role written next year will not have it. It also sits
directly beneath the *"several messages"* invitation it exists to reconcile.

### Static Schema + Prompt-Carried Reply Protocol

`act()` forwards the `output_type` it was handed straight to the REACT loop — there is no
per-call subclass and no `type()` metaprogramming on the hot path:

```python
output = self._react_agent.run_sync(prompt, deps=self, output_type=output_type)
```

`receiveMsg_AgentMessage()` calls `act(message, StructuredOutput)`, so the team
delegation path reasons against the **static** `StructuredOutput` type.

`Request.recipient` is a **plain string** with no `enum` constraint. Recipient validity is
enforced at **routing time** in `_route_output()`, not in the schema:

| Recipient format | Resolution |
|---|---|
| `@MemberName` | `get_team_member(name)` → direct send (skipped if not found) |
| `RoleName` | `hire_member(role)` → create actor → send |

The reply-protocol guidance lives where the LLM actually reads it — the **prompt**.
`AgentMessage.rendering()` puts a one-line protocol (keyed on the message's own
type via `REPLY_PROTOCOLS`) in front of the content, and `act()` calls it:

```
You received a request from @Human. A reply is expected: respond to @Human with the result.

<raw message content>
```

> **Note:** This supersedes the schema-constrained-recipient + docstring-injection
> mechanism from Story 5.1 / ADR-004. The intent-driven 5-type protocol is unchanged —
> only its enforcement moved from a per-call schema to the prompt + routing-time validation.

### Routing and Delivery

`receiveMsg_AgentMessage()` runs one LLM turn and hands the result to `_route_output()`,
which resolves each `Request.recipient` (see the table above) and sends the **raw**
`request.message` as an `AgentMessage`. The sender does not enrich the content —
the reply-protocol prefix is added by the *delivered message itself*, when `act()` renders
it on the receiving side, so the guidance is always keyed to the intent that agent actually
received:

```python
# AgentMessage.rendering(), called once by act() before the LLM turn:
return (
    f"You received {article} {self.type} from {sender_name}. "
    f"{REPLY_PROTOCOLS.get(self.type, '').format(sender=sender_name)}"
    f"\n\n{self.content}"
)
```

`_route_output()` returns **whether anything was actually delivered**. Nothing reads that bool
today — the guard that used to is retired — but it is what a caller would need to tell a real
answer from one that resolved no recipient and sent nothing.

A usage-limit breach escalates: `notify_human()` to the team's user-proxy member — found
structurally through `ActorAddress.is_user_proxy`, so any role string works; when the team has
none, the notice is logged and dropped — then `WarningError`. **Both tiers, identically**, and
from the `@guard_usage_limits()` decorator on `act()` rather than from any handler.

A run-tier breach that reaches that point has already had its second chance: `akgentic-llm`
concludes a breached turn itself, and the conclusion returns through `act()` as an ordinary
`StructuredOutput` that routes through this same `_route_output()`. There is nothing left for
this package to attempt, and nothing to tell the tiers apart *for*. See
[What happens when a limit is hit](#what-happens-when-a-limit-is-hit) and
[Writing a second agent class](#writing-a-second-agent-class).

### Writing a second agent class

`BaseAgent` handles one message type against one schema. A subclass that wants its own — its
own structured output, its own `receiveMsg_*` — **declares nothing at all** for usage limits.
`@guard_usage_limits()` sits on `BaseAgent.act` and `BaseAgent.compact`, the two methods that
reach the model, so a subclass inherits the policy by calling `act()` — which is the only way
it can reach the LLM in the first place.

That placement is the point. While the decorator was on the handlers it was a caller
obligation: invisible from `usage_limits.py`, and silently dropped by the one handler that
forgot it. Now it cannot be skipped, because *making the LLM call* is what triggers it. (The
one way to lose it is to override `act()` outright instead of calling `super().act()`.)

Two modules exist for what a subclass *does* need, and neither imports `agent.py` — they are
what a *new* agent class needs, so a dependency in that direction would make them unusable
from the module that defines the base class. What each needs from an agent is stated as a
`Protocol`:

| module | holds |
|---|---|
| `usage_limits.py` | `AgentLike`, `guard_usage_limits`, `escalate_usage_limit` |
| `utils.py` | `TeamResolver`, `resolve_recipient` — the team addressing convention (`@member` vs role to hire) |

So the subclass is only its own work:

```python
from akgentic.agent.utils import resolve_recipient


class CustomAgent(BaseAgent):
    def _route_triage(self, output: TriageOutput) -> bool:
        """Deliver the output — and report whether anything actually went out."""
        delivered = False
        for handoff in output.handoffs:
            member = resolve_recipient(self, handoff.recipient)
            if member is None:
                continue
            self.send(
                member,
                AgentMessage(content=handoff.task, type="request", recipient=member),
            )
            delivered = True
        return delivered

    def receiveMsg_TriageMessage(self, message: TriageMessage, sender: ActorAddress) -> None:
        output = self.act(message, TriageOutput)
        self._route_triage(output)
```

Four things follow, and they are the whole point:

- **The handler is undecorated, and carries no usage-limit handling.** The policy arrives with
  the `act()` call; the body reads as just the work.
- **A breached turn still concludes in `TriageOutput`** — because `akgentic-llm` reuses the
  `output_type` this body already passes to `act()`. The subclass declares no schema for the
  conclusion because it already declared one for the turn.
- **A concluded turn is indistinguishable from an ordinary one.** It comes back from `act()`
  as a `TriageOutput` and goes through `_route_triage` like anything else. Nothing branches.
- **The handler carries no `except` at all — not for cancellation, not for budget.** The
  cancel capability is unconditional on every `BaseAgent` subclass, so every run is
  interruptible, but the interruption never reaches the handler: `act()` absorbs
  `RunInterruptedError` itself, notifies the human once, and returns a default
  `TriageOutput()`. A usage breach is absorbed in the same place, by the guard (see
  [Run Cancellation](#run-cancellation)).

**Known gap.** A conclusion that hands off to nobody is silent: an ordinary success that routes
nothing and notifies no one. `akgentic-llm` cannot see it (`TriageOutput` has no `.messages`,
and it receives the output as `Any`) and the guard never sees the output at all. Tracked as
ADR-021 §Q2, and pinned by
`test_custom_agent.py::TestCustomAgentUsageBreach::test_a_triage_with_no_handoffs_is_silent`.

The runnable version is `src/akgentic/agent/custom_agent.py`.

### HumanProxy

`HumanProxy` extends `UserProxy` from `akgentic-core`. It serves two roles:

- **Message sink** — `receiveMsg_AgentMessage()` logs receipt. The base implementation
  publishes nothing of its own; subscribers see the content through the `SentMessage` the
  *sending* agent emits. Override the hook to queue for a console printer, a WebSocket to a
  frontend, WhatsApp, email, etc.
- **Human input bridge** — `process_human_input()` routes a human's reply back to
  `message.sender` as an `AgentMessage` with `type="response"`.

```python
# Send a message from the human to an agent
human_proxy = actor_system.proxy_tell(human_addr, HumanProxy)
human_proxy.send(agent_addr, AgentMessage(content="Do X"))

# Route a human reply back to the agent that asked
human_proxy.process_human_input("My answer", original_message)
```

## Message Protocol

The 5-type intent protocol controls conversation flow. `AgentMessage.rendering()` puts
the `REPLY_PROTOCOLS` instruction matching its own `type` into the **user prompt** (not the
output schema), so the LLM reads the guidance inline with the content:

| Intent | Receiver instruction (`REPLY_PROTOCOLS`) |
|---|---|
| `request` | "A reply is expected: respond to {sender} with the result." |
| `response` | "This is a reply to something you asked. Take it into account and continue." |
| `instruction` | "Carry it out; acknowledge to {sender} only if asked to." |
| `notification` | "Informational message. No reply is expected." |
| `acknowledgment` | "Receipt confirmed. No further action needed." |

**These lines state message mechanics only** — what kind of message arrived, and whether a
reply is expected. They deliberately say nothing about *who* should do the work or whether
to delegate, because they sit at the most salient position in every agent's prompt, in every
team. Team policy that is stated there is stated to everyone at once: the earlier `request`
text read "Carry out the task, then respond to {sender}. You may also delegate to others",
and that single line is what made coordinators do their specialists' work. Wording it the
other way round is the same mistake with the sign flipped — it makes specialists fan out to
each other. Division of labour is per-role, so it belongs in the agents' own prompts.

The protocol is **soft guidance**, not framework enforcement. The LLM is guided to send no
further messages for `notification` and `acknowledgment`, but the framework processes
whatever the LLM returns. This is intentional: LLMs are probabilistic, and rigid
enforcement would be brittle.

**No reply does not mean no processing.** When the protocol says "return an empty list",
the LLM still runs — it absorbs the message into its context, which may inform future
decisions. The empty list simply means no outbound messages are sent.

## Team Composition

### AgentCard

Declarative role definition registered with the Orchestrator. Acts as a blueprint: agents
can be instantiated from it on demand without hard-coding actor addresses.

```python
AgentCard(
    description="Writes and reviews code",
    skills=["python", "testing"],
    agent_class="akgentic.agent.BaseAgent",   # FQCN string or class reference
    config=AgentConfig(name="@Developer", role="Developer"),
    routes_to=["Reviewer", "Tester"],         # roles this agent can hire
)
```

**`role` is not a constructor keyword.** `AgentCard.role` is a read-only property that reads
`config.role`, so that config field is the single source of truth. Passing `role=` to the
constructor is silently ignored by Pydantic — the card would end up with whatever
`config.role` says, or an empty role if you set neither.

`register_agent_profiles([card, ...])` stores cards in the Orchestrator so any agent can
hire a role by name without knowing the class.

`AgentCard.get_config_copy()` returns a fresh `AgentConfig` suitable for `createActor()`.

### Dynamic Hiring

When `_route_output()` sees `recipient="Developer"` (no `@` prefix), `resolve_recipient()`
calls `hire_member("Developer")`, which resolves the registry's typed `hire_member` command
(`TeamTool`) and invokes it. That command:

1. Looks up the `AgentCard` for `"Developer"` in the Orchestrator
2. Calls `createActor(agent_class, config=card.get_config_copy())`
3. Returns the new actor address for immediate message delivery

If no `hire_member` command is registered — `TeamTool` was removed from `config.tools` — the
method raises `RuntimeError` rather than failing silently.

The LLM in the sending agent triggers this transparently by naming a role instead of a
team member.

### EventSubscriber

Attach an `EventSubscriber` to the Orchestrator to observe all messages and events:

```python
from akgentic.core import EventSubscriber
from akgentic.core.messages import Message
from akgentic.core.messages.orchestrator import EventMessage, SentMessage
from akgentic.llm import ToolCallEvent

class MessagePrinter(EventSubscriber):
    def on_message(self, message: Message) -> None:
        if isinstance(message, SentMessage):
            print(f"[{message.sender.name}] → {message.recipient.name}: {message.message.content}")
        elif isinstance(message, EventMessage) and isinstance(message.event, ToolCallEvent):
            print(f"TOOL: {message.event.tool_name}")

orchestrator_proxy.subscribe(MessagePrinter())
```

## Configuration

### AgentConfig

Extends `BaseConfig` from `akgentic-core`:

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | `PromptTemplate` | `PromptTemplate()` | Agent backstory rendered into `AgentState.backstory` and injected as LLM system prompt |
| `model_cfg` | `ModelConfig` | `ModelConfig()` | LLM provider, model name, API settings. Accepts `ModelConfig \| list[ModelConfig]` **at the input boundary only** — see [The model roster](#the-model-roster) |
| `model_roster` | `list[ModelConfig]` | `[]` | The full declared roster, in declaration order, including the active entry. Empty means a single-model agent, for which switching is unavailable |
| `runtime_cfg` | `RuntimeConfig` | `RuntimeConfig()` | Retries, tool-call end strategy, parallel tools, HTTP client settings |
| `run_usage_limits` | `RunUsageLimits` | `RunUsageLimits()` | Budget for **one** `run()` — token and request caps that reset every run |
| `agent_usage_limits` | `AgentUsageLimits` | `AgentUsageLimits()` | Budget for the agent's **whole lifetime** — runs and tokens, accumulated across every run |
| `compaction_cfg` | `CompactionConfig` | `CompactionConfig()` | Context-compaction strategy and auto-trigger (opt-in; off unless `model_cfg.context_length` is set) |
| `tools` | `list[ToolCard]` | `[]` | Tool cards; `TeamTool` and `MailboxTool` are always prepended automatically |

#### The model roster

`model_cfg` and `model_roster` are one declaration with two spellings. Passing a **list** to
`model_cfg` is a convenience for declaring a roster; passing a single `ModelConfig` — which is
what every existing config does — declares no roster at all.

```python
config = AgentConfig(
    name="@Manager",
    role="Manager",
    model_cfg=[                                              # a list, at the input boundary
        ModelConfig(provider="openai", model="gpt-4.1"),     # element 0 = the active model
        ModelConfig(provider="anthropic", model="claude-sonnet-4-5"),
    ],
)
assert config.model_cfg.model == "gpt-4.1"      # stored shape is always a single ModelConfig
assert len(config.model_roster) == 2            # the whole list became the roster
```

Four rules govern the boundary, and a caller can trip over each of them:

- **Given a list, element 0 becomes the active model and the whole list becomes the roster**
  (`config.py:128-142`). The active entry is a member of its own roster by construction.
- **An empty list is rejected.** There is no "roster of nothing" — declare a single
  `ModelConfig` instead.
- **A list `model_cfg=` passed together with an explicit `model_roster=` is rejected**, because
  which one wins would depend on argument order.
- **Two entries producing the same `provider:model` key are rejected** (`config.py:144-161`). The
  key is the identity a switch is named by, so a duplicate would not raise later — it would
  produce a switch that silently matches one of two entries.

**The stored shape is always a single `ModelConfig`.** No read path downstream branches on the
union: the list exists at validation time and nowhere else, which is what makes accepting it safe.

**An empty roster is the default, and it means switching is unavailable.** Every existing
single-model agent, catalog entry and example is unchanged by this feature: `model_roster == []`,
no `list_models` or `switch_model` commands, no `LLM_CONTEXT` block naming a model, and no cost. If
you are upgrading and you declare one model, nothing changed for you.

Hand-setting `model_roster=` directly is legal, and has consequences worth knowing before you do
it — see [When the active model is not in the roster](#when-the-active-model-is-not-in-the-roster).

#### Usage limits: two tiers

The two budgets answer different questions, and both are carried into the `ReactAgent` that
`BaseAgent` builds. **Neither is enforced in `akgentic-agent`** — this package configures,
`akgentic-llm` enforces.

| tier | class | bounds | enforced by |
|---|---|---|---|
| run | `RunUsageLimits` | one `run()` call — requests, tool calls and tokens *within* it | pydantic-ai, mid-run; counts reset every run |
| agent | `AgentUsageLimits` | the agent's whole lifetime — `run()` calls and cumulative tokens | `ReactAgent`, pre-flight before each run |

```python
from akgentic.agent.config import AgentConfig
from akgentic.llm import AgentUsageLimits, ModelConfig, RunUsageLimits

config = AgentConfig(
    name="@Manager",
    role="Manager",
    model_cfg=ModelConfig(provider="openai", model="gpt-4.1"),
    run_usage_limits=RunUsageLimits(run_request_limit=50, total_tokens_limit=100_000),
    agent_usage_limits=AgentUsageLimits(agent_request_limit=200, total_tokens_limit=2_000_000),
)
```

Both defaults are safe to leave alone. `RunUsageLimits()` keeps a 50-request-per-run brake;
`AgentUsageLimits()` is all-`None`, and an all-`None` budget never blocks — that is why the
field is never `None` itself, and why adding a lifetime cap is opt-in rather than a
behaviour change.

**The agent tier survives a resume.** Its counters are not persisted and are not part of
`AgentState`. On restore, `ReactAgent` recomputes them from the replayed usage events the
team restorer already feeds through `init_llm_context()` — so an agent that has spent 180 of
its 200 runs comes back with 20 left, not 200. Two consequences worth knowing:

- The lifetime token limits bound where a run may **start**, not where it may end. A run's
  cost is unknown until it completes, so the run that crosses the line finishes and only the
  next one is refused.
- `agent_request_limit` is consumed *before* the call executes, so a run that fails partway
  still counts against the lifetime budget.

**A retrying tool now costs an extra model turn.** Under `end_strategy="exhaustive"` (the
default), pydantic-ai v2 suppresses an output produced in the same round as a function tool
that raised `ModelRetry`, and keeps the run open for another model turn. Since agents here
routinely emit a `StructuredOutput` alongside a tool call, and tools raise `ModelRetry` by
design, that second turn is charged to *both* tiers — so an agent near either budget can
trip a limit on a turn that previously completed. Budget for it when sizing tight limits.

The two tiers raise **two distinct classes** — `RunUsageLimitError` for the run tier,
`AgentUsageLimitError` for the agent tier — and both subclass the `UsageLimitError` that
predates the split. A caller that already writes `except UsageLimitError` therefore still
catches both and needs no change. Code that has to tell the tiers apart does so by **class**
(`isinstance`, or the order of its `except` clauses), never by reading the message text. All
three classes are `akgentic-llm`'s.

##### What happens when a limit is hit

**Every usage-limit error that reaches this package produces the same thing: the human is
notified and the turn ends.** The `@guard_usage_limits()` decorator on `act()` catches the
**base** `UsageLimitError`, so both tiers — and any tier added later — take that one path.

That is not the whole story of a breached turn, though: most run-tier breaches never reach this
package at all.

| where | what happens |
|---|---|
| `akgentic-llm`, run-tier breach | `LimitRecoveryCapability` decides whether the turn degrades, and by default drives one **tool-free conclusion** through the `output_type` this agent already asked for |
| conclusion succeeds | returns through `act()` as an ordinary output, routes through the normal path — **nothing here notices, and no human is told** |
| conclusion declined or failed | `akgentic-llm` re-raises the **ORIGINAL** breach as `RunUsageLimitError` → notify, `WarningError` |
| `akgentic-llm`, agent-tier breach | raised pre-flight as `AgentUsageLimitError`, terminal → notify, `WarningError` |

A run-tier breach means *this turn* ran out of requests, tool calls or tokens. The agent itself
usually still has lifetime budget, and by that point usually has most of what it was asked for:
it can no longer call a tool, but it can still answer — so `akgentic-llm` asks it to, once. The
prompt for that final call tells the model to answer now and to **state explicitly which parts
it could not check or finish**. An answer produced this way is expected to be incomplete and to
say so; read it as a partial result, not a finished one.

An agent-tier breach means the *lifetime* budget is spent — the budget that would pay for a
closing call is exactly what ran out — so there is no conclusion to attempt. That is terminal
by construction rather than by policy, which is a known limitation (ADR-021 §Q1): an exhausted
agent stops mid-conversation with no final word.

There is no retry counter, and none is needed *where a lifetime budget is set*: the agent tier
is consumed before every call, the conclusion call included, so an agent that keeps breaching
the run tier walks into its terminal tier by construction. That bound is only as real as the
budget behind it — the default `AgentUsageLimits()` is all-`None` and never blocks, so an agent
left on the defaults breaches, concludes and breaches again with nothing to stop it and no one
told. Set `agent_usage_limits` if you want that backstop.

**Known gap: a conclusion that succeeds emptily is silent.** A `StructuredOutput` with no
requests is an ordinary success — nothing raises, nothing is routed, nobody is notified.
`akgentic-llm` cannot judge it (it sees the output as `Any`) and the guard never sees the output
at all. Tracked as ADR-021 §Q2; today's behaviour is pinned by
`test_usage_limit_handling.py::TestARescuedTurnIsIndistinguishable::test_a_conclusion_that_routes_nothing_is_silent`
so the day it changes is visible in the diff.

**Who owns what.** `UsageLimitError`, `RunUsageLimitError`, `AgentUsageLimitError`, and the
whole of degradation — whether a breached turn concludes, with what prompt, and whether the
result was worth returning — belong to **`akgentic-llm`**. This package imports the classes and
never redefines them, and it no longer concludes anything. What belongs to **`akgentic-agent`**,
in `usage_limits.py`, is one response to an error that has already exhausted its options:
notify, then stop. Change enforcement *and* degradation policy in `akgentic-llm` — the seam for
the latter is `LimitRecoveryCapability.handle_limit_exceeded`.

##### Migrating from `usage_limits`

`AgentConfig.usage_limits` was the single pre-split budget. It is now the run tier under a
new name:

```python
# before
AgentConfig(usage_limits=UsageLimits(request_limit=50, total_tokens_limit=100_000))

# after
AgentConfig(run_usage_limits=RunUsageLimits(run_request_limit=50, total_tokens_limit=100_000))
```

The old spelling still works: passing `usage_limits=` emits a `DeprecationWarning` and populates
`run_usage_limits`, and reading `config.usage_limits` returns the run tier. Passing both
`usage_limits=` and `run_usage_limits=` raises `ValueError` rather than silently picking one.
**Both are removed in akgentic-agent 2.0.0.**

`UsageLimits` — the pre-split class itself — is a separate, `akgentic-llm`-owned deprecated alias
of `RunUsageLimits`. It still ships and still warns; **its removal is not scheduled for a named
release.** Only the two `AgentConfig` shims above carry a fixed removal target.

### AgentState

Runtime state extending `BaseState`:

| Field | Type | Description |
|---|---|---|
| `backstory` | `str` | `config.prompt` rendered at `on_start()`, injected as LLM system context on every call |
| `tool_state` | `ToolState` | The tool layer's persistent per-agent slot — context-update baselines, the block counter, and `active_model`. See [Context updates](#context-updates) and [Runtime Model Switching](#runtime-model-switching) |

`tool_state` carries **three** things, and they are harmless to lose for two different reasons:

- **The baselines and the block counter** are a **cache, never a record**. The message history is
  the record of what the model was told, so a lost or stale slot costs at most one full-snapshot
  re-send and never a lost update.
- **`active_model`** (`ToolState.active_model`, a roster key or `None`) is **not** a cache — it is
  the only place the remembered model choice lives. Losing it is still harmless, but for the other
  reason: the agent degrades to the *declared* active entry rather than to a re-send. `None` means
  the agent expresses no preference, which is also what a payload persisted before the field
  existed restores to — so no migration step is needed.

## Tool Channels

`ToolFactory` organises tool cards into four channels:

| Channel | Consumer | Examples |
|---|---|---|
| `TOOL_CALL` | LLM via pydantic-ai tools | `hire_members()`, `fire_members()`, `read_mailbox()`, `web_search()`, `workspace_read()`, `list_models()`, `switch_model()` |
| `SYSTEM_PROMPT` | LLM system prompt — rendered into the frozen system block | backstory, current date |
| `LLM_CONTEXT` | LLM via a per-turn appended **Context update** block | team roster, role profiles, planning summary, knowledge-graph summary, the model in force |
| `COMMAND` | `CommandRegistry` — in-agent Python and `/`-prefixed messages | `hire_member`, `fire_member`, `team_members`, `team_roles`, `planning_summary`, `stop`, `list_models`, `switch_model` |

`list_models` and `switch_model` serve **both** `TOOL_CALL` and `COMMAND`
(`akgentic-tool/.../model/tool.py:131-162`), and that is the point: the human and the model reach
the same capability rather than two parallel implementations of it. They are present only when
`ModelTool` is in `config.tools` — see [Runtime Model Switching](#runtime-model-switching).

`TeamTool` **and** `MailboxTool` are always prepended to `config.tools` if not already
present, so every `BaseAgent` can hire and fire members (`TeamTool`) and carries the two
mailbox surfaces — the `read_mailbox` tool, which takes the **id** of one waiting message and
acknowledges it, and `/stop` (`MailboxTool`). The tool consumes nothing itself:
`MailboxCapability.after_tool_execute` absorbs exactly the message the model named, so that
one is not delivered again as its own turn, and injects that message's own
`rendering()`. Mail the model does not name stays queued, and a cancel is never offered
and never absorbed. A card already supplied in `config.tools` wins over the prepended
default, and `config.tools` itself is never mutated — `on_start()` copies the list.

### Assembly: what `on_start` collects

`on_start()` walks the card set once through the `ToolFactory` and consumes each channel
exactly once:

| Channel | Collected in `on_start` via | Consumed |
|---|---|---|
| `TOOL_CALL` | `tool_factory.get_tools()` / `get_toolsets()` | handed to the `ReactAgent` at build |
| `SYSTEM_PROMPT` | `tool_factory.get_system_prompts()` | registered once into the frozen system block |
| `LLM_CONTEXT` | `tool_factory.get_context_states()` | providers held for the agent's lifetime; diffed and delivered per turn by `_deliver_context_update` |
| `COMMAND` | `tool_factory.get_command_registry(extra_commands=[compact, clear])` | one `CommandRegistry`, announced once via `CommandsAnnouncedEvent` |

**`BaseAgent` grows behaviour by hosting cards, not by accreting methods.** `MailboxTool`
is the worked example: two capabilities, each riding its own channel — `read_mailbox` on
`TOOL_CALL`, `/stop` on `COMMAND` — and the agent gained both without a single new method.
The card serves no `LLM_CONTEXT` at all: mailbox awareness reaches the model through the
agent's mid-run arrival notice alone, a second card-side carrier having only narrated the
same arrivals twice. When the next feature is a capability the LLM, the
operator, or the context should see, write it as a card and let the table above route each
piece to the hook that serves it. The card-author side of this contract is the
`akgentic-tool` README's *Building a feature as a card* authoring guide.

### Adding a capability of your own

A card is the right shape for anything the LLM, the operator or the context should see. A
**pydantic-ai capability** is the right shape for something that has to run *around every
model request* — an observability wrapper, a domain guard, a tenant-resolution hook. Those
have no channel to ride, so `BaseAgent` offers a hook instead:

```python
from typing import Any

from pydantic_ai import AgentCapability

from akgentic.agent.agent import BaseAgent


class AuditedAgent(BaseAgent):
    def extra_capabilities(self) -> list[AgentCapability[Any]]:
        return [AuditCapability(self.config.name)]
```

`AuditCapability` is yours to write — any `pydantic_ai.capabilities.AbstractCapability`
subclass, or a plain capability function.

`_assemble_capabilities` builds the list once, and `on_start` hands it to whichever build
site runs:

```python
self._capabilities = [self._mailbox_capability, *self.extra_capabilities()]
```

Two things follow, and both are deliberate:

- **The framework's own capability is prepended, never returned by the hook.** Cancellation
  is unconditional — a subclass that forgot to call `super()` would otherwise silently lose
  the ability to be stopped. Do not return `MailboxCapability` from `extra_capabilities()`;
  you would get two of it.
- **Mailbox-first is an ordering guarantee, not an accident.** Hook order is registration
  order, so the cancel check runs before any custom capability's work: a run that is about
  to be cancelled does not first pay for a third party's `before_model_request`.

`extra_capabilities()` is called from `_assemble_capabilities`, during `on_start` and *before*
`self._react_agent` exists — so an override may read `self.config`, but must not touch the
ReactAgent or anything built later in `on_start`. `AgentCapability` is the union of
`AbstractCapability` and a plain capability function, so either shape is accepted. There is
no per-capability lifecycle for the framework to drive: a capability that needs per-run state
resets it in its own `before_run` hook, which is what `MailboxCapability` does for its
announced-id set. The framework never iterates the list. `CustomAgent` in `custom_agent.py`
carries a runnable version of the above.

### Context updates

Volatile, team-shared state — the roster, role profiles, planning, the knowledge-graph summary
— never enters the system prompt: the system block holds the backstory and the current date only, and
stays byte-identical run to run so the prompt-cache prefix survives. Instead, before each run the
agent appends **at most one block** at the tail of the conversation carrying what changed since the
last block it delivered.

The engine that composes that block is not in this package. `akgentic.tool.core.ContextUpdater` —
built once by `ToolFactory.get_context_updater()` at `on_start()` and held for the agent's lifetime
— reads the state providers, diffs them against the baselines, composes the block and advances the
counter; `akgentic-tool` owns those semantics along with the cards that produce the state.
`BaseAgent` contributes only *when* — one delivery site, at the top of `act()`, before the run — and
*how* — the append goes through `ContextManager.append_user_prompt`, so a fresh agent's first
block is folded into the first run's user prompt instead of suppressing system-prompt injection.

The block opens with a marker line, `**Context update N**`, followed by one of two **fixed**
suffixes:

```
**Context update N** — current state.
**Context update N** — state has changed since the last update.
```

- `N` is monotonic per agent and advances only when a block is actually appended.
- The *current state* wording is used when no diff baselines survive as delivery begins — the first
  block of an agent's life, and any block after `/clear` or an eviction. Every section in such a
  block is a full snapshot.
- The *state has changed* wording is used when the block was diffed against surviving baselines:
  its sections are deltas, plus a full rendering for any provider contributing for the first time.
- **When nothing changed, nothing is appended** — an idle turn adds only the user's own message.

#### Where the baselines live

The baselines and the block counter persist on `AgentState.tool_state`. They ride the state
checkpoints the agent already emits — **no new event, no forced publish**: the engine mutates the
slot in place, and change detection compares serializations, so the existing `notify_if_changed()`
picks it up on its own.

The slot is a **cache, never a record.** The message history remains the durable record of what the
model was actually told, so a slot that is lost or stale costs at most one full-snapshot re-send,
and never a lost update. That is what makes persisting it lazily safe.

The payoff is on restore. A restored agent whose history still contains its last **Context update**
block resumes delta delivery — it says only what changed while it was gone, which is **usually
nothing** — instead of re-appending the whole roster, planning and knowledge-graph snapshot on every
restart. Only an agent whose history lost its blocks (compaction, `/clear`, a sliding-window trim)
falls back to a full snapshot.

The mechanism is self-healing: before trusting its baselines the engine reconciles them against the
markers still visible in the history.

- **The last delivered marker is still there** — the baselines are trusted and the next block is a
  delta.
- **The marker is gone** — every baseline is dropped and the next block is a full snapshot. The
  counter is *not* reset: a partially trimmed history may still show older numbers, so `N` stays
  monotonic.
- **The persisted counter is behind the history** — a crash between the append and the next
  checkpoint. The counter catches up to the highest visible marker and the baselines are **kept**,
  so the next block re-states what the missed blocks said: a repeat, never an omission.

`clear()` is the one legitimate zeroing of the counter — it empties the history and the slot
together, so the next block is `**Context update 1** — current state.` `compact()` gets no reset of
its own: the reconciliation above catches a compacted-away block either way.

> **Never cache `state.tool_state`.** `init_state()` replaces the whole state object on restore, so
> a held reference goes silently stale. Read the slot through `self.state` on every use — which is
> exactly what the engine does, on every call.

In the transcript, context-update blocks appear as **user-role messages**, the same way operator
actions do. The marker line is the stable handle for finding, collapsing, or styling them.

### The Command Registry

`on_start()` builds **one** `CommandRegistry` from every `COMMAND`-channel capability of the
agent's tool cards, adds `compact` and `clear` as command-only built-ins, and announces the whole
set once as a `CommandsAnnouncedEvent`:

```python
self._command_registry = tool_factory.get_command_registry(
    extra_commands=[self.compact, self.clear]
)
self.notify_event(
    CommandsAnnouncedEvent(
        agent=self.myAddress,
        commands=self._command_registry.descriptors(),
    )
)
```

Commands are keyed by the **callable's `__name__`**. The canonical names are therefore
`hire_member`, `fire_member`, `team_members`, … — there is no `cmd_` prefix on any of them.

Two surfaces reach the same table:

| Surface | Call | Returns |
|---|---|---|
| **human / text** | `registry.dispatch("/hire_member Developer")` | `str` — the result, rendered |
| **typed / in-agent** | `registry.callable("hire_member")("Developer")` | the command's **native** value (here an `ActorAddress`) |

`registry.has(name)` tests availability before either call, and `registry.descriptors()` returns
serializable discovery metadata — name, description, argument schema, and owning tool card.
`BaseAgent` uses both surfaces itself: `hire_member()` resolves the typed callable, and `act()`
expands media references the same way.

```python
if not self._command_registry.has("hire_member"):
    raise RuntimeError("hire_member command not available — TeamTool not configured")

hire = self._command_registry.callable("hire_member")
return cast(ActorAddress, hire(role))
```

#### Slash commands: how a human drives an agent

A message whose content starts with `/` is intercepted in `receiveMsg_AgentMessage()` **before**
the LLM path and handed to `_dispatch_command()`. That method dispatches the text, replies to the
sender with a `notification` `AgentMessage` carrying the result, and records one
human-attributed operator action in the agent's LLM context — so the agent reasons about what the
human did on its next turn, without mistaking it for its own tool call.

```python
human_addr.send(manager_addr, AgentMessage(content="/team_members"))
human_addr.send(manager_addr, AgentMessage(content="/hire_member DevOpsEngineer"))
human_addr.send(manager_addr, AgentMessage(content="/fire_member @DevOpsEngineer456"))
```

An unrecognised leading token raises `CommandNotRecognized`, which `_dispatch_command()` swallows
so the message falls through to the normal LLM path with its original content — a sentence that
happens to start with a slash is never lost, and nothing is injected into the context. Failures
*after* a command has been identified (missing or malformed arguments, or the command body
raising) are caught inside `dispatch()` and returned as a result string; those never fall back to
the LLM.

A dispatched command may also return `None`, meaning *handled, say nothing*: the message counts
as handled — it never reaches the LLM — but there is no reply to the sender and no operator
action recorded. That is the outcome for a command whose whole effect happens elsewhere, which
an empty reply would only double-report.

Arguments are `shlex`-split and coerced against the command's signature. A token is treated as a
keyword only when the text before its first `=` names a real parameter, so
`/hire_member Developer name=@Ada` binds both, while a positional value containing `=` is left
intact.

#### Which commands exist

The registry contents follow from the tool cards attached to the agent:

| Command | Provided by | Description |
|---|---|---|
| `hire_member(role, name=None)` | `TeamTool` | Hire by role; native return is the new `ActorAddress` |
| `fire_member(name)` | `TeamTool` | Fire a member by name |
| `team_members()` | `TeamTool` | Current team roster |
| `team_roles()` | `TeamTool` | Available roles and descriptions |
| `planning_summary()` | `PlanningTool` | Full team planning text |
| `get_planning_task(task_id)` | `PlanningTool` | Single planning task by ID |
| `search_planning(...)` | `PlanningTool` | Search the shared task board |
| `stop()` | `MailboxTool` | Cancel the current run; the mid-run effect is the cancel hook's (see [Run Cancellation](#run-cancellation)) |
| `list_models()` | `ModelTool` | The roster this agent may switch within, one entry per line, the entry in force marked |
| `switch_model(model)` | `ModelTool` | Make one roster entry the model in force, from the next turn (see [Runtime Model Switching](#runtime-model-switching)) |
| `compact()` / `clear()` | `BaseAgent` built-ins | Compact or clear the conversation context |

`ModelTool` is **not** auto-injected, so its two rows are present only for an agent whose
`config.tools` declares the card. Note also that the card's parameter is named `model` while the
observer's is `key` — two contracts, two names, deliberately: the command descriptor a frontend
reads advertises `model` (`akgentic-tool/.../model/tool.py:200-213`), so a human types
`/switch_model openai:gpt-4.1`.

Do not hand-transcribe this table into your own code: read the set from
`registry.descriptors()`, or from the `CommandsAnnouncedEvent` the agent emits at start-up. Those
cannot drift from the registry; a copied list can.

### Methods on the Pykka proxy

Separately from the command channel, `BaseAgent`'s own public methods are reachable through
`actor_system.proxy_ask(agent_addr, BaseAgent)`:

| Method | Returns | Description |
|---|---|---|
| `get_usage_summary(by_run)` | `AgentUsageSummary` | Aggregated LLM usage and cost; queries the orchestrator for `LlmUsageEvent`s and folds them via `aggregate_usage()` from `akgentic.llm`. Pass `by_run=True` for a per-run breakdown. |

### Media Expansion

When the registry carries an `_expand_media_refs` command — `WorkspaceTool` is what provides it —
`act()` expands inline file references before the LLM call. Expansion runs on the **rendered**
string, after `message.rendering()`, so a `!!glob` written anywhere in a message's own
framing expands exactly as one written in its content does:

```
!!file.png               → BinaryContent injected into the prompt
!!"my screenshot.png"    → same, for paths with spaces
!!*.png                  → glob — every matching image, sorted by path
!!report.pdf             → "!!report.pdf[=> Use workspace_read tool]" forwarded to the LLM
!!nonexistent.png        → "!!nonexistent.png[Error: no image found in the workspace]"
```

Expansion happens in `act()` between the render and `run_sync()`, and only when the expansion
actually changed something: if the command returns the rendered string unchanged, that plain string
is sent as-is. Errors and document hints are forwarded to the LLM rather than silently dropped.
Agents whose registry has no `_expand_media_refs` are unaffected — the block is a no-op.

## Runtime Model Switching

An agent can declare a **roster** of models and move between them while it is running — chosen by
a human typing `/switch_model`, or by the model itself calling the `switch_model` tool. The
selection persists, so a restarted agent answers on the model it was switched to.

The feature is built across three packages and no one of them can show you the whole of it, so
this section starts with the path end to end.

### The path, end to end

```
a human types /switch_model openai:gpt-4.1     ─┐
   OR the model calls the switch_model tool    ─┴─►  ModelTool's switch_model closure
                                                       akgentic-tool  model/tool.py:216-248
  ─►  BaseAgent.switch_model(key)                      THIS PACKAGE   agent.py:537-579
  ─►  ReactAgent.switch_model(key)                     akgentic-llm   agent.py:356-430
        · resolves the key against the roster, or refuses  (agent.py:329-354)
        · builds the model on the agent's EXISTING http client
        · re-checks the compaction bounds
        · model_copy(update={"model_cfg": entry}) — no rebuild, no mutation
  ─►  the NEXT run() carries model=self._model         akgentic-llm   agent.py:635
        a per-run argument; the pydantic-ai Agent is never rebuilt, so tools,
        toolsets, system prompts, history, usage counters and the HTTP
        connection pool all survive the switch untouched
  ─►  ModelTool writes ToolState.active_model = key    akgentic-tool  model/tool.py:244
        LAST, and only after the observer returned normally — which is why a
        refusal must raise rather than return a message
  ─►  the key rides AgentState's existing checkpoints into the event store
        no new event, no forced publish
  ─►  on restore, init_state() brings the slot back, and
      BaseAgent._restore_active_model() re-applies it at the top of act(),
      before the turn's context block                  THIS PACKAGE   agent.py:581-628,
                                                                      called at agent.py:736
```

Who owns which hop:

| Package | Owns |
|---|---|
| `akgentic-tool` | `ModelTool` (the card and its three capabilities), `ModelRow`, `ActiveModelState`, `ModelSwitchToolObserver`, and the `ToolState.active_model` slot |
| `akgentic-llm` | the roster on `ReactAgentConfig`, `ReactAgent.switch_model()`, `ModelSwitchError`, and the `provider:model` key grammar |
| `akgentic-agent` | the roster on `AgentConfig`, the observer implementation, the card wiring, and the restore |

The observer implementation lives **here** and only here because this is the one package that may
import both `akgentic-llm` and `akgentic-tool` — so the `ModelConfig` → `ModelRow` mapping has a
legal home and neither of those packages gains an import edge to the other
(`agent.py:500-535`).

### Enabling it: the card is opt-in

Two things are needed, and both are yours to declare: a roster, and the `ModelTool` card.

```python
from akgentic.agent import AgentConfig
from akgentic.llm import ModelConfig, PromptTemplate
from akgentic.tool.model import ModelTool

config = AgentConfig(
    name="@Manager",
    role="Manager",
    prompt=PromptTemplate(template="You are a project manager."),
    model_cfg=[
        ModelConfig(provider="openai", model="gpt-4.1", context_length=1_000_000),
        ModelConfig(provider="anthropic", model="claude-sonnet-4-5", context_length=200_000),
    ],
    tools=[ModelTool()],
)
```

**`ModelTool` is not auto-injected.** `BaseAgent` auto-adds `TeamTool` and `MailboxTool` and
nothing else (`agent.py:254-260`); an agent gets `ModelTool` only because its card list says so.
That is a decision, not an omission: granting every agent the standing power to change its own
model is a cost and governance question, and it belongs to whoever writes the card list rather
than arriving unannounced with an upgrade (ADR-018 §5).

**This is a consumer contract, and this README is where it is stated.** Nothing in `akgentic-tool`
can enforce it — `ModelTool` is an ordinary `ToolCard` and any consumer could prepend it to every
agent it builds. This package is the consumer that chooses not to.

The two commands reach a UI with no UI-specific code: they are ordinary `CommandRegistry` entries,
announced in the single `CommandsAnnouncedEvent` the agent emits at start-up
(`agent.py:290-295`) alongside every other command. No frontend work was needed to surface them.

### What the model is told: the `LLM_CONTEXT` block

`ModelTool` contributes a fifth `LLM_CONTEXT` provider
(`akgentic-tool/.../model/tool.py:164-169`), so the model in force appears in the per-turn
**Context update** block like any other volatile state — full on first delivery, a delta
afterwards (`akgentic-tool/.../model/state.py:64-72`):

```
**Active model:** openai:gpt-4.1
**Active model changed:** openai:gpt-4.1 → anthropic:claude-sonnet-4-5
```

**The block renders the roster's own `active` flag, never `ToolState.active_model`**
(`akgentic-tool/.../model/tool.py:273-279`). The slot is a persisted *preference*; rendering it
would show a stale key as though it were the model answering.

### The restore rule, and how it degrades

After `init_state()` and before the first turn, `BaseAgent` re-applies
`state.tool_state.active_model` (`agent.py:581-628`), as the **first** statement of `act()`
(`agent.py:736`) — before `_deliver_context_update()`. That order is load-bearing: reversed, the
first block after a restart would advertise the declared model while the restored one answered.

- **`active_model is None` is a no-op.** The declared active entry wins.
- **A key that no longer resolves is dropped with one `logging.WARNING`, and the declared active
  entry wins.** The restore is never fatal. An agent that refused to start because of a remembered
  choice would strand the whole team (ADR-018 §4).

#### A permanently stale key warns once per turn

The `_restored_model_key` latch is set on the **success path only** (`agent.py:628`, and
`agent.py:574` for a switch the agent made itself). A key that never resolves is therefore never
latched, and is retried at the top of every `act()` — one `WARNING` per turn, for the life of the
agent. Four things to know about it:

- **It is the design, not a defect.** The roster is mutable within a session, so latching a
  refusal would permanently and silently forfeit a key that may become valid later.
- **The cheap case** is an unknown key. `ReactAgent._resolve_roster_entry`
  (`akgentic-llm/.../agent.py:329-354`) refuses it before anything is built — a dictionary miss
  and a log line.
- **The expensive case is the one hit in production.** A key that *resolves* but whose provider
  constructor fails — a missing `OPENAI_API_KEY`, a missing `AZURE_OPENAI_ENDPOINT` — reaches a
  third-party constructor once per turn, for the life of the agent
  (`akgentic-llm/.../agent.py:405-408`).
- **The cure is clearing the persisted selection**, i.e. getting `ToolState.active_model` back to
  `None` or to a key that resolves. The in-band way is to switch to a valid key — a successful
  switch overwrites the slot. Otherwise supply the missing credential, or restore the agent from a
  state whose slot is `None`. Fixing the *roster* alone does not stop the warning if the persisted
  key is still absent from it.

### `ModelSwitchError` is the one class this layer catches

`BaseAgent.switch_model` catches `ModelSwitchError` and nothing else — never `except Exception`
(`agent.py:568-571`). One `except` is sufficient because of what that class now carries:
`akgentic-llm` translates a provider constructor's own failure into it, since pydantic-ai raises
`UserError` — a `RuntimeError`, not a `ValueError` — for a missing API key
(`akgentic-llm/.../agent.py:405-408`). That is the production case, and it would escape any
`except ValueError`.

A refusal **raises**; it is never returned as a message. `ModelTool` records
`ToolState.active_model` immediately after the observer returns normally
(`akgentic-tool/.../model/tool.py:244`), so an error string would be read as a success and would
persist a key the llm layer had just refused. The refusal reaches the model as a `RetriableError`
it can correct, and reaches a human as a dispatched string.

### Three boundaries decided upstream

These were settled in `akgentic-llm` Epic 22 and in ADR-018; they are stated here rather than
re-argued, because a reader who does not know them will re-litigate all three.

- **A mid-run switch lands on the NEXT run.** pydantic-ai binds the model once per `run()`
  (`akgentic-llm/.../agent.py:376-380`), which is why the confirmation string
  `BaseAgent.switch_model` returns says so (`agent.py:575-579`) — the model reading it is the one
  that would otherwise be surprised. The one exception the llm layer names: the auto-compaction
  gate reads `context_length` live and *does* move mid-run.
- **A switch does no history sanitization.** Provider-specific parts already in the history —
  thinking parts, reasoning items, provider-native tool payloads — are handed to the next provider
  as pydantic-ai maps them. Behaviour across a **heterogeneous** switch is therefore best-effort;
  the mitigation available today is `/compact` before switching (ADR-018 §Traps 1). This is not a
  provider-neutrality guarantee, and it is worth saying plainly rather than implying one.
- **A roster is not a fallback chain.** `fallback_models` is automatic, failure-driven and
  invisible to the model; a roster entry is chosen deliberately, by a human or by the model. The
  two compose — each roster entry may carry its own chain — and neither replaces the other
  (ADR-018 §1, §Traps 2). A roster entry that cannot be built fails **at switch time**, not at
  construction: the deliberate trade for not eagerly building every entry at start-up.

### When the active model is not in the roster

Hand-setting `model_roster=` is legal. `AgentConfig` carries the duplicate-key guard but
deliberately **not** the membership rule (`config.py:144-161`) — normalization satisfies membership
by construction on the list path, and the `ReactAgentConfig` that `on_start` builds enforces it one
layer later. So an `AgentConfig` can carry an active model that its own roster does not contain.
What follows is surprising and deliberate:

- **`active` is computed by key equality**, so an active model absent from the roster yields rows
  that are **all `False`** — nothing is synthesized and nothing raises (`agent.py:500-535`).
- **`ModelTool` then composes no `LLM_CONTEXT` block that turn**
  (`akgentic-tool/.../model/tool.py:276-279`). Designed degradation, not a bug: it would rather say
  nothing than name a model it cannot confirm.
- **An empty roster returns `[]`**, with no row synthesized for the active model. That emptiness is
  load-bearing — `ModelTool`'s own "no roster" line depends on it
  (`akgentic-tool/.../model/tool.py:48,85`): *"This agent has no model roster, so there is nothing
  to switch within."* A synthesized single row would replace that honest answer with a listing of
  one model the agent cannot switch away from.

## Run Cancellation

A running turn can be interrupted. The design is **two surfaces, one predicate, one hook**:

- **Two surfaces.** `/stop` is a `MailboxTool` command, announced to every frontend through
  the same `CommandsAnnouncedEvent` as any other command; `CancelMessage`
  (`akgentic.core.messages`) is the typed carrier for programmatic senders. Both land in the
  agent's mailbox like any other message.
- **One predicate.** `is_cancel`, defined once in `akgentic.tool.mailbox`, recognises both
  forms — nothing else in the system parses cancel vocabulary.
- **One hook.** `MailboxCapability.before_model_request` (same module), built
  **unconditionally** by `BaseAgent` — never contributed by a card, so cancellation works even
  on an agent configured without `MailboxTool`. The agent owns *both* the vocabulary and the
  enforcement, and the first is a consequence of the second: a card-less agent has no card to
  borrow a predicate from, so a predicate that shipped with the card could not make that agent
  interruptible. What the card still owns is its own surface — the id-taking `read_mailbox`
  tool and the `/stop` command registration whose string form `is_cancel` recognises without
  importing anything from the card.

A cancel arrives in one of two situations, and they are handled in completely different
places:

| | **Mid-run** — it arrives while a message is being processed | **Idle** — it arrives with nothing running |
|---|---|---|
| Who sees it | the hook, at the next step boundary | nobody, until the actor dequeues it normally |
| Mailbox | **purged at recognition** — never dequeued, never dispatched | dequeued in the ordinary way |
| Effect | `RunInterruptedError`; the run dies; `act()` tells the human | there is no run to cancel |
| What the human sees | the interruption notification — which is why nothing else speaks | an explicit answer that there is no run to cancel |

**Mid-run.** While a run is in progress, the hook peeks the mailbox before every model
request. Once a cancel is pending, it purges *every* pending cancel from the mailbox through
`consume_mailbox` — one `HandledMessage` per removal, emitted by that primitive rather than
by the hook — and only then raises `RunInterruptedError`. `act()` absorbs it: the human is
notified ("Run interrupted.") and `act()` returns a default instance of the output type the
caller named — an empty `StructuredOutput` on the team path, which `_route_output` delivers
as nothing. **No handler writes a catch**, in this package or in yours, and the handler
returns normally — **the run dies, the agent survives**. The one case a caller still sees the
error is an output type that cannot be default-constructed (a model with a required field);
`act()` then re-raises the original interruption unchanged. Because the cancel was purged, it
never gets a turn of its own: the next message the actor dequeues is ordinary mail, and the
human is told once, by the interruption.

**Idle.** A cancel that arrives with nothing running is seen by no hook, and is dequeued like
any other message. A `CancelMessage` lands on `receiveMsg_CancelMessage`, an
acknowledge-and-log no-op; a `/stop` answers through ordinary command dispatch, with an
explicit answer that there is no run to cancel. Neither needs to check whether a run is in
flight — after the purge, *reaching* either of them is the proof that none is.

**The mailbox is the cancellation's single source of truth.** There is no cancel flag and no
clear step: recognising the cancel and consuming it are one atomic act, performed by the hook
at recognition, so a cancel can never go stale and cancel the next run.

### The mid-run arrival notice

**`read_mailbox=False` turns it off completely.** The notice exists to offer the model a way to
take a waiting message on now, so without that tool it is an instruction the model cannot
follow, rendered on every step boundary of every run. `MailboxCapability` reads `read_mailbox` off
the card it is given; with reads disabled nothing is rendered and nothing is enqueued. Mail is not lost — it still arrives as its own turn, which is the fallback the design
already specifies. **Cancellation is unaffected**: the purge-and-raise runs *ahead* of the
notice and is not configurable from any card, so a `/stop` or `CancelMessage` still ends the run
— including for an agent carrying no `MailboxTool` at all.

> This is a **behaviour change** for a deployment already running `read_mailbox=False`: it loses
> the informational "1 new message arrived — finish your current work first" notice. Deliberate.
> That signal is not worth a render on every step boundary when the run cannot act on it.
>
> It is also the **only** switch: there is no per-handler setting any more. The card field that
> named handlers by dotted path is gone, replaced by the message type itself — see the offer rule
> below.

The same hook, after the cancel check, announces mail that arrived during the run: new
pending messages are announced **once**, by a **durable** notice (rendered by
`render_arrival_notice`) delivered through `ctx.enqueue(notice, priority="asap")` —
pydantic-ai's supported injection path. The auto-injected drain capability delivers it into
the model request at the next step boundary and records it in the agent's history and the
event store as its own user-role message — that record **is** the audit trail that the
doorbell rang. Announced-id tracking is run-local: `MailboxCapability.before_run` clears it
at each run start, so it dies with the run.

**The run's end is the boundary — a notice still queued there is withdrawn, not delivered.**
An `'asap'` enqueue is always one step late (the drain is mounted outermost, and `before_*`
hooks walk the chain forwards, so this step's drain has already run when the notice is
enqueued). If the model produces its final output on that same step, pydantic-ai's drain
would **discard** the run's own `End(FinalResult)` and redirect through one more model
request so the queued content is not lost — and `run_sync` then returns only that second
output, so the answer the agent had already written reaches nobody. That redirect is right
for content with no other delivery path and wrong for a doorbell: the message behind the
notice is still sitting in the actor mailbox and **arrives as its own turn** regardless. So
`MailboxCapability.after_node_run` withdraws the notice once the run has reached its end,
the `End` survives, and the turn returns the answer it produced. It is possible only on that
hook — `after_*` walks the capability chain **backwards**, putting this capability ahead of
the outermost drain.

The withdrawal is narrow, and deliberately so. It is keyed on the `enqueue_id` that
`ctx.enqueue` returned, never on the notice's text, and it covers only the ids this hook
recorded: content enqueued by any other producer keeps the redirect, and the rendering
`after_tool_execute` injects for an **absorbed** message is never withdrawn — that message
has already been consumed from the mailbox, so the queue is the only thing still holding it.

**Every announced message is listed; only some carry an id.** A message this run can take on
renders as its own `rendering_preview()` followed by `(id: …)` — the only way the model can name
it. Everything else renders as the fixed line `- Message cannot be handled in the run`, with no
id and no content. That missing id *is* the constraint: such a message is **visible but
unaskable** — not rejected, not validated, not refused. The affordance simply is not offered.

A message is offered an id only when **all three** of these hold:

1. **It is a `MailboxMessage`** (`akgentic.tool.mailbox`). Extending that base is how a class
   declares it can travel through a mailbox, and it owes both `rendering()` and
   `rendering_preview()` — either one left unanswered raises. A class that renders but should
   never be absorbed mid-run simply does not extend it; `TriageMessage` in `custom_agent.py` is
   the worked case, renderable by `act()` and never offered.
2. **Its class is exactly the class of the message being handled.** Same class means same
   handler means same output type, so an absorbed message is answered in the shape its own
   handler would have produced. An exact class check, not `isinstance`.
3. **It is not a cancel.** A `/stop` arrives as an ordinary `AgentMessage` and does have a
   preview, so offering its id would let the model read its way out of being cancelled.

> There used to be a fourth condition — a `mailbox_preview_handlers` list on the card, naming
> handlers by dotted path. It is gone and nothing replaced it: 1 and 2 already decide exactly what
> it decided, and it cost a dotted-path resolution, four `ValueError` shapes and a typo that
> surfaced only at agent init.

The closing line follows the same rule. It points at `read_mailbox` "with one of the ids above"
only when at least one id is on offer; otherwise it says only "Finish your current work first —
you will get them just after", because promising a read for a listing that carries no id would
be an instruction the model cannot follow. When it does point, it also gives the one reason that
decides the timing: a message that may add to or change the work in flight is worth taking on
before that work is finished, and is worth nothing after. The card-level gate above is that same
principle one step earlier: a notice that cannot name an id is degraded, and a notice whose
*tool* does not exist is not rendered at all.

**Naming an id absorbs that one message.** `read_mailbox` takes the id and acknowledges it;
`MailboxCapability.after_tool_execute` consumes exactly the message named and enqueues that
message's own `rendering()` at `"asap"`, so its content arrives as its own injected turn
rather than as a tool result. An absent or unknown id is a silent no-op. Whatever the model
leaves unnamed stays queued and arrives as its own turn once the run ends, and a cancel is
never offered and never absorbed.

**The injected turn is prefixed with `ABSORBED_PREFIX`, and that prefix is load-bearing.**
(`ABSORBED_PREFIX` and `ARRIVAL_CLOSING` are both exported from `akgentic.tool.mailbox`, so a
caller overriding one can build on the shipped wording rather than replace it blind.)
`rendering()` renders a message the way its *own handler* would receive it — imperative and
self-contained ("You received a request from @X. A reply is expected."). Injected mid-run that
reads as a **new assignment**, and the model answers it *instead of* what it was already doing.
Observed in the field: an agent that had just finished a report answered only the newer question,
and the report answer reached nobody. The prefix says the work is *additional* and that it does
not replace the current request.

**How many answers are owed, it does not assert — it asks.** A mid-run arrival is either a
separate request (two answers, one message each in the output's `messages` list) or an addition
to, or correction of, the request already in flight, where one message answers both. The
capability cannot tell which: it has not read the message, and a classification made in code
would be invisible and unrecoverable, where one made by the model is right there in the output.
So the prefix states both cases and names a default for the doubtful ones — **answer
separately** — because the two failure modes are not symmetric: a redundant second message is
noise, a swallowed report reaches nobody.

The prefix belongs to the capability, not the message: **rendering a message is the message's
job, delivering one is the capability's**, and framing a delivery is part of delivering it — so
every class that grows a `rendering()` inherits it for free.

**The card is what gets injected, and the wording is not on it.**
`MailboxCapability(observer=self, card=mailbox_card)` is the whole of the wiring:
`BaseAgent._assemble_capabilities` hands the card over and inspects none of it, and the capability
reads `read_mailbox` off it itself to decide whether the doorbell rings. Which fields the mailbox
needs is knowledge the *consumer* holds — the agent does not repeat it.

The two injected strings are **`MailboxCapability`'s own**, keyword-only constructor parameters
defaulting to `ABSORBED_PREFIX` and `ARRIVAL_CLOSING`. They lived on the card for one release and
came back off it: the catalog persists a card with a plain `model_dump(mode="json")` and no
`exclude_defaults`, so a literal field default was written into every stored entry and each team
froze a private copy of prose that is expected to keep improving. `BaseAgent` passes neither
parameter, so every deployment runs the shipped wording and improving a sentence reaches every
existing team on upgrade rather than only teams created afterwards. Overriding is still possible,
in process, by a caller constructing the capability directly.

The card is **required**, so a card published before these fields existed fails loudly rather than
quietly running text nobody can see in the catalog; `akgentic-agent` depends on an `akgentic-tool`
release carrying them. The one value that does not reach the model is the empty string: `""` falls
back to the module constant, because a prefix set empty is a configuration mistake whose failure
mode is a mid-run injection with no framing at all. That is `or`, not `is None`: an empty string is
a configuration mistake rather than a choice.

Two deliberate asymmetries a reader will otherwise re-litigate. First, the closing line reaches
`render_arrival_notice` as a *function parameter* while the prefix arrives at construction: the
renderer is a module-level function, which constructor injection cannot reach. Second, the
id-less closing is **not** configurable and takes no parameter — a listing carrying no id may not
promise a read whatever a deployment sets. Cancellation consults neither string: the
purge-and-raise runs above the notice, so a capability built with empty text for both is still
interruptible.

### Honest limitations

- **An interruption is a clean end, not a failure.** It never routes through the failure
  path: no `ErrorMessage` is emitted, and the handler returns normally. (An exception
  escaping a handler would not stop the actor either — `Akgent._handle_failure` in
  `akgentic-core` keeps the actor loop alive and emits an `ErrorMessage` to the
  orchestrator; actor death on failure is stock-pykka behaviour only. The catch site's
  invariant is stronger than survival: the failure path is never entered at all.)
- **Granularity is the next step boundary, never mid-stream.** The hook fires before every
  model request inside the REACT loop — bracketing every tool call and reasoning step — but
  it does not abort an in-flight provider stream: a single very long model response is
  uninterruptible from inside. A tool-free single completion has no step boundary at all, so
  it can neither be cancelled nor see mid-run mail — accepted, since that run is ending
  anyway.
- **A cancel pending during a run-tier-breach conclusion escalates the breach.** The
  tool-free conclusion is `akgentic-llm`'s now (see
  [What happens when a limit is hit](#what-happens-when-a-limit-is-hit)), but the outcome is
  unchanged: its blanket `except` treats a pending cancel as a failed conclusion and re-raises
  the **original** breach rather than reading it as an interruption. The guard on `act()` then
  notifies. A safe, known, accepted outcome — the turn ends either way — and the cancel is
  still purged before the raise, so it does not survive to be dispatched afterwards.

## Examples

```bash
cd packages/akgentic-agent
uv run python examples/simple_team.py
```

| Script | Topic |
|---|---|
| `simple_team.py` | Three-role interactive team with search, workspace, planning, `/commands`, and `/usage` for per-agent cost reporting via `EventSubscriber` |

See the [Examples README](https://github.com/b12consulting/akgentic-agent/blob/master/examples/README.md) for full descriptions and running instructions.

## Documentation

- [Agent Collaboration System](https://github.com/b12consulting/akgentic-agent/blob/master/docs/agent-collaboration.md) — Collaboration model,
  routing mechanics, delegation patterns, and typed protocol walkthrough

## Development

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

From this repository's root — `akgentic-core`, `akgentic-llm` and `akgentic-tool`
resolve from PyPI under the floors in `pyproject.toml`. This is what CI does:

```bash
uv venv
uv pip install -e ".[dev]"
```

To exercise this package against unreleased sibling code, work from the
[akgentic-framework](https://github.com/b12consulting/akgentic-framework) bundle
in source mode instead — see
[Working on the package itself](#working-on-the-package-itself).

### Commands

From this repository's root:

```bash
# Run tests
uv run pytest tests/

# Run tests with coverage
uv run pytest tests/ --cov=akgentic.agent --cov-fail-under=80

# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/

# Type check
uv run mypy src/
```

`addopts = "-m 'not integration'"` deselects the integration tests by default: they make real LLM
calls (gated by `OPENAI_API_KEY`) and poll for actor quiescence. Run them explicitly with
`uv run pytest tests/ -m integration`.

### CI Pipeline

The package uses GitHub Actions for continuous integration. On every push, on pull requests
against `master`, and on manual dispatch, the pipeline:

1. Checks out **this repository only** — no workspace, no submodules
2. Installs uv and Python 3.12, and creates a virtualenv
3. Installs the package and its dev extra with `uv pip install -e ".[dev]"`, so the sibling
   akgentic packages come from PyPI at their declared floors
4. Runs **mypy** on `src/` (strict type checking)
5. Runs **ruff check** on `src/`
6. Runs **pytest** on `tests/` with coverage over `akgentic.agent` (`--cov-fail-under=80`)
7. Updates the coverage badge gist — only on `master` pushes

Because step 3 resolves the siblings from PyPI, a change that depends on an unreleased
`akgentic-core`/`llm`/`tool` commit will be red here until that package ships, even when the
workspace is green locally. That is a merge-order signal, not a defect in this package.

**The id-based mailbox read is exactly that case, and it is worth stating plainly.** The two
halves live in two packages — `read_mailbox(message_id)` in `akgentic-tool`, the absorption and
injection in this one — and **neither may be released alone**. A tool that takes an id with no
agent-side injection acknowledges and delivers nothing; an injection with no id-taking signature
never receives an id. The order is `akgentic-tool` first, then this package **with its
`akgentic-tool` floor raised** — a floor raise that is owed and deliberately not yet made, because
the version to pin does not exist yet. In the gap the tool has shipped and this package has not, so
an agent still running the older code announces mail the old way and tells the model to call
`read_mailbox` with no id at all; the new signature requires one, rejects the call, and the model
burns retries on it: **inert but noisy**. Nothing is lost — the new tool consumes nothing, so unread
mail stays queued and arrives as its own turn — and the noise stops of its own accord once this
package's release lands on the raised floor.

> **Note:** No pre-commit hooks are configured in this package. Quality checks run
> exclusively in CI.

### Project Structure

```
src/akgentic/agent/
    __init__.py          # Public API (__all__): __version__, AgentConfig, HumanProxy,
                         #   BaseAgent, RunInterruptedError, AgentMessage, LlmRenderable
    agent.py             # BaseAgent — actor + LLM + tool composition, routing logic
    config.py            # AgentConfig, AgentState
    custom_agent.py      # Worked example: a second agent class with its own schema
    human_proxy.py       # HumanProxy — human-in-the-loop bridge
    messages.py          # AgentMessage with its typed protocol and its two renderings;
                         #   LlmRenderable — the Protocol act() accepts
    output_models.py     # StructuredOutput, Request, REPLY_PROTOCOLS
    usage_limits.py      # guard_usage_limits decorator + escalation (no agent.py import)
    utils.py             # resolve_recipient — the team addressing convention
examples/                # Runnable examples with README
tests/                   # Tests organised by module
docs/
    agent-collaboration.md
```

## License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](https://github.com/b12consulting/akgentic-agent/blob/master/LICENSE).

> **Dual licensing & CLA** — Akgentic is available under the AGPL-3.0 open-source license. A commercial license is also planned for organizations that require alternative terms. Contact [Yuma](https://www.weareyuma.com/en/contact) for more information. External contributions will be accepted once a Contributor License Agreement (CLA) is in place. Until then, please hold off on submitting pull requests.
