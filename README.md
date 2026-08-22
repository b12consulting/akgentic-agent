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
                    receiveMsg_AgentMessage()   ← @guard_usage_limits(...)
                             │  prepend reply protocol to the prompt:
                             │  "You received a request from @Human. A reply is
                             │   expected: respond to @Human with the result."
                             │
                    act(prompt, StructuredOutput)
                      │
                      ├─ append **Context update N** block (if shared state changed)
                      ├─ expand !!glob_pattern refs (if WorkspaceTool present)
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
                         The receiving agent prepends its own reply
                         protocol when it runs the loop above.
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

2. **When receiving** — `receiveMsg_AgentMessage()` prepends a one-line reply protocol, keyed on
   the incoming `type` via `REPLY_PROTOCOLS`, to the raw content before handing it to the LLM.
   The guidance is therefore always the one matching the intent *that* agent received, and it
   reaches the LLM through the **prompt** — not through the output schema.

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

### Static Schema + Prompt-Carried Reply Protocol

`act()` forwards the `output_type` it was handed straight to the REACT loop — there is no
per-call subclass and no `type()` metaprogramming on the hot path:

```python
output = self._react_agent.run_sync(prompt, deps=self, output_type=output_type)
```

`receiveMsg_AgentMessage()` calls `act(prefixed_content, StructuredOutput)`, so the team
delegation path reasons against the **static** `StructuredOutput` type.

`Request.recipient` is a **plain string** with no `enum` constraint. Recipient validity is
enforced at **routing time** in `_route_output()`, not in the schema:

| Recipient format | Resolution |
|---|---|
| `@MemberName` | `get_team_member(name)` → direct send (skipped if not found) |
| `RoleName` | `hire_member(role)` → create actor → send |

The reply-protocol guidance lives where the LLM actually reads it — the **prompt**.
`receiveMsg_AgentMessage()` prepends a one-line protocol (keyed on the incoming message
type via `REPLY_PROTOCOLS`) to the raw content before handing it to the LLM:

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
the reply-protocol prefix is added by the *receiving* agent's `receiveMsg_AgentMessage()`,
so the guidance is always keyed to the intent that agent actually received:

```python
# In the receiver's receiveMsg_AgentMessage(), before the LLM turn:
prompt = f"You received a request from @Manager. {reply_protocol}\n\n{message.content}"
```

`_route_output()` returns **whether anything was actually delivered**. That is what lets the
usage-limit guard tell a real conclusion from one that resolved no recipient and sent
nothing — it cannot inspect the output itself, since the schema belongs to the caller.

A usage-limit breach is handled by **tier**, by the `@guard_usage_limits` decorator on the
handler rather than by any code inside it. A run-tier breach — this turn ran out of its own
budget — first gets one tool-free conclusion attempt, whose answer is delivered through this
same `_route_output()` path, which is why a concluded answer is routed exactly like any other
message. An agent-tier breach (the agent's lifetime budget is spent), or a conclusion attempt
that produces no answer, escalates via `notify_human()` to the team's user-proxy member —
found structurally through `ActorAddress.is_user_proxy`, so any role string works; when the
team has none, the notice is logged and dropped. See
[What happens when a limit is hit](#what-happens-when-a-limit-is-hit) and
[Writing a second agent class](#writing-a-second-agent-class).

### Writing a second agent class

`BaseAgent` handles one message type against one schema. A subclass that wants its own — its
own structured output, its own `receiveMsg_*` — needs the usage-limit policy too, and must
**not** copy it. The `except` ordering is load-bearing: `RunUsageLimitError` and
`AgentUsageLimitError` both subclass `UsageLimitError`, so a base clause placed first catches
both and the tier branch never runs. A wrong copy still compiles, still passes an ordinary
test, and simply stops concluding on a run-tier breach.

Two modules exist for exactly this, and neither imports `agent.py` — they are what a *new*
agent class needs, so a dependency in that direction would make them unusable from the module
that defines the base class. What each needs from an agent is stated as a `Protocol`:

| module | holds |
|---|---|
| `usage_limits.py` | `AgentLike`, `guard_usage_limits`, `escalate_usage_limit`, `try_conclude_without_tools` |
| `utils.py` | `TeamResolver`, `resolve_recipient` — the team addressing convention (`@member` vs role to hire) |

Apply the decorator to every `receiveMsg_*` that can reach the LLM, handing it *your* schema
and *your* router:

```python
from akgentic.agent import RunInterruptedError
from akgentic.agent.usage_limits import guard_usage_limits
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

    @guard_usage_limits(output_type=TriageOutput, route=_route_triage)
    def receiveMsg_TriageMessage(self, message: TriageMessage, sender: ActorAddress) -> None:
        try:
            output = self.act(prompt, TriageOutput)
        except RunInterruptedError:
            self.notify_human("Run interrupted.")
            return
        self._route_triage(output)
```

Four things follow, and they are the whole point:

- **The handler carries no usage-limit handling of its own.** The tier policy is entirely the
  decorator's; the body reads as just the work.
- **A run-tier breach concludes in `TriageOutput`**, delivered by `_route_triage` — because
  the schema and the routing are decorator *arguments*, not overrides. `CustomAgent` overrides
  nothing.
- **The router returns `bool`.** The guard is handed a schema it cannot inspect — `TriageOutput`
  has no `.messages` — so "did anything actually go out?" is the router's answer to give, and
  it is what separates a real conclusion from one that routed nothing.
- **The one `except` the handler does carry is the `RunInterruptedError` catch around
  `act()`.** The cancel capability is unconditional on every `BaseAgent` subclass, so every
  run is interruptible — but the catch is not inherited: every subclass handler that calls
  `act()` must supply it itself, exactly as `receiveMsg_AgentMessage` does (notify the human,
  route nothing, return). An escape ends the turn through the actor failure path instead of
  the designed clean end (see [Run Cancellation](#run-cancellation)).

`_route_triage` is defined **before** the handler: the decorator names it as an argument, which
is evaluated while the class body runs. The requester is lifted off the handler's own message
argument, so nothing is threaded through the signature.

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

The 5-type intent protocol controls conversation flow. When an incoming message is
received, the matching `REPLY_PROTOCOLS` instruction is prepended to the **user prompt**
(not the output schema), so the LLM reads the guidance inline with the content:

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
| `model_cfg` | `ModelConfig` | `ModelConfig()` | LLM provider, model name, API settings |
| `runtime_cfg` | `RuntimeConfig` | `RuntimeConfig()` | Retries, tool-call end strategy, parallel tools, HTTP client settings |
| `run_usage_limits` | `RunUsageLimits` | `RunUsageLimits()` | Budget for **one** `run()` — token and request caps that reset every run |
| `agent_usage_limits` | `AgentUsageLimits` | `AgentUsageLimits()` | Budget for the agent's **whole lifetime** — runs and tokens, accumulated across every run |
| `compaction_cfg` | `CompactionConfig` | `CompactionConfig()` | Context-compaction strategy and auto-trigger (opt-in; off unless `model_cfg.context_length` is set) |
| `tools` | `list[ToolCard]` | `[]` | Tool cards; `TeamTool` and `MailboxTool` are always prepended automatically |

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

The `@guard_usage_limits` decorator reacts differently to each tier, on behalf of whichever
handler carries it. The exception class decides; the message text is never parsed.

| tier | class raised | reaction | human notified? |
|---|---|---|---|
| run | `RunUsageLimitError` | one **tool-free conclusion** attempt — the agent is asked to answer the requester with what it has already gathered, delivered through the ordinary routing path | no, when the attempt produces an answer |
| agent | `AgentUsageLimitError` | terminal and unchanged: `notify_human()`, then `WarningError` | yes |

A run-tier breach means *this turn* ran out of requests, tool calls or tokens. The agent
itself usually still has lifetime budget, and by that point usually has most of what it was
asked for: it can no longer call a tool, but it can still answer — so it is asked to, once.
The prompt for that final call tells the model it has no tools left, to answer the named
requester now, and to **state explicitly which parts it could not check or finish**. An
answer produced this way is expected to be incomplete and to say so; read it as a partial
result, not a finished one. Nothing extra is written to the agent's own context: the
conclusion's own exchange lands in the history like any other turn, and the prompt above
already says the turn was cut short.

That attempt is exactly one attempt, and it can fail. It is skipped altogether when the
incoming message carried no sender — there is nobody to answer — and it falls through to the
escalation above when the call raises (including on a second breach) or returns no message at
all. In each of those cases the human is notified and `WarningError` is raised, reporting the
**original** run-tier breach rather than any secondary failure. So a run-tier breach
*attempts* a conclusion; it does not guarantee a reply — a `@recipient` the model names but
the team does not have is skipped at routing time, as always, and an answer addressed that way
still counts as a conclusion, so no one is notified about it either. **The human is therefore
notified only when the lifetime budget is spent, or when the conclusion attempt failed.**

There is no retry counter, and none is needed *where a lifetime budget is set*: the agent tier
is consumed before every call, the conclusion call included, so an agent that keeps breaching
the run tier walks into its terminal tier by construction. That bound is only as real as the
budget behind it — the default `AgentUsageLimits()` is all-`None` and never blocks, so an agent
left on the defaults breaches, concludes and breaches again with nothing to stop it and no one
told. Set `agent_usage_limits` if you want that backstop.

**Who owns what.** `UsageLimitError`, `RunUsageLimitError`, `AgentUsageLimitError` and the
conclusion mechanism (`ReactAgent.conclude_without_tools()` and its `_sync` bridge) belong to
**`akgentic-llm`** — this package imports them and never redefines them. *Which tier gets
which reaction*, the fall-through cases and the routing of the answer belong to
**`akgentic-agent`**, in `usage_limits.py` — not in any agent class. Change enforcement in
`akgentic-llm`; change policy there. To apply that policy to a handler of your own, see
[Writing a second agent class](#writing-a-second-agent-class).

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
| `tool_state` | `ToolState` | The tool layer's persistent per-agent slot — context-update baselines and the block counter. A **cache, never a record**: the message history is the record, so a lost or stale slot can only cost a full-snapshot re-send, never a lost update. See [Context updates](#context-updates) |

## Tool Channels

`ToolFactory` organises tool cards into four channels:

| Channel | Consumer | Examples |
|---|---|---|
| `TOOL_CALL` | LLM via pydantic-ai tools | `hire_members()`, `fire_members()`, `read_mailbox()`, `web_search()`, `workspace_read()` |
| `SYSTEM_PROMPT` | LLM system prompt — rendered into the frozen system block | backstory, current date |
| `LLM_CONTEXT` | LLM via a per-turn appended **Context update** block | team roster, role profiles, planning summary, knowledge-graph summary, mailbox status |
| `COMMAND` | `CommandRegistry` — in-agent Python and `/`-prefixed messages | `hire_member`, `fire_member`, `team_members`, `team_roles`, `planning_summary`, `stop` |

`TeamTool` **and** `MailboxTool` are always prepended to `config.tools` if not already
present, so every `BaseAgent` can hire and fire members (`TeamTool`) and carries the
mailbox surfaces — mailbox status as context state, the `read_mailbox` peek tool, and
`/stop` (`MailboxTool`). A card already supplied in `config.tools` wins over the prepended
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
is the worked example: three capabilities, each riding its own channel — `MailboxState` on
`LLM_CONTEXT`, `read_mailbox` on `TOOL_CALL`, `/stop` on `COMMAND` — and the agent gained
all three without a single new method. When the next feature is a capability the LLM, the
operator, or the context should see, write it as a card and let the table above route each
piece to the hook that serves it. The card-author side of this contract is the
`akgentic-tool` README's *Building a feature as a card* authoring guide.

### Context updates

Volatile, team-shared state — the roster, role profiles, planning, the knowledge-graph summary,
the mailbox status — never enters the system prompt: the system block holds the backstory and the current date only, and
stays byte-identical run to run so the prompt-cache prefix survives. Instead, before each run the
agent appends **at most one block** at the tail of the conversation carrying what changed since the
last block it delivered.

The engine that composes that block is not in this package. `akgentic.tool.core.ContextUpdater` —
built once by `ToolFactory.get_context_updater()` at `on_start()` and held for the agent's lifetime
— reads the state providers, diffs them against the baselines, composes the block and advances the
counter; `akgentic-tool` owns those semantics along with the cards that produce the state.
`BaseAgent` contributes only *when* — one delivery site, at the top of `act()`, before the run — and
*how* — the append goes through `ContextManager.record_operator_action`, so a fresh agent's first
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
| `stop()` | `MailboxTool` | Cancel the current run; dispatched while idle it only replies that nothing is running — the mid-run effect is the cancel hook's (see [Run Cancellation](#run-cancellation)) |
| `compact()` / `clear()` | `BaseAgent` built-ins | Compact or clear the conversation context |

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
`act()` expands inline file references before the LLM call:

```
!!file.png               → BinaryContent injected into the prompt
!!"my screenshot.png"    → same, for paths with spaces
!!*.png                  → glob — every matching image, sorted by path
!!report.pdf             → "!!report.pdf[=> Use workspace_read tool]" forwarded to the LLM
!!nonexistent.png        → "!!nonexistent.png[Error: no image found in the workspace]"
```

Expansion happens in `act()` before `run_sync()`, and only when the expansion actually changed
something: if the command returns the prompt unchanged, the plain string is sent as-is. Errors and
document hints are forwarded to the LLM rather than silently dropped. Agents whose registry has no
`_expand_media_refs` are unaffected — the block is a no-op.

## Run Cancellation

A running turn can be interrupted. The design is **two surfaces, one predicate, one hook**:

- **Two surfaces.** `/stop` is a `MailboxTool` command, announced to every frontend through
  the same `CommandsAnnouncedEvent` as any other command; `CancelMessage`
  (`akgentic.core.messages`) is the typed carrier for programmatic senders. Both land in the
  agent's mailbox like any other message.
- **One predicate.** `is_cancel`, defined once in `akgentic.tool.mailbox`, recognises both
  forms — nothing else in the system parses cancel vocabulary.
- **One hook.** `MailboxCancelCapability.before_model_request`, built **unconditionally** by
  `BaseAgent` — never contributed by a card, so cancellation works even on an agent
  configured without `MailboxTool`. The card owns the *vocabulary*; the agent owns the
  *enforcement*.

The flow: while a run is in progress, the hook peeks the mailbox before every model request
and raises `RunInterruptedError` at the next step boundary once a cancel is pending.
`receiveMsg_AgentMessage` catches it around `act()`: the human is notified
("Run interrupted."), nothing is routed, and the handler returns normally — **the run dies,
the agent survives**. The actor loop then dequeues the `/stop` itself, which answers through
ordinary command dispatch (its reply: nothing is running any more). A `CancelMessage`
dequeued while the agent is idle lands on `receiveMsg_CancelMessage`, an
acknowledge-and-log no-op — by that point there is nothing to cancel.

**The mailbox is the cancellation's single source of truth.** There is no cancel flag and no
clear step: recognising the cancel (the hook's peek) and consuming it (the actor loop's
dequeue) are the same message leaving the same queue, so a cancel can never go stale and
cancel the next run.

### The mid-run arrival notice

The same hook, after the cancel check, announces mail that arrived during the run: new
pending messages are announced **once**, by a **durable** notice (rendered by
`render_arrival_notice`) delivered through `ctx.enqueue(notice, priority="asap")` —
pydantic-ai's supported injection path. The auto-injected drain capability delivers it into
the model request at the next step boundary and records it in the agent's history and the
event store as its own user-role message — that record **is** the audit trail that the
doorbell rang. When the run would otherwise end at that boundary, pydantic-ai's drain
redirects through one final model request so an already-enqueued notice is delivered rather
than lost — an occasional extra model call, by design. Announced-id tracking is run-local:
`act()` resets it at each run start, so it dies with the run. The durable record exists
regardless: the `read_mailbox` tool return if the model chooses to look, and the message's
own turn either way — every pending message is still delivered as its own turn after the
run ends.

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
- **A cancel pending during a run-tier-breach conclusion escalates the breach.** When the
  `@guard_usage_limits` tool-free conclusion is running (see
  [What happens when a limit is hit](#what-happens-when-a-limit-is-hit)), a pending cancel
  is caught by `try_conclude_without_tools`'s blanket `except` and **escalates the original
  breach** rather than reading as an interruption — a safe, known, accepted outcome: the
  turn ends either way, and the queued `/stop` still answers through dispatch.

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

> **Note:** No pre-commit hooks are configured in this package. Quality checks run
> exclusively in CI.

### Project Structure

```
src/akgentic/agent/
    __init__.py          # Public API: BaseAgent, AgentConfig, HumanProxy, AgentMessage
    agent.py             # BaseAgent — actor + LLM + tool composition, routing logic
    config.py            # AgentConfig, AgentState
    custom_agent.py      # Worked example: a second agent class with its own schema
    human_proxy.py       # HumanProxy — human-in-the-loop bridge
    messages.py          # AgentMessage with typed protocol
    output_models.py     # StructuredOutput, Request, REPLY_PROTOCOLS
    usage_limits.py      # guard_usage_limits decorator + the tier policy (no agent.py import)
    utils.py             # resolve_recipient — the team addressing convention
examples/                # Runnable examples with README
tests/                   # Tests organised by module
docs/
    agent-collaboration.md
```

## License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](https://github.com/b12consulting/akgentic-agent/blob/master/LICENSE).

> **Dual licensing & CLA** — Akgentic is available under the AGPL-3.0 open-source license. A commercial license is also planned for organizations that require alternative terms. Contact [Yuma](https://www.weareyuma.com/en/contact) for more information. External contributions will be accepted once a Contributor License Agreement (CLA) is in place. Until then, please hold off on submitting pull requests.
