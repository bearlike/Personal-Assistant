#!/usr/bin/env python3

# Standard library modules
import os
import warnings
from collections.abc import Callable
from typing import cast

from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser

# Third-party modules
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core._api.beta_decorator import LangChainBetaWarning
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.ai import AIMessage
from langchain_openai import ChatOpenAI
from langfuse.callback import CallbackHandler

# User-defined modules
from core.classes import ActionStep, OrchestrationState, TaskQueue, get_task_master_examples
from core.common import (
    get_logger,
    get_mock_speaker,
    get_system_prompt,
    get_unique_timestamp,
    num_tokens_from_string,
)
from core.compaction import should_compact, summarize_events
from core.hooks import HookManager, default_hook_manager
from core.permissions import (
    PermissionDecision,
    PermissionPolicy,
    approval_callback_from_env,
    load_permission_policy,
)
from core.session_store import SessionStore
from core.token_budget import get_token_budget
from core.tool_registry import ToolRegistry, load_registry
from core.types import ActionStepPayload, Event

logging = get_logger(name="core.task_master")

# C0116,
# Filter out LangChainBetaWarning specifically
warnings.simplefilter("ignore", LangChainBetaWarning)
load_dotenv()


def _build_direct_response(message: str) -> TaskQueue:
    """Create a TaskQueue with a direct talk-to-user response."""
    step = ActionStep(
        action_consumer="talk_to_user_tool",
        action_type="set",
        action_argument=message,
    )
    task_queue = TaskQueue(action_steps=[step])
    MockSpeaker = get_mock_speaker()
    step.result = MockSpeaker(content=message)
    task_queue.task_result = message
    return task_queue


def _augment_system_prompt(
    system_prompt: str,
    tool_registry: ToolRegistry | None,
    session_summary: str | None = None,
) -> str:
    sections = [system_prompt]
    if session_summary:
        sections.append(f"Session summary:\n{session_summary}")
    if tool_registry is not None:
        catalog = tool_registry.tool_catalog()
        if catalog:
            tool_lines = "\n".join(
                f"- {tool['tool_id']}: {tool['description']}" for tool in catalog
            )
            sections.append(f"Available tools:\n{tool_lines}")
    return "\n\n".join(sections)


def generate_action_plan(
    user_query: str,
    model_name: str | None = None,
    tool_registry: ToolRegistry | None = None,
    session_summary: str | None = None,
) -> TaskQueue:
    """
    Use the LangChain pipeline to generate an action plan based on the user query.

    Args:
        user_query (str): The user query to generate the action plan.

    Returns:
        List[dict]: The generated action plan as a list of dictionaries.
    """
    if tool_registry is None:
        tool_registry = load_registry()

    user_id = "meeseeks-task-master"
    session_id = f"action-queue-id-{get_unique_timestamp()}"
    trace_name = user_id
    version = os.getenv("VERSION", "Not Specified")
    release = os.getenv("ENVMODE", "Not Specified")

    langfuse_handler = CallbackHandler(
        user_id=user_id,
        session_id=session_id,
        trace_name=trace_name,
        version=version,
        release=release
    )

    model_name = cast(
        str,
        model_name
        or os.getenv("ACTION_PLAN_MODEL")
        or os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo"),
    )

    model = ChatOpenAI(
        openai_api_base=os.getenv("OPENAI_API_BASE"),  # type: ignore[call-arg]
        model=model_name,
        temperature=0.4
    )

    parser = PydanticOutputParser(pydantic_object=TaskQueue)  # type: ignore[type-var]
    logging.debug(
        "Generating action plan <model='%s'; user_query='%s'>", model_name, user_query)

    prompt = ChatPromptTemplate(
        messages=[
            SystemMessage(
                content=_augment_system_prompt(
                    get_system_prompt(),
                    tool_registry,
                    session_summary=session_summary,
                )
            ),
            HumanMessage(content="Turn on strip lights and heater."),
            AIMessage(content=get_task_master_examples(example_id=0)),
            HumanMessage(content="What is the weather today?"),
            AIMessage(content=get_task_master_examples(example_id=1)),
            HumanMessagePromptTemplate.from_template(
                
                    "## Format Instructions\n{format_instructions}\n"
                    "## Generate a task queue for the user query\n{user_query}"
                
            ),
        ],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        },
        input_variables=["user_query"]
    )

    estimator = num_tokens_from_string(str(prompt))
    logging.info("Input Prompt Token length is `%s`.", estimator)

    action_plan = (prompt | model | parser).invoke(
        {"user_query": user_query.strip()},
        config={"callbacks": [langfuse_handler]}
    )

    action_plan.human_message = user_query
    logging.info("Action plan generated <%s>", action_plan)
    return action_plan


def run_action_plan(
    task_queue: TaskQueue,
    tool_registry: ToolRegistry | None = None,
    event_logger: Callable[[Event], None] | None = None,
    permission_policy: PermissionPolicy | None = None,
    approval_callback: Callable[[ActionStep], bool] | None = None,
    hook_manager: HookManager | None = None,
) -> TaskQueue:
    """
    Run the generated action plan.

    Args:
        task_queue (TaskQueue): The action plan to run.

    Returns:
        TaskQueue: The updated action plan after running.
    """
    if tool_registry is None:
        tool_registry = load_registry()
    if permission_policy is None:
        permission_policy = load_permission_policy()
    if approval_callback is None:
        approval_callback = approval_callback_from_env()
    if hook_manager is None:
        hook_manager = default_hook_manager()

    results: list[str] = []

    for idx, action_step in enumerate(task_queue.action_steps):
        logging.debug("Processing ActionStep: %s", action_step)
        decision = permission_policy.decide(action_step)
        decision = hook_manager.run_permission_request(action_step, decision)
        decision_logged = False
        if decision == PermissionDecision.ASK:
            approved = approval_callback(action_step) if approval_callback else False
            decision = PermissionDecision.ALLOW if approved else PermissionDecision.DENY
            if event_logger is not None:
                event_logger(
                    {
                        "type": "permission",
                        "payload": {
                            "action_consumer": action_step.action_consumer,
                            "action_type": action_step.action_type,
                            "action_argument": action_step.action_argument,
                            "decision": decision.value,
                        },
                    }
                )
                decision_logged = True
        if decision == PermissionDecision.DENY:
            MockSpeaker = get_mock_speaker()
            message = (
                "Permission denied for "
                f"{action_step.action_consumer}:{action_step.action_type}."
            )
            action_step.result = MockSpeaker(content=message)
            results.append(message)
            if event_logger is not None and not decision_logged:
                event_logger(
                    {
                        "type": "permission",
                        "payload": {
                            "action_consumer": action_step.action_consumer,
                            "action_type": action_step.action_type,
                            "action_argument": action_step.action_argument,
                            "decision": decision.value,
                        },
                    }
                )
            continue

        action_step = hook_manager.run_pre_tool_use(action_step)
        task_queue.action_steps[idx] = action_step
        tool = tool_registry.get(action_step.action_consumer)

        if tool is None:
            logging.error(
                "No tool found for consumer: %s", action_step.action_consumer
            )
            continue

        try:
            action_result = tool.run(action_step)
            action_result = hook_manager.run_post_tool_use(action_step, action_result)
            action_step.result = action_result
            content = getattr(action_result, "content", None)
            if content is None:
                content = "" if action_result is None else str(action_result)
            results.append(content)
            if event_logger is not None:
                event_logger(
                    {
                        "type": "tool_result",
                        "payload": {
                            "action_consumer": action_step.action_consumer,
                            "action_type": action_step.action_type,
                            "action_argument": action_step.action_argument,
                            "result": content,
                        },
                    }
                )
        except Exception as e:
            logging.error(f"Error processing action step: {e}")
            action_step.result = None
            if event_logger is not None:
                event_logger(
                    {
                        "type": "tool_result",
                        "payload": {
                            "action_consumer": action_step.action_consumer,
                            "action_type": action_step.action_type,
                            "action_argument": action_step.action_argument,
                            "result": None,
                            "error": str(e),
                        },
                    }
                )

    task_queue.task_result = " ".join(results).strip()

    return task_queue


def _action_steps_complete(task_queue: TaskQueue) -> bool:
    return all(step.result is not None for step in task_queue.action_steps)


def _maybe_auto_compact(
    session_store: SessionStore,
    session_id: str,
    model_name: str | None,
    hook_manager: HookManager,
) -> str | None:
    events = session_store.load_transcript(session_id)
    events = hook_manager.run_pre_compact(events)
    summary = session_store.load_summary(session_id)
    budget = get_token_budget(events, summary, model_name)
    if budget.needs_compact or should_compact(events):
        summary = summarize_events(events)
        session_store.save_summary(session_id, summary)
        return summary
    return None


def orchestrate_session(
    user_query: str,
    model_name: str | None = None,
    max_iters: int = 3,
    initial_task_queue: TaskQueue | None = None,
    return_state: bool = False,
    session_id: str | None = None,
    session_store: SessionStore | None = None,
    tool_registry: ToolRegistry | None = None,
    permission_policy: PermissionPolicy | None = None,
    approval_callback: Callable[[ActionStep], bool] | None = None,
    hook_manager: HookManager | None = None,
) -> TaskQueue | tuple[TaskQueue, OrchestrationState]:
    """
    Orchestrate a session using a plan-act-observe-decide loop.
    """
    if tool_registry is None:
        tool_registry = load_registry()

    if session_store is None:
        session_store = SessionStore()

    if session_id is None:
        session_id = session_store.create_session()

    state = OrchestrationState(goal=user_query, session_id=session_id)
    state.summary = session_store.load_summary(session_id)
    if state.tool_results is None:
        state.tool_results = []
    if state.open_questions is None:
        state.open_questions = []
    if hook_manager is None:
        hook_manager = default_hook_manager()

    session_store.append_event(
        session_id, {"type": "user", "payload": {"text": user_query}}
    )

    updated_summary = _maybe_auto_compact(
        session_store,
        session_id,
        model_name,
        hook_manager,
    )
    if updated_summary:
        state.summary = updated_summary

    if user_query.strip() == "/compact":
        events = session_store.load_transcript(session_id)
        summary = summarize_events(events)
        session_store.save_summary(session_id, summary)
        state.summary = summary
        state.done = True
        state.done_reason = "compacted"
        task_queue = _build_direct_response(
            f"Compaction complete. Summary: {summary}"
        )
        if return_state:
            return task_queue, state
        return task_queue

    if initial_task_queue is None:
        task_queue = generate_action_plan(
            user_query=user_query,
            model_name=model_name,
            tool_registry=tool_registry,
            session_summary=state.summary,
        )
    else:
        task_queue = initial_task_queue
    state.plan = task_queue.action_steps
    steps: list[ActionStepPayload] = [
        {
            "action_consumer": step.action_consumer,
            "action_type": step.action_type,
            "action_argument": step.action_argument,
        }
        for step in task_queue.action_steps
    ]
    session_store.append_event(
        session_id,
        {
            "type": "action_plan",
            "payload": {"steps": steps},
        },
    )

    for iteration in range(max_iters):
        task_queue = run_action_plan(
            task_queue,
            tool_registry=tool_registry,
            event_logger=lambda event: session_store.append_event(session_id, event),
            permission_policy=permission_policy,
            approval_callback=approval_callback,
            hook_manager=hook_manager,
        )
        state.tool_results.append(task_queue.task_result or "")

        if _action_steps_complete(task_queue):
            state.done = True
            state.done_reason = "completed"
            break

        if iteration < max_iters - 1:
            revised_query = (
                f"{user_query}\n\nPrevious tool results:\n{task_queue.task_result or ''}\n"
                "Please revise the action plan to resolve remaining tasks."
            )
            task_queue = generate_action_plan(
                user_query=revised_query,
                model_name=model_name,
                tool_registry=tool_registry,
                session_summary=state.summary,
            )
            state.plan = task_queue.action_steps
            revised_steps: list[ActionStepPayload] = [
                {
                    "action_consumer": step.action_consumer,
                    "action_type": step.action_type,
                    "action_argument": step.action_argument,
                }
                for step in task_queue.action_steps
            ]
            session_store.append_event(
                session_id,
                {
                    "type": "action_plan",
                    "payload": {"steps": revised_steps},
                },
            )

    if not state.done:
        state.done_reason = "max_iterations_reached"

    session_store.append_event(
        session_id,
        {
            "type": "completion",
            "payload": {
                "done": state.done,
                "done_reason": state.done_reason,
                "task_result": task_queue.task_result,
            },
        },
    )

    updated_summary = _maybe_auto_compact(
        session_store,
        session_id,
        model_name,
        hook_manager,
    )
    if updated_summary:
        state.summary = updated_summary

    if return_state:
        return task_queue, state

    return task_queue
