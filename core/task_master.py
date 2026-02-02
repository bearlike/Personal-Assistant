#!/usr/bin/env python3

# Standard library modules
import os
import warnings
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
from core.session_store import SessionStore
from core.tool_registry import ToolRegistry, load_registry

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
    event_logger=None,
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

    results = []

    for action_step in task_queue.action_steps:
        logging.debug(f"Processing ActionStep: {action_step}")
        tool = tool_registry.get(action_step.action_consumer)

        if tool is None:
            logging.error(
                f"No tool found for consumer: {action_step.action_consumer}")
            continue

        try:
            action_result = tool.run(action_step)
            action_step.result = action_result
            results.append(
                action_result.content if action_result.content is not None else "")
            if event_logger is not None:
                event_logger(
                    {
                        "type": "tool_result",
                        "payload": {
                            "action_consumer": action_step.action_consumer,
                            "action_type": action_step.action_type,
                            "action_argument": action_step.action_argument,
                            "result": action_result.content,
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


def orchestrate_session(
    user_query: str,
    model_name: str | None = None,
    max_iters: int = 3,
    initial_task_queue: TaskQueue | None = None,
    return_state: bool = False,
    session_id: str | None = None,
    session_store: SessionStore | None = None,
    tool_registry: ToolRegistry | None = None,
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

    session_store.append_event(session_id, {"type": "user", "payload": {"text": user_query}})

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
    session_store.append_event(
        session_id,
        {
            "type": "action_plan",
            "payload": {
                "steps": [
                    {
                        "action_consumer": step.action_consumer,
                        "action_type": step.action_type,
                        "action_argument": step.action_argument,
                    }
                    for step in task_queue.action_steps
                ]
            },
        },
    )

    for iteration in range(max_iters):
        task_queue = run_action_plan(
            task_queue,
            tool_registry=tool_registry,
            event_logger=lambda event: session_store.append_event(session_id, event),
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
            session_store.append_event(
                session_id,
                {
                    "type": "action_plan",
                    "payload": {
                        "steps": [
                            {
                                "action_consumer": step.action_consumer,
                                "action_type": step.action_type,
                                "action_argument": step.action_argument,
                            }
                            for step in task_queue.action_steps
                        ]
                    },
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

    events = session_store.load_transcript(session_id)
    if should_compact(events):
        summary = summarize_events(events)
        session_store.save_summary(session_id, summary)
        state.summary = summary

    if return_state:
        return task_queue, state

    return task_queue
