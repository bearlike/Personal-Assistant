#!/usr/bin/env python3

# Standard library modules
import warnings
from typing import List
import os

# Third-party modules
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.messages.ai import AIMessage
from langchain.output_parsers import PydanticOutputParser
from langchain_community.document_loaders import JSONLoader
from langfuse.callback import CallbackHandler
from langchain_core._api.beta_decorator import LangChainBetaWarning
from dotenv import load_dotenv

# User-defined modules
from core.classes import TaskQueue, get_task_master_examples
from core.common import get_unique_timestamp
from core.common import get_logger, get_system_prompt
from core.common import num_tokens_from_string
from tools.integration.homeassistant import HomeAssistant
from tools.core.talk_to_user import TalkToUser

logging = get_logger(name="core.task_master")


# Filter out LangChainBetaWarning specifically
warnings.simplefilter("ignore", LangChainBetaWarning)
load_dotenv()


def generate_action_plan(
        user_query: str, model_name: str = None) -> List[dict]:
    """
        Use the LangChain pipeline to generate an action plan
        based on the user query.

    Args:
        user_query (str): The user query to generate the action plan.

    Returns:
        List[dict]: The generated action plan as a list of dictionaries.
    """
    langfuse_handler = CallbackHandler(
        user_id="homeassistant_kk",
        session_id=f"action-queue-id-{get_unique_timestamp()}",
        trace_name="meeseeks-task-master",
        version=os.getenv("VERSION", "Not Specified"),
        release=os.getenv("ENVMODE", "Not Specified")
    )

    if model_name is None:
        default_model = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
        model_name = os.getenv("ACTION_PLAN_MODEL", default_model)

    model = ChatOpenAI(
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        model=model_name,
        temperature=0.4
    )
    # Instantiate the parser with the new model.
    parser = PydanticOutputParser(pydantic_object=TaskQueue)
    logging.debug(
        "Generating action plan <model='%s'; user_query='%s'>",
        model_name, user_query)
    # Update the prompt to match the new query and desired format.
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessage(
                content=get_system_prompt()
            ),
            HumanMessage(
                content="Turn on strip lights and heater."
            ),
            AIMessage(get_task_master_examples(id=0)),
            HumanMessage(
                content="What is the weather today?"
            ),
            AIMessage(get_task_master_examples(id=1)),
            HumanMessagePromptTemplate.from_template(
                "## Format Instructions\n{format_instructions}\n## Generate a task queue for the user query\n{user_query}"
            ),
        ],
        partial_variables={
            "format_instructions": parser.get_format_instructions()},
        input_variables=["user_query"]
    )
    estimator = num_tokens_from_string(str(prompt))
    logging.info("Input Prompt Token length is `%s`.", estimator)
    chain = prompt | model | parser

    action_plan = chain.invoke({"user_query": user_query.strip()},
                               config={"callbacks": [langfuse_handler]})
    action_plan.human_message = user_query
    logging.info("Action plan generated <%s>", action_plan)
    return action_plan


def run_action_plan(task_queue: TaskQueue) -> TaskQueue:
    """
    Run the generated action plan.

    Args:
        task_queue (TaskQueue): The action plan to run.

    Returns:
        TaskQueue: The updated action plan after running.
    """
    tool_dict = {
        "home_assistant_tool": HomeAssistant(),
        "talk_to_user_tool": TalkToUser()
    }
    for idx, action_step in enumerate(task_queue.action_steps):
        logging.debug(f"<ActionStep({action_step})>")
        tool = tool_dict[action_step.action_consumer]
        action_plan = tool.run(action_step)
        task_queue.action_steps[idx].result = action_plan

    return task_queue
