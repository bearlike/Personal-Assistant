#!/usr/bin/env python3
import logging
import copy
import os

from typing import Optional  # Add the missing import statement
from typing import List

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage

from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain.output_parsers import OutputFixingParser
from langchain.schema import OutputParserException

from langchain_core.prompts import PromptTemplate
from langfuse.callback import CallbackHandler

from dotenv import load_dotenv
from pprint import pprint
import time

load_dotenv()


class ActionStep(BaseModel):
    action_type: str = Field(
        description="Specify either 'task' or 'talk' to indicate the action type.")
    action_argument: str = Field(
        description="Provide details for the action. If 'task', specify the task to perform. If 'talk', include the message to speak to the user.")
    is_executed: Optional[bool] = Field(
        alias="_is_executed", description='Private field to indicate if the action has been executed.')


class TaskQueue(BaseModel):
    ActionSteps: Optional[List[ActionStep]] = None

    @validator("ActionSteps")
    def validate_actions(cls, field):
        for ActionID, _ in enumerate(field):
            field[ActionID].action_type = field[ActionID].action_type.lower()
            if field[ActionID].action_type not in ['task', 'talk']:
                raise ValueError(
                    f"{field[ActionID].action_type} is not a valid Assistant action type.")
        return field


def get_unique_timestamp():
    # Get the number of seconds since epoch (Jan 1, 1970) as a float
    current_timestamp = int(time.time())
    # Convert it to string for uniqueness and consistency
    unique_timestamp = str(current_timestamp)
    # Return the integer version of this string timestamp
    return int(''.join(str(x) for x in map(int, unique_timestamp)))


def get_system_prompt() -> str:
    """ Get the system prompt for the task queue.

    Returns:
        str: The system prompt for the task queue.
    """
    system_prompt_path = os.path.join(
        os.path.dirname(__file__), "..", "prompts", "ha-tools-system.txt")
    with open(system_prompt_path, "r", encoding="utf-8") as system_prompt_file:
        system_prompt = system_prompt_file.read()
    return system_prompt


def main():
    langfuse_handler = CallbackHandler(
        user_id="homeassistant_kk",
        session_id=f"ha-tool-trace-id-{get_unique_timestamp()}",
        trace_name="ha-tool-task-queue",
        version=os.getenv("VERSION", "Not Specified"),
        release=os.getenv("ENVMODE", "Not Specified")
    )

    model = ChatOpenAI(
        openai_api_base="http://llm.hurricane.home/",
        model="microsoft/phi3",
        temperature=0.1
    )
    # Instantiate the parser with the new model.
    parser = PydanticOutputParser(pydantic_object=TaskQueue)

    # Update the prompt to match the new query and desired format.
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessage(
                content=get_system_prompt()
            ),
            HumanMessagePromptTemplate.from_template(
                "## Format Instructions\n{format_instructions}\n## Generate a task queue for the user query\n{user_query}"
            )
        ],
        partial_variables={
            "format_instructions": parser.get_format_instructions()},
        input_variables=["user_query"]
    )

    chain = prompt | model | parser
    answer = chain.invoke({"user_query": "Who is Barack Obama?"},
                          config={"callbacks": [langfuse_handler]})
    pprint(answer.ActionSteps)


if __name__ == "__main__":
    main()
