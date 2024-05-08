#!/usr/bin/env python3
import abc
import os
import json
from typing import Optional
from typing import List, Any
# Third-party modules
from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langfuse.callback import CallbackHandler
from dotenv import load_dotenv
# User-defined modules
from utils.common import get_logger, get_unique_timestamp, get_mock_speaker

load_dotenv()
logging = get_logger(name="classes")
AVAILABLE_TOOLS = ["home_assistant_tool", "talk_to_user_tool"]


class ActionStep(BaseModel):
    action_consumer: str = Field(
        description=f"Specify one of {AVAILABLE_TOOLS} to indicate the action consumer."
    )
    action_type: str = Field(
        description="Specify either 'get' or 'set' to indicate the action type."
    )
    action_argument: str = Field(
        description="Provide details for the action. If 'task', specify the task to perform. If 'talk', include the message to speak to the user."
    )
    result: Optional[Any] = Field(
        alias="_result",
        default="Not executed yet.",
        description='Private field to persist the action status and other data.'
    )


class TaskQueue(BaseModel):
    human_message: Optional[str] = Field(
        alias="_human_message",
        description='Human message associated with the task queue.'
    )
    action_steps: Optional[List[ActionStep]] = None

    @validator("action_steps", allow_reuse=True)
    def validate_actions(cls, field):
        for action in field:
            # Normalize once and store it
            action.action_consumer = action.action_consumer.lower()
            action.action_type = action.action_type.lower()
            error_msg_list = []

            # Check if action_consumer is valid
            if action.action_consumer not in AVAILABLE_TOOLS:
                error_msg_list.append(
                    f"`{action.action_consumer}` is not a valid Assistant consumer.")

            # Check if action_type is valid
            if action.action_type not in ["get", "set"]:
                error_msg = f"`{action.action_type}` is not a valid action type."
                error_msg_list.append(error_msg)

            # Specific checks for "talk_to_user" consumer
            if action.action_consumer == "talk_to_user" and \
                    action.action_type == "get":
                error_msg = f"`{action.action_consumer}` does not support 'get' action type."
                error_msg_list.append(error_msg)

            # Check for None argument
            if action.action_argument is None:
                error_msg_list.append("Action argument cannot be None.")

            # Handle errors if any
            if error_msg_list:
                error_msg = "\n".join(error_msg_list)
                for msg in error_msg_list:
                    logging.error(msg)  # Log each error message
                raise ValueError(error_msg)

        return field


class AbstractTool(abc.ABC):
    def __init__(self, name, description, model_name=None):
        # Data Validation
        if model_name is None:
            default_model = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
            self.model_name = os.getenv("TOOL_MODEL", default_model)
        else:
            self.model_name = model_name

        # Set the tool attributes
        self.name = name
        self._id = f"{name.lower().replace(' ', '_')}_tool"
        self.description = description
        session_id = f"{self._id}-tool-id-{get_unique_timestamp()}"
        logging.info(f"Tool created <name={name}; session_id={session_id};>")
        self.langfuse_handler = CallbackHandler(
            user_id="homeassistant_kk",
            session_id=session_id,
            trace_name=f"meeseeks-{self._id}-tool",
            version=os.getenv("VERSION", "Not Specified"),
            release=os.getenv("ENVMODE", "Not Specified")
        )
        self.model = ChatOpenAI(
            openai_api_base=os.getenv("OPENAI_API_BASE"),
            model=self.model_name,
            temperature=0.3
        )
        cache_dir = os.path.join(os.path.dirname(
            __file__), "..", ".cache", self._id)
        self.cache_dir = os.path.abspath(cache_dir)

    def _save_json(self, data, filename):
        """Save a dictionary to a JSON file."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        filename = os.path.join(self.cache_dir, filename)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        logging.info(f"Data saved to {filename}.")

    def _load_rag_json(self, filename) -> list:
        """Load a dictionary from a JSON file."""
        logging.debug("RAG directory is %s.", self.cache_dir)
        logging.info(f"Loading `{filename}` as JSON.")
        filename = os.path.join(self.cache_dir, filename)
        filename = os.path.abspath(filename)
        loader = JSONLoader(
            file_path=filename,
            jq_schema='.',
            text_content=False)
        data = loader.load()
        return data

    def _load_rag_documents(self, filenames: List[str]) -> list:
        rag_documents = []
        for rag_file in filenames:
            data = self._load_rag_json(rag_file)
            rag_documents.extend(data)
        return rag_documents

    def set_state(self, action_step: ActionStep = None) -> "MockSpeaker":
        """
        An abstract method that subclasses should implement,
        performing the desired action.

        Returns:
            str: A message indicating the result of the action.
        """
        MockSpeaker = get_mock_speaker()
        return MockSpeaker(content="Not implemented yet.")

    def get_state(self, action_step: ActionStep = None) -> "MockSpeaker":
        """
        An abstract method that subclasses should implement,
        performing the desired action.

        Returns:
            str: A message indicating the result of the action.
        """
        MockSpeaker = get_mock_speaker()
        return MockSpeaker(content="Not implemented yet.")

    def run(self, action_step: ActionStep) -> str:
        """
        Executes the action based on the action type.

        Arguments:
            action_step (ActionStep): An ActionStep object with the action details.

        Returns:
            str: A message indicating the result of the action.
        """
        if action_step.action_type == "set":
            return self.set_state(action_step)
        elif action_step.action_type == "get":
            return self.get_state(action_step)
        else:
            raise ValueError(f"Invalid action type: {action_step.action_type}")


def create_task_queue(
    action_data: List[dict] = None, is_example=True
) -> TaskQueue:
    """
    Creates a new TaskQueue object and assigns values from action_data.

    Arguments:
        action_data (List[dict]): List of dictionaries, where each
                    dictionary represents an action step with keys
                    'action_consumer' and 'action_argument'.

    Returns:
        A TaskQueue object with the provided action steps.
    """
    if action_data is None:
        ValueError("Action data cannot be None.")

    # Convert the input data to ActionStep objects
    action_steps = [
        ActionStep(action_consumer=action['action_consumer'],
                   action_argument=action['action_argument'],
                   action_type=action['action_type'])
        for action in action_data
    ]
    # Create a TaskQueue object with the action steps
    task_queue = TaskQueue(action_steps=action_steps)
    if is_example:
        del task_queue.human_message
    return task_queue


def get_task_master_examples(id: int = 0):
    """Get the example task queue data."""
    examples = [
        [
            {"action_consumer": "home_assistant_tool", "action_type": "set",
                "action_argument": "Power on the strip lights."},
            {"action_consumer": "home_assistant_tool", "action_type": "set",
                "action_argument": "Power on the Heater."},
            {"action_consumer": "talk_to_user_tool", "action_type": "set",
                "action_argument": "Got it, boss! I'm using Home Assistant to power on the strip lights and the heater."}
        ],
        [
            {"action_consumer": "home_assistant_tool", "action_type": "get",
                "action_argument": "Get today's weather."},
        ]
    ]
    if id not in range(0, len(examples)):
        raise ValueError(f"Invalid example ID: {id}")

    return create_task_queue(action_data=examples[id], is_example=True).json()
