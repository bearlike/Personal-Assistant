#!/usr/bin/env python3
# Third-party modules
from dotenv import load_dotenv
# User-defined modules
from utils.common import get_logger, get_mock_speaker
from utils.classes import ActionStep, AbstractTool

load_dotenv()
logging = get_logger(name="talk_to_user")

class TalkToUser(AbstractTool):
    """A service to manage and interact with Home Assistant."""

    def __init__(self):
        super().__init__(
            name="Talk to User",
            description="Directly talk to the user."
        )

    def set_state(self, action_step: ActionStep) -> "MockSpeaker":
        """
        An abstract method that subclasses should implement,
        performing the desired action.

        Returns:
            str: A message indicating the result of the action.
        """
        MockSpeaker = get_mock_speaker()
        return MockSpeaker(content=action_step.action_argument)
