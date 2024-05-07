#!/usr/bin/env python3
from utils.task_master import generate_action_plan
from pprint import pprint
from utils.homeassistant import HomeAssistant
from utils.common import get_logger

logging = get_logger(name="chat-cli")


if __name__ == "__main__":
    task_queue = generate_action_plan(user_query="How is hurricane server doing?")

    for action_step in task_queue.action_steps:
        logging.debug(f"ActionStep({action_step})")
        if action_step.action_consumer == "home_assistant":
            ha = HomeAssistant()
            speak = ha.run(action_step)
            print(f"AI: {speak.content}")
