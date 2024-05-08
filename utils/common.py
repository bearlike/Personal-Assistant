#!/usr/bin/env python3
import time
import os
import tqdm
from collections import namedtuple
from jinja2 import Environment, FileSystemLoader, TemplateNotFound
import coloredlogs
import logging as logging_real
import tiktoken


def get_mock_speaker():
    """ Return a mock speaker for testing. """
    return namedtuple('MockSpeaker', 'content')


def get_logger(name=None) -> logging_real.Logger:
    """ Get the logger for the module.

    Args:
        name (str, optional): Name of the logger. Defaults to __name__.

    Returns:
        logging.Logger: The logger object.
    """
    logging_real.basicConfig(level=logging_real.DEBUG,
                             format='%(asctime)s - %(levelname)s - %(message)s')
    loggers_to_suppress = [
        'request', 'httpcore', 'urllib3.connectionpool', 'openai._base_client',
        'aiohttp_client_cache.signatures', 'LangChainDeprecationWarning',
        'watchdog.observers.inotify_buffer', 'PIL.PngImagePlugin'
    ]
    for logger_name in loggers_to_suppress:
        logging_real.getLogger(logger_name).setLevel(logging_real.ERROR)

    if not name:
        name = __name__
    logger = logging_real.getLogger(name)

    coloredlogs.install(logger=logger, level='DEBUG')
    return logger


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """ Get the number of tokens in a string using a specific model.

    Args:
        string (str): The string for which the token length is required.
        encoding_name (str): The name of the model.

    Returns:
        int: Number of tokens for the string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_unique_timestamp():
    """ Get a unique timestamp for the task queue. """
    # Get the number of seconds since epoch (Jan 1, 1970) as a float
    current_timestamp = int(time.time())
    # Convert it to string for uniqueness and consistency
    unique_timestamp = str(current_timestamp)
    # Return the integer version of this string timestamp
    return int(''.join(str(x) for x in map(int, unique_timestamp)))


def get_system_prompt(name: str = "action-planner") -> str:
    """ Get the system prompt for the task queue.

    Returns:
        str: The system prompt for the task queue.
    """
    logging = get_logger(name="get_system_prompt")
    system_prompt_path = os.path.join(
        os.path.dirname(__file__), "..", "prompts", f"{name}.txt")
    with open(system_prompt_path, "r", encoding="utf-8") as system_prompt_file:
        system_prompt = system_prompt_file.read()
    logging.debug(f"Getting system prompt from `{system_prompt_path}`")
    del logging
    return system_prompt.strip()


def ha_render_system_prompt(
        all_entities=None, name="homeassistant-set-state") -> str:
    """ Render the system j2 prompt. Need to make it more generic.

    Returns:
        str: The system prompt for the task queue.
    """
    if all_entities is not None:
        all_entities = str(all_entities).strip()
    logging = get_logger(name="render_system_prompt")
    env = Environment(loader=FileSystemLoader("prompts"))
    template = env.get_template(f"{name}.txt")
    logging.debug(f"Render system prompt for `{name}`")
    del logging
    return template.render(ALL_ENTITIES=all_entities)
