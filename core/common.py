#!/usr/bin/env python3
"""Common helpers shared across the assistant runtime."""
from __future__ import annotations

import logging as logging_real
import os
import time
from typing import Any, NamedTuple

import coloredlogs
import tiktoken
from jinja2 import Environment, FileSystemLoader


class MockSpeaker(NamedTuple):
    """Simple mock response container used across tools and tests.

    Attributes:
        content: Text content returned by a tool.
    """
    content: str


def get_mock_speaker() -> type[MockSpeaker]:
    """Return a mock speaker for testing.

    Returns:
        MockSpeaker class for constructing responses.
    """
    return MockSpeaker


def get_logger(name: str | None = None) -> logging_real.Logger:
    """Get the logger for the module.

    Args:
        name: Name of the logger, defaults to __name__.

    Returns:
        Logger configured with colored output.
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

    coloredlogs.install(logger=logger, level=os.getenv("LOG_LEVEL", "ERROR"))
    return logger


def num_tokens_from_string(
        string: str, encoding_name: str = "cl100k_base") -> int:
    """Get the number of tokens in a string using a specific model.

    Args:
        string: Text to tokenize.
        encoding_name: Encoding name used for tokenization.

    Returns:
        Number of tokens for the string.
    """
    # TODO: Add support for dynamic model selection
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_unique_timestamp() -> int:
    """Get a unique timestamp for the task queue.

    Returns:
        Integer timestamp suitable for unique IDs.
    """
    # Get the number of seconds since epoch (Jan 1, 1970) as a float
    current_timestamp = int(time.time())
    # Convert it to string for uniqueness and consistency
    unique_timestamp = str(current_timestamp)
    # Return the integer version of this string timestamp
    return int(''.join(str(x) for x in map(int, unique_timestamp)))


def get_system_prompt(name: str = "action-planner") -> str:
    """Get the system prompt for the task queue.

    Args:
        name: Prompt file name without extension.

    Returns:
        System prompt string.

    Raises:
        OSError: If the prompt file cannot be read.
    """
    logging = get_logger(name="core.common.get_system_prompt")
    system_prompt_path = os.path.join(
        os.path.dirname(__file__), "..", "prompts", f"{name}.txt")
    with open(system_prompt_path, encoding="utf-8") as system_prompt_file:
        system_prompt = system_prompt_file.read()
    logging.debug("Getting system prompt from `%s`", system_prompt_path)
    del logging
    return system_prompt.strip()


def ha_render_system_prompt(
    all_entities: Any | None = None,
    env: str = "prompts",
    name: str = "homeassistant-set-state",
) -> str:
    """Render the Home Assistant Jinja2 system prompt.

    Args:
        all_entities: Optional entity list for template substitution.
        env: Template root directory name.
        name: Template file name without extension.

    Returns:
        Rendered system prompt string.
    """
    if all_entities is not None:
        all_entities = str(all_entities).strip()
    logging = get_logger(name="core.common.render_system_prompt")

    template_root = os.path.join(__name__, "..", "..", "prompts")
    template_root = os.path.abspath(template_root)
    logging.debug("Compiling %s from %s.", name, template_root)
    # TODO: Catch and log TemplateNotFound when necessary.
    template_env = Environment(loader=FileSystemLoader(template_root))
    template = template_env.get_template(f"{name}.txt")
    logging.debug("Render system prompt for `%s`", name)
    del logging

    return template.render(ALL_ENTITIES=all_entities)
