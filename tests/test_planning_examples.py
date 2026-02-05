"""Tests for planner example message wrappers."""

from langchain_core.messages import AIMessage, HumanMessage
from meeseeks_core.planning import Planner


def test_planner_examples_are_wrapped():
    """Ensure planner examples are tagged as illustrative only."""
    messages = Planner._build_example_messages(["home_assistant_tool"])
    assert len(messages) == 4
    for message in messages:
        assert isinstance(message, HumanMessage | AIMessage)
        assert message.content.startswith(
            '<example desc="Illustrative only; not part of the live conversation">'
        )
        assert message.content.endswith("</example>")


def test_planner_examples_wrap_when_tools_missing():
    """Ensure example tags exist even when tool examples are empty."""
    messages = Planner._build_example_messages([])
    assert len(messages) == 4
    for message in messages:
        assert message.content.startswith(
            '<example desc="Illustrative only; not part of the live conversation">'
        )
        assert message.content.endswith("</example>")
