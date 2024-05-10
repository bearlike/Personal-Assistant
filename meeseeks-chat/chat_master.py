#!/usr/bin/env python3
"""
Streamlit Chat App

This module implements a chat application using Streamlit and Meeseeks.
It allows users to interact with an AI assistant
through a chat interface.

# Please start Meeseeks Chat using the following command:
# streamlit run chat_master.py
"""
# Standard library modules
import time
import os
import sys
# Third-party modules
import streamlit as st
from langchain.memory import ConversationBufferWindowMemory

# TODO: Need to package the application and import it as module
# Adding the parent directory to the path before importing the custom modules
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Custom imports - Meeseeks core modules
from core.task_master import generate_action_plan, run_action_plan
from core.common import get_logger
from core.classes import TaskQueue

logging = get_logger(name="Meeseeks-Chat")


def generate_action_plan_helper(user_input: str):
    action_plan_list = []
    task_queue = generate_action_plan(user_query=user_input)
    for action_step in task_queue.action_steps:
        # * Append action step to the action plan list
        action_plan_list.append(
            f"Using `{action_step.action_consumer}` with "
            f"`{action_step.action_type}` to `{action_step.action_argument}`"
        )
    return action_plan_list, task_queue


def run_action_plan_helper(task_queue: TaskQueue):
    ai_response = []
    task_queue = run_action_plan(task_queue)
    for action_step in task_queue.action_steps:
        ai_response.append(action_step.result.content)

    ai_response = " ".join(ai_response)
    return ai_response


def main():
    """
    Main function to run the chat application.
    """
    st.set_page_config(
        page_title="Meeseeks | Bedroom AI",
        page_icon=":speech_balloon:",
    )
    image_path = os.path.join("static", "img", "banner.png")
    css_path = os.path.join("static", "css", "streamlit_custom.css")
    st.image(image_path, use_column_width=True)

    # Load css file as string into page_bg_img
    with open(css_path, encoding="utf-8") as f:
        page_bg_img = f.read()
    st.markdown(f"<style>{page_bg_img}</style>", unsafe_allow_html=True)

    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = ConversationBufferWindowMemory(
            k=5)
    conversation_memory = st.session_state.conversation_memory

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello Boss, How may I help you?"}
        ]

    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="👨‍💻"):
                st.markdown(message["content"])

        if message["role"] == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message["content"])

        if message["role"] == "thought":
            with st.chat_message("thought", avatar="🧠"):
                with st.expander("**Action Plan (Click to Expand)**"):
                    st.caption(message["content"])

    user_input = st.chat_input("Ask me anything ✏️")
    if user_input:
        user_input = user_input.strip()
        with st.chat_message("user", avatar="👨‍💻"):
            st.markdown(user_input)
            st.session_state.messages.append(
                {"role": "user", "content": user_input})

        with st.chat_message("thought", avatar="🧠"):
            with st.spinner("Creating Action Plan ..."):
                time.sleep(0.5)
                # * User query is processed here
                action_plan_list, task_queue = generate_action_plan_helper(
                    user_input)
                action_plan_caption = ""
                action_plan_caption = "\n* ".join(action_plan_list)
                if action_plan_list:
                    st.session_state.messages.append(
                        {"role": "thought",
                            "content": action_plan_caption}
                    )
                    with st.expander("**Action Plan (Click to Expand)**"):
                        st.caption(action_plan_caption)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Running Action Plan ..."):
                time.sleep(0.5)
                response = run_action_plan_helper(task_queue)
                st.markdown(response)
                conversation_memory.save_context(
                    {"input": user_input}, {"output": response})
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

    st.session_state.conversation_memory = conversation_memory


if __name__ == "__main__":
    main()
