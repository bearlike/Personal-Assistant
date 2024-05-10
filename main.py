#!/usr/bin/env python3
"""
Streamlit Chat App

This module implements a chat application using Streamlit, Langchain, and
OpenAI's GPT-3.5-turbo model. It allows users to interact with an AI assistant
through a chat interface.
"""

import requests
import os
import time
import streamlit as st
from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langfuse.callback import CallbackHandler
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
# Custom imports
from tools.homeassistant import HomeAssistant
from core.common import get_unique_timestamp
from core.common import get_logger

logging = get_logger(name="ChatApp")


load_dotenv()
langfuse_handler = CallbackHandler(
    user_id="homeassistant_kk",
    session_id=f"ha-tool-trace-id-{get_unique_timestamp()}",
    trace_name="meeseeks-chat-app",
    version=os.getenv("VERSION", "Not Specified"),
    release=os.getenv("ENVMODE", "Not Specified")

)
tools = {
    "homeassistant": HomeAssistant()
}


def get_models():
    """
    Returns a list of available models from OpenAI's API.

    Returns:
        list: A list of available models.
    """
    models = [
        "microsoft/phi3",
        "openai/gpt-3.5-turbo",
        "openai/gpt-4-turbo",
        "mistralai/mixtral-8x7b-32768",
        "custom/nous-hermes-2-mixtral-8x7b-dpo",
        "custom/WizardLM-2-8x22B-AWQ",
        "anthropic/claude-3-opus",
        "anthropic/claude-3-sonnet",
        "anthropic/claude-3-haiku",
        "anthropic/claude-instant-1.2",
        "meta/llama3-70b-8192",
        "meta/llama3-8b-8192",
        "google/gemini-pro",
        "google/gemini-1.5-pro"
    ]
    return models


def render_ha_entities_template(conversation_memory) -> str:
    """
    Renders a Jinja2 template designed for Home Assistant entities, with a
    specific HA_Entities value.

    Returns:
        str: The rendered template as a string.

    Raises:
        FileNotFoundError: If the specified template file does not exist.
        TemplateNotFound: If the j2 template is not found in the path.
    """

    # Hardcoded template path
    template_path = 'template.txt.j2'
    conversation_history = conversation_memory.load_memory_variables({})[
        "history"]
    print(f"conversation_memory={conversation_history}")

    HA_Entities = render_template()
    if not HA_Entities:
        logging.error("Failed to render template")
        raise RuntimeError("Failed to render template from Home Assistant")
    try:
        # Setup the environ. with the directory containing your template files
        env = Environment(loader=FileSystemLoader('templates'))
        # Load the template from the file
        template = env.get_template(template_path)
    except TemplateNotFound:
        raise FileNotFoundError(
            f"Template {template_path} not found in the current directory.")

    # Render the template with the HA_Entities variable
    rendered_template = template.render(
        HA_Entities=HA_Entities, conversation_history=conversation_history)

    return rendered_template


def render_template():
    template_path = os.path.join(
        os.path.dirname(__file__), "templates", "entity_info.txt.j2")
    return tools["homeassistant"].render_template(template_path)


def main():
    """
    Main function to run the chat application.
    """
    st.set_page_config(
        page_title="Meeseeks | Bedroom AI",
        page_icon=":speech_balloon:",
        layout="wide",
    )
    st.title("🚀 Meeseeks: Personal Assistant 🗨️")

    models = get_models()
    selected_model = st.selectbox("Select a model", list(models))

    # llm = OpenAI()

    # Load the model from OpenAI's API and create a chat object with it
    chat_model = ChatOpenAI(
        model=selected_model,
        temperature=0.65,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = ConversationBufferWindowMemory(
            k=5)
    conversation_memory = st.session_state.conversation_memory

    # Compile the system message jinja2 prompt template from template.txt.j2
    system_message_prompt = render_ha_entities_template(conversation_memory)

    # Define the human message prompt
    human_template = "{input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        human_template)

    # Create the chat prompt template
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt])

    # Create the conversation chain
    conversation = LLMChain(llm=chat_model, prompt=chat_prompt)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello Boss, How may I help you?"}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Say Something")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
            st.session_state.messages.append(
                {"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                time.sleep(0.5)
                response = conversation.predict(
                    input=user_input, callbacks=[langfuse_handler])
                conversation_memory.save_context(
                    {"input": user_input}, {"output": response})
                st.write(response)
        st.session_state.messages.append(
            {"role": "assistant", "content": response})

    st.session_state.conversation_memory = conversation_memory


if __name__ == "__main__":
    main()
