#!/usr/bin/env python3
import copy
import json
import os
import time
import warnings
from typing import Optional, Tuple, List

import requests
import tqdm
from dotenv import load_dotenv
from homeassistant_api import Client
from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from pprint import pprint

from utils.classes import AbstractTool, ActionStep, TaskQueue
from utils.common import get_logger, ha_render_system_prompt, num_tokens_from_string

logging = get_logger(name="utils.homeassistant")
load_dotenv()


def cache_monitor(func):
    """Decorator to monitor and update the cache."""
    @staticmethod
    def sort_by_entity_id(dict_list: List[dict]) -> List[dict]:
        return sorted(dict_list, key=lambda x: x['entity_id'])

    def clean_entities(self, forbidden_prefixes, forbidden_substrings):
        for idx, entity in enumerate(self.cache['entities']):
            if 'context' in entity:
                self.cache['entities'][idx].pop('context')
                self.cache['entities'][idx].pop('last_changed')
                self.cache['entities'][idx].pop('last_reported')
                self.cache['entities'][idx].pop('last_updated')

            if 'attributes' in entity:
                self.cache['entities'][idx]['attributes'].pop('icon', None)
                self.cache['entities'][idx]['attributes'].pop(
                    "monitor_cert_days_remaining", None)
                self.cache['entities'][idx]['attributes'].pop(
                    "monitor_cert_is_valid", None)
                self.cache['entities'][idx]['attributes'].pop(
                    "monitor_hostname", None)
                self.cache['entities'][idx]['attributes'].pop(
                    "monitor_port", None)

            if any(entity['entity_id'].startswith(prefix) for prefix in forbidden_prefixes):
                self.cache['entities'].remove(entity)

            if any(substring in entity['entity_id'] for substring in forbidden_substrings):
                self.cache['entities'].remove(entity)

            if entity['entity_id'].startswith('scene.'):
                self.cache['entities'][idx].pop('state', None)

            if entity['entity_id'].startswith('sensor.'):
                self.cache['sensors'].append(entity)
                self.cache['entities'].pop(idx)
        self.cache['entities'] = sort_by_entity_id(self.cache['entities'])
        self.cache['sensors'] = sort_by_entity_id(self.cache['sensors'])
        return self.cache

    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)

        forbidden_prefixes = [
            'person.', 'tts.', 'stt.', 'sun.', 'sensor.hacs',
            'media_player.fire_tv_192_168_1_12', 'camera.', 'automation.',
            'media_player.axios', 'media_player.axios_2', 'zone.home',
            'media_player.chrome', 'binary_sensor.remote_ui', 'switch.adam',
            'switch.', 'update.', 'sensor.sun', 'sensor.sonarr_commands',
            'script.higher', 'conversation', 'remote.', 'sensor.uptimekuma_',
            'alarm_control_panel.', 'sensor.kraken_raspberry_pi_5_', "climate",
            'switch.bedroom_camera_camera_motion_detection', "sensor.hacs",
            'device_tracker.kraken_raspberry_pi_5',
        ]
        forbidden_substrings = ['blink_kk_bedroom']
        self.cache["sensor"] = []
        # Clean entities
        self.cache = clean_entities(
            self, forbidden_prefixes, forbidden_substrings)

        # Clean services
        self.cache['services'] = [
            service for service in self.cache['services'] if service['domain'] in self.cache["allowed_domains"]
        ]

        # Retrieve entity and sensor IDs
        self.cache['entity_ids'] = sorted(
            [item for item in self.cache['entity_ids']])
        self.cache['sensor_ids'] = sorted(
            [item for item in self.cache['sensor_ids']])

        logging.info(
            "Cache status: <Entity IDs: %s; Entities: %s; Sensors: %s; Services: %s;>",
            len(self.cache['entity_ids']), len(self.cache['entities']),
            len(self.cache['sensors']), len(self.cache['services'])
        )

        return result
    return wrapper


class HomeAssistantCall(BaseModel):
    cache: Optional[dict] = Field(alias="_ha_cache", default={})
    domain: str = Field(
        description="The category of the service to call, such as 'light', 'switch', or 'scene'.")
    service: str = Field(
        description="The specific action to perform within the domain, such as 'turn_on', 'turn_off', or 'set_temperature'.")
    entity_id: str = Field(
        description="The ID of the specific device or entity within the domain to apply the service to, such as 'scene.heater'.")

    def __init__(self, ha_cache: dict = None, **data):
        super().__init__(**data)
        if ha_cache:
            self.cache: Optional[dict] = Field(
                alias="_ha_cache", default=ha_cache)

    @validator("entity_id")
    def validate_entity_id(cls, entity_id, values, **kwargs):
        # ! BUG: The entity_id is not being validated correctly as
        # !     cache is not being passed to the validator.
        ha_cache = values.get("ha_cache")
        if ha_cache and entity_id not in ha_cache.cache["entity_ids"]:
            raise ValueError(
                f"Entity ID '{entity_id}' is not in the Home Assistant cache.")
        return entity_id

    @validator("domain")
    def validate_domain(cls, domain, values, **kwargs):
        # ! BUG: The entity_id is not being validated correctly as
        # !     cache is not being passed to the validator.
        ha_cache = values.get("ha_cache")
        if ha_cache and domain not in ha_cache.cache["allowed_domains"]:
            raise ValueError(
                f"Domain '{domain}' is not in the Home Assistant cache.")
        return domain


class HomeAssistant(AbstractTool):
    """A service to manage and interact with Home Assistant."""

    def __init__(self):
        super().__init__(
            name="Home Assistant",
            description="A service to manage and interact with Home Assistant"
        )
        self.base_url = os.getenv("HA_URL", None)
        self._api_token = os.getenv("HA_TOKEN", None)
        self.cache = {
            "entity_ids": [],
            "sensor_ids": [],
            "entities": [],
            "services": [],
            "sensors": [],
            "allowed_domains": ["scene", "switch", "weather", "kodi", "automation"]
        }

        if not self.base_url or not self._api_token:
            raise ValueError(
                "HA_URL and HA_TOKEN must be set in the environment.")

        self.api_headers = {
            "Authorization": f"Bearer {self._api_token}",
            "Content-Type": "application/json"
        }

    @cache_monitor
    def update_services(self):
        """Update the list of services from Home Assistant."""
        url = f"{self.base_url}/services"
        try:
            response = requests.get(url, headers=self.api_headers, timeout=30)
            response.raise_for_status()
            self.cache["services"] = response.json()
            self._save_json(self.cache["services"], "services.json")
            logging.info("Services updated.")
            return True
        except requests.exceptions.RequestException as e:
            logging.error("Error: %s", e)
            return False

    @cache_monitor
    def update_entities(self):
        """Update the list of entities from Home Assistant."""
        url = f"{self.base_url}/states"
        try:
            response = requests.get(url, headers=self.api_headers, timeout=30)
            response.raise_for_status()
            self.cache["entities"] = response.json()
            logging.info("Entities updated.")
            return True
        except requests.exceptions.RequestException as e:
            logging.error("Error: %s", e)
            return False

    @cache_monitor
    def update_entity_ids(self, is_blacklist=True):
        """Update the list of entity IDs from Home Assistant."""
        self.update_entities()
        entities = self.cache["entities"]
        if not entities:
            raise ValueError("No entities found.")
        self.cache["entity_ids"] = [entity["entity_id"] for entity in entities]
        logging.info("Entity IDs updated.")
        return True

    @cache_monitor
    def update_cache(self):
        """Update the entire cache."""
        self.update_entity_ids()
        self.update_services()
        self._save_json(self.cache["entities"], "entities.json")
        self._save_json(self.cache["sensors"], "sensors.json")

    def render_template(self, template_path, variables=None):
        """Render a template with the provided variables."""
        if not variables:
            variables = {}

        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()

        payload = {
            "template": template,
            "variables": variables
        }

        url = f"{self.base_url}/template"
        try:
            response = requests.post(
                url, headers=self.api_headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logging.error("Error: %s", e)
            return False

    def call_service(self, domain: str, service: str, entity_id: str, data: Optional[dict] = None) -> Tuple[bool, list]:
        """Call a service in Home Assistant."""
        if domain not in self.cache["allowed_domains"]:
            raise ValueError(f"Invalid domain: {domain}")

        url = f"{self.base_url}/services/{domain}/{service}"
        payload = {"entity_id": entity_id}
        if data:
            payload.update(data)

        try:
            response = requests.post(
                url, headers=self.api_headers, json=payload, timeout=30)
            response.raise_for_status()
            logging.info("Service <%s.%s> called on entity <%s> returned `%s`.",
                         domain, service, entity_id, response.text)
            return True, response.json()
        except requests.exceptions.RequestException as e:
            logging.error(
                "Unable to call service <%s.%s> on entity <%s>: %s", domain, service, entity_id, e)
            return False, []

    def _create_set_prompt(self, system_prompt: str, parser: PydanticOutputParser) -> ChatPromptTemplate:
        example = HomeAssistantCall(
            domain="scene", service="turn_on", entity_id="scene.lamp_power_on")
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessage(content=system_prompt),
                HumanMessage(content="Turn on the lamp lights."),
                AIMessage(example.json()),
                HumanMessagePromptTemplate.from_template(
                    "The user asked you to `{action_step}`. You must use the information provided to pick the right Home Assistant service call values.\n\n## Format Instructions\n{format_instructions}\n\n## Home Assistant Entities and Domain-Services\n```\n{context}```\n"
                ),
            ],
            partial_variables={
                "format_instructions": parser.get_format_instructions()},
            input_variables=["action_step"]
        )
        return prompt

    def _create_get_prompt(self, system_prompt: str) -> ChatPromptTemplate:
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessage(content=system_prompt),
                HumanMessage(content="How is the air quality today?"),
                AIMessage(content="AccuWeather reported today's air quality in your home as good. This level of air quality ensures that the environment is healthy, supporting your daily activities and wellbeing without any air quality-related risks."),
                HumanMessagePromptTemplate.from_template(
                    "The user asked you to `{action_step}`. You must use the sensor information to answer the user's query. Keep your answer analytical, brief and useful.\n\n## Home Assistant Sensors\n```\n{context}```\n"
                ),
            ],
            input_variables=["action_step"]
        )
        return prompt

    def set_state(self, action_step: ActionStep) -> str:
        """Perform the action defined by this service."""
        self.update_cache()
        rag_documents = self._load_rag_documents([
            "entities.json", "services.json"
        ])
        system_prompt = ha_render_system_prompt(
            name="homeassistant-set-state", all_entities=self.cache["entity_ids"])

        parser = PydanticOutputParser(pydantic_object=HomeAssistantCall)
        prompt = self._create_set_prompt(system_prompt, parser)
        chain = prompt | self.model | parser

        logging.info("Invoking `set` action chain using `%s`.",
                     self.model_name)
        call_service_values = chain.invoke(
            {
                "action_step": action_step.action_argument.strip(),
                "context": rag_documents,
                "cache": self.cache
            },
        )
        status_bool, response_json = self.call_service(
            domain=call_service_values.domain,
            service=call_service_values.service,
            entity_id=call_service_values.entity_id,
        )
        if status_bool:
            return f"Successfully called service: {response_json}"
        else:
            return f"Failed to call service: {response_json}"

    @staticmethod
    def _clean_answer(answer: str) -> str:
        """Clean the answer by removing/replacing characters."""
        # Replace confusing sensor names.
        answer = answer.replace("RealFeel ", "")
        # (confident) Abbreviations
        answer = answer.replace("km/h", " kilometer per hour")
        answer = answer.replace("°C", " degrees celsius")
        answer = answer.replace("%", " percent")
        answer = answer.replace("mm/h", " millimeter per hour")
        answer = answer.replace("Gb/s", " gigabits per second")
        answer = answer.replace("GHz", "Gigahertz")
        # Remove extra spaces
        answer = answer.replace("\n", " ")
        answer = answer.replace("   ", " ")
        answer = answer.replace("  ", " ")
        answer = answer.replace("\"", "")
        return answer

    def get_state(self, action_step: ActionStep) -> str:
        """Perform the action defined by this service."""
        self.update_cache()
        rag_documents = self._load_rag_documents(["sensors.json"])

        system_prompt = ha_render_system_prompt(name="homeassistant-get-state")

        prompt = self._create_get_prompt(system_prompt)
        chain = prompt | self.model

        logging.info("Invoking `get` action chain using `%s`.",
                     self.model_name)
        message = chain.invoke(
            {
                "action_step": action_step.action_argument.strip(),
                "context": rag_documents,
            },
        )
        # Remove newline characters in the message.content
        message.content = self._clean_answer(message.content)
        return message


def test_homeassistant():
    """Test the HomeAssistant class."""
    ha = HomeAssistant()
    ha.update_cache()
    ha.call_service("scene", "turn_on", "scene.strip_lights_white")
    return ha
