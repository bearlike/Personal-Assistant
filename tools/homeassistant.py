#!/usr/bin/env python3
import os
import time
import json
import logging
from typing import Tuple
from dotenv import load_dotenv
import requests
from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from homeassistant_api import Client
from pprint import pprint
import coloredlogs
import tqdm


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


# Configure logging
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('request').setLevel(logging.ERROR)
logging.getLogger('httpcore').setLevel(logging.ERROR)

# Create a logger
logger = logging.getLogger(__name__)
logger.addHandler(TqdmLoggingHandler())
coloredlogs.install(logger=logger)
coloredlogs.install()
load_dotenv()


def cache_monitor(func):
    def wrapper(self, *args, **kwargs):
        # Execute the original function
        result = func(self, *args, **kwargs)

        # Define the prefixes or substrings you want to exclude
        forbidden_prefixes = [
            'person.', 'tts.', 'stt.', 'sun.', 'sensor.hacs',
            'media_player.fire_tv_192_168_1_12', 'camera.', 'automation.',
            'media_player.axios', 'media_player.axios_2', 'zone.home',
            'alarm_control_panel.', 'switch.bedroom_camera_camera_motion_detection',
            'media_player.chrome', 'binary_sensor.remote_ui', 'switch.adam',
            'switch.', 'update.', 'sensor.sun', 'sensor.sonarr_commands',
            'sensor.kraken_raspberry_pi_5_', 'device_tracker.kraken_raspberry_pi_5',
            'script.higher'
        ]
        forbidden_substrings = ['blink_kk_bedroom']

        if 'entities' in self.cache:
            for entity in self.cache['entities']:
                if any(entity['entity_id'].startswith(prefix) for prefix in forbidden_prefixes):
                    self.cache['entities'].remove(entity)
                elif any(substring in entity['entity_id'] for substring in forbidden_substrings):
                    self.cache['entities'].remove(entity)

        if 'services' in self.cache:
            domain_whitelist = [
                "scene", "switch", "weather", "kodi", "automation"
            ]
            self.cache['services'] = [
                service for service in self.cache['services'] if service['domain'] in domain_whitelist
            ]

        # Now check and modify self.cache["entity_ids"] if needed
        if 'entity_ids' in self.cache:
            # Use list comprehension to filter out unwanted items
            self.cache['entity_ids'] = sorted(
                [item for item in self.cache['entity_ids']
                 if not any(item.startswith(prefix) for prefix in forbidden_prefixes)
                 and not any(substring in item for substring in forbidden_substrings)]
            )

        return result
    return wrapper


class HomeAssistant:
    def __init__(self):
        self.base_url = os.getenv("HA_URL", None)
        self._api_token = os.getenv("HA_TOKEN", None)
        self.allowed_domains = [
            "scene", "switch", "weather", "kodi", "automation"
        ]
        self.cache = {
            "entity_ids": [],  # List of all entity IDs
            "entities": [],  # List of all entities
            "services": [],  # List of all available services
        }

        # Data validation
        if not self.base_url or not self._api_token:
            raise ValueError(
                "HA_URL and HA_TOKEN must be set in the environment.")

        # Common headers
        self.api_headers = {
            "Authorization": f"Bearer {self._api_token}",
            "Content-Type": "application/json"
        }

        self.jinja_env = Environment(loader=FileSystemLoader("templates"))

    @cache_monitor
    def update_services(self):
        """
        Returns a list of entities from Home Assistant.

        Returns: None
        """
        url = f"{self.base_url}/services"
        try:
            response = requests.get(url, headers=self.api_headers, timeout=30)
            response.raise_for_status()
            # Parse the response as JSON
            self.cache["services"] = response.json()
            logging.info("Services updated.")
            return True
        except requests.exceptions.RequestException as e:
            logging.error("Error: %s", e)
            return False

    @cache_monitor
    def update_entities(self):
        """
        Updates the list of entities from Home Assistant to cache.

        Returns:
            bool: True if successful, False otherwise.
        """
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
        """
        Returns a list of all entity IDs.
        TODO: is_blacklist equals False is not implemented yet.
        Args:
            is_blacklist (bool): If True, the blacklist is used.
        Returns:
            bool: True if successful, False otherwise.
        """
        self.update_entities()
        entities = self.cache["entities"]
        if not entities:
            raise ValueError("No entities found.")
        self.cache["entity_ids"] = [entity["entity_id"] for entity in entities]
        logging.info("Entity IDs updated.")
        return True

    def update_cache(self):
        self.update_entity_ids()
        self.update_services()

    def render_template(self, template_path, variables=None):
        if not variables:
            variables = {}
        # Load the template file
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()

        # Prepare the payload
        payload = {
            "template": template,
            "variables": variables  # Add any required variables here
        }

        url = f"{self.base_url}/template"
        try:
            response = requests.post(url, headers=self.api_headers,
                                     json=payload, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logging.error("Error: %s", e)
            return False

    def call_service(
            self, domain, service, entity_id, data=None) -> Tuple[bool, list]:
        """
        Calls a service in Home Assistant.

        Args:
            domain (str): The domain of the service.
            service (str): The service to call.
            entity_id (str): The entity ID to call the service on.
            data (dict): Additional data to pass to the service.
        Returns:
            (bool, list): True if successful, False otherwise
                            list of entities changed if successful
        """
        # Data validation
        if domain not in self.allowed_domains:
            raise ValueError(f"Invalid domain: {domain}")

        url = f"{self.base_url}/services/{domain}/{service}"
        payload = {
            "entity_id": entity_id
        }
        if data:
            payload.update(data)

        try:
            response = requests.post(url, headers=self.api_headers,
                                     json=payload, timeout=30)
            response.raise_for_status()
            return (
                True, response.json()
            )
        except requests.exceptions.RequestException as e:
            logging.error("Unable to call service <%s.%s> on entity <%s>: %s",
                          domain, service, entity_id, e)
            return (False, [])


def _save_json(data, filename):
    """ Function to save a dictionary to a JSON file.
        Create .cache directory if it doesn't exist.
    """
    if not os.path.exists(".cache"):
        os.makedirs(".cache")
    filename = os.path.join(".cache", filename)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    logging.info(f"Data saved to {filename}.")


def test_homeassistant():
    """ Test the HomeAssistant class.
    """
    ha = HomeAssistant()
    entities = ha.update_cache()
    pprint(sorted(ha.cache["entity_ids"]))
    _save_json(ha.cache["services"], "services.json")
    _save_json(ha.cache["entities"], "entities.json")
    ha.call_service("scene", "turn_on", "scene.strip_lights_white")


if __name__ == "__main__":
    test_homeassistant()
