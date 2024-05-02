import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Home Assistant instance URL
instance_url = "http://192.168.1.206:8123"

# Your Home Assistant access token
access_token = os.getenv("HA_TOKEN")

# Load the template file
with open("entity_info.txt.j2", "r", encoding="utf-8") as f:
    template = f.read()

# Prepare the payload
payload = {
    "template": template,
    "variables": {}  # Add any required variables here
}

# Authenticate and send the request
headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json"
}

url = f"{instance_url}/api/template"
response = requests.post(url, headers=headers, json=payload, timeout=30)

# Check the response
if response.status_code == 200:
    rendered_template = response.text
    print(rendered_template) # Reeturn This
else:
    print(f"Error: {response.status_code} - {response.text}") # Return Fals in this case
