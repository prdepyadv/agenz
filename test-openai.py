import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

os.environ["OPENAI_API_KEY"] = api_key
# os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
# os.environ["SSL_CERT_FILE"] = certifi.where()

#Sample CURL
#curl -X POST "https://api.openai.com/v1/chat/completions" -H "Authorization: Bearer <TOKEN>" -H "Content-Type: application/json" -d '{"model": "gpt-4o","messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello, how are you?"}],"max_tokens": 50}'
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
data = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 50
}
url = "https://api.openai.com/v1/chat/completions"
response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json())