import requests

url = f"http://127.0.0.1:{30000}/v1/chat/completions"

data = {
    "model": "WePoints",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "图片里面有什么?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "test.png"},
                },
            ],
        }
    ],
    "max_tokens": 15000,
    "temperature": 0,
}

response = requests.post(url, json=data)
print(response.text)
