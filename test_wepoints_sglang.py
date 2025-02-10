import requests

url = "http://127.0.0.1:30000/v1/chat/completions"

data = {
    "model": "WePoints",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "please describe the image in detail"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://github.com/user-attachments/assets/83258e94-5d61-48ef-a87f-80dd9d895524"
                    },
                },
            ],
        }
    ],
    "max_new_tokens": 1024,
    "temperature": 0.0,
}

response = requests.post(url, json=data)
print(response.text)
