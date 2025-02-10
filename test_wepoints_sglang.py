import requests

url = "http://127.0.0.1:30000/v1/chat/completions"

data = {
    "model": "WePoints",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请用中文描述图片里的内容"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://wiki.01studio.cc/en/assets/images/mnist1-cc229fb1d4374aa0aef6b4d26b2d22a3.jpg"
                    },
                },
            ],
        }
    ],
    "max_new_tokens": 1024,
    "temperature": 0.0,
    "do_sample": False,
}

response = requests.post(url, json=data)
print(response.text)
