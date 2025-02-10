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
                        "url": "https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png"
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
