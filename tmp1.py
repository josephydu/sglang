import time

import requests

url = f"http://127.0.0.1:{30000}/v1/chat/completions"

data1 = {
    "model": "WePoints",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "图片里面有什么?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "test1.png"},
                },
            ],
        }
    ],
    "max_tokens": 4096,
    "temperature": 0,
}

data2 = {
    "model": "WePoints",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "图片里面有什么?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "test2.png"},
                },
            ],
        }
    ],
    "max_tokens": 4096,
    "temperature": 0,
}

try:
    while True:
        # 发送 data1
        response = requests.post(url, json=data1)
        print("Response for test1.png:", response.text)
        time.sleep(0.5)  # 等待0.5秒

        # 发送 data2
        response = requests.post(url, json=data2)
        print("Response for test2.png:", response.text)
        time.sleep(0.5)  # 等待0.5秒

except KeyboardInterrupt:
    print("\nStopped by user")
