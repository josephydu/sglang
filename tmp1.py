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
                    "image_url": {"url": "test.png"},
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
        requests.post(url, json=data1)
        time.sleep(0.1)  # 等待0.5秒

        # 发送 data2
        requests.post(url, json=data2)
        time.sleep(0.1)  # 等待0.5秒

except KeyboardInterrupt:
    print("\nStopped by user")
