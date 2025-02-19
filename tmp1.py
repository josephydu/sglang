import time

import requests

url = f"http://127.0.0.1:{30000}/v1/chat/completions"

test_pngs = [
    "test_pngs/test1.png",
    "test_pngs/test2.png",
    "test_pngs/test3.png",
    "test_pngs/test4.png",
    "test_pngs/test5.png",
]


def construct_data(png_path):
    data = {
        "model": "WePoints",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "图片里面有什么?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"{png_path}"},
                    },
                ],
            }
        ],
        "max_tokens": 4096,
        "temperature": 0,
    }
    return data


for test_png in test_pngs:
    for png in test_pngs:
        data = construct_data(png)
        response = requests.post(url, json=data)
        print(f"Response for {png}:", response.text)
        time.sleep(0.5)  # 等待0.5秒
