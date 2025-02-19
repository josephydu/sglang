import time

import requests

url = f"http://127.0.0.1:{30000}/v1/chat/completions"

test_pngs = [
    "test_pngs/test1.png",
    "test_pngs/test2.png",
    "test_pngs/test3.png",
    "test_pngs/test4.png",
]


def construct_data(png_path):
    data = {
        "model": "WePoints",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "识别图片中的文字。"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"{png_path}"},
                    },
                ],
            }
        ],
        "max_tokens": 1024,
        "temperature": 0,
    }
    return data


repeats = 1
with open("spda.txt", "w") as f:
    for repeat in range(repeats):
        for png in test_pngs:
            f.write(
                f"============================turn={repeat},png={png}============================\n"
            )
            data = construct_data(png)
            response = requests.post(url, json=data)
            output = response.json()["choices"][0]["message"]["content"]
            f.write(f"{output}\n")
            time.sleep(0.5)  # 等待0.5秒
