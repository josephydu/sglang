import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

for _ in range(10):
    response = client.chat.completions.create(
        model="/workspace/Qwen2-7B-Instruct",
        messages=[
            {
                "role": "user",
                "content": 'What are the mental triggers in Jeff Walker\'s Product Launch Formula and "Launch" book?',
            },
        ],
        temperature=0,
        max_tokens=1024,
    )
    completion_tokens = response.usage.completion_tokens
    print(f"{completion_tokens=}")
