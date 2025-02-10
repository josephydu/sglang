from io import BytesIO

import requests

# from wepoints.utils.images import Qwen2ImageProcessorForPOINTSV15
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

model_path = "/WePoints"
model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
image_processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What is in this image?",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"
                },
            },
        ],
    }
]

generation_config = {
    "max_new_tokens": 1024,
    "temperature": 0.0,
    "top_p": 0.0,
    "num_beams": 1,
}
response = model.chat(messages, tokenizer, image_processor, generation_config)
print(response)
