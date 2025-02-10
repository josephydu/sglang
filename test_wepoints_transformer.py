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
image_url = (
    "https://github.com/user-attachments/assets/83258e94-5d61-48ef-a87f-80dd9d895524"
)
response = requests.get(image_url)
image_data = BytesIO(response.content)
pil_image = Image.open(image_data)
pil_image = pil_image.save("image.jpg")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What is in this image?",
            },
            {"type": "image", "image": "image.jpg"},
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
