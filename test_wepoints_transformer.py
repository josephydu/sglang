# from io import BytesIO

# import requests

# # from wepoints.utils.images import Qwen2ImageProcessorForPOINTSV15
# import torch
# from PIL import Image
# from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

# model_path = "/WePoints"
# model = AutoModelForCausalLM.from_pretrained(
#     model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="cuda"
# )
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# image_processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
# image_url = "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"
# response = requests.get(image_url)
# image_data = BytesIO(response.content)
# pil_image = Image.open(image_data)
# pil_image = pil_image.save("image.jpg")
# content = [
#     dict(type="image", image="image.jpg"),
#     dict(type="text", text="What is in this image?"),
# ]
# messages = [{"role": "user", "content": content}]

from io import BytesIO

import requests
import torch
from PIL import Image

# generation_config = {
#     "max_new_tokens": 1024,
#     "temperature": 0.0,
#     "top_p": 0.0,
#     "num_beams": 1,
# }
# response = model.chat(messages, tokenizer, image_processor, generation_config)
# print(response)
from transformers import AutoModelForCausalLM, AutoTokenizer
from wepoints.utils.images import Qwen2ImageProcessorForPOINTSV15

model_path = "/WePoints/"
model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
image_processor = Qwen2ImageProcessorForPOINTSV15.from_pretrained(model_path)
print("==============")
print(image_processor)
print("==============")

image_url = (
    "https://github.com/user-attachments/assets/83258e94-5d61-48ef-a87f-80dd9d895524"
)
response = requests.get(image_url)
image_data = BytesIO(response.content)
pil_image = Image.open(image_data)
pil_image = pil_image.save("image.jpg")
prompt = "please describe the image in detail"

content = [dict(type="image", image="image.jpg"), dict(type="text", text=prompt)]
messages = [{"role": "user", "content": content}]
generation_config = {
    "max_new_tokens": 1024,
    "temperature": 0.0,
    "top_p": 0.5,
    "num_beams": 1,
}
response = model.chat(messages, tokenizer, image_processor, generation_config)
print(response)
