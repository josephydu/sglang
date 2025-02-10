import time

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor

# 配置参数
model_name = "Qwen/Qwen2-VL-7B-Instruct"
batch_size = 2  # 根据GPU显存调整
device = "cuda:1" if torch.cuda.is_available() else "cpu"

# 加载模型（优化版）
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="flash_attention_2",
).eval()

processor = AutoProcessor.from_pretrained(model_name)


# 数据集加载与预处理
def collate_fn(batch):

    print(batch)
    images = [item["image"].convert("RGB") for item in batch]
    texts = [item["question"] for item in batch]

    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
        return_attention_mask=True,
    ).to(device)

    return inputs


dataset = load_dataset("lmarena-ai/vision-arena-bench-v0.1", split="train[:100]")
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

# 推理计时
total_time = 0
with torch.no_grad():
    for batch in dataloader:
        # CUDA事件精确计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()

        # 批量生成
        outputs = model.generate(
            **batch,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

        end_event.record()
        torch.cuda.synchronize()

        batch_time = start_event.elapsed_time(end_event) / 1000  # 转换为秒
        total_time += batch_time

print(f"Average time per batch: {total_time/len(dataloader):.4f}s")
print(f"Total tokens generated: {sum(len(seq) for seq in outputs)}")
