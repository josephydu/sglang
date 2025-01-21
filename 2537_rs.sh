python3 -m sglang.launch_server --model /workspace/Qwen2-7B-Instruct \
--max-prefill-tokens 16384  --trust-remote-code --tp 1 --dp 1  --mem-fraction-static 0.5 \
--speculative-draft /workspace/EAGLE-Qwen2-7B-Instruct \
--speculative-num-steps 4 --speculative-eagle-topk 2 --speculative-num-draft-tokens 8 --speculative-algo EAGLE \
--disable-radix-cache
