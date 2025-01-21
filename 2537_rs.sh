python3 -m sglang.launch_server --model /workspace/Llama-2-7b-chat-hf \
--max-prefill-tokens 16384  --trust-remote-code --tp 1 --dp 1  --mem-fraction-static 0.5 \
--speculative-draft /workspace/sglang-EAGLE-llama2-chat-7B \
--speculative-num-steps 4 --speculative-eagle-topk 2 --speculative-num-draft-tokens 8 --speculative-algo EAGLE \
--disable-radix-cache
