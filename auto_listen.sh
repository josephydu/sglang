#!/bin/bash

# 定义要检查的 URL
URL="http://127.0.0.1:30000/health"

# 使用 curl 检查服务健康状态
response=$(curl -s -o /dev/null -w "%{http_code}" "$URL")
MEM_FRACTION_STATIC="--mem-fraction-static 0.6"
MODEL_PATH="--model-path /WePoints/"
# 检查 HTTP 状态码
if [ "$response" -eq 200 ]; then
    echo "服务正常运行 (HTTP 状态码: $response)"
else
    echo "服务异常 (HTTP 状态码: $response)，正在尝试启动服务..."
    python3 -m sglang.launch_server $MEM_FRACTION_STATIC $MODEL_PATH--trust-remote-code --chat-template qwen2-vl --chunked-prefill-size -1
fi
