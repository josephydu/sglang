#!/bin/bash

# 定义要检查的 URL
URL="http://127.0.0.1:30000/health"

# 使用 curl 检查服务健康状态
response=$(curl -s -o /dev/null -w "%{http_code}" "$URL")

# 检查 HTTP 状态码
if [ "$response" -eq 200 ]; then
    echo "服务正常运行 (HTTP 状态码: $response)"
else
    echo "服务异常 (HTTP 状态码: $response)"
fi
