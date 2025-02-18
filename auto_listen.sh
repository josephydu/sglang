#!/bin/bash

# 定义要检查的 URL
URL="http://127.0.0.1:30000/health"
CHECK_INTERVAL=30  # 检查间隔
WAIT_SERVER_TIME=60  # 等待时间


# 先清除服务
echo "正在清除其他服务..."
ps -elf | grep sglang  | awk '{print $4}' | xargs  kill -s 9
sleep 10  # 等待服务关闭
echo "清除完毕..."

# 设置启动命令
MEM_FRACTION_STATIC="0.6"
MODEL_PATH="/WePoints/"
DP="1"
START_COMMAND="python3 -m sglang.launch_server --model-path $MODEL_PATH --trust-remote-code --chat-template qwen2-vl --mem-fraction-static $MEM_FRACTION_STATIC --dp $DP"

SERVER_LOG_FILE="server.log"
# 尝试第一次启动服务
echo "正在启动服务..."
echo "启动命令"
echo "$START_COMMAND"
# 向日志文件中写入此时的时间
echo "==================启动时间: $(date)==================" >> "$SERVER_LOG_FILE"
eval $START_COMMAND >> "$SERVER_LOG_FILE" 2>&1 &
sleep $WAIT_SERVER_TIME  # 等待服务启动
echo "启动成功，开始监听服务状态..."


while true; do
    # 使用 curl 检查服务健康状态
    response=$(curl -s -o /dev/null -w "%{http_code}" "$URL")
    # 检查 HTTP 状态码
    if [ "$response" -eq 200 ]; then
        echo "服务正常运行 (HTTP 状态码: $response)"
    else
        echo "服务异常 (HTTP 状态码: $response)，正在尝试重新启动服务..."

        # 清除服务
        echo "1.尝试清除服务..."
        ps -elf | grep sglang  | awk '{print $4}' | xargs  kill -s 9
        sleep 10  # 等待10s服务关闭

        # 打印启动命令
        echo "2.打印启动命令..."
        echo "启动命令: $START_COMMAND"

        # 启动服务的命令
        echo "3.尝试启动服务..."
        echo "==================启动时间: $(date)==================" >> "$SERVER_LOG_FILE"
        eval $START_COMMAND >> "$SERVER_LOG_FILE" 2>&1 &
        sleep $WAIT_SERVER_TIME # 等待服务启动
    fi
    sleep $CHECK_INTERVAL
done
