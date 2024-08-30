#!/bin/bash

# 获取当前时间并格式化为所需的文件名格式
current_time=$(date +"%Y%m%d_%H%M%S")

# 根据当前时间生成日志文件名
LOG_FILE="test_load_method_${current_time}.log"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# 在日志文件中写入分隔符和当前时间
echo "====================== $(date) ======================" >> "$LOG_FILE"

# 启动服务并将其放到后台，重定向输出到日志文件
echo "Running with setting: dp8 resources_aware ======================================================" >> "$LOG_FILE"
/workspace/bin/micromamba run -n sglang python3 -m sglang.launch_server --model-path Qwen/Qwen2-7B \
    --host 0.0.0.0 --port 8080 --mem-fraction-static 0.8 \
    --dp-size 8 --load-balance-method resources_aware \
    --chunked-prefill-size 2048 --disable-radix-cache >> "$LOG_FILE" 2>&1 &
export LOAD_BALANCE_METHOD="resources_aware"

# 等待一段时间以确保服务启动
sleep 300

# 启动 benchmark 脚本，重定向输出到日志文件
# 循环从 16 到 20，步进为 0.1
for rate in $(seq 16 0.1 20); do
    echo "Running with request-rate: $rate" | tee -a "$LOG_FILE"  # 输出当前的 request-rate 值并追加到日志文件
    /workspace/bin/micromamba run -n sglang python3 -m sglang.bench_serving --backend sglang \
            --host 127.0.0.1 --port 8080 --dataset-name random \
            --tokenizer Qwen/Qwen2-7B --model Qwen/Qwen2-7B \
            --random-output-len 1024 --random-input-len 4096 \
            --random-range-ratio 0.5 --seed 1234 \
            --num-prompts 20000 --request-rate $rate >> "$LOG_FILE" 2>&1
    sleep 100
done
ps -elf | grep python  | awk '{print $4}' | xargs  kill -s 9
sleep 300

#=============================================================================================================================================================


# 启动服务并将其放到后台，重定向输出到日志文件
echo "Running with setting: dp8 power_of_2_choice ======================================================" >> "$LOG_FILE"
/workspace/bin/micromamba run -n sglang python3 -m sglang.launch_server --model-path Qwen/Qwen2-7B \
    --host 0.0.0.0 --port 8080 --mem-fraction-static 0.8 \
    --dp-size 8 --load-balance-method power_of_2_choice \
    --chunked-prefill-size 2048 --disable-radix-cache >> "$LOG_FILE" 2>&1 &
export LOAD_BALANCE_METHOD="power_of_2_choice"

sleep 300

# 启动 benchmark 脚本，重定向输出到日志文件
# 循环从 16 到 20，步进为 0.1
for rate in $(seq 16 0.1 20); do
    echo "Running with request-rate: $rate" | tee -a "$LOG_FILE"  # 输出当前的 request-rate 值并追加到日志文件
    /workspace/bin/micromamba run -n sglang python3 -m sglang.bench_serving --backend sglang \
            --host 127.0.0.1 --port 8080 --dataset-name random \
            --tokenizer Qwen/Qwen2-7B --model Qwen/Qwen2-7B \
            --random-output-len 1024 --random-input-len 4096 \
            --random-range-ratio 0.5 --seed 1234 \
            --num-prompts 20000 --request-rate $rate >> "$LOG_FILE" 2>&1
    sleep 100
done
ps -elf | grep python  | awk '{print $4}' | xargs  kill -s 9
sleep 300

#=============================================================================================================================================================

# 启动服务并将其放到后台，重定向输出到日志文件
echo "Running with setting: dp8 round_robin ======================================================" >> "$LOG_FILE"
/workspace/bin/micromamba run -n sglang python3 -m sglang.launch_server --model-path Qwen/Qwen2-7B \
    --host 0.0.0.0 --port 8080 --mem-fraction-static 0.8 \
    --dp-size 8 --load-balance-method round_robin \
    --chunked-prefill-size 2048 --disable-radix-cache >> "$LOG_FILE" 2>&1 &
export LOAD_BALANCE_METHOD="round_robin"

sleep 300

# 启动 benchmark 脚本，重定向输出到日志文件
# 循环从 16 到 20，步进为 0.1
for rate in $(seq 16 0.1 20); do
    echo "Running with request-rate: $rate" | tee -a "$LOG_FILE"  # 输出当前的 request-rate 值并追加到日志文件
    /workspace/bin/micromamba run -n sglang python3 -m sglang.bench_serving --backend sglang \
            --host 127.0.0.1 --port 8080 --dataset-name random \
            --tokenizer Qwen/Qwen2-7B --model Qwen/Qwen2-7B \
            --random-output-len 1024 --random-input-len 4096 \
            --random-range-ratio 0.5 --seed 1234 \
            --num-prompts 20000 --request-rate $rate >> "$LOG_FILE" 2>&1
    sleep 100
done
ps -elf | grep python  | awk '{print $4}' | xargs  kill -s 9
sleep 300


#=============================================================================================================================================================


# echo "Running with setting: dp=1 tp=8 ======================================================" >> "$LOG_FILE"

# /workspace/bin/micromamba run -n sglang python3 -m sglang.launch_server --model-path Qwen/Qwen2-7B \
#     --host 0.0.0.0 --port 8080 --mem-fraction-static 0.8 \
#     --dp-size 1 --tp-size 8 --load-balance-method resources_aware \
#     --chunked-prefill-size 2048 --disable-radix-cache >> "$LOG_FILE" 2>&1 &
# sleep 300
# for rate in $(seq 3.5 0.25 6); do
#     echo "Running with request-rate: $rate" | tee -a "$LOG_FILE"  # 输出当前的 request-rate 值并追加到日志文件
#     /workspace/bin/micromamba run -n sglang python3 -m sglang.bench_serving --backend sglang \
#                     --host 127.0.0.1 --port 8080 --dataset-name random \
#                     --tokenizer Qwen/Qwen2-7B --model Qwen/Qwen2-7B \
#                     --random-output-len 1024 --random-input-len 4096 \
#                     --random-range-ratio 0.5 --seed 1234 \
#                     --num-prompts 5000 --request-rate $rate >> "$LOG_FILE" 2>&1
#     sleep 100
# done
# ps -elf | grep python  | awk '{print $4}' | xargs  kill -s 9
# sleep 100
# #=============================================================================================================================================================