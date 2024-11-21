#!/bin/bash

# 获取当前时间并格式化为所需的文件名格式
current_time=$(date +"%Y%m%d_%H%M%S")

# 根据当前时间生成日志文件名
LOG_FILE="test_load_method_${current_time}.log"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 在日志文件中写入分隔符和当前时间
echo "====================== $(date) ======================" >> "$LOG_FILE"

# 定义不同的设置
declare -A settings
# settings["resources_aware"]="dp8 resources_aware"
settings["pre_radix"]="dp8 pre_radix"
settings["round_robin"]="dp8 round_robin"

for random_output_len in 128 256 512 1024 2048 4096; do
    for random_input_len in 128 256 512 1024 2048 4096; do
            # 循环处理每个设置
            for method in "${!settings[@]}"; do
                setting=${settings[$method]}
                echo "Running with setting: $setting ======================================================" >> "$LOG_FILE"
                
                # 启动服务并将其放到后台，重定向输出到日志文件
                /workspace/bin/micromamba run -n sglang python3 -m sglang.launch_server \
                    --model-path Llama-3.1-8B-Instruct \
                    --host 127.0.0.1 --port 8080 --mem-fraction-static 0.7 \
                    --dp-size 8 \
                    --load-balance-method $method >> "$LOG_FILE" 2>&1 &

                # 等待一段时间以确保服务启动
                sleep 300

                # for rate in $(seq 16 0.1 16.2); do
                # 启动 benchmark 脚本，重定向输出到日志文件
                # 循环从 16 到 20，步进为 0.1
                /workspace/bin/micromamba run -n sglang python benchmark/multi_turn_chat/long_prompt_multi_turn.py \
                --tokenizer Llama-3.1-8B-Instruct --port 30000 \
                --parallel 155 \
                --len-q $random_input_len --len-a $random_output_len \
                --turns 11 --num-qa 1024

                sleep 100
                # done

                # 杀死特定的 Python 进程
                ps -elf | grep '[p]ython' | awk '{print $4}' | xargs kill -s 9
                ps -elf | grep '[p]ython' | awk '{print $4}' | xargs kill -s 9

                # 等待一段时间以确保进程被杀死
                sleep 30
            done
        done
    done
done