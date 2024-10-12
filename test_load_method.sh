# #!/bin/bash


#!/bin/bash

# 获取当前时间并格式化为所需的文件名格式
unset http_proxy
unset https_proxy
current_time=$(date +"%Y%m%d_%H%M%S")

# 根据当前时间生成日志文件名
LOG_FILE="v0.3.3.post1_test_load_method_${current_time}.log"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 在日志文件中写入分隔符和当前时间
echo "====================== $(date) ======================" >> "$LOG_FILE"

# 定义不同的设置
declare -A settings
# settings["power_of_2_choice"]="dp8 power_of_2_choice"
settings["resources_aware"]="dp8 resources_aware"
settings["round_robin"]="dp8 round_robin"

# for rate in $(seq 16 0.1 16.2); do
for rate in 9.1 9.2 9.3 9.4 9.58 9.6 9.65 9.7 10.0; do
    # 循环处理每个设置
    for method in "${!settings[@]}"; do
        setting=${settings[$method]}
        echo "Running with setting: $setting ======================================================" >> "$LOG_FILE"
        
        # 启动服务并将其放到后台，重定向输出到日志文件
        /workspace/bin/micromamba run -n sglang python3 -m sglang.launch_server \
            --model-path /root/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
            --host 127.0.0.1 --port 8080 --mem-fraction-static 0.7 \
            --dp-size 8 \
            --load-balance-method $method >> "$LOG_FILE" 2>&1 &
        export LOAD_BALANCE_METHOD=$method

        # 等待一段时间以确保服务启动
        sleep 300

        # for rate in $(seq 16 0.1 16.2); do
        # 启动 benchmark 脚本，重定向输出到日志文件
        # 循环从 16 到 20，步进为 0.1
        echo "Running with request-rate: $rate" | tee -a "$LOG_FILE"  # 输出当前的 request-rate 值并追加到日志文件
        /workspace/bin/micromamba run -n sglang python3 -m sglang.bench_serving --backend sglang \
                --host 127.0.0.1 --port 8080 --dataset-name random \
                --tokenizer /root/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
                --model /root/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
                --random-output-len 1024 --random-input-len 4096 \
                --random-range-ratio 0.5 --seed 1234 \
                --num-prompts 1000 --request-rate $rate >> "$LOG_FILE" 2>&1
        sleep 100
        # done

        # 杀死特定的 Python 进程
        ps -elf | grep '[p]ython' | awk '{print $4}' | xargs kill -s 9

        # 等待一段时间以确保进程被杀死
        sleep 300
    done
done