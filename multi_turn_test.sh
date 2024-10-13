#!/bin/bash

# 获取当前时间并格式化为所需的文件名格式
current_time=$(date +"%Y%m%d_%H%M%S")
unset http_proxy
unset https_proxy
# 根据当前时间生成日志文件名
LOG_FILE="multi_turn_${current_time}_test_load_method.log"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 在日志文件中写入分隔符和当前时间
echo "====================== $(date) ======================" >> "$LOG_FILE"

# 定义不同的设置

turns=(20)

num_qa=(128 256 512 1024 2048 4096)

methods1=(
    # "--load-balance-method resources_aware"
    "--load-balance-method pre_radix" 
    "--load-balance-method bucket"
    # "--load-balance-method round_robin"
)
# methods2=(
    # "--dp-size 1"
    # "--dp-size 1 --disable-radix-cache"
# )


for turn in "${turns[@]}"; do
    for qa in "${num_qa[@]}"; do

        # 循环处理每个设置
        for method in "${methods1[@]}"; do
            unset http_proxy
            unset https_proxy

            echo "====================== $method turn=$turn qa=$qa start ======================" >> "$LOG_FILE"
            # 启动服务并将其放到后台，重定向输出到日志文件
            /workspace/bin/micromamba run -n sglang python3 -m sglang.launch_server \
                --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen2-7B/snapshots/453ed1575b739b5b03ce3758b23befdb0967f40e \
                --host 127.0.0.1 --port 8080 --mem-fraction-static 0.7 \
                --dp-size 8 \
                $method >> "$LOG_FILE" 2>&1 &
            sleep 300

            unset http_proxy
            unset https_proxy

            /workspace/bin/micromamba run -n sglang python3 /workspace/sglang/benchmark/multi_turn_chat/bench_sglang.py \
            --tokenizer /root/.cache/huggingface/hub/models--Qwen--Qwen2-7B/snapshots/453ed1575b739b5b03ce3758b23befdb0967f40e \
            --port 8080 --parallel 512 \
            --min-len-q 128 --max-len-q 256 \
            --min-len-a 256 --max-len-a 512 \
            --turns $turn --num-qa $qa >> "$LOG_FILE" 2>&1
            sleep 30
            # done

            # 杀死特定的 Python 进程
            ps -elf | grep python  | awk '{print $4}' | xargs  kill -s 9 
            ps -elf | grep python  | awk '{print $4}' | xargs  kill -s 9 
            # 等待一段时间以确保进程被杀死
            sleep 20
            echo "====================== $method turn=$turn qa=$qa end ======================" >> "$LOG_FILE"
        done
    done
done