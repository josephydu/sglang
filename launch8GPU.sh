#!/bin/bash

# 获取 GPU 数量
n=$(nvidia-smi -L | wc -l)

# 起始端口
start_port=25000

# 用于存储已生成的端口
declare -A ports

# 启动服务的进程 ID 列表
pids=()

# 定义清理函数
cleanup() {
    echo "Cleaning up..."
    for pid in "${pids[@]}"; do
        echo "Killing process $pid"
        kill "$pid" 2>/dev/null
    done
    exit
}

# 捕获 EXIT 信号
trap cleanup EXIT

# 生成随机端口并启动服务
while [ ${#ports[@]} -lt $n ]; do
    # 生成一个随机端口号，范围在 25000 到 25999 之间
    port=$((RANDOM % 1000 + start_port))
    
    # 确保端口唯一
    if [[ -z ${ports[$port]} ]]; then
        ports[$port]=1
        
        # 获取当前 GPU 的索引
        gpu_index=${#ports[@]} - 1  # 从 0 开始计数
        
        # 启动服务
        echo "Starting service on GPU $gpu_index with port $port"
        CUDA_VISIBLE_DEVICES=$gpu_index python3 -m sglang.launch_server --model-path Qwen/Qwen2-7B --host 0.0.0.0 --mem-fraction-static 0.7 --port $port &
        
        # 保存进程 ID
        pids+=($!)
    fi
done

# 等待所有后台进程完成
wait