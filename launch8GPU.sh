#!/bin/bash

# 获取 GPU 数量
n=$(nvidia-smi -L | wc -l)

# 起始端口
start_port=25000

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

# 生成端口并启动服务
for (( gpu_index=0; gpu_index<n; gpu_index++ )); do
    # 计算当前 GPU 对应的端口
    port=$((start_port + gpu_index))
    
    # 启动服务
    echo "Starting service on GPU $gpu_index with port $port"
    CUDA_VISIBLE_DEVICES=$gpu_index python3 -m sglang.launch_server --model-path Qwen/Qwen2-7B --host 0.0.0.0 --mem-fraction-static 0.7 --port $port &
    
    # 保存进程 ID
    pids+=($!)
done

# 等待所有后台进程完成
wait