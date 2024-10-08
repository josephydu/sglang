#!/bin/bash

# 获取 GPU 数量
n=$(nvidia-smi -L | wc -l)

# 起始端口
start_port=25000

# 端口增量
port_increment=100

# 启动服务的进程 ID 列表
pids=()

# 定义清理函数
cleanup() {
    echo "Cleaning up..."
    ps -elf | grep sglang  | awk '{print $4}' | xargs  kill -s 9 
    exit
}

# 捕获 EXIT 和 SIGINT 信号
trap cleanup EXIT SIGINT

# 生成端口并启动服务
for (( gpu_index=0; gpu_index<n; gpu_index++ )); do
    # 计算当前 GPU 对应的端口
    port=$((start_port + gpu_index * port_increment))
    
    # 启动服务
    echo "Starting service on GPU $gpu_index with port $port"
    CUDA_VISIBLE_DEVICES=$gpu_index python3 -m sglang.launch_server --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen2-7B/snapshots/453ed1575b739b5b03ce3758b23befdb0967f40e --host 127.0.0.1 --mem-fraction-static 0.7 --port $port &
    
    # 保存进程 ID
    pids+=($!)
done




# 等待所有后台进程完成
wait