import os
import time

import torch
import torch.distributed as dist
from torch.multiprocessing import Process


def run_all_reduce(rank, size, num_elements):
    """执行all_reduce操作并测量时间"""
    # 初始化进程组
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        init_method="env://",
        rank=rank,
        world_size=size,
    )

    # 创建一个随机张量
    tensor = torch.rand(num_elements).to(
        f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    )

    # 同步所有进程
    dist.barrier()

    # 开始计时
    start_time = time.time()

    # 执行all_reduce操作
    for _ in range(1000):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # 同步所有进程
    dist.barrier()

    # 结束计时
    end_time = time.time()

    # 输出性能结果
    if rank == 0:
        print(
            f"[rank0]All Reduce completed with {num_elements} elements in {end_time - start_time:.4f} seconds"
        )
    else:
        print(
            f"[rank1]All Reduce completed with {num_elements} elements in {end_time - start_time:.4f} seconds"
        )

    # 清理进程组
    dist.destroy_process_group()


def init_process(rank, size, num_elements, fn):
    """初始化进程并执行all_reduce操作"""
    fn(rank, size, num_elements)


if __name__ == "__main__":
    # 设置分布式环境变量
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "23456"

    # 定义张量的大小
    num_elements = 1000000  # 1百万个元素

    # 定义进程数量
    world_size = 2  # 2个进程

    # 创建进程
    processes = []
    for rank in range(world_size):
        p = Process(
            target=init_process, args=(rank, world_size, num_elements, run_all_reduce)
        )
        p.start()
        processes.append(p)

    # 等待所有进程完成
    for p in processes:
        p.join()
