import os
import time

import torch
import torch.distributed as dist
from torch.multiprocessing import Process


def run_all_reduce_with_gemm(rank, size, num_elements, matrix_size):
    """执行all_reduce操作并在另一个CUDA Stream中并行执行GEMM"""
    # 初始化进程组
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        init_method="env://",
        rank=rank,
        world_size=size,
    )

    # 创建一个随机张量用于all_reduce
    tensor = torch.rand((num_elements, num_elements)).to(
        f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    )
    print(tensor.dtype)

    # 创建两个随机矩阵用于GEMM
    matrix_a = torch.rand(matrix_size, matrix_size).to(
        f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    )
    matrix_b = torch.rand(matrix_size, matrix_size).to(
        f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    )

    # 创建两个新的CUDA Stream
    # gemm_stream = torch.cuda.Stream(device=f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    comm_stream = torch.cuda.Stream(
        device=f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    )

    # 同步所有进程
    dist.barrier()

    # 开始计时
    start_time = time.time()

    # 在通信流中执行all_reduce
    with torch.cuda.stream(comm_stream):
        # start_comm = time.time()
        for _ in range(100):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        # comm_stream.synchronize()  # 确保all_reduce完成
        # end_comm = time.time()

    # 在GEMM流中执行GEMM
    # with torch.cuda.stream(gemm_stream):
    # start_gemm = time.time()
    # for _ in range(100):
    # torch.mm(matrix_a, matrix_b)
    # gemm_stream.synchronize()  # 确保GEMM完成
    # end_gemm = time.time()

    # 同步所有进程
    dist.barrier()
    torch.cuda.synchronize()

    # 结束计时
    end_time = time.time()

    # 输出性能结果
    if rank == 0:
        # print("rank0---------------------")
        # print(f"All Reduce completed in {end_comm - start_comm:.4f} seconds")
        # print(f"GEMM completed in {end_gemm - start_gemm:.4f} seconds")
        print(f"Total time: {end_time - start_time:.4f} seconds")
    else:
        # print("rank1---------------------")
        # print(f"All Reduce completed in {end_comm - start_comm:.4f} seconds")
        # print(f"GEMM completed in {end_gemm - start_gemm:.4f} seconds")
        print(f"Total time: {end_time - start_time:.4f} seconds")

    # 清理进程组
    dist.destroy_process_group()


def init_process(rank, size, num_elements, matrix_size, fn):
    """初始化进程并执行all_reduce和GEMM操作"""
    fn(rank, size, num_elements, matrix_size)


if __name__ == "__main__":
    # 设置分布式环境变量
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"

    # 定义张量的大小
    num_elements = 20480 * 4 // 10

    # 定义矩阵的大小
    matrix_size = 2048

    # 定义进程数量
    world_size = 2

    # 创建进程
    processes = []
    for rank in range(world_size):
        p = Process(
            target=init_process,
            args=(
                rank,
                world_size,
                num_elements,
                matrix_size,
                run_all_reduce_with_gemm,
            ),
        )
        p.start()
        processes.append(p)

    # 等待所有进程完成
    for p in processes:
        p.join()
