import argparse
import time

import torch
import torch.distributed as dist


class Workload:
    def __init__(self, data_size, warmup, repeat, rank, world_size):
        self.stream = torch.cuda.Stream()
        self.num_warmup = warmup
        self.num_repeat = repeat
        self.rank = rank
        self.device = "cuda:" + str(rank)
        self.data = self.create_data(data_size)
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.world_size = world_size

    def create_data(self, data_size):
        raise NotImplementedError

    def work(self, data):
        raise NotImplementedError

    def warmup(self):
        with torch.cuda.stream(self.stream):
            for i in range(self.num_warmup):
                self.work(self.data)

    def test(self):
        with torch.cuda.stream(self.stream):
            self.start.record(self.stream)
            for i in range(self.num_repeat):
                self.work(self.data)
            self.end.record()

    def print_profile(self):
        raise NotImplementedError


class AllReduceWorkload(Workload):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_data(self, data_size):
        return torch.randn(
            data_size, data_size, dtype=torch.float32, device=self.device
        )

    def work(self, data):
        dist.all_reduce(data, op=dist.ReduceOp.SUM)

    def print_profile(self):
        if self.rank == 0:
            elapsed_time = self.start.elapsed_time(self.end) / 1e3
            data_gb = (self.data.element_size() * self.data.nelement()) / (1024**3)
            total_data = (
                data_gb * self.num_repeat * 2 * (self.world_size - 1)
            )  # 总传输数据量
            bandwidth = total_data / elapsed_time  # GB/s
            print("===========AllReduceWorkload===========")
            print(f"World size: {self.world_size}")
            print(f"Data per iteration: {data_gb:.2f}GB")
            print(f"Total time: {elapsed_time:.3f} seconds")
            print(f"Average bandwidth: {bandwidth:.2f} GB/s")


class GemmWorkload(Workload):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_data(self, data_size):
        return torch.randn(
            data_size, data_size, dtype=torch.float16, device=self.device
        )

    def work(self, data):
        torch.matmul(data, data)

    def print_profile(self):
        if self.rank == 0:
            elapsed_time = self.start.elapsed_time(self.end) / 1e3
            data_size = self.data.shape[0]
            flops = 2 * (data_size**3) * self.num_repeat
            tflops = flops / elapsed_time / (1024**4)  # TFlops
            print("===========GemmWorkload===========")
            print(f"Total time: {elapsed_time:.3f} seconds")
            print(f"Matrix size: {data_size} seconds")
            print(f"Average TFlops: {tflops:.2f} Tflops")


def benchmark_all_reduce(
    backend="nccl", device="cuda", data_size=1024**3, num_iters=10
):
    # 初始化进程组
    dist.init_process_group(backend)
    rank = dist.get_rank()
    if device == "cuda":
        device = "cuda:" + str(rank)
    world_size = dist.get_world_size()

    all_reduce_worker = AllReduceWorkload(8192, 5, num_iters, rank, world_size)
    gemm_worker = GemmWorkload(10240, 5, num_iters, rank, world_size)
    dist.barrier()
    all_reduce_worker.warmup()
    gemm_worker.warmup()

    torch.cuda.synchronize()
    dist.barrier()
    time.sleep(4)

    start_time = time.time()
    all_reduce_worker.test()
    gemm_worker.test()
    torch.cuda.synchronize()
    dist.barrier()
    elapsed_time = time.time() - start_time
    print(elapsed_time)

    all_reduce_worker.print_profile()
    gemm_worker.print_profile()

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"])
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--data-size", type=int, default=1024**3)
    parser.add_argument("--num-iters", type=int, default=10)
    args = parser.parse_args()

    benchmark_all_reduce(
        backend=args.backend,
        device=args.device,
        data_size=args.data_size,
        num_iters=args.num_iters,
    )
