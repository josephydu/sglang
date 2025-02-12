import argparse
import time
from contextlib import contextmanager
from typing import Optional

import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton.profiler as proton
import triton.tools.experimental_descriptor

if torch.cuda.is_available():
    from triton._C.libtriton import nvidia
else:
    cublas = None


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


HAS_TMA_DESC = "nv_tma_desc_type" in dir(tl)

if HAS_TMA_DESC:
    print(
        "TMA benchmarks will be running with experimental grid constant TMA descriptor.",
    )
else:
    print(
        "TMA benchmarks will be running without grid constant TMA descriptor.",
    )


class TmaAutoTuneHelper:

    # duck typing wrapper to implement the same interface as TmaDescKernelParam in Triton PR #4498
    class KernelParamWrapper:

        def __init__(self, desc):
            self.desc = desc

        def tma_desc_cpu_ptr(self):
            return self.desc.data_ptr()

    TMA_SIZE = 128

    def __init__(self):
        self.fill_1d_tma_descriptor_inner = (
            triton.runtime.driver.active.utils.fill_1d_tma_descriptor
        )
        self.fill_2d_tma_descriptor_inner = (
            triton.runtime.driver.active.utils.fill_2d_tma_descriptor
        )
        if HAS_TMA_DESC:
            self.descriptors = {}
        else:
            self.cuda_descriptors = {}

    # Call this method outside of the lambda function for grid size
    def init_tma_descriptor(self, name):
        if HAS_TMA_DESC:
            self.descriptors[name] = torch.empty(
                TmaAutoTuneHelper.TMA_SIZE, device="cpu", dtype=torch.int8
            )
        else:
            self.cuda_descriptors[name] = torch.empty(
                TmaAutoTuneHelper.TMA_SIZE, device="cuda", dtype=torch.int8
            )

    # Call this method inside the lambda function for grid size
    def fill_1d_tma_descriptor(self, name, ptr, dim, block_dim, element_size):
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            assert desc_x.data_ptr() % 64 == 0
            self.fill_1d_tma_descriptor_inner(
                ptr, dim, block_dim, element_size, desc_x.data_ptr()
            )
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_1d_tma_descriptor_inner(
                ptr, dim, block_dim, element_size, buf_x.data_ptr()
            )
            desc_x.copy_(buf_x, non_blocking=True)

    # Call this method inside the lambda function for grid size
    def fill_2d_tma_descriptor(
        self, name, ptr, dim1, dim0, block_dim1, block_dim0, element_size
    ):
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            assert desc_x.data_ptr() % 64 == 0
            self.fill_2d_tma_descriptor_inner(
                ptr, dim1, dim0, block_dim1, block_dim0, element_size, desc_x.data_ptr()
            )
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_2d_tma_descriptor_inner(
                ptr, dim1, dim0, block_dim1, block_dim0, element_size, buf_x.data_ptr()
            )
            desc_x.copy_(buf_x, non_blocking=True)

    def get_tma_descriptor_kernel_param(self, name):
        if HAS_TMA_DESC:
            assert self.descriptors[name] is not None
            return self.KernelParamWrapper(self.descriptors[name])
        else:
            assert self.cuda_descriptors[name] is not None
            return self.cuda_descriptors[name]


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


def matmul_tma_persistent_get_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "GROUP_SIZE_M": 8,
                "EPILOGUE_SUBTILE": SUBTILE,
            },
            num_stages=s,
            num_warps=w,
        )
        for BM in [128]
        for BN in [128, 256]
        for BK in [64, 128]
        for s in ([3, 4])
        for w in [4, 8]
        for SUBTILE in [True, False]
    ]


@triton.autotune(
    configs=matmul_tma_persistent_get_configs(),
    key=["M", "N", "K"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_tma_persistent(
    a_desc_ptr,
    b_desc_ptr,
    c_desc_ptr,  #
    M,
    N,
    K,  #
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    EPILOGUE_SUBTILE: tl.constexpr,  #
    NUM_SMS: tl.constexpr,
):  #
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = tl._experimental_descriptor_load(
                a_desc_ptr, [offs_am, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], dtype
            )
            b = tl._experimental_descriptor_load(
                b_desc_ptr, [offs_bn, offs_k], [BLOCK_SIZE_N, BLOCK_SIZE_K], dtype
            )
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(
            tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_am_c = pid_m * BLOCK_SIZE_M
        offs_bn_c = pid_n * BLOCK_SIZE_N

        # Epilogue subtiling is a technique to break our computation and stores into multiple pieces
        # By subtiling we can reduce shared memory consumption by the epilogue and instead use that
        # memory to increase our stage count.
        # In this case we partition the accumulator into 2 BLOCK_SIZE_M x BLOCK_SIZE_N // 2 tensors
        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            tl._experimental_descriptor_store(c_desc_ptr, c0, [offs_am_c, offs_bn_c])
            c1 = acc1.to(dtype)
            tl._experimental_descriptor_store(
                c_desc_ptr, c1, [offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2]
            )
        else:
            accumulator = accumulator.to(dtype)
            tl._experimental_descriptor_store(
                c_desc_ptr, accumulator, [offs_am_c, offs_bn_c]
            )


def matmul_tma_persistent(a, b, num_sms):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    desc_helper = TmaAutoTuneHelper()
    desc_helper.init_tma_descriptor("a")
    desc_helper.init_tma_descriptor("b")
    desc_helper.init_tma_descriptor("c")

    # NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    # print(f"NUM_SMS===================={NUM_SMS}")
    NUM_SMS = num_sms

    def grid(META):
        nonlocal desc_helper
        desc_helper.fill_2d_tma_descriptor(
            "a",
            a.data_ptr(),
            M,
            K,
            META["BLOCK_SIZE_M"],
            META["BLOCK_SIZE_K"],
            a.element_size(),
        )

        desc_helper.fill_2d_tma_descriptor(
            "b",
            b.data_ptr(),
            N,
            K,
            META["BLOCK_SIZE_N"],
            META["BLOCK_SIZE_K"],
            b.element_size(),
        )

        store_block_n = META["BLOCK_SIZE_N"]

        if META["EPILOGUE_SUBTILE"]:
            store_block_n = store_block_n // 2

        desc_helper.fill_2d_tma_descriptor(
            "c",
            c.data_ptr(),
            M,
            N,
            META["BLOCK_SIZE_M"],
            store_block_n,
            c.element_size(),
        )

        return (
            min(
                NUM_SMS,
                triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            ),
        )

    desc_a = desc_helper.get_tma_descriptor_kernel_param("a")
    desc_b = desc_helper.get_tma_descriptor_kernel_param("b")
    desc_c = desc_helper.get_tma_descriptor_kernel_param("c")

    matmul_kernel_tma_persistent[grid](
        desc_a,
        desc_b,
        desc_c,  #
        M,
        N,
        K,  #
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
        NUM_SMS=NUM_SMS,  #
    )
    return c


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
            (data_size, data_size), dtype=torch.float32, device=self.device
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
            (data_size, data_size), dtype=torch.float16, device=self.device
        )

    def work(self, data):
        # torch.matmul(data, data)
        # print(f'rank = {self.rank}, {data.shape}, {data.dtype}, {data.device}')
        matmul_tma_persistent(data, data, num_sms=78)
        # matmul(data, data)

    def print_profile(self):
        if self.rank == 0:
            elapsed_time = self.start.elapsed_time(self.end) / 1e3
            data_size = self.data.shape[0]
            flops = 2 * (data_size**3) * self.num_repeat
            tflops = flops / elapsed_time / (1024**4)  # TFlops
            print("===========GemmWorkload===========")
            print(f"Total time: {elapsed_time:.3f} seconds")
            print(f"Matrix size: {data_size}")
            print(f"Average TFlops: {tflops:.2f} Tflops")


def benchmark_all_reduce(
    backend="nccl", device="cuda", data_size=1024**3, num_iters=10
):
    # 初始化进程组
    dist.init_process_group(backend)
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
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
    print(f"total time = {elapsed_time}")

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
