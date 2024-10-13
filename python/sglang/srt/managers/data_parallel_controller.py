"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""A controller that dispatches requests to multiple data parallel workers."""

import logging
import multiprocessing as mp
from enum import Enum, auto

import zmq

from sglang.srt.managers.io_struct import (
    ControllerInfo,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    TokenizedRewardReqInput,
)
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    configure_logger,
    kill_parent_process,
    suppress_other_loggers,
)
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)

import random


class LoadBalanceMethod(Enum):
    """Load balance method."""

    ROUND_ROBIN = auto()
    SHORTEST_QUEUE = auto()
    RESOURCES_AWARE = auto()
    PRE_RADIX = auto()
    BUCKET = auto()

    @classmethod
    def from_str(cls, method: str):
        method = method.upper()
        try:
            return cls[method]
        except KeyError as exc:
            raise ValueError(f"Invalid load balance method: {method}") from exc


class DataParallelController:
    """A controller that dispatches requests to multiple data parallel workers."""

    def __init__(self, server_args, port_args) -> None:
        # Parse args
        self.server_args = server_args
        self.port_args = port_args
        self.load_balance_method = LoadBalanceMethod.from_str(
            server_args.load_balance_method
        )

        # Init inter-process communication
        self.context = zmq.Context(1 + server_args.dp_size)
        self.recv_from_tokenizer = self.context.socket(zmq.PULL)
        self.recv_from_tokenizer.bind(f"ipc://{port_args.scheduler_input_ipc_name}")

        # Dispatch method
        self.round_robin_counter = 0
        dispatch_lookup = {
            LoadBalanceMethod.ROUND_ROBIN: self.round_robin_scheduler,
            LoadBalanceMethod.SHORTEST_QUEUE: self.shortest_queue_scheduler,
            LoadBalanceMethod.RESOURCES_AWARE: self.resources_aware_scheduler,
            LoadBalanceMethod.PRE_RADIX: self.pre_radix_scheduler,
            LoadBalanceMethod.BUCKET: self.bucket_scheduler,
        }
        self.dispatching = dispatch_lookup[self.load_balance_method]

        # For resources aware
        self.dp_size = server_args.dp_size
        self.controller_info = ControllerInfo(server_args.dp_size)
        self.pre_available_kv_cache = []
        self.main_available_kv_cache = []

        self.pre_num_running_req = []
        self.main_num_running_req = []

        self.pre_num_waiting_req = []
        self.main_num_waiting_req = []
        
        # For pre_radix
        self.choosen_gpu_per_req = {}

        # Start data parallel workers
        base_gpu_id = 0
        self.workers = []
        for dp_rank in range(server_args.dp_size):
            tmp_port_args = PortArgs.init_new(server_args)
            tmp_port_args.detokenizer_ipc_name = port_args.detokenizer_ipc_name

            send_to = self.launch_tensor_parallel_group(
                server_args, tmp_port_args, base_gpu_id, dp_rank, self.controller_info
            )

            self.workers.append(send_to)
            base_gpu_id += server_args.tp_size

    def launch_tensor_parallel_group(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: int,
        controller_info: ControllerInfo,
    ):
        # Launch tensor parallel scheduler processes
        scheduler_procs = []
        scheduler_pipe_readers = []
        tp_size_per_node = server_args.tp_size // server_args.nnodes
        tp_rank_range = range(
            tp_size_per_node * server_args.node_rank,
            tp_size_per_node * (server_args.node_rank + 1),
        )
        for tp_rank in tp_rank_range:
            reader, writer = mp.Pipe(duplex=False)
            gpu_id = base_gpu_id + tp_rank % tp_size_per_node
            proc = mp.Process(
                target=run_scheduler_process,
                args=(
                    server_args,
                    port_args,
                    gpu_id,
                    tp_rank,
                    dp_rank,
                    writer,
                    controller_info,
                ),
            )
            proc.start()
            scheduler_procs.append(proc)
            scheduler_pipe_readers.append(reader)

        send_to = self.context.socket(zmq.PUSH)
        send_to.connect(f"ipc://{port_args.scheduler_input_ipc_name}")

        # Wait for model to finish loading
        for i in range(len(scheduler_pipe_readers)):
            scheduler_pipe_readers[i].recv()

        return send_to

    def round_robin_scheduler(self, req):
        self.workers[self.round_robin_counter].send_pyobj(req)
        self.round_robin_counter = (self.round_robin_counter + 1) % len(self.workers)

    def resources_aware_scheduler(self, req):
        available_mem = [k.value for k in self.controller_info.available_kv_cache]
        num_reqs_running = [k.value for k in self.controller_info.running_reqs]
        num_reqs_waiting = [k.value for k in self.controller_info.waiting_reqs]
        if not self.pre_available_kv_cache:
            self.pre_available_kv_cache = available_mem.copy()

        if not self.main_available_kv_cache:
            self.main_available_kv_cache = available_mem.copy()

        if self.pre_available_kv_cache == available_mem:
            # 使用备份的available_mem
            pass
        else:
            # logger.info(
            #     f"update main_available_kv_cache: main{self.main_available_kv_cache}=>pre{self.pre_available_kv_cache}=>now{available_mem}"
            # )
            self.pre_available_kv_cache = available_mem.copy()
            self.main_available_kv_cache = available_mem.copy()
        # ===============================================================================
        if not self.pre_num_running_req:
            self.pre_num_running_req = num_reqs_running.copy()

        if not self.main_num_running_req:
            self.main_num_running_req = num_reqs_running.copy()

        if self.pre_num_running_req == num_reqs_running:
            # use_num_reqs_running = self.main_available_kv_cache
            pass
        else:
            # logger.info(
            #     f"update main_num_running_req: main{self.main_num_running_req}=>pre{self.pre_num_running_req}=>now{num_reqs_running}"
            # )
            self.main_num_running_req = num_reqs_running.copy()
            self.pre_num_running_req = num_reqs_running.copy()

        # =================================================================================
        if not self.pre_num_waiting_req:
            self.pre_num_waiting_req = num_reqs_waiting.copy()

        if not self.main_num_waiting_req:
            self.main_num_waiting_req = num_reqs_waiting.copy()

        if self.pre_num_waiting_req == num_reqs_waiting:
            # use_num_reqs_running = self.main_available_kv_cache
            pass
        else:
            # logger.info(
            #     f"update main_num_waiting_req: main{self.main_num_waiting_req}=>pre{self.pre_num_waiting_req}=>now{num_reqs_waiting}"
            # )
            self.main_num_waiting_req = num_reqs_waiting.copy()
            self.pre_num_waiting_req = num_reqs_waiting.copy()

        all_waitting = False
        if min(self.main_num_waiting_req) > 0:
            # 最小值都大于0，全部waiting
            all_waitting = True
        else:
            # 最小值都是0， 则全部waiting
            all_waitting = False

        no_waiting = [1 if waiting == 0 else 0 for waiting in self.main_num_waiting_req]
        if all_waitting:
            ratio = [
                run / wait
                for run, wait in zip(
                    self.main_num_running_req, self.main_num_waiting_req
                )
            ]
            max_raio = max(ratio)
            indices = [i for i, x in enumerate(ratio) if x == max_raio]
            gpu_idx = random.choice(indices)
            self.main_num_waiting_req[gpu_idx] += 1
            self.main_available_kv_cache[gpu_idx] -= len(req.input_ids)
        else:
            filter_result = [
                a * b for a, b in zip(no_waiting, self.main_available_kv_cache)
            ]
            # 找到最大值
            max_value = max(filter_result)

            # 找到所有最大值的索引
            max_indices = [
                index for index, value in enumerate(filter_result) if value == max_value
            ]

            # 随机选择一个索引
            gpu_idx = random.choice(max_indices)
            self.main_available_kv_cache[gpu_idx] -= len(req.input_ids)
        self.workers[gpu_idx].send_pyobj(req)


    def pre_radix_scheduler(self, req):
        available_mem = [k.value for k in self.controller_info.available_kv_cache]
        num_reqs_running = [k.value for k in self.controller_info.running_reqs]
        num_reqs_waiting = [k.value for k in self.controller_info.waiting_reqs]
        if not self.pre_available_kv_cache:
            self.pre_available_kv_cache = available_mem.copy()

        if not self.main_available_kv_cache:
            self.main_available_kv_cache = available_mem.copy()

        if self.pre_available_kv_cache == available_mem:
            # 使用备份的available_mem
            pass
        else:
            # logger.info(
            #     f"update main_available_kv_cache: main{self.main_available_kv_cache}=>pre{self.pre_available_kv_cache}=>now{available_mem}"
            # )
            self.pre_available_kv_cache = available_mem.copy()
            self.main_available_kv_cache = available_mem.copy()
        # ===============================================================================
        if not self.pre_num_running_req:
            self.pre_num_running_req = num_reqs_running.copy()

        if not self.main_num_running_req:
            self.main_num_running_req = num_reqs_running.copy()

        if self.pre_num_running_req == num_reqs_running:
            # use_num_reqs_running = self.main_available_kv_cache
            pass
        else:
            # logger.info(
            #     f"update main_num_running_req: main{self.main_num_running_req}=>pre{self.pre_num_running_req}=>now{num_reqs_running}"
            # )
            self.main_num_running_req = num_reqs_running.copy()
            self.pre_num_running_req = num_reqs_running.copy()

        # =================================================================================
        if not self.pre_num_waiting_req:
            self.pre_num_waiting_req = num_reqs_waiting.copy()

        if not self.main_num_waiting_req:
            self.main_num_waiting_req = num_reqs_waiting.copy()

        if self.pre_num_waiting_req == num_reqs_waiting:
            # use_num_reqs_running = self.main_available_kv_cache
            pass
        else:
            # logger.info(
            #     f"update main_num_waiting_req: main{self.main_num_waiting_req}=>pre{self.pre_num_waiting_req}=>now{num_reqs_waiting}"
            # )
            self.main_num_waiting_req = num_reqs_waiting.copy()
            self.pre_num_waiting_req = num_reqs_waiting.copy()

        all_waitting = False
        if min(self.main_num_waiting_req) > 0:
            # 最小值都大于0，全部waiting
            all_waitting = True
        else:
            # 最小值都是0， 则全部waiting
            all_waitting = False

        no_waiting = [1 if waiting == 0 else 0 for waiting in self.main_num_waiting_req]
        len_r = len(req.input_ids)
        if len_r < 10:
            rid = 0
            for i in range(10):
                rid += req.input_ids[i % len_r] 
        else:
            rid = sum(req.input_ids[:10])

        if rid not in self.choosen_gpu_per_req:
            if all_waitting:
                ratio = [
                    run / wait
                    for run, wait in zip(
                        self.main_num_running_req, self.main_num_waiting_req
                    )
                ]
                max_raio = max(ratio)
                indices = [i for i, x in enumerate(ratio) if x == max_raio]
                gpu_idx = random.choice(indices)
                self.main_num_waiting_req[gpu_idx] += 1
                self.main_available_kv_cache[gpu_idx] -= len(req.input_ids)
            else:
                filter_result = [
                    a * b for a, b in zip(no_waiting, self.main_available_kv_cache)
                ]
                # 找到最大值
                max_value = max(filter_result)

                # 找到所有最大值的索引
                max_indices = [
                    index
                    for index, value in enumerate(filter_result)
                    if value == max_value
                ]

                # 随机选择一个索引
                gpu_idx = random.choice(max_indices)

                self.main_available_kv_cache[gpu_idx] -= len(req.input_ids)
            self.choosen_gpu_per_req[rid] = gpu_idx
        else:
            gpu_idx = self.choosen_gpu_per_req[rid]
        self.workers[gpu_idx].send_pyobj(req)

    def shortest_queue_scheduler(self, input_requests):
        raise NotImplementedError()

    def bucket_scheduler(self, req):
        gpu_idx = req.input_ids[0] % self.dp_size

        self.workers[gpu_idx].send_pyobj(req)
    def event_loop(self):
        while True:
            while True:
                try:
                    recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break

                if isinstance(
                    recv_req,
                    (
                        TokenizedGenerateReqInput,
                        TokenizedEmbeddingReqInput,
                        TokenizedRewardReqInput,
                    ),
                ):
                    self.dispatching(recv_req)
                else:
                    # Send other control messages to all workers
                    for worker in self.workers:
                        worker.queue.put(recv_req)


def run_data_parallel_controller_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
):
    configure_logger(server_args)
    suppress_other_loggers()

    try:
        controller = DataParallelController(server_args, port_args)
        pipe_writer.send("ready")
        controller.event_loop()
    except Exception:
        msg = get_exception_traceback()
        logger.error(msg)
        kill_parent_process()
