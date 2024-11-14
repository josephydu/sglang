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
import multiprocessing.connection
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

import queue
import random
import time


# for pre radix scheduler
def _key_match(key0, key1):
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i


def get_match_len(node, key, match_length: int) -> int:
    if len(key) == 0:
        return match_length

    if key[0] in node.children.keys():
        child = node.children[key[0]]
        prefix_len = _key_match(child.key, key)
        match_length += prefix_len
        if prefix_len < len(child.key):
            return match_length
        else:
            return get_match_len(child, key[prefix_len:], match_length)
    else:
        return match_length


class LoadBalanceMethod(Enum):
    """Load balance method."""

    ROUND_ROBIN = auto()
    SHORTEST_QUEUE = auto()
    RESOURCES_AWARE = auto()
    PRE_RADIX = auto()
    POWER_OF_2_CHOICE = auto()

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
            LoadBalanceMethod.POWER_OF_2_CHOICE: self.power_of_2_choice_scheduler,
        }
        self.dispatching = dispatch_lookup[self.load_balance_method]

        # For resources aware
        self.dp_size = server_args.dp_size
        self.controller_info = ControllerInfo(server_args.dp_size)
        self.pre_available_kv_cache = []
        self.main_available_kv_cache = []
        
        self.pre_evictable_kv_cache = []
        self.main_evictable_kv_cache = []

        self.pre_num_running_req = []
        self.main_num_running_req = []

        self.pre_num_waiting_req = []
        self.main_num_waiting_req = []

        # For pre_radix
        self.zmq_raidx = server_args.load_balance_method == "pre_radix"

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

        if self.zmq_raidx:
            import threading

            self.newest_tree_cache = {}

            self.recv_tree_cache_lock = threading.Lock()
            self.recv_tree_cache_thread = threading.Thread(
                target=self.loop_for_recv_tree_cache
            )
        else:
            self.newest_tree_cache = None
            self.recv_tree_cache_thread = None
            
        self.loop_time = 0

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

    def loop_for_recv_tree_cache(self):
        # self.cnt = 0
        while True:
            self.recv_tree_cache()

    def recv_tree_cache(self):
        recv_radix_cache = self.controller_info.radix_queue.get()
        if recv_radix_cache:
            # logger.info('[recv_tree_cache] receive new data')
            gpu_id = recv_radix_cache.gpu_id
            if (
                gpu_id not in self.newest_tree_cache
                or recv_radix_cache.time > self.newest_tree_cache[gpu_id].time
            ):
                with self.recv_tree_cache_lock:
                    if gpu_id in self.newest_tree_cache:
                        del self.newest_tree_cache[gpu_id]
                    self.newest_tree_cache[gpu_id] = recv_radix_cache
            del recv_radix_cache

        # if self.cnt % 100 == 0:
        #     t2 = time.time()
        #     logger.info(f"[loop_for_recv_tree_cache]time={t2 - t1:.8f}")
        #     self.cnt += 1

    # 比较两个worker的指标
    def compare_metrics(self, ins1, ins2):
        if self.main_num_waiting_req[ins1] != self.main_num_waiting_req[ins2]:
            return (
                ins1
                if self.main_num_waiting_req[ins1] < self.main_num_waiting_req[ins2]
                else ins2
            )
        if self.main_num_running_req[ins1] != self.main_num_running_req[ins2]:
            return (
                ins1
                if self.main_num_running_req[ins1] < self.main_num_running_req[ins2]
                else ins2
            )
        if self.main_available_kv_cache[ins1] != self.main_available_kv_cache[ins2]:
            return (
                ins1
                if self.main_available_kv_cache[ins1]
                > self.main_available_kv_cache[ins2]
                else ins2
            )
        return ins1

    def power_of_2_choice_scheduler(self, req):
        self.update_memory_and_requests()
        instances_len = len(self.workers)

        ins1, ins2 = random.sample(range(0, instances_len), 2)
        ins_end = self.compare_metrics(ins1, ins2)
        self.main_available_kv_cache[ins_end] -= len(req.input_ids)
        self.main_num_running_req[ins_end] += 1
        self.workers[ins_end].send_pyobj(req)

    def round_robin_scheduler(self, req):
        self.workers[self.round_robin_counter].send_pyobj(req)
        self.round_robin_counter = (self.round_robin_counter + 1) % len(self.workers)
    def update_memory_and_requests(self):
        # 从控制器获取最新的内存和请求状态
        evictable_mem = [k.value for k in self.controller_info.evictable_kv_cache]
        available_mem = [k.value for k in self.controller_info.available_kv_cache]
        num_reqs_running = [k.value for k in self.controller_info.running_reqs]
        num_reqs_waiting = [k.value for k in self.controller_info.waiting_reqs]

        # 初始化或更新缓存
        def update_cache(pre_cache, main_cache, new_data):
            if not pre_cache:
                pre_cache.extend(new_data)
                main_cache.extend(new_data)
            else:
                for i, (pre, new) in enumerate(zip(pre_cache, new_data)):
                    if pre != new:
                        pre_cache[i] = new
                        main_cache[i] = new

        # 更新各种缓存
        update_cache(self.pre_available_kv_cache if hasattr(self, 'pre_available_kv_cache') else [],
                    self.main_available_kv_cache if hasattr(self, 'main_available_kv_cache') else [],
                    available_mem)

        update_cache(self.pre_evictable_kv_cache if hasattr(self, 'pre_evictable_kv_cache') else [],
                    self.main_evictable_kv_cache if hasattr(self, 'main_evictable_kv_cache') else [],
                    evictable_mem)

        update_cache(self.pre_num_running_req if hasattr(self, 'pre_num_running_req') else [],
                    self.main_num_running_req if hasattr(self, 'main_num_running_req') else [],
                    num_reqs_running)

        update_cache(self.pre_num_waiting_req if hasattr(self, 'pre_num_waiting_req') else [],
                    self.main_num_waiting_req if hasattr(self, 'main_num_waiting_req') else [],
                    num_reqs_waiting)


    def allocate_gpu(self, req, all_waiting, no_waiting):


        if all_waiting:
            ratio = [
                run / wait
                for run, wait in zip(
                    self.main_num_running_req, self.main_num_waiting_req
                )
            ]
            max_ratio = max(ratio)
            indices = [i for i, x in enumerate(ratio) if x == max_ratio]
            gpu_idx = random.choice(indices)
        else:
            filter_result = [
                a * b for a, b in zip(no_waiting, self.main_available_kv_cache)
            ]
            max_value = max(filter_result)
            max_indices = [
                index for index, value in enumerate(filter_result) if value == max_value
            ]
            gpu_idx = random.choice(max_indices)

        return gpu_idx

    def resources_aware_scheduler(self, req):
        self.update_memory_and_requests()
        all_waiting = min(self.main_num_waiting_req) > 0
        no_waiting = [1 if waiting == 0 else 0 for waiting in self.main_num_waiting_req]
        gpu_idx = self.allocate_gpu(req, all_waiting, no_waiting)
        self.main_available_kv_cache[gpu_idx] = self.main_available_kv_cache[gpu_idx] - len(req.input_ids)
        if all_waiting:
            self.main_num_waiting_req[gpu_idx] += 1
        self.workers[gpu_idx].send_pyobj(req)

    def pre_radix_scheduler(self, req):
        prefix_lens = [0] * self.dp_size
        req_lens = [len(req.input_ids)] * self.dp_size

        with self.recv_tree_cache_lock:
            for gpu_id, radix_cache in self.newest_tree_cache.items():
                pre_len = get_match_len(radix_cache.root_node, req.input_ids, 0)
                prefix_lens[gpu_id] = pre_len
        # NOTE: 100 is used to reduce the influence of random input
        # e.g. If the match nums is [1, 2, 0, 0, 0, 0], we think the scheduer method should be resources aware
        occipuied_lens = [(req_len - prefix_len) for req_len, prefix_len in zip(req_lens, prefix_lens)]
        logger.info(f'[occipuied_lens]{occipuied_lens}')
        
        logger.info(f'[before update]{self.main_available_kv_cache}')
        self.update_memory_and_requests()
        logger.info(f'[after update]{self.main_available_kv_cache}')
        all_waiting = min(self.main_num_waiting_req) > 0
        no_waiting = [1 if waiting == 0 else 0 for waiting in self.main_num_waiting_req]
        if max(prefix_lens) <= 100 or all_waiting:
            gpu_idx = self.allocate_gpu(req, all_waiting, no_waiting)
            
            logger.info(f'[before minus]{self.main_available_kv_cache}')
            self.main_available_kv_cache[gpu_idx] = self.main_available_kv_cache[gpu_idx] - occipuied_lens[gpu_idx]
            logger.info(f'[after minus]{self.main_available_kv_cache}')
            if all_waiting:
                self.main_num_waiting_req[gpu_idx] += 1
        else:
            forward_mems = [(availiable - occipuied - evictbale) if no_wait == 1 else (-100000) for availiable, occipuied, no_wait, evictbale in zip(self.main_available_kv_cache, occipuied_lens, no_waiting, self.main_evictable_kv_cache)]
            gpu_idx = forward_mems.index(max(forward_mems))
            logger.info(f'[before minus]{self.main_available_kv_cache}')
            self.main_available_kv_cache[gpu_idx] = self.main_available_kv_cache[gpu_idx] - occipuied_lens[gpu_idx]
            logger.info(f'[after minus]{self.main_available_kv_cache}')
        logger.info(f'[request_id]{sum(req.input_ids[:1000])} go to => [gpu_idx]{gpu_idx}')
        self.workers[gpu_idx].send_pyobj(req)

    def shortest_queue_scheduler(self, input_requests):
        raise NotImplementedError()

    def event_loop(self):
        while True:
            while True:
                # logger.info("1")
                try:
                    recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                # logger.info("2")
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
                # logger.info("3")


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
        if controller.recv_tree_cache_thread:
            controller.recv_tree_cache_thread.start()
        controller.event_loop()
    except Exception:
        msg = get_exception_traceback()
        logger.error(msg)
        kill_parent_process()
