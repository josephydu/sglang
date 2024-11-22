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

import random
import threading
import time

import logging
import multiprocessing as mp
from enum import Enum, auto
from collections import defaultdict

import zmq

from sglang.srt.managers.io_struct import (
    ControllerInfo,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.mem_cache.radix_cache import TreeNode
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    configure_logger,
    get_zmq_socket,
    kill_parent_process,
    suppress_other_loggers,
)

from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)

class TreeNodeSend:
    def __init__(self) -> None:
        self.children = defaultdict(TreeNodeSend)
        self.parent = None
        self.key = None
        self.gpu_id = None


class LoadBalanceMethod(Enum):
    """Load balance method."""

    ROUND_ROBIN = auto()
    SHORTEST_QUEUE = auto()
    RESOURCES_AWARE = auto()
    CACHE_AWARE = auto()

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
        self.recv_from_tokenizer = get_zmq_socket(
            self.context, zmq.PULL, port_args.scheduler_input_ipc_name
        )

        # Dispatch method
        self.round_robin_counter = 0
        dispatch_lookup = {
            LoadBalanceMethod.ROUND_ROBIN: self.round_robin_scheduler,
            LoadBalanceMethod.SHORTEST_QUEUE: self.shortest_queue_scheduler,
            LoadBalanceMethod.RESOURCES_AWARE: self.resources_aware_scheduler,
            LoadBalanceMethod.CACHE_AWARE: self.cache_aware_scheduler,
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

        # For cache_aware
        self.cache_aware = server_args.load_balance_method == LoadBalanceMethod.CACHE_AWARE

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

        if self.cache_aware:
            self.newest_tree_cache = {}
            self.recv_tree_cache_lock = threading.Lock()
            self.recv_tree_cache_thread = threading.Thread(
                target=self.loop_for_recv_tree_cache
            ).start()
        else:
            self.newest_tree_cache = None
            self.recv_tree_cache_thread = None

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

        send_to = get_zmq_socket(
            self.context, zmq.PUSH, port_args.scheduler_input_ipc_name
        )

        # Wait for model to finish loading
        for i in range(len(scheduler_pipe_readers)):
            scheduler_pipe_readers[i].recv()

        return send_to

    def loop_for_recv_tree_cache(self):
        while True:
            self.recv_tree_cache()

    def recv_tree_cache(self):
        recv_radix_cache = self.controller_info.radix_queue.get()
        if recv_radix_cache:
            gpu_id = recv_radix_cache.gpu_id
            with self.recv_tree_cache_lock:
                self.newest_tree_cache[gpu_id] = recv_radix_cache
            del recv_radix_cache

    def round_robin_scheduler(self, req):
        self.workers[self.round_robin_counter].send_pyobj(req)
        self.round_robin_counter = (self.round_robin_counter + 1) % len(self.workers)

    # get newest gpu info from nodes before scheduler request
    def update_memory_and_requests_info(self):
        # pre_xx is used to check whether nodes have updated their relevant information
        # main_xx is used in scheduler methods so that the infomation we used is the newest
        
        available_mem = [k.value for k in self.controller_info.available_kv_cache]
        num_reqs_running = [k.value for k in self.controller_info.running_reqs]
        num_reqs_waiting = [k.value for k in self.controller_info.waiting_reqs]

        def update_cache(pre_cache, main_cache, new_data):
            if not pre_cache:
                pre_cache.extend(new_data)
                main_cache.extend(new_data)
            else:
                for i, (pre, new) in enumerate(zip(pre_cache, new_data)):
                    if pre != new:
                        pre_cache[i] = new
                        main_cache[i] = new

        update_cache(self.pre_available_kv_cache if hasattr(self, 'pre_available_kv_cache') else [],
                    self.main_available_kv_cache if hasattr(self, 'main_available_kv_cache') else [],
                    available_mem)

        update_cache(self.pre_num_running_req if hasattr(self, 'pre_num_running_req') else [],
                    self.main_num_running_req if hasattr(self, 'main_num_running_req') else [],
                    num_reqs_running)

        update_cache(self.pre_num_waiting_req if hasattr(self, 'pre_num_waiting_req') else [],
                    self.main_num_waiting_req if hasattr(self, 'main_num_waiting_req') else [],
                    num_reqs_waiting)

    # get specific gpu_idx by gpu info
    def allocate_gpu(self, all_waiting, no_waiting):
        if all_waiting:
            # if all nodes are waiting, we select the gpu from the node with the smallest number of nodes
            min_num_waitng = min(self.main_num_waiting_req)
            # if there are multi nodes that meet the criteria, we random choose them.
            indices = [i for i, x in enumerate(self.main_num_waiting_req) if x == min_num_waitng]
            gpu_idx = random.choice(indices)
        else:
            # else we select the gpu from no wait nodes with the most gpu memory
            # firstly, remove the waiting queue. 
            mems = [mem if no_wait == 1 else 0 for mem, no_wait in zip(self.main_available_kv_cache, no_waiting)]
            max_value = max(mems)
            max_indices = [
                index for index, value in enumerate(mems) if value == max_value
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
        else:
            self.main_num_running_req[gpu_idx] += 1
        self.workers[gpu_idx].send_pyobj(req)

    def cache_aware_scheduler(self, req):
        match_lens = [0] * self.dp_size
        req_lens = [len(req.input_ids)] * self.dp_size

        with self.recv_tree_cache_lock:
            for gpu_id, radix_cache in self.newest_tree_cache.items():
                pre_len = get_match_len(radix_cache, req.input_ids, 0)
                match_lens[gpu_id] = int(pre_len)

        # NOTE: 100 is used to reduce the influence of random input
        # e.g. If the match nums is [1, 2, 0, 0, 0, 0], we think the scheduer method should be resources aware
        self.update_memory_and_requests_info()
        all_waiting = min(self.main_num_waiting_req) > 0
        no_waiting = [1 if waiting <= 0 else 0 for waiting in self.main_num_waiting_req]
        
        # occipuied_lens means that the number of tokens this request will occupy
        occipuied_lens = [(req_len - prefix_len + req.sampling_params.max_new_tokens * 0.5) for req_len, prefix_len in zip(req_lens, prefix_lens)]
        
        if max(match_lens) <= 100 or all_waiting:
            # this is the logic of resources_aware
            gpu_idx = self.allocate_gpu(req, all_waiting, no_waiting)
            self.main_available_kv_cache[gpu_idx] = self.main_available_kv_cache[gpu_idx] - occipuied_lens[gpu_idx]
            if all_waiting:
                self.main_num_waiting_req[gpu_idx] += 1
            else:
                self.main_num_running_req[gpu_idx] += 1
            self.workers[gpu_idx].send_pyobj(req)
        else:
            # We get the matching length of the no waiting nodes, then we get the max match nodes
            pre_lens = [pre if no_wait == 1 else 0 for pre, no_wait in zip(match_lens, no_waiting)]
            max_value = max(pre_lens)
            max_indices = [
                index for index, value in enumerate(pre_lens) if value == max_value
            ]
            gpu_idx = random.choice(max_indices)
            
            self.main_available_kv_cache[gpu_idx] = self.main_available_kv_cache[gpu_idx] - occipuied_lens[gpu_idx]
            self.main_num_running_req[gpu_idx] += 1
            self.workers[gpu_idx].send_pyobj(req)

    def shortest_queue_scheduler(self, input_requests):
        raise NotImplementedError()

    def event_loop(self):
        while True:
            while True:
                try:
                    recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break

                # logger.info(f"[event_loop]{type(recv_req)}")
                if isinstance(
                    recv_req,
                    (
                        TokenizedGenerateReqInput,
                        TokenizedEmbeddingReqInput,
                    ),
                ):
                    self.dispatching(recv_req)
                else:
                    # Send other control messages to all workers
                    for worker in self.workers:
                        worker.send_pyobj(recv_req)


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


# for cache aware scheduler
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
