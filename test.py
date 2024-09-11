import time
from collections import defaultdict
from multiprocessing import Manager, Process


class TreeNode:
    def __init__(self):
        self.children = defaultdict(TreeNode)
        self.parent = None
        self.key = None
        self.value = None
        self.lock_ref = 0
        self.last_access_time = time.time()

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def worker(shared_dict, gpu_id):
    # 模拟工作
    node = TreeNode()
    node.key = f"Key-{gpu_id}"
    node.value = f"Value-{gpu_id}"
    shared_dict[gpu_id] = node
    print(f"Process {gpu_id} updated the shared dictionary.")


if __name__ == "__main__":
    with Manager() as manager:
        shared_dict = manager.dict()
        processes = []
        for i in range(8):  # 假设有8个GPU
            p = Process(target=worker, args=(shared_dict, i))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        for key, node in shared_dict.items():
            print(f"GPU {key}: {node.key}, {node.value}")
