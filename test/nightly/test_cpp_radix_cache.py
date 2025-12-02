import unittest
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
<<<<<<<< HEAD:test/nightly/test_cpp_radix_cache.py
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
========
from sglang.test.few_shot_gsm8k import run_eval
>>>>>>>> origin:test/srt/models/test_glm4_moe_models.py
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=60, suite="nightly-1-gpu", nightly=True)

<<<<<<<< HEAD:test/nightly/test_cpp_radix_cache.py

class TestCppRadixCache(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        envs.SGLANG_EXPERIMENTAL_CPP_RADIX_TREE.set(True)
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
========
class TestGLM4MoE(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "zai-org/GLM-4.5-Air-FP8"
>>>>>>>> origin:test/srt/models/test_glm4_moe_models.py
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
<<<<<<<< HEAD:test/nightly/test_cpp_radix_cache.py
========
            other_args=[
                "--tp-size",
                "2",
            ],
>>>>>>>> origin:test/srt/models/test_glm4_moe_models.py
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=100,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
<<<<<<<< HEAD:test/nightly/test_cpp_radix_cache.py
        print(metrics)
        self.assertGreaterEqual(metrics["score"], 0.65)
========
        print(f"{metrics=}")
        self.assertGreater(metrics["accuracy"], 0.8)
>>>>>>>> origin:test/srt/models/test_glm4_moe_models.py


if __name__ == "__main__":
    unittest.main()
