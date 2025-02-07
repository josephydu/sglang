import copy
from typing import Any, Dict

from transformers import PretrainedConfig

try:
    from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig
except ImportError:
    print("Please upgrade transformers to version 4.46.3 or higher")

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class POINTSV15ChatConfig(PretrainedConfig):
    model_type = "pointsv1.5_chat"
    is_composition = True
    """Configuration class for `POINTSV1.5`."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        vision_config = kwargs.pop("vision_config", None)
        llm_config = kwargs.pop("llm_config", None)
        if isinstance(vision_config, dict):
            self.vision_config = Qwen2VLVisionConfig(**vision_config)
        else:
            self.vision_config = vision_config

        # change it to adapter sglang
        # if isinstance(llm_config, dict):
        #     self.llm_config = CustomLlamaConfig(**llm_config)
        # else:
        #     self.llm_config = llm_config
        # print(f"[POINTSV15ChatConfig] => llm_config{llm_config}")
        super().__init__(
            bos_token_id=llm_config["bos_token_id"],
            eos_token_id=llm_config["eos_token_id"],
            **kwargs,
        )
        self.vocab_size = llm_config["vocab_size"]
        self.max_position_embeddings = llm_config["max_position_embeddings"]
        self.hidden_size = llm_config["hidden_size"]
        self.num_layers = llm_config["num_layers"]
        self.num_attention_heads = llm_config["num_attention_heads"]
        self.num_kv_heads = llm_config["num_kv_heads"]
        self.ffn_hidden_size = llm_config["ffn_hidden_size"]
        self.hidden_act = llm_config["hidden_act"]
        self.rotary_pct = llm_config["rotary_pct"]
        self.rotary_emb_base = llm_config["rotary_emb_base"]
        self.rotary_compress = llm_config["rotary_compress"]
        self.initializer_range = llm_config["initializer_range"]
        self.layernorm_epsilon = llm_config["layernorm_epsilon"]
        self.use_cache = llm_config["use_cache"]
        if llm_config.get("rms_norm", None) is not None:
            self.norm_type = "rms_norm" if llm_config["rms_norm"] else "layer_norm"
        else:
            self.norm_type = llm_config["norm_type"]
        self.qkv_proj_bias = llm_config["qkv_proj_bias"]
        self.out_proj_bias = llm_config["out_proj_bias"]
        self.mlp_fc1_bias = llm_config["mlp_fc1_bias"]
        self.mlp_fc2_bias = llm_config["mlp_fc2_bias"]
        self.num_hidden_layers = llm_config["num_layers"]

    # def to_dict(self) -> Dict[str, Any]:
    #     output = copy.deepcopy(self.__dict__)
    #     output["vision_config"] = self.vision_config.to_dict()
    #     output["llm_config"] = self.llm_config.to_dict()
    #     return output
