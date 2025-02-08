import copy
import os
from typing import Any, Dict, Union

from transformers import PretrainedConfig, Qwen2Config


class Qwen2VLVisionConfig(PretrainedConfig):
    model_type = "qwen2_vl"

    def __init__(
        self,
        depth=32,
        embed_dim=1280,
        hidden_size=3584,
        hidden_act="quick_gelu",
        mlp_ratio=4,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )

        if config_dict.get("model_type") == "qwen2_vl":
            config_dict = config_dict["vision_config"]

        return cls.from_dict(config_dict, **kwargs)


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
        elif vision_config is None:
            self.vision_config = Qwen2VLVisionConfig()

        # =========== Adapt for WePoints-Sglang

        # if isinstance(llm_config, dict):
        #     self.llm_config = Qwen2Config(**llm_config)
        # else:
        #     self.llm_config = llm_config

        print(f"[POINTSV15ChatConfig]===========>\n{llm_config}")
        self.vocab_size = llm_config["vocab_size"]
        self.max_position_embeddings = llm_config["max_position_embeddings"]
        self.hidden_size = llm_config["hidden_size"]
        self.intermediate_size = llm_config["intermediate_size"]
        self.num_hidden_layers = llm_config["num_hidden_layers"]
        self.num_attention_heads = llm_config["num_attention_heads"]
        self.use_sliding_window = llm_config["use_sliding_window"]
        self.sliding_window = llm_config["sliding_window"]
        self.max_window_layers = llm_config["max_window_layers"]

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = llm_config["num_attention_heads"]

        self.num_key_value_heads = llm_config["num_key_value_heads"]
        self.hidden_act = llm_config["hidden_act"]
        self.initializer_range = llm_config["initializer_range"]
        self.rms_norm_eps = llm_config["rms_norm_eps"]
        self.use_cache = llm_config["use_cache"]
        self.attention_dropout = llm_config["attention_dropout"]
        self.rope_scaling = llm_config["rope_scaling"]

        if self.rope_scaling is not None and "type" in self.rope_scaling:
            if self.rope_scaling["type"] == "mrope":
                self.rope_scaling["type"] = "default"
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        super().__init__(
            tie_word_embeddings=llm_config["tie_word_embeddings"], **kwargs
        )
