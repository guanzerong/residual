from __future__ import annotations

import copy
import json
import logging
import math
import os
from collections import deque
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from resfit.lerobot.policies.pi05._runtime import ensure_pi05_dependencies

ensure_pi05_dependencies()

from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, GemmaTokenizer
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma
from transformers.models.gemma.modeling_gemma import GemmaForCausalLM
from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration

from resfit.lerobot.policies.pi05.configuration_pi05 import PI05Config
from resfit.lerobot.policies.pretrained import PreTrainedPolicy

ACTION = "action"
OBS_STATE = "observation.state"
OBS_LANGUAGE_TOKENS = "observation.language.tokens"
OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"
OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38

_DATASET_STATS_FILENAME = "resfit_dataset_stats.pt"
_POLICY_METADATA_FILENAME = "resfit_pi05_policy.json"


def _clone_cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def get_safe_dtype(target_dtype, device_type):
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def _load_pi05_tokenizer(config: "PI05Config", processor_name_or_path: str | Path | None = None):
    tokenizer_name_or_path = os.environ.get("RESFIT_PI05_TOKENIZER")
    if tokenizer_name_or_path is None:
        tokenizer_name_or_path = str(processor_name_or_path) if processor_name_or_path is not None else None
    if tokenizer_name_or_path is None:
        tokenizer_name_or_path = config.tokenizer_name_or_path
    try:
        return AutoProcessor.from_pretrained(tokenizer_name_or_path).tokenizer
    except Exception as processor_exc:
        error_text = str(processor_exc).lower()
        if "gated repo" in error_text or "403" in error_text or "authorized list" in error_text:
            raise RuntimeError(
                "Failed to load the PI05 tokenizer from "
                f"'{tokenizer_name_or_path}'. The default source is a gated Hugging Face repo. "
                "Either request access and run `huggingface-cli login`, or point "
                "`tokenizer_name_or_path` (for example via `--policy_kwargs`) or the "
                "`RESFIT_PI05_TOKENIZER` environment variable to a local downloaded copy."
            ) from processor_exc
        try:
            return GemmaTokenizer.from_pretrained(tokenizer_name_or_path)
        except Exception as tokenizer_exc:
            tokenizer_error_text = str(tokenizer_exc).lower()
            if "sentencepiece" in tokenizer_error_text:
                raise RuntimeError(
                    "Failed to load the PI05 tokenizer because `sentencepiece` is missing. "
                    "Install `sentencepiece` in the pi05 runtime or point "
                    "`tokenizer_name_or_path` / `RESFIT_PI05_TOKENIZER` to a locally prepared tokenizer."
                ) from tokenizer_exc
            raise RuntimeError(
                f"Failed to load the PI05 tokenizer from '{tokenizer_name_or_path}'. "
                "Tried both AutoProcessor and GemmaTokenizer loading paths."
            ) from tokenizer_exc


def _extract_tokenizer_name_from_policy_preprocessor(preprocessor_data: dict[str, Any]) -> str | None:
    for step in preprocessor_data.get("steps", []):
        if step.get("registry_name") != "tokenizer_processor":
            continue
        step_config = step.get("config", {})
        tokenizer_name = step_config.get("tokenizer_name")
        if isinstance(tokenizer_name, str) and tokenizer_name:
            return tokenizer_name
    return None


def _resolve_pi05_tokenizer_source(
    *,
    config: "PI05Config",
    pretrained_name_or_path: str | Path,
    explicit_source: str | Path | None = None,
) -> str | Path:
    if os.environ.get("RESFIT_PI05_TOKENIZER") is not None:
        return os.environ["RESFIT_PI05_TOKENIZER"]
    if explicit_source is not None:
        return explicit_source

    policy_path = Path(pretrained_name_or_path)
    if policy_path.is_dir():
        if (policy_path / "tokenizer.model").exists():
            return policy_path
        preprocessor_path = policy_path / "policy_preprocessor.json"
        if preprocessor_path.exists():
            with preprocessor_path.open() as f:
                tokenizer_name = _extract_tokenizer_name_from_policy_preprocessor(json.load(f))
            if tokenizer_name is not None:
                return tokenizer_name
    else:
        try:
            preprocessor_path = hf_hub_download(str(pretrained_name_or_path), "policy_preprocessor.json")
            with open(preprocessor_path) as f:
                tokenizer_name = _extract_tokenizer_name_from_policy_preprocessor(json.load(f))
            if tokenizer_name is not None:
                return tokenizer_name
        except Exception:
            pass

    return config.tokenizer_name_or_path


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device="cpu",
) -> Tensor:
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size,)`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector, new_dim):
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def resize_with_pad_torch(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    if images.shape[-1] <= 4:
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = images.permute(0, 3, 1, 2)
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)

    batch_size, channels, cur_height, cur_width = images.shape
    del batch_size, channels

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w
    constant_value = 0 if images.dtype == torch.uint8 else -1.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),
        mode="constant",
        value=constant_value,
    )

    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)

    return padded_images


def compute_layer_complete(
    layer_idx,
    inputs_embeds,
    attention_mask,
    position_ids,
    adarms_cond,
    paligemma,
    gemma_expert,
):
    models = [paligemma.language_model, gemma_expert.model]
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)

    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    batch_size = query_states.shape[0]
    scaling = paligemma.language_model.layers[layer_idx].self_attn.scaling
    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma.language_model.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    head_dim = paligemma.language_model.layers[layer_idx].self_attn.head_dim
    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
        after_first_residual = out_emb.clone()
        out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


class GemmaConfig:
    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaConfig:
    if variant == "gemma_300m":
        return GemmaConfig(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256)
    if variant == "gemma_2b":
        return GemmaConfig(width=2048, depth=18, mlp_dim=16_384, num_heads=8, num_kv_heads=1, head_dim=256)
    raise ValueError(f"Unknown variant: {variant}")


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None
        self.to_bfloat16_for_selected_params(precision)
        self.set_requires_grad()

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()
            for params in self.paligemma.vision_tower.parameters():
                params.requires_grad = False

        if self.train_expert_only:
            self.paligemma.eval()
            for params in self.paligemma.parameters():
                params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)

        if self.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()

        if self.train_expert_only:
            self.paligemma.eval()

        return self

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )

            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values


class PI05Pytorch(nn.Module):
    def __init__(self, config: PI05Config):
        super().__init__()
        self.config = config

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
        )
        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)
        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.gradient_checkpointing_enabled = False

        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)

        msg = "The pi05 runtime requires the patched transformers build."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

    def _apply_checkpoint(self, func, *args, **kwargs):
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func,
                *args,
                use_reentrant=False,
                preserve_rng_state=False,
                **kwargs,
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha,
            self.config.time_sampling_beta_beta,
            bsize,
            device,
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(self, images, img_masks, tokens, masks):
        embs = []
        pad_masks = []
        att_masks = []

        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self._apply_checkpoint(self.paligemma_with_expert.embed_image, img)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            return lang_emb * math.sqrt(lang_emb.shape[-1])

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(pad_masks.shape[0], len(att_masks))
        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)
        action_emb = self._apply_checkpoint(self.action_in_proj, noisy_actions)

        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        adarms_cond = time_emb

        embs = action_emb
        bsize, action_time_dim = embs.shape[:2]
        pad_masks = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        att_masks = [1] + ([0] * (self.config.chunk_size - 1))
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, images, img_masks, tokens, masks, actions, noise=None, time=None) -> Tensor:
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func,
            prefix_embs,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self._apply_checkpoint(self.action_out_proj, suffix_out)
        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, images, img_masks, tokens, masks, noise=None, num_steps=None) -> Tensor:
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device
        if noise is None:
            noise = self.sample_noise((bsize, self.config.chunk_size, self.config.max_action_dim), device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            v_t = self.denoise_step(prefix_pad_masks, past_key_values, x_t, time.expand(bsize))
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


class PI05Policy(PreTrainedPolicy):
    config_class = PI05Config
    name = "pi05"

    def __init__(
        self,
        config: PI05Config,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        default_task: str | None = None,
        processor_name_or_path: str | Path | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.dataset_stats = self._prepare_dataset_stats(dataset_stats)
        self.default_task = default_task
        self.model = PI05Pytorch(config)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.tokenizer = _load_pi05_tokenizer(config, processor_name_or_path=processor_name_or_path)
        self.model.to(config.device)
        self.reset()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: PI05Config | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        default_task: str | None = None,
        **kwargs,
    ):
        policy_dir = Path(pretrained_name_or_path)
        processor_name_or_path = kwargs.pop("processor_name_or_path", None)
        if dataset_stats is None and policy_dir.is_dir():
            stats_path = policy_dir / _DATASET_STATS_FILENAME
            if stats_path.exists():
                dataset_stats = torch.load(stats_path, map_location="cpu")
        if default_task is None and policy_dir.is_dir():
            metadata_path = policy_dir / _POLICY_METADATA_FILENAME
            if metadata_path.exists():
                with metadata_path.open() as f:
                    default_task = json.load(f).get("default_task")
        processor_name_or_path = _resolve_pi05_tokenizer_source(
            config=config if config is not None else cls.config_class(device="cpu"),
            pretrained_name_or_path=pretrained_name_or_path,
            explicit_source=processor_name_or_path,
        )
        if config is not None:
            config.tokenizer_name_or_path = str(processor_name_or_path)

        return super().from_pretrained(
            pretrained_name_or_path,
            config=config,
            dataset_stats=dataset_stats,
            default_task=default_task,
            processor_name_or_path=processor_name_or_path,
            **kwargs,
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        super()._save_pretrained(save_directory)
        if hasattr(self, "tokenizer"):
            self.tokenizer.save_pretrained(save_directory)
        if self.dataset_stats is not None:
            torch.save(_clone_cpu_tree(self.dataset_stats), save_directory / _DATASET_STATS_FILENAME)
        with (save_directory / _POLICY_METADATA_FILENAME).open("w") as f:
            json.dump({"default_task": self.default_task}, f, indent=2)

    def _prepare_dataset_stats(self, dataset_stats):
        if dataset_stats is None:
            return None
        prepared = _clone_cpu_tree(dataset_stats)
        for key in (OBS_STATE, ACTION):
            stats = prepared.get(key)
            if not stats:
                continue
            if "q01" not in stats or "q99" not in stats:
                if "min" in stats and "max" in stats:
                    stats["q01"] = _clone_cpu_tree(stats["min"])
                    stats["q99"] = _clone_cpu_tree(stats["max"])
                elif "q10" in stats and "q90" in stats:
                    stats["q01"] = _clone_cpu_tree(stats["q10"])
                    stats["q99"] = _clone_cpu_tree(stats["q90"])
                elif "mean" in stats and "std" in stats:
                    stats["q01"] = _clone_cpu_tree(stats["mean"] - stats["std"])
                    stats["q99"] = _clone_cpu_tree(stats["mean"] + stats["std"])
        return prepared

    def _batch_size(self, batch: dict[str, Any]) -> int:
        for key in (OBS_STATE, ACTION):
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                return value.shape[0] if value.dim() >= 2 else 1
        for key, value in batch.items():
            if key.startswith("observation.images.") and isinstance(value, torch.Tensor):
                return value.shape[0] if value.dim() >= 4 else 1
        return 1

    def _normalize_tensor(self, key: str, tensor: Tensor) -> Tensor:
        if self.dataset_stats is None or key not in self.dataset_stats:
            raise ValueError(f"Missing dataset stats for '{key}'.")
        stats = self.dataset_stats[key]
        low = stats.get("q01", stats.get("min"))
        high = stats.get("q99", stats.get("max"))
        if low is None or high is None:
            raise ValueError(f"Missing quantile/min-max stats for '{key}'.")
        low = torch.as_tensor(low, device=tensor.device, dtype=tensor.dtype)
        high = torch.as_tensor(high, device=tensor.device, dtype=tensor.dtype)
        denom = torch.where(
            (high - low) == 0,
            torch.full_like(high - low, 1e-8),
            high - low,
        )
        return 2.0 * (tensor - low) / denom - 1.0

    def _unnormalize_tensor(self, key: str, tensor: Tensor) -> Tensor:
        if self.dataset_stats is None or key not in self.dataset_stats:
            raise ValueError(f"Missing dataset stats for '{key}'.")
        stats = self.dataset_stats[key]
        low = stats.get("q01", stats.get("min"))
        high = stats.get("q99", stats.get("max"))
        if low is None or high is None:
            raise ValueError(f"Missing quantile/min-max stats for '{key}'.")
        low = torch.as_tensor(low, device=tensor.device, dtype=tensor.dtype)
        high = torch.as_tensor(high, device=tensor.device, dtype=tensor.dtype)
        denom = torch.where(
            (high - low) == 0,
            torch.full_like(high - low, 1e-8),
            high - low,
        )
        return (tensor + 1.0) * denom / 2.0 + low

    def _prepare_tasks(self, batch: dict[str, Any], batch_size: int, norm_state: Tensor) -> list[str]:
        tasks = batch.get("task")
        if tasks is None:
            tasks = [self.default_task or "perform the task"] * batch_size
        elif isinstance(tasks, str):
            tasks = [tasks] * batch_size
        elif isinstance(tasks, tuple):
            tasks = list(tasks)
        elif not isinstance(tasks, list):
            tasks = list(tasks)

        if len(tasks) != batch_size:
            raise ValueError(f"Expected {batch_size} task prompts, got {len(tasks)}.")

        padded_state = pad_vector(norm_state, self.config.max_state_dim)
        discretized_states = np.digitize(
            padded_state.detach().cpu().numpy(),
            bins=np.linspace(-1, 1, 256 + 1)[:-1],
        ) - 1

        prompts = []
        for idx, task in enumerate(tasks):
            cleaned_text = str(task).strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized_states[idx]))
            prompts.append(f"Task: {cleaned_text}, State: {state_str};\nAction: ")
        return prompts

    def _tokenize_tasks(self, prompts: list[str], device: torch.device) -> tuple[Tensor, Tensor]:
        tokenized = self.tokenizer(
            prompts,
            max_length=self.config.tokenizer_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokens = tokenized["input_ids"].to(device)
        masks = tokenized["attention_mask"].to(device=device, dtype=torch.bool)
        return tokens, masks

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        images = []
        img_masks = []
        device = next(self.parameters()).device
        batch_size = self._batch_size(batch)

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]
        if len(present_img_keys) == 0 and len(missing_img_keys) == 0:
            raise ValueError("PI05 requires at least one image input or empty_cameras > 0.")

        template_img = None
        template_mask = None
        for key in present_img_keys:
            img = batch[key]
            if img.device != device:
                img = img.to(device)
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            is_channels_first = img.shape[1] == 3
            if is_channels_first:
                img = img.permute(0, 2, 3, 1)
            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)
            img = img * 2.0 - 1.0
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)

            mask = torch.ones(img.shape[0], dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)
            template_img = img
            template_mask = mask

        if template_img is None:
            template_img = torch.full(
                (batch_size, 3, *self.config.image_resolution),
                -1.0,
                dtype=torch.float32,
                device=device,
            )
            template_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in missing_img_keys:
            images.append(torch.full_like(template_img, -1.0))
            img_masks.append(torch.zeros_like(template_mask))

        return images, img_masks

    def _prepare_inference_inputs(self, batch: dict[str, Any]) -> tuple[list[Tensor], list[Tensor], Tensor, Tensor]:
        device = next(self.parameters()).device
        norm_state = self._normalize_tensor(OBS_STATE, batch[OBS_STATE].to(device=device, dtype=torch.float32))
        prompts = self._prepare_tasks(batch, norm_state.shape[0], norm_state)
        tokens, masks = self._tokenize_tasks(prompts, device)
        images, img_masks = self._preprocess_images(batch)
        return images, img_masks, tokens, masks

    def reset(self, env_ids: list[int] | None = None):
        del env_ids
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        return [p for p in self.parameters() if p.requires_grad]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        images, img_masks, tokens, masks = self._prepare_inference_inputs(batch)
        actions = self.model.sample_actions(images, img_masks, tokens, masks)
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]
        return self._unnormalize_tensor(ACTION, actions)

    @torch.no_grad()
    def select_action_chunk(self, batch: dict[str, Tensor], n_steps: int | None = None) -> Tensor:
        actions = self.predict_action_chunk(batch)
        if n_steps is not None:
            actions = actions[:, :n_steps]
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        images, img_masks, tokens, masks = self._prepare_inference_inputs(batch)
        actions = self._normalize_tensor(ACTION, batch[ACTION].to(tokens.device, dtype=torch.float32))
        actions = pad_vector(actions, self.config.max_action_dim)
        losses = self.model.forward(images, img_masks, tokens, masks, actions)
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]
        loss = losses.mean()
        return loss, {
            "loss": float(loss.item()),
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
        }
