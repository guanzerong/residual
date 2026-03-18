# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

import einops
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.init import trunc_normal_


class PatchEmbed1(nn.Module):
    def __init__(self, embed_dim, patch_size=8, stride=-1):
        super().__init__()
        if stride <= 0:
            stride = patch_size
        self.conv = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=stride)

        self.patch_dim = embed_dim

    def get_output_grid_size(self, image_size: tuple[int, int]) -> tuple[int, int]:
        with torch.no_grad():
            y = self.conv(torch.zeros(1, 3, *image_size))
        return int(y.shape[-2]), int(y.shape[-1])

    def forward(self, x: torch.Tensor, return_grid_size: bool = False):
        y = self.conv(x)
        grid_size = (int(y.shape[-2]), int(y.shape[-1]))
        y = einops.rearrange(y, "b c h w -> b (h  w) c")
        if return_grid_size:
            return y, grid_size
        return y  # noqa: RET504


class PatchEmbed2(nn.Module):
    def __init__(self, embed_dim, use_norm, patch_size=8, stride=-1):
        super().__init__()
        if stride <= 0:
            stride = max(1, patch_size // 2)
        layers = [
            nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=stride),
            nn.GroupNorm(embed_dim, embed_dim) if use_norm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2),
        ]
        self.embed = nn.Sequential(*layers)

        self.patch_dim = embed_dim

    def get_output_grid_size(self, image_size: tuple[int, int]) -> tuple[int, int]:
        with torch.no_grad():
            y = self.embed(torch.zeros(1, 3, *image_size))
        return int(y.shape[-2]), int(y.shape[-1])

    def forward(self, x: torch.Tensor, return_grid_size: bool = False):
        y = self.embed(x)
        grid_size = (int(y.shape[-2]), int(y.shape[-1]))
        y = einops.rearrange(y, "b c h w -> b (h  w) c")
        if return_grid_size:
            return y, grid_size
        return y  # noqa: RET504


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_head):
        super().__init__()
        assert embed_dim % num_head == 0

        self.num_head = num_head
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, attn_mask):
        """
        x: [batch, seq, embed_dim]
        """
        qkv = self.qkv_proj(x)
        q, k, v = einops.rearrange(qkv, "b t (k h d) -> b k h t d", k=3, h=self.num_head).unbind(1)
        # force flash/mem-eff attention, it will raise error if flash cannot be applied
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            attn_v = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=attn_mask)
        attn_v = einops.rearrange(attn_v, "b h t d -> b t (h d)")
        return self.out_proj(attn_v)


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_head, dropout):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(embed_dim, num_head)

        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.linear2 = nn.Linear(4 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        x = x + self.dropout(self.mha(self.layer_norm1(x), attn_mask))
        x = x + self.dropout(self._ff_block(self.layer_norm2(x)))
        return x  # noqa: RET504

    def _ff_block(self, x):
        x = self.linear2(nn.functional.gelu(self.linear1(x)))
        return x  # noqa: RET504


class MinVit(nn.Module):
    def __init__(self, embed_style, embed_dim, embed_norm, num_head, depth, image_size, patch_size=8, stride=-1):
        super().__init__()

        if embed_style == "embed1":
            self.patch_embed = PatchEmbed1(embed_dim, patch_size=patch_size, stride=stride)
        elif embed_style == "embed2":
            self.patch_embed = PatchEmbed2(embed_dim, use_norm=embed_norm, patch_size=patch_size, stride=stride)
        else:
            raise NotImplementedError(f"Unknown embed style {embed_style}")

        self.base_grid_size = self.patch_embed.get_output_grid_size(image_size)
        self.num_patches = self.base_grid_size[0] * self.base_grid_size[1]
        # Keep the old attribute name for compatibility with callers that inspect the patch embed.
        self.patch_embed.num_patch = self.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        layers = [TransformerLayer(embed_dim, num_head, 0) for _ in range(depth)]

        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(embed_dim)

        # weight init
        trunc_normal_(self.pos_embed, std=0.02)
        named_apply(init_weights_vit_timm, self)

    def _resize_pos_embed(self, grid_size: tuple[int, int], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if grid_size == self.base_grid_size:
            return self.pos_embed.to(device=device, dtype=dtype)

        pos_embed = self.pos_embed.reshape(1, self.base_grid_size[0], self.base_grid_size[1], -1)
        pos_embed = pos_embed.permute(0, 3, 1, 2)
        pos_embed = torch.nn.functional.interpolate(
            pos_embed,
            size=grid_size,
            mode="bicubic",
            align_corners=False,
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)
        return pos_embed.to(device=device, dtype=dtype)

    def forward(self, x):
        x, grid_size = self.patch_embed(x, return_grid_size=True)
        x = x + self._resize_pos_embed(grid_size, dtype=x.dtype, device=x.device)
        x = self.net(x)
        return self.norm(x)


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def named_apply(fn, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        full_child_name = f"{name}.{child_name}" if name else child_name
        named_apply(fn=fn, module=child_module, name=full_child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def test_patch_embed():
    print("embed 1")
    embed = PatchEmbed1(128)
    x = torch.rand(10, 3, 84, 84)
    y = embed(x)
    print(y.size())

    print("embed 2")
    embed = PatchEmbed2(128, True)
    x = torch.rand(10, 3, 84, 84)
    y = embed(x)
    print(y.size())


def test_transformer_layer():
    embed = PatchEmbed1(128)
    x = torch.rand(10, 3, 84, 84)
    y = embed(x)
    print(y.size())

    transformer = TransformerLayer(
        num_head=4,
        dropout=0,
        embed_dim=128,
    )
    z = transformer(y)
    print(z.size())


if __name__ == "__main__":
    test_patch_embed()
    test_transformer_layer()
