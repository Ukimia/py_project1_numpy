"""NumPy implementation of the OpenCLIP image encoder using supplied weights."""

import numpy as np

from helpers import GELU, LayerNorm, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    def __init__(self, weights, prefix, heads):
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']
        self.out_weight = weights[prefix + '.out_proj.weight']
        self.out_bias = weights[prefix + '.out_proj.bias']
        self.width = self.out_bias.shape[0]
        if not isinstance(heads, (int, np.integer)) or heads <= 0 or self.width % heads:
            raise ValueError("heads must be a positive integer that divides the feature width")
        self.heads = heads
        self.head_dim = self.width // heads

    def split_heads(self, x):
        """TODO-02: Reorder [B,N,D] to [B,H,N,D/H]."""
        if x.ndim != 3 or x.shape[-1] != self.width:
            raise ValueError(f"Expected attention input with shape [B,N,{self.width}]")
        batch_size, token_count, _ = x.shape
        grouped = x.reshape(batch_size, token_count, self.heads, self.head_dim)
        return grouped.transpose(0, 2, 1, 3)

    def __call__(self, x, causal=False):
        """TODO-03: Calculate scaled attention and merge the heads."""
        # Stored linear weights use [out,in].
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2 * self.width])
        v = self.split_heads(qkv[:, :, 2 * self.width:])

        scores = q @ k.swapaxes(-1, -2)
        # Use a float32 divisor so scores retain the checkpoint dtype.
        scores = scores / np.sqrt(np.float32(self.head_dim))
        if causal:
            scores = causal_mask(scores)

        coefficients = attention_weights(scores)
        context = coefficients @ v  # [B,H,N,D/H]
        batch_size, _, token_count, _ = context.shape
        merged = context.transpose(0, 2, 1, 3).reshape(
            batch_size, token_count, self.width
        )
        return merged @ self.out_weight.T + self.out_bias


class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(
            weights[prefix + '.ln_1.weight'], weights[prefix + '.ln_1.bias']
        )
        self.norm2 = LayerNorm(
            weights[prefix + '.ln_2.weight'], weights[prefix + '.ln_2.bias']
        )
        self.attention = MultiHeadAttention(weights, prefix + '.attn', heads)
        self.fc1_weight = weights[prefix + '.mlp.c_fc.weight']
        self.fc1_bias = weights[prefix + '.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix + '.mlp.c_proj.weight']
        self.fc2_bias = weights[prefix + '.mlp.c_proj.bias']
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        """TODO-04: Apply attention and MLP updates with residual connections."""
        attention_update = self.attention(self.norm1(x), causal=causal)
        residual = x + attention_update

        mlp_input = self.norm2(residual)
        hidden = mlp_input @ self.fc1_weight.T + self.fc1_bias
        activated = self.gelu(hidden)
        mlp_update = activated @ self.fc2_weight.T + self.fc2_bias
        return residual + mlp_update


class ImageEncoder:
    # Fixed dimensions of the course's ViT-B-32 checkpoint.
    IMAGE_SIZE = 224
    CHANNELS = 3
    PATCH_SIZE = 32
    PATCH_GRID_SIZE = IMAGE_SIZE // PATCH_SIZE
    PATCH_FEATURES = CHANNELS * PATCH_SIZE ** 2
    WIDTH = 768
    HEAD_COUNT = 12
    BLOCK_COUNT = 12

    def __init__(self, weights):
        self.patch_weight = weights['visual.conv1.weight'].reshape(
            self.WIDTH, self.PATCH_FEATURES
        )
        self.cls = weights['visual.class_embedding'].reshape(1, 1, self.WIDTH)
        self.position = weights['visual.positional_embedding']
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [
            TransformerBlock(weights, f'visual.transformer.resblocks.{index}', self.HEAD_COUNT)
            for index in range(self.BLOCK_COUNT)
        ]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']  # [in,out]

    def patchify(self, images):
        """TODO-01: Extract patches in row-major order, with C,H,W within each patch."""
        expected_shape = (self.CHANNELS, self.IMAGE_SIZE, self.IMAGE_SIZE)
        if not isinstance(images, np.ndarray):
            raise TypeError("images must be a NumPy array")
        if images.ndim != 4 or images.shape[1:] != expected_shape:
            raise ValueError(f"Expected images with shape [B,{self.CHANNELS},{self.IMAGE_SIZE},{self.IMAGE_SIZE}]")
        if images.shape[0] == 0:
            raise ValueError("images must contain at least one image")
        if images.dtype != np.float32:
            raise TypeError("images must have dtype float32 after preprocessing")
        if not np.isfinite(images).all():
            raise ValueError("images must contain only finite pixel values")

        batch_size = images.shape[0]
        grid_size = self.PATCH_GRID_SIZE
        patch_size = self.PATCH_SIZE
        grid = images.reshape(
            batch_size, self.CHANNELS, grid_size, patch_size, grid_size, patch_size
        )
        ordered_patches = grid.transpose(0, 2, 4, 1, 3, 5)
        return ordered_patches.reshape(batch_size, grid_size ** 2, self.PATCH_FEATURES)

    def embed(self, images):
        """TODO-05: Project patches, prepend CLS, and add position embeddings."""
        patches = self.patchify(images)
        patch_tokens = patches @ self.patch_weight.T
        cls_tokens = np.tile(self.cls, (patch_tokens.shape[0], 1, 1))
        tokens = np.concatenate([cls_tokens, patch_tokens], axis=1)
        return tokens + self.position  # Positions are shared across the batch.

    def encode_tokens(self, images):
        """Encode the full token sequence through all Transformer blocks."""
        tokens = self.pre_norm(self.embed(images))
        for block in self.blocks:
            tokens = block(tokens)
        return tokens

    def __call__(self, images):
        """TODO-06: Project and normalize the CLS feature for each image."""
        tokens = self.encode_tokens(images)
        cls_tokens = self.post_norm(tokens[:, 0, :])
        return unit_vector(cls_tokens @ self.projection)
