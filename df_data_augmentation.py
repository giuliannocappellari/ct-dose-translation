# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Tuple

import cv2
import kornia
import numpy as np
import random

# Try to import skimage for histogram matching, fallback if not available
try:
    from skimage.exposure import match_histograms
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: skimage not available. Histogram matching will be skipped.")

# Try to import LPIPS, fallback if not available
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not available. Composite loss will use MSE + L1 only.")
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm


def exists(x):
    return x is not None


def default(val, d):
    return d if val is None else val


def load_checkpoint(model: torch.nn.Module, ckpt_path: str | Path, map_location):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(ckpt)
    model.eval()
    return model


class SinusoidalPosEmb(nn.Module):
    """From "Improved DDPM"—encodes timestep t into a 1‑D embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class ResnetBlockTime(nn.Module):
    """ResNet block with timestep conditioning (FiLM‑style)."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(Swish(), nn.Linear(time_dim, out_ch * 2))
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch), Swish(), nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch), Swish(), nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        )
        self.res_conv = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        h = self.block1(x)
        scale, shift = self.mlp(t_emb).chunk(2, dim=1)
        h = h * (1 + scale[..., None, None]) + shift[..., None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class ResnetBlockTimeAdaptive(nn.Module):
    """ResNet block with timestep conditioning and adaptive GroupNorm for small channel counts."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        # Find largest divisor of channels that's <= 32 for GroupNorm
        def get_groups(ch):
            for g in [32, 16, 8, 4, 2, 1]:
                if ch % g == 0:
                    return g
            return 1
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(get_groups(in_ch), in_ch), Swish(), nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        )
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.block2 = nn.Sequential(
            nn.GroupNorm(get_groups(out_ch), out_ch), Swish(), nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        )
        self.res_conv = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        h = self.block1(x)
        h += self.time_proj(t_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class AttentionBlock(nn.Module):
    """Vanilla QKV self‑attention over H×W positions."""

    def __init__(self, ch: int, num_heads: int = 1):
        super().__init__()
        self.num_heads = num_heads
        # Find largest divisor of channels that's <= 32 for GroupNorm
        def get_groups(ch):
            for g in [32, 16, 8, 4, 2, 1]:
                if ch % g == 0:
                    return g
            return 1
        
        self.norm = nn.GroupNorm(get_groups(ch), ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)  # each [B,C,H,W]
        q = q.view(B, C, H * W).transpose(1, 2)  # [B,HW,C]
        k = k.view(B, C, H * W).transpose(1, 2)  # [B,HW,C]
        v = v.view(B, C, H * W).transpose(1, 2)  # [B,HW,C]
        attn = torch.softmax(q @ k.transpose(-2, -1) / (C**0.5), dim=-1)
        out = attn @ v  # [B,HW,C]
        out = out.transpose(1, 2).view(B, C, H, W)  # [B,C,H,W]
        out = self.proj(out)
        return x + out


class CrossAttentionBlock(nn.Module):
    """Cross-attention between noisy input (query) and conditioning image (key/value)."""
    
    def __init__(self, query_ch: int, context_ch: int, num_heads: int = 1):
        super().__init__()
        self.num_heads = num_heads
        self.query_ch = query_ch
        self.context_ch = context_ch
        
        # Find largest divisor of channels that's <= 32 for GroupNorm
        def get_groups(ch):
            for g in [32, 16, 8, 4, 2, 1]:
                if ch % g == 0:
                    return g
            return 1
        
        # Normalization layers
        self.norm_query = nn.GroupNorm(get_groups(query_ch), query_ch)
        self.norm_context = nn.GroupNorm(get_groups(context_ch), context_ch)
        
        # Projection layers
        self.to_q = nn.Conv2d(query_ch, query_ch, 1)
        self.to_k = nn.Conv2d(context_ch, query_ch, 1)  # Project context to query dim
        self.to_v = nn.Conv2d(context_ch, query_ch, 1)  # Project context to query dim
        self.to_out = nn.Conv2d(query_ch, query_ch, 1)
        
    def forward(self, query: torch.Tensor, context: torch.Tensor):
        """Cross-attention between query (noisy input) and context (conditioning).
        
        Args:
            query: Noisy input features [B, query_ch, H, W]
            context: Conditioning features [B, context_ch, H_ctx, W_ctx]
            
        Returns:
            Cross-attended features [B, query_ch, H, W]
        """
        B, C_q, H_q, W_q = query.shape
        B, C_ctx, H_ctx, W_ctx = context.shape
        
        # Normalize inputs
        query_norm = self.norm_query(query)
        context_norm = self.norm_context(context)
        
        # Resize context to match query spatial dimensions if needed
        if (H_ctx, W_ctx) != (H_q, W_q):
            context_norm = F.interpolate(context_norm, size=(H_q, W_q), mode='bilinear', align_corners=False)
        
        # Generate Q, K, V
        q = self.to_q(query_norm)  # [B, C_q, H, W]
        k = self.to_k(context_norm)  # [B, C_q, H, W] 
        v = self.to_v(context_norm)  # [B, C_q, H, W]
        
        # Reshape for attention computation
        q = q.view(B, C_q, H_q * W_q).transpose(1, 2)  # [B, HW, C_q]
        k = k.view(B, C_q, H_q * W_q).transpose(1, 2)  # [B, HW, C_q]
        v = v.view(B, C_q, H_q * W_q).transpose(1, 2)  # [B, HW, C_q]
        
        # Compute cross-attention
        scale = C_q ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)  # [B, HW, HW]
        out = attn @ v  # [B, HW, C_q]
        
        # Reshape back to spatial dimensions
        out = out.transpose(1, 2).view(B, C_q, H_q, W_q)  # [B, C_q, H, W]
        out = self.to_out(out)
        
        # Residual connection
        return query + out


class CrossAttentionIdentity(nn.Module):
    """Identity module for cross-attention that accepts two arguments."""
    
    def forward(self, query: torch.Tensor, context: torch.Tensor):
        """Return query unchanged, ignoring context."""
        return query


class GuidedFilterModule(nn.Module):
    """Learnable Guided Filter Module for edge-conditioned skip connections."""
    def __init__(self, skip_ch: int, edge_ch: int = 1):
        super().__init__()
        # Learnable convolutional block to combine skip features with edge guidance
        self.conv = nn.Sequential(
            nn.Conv2d(skip_ch + edge_ch, skip_ch, 3, padding=1),
            nn.GroupNorm(min(32, skip_ch), skip_ch),  # Adaptive group norm
            Swish()
        )
    
    def forward(self, skip_feat: torch.Tensor, edge_feat: torch.Tensor):
        # Resize edge features to match skip features if needed
        if edge_feat.shape[-2:] != skip_feat.shape[-2:]:
            edge_feat = F.interpolate(edge_feat, size=skip_feat.shape[-2:], mode='nearest')
        
        # Concatenate and apply learned transformation
        combined = torch.cat([skip_feat, edge_feat], dim=1)
        return self.conv(combined)


class ICRHead(nn.Module):
    """Image Coordinate Refinement head with coordinate channels."""
    
    def __init__(self, in_ch: int, out_ch: int = 1, use_coordinates: bool = True):
        super().__init__()
        self.use_coordinates = use_coordinates
        self.out_ch = out_ch
        
        # Coordinate channels: x, y (normalized to [-1, 1])
        coord_ch = 2 if use_coordinates else 0
        total_in_ch = in_ch + coord_ch
        
        # Find largest divisor of channels that's <= 32 for GroupNorm
        def get_groups(ch):
            for g in [32, 16, 8, 4, 2, 1]:
                if ch % g == 0:
                    return g
            return 1
        
        # ICR refinement network
        self.icr_net = nn.Sequential(
            nn.Conv2d(total_in_ch, in_ch, 3, padding=1),
            nn.GroupNorm(get_groups(in_ch), in_ch),
            Swish(),
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1),
            nn.GroupNorm(get_groups(in_ch // 2), in_ch // 2),
            Swish(),
            nn.Conv2d(in_ch // 2, out_ch, 3, padding=1),
        )
    
    def create_coordinate_channels(self, x: torch.Tensor) -> torch.Tensor:
        """Create normalized coordinate channels.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Coordinate tensor [B, 2, H, W] with x,y coordinates in [-1, 1]
        """
        B, C, H, W = x.shape
        device = x.device
        
        # Create coordinate grids
        y_coords = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        x_coords = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        
        coords = torch.cat([x_coords, y_coords], dim=1)  # [B, 2, H, W]
        return coords
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ICR head with coordinate refinement.
        
        Args:
            x: Input features [B, in_ch, H, W]
            
        Returns:
            Refined output [B, out_ch, H, W]
        """
        if self.use_coordinates:
            # Add coordinate channels
            coords = self.create_coordinate_channels(x)
            x_with_coords = torch.cat([x, coords], dim=1)
        else:
            x_with_coords = x
        
        # Apply ICR refinement
        return self.icr_net(x_with_coords)


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator for adversarial training.
    
    Discriminates between real and generated CT image patches.
    Uses spectral normalization for training stability.
    """
    
    def __init__(self, input_ch=2, base_ch=64, n_layers=3, use_spectral_norm=True):
        super().__init__()
        self.input_ch = input_ch
        self.use_spectral_norm = use_spectral_norm
        
        # Helper function to optionally apply spectral normalization
        def maybe_spectral_norm(layer):
            if use_spectral_norm:
                return nn.utils.spectral_norm(layer)
            return layer
        
        # Initial convolution (no normalization)
        layers = [
            maybe_spectral_norm(nn.Conv2d(input_ch, base_ch, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Intermediate layers with increasing channels
        curr_ch = base_ch
        for i in range(n_layers):
            next_ch = min(curr_ch * 2, 512)  # Cap at 512 channels
            stride = 2 if i < n_layers - 1 else 1  # Last layer has stride 1
            
            layers.extend([
                maybe_spectral_norm(nn.Conv2d(curr_ch, next_ch, 4, stride=stride, padding=1)),
                nn.InstanceNorm2d(next_ch),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            curr_ch = next_ch
        
        # Final classification layer
        layers.append(
            maybe_spectral_norm(nn.Conv2d(curr_ch, 1, 4, stride=1, padding=1))
        )
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x, conditioning=None):
        """Forward pass through discriminator.
        
        Args:
            x: Input image tensor [B, C, H, W]
            conditioning: Optional conditioning image [B, C, H, W]
            
        Returns:
            Discriminator output [B, 1, H_patch, W_patch]
        """
        if conditioning is not None:
            # Conditional discriminator: concatenate input and conditioning
            input_tensor = torch.cat([x, conditioning], dim=1)
        else:
            # Unconditional discriminator: if model expects 2 channels but we only have 1,
            # duplicate the input channel to match expected input channels
            if x.shape[1] == 1 and self.input_ch == 2:
                input_tensor = torch.cat([x, x], dim=1)  # Duplicate channel
            else:
                input_tensor = x
        
        return self.model(input_tensor)


class AdversarialLoss(nn.Module):
    """Adversarial loss for GAN training."""
    
    def __init__(self, loss_type: str = 'lsgan', target_real_label: float = 1.0, target_fake_label: float = 0.0):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.target_real_label = target_real_label
        self.target_fake_label = target_fake_label
        
        if loss_type == 'lsgan':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'vanilla':
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif loss_type == 'wgangp':
            self.loss_fn = None  # WGAN-GP uses different loss computation
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Create target tensor with same shape as prediction."""
        target_label = self.target_real_label if target_is_real else self.target_fake_label
        return torch.full_like(prediction, target_label, device=prediction.device)
    
    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Compute adversarial loss.
        
        Args:
            prediction: Discriminator output [B, 1, H_patch, W_patch]
            target_is_real: Whether the target should be real (True) or fake (False)
            
        Returns:
            Adversarial loss scalar
        """
        if self.loss_type == 'wgangp':
            # WGAN-GP loss
            if target_is_real:
                return -prediction.mean()
            else:
                return prediction.mean()
        else:
            # LSGAN or vanilla GAN loss
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            return self.loss_fn(prediction, target_tensor)


class PixelDiffusionUNetConditional(nn.Module):
    """UNet that predicts ε given (x_t, cond) with edge guidance and cross-attention in **pixel space**."""

    def __init__(
        self,
        base_ch: int = 64,
        ch_mults: Tuple[int, ...] = (1, 2, 4, 8),
        time_dim: int = 256,
        attn_res: Tuple[int, ...] = (32, 64),  # self‑attn at 32×32 and 64×64
        use_cross_attention: bool = True,
        use_icr_head: bool = True,
    ):
        super().__init__()
        self.use_cross_attention = use_cross_attention
        self.use_icr_head = use_icr_head
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            Swish(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # init conv: concat noisy x, LD condition, and edge map → base_ch (3 channels total)
        self.init_conv = nn.Conv2d(3, base_ch, 3, padding=1)
        
        # Conditioning feature extractor (processes LD conditioning image)
        if self.use_cross_attention:
            self.cond_encoder = nn.Sequential(
                nn.Conv2d(1, base_ch // 2, 3, padding=1),
                nn.GroupNorm(min(32, base_ch // 2), base_ch // 2),
                Swish(),
                nn.Conv2d(base_ch // 2, base_ch, 3, padding=1),
                nn.GroupNorm(min(32, base_ch), base_ch),
                Swish(),
            )

        # Downsampling path
        curr_ch = base_ch
        self.downs = nn.ModuleList()
        self.cross_attns_down = nn.ModuleList()  # Cross-attention blocks for downsampling
        resolutions = []
        
        for i, mult in enumerate(ch_mults):
            out_ch = base_ch * mult
            # Adaptive group norm for small channel counts
            block = nn.ModuleList(
                [
                    ResnetBlockTimeAdaptive(curr_ch, out_ch, time_dim),
                    ResnetBlockTimeAdaptive(out_ch, out_ch, time_dim),
                    AttentionBlock(out_ch)
                    if 256 // (2**i) in attn_res
                    else nn.Identity(),
                    Downsample(out_ch) if i < len(ch_mults) - 1 else nn.Identity(),
                ]
            )
            self.downs.append(block)
            
            # Add cross-attention for conditioning at key resolutions
            if self.use_cross_attention and 256 // (2**i) in attn_res:
                self.cross_attns_down.append(CrossAttentionBlock(out_ch, base_ch))
            else:
                self.cross_attns_down.append(CrossAttentionIdentity())
            
            resolutions.append(out_ch)
            curr_ch = out_ch

        # Bottleneck
        self.mid = nn.ModuleList(
            [
                ResnetBlockTimeAdaptive(curr_ch, curr_ch, time_dim),
                AttentionBlock(curr_ch),
                ResnetBlockTimeAdaptive(curr_ch, curr_ch, time_dim),
            ]
        )

        # Upsampling path (with skip connections, GFM, and cross-attention)
        self.ups = nn.ModuleList()
        self.gfms = nn.ModuleList()  # Guided Filter Modules
        self.cross_attns_up = nn.ModuleList()  # Cross-attention blocks for upsampling
        
        for i, mult in enumerate(reversed(ch_mults[:-1])):  # mult = 4,2,1
            skip_ch = base_ch * mult  # 256,128,64
            in_ch = curr_ch + skip_ch  # 512+256, 256+128, 128+64
            out_ch = skip_ch  # 256,128,64

            # Add GFM for this skip connection
            self.gfms.append(GuidedFilterModule(skip_ch))
            
            # Add cross-attention for conditioning at key resolutions
            if self.use_cross_attention and 256 // (2 ** (len(ch_mults) - 2 - i)) in attn_res:
                self.cross_attns_up.append(CrossAttentionBlock(out_ch, base_ch))
            else:
                self.cross_attns_up.append(CrossAttentionIdentity())

            block = nn.ModuleDict(
                {
                    # always upsample; use curr_ch so conv sees correct in‑channels
                    "up": Upsample(curr_ch),
                    "res1": ResnetBlockTimeAdaptive(in_ch, out_ch, time_dim),
                    "res2": ResnetBlockTimeAdaptive(out_ch, out_ch, time_dim),
                    "attn": AttentionBlock(out_ch)
                    if 256 // (2 ** (len(ch_mults) - 2 - i)) in attn_res
                    else nn.Identity(),
                }
            )
            self.ups.append(block)
            curr_ch = out_ch  # 512→256→128→64

        # Final output layer: ICR head or simple conv
        if self.use_icr_head:
            self.final = ICRHead(curr_ch, out_ch=1, use_coordinates=True)
        else:
            # Find largest divisor of channels that's <= 32 for GroupNorm
            def get_groups(ch):
                for g in [32, 16, 8, 4, 2, 1]:
                    if ch % g == 0:
                        return g
                return 1
            
            self.final = nn.Sequential(
                nn.GroupNorm(get_groups(curr_ch), curr_ch),  # Adaptive group norm
                Swish(),
                nn.Conv2d(curr_ch, 1, 3, padding=1),
            )

    def create_edge_pyramid(self, edge_full: torch.Tensor) -> list:
        """Create multi-scale edge maps for skip connections."""
        edge_pyramid = []
        B, C, H, W = edge_full.shape
        
        # Create edge maps at different scales matching the number of skip connections
        # Number of skip connections = len(ch_mults) - 1
        num_skips = len(self.ups)  # This equals len(ch_mults) - 1
        
        # Create scales for each skip connection level
        scales = [2 ** (i + 1) for i in range(num_skips)]
        for scale in scales:
            edge_scaled = F.avg_pool2d(edge_full, kernel_size=scale, stride=scale)
            edge_pyramid.append(edge_scaled)
        
        return edge_pyramid

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, edge_map: torch.Tensor = None):
        # 1) time embedding
        t_emb = self.time_mlp(t)

        # 2) Extract edges from conditioning image if not provided
        if edge_map is None:
            # Convert cond from [-1,1] to [0,1] for edge detection
            cond_01 = (cond + 1.0) / 2.0
            edge_map = torch.zeros_like(cond)
            
            for i in range(cond.size(0)):
                # Extract edges using Canny
                img_np = cond_01[i, 0].cpu().numpy()
                edges_np = extract_edges_canny(img_np)
                edge_map[i, 0] = torch.from_numpy(edges_np).to(cond.device)

        # Create edge pyramid for multi-scale guidance
        edge_pyramid = self.create_edge_pyramid(edge_map)
        
        # 3) Extract conditioning features for cross-attention
        if self.use_cross_attention:
            cond_features = self.cond_encoder(cond)  # [B, base_ch, H, W]
        else:
            cond_features = None

        # 4) concat noisy x, LD condition, and edge map
        h = self.init_conv(torch.cat([x_t, cond, edge_map], dim=1))

        skips = []
        # ---- DOWN SAMPLING -------------------------------------------------
        for i, (block, cross_attn) in enumerate(zip(self.downs, self.cross_attns_down)):  # i = 0..len(downs)-1
            res1, res2, attn, down = block

            h = res1(h, t_emb)
            h = res2(h, t_emb)
            h = attn(h)
            
            # Apply cross-attention with conditioning features
            if self.use_cross_attention and cond_features is not None:
                h = cross_attn(h, cond_features)

            # save skip for all but the deepest level
            if i < len(self.downs) - 1:  # ← only 0,1,2 for 4‑level UNet
                skips.append(h)

            h = down(h)  # downsample (no‑op at bottom)

        # ---- BOTTLENECK ----------------------------------------------------
        for layer in self.mid:
            h = layer(h, t_emb) if isinstance(layer, ResnetBlockTimeAdaptive) else layer(h)

        # ---- UP SAMPLING with Edge Guidance and Cross-Attention -----------
        for i, (block, gfm, cross_attn) in enumerate(zip(self.ups, self.gfms, self.cross_attns_up)):
            h = block["up"](h)  # 32→64→128→256
            
            # Get skip connection and apply edge guidance
            skip_feat = skips.pop()  # spatial sizes now equal
            edge_feat = edge_pyramid[i]  # corresponding edge map
            
            # Apply GFM to condition skip features with edges
            skip_guided = gfm(skip_feat, edge_feat)
            
            h = torch.cat([h, skip_guided], 1)  # concatenate guided skip
            h = block["res1"](h, t_emb)
            h = block["res2"](h, t_emb)
            h = block["attn"](h)
            
            # Apply cross-attention with conditioning features
            if self.use_cross_attention and cond_features is not None:
                h = cross_attn(h, cond_features)

        return self.final(h)


class CTPairsDataset(Dataset):
    """Loads paired low‑dose (LD) and normal‑dose (ND) CT slices with data augmentations.
    
    Supports three types of augmentations:
    1. Physics-aware noise injection (Poisson noise modeling)
    2. Spectral intensity jitter (gamma, brightness/contrast, histogram matching)
    3. MixUp and PatchMix augmentations
    """

    def __init__(self, root: str | Path, transform=None, 
                 enable_augmentations=True,
                 noise_injection_prob=0.3,
                 spectral_jitter_prob=0.4, 
                 mixup_prob=0.15,
                 patchmix_prob=0.15):
        root = Path(root)

        # collect all your low/high folders however you like
        self.pairs = self.collect_pairs_by_position(root)
        self.transform = default(
            transform,
            transforms.Compose(
                [
                    transforms.CenterCrop(256),
                    transforms.ToTensor(),
                    transforms.Normalize(0.5, 0.5),
                ]
            ),
        )
        
        # Store augmentation settings
        self.enable_augmentations = enable_augmentations
        self.noise_injection_prob = noise_injection_prob
        self.spectral_jitter_prob = spectral_jitter_prob
        self.mixup_prob = mixup_prob
        self.patchmix_prob = patchmix_prob
        
        print(f"CTPairsDataset initialized with {len(self.pairs)} pairs")
        if enable_augmentations:
            print(f"Augmentations enabled: noise_inj={noise_injection_prob:.2f}, "
                  f"spectral_jitter={spectral_jitter_prob:.2f}, mixup={mixup_prob:.2f}, "
                  f"patchmix={patchmix_prob:.2f}")

    def __len__(self):
        return len(self.pairs)

    def collect_pairs_by_position(self, root: str, sort: bool = True):
        """
        Walk quarter_3mm ↔ full_3mm and quarter_1mm ↔ full_1mm in parallel,
        and pair the i-th image in each patient directory by position.
        """
        root = Path(root)
        mapping = {
            "quarter_3mm": "full_3mm",
            "quarter_1mm": "full_1mm",
        }

        pairs = []
        for small_name, full_name in mapping.items():
            small_root = root / small_name
            full_root = root / full_name

            # each subfolder under small_root is a patient ID
            for patient_dir in sorted(small_root.iterdir()):
                if not patient_dir.is_dir():
                    continue

                # match the same patient under the full folder
                full_patient_dir = full_root / patient_dir.name
                if not full_patient_dir.exists():
                    continue

                # grab all images anywhere under that patient (rglob)
                small_imgs = list(patient_dir.rglob("*.IMA"))
                full_imgs = list(full_patient_dir.rglob("*.IMA"))

                if sort:
                    small_imgs.sort()
                    full_imgs.sort()

                # pair by index
                for small_img, full_img in zip(small_imgs, full_imgs):
                    pairs.append((small_img, full_img))

        return pairs
    
    def process_to_uint8(self, pixel_array):
        """Convert DICOM pixel array to normalized 8-bit image."""
        img = pixel_array.astype(np.float32)
        # Normalize to [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        # Convert to 8-bit
        return (img * 255).astype(np.uint8)
    
    def apply_physics_noise_injection(self, img_q_array, img_f_array):
        """Apply physics-aware Poisson noise injection."""
        # Convert to [0,1] float
        arr = img_f_array.astype(np.float32) / 255.0
        
        # Simulate quarter-dose from full-dose with Poisson noise
        N0 = 10000  # base photon count
        intensity = np.clip(arr, 1e-6, 1)  # intensity values
        
        # Poisson noise on counts
        counts = np.random.poisson(intensity * N0)
        intensity_noisy = counts.astype(np.float32) / N0
        
        # Back to [0,1] then to uint8
        intensity_noisy = np.clip(intensity_noisy, 0, 1)
        return (intensity_noisy * 255).astype(np.uint8)
    
    def apply_spectral_jitter(self, img_q_array, img_f_array):
        """Apply spectral intensity jitter (gamma, brightness/contrast, histogram matching)."""
        # Convert to [0,1] float
        arr_q = img_q_array.astype(np.float32) / 255.0
        arr_f = img_f_array.astype(np.float32) / 255.0
        
        # Gamma jitter
        gamma = np.random.uniform(0.8, 1.2)
        arr_q = np.clip(arr_q ** gamma, 0, 1)
        
        # Brightness/contrast jitter
        alpha = np.random.uniform(0.9, 1.1)
        beta = np.random.uniform(-0.05, 0.05)
        arr_q = np.clip(arr_q * alpha + beta, 0, 1)
        arr_f = np.clip(arr_f * alpha + beta, 0, 1)  # Apply same α,β to full-dose
        
        # Optional histogram matching (10% of the time)
        if SKIMAGE_AVAILABLE and random.random() < 0.1:
            arr_q = match_histograms(arr_q, arr_f, channel_axis=None)
            arr_q = np.clip(arr_q, 0, 1)
        
        # Convert back to uint8
        img_q_jittered = (arr_q * 255).astype(np.uint8)
        img_f_jittered = (arr_f * 255).astype(np.uint8)
        
        return img_q_jittered, img_f_jittered
    
    def apply_mixup(self, idx):
        """Apply MixUp augmentation by blending with another random sample."""
        # Pick a different random index
        j = random.randrange(len(self.pairs))
        if j == idx:  # Avoid self-mixing
            j = (j + 1) % len(self.pairs)
        
        # Load second sample
        ld_path2, nd_path2 = self.pairs[j]
        ds_q2 = pydicom.dcmread(ld_path2)
        ds_f2 = pydicom.dcmread(nd_path2)
        img_q2 = self.process_to_uint8(ds_q2.pixel_array)
        img_f2 = self.process_to_uint8(ds_f2.pixel_array)
        
        return img_q2, img_f2
    
    def apply_patchmix(self, img_q_array, img_f_array, img_q2_array, img_f2_array):
        """Apply PatchMix augmentation by splicing patches from second sample."""
        H, W = img_q_array.shape
        
        # Random patch size (between 1/4 and 1/2 of image)
        patch_ratio = random.uniform(0.25, 0.5)
        ph = int(H * patch_ratio)
        pw = int(W * patch_ratio)
        
        # Random patch location
        y = random.randint(0, H - ph)
        x = random.randint(0, W - pw)
        
        # Copy arrays to avoid modifying originals
        arr_q = img_q_array.copy()
        arr_f = img_f_array.copy()
        
        # Replace patch region
        arr_q[y:y+ph, x:x+pw] = img_q2_array[y:y+ph, x:x+pw]
        arr_f[y:y+ph, x:x+pw] = img_f2_array[y:y+ph, x:x+pw]
        
        return arr_q, arr_f

    def __getitem__(self, idx):
        ld_path, nd_path = self.pairs[idx]
        ds_q = pydicom.dcmread(ld_path)
        ds_f = pydicom.dcmread(nd_path)
        
        # Process to normalized 8-bit arrays
        img_q_array = self.process_to_uint8(ds_q.pixel_array)
        img_f_array = self.process_to_uint8(ds_f.pixel_array)
        
        # Apply augmentations if enabled
        if self.enable_augmentations:
            # Check for MixUp augmentation (highest priority)
            if random.random() < self.mixup_prob:
                # Load second sample for mixing
                img_q2_array, img_f2_array = self.apply_mixup(idx)
                
                # MixUp blending
                lambda_mix = np.random.uniform(0.3, 0.7)
                img_q_array = cv2.addWeighted(img_q_array, lambda_mix, img_q2_array, 1-lambda_mix, 0)
                img_f_array = cv2.addWeighted(img_f_array, lambda_mix, img_f2_array, 1-lambda_mix, 0)
                
            # Check for PatchMix augmentation (mutually exclusive with MixUp)
            elif random.random() < self.patchmix_prob:
                # Load second sample for patch mixing
                img_q2_array, img_f2_array = self.apply_mixup(idx)  # Reuse same loading logic
                
                # Apply PatchMix
                img_q_array, img_f_array = self.apply_patchmix(
                    img_q_array, img_f_array, img_q2_array, img_f2_array
                )
            
            # Physics-aware noise injection (can combine with mixing)
            if random.random() < self.noise_injection_prob:
                img_q_array = self.apply_physics_noise_injection(img_q_array, img_f_array)
            
            # Spectral intensity jitter (can combine with other augmentations)
            if random.random() < self.spectral_jitter_prob:
                img_q_array, img_f_array = self.apply_spectral_jitter(img_q_array, img_f_array)
        
        # Convert to PIL Images
        img_q = Image.fromarray(img_q_array)
        img_f = Image.fromarray(img_f_array)
        
        # Apply transforms (crop, tensor conversion, normalization)
        ld, nd = self.transform(img_q), self.transform(img_f)
        return ld, nd


def extract_edges_canny(img):
    """Extract edges using Canny edge detection."""
    # Convert to uint8 if needed
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    edges = cv2.Canny(img, 100, 200)
    return edges.astype(np.float32) / 255.0


def forward_diffusion_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    sqrt_alphas_cumprod,
    sqrt_one_minus_alphas_cumprod,
):
    """Returns noisy x_t, plus the noise (ε) used to generate it."""
    noise = torch.randn_like(x0)
    sqrt_ac = sqrt_alphas_cumprod[t][:, None, None, None]
    sqrt_om = sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
    return sqrt_ac * x0 + sqrt_om * noise, noise


def bridge_forward_diffusion(
    ld_img: torch.Tensor,
    nd_img: torch.Tensor, 
    t: torch.Tensor,
    sqrt_alphas_cumprod,
    sqrt_one_minus_alphas_cumprod,
):
    """Schrödinger-Bridge forward process: interpolate from LD to ND.
    
    Args:
        ld_img: Low-dose image (starting point) [B, C, H, W]
        nd_img: Normal-dose image (target) [B, C, H, W]
        t: Time steps [B]
        sqrt_alphas_cumprod: sqrt(α̅_t) values
        sqrt_one_minus_alphas_cumprod: sqrt(1 - α̅_t) values
        
    Returns:
        x_t: Interpolated image at time t [B, C, H, W]
        target: What the model should predict (direction from LD to ND)
    """
    # Bridge interpolation: x_t = α̅_t * ND + (1 - α̅_t) * LD
    sqrt_ac = sqrt_alphas_cumprod[t][:, None, None, None]  # α̅_t weight for ND
    sqrt_om = sqrt_one_minus_alphas_cumprod[t][:, None, None, None]  # (1-α̅_t) weight for LD
    
    # Interpolate between LD and ND
    x_t = sqrt_ac * nd_img + sqrt_om * ld_img
    
    # Target: direction from current interpolation toward ND
    # This is what the model should predict to move from x_t toward ND
    target = nd_img - ld_img  # Direction vector from LD to ND
    
    return x_t, target


@torch.no_grad()
def sample(model, ld_batch, scheduler, guidance=5.0):
    """DDIM‑like sampling with classifier‑free guidance."""
    model.eval()
    device = ld_batch.device
    B = ld_batch.size(0)
    x = torch.randn_like(ld_batch)
    for i, t in enumerate(reversed(scheduler.timesteps)):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        eps_cond = model(x, t_batch, ld_batch)
        eps_uncond = model(x, t_batch, torch.zeros_like(ld_batch))
        eps = eps_uncond + guidance * (eps_cond - eps_uncond)
        out = scheduler.step(eps, t_batch[0], x)
        x = out.prev_sample
    return x.clamp(-1, 1)


def get_transform():
    return transforms.Compose(
        [
            transforms.CenterCrop(256),
            transforms.ToTensor(),  # → [0,1]
            transforms.Normalize(0.5, 0.5),  # → [‑1,1]
        ]
    )


def load_and_preprocess(dicom_path: str | Path, tfm) -> torch.Tensor:
    """Returns LD slice as **[1,1,256,256]** in [‑1,1]."""
    ds = pydicom.dcmread(str(dicom_path))
    arr = ds.pixel_array.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-5)
    pil = Image.fromarray((arr * 255).astype("uint8"))
    return tfm(pil).unsqueeze(0)  # [1,1,H,W]


@torch.no_grad()
def sample_ddim_guided(
    ld_img: torch.Tensor, scheduler, diffusion, guidance_scale: float = 5.0
):
    """Generate a ND prediction from a batch of LD images (pixel‑space)."""

    device = ld_img.device
    B = ld_img.size(0)
    x = torch.randn_like(ld_img)  # start from pure noise

    for t in scheduler.timesteps:
        t_int = int(t.item() if isinstance(t, torch.Tensor) else t)
        t_batch = torch.full((B,), t_int, device=device, dtype=torch.long)

        # unconditional (no LD condition)
        eps_uncond = diffusion(x, t_batch, cond=torch.zeros_like(ld_img))
        # conditional
        eps_cond = diffusion(x, t_batch, cond=ld_img)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        out = scheduler.step(eps, t_int, x, return_dict=True)
        x = out.prev_sample
    return x  # still in [-1,1]


@torch.no_grad()
def sample_bridge_ddim(
    ld_img: torch.Tensor, 
    scheduler, 
    diffusion, 
    guidance_scale: float = 5.0,
    use_bridge: bool = True
):
    """Schrödinger-Bridge DDIM sampling: start from LD, move toward ND.
    
    Args:
        ld_img: Low-dose conditioning image [B, 1, H, W]
        scheduler: DDIM scheduler
        diffusion: UNet model
        guidance_scale: Classifier-free guidance scale
        use_bridge: If True, use bridge process; if False, use standard diffusion
        
    Returns:
        Generated normal-dose image [B, 1, H, W]
    """
    device = ld_img.device
    B = ld_img.size(0)
    
    if use_bridge:
        # Bridge process: start from LD image
        x = ld_img.clone()  # Start from low-dose image
    else:
        # Standard diffusion: start from noise
        x = torch.randn_like(ld_img)
    
    # Reverse diffusion process
    for t in scheduler.timesteps:
        t_int = int(t.item() if isinstance(t, torch.Tensor) else t)
        t_batch = torch.full((B,), t_int, device=device, dtype=torch.long)
        
        if use_bridge:
            # Bridge sampling: predict direction from LD to ND
            # Unconditional: no guidance
            direction_uncond = diffusion(x, t_batch, cond=torch.zeros_like(ld_img))
            # Conditional: with LD guidance
            direction_cond = diffusion(x, t_batch, cond=ld_img)
            
            # Apply classifier-free guidance
            direction = direction_uncond + guidance_scale * (direction_cond - direction_uncond)
            
            # Bridge step: move in predicted direction
            # Scale the direction by the current timestep
            alpha_t = scheduler.alphas_cumprod[t_int] if hasattr(scheduler, 'alphas_cumprod') else 0.5
            step_size = (1 - alpha_t) * 0.1  # Adaptive step size
            x = x + step_size * direction
            
        else:
            # Standard DDIM sampling
            eps_uncond = diffusion(x, t_batch, cond=torch.zeros_like(ld_img))
            eps_cond = diffusion(x, t_batch, cond=ld_img)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            
            out = scheduler.step(eps, t_int, x, return_dict=True)
            x = out.prev_sample
    
    return x  # Bridge result or standard diffusion result


@torch.no_grad()
def predict(ld_img: torch.Tensor, model, scheduler):
    """Runs diffusion and returns prediction in **[0,1]** as [1,1,H,W]."""
    pred = sample_ddim_guided(ld_img, scheduler, model)
    return (pred + 1) / 2  # to [0,1]


# simple CT loader (identical to training)
_transform = transforms.Compose(
    [transforms.CenterCrop(256), transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
)


def evaluate(model, val_paths, scheduler, device, tfm):
    ssim_mod = kornia.metrics.SSIM(window_size=11, max_val=1.0, padding="same").to(
        device
    )
    scores = []
    scores_2 = []
    pbar = tqdm(val_paths, desc="val", unit="img")
    for ld_path in pbar:
        ld_t = load_and_preprocess(ld_path, tfm).to(device)
        with torch.no_grad():
            pred_t = sample_ddim_guided(ld_t, scheduler, model)  # [-1,1]
            pred_t = (pred_t + 1) / 2  # [0,1]
        gt_t = load_and_preprocess(ld_path, tfm).to(device)  # [1,1,H,W]
        gt = (gt_t * 0.5 + 0.5).clamp(0, 1)  # [1,1,H,W]
        # 4) compute SSIM map & reduce
        with torch.no_grad():
            ssim_map = ssim_mod(pred_t, gt)  # [1,1,H,W]
            ssim_score = ssim_map.mean().item()  # scalar
        scores.append({"path": ld_path, "ssim": ssim_score})
        scores_2.append(ssim_score)
        pbar.set_postfix(ssim=f"{ssim_score:.4f}")
        json.dump(scores, open("scores.json", "w"))
    return sum(scores_2) / len(scores_2)


class CompositeLoss(nn.Module):
    """Composite loss combining MSE, L1, LPIPS, and optionally adversarial loss for better training quality."""
    
    def __init__(self, device, mse_weight=1.0, l1_weight=0.5, lpips_weight=0.1, adv_weight=0.01, discriminator=None):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.lpips_weight = lpips_weight if LPIPS_AVAILABLE else 0.0
        self.adv_weight = adv_weight
        self.device = device
        self.discriminator = discriminator
        
        # Initialize LPIPS network if available
        if LPIPS_AVAILABLE and self.lpips_weight > 0:
            self.lpips_net = lpips.LPIPS(net='vgg').to(device)
            # Freeze LPIPS network parameters
            for param in self.lpips_net.parameters():
                param.requires_grad = False
            print(f"Initialized LPIPS loss with weight {self.lpips_weight}")
        else:
            self.lpips_net = None
            if not LPIPS_AVAILABLE:
                print("LPIPS not available - using MSE + L1 loss only")
            else:
                print("LPIPS weight is 0 - using MSE + L1 loss only")
        
        # Initialize adversarial loss if discriminator is provided
        if self.discriminator is not None and self.adv_weight > 0:
            self.adv_loss_fn = AdversarialLoss(loss_type='lsgan')
            print(f"Initialized adversarial loss with weight {self.adv_weight}")
        else:
            self.adv_loss_fn = None
            if self.adv_weight > 0:
                print("Adversarial weight > 0 but no discriminator provided")
    
    def forward(self, pred, target, generated_image=None, real_image=None, conditioning=None):
        """Compute composite loss.
        
        Args:
            pred: Predicted noise/image tensor
            target: Target noise/image tensor
            generated_image: Generated image for adversarial loss (optional)
            real_image: Real image for reference (optional)
            conditioning: Conditioning image for discriminator (optional)
            
        Returns:
            Dictionary with individual loss components and total loss
        """
        # MSE Loss (primary diffusion loss)
        mse_loss = F.mse_loss(pred, target)
        
        # L1 Loss (better pixel-wise reconstruction)
        l1_loss = F.l1_loss(pred, target)
        
        # LPIPS Loss (perceptual quality) - only if available
        if self.lpips_net is not None:
            # LPIPS expects inputs in [-1, 1] range and 3-channel images
            # Convert single-channel to 3-channel for LPIPS
            pred_3ch = pred.repeat(1, 3, 1, 1) if pred.size(1) == 1 else pred
            target_3ch = target.repeat(1, 3, 1, 1) if target.size(1) == 1 else target
            
            # Ensure inputs are in [-1, 1] range for LPIPS
            pred_norm = torch.clamp(pred_3ch, -1, 1)
            target_norm = torch.clamp(target_3ch, -1, 1)
            
            lpips_loss = self.lpips_net(pred_norm, target_norm).mean()
        else:
            # LPIPS not available, set to zero
            lpips_loss = torch.tensor(0.0, device=pred.device)
        
        # Adversarial Loss (generator loss to fool discriminator)
        if self.adv_loss_fn is not None and generated_image is not None:
            # Discriminator prediction on generated image
            with torch.no_grad():
                # Don't update discriminator during generator training
                disc_pred_fake = self.discriminator(generated_image, conditioning)
            
            # Generator wants discriminator to think generated images are real
            adv_loss = self.adv_loss_fn(disc_pred_fake, target_is_real=True)
        else:
            # No adversarial loss
            adv_loss = torch.tensor(0.0, device=pred.device)
        
        # Composite loss
        total_loss = (
            self.mse_weight * mse_loss +
            self.l1_weight * l1_loss +
            self.lpips_weight * lpips_loss +
            self.adv_weight * adv_loss
        )
        
        return {
            'total': total_loss,
            'mse': mse_loss,
            'l1': l1_loss,
            'lpips': lpips_loss,
            'adv': adv_loss
        }


def train_with_eval(
    model,
    train_loader,
    val_paths,
    scheduler,
    optimizer,
    sqrt_ac,
    sqrt_om,
    device,
    drop_prob=0.1,
    epochs=10,
    use_composite_loss=True,
    use_bridge_diffusion=True,
    use_adversarial_loss=False,
    discriminator=None,
    disc_optimizer=None,
):
    import torch.nn.functional as F

    def forward_diff(x0, t):
        """Standard forward diffusion (noise-based)."""
        noise = torch.randn_like(x0)
        x = (
            sqrt_ac[t][:, None, None, None] * x0
            + sqrt_om[t][:, None, None, None] * noise
        )
        return x, noise
    
    def forward_bridge(ld, nd, t):
        """Bridge forward diffusion (LD to ND)."""
        return bridge_forward_diffusion(ld, nd, t, sqrt_ac, sqrt_om)

    # Initialize loss function
    if use_composite_loss:
        if use_adversarial_loss and discriminator is not None:
            print("Using composite loss (MSE + L1 + LPIPS + Adversarial)")
            loss_fn = CompositeLoss(device, mse_weight=1.0, l1_weight=0.5, lpips_weight=0.1, 
                                  adv_weight=0.01, discriminator=discriminator)
            adv_loss_fn = AdversarialLoss(loss_type='lsgan')
        else:
            print("Using composite loss (MSE + L1 + LPIPS)")
            loss_fn = CompositeLoss(device, mse_weight=1.0, l1_weight=0.5, lpips_weight=0.1)
            adv_loss_fn = None
    else:
        print("Using MSE loss only")
        loss_fn = None
        adv_loss_fn = None
    
    # Print diffusion mode
    if use_bridge_diffusion:
        print("Using Schrödinger-Bridge diffusion (LD → ND)")
    else:
        print("Using standard diffusion (noise → ND)")

    best_ssim = 0.0
    for ep in range(1, epochs + 1):
        print(f"Epoch {ep}/{epochs}")
        model.train()
        running_total = 0.0
        running_mse = 0.0
        running_l1 = 0.0
        running_lpips = 0.0
        running_adv = 0.0
        running_disc = 0.0
        
        pbar = tqdm(train_loader, desc=f"train {ep}")
        for ld, nd in pbar:
            ld, nd = ld.to(device), nd.to(device)
            B = nd.size(0)
            t = torch.randint(
                0, scheduler.num_train_timesteps, (B,), device=device, dtype=torch.long
            )
            
            # Choose diffusion process
            if use_bridge_diffusion:
                # Bridge diffusion: interpolate from LD to ND
                x_t, target = forward_bridge(ld, nd, t)
            else:
                # Standard diffusion: add noise to ND
                x_t, target = forward_diff(nd, t)  # target is noise
            
            # Classifier-free guidance: randomly drop conditioning
            mask = (torch.rand(B, device=device) < drop_prob).view(B, 1, 1, 1)
            cond = torch.where(mask, torch.zeros_like(ld), ld)
            
            # ===== DISCRIMINATOR TRAINING =====
            disc_loss = torch.tensor(0.0, device=device)
            if use_adversarial_loss and discriminator is not None and disc_optimizer is not None:
                # Generate fake images for discriminator training
                with torch.no_grad():
                    pred_noise = model(x_t, t, cond)
                    if use_bridge_diffusion:
                        # For bridge diffusion, pred is direction, so apply it
                        fake_images = x_t + 0.1 * pred_noise  # Small step toward ND
                    else:
                        # For standard diffusion, reconstruct image from noise prediction
                        fake_images = (x_t - torch.sqrt(1 - scheduler.alphas_cumprod[t[0]]) * pred_noise) / torch.sqrt(scheduler.alphas_cumprod[t[0]])
                    fake_images = torch.clamp(fake_images, -1, 1)
                
                # Train discriminator
                disc_optimizer.zero_grad()
                
                # Real images (ND)
                disc_real = discriminator(nd, ld)  # Conditional discriminator
                disc_loss_real = adv_loss_fn(disc_real, target_is_real=True)
                
                # Fake images
                disc_fake = discriminator(fake_images.detach(), ld)
                disc_loss_fake = adv_loss_fn(disc_fake, target_is_real=False)
                
                # Total discriminator loss
                disc_loss = (disc_loss_real + disc_loss_fake) * 0.5
                disc_loss.backward()
                disc_optimizer.step()
                
                running_disc += disc_loss.item() * B
            
            # ===== GENERATOR TRAINING =====
            # Model prediction
            pred = model(x_t, t, cond)
            
            # Compute loss
            if use_composite_loss:
                # Generate image for adversarial loss
                if use_adversarial_loss and discriminator is not None:
                    if use_bridge_diffusion:
                        generated_image = x_t + 0.1 * pred  # Apply predicted direction
                    else:
                        generated_image = (x_t - torch.sqrt(1 - scheduler.alphas_cumprod[t[0]]) * pred) / torch.sqrt(scheduler.alphas_cumprod[t[0]])
                    generated_image = torch.clamp(generated_image, -1, 1)
                else:
                    generated_image = None
                
                loss_dict = loss_fn(pred, target, generated_image=generated_image, 
                                  real_image=nd, conditioning=ld)
                loss = loss_dict['total']
                
                # Track individual loss components
                running_total += loss.item() * B
                running_mse += loss_dict['mse'].item() * B
                running_l1 += loss_dict['l1'].item() * B
                running_lpips += loss_dict['lpips'].item() * B
                running_adv += loss_dict['adv'].item() * B
                
                mode = "bridge" if use_bridge_diffusion else "std"
                if use_adversarial_loss:
                    pbar.set_postfix(
                        mode=mode,
                        total=f"{loss.item():.4f}",
                        mse=f"{loss_dict['mse'].item():.4f}",
                        l1=f"{loss_dict['l1'].item():.4f}",
                        adv=f"{loss_dict['adv'].item():.4f}",
                        disc=f"{disc_loss.item():.4f}"
                    )
                else:
                    pbar.set_postfix(
                        mode=mode,
                        total=f"{loss.item():.4f}",
                        mse=f"{loss_dict['mse'].item():.4f}",
                        l1=f"{loss_dict['l1'].item():.4f}",
                        lpips=f"{loss_dict['lpips'].item():.4f}"
                    )
            else:
                loss = F.mse_loss(pred, target)
                running_total += loss.item() * B
                mode = "bridge" if use_bridge_diffusion else "std"
                pbar.set_postfix(mode=mode, loss=f"{loss.item():.4f}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        train_loss = running_total / len(train_loader.dataset)
        
        if use_composite_loss:
            train_mse = running_mse / len(train_loader.dataset)
            train_l1 = running_l1 / len(train_loader.dataset)
            train_lpips = running_lpips / len(train_loader.dataset)
            print(f"Train - Total: {train_loss:.4f}, MSE: {train_mse:.4f}, L1: {train_l1:.4f}, LPIPS: {train_lpips:.4f}")
        else:
            print(f"Train loss: {train_loss:.4f}")

        # validation SSIM
        model.eval()
        avg_ssim = evaluate(model, val_paths, scheduler, device, get_transform())
        print(f"Epoch {ep}: train loss {train_loss:.4f} | val SSIM {avg_ssim:.4f}")
        torch.save(
            model.state_dict(),
            f"/content/drive/MyDrive/CT Models/DiffusionNonLatent/diffusion_epoch_{ep}.pth",
        )
    best_ssim = max(best_ssim, avg_ssim)
    print(f"Best SSIM across epochs: {best_ssim:.4f}")


def main():
    """Main training function - only runs when script is executed directly."""
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        beta_schedule="squaredcos_cap_v2",  # or the schedule of your choice
    )
    betas = scheduler.betas  # [T]  tensor
    # Auto-detect device: use CUDA if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    betas = betas.to(device)

    # 2) derive α, cumulative ᾱ, and the two square‑roots
    alphas = 1.0 - betas  # α_t
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # ᾱ_t
    sqrt_ac = torch.sqrt(alphas_cumprod)  # √ᾱ_t   (your √ac)
    sqrt_om = torch.sqrt(1.0 - alphas_cumprod)  # √(1−ᾱ_t) (your √om)

    # load model
    model = PixelDiffusionUNetConditional().to(device)
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=1e-4,  # good starting point for 256² CT slices
        betas=(0.9, 0.95),  # slightly lower β₂ improves noise prediction
        weight_decay=1e-2,  # mild L2 regularisation on weights
    )

    # Create dataset with data augmentations enabled
    ds = CTPairsDataset(
        "/content/drive/MyDrive/CT/", 
        transform=get_transform(),
        enable_augmentations=True,
        noise_injection_prob=0.3,
        spectral_jitter_prob=0.4,
        mixup_prob=0.15,
        patchmix_prob=0.15
    )
    n_val = int(len(ds) * 0.1)
    n_train = len(ds) - n_val

    train_ds, vaal_ds = random_split(ds, [n_train, n_val])

    # Reduce batch size and num_workers for CPU compatibility
    batch_size = 4 if device.type == "cpu" else 12
    num_workers = 0 if device.type == "cpu" else 4
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    val_paths = json.load(open("/content/drive/MyDrive/CT/image_paths.json"))
    len(val_paths[0:100])

    # Reduce epochs for CPU testing
    epochs = 10 if device.type == "cpu" else 140
    
    # Use composite loss for better training quality
    use_composite_loss = True
    print(f"Training with composite loss: {use_composite_loss}")
    
    # Use Schrödinger-Bridge diffusion for LD → ND process
    use_bridge_diffusion = True
    print(f"Training with bridge diffusion: {use_bridge_diffusion}")
    
    train_with_eval(
        model,
        train_loader,
        val_paths[0:100],
        scheduler,
        optimizer,
        sqrt_ac,
        sqrt_om,
        device,
        epochs=epochs,
        use_composite_loss=use_composite_loss,
        use_bridge_diffusion=use_bridge_diffusion,
    )


if __name__ == "__main__":
    main()
