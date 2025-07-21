#!/usr/bin/env python3
"""
Test script for Stage 5: Schrödinger-Bridge Diffusion Implementation
Tests the bridge diffusion process that interpolates from LD to ND images.
"""

import sys
import traceback

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler

# Import our modules
from df_without_latent_with_filters import (
    bridge_forward_diffusion,
    sample_bridge_ddim,
    PixelDiffusionUNetConditional,
)


def test_bridge_forward_diffusion():
    """Test bridge forward diffusion process."""
    print("Testing bridge forward diffusion...")
    
    batch_size = 2
    H, W = 32, 32
    
    # Create test images
    ld_img = torch.randn(batch_size, 1, H, W) * 0.5  # Lower intensity (LD)
    nd_img = torch.randn(batch_size, 1, H, W) * 1.0  # Higher intensity (ND)
    
    # Create scheduler for alpha values
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    sqrt_ac = torch.sqrt(scheduler.alphas_cumprod)
    sqrt_om = torch.sqrt(1 - scheduler.alphas_cumprod)
    
    # Test different time steps
    for t_val in [0, 250, 500, 750, 999]:
        t = torch.full((batch_size,), t_val, dtype=torch.long)
        
        # Forward bridge diffusion
        x_t, target = bridge_forward_diffusion(ld_img, nd_img, t, sqrt_ac, sqrt_om)
        
        # Check output shapes
        assert x_t.shape == ld_img.shape, f"x_t shape {x_t.shape} should match input {ld_img.shape}"
        assert target.shape == ld_img.shape, f"target shape {target.shape} should match input {ld_img.shape}"
        
        # Check that x_t is interpolation between LD and ND
        alpha_t = scheduler.alphas_cumprod[t_val].item()
        expected_x_t = torch.sqrt(torch.tensor(alpha_t)) * nd_img + torch.sqrt(torch.tensor(1 - alpha_t)) * ld_img
        
        assert torch.allclose(x_t, expected_x_t, atol=1e-6), f"x_t should be correct interpolation at t={t_val}"
        
        # Check that target is direction from LD to ND
        expected_target = nd_img - ld_img
        assert torch.allclose(target, expected_target, atol=1e-6), f"target should be LD→ND direction at t={t_val}"
        
        # At t=0 (start), x_t should be close to ND
        if t_val == 0:
            assert torch.allclose(x_t, nd_img, atol=1e-1), "At t=0, x_t should be close to ND"
        
        # At t=999 (end), x_t should be close to LD
        if t_val == 999:
            assert torch.allclose(x_t, ld_img, atol=1e-1), "At t=999, x_t should be close to LD"
    
    print("✓ Bridge forward diffusion test passed")


def test_bridge_forward_diffusion_gradients():
    """Test gradient flow through bridge forward diffusion."""
    print("Testing bridge forward diffusion gradients...")
    
    batch_size = 1
    H, W = 16, 16
    
    # Create test images with gradients
    ld_img = torch.randn(batch_size, 1, H, W, requires_grad=True)
    nd_img = torch.randn(batch_size, 1, H, W, requires_grad=True)
    
    # Create scheduler
    scheduler = DDPMScheduler(num_train_timesteps=100)
    sqrt_ac = torch.sqrt(scheduler.alphas_cumprod)
    sqrt_om = torch.sqrt(1 - scheduler.alphas_cumprod)
    
    t = torch.randint(0, 100, (batch_size,))
    
    # Forward pass
    x_t, target = bridge_forward_diffusion(ld_img, nd_img, t, sqrt_ac, sqrt_om)
    loss = x_t.mean() + target.mean()
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist
    assert ld_img.grad is not None, "LD image should have gradients"
    assert nd_img.grad is not None, "ND image should have gradients"
    
    # Check gradient shapes
    assert ld_img.grad.shape == ld_img.shape, "LD gradient shape should match input"
    assert nd_img.grad.shape == nd_img.shape, "ND gradient shape should match input"
    
    print("✓ Bridge forward diffusion gradients test passed")


def test_sample_bridge_ddim_initialization():
    """Test bridge DDIM sampling initialization."""
    print("Testing bridge DDIM sampling initialization...")
    
    batch_size = 1
    H, W = 32, 32
    
    # Create test inputs
    ld_img = torch.randn(batch_size, 1, H, W)
    
    # Create simple model and scheduler
    model = PixelDiffusionUNetConditional(
        base_ch=32,
        ch_mults=(1, 2),
        time_dim=64,
        attn_res=(32,),
        use_cross_attention=False,
        use_icr_head=False
    )
    
    scheduler = DDPMScheduler(num_train_timesteps=50)  # Small for testing
    
    # Test bridge sampling (should not crash)
    with torch.no_grad():
        output_bridge = sample_bridge_ddim(ld_img, scheduler, model, guidance_scale=1.0, use_bridge=True)
        output_standard = sample_bridge_ddim(ld_img, scheduler, model, guidance_scale=1.0, use_bridge=False)
    
    # Check output shapes
    assert output_bridge.shape == ld_img.shape, f"Bridge output shape {output_bridge.shape} should match input"
    assert output_standard.shape == ld_img.shape, f"Standard output shape {output_standard.shape} should match input"
    
    # Check outputs are finite
    assert torch.isfinite(output_bridge).all(), "Bridge output should be finite"
    assert torch.isfinite(output_standard).all(), "Standard output should be finite"
    
    print("✓ Bridge DDIM sampling initialization test passed")


def test_bridge_vs_standard_diffusion():
    """Test that bridge and standard diffusion produce different results."""
    print("Testing bridge vs standard diffusion...")
    
    batch_size = 1
    H, W = 16, 16
    
    # Create test inputs
    ld_img = torch.randn(batch_size, 1, H, W)
    nd_img = torch.randn(batch_size, 1, H, W)
    
    # Create scheduler
    scheduler = DDPMScheduler(num_train_timesteps=10)  # Very small for testing
    sqrt_ac = torch.sqrt(scheduler.alphas_cumprod)
    sqrt_om = torch.sqrt(1 - scheduler.alphas_cumprod)
    
    t = torch.randint(0, 10, (batch_size,))
    
    # Bridge diffusion
    x_t_bridge, target_bridge = bridge_forward_diffusion(ld_img, nd_img, t, sqrt_ac, sqrt_om)
    
    # Standard diffusion (noise-based)
    noise = torch.randn_like(nd_img)
    sqrt_ac_t = sqrt_ac[t][:, None, None, None]
    sqrt_om_t = sqrt_om[t][:, None, None, None]
    x_t_standard = sqrt_ac_t * nd_img + sqrt_om_t * noise
    target_standard = noise
    
    # Results should be different
    assert not torch.allclose(x_t_bridge, x_t_standard, atol=1e-3), "Bridge and standard x_t should be different"
    assert not torch.allclose(target_bridge, target_standard, atol=1e-3), "Bridge and standard targets should be different"
    
    # Bridge target should be direction vector (ND - LD)
    expected_bridge_target = nd_img - ld_img
    assert torch.allclose(target_bridge, expected_bridge_target, atol=1e-6), "Bridge target should be ND - LD"
    
    print("✓ Bridge vs standard diffusion test passed")


def test_bridge_interpolation_properties():
    """Test mathematical properties of bridge interpolation."""
    print("Testing bridge interpolation properties...")
    
    batch_size = 2
    H, W = 8, 8
    
    # Create test images with known relationship
    ld_img = torch.ones(batch_size, 1, H, W) * 0.2  # Low intensity
    nd_img = torch.ones(batch_size, 1, H, W) * 0.8  # High intensity
    
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    sqrt_ac = torch.sqrt(scheduler.alphas_cumprod)
    sqrt_om = torch.sqrt(1 - scheduler.alphas_cumprod)
    
    # Test at specific time points
    test_times = [0, 250, 500, 750, 999]
    
    for t_val in test_times:
        t = torch.full((batch_size,), t_val, dtype=torch.long)
        x_t, target = bridge_forward_diffusion(ld_img, nd_img, t, sqrt_ac, sqrt_om)
        
        alpha_t = scheduler.alphas_cumprod[t_val].item()
        
        # Check interpolation bounds (accounting for sqrt scaling)
        # x_t = sqrt(α_t) * ND + sqrt(1-α_t) * LD
        sqrt_alpha_t = torch.sqrt(torch.tensor(alpha_t))
        sqrt_one_minus_alpha_t = torch.sqrt(torch.tensor(1 - alpha_t))
        
        # Expected bounds based on the interpolation formula
        expected_min = (sqrt_alpha_t * nd_img.min() + sqrt_one_minus_alpha_t * ld_img.min()).item()
        expected_max = (sqrt_alpha_t * nd_img.max() + sqrt_one_minus_alpha_t * ld_img.max()).item()
        
        # Allow some tolerance for numerical precision
        tolerance = 1e-5
        assert x_t.min().item() >= expected_min - tolerance, f"x_t min should be >= {expected_min} at t={t_val}"
        assert x_t.max().item() <= expected_max + tolerance, f"x_t max should be <= {expected_max} at t={t_val}"
        
        # Target should always be the same (direction from LD to ND)
        expected_target = nd_img - ld_img
        assert torch.allclose(target, expected_target, atol=1e-6), f"Target should be constant at t={t_val}"
    
    print("✓ Bridge interpolation properties test passed")


def test_bridge_training_compatibility():
    """Test that bridge diffusion is compatible with training loop."""
    print("Testing bridge training compatibility...")
    
    batch_size = 1
    H, W = 16, 16
    
    # Create model
    model = PixelDiffusionUNetConditional(
        base_ch=16,
        ch_mults=(1, 2),
        time_dim=32,
        attn_res=(),  # No attention for speed
        use_cross_attention=False,
        use_icr_head=False
    )
    
    # Create test data
    ld_img = torch.randn(batch_size, 1, H, W)
    nd_img = torch.randn(batch_size, 1, H, W)
    
    # Create scheduler
    scheduler = DDPMScheduler(num_train_timesteps=100)
    sqrt_ac = torch.sqrt(scheduler.alphas_cumprod)
    sqrt_om = torch.sqrt(1 - scheduler.alphas_cumprod)
    
    t = torch.randint(0, 100, (batch_size,))
    
    # Forward bridge diffusion
    x_t, target = bridge_forward_diffusion(ld_img, nd_img, t, sqrt_ac, sqrt_om)
    
    # Model prediction
    pred = model(x_t, t, ld_img)  # Use LD as conditioning
    
    # Compute loss
    loss = F.mse_loss(pred, target)
    
    # Backward pass
    loss.backward()
    
    # Check that model has gradients
    has_gradients = False
    for param in model.parameters():
        if param.grad is not None:
            has_gradients = True
            break
    
    assert has_gradients, "Model should have gradients after bridge training step"
    assert torch.isfinite(loss).item(), "Loss should be finite"
    
    print("✓ Bridge training compatibility test passed")


def test_bridge_time_consistency():
    """Test that bridge diffusion is consistent across time steps."""
    print("Testing bridge time consistency...")
    
    batch_size = 1
    H, W = 8, 8
    
    # Create test images
    ld_img = torch.randn(batch_size, 1, H, W)
    nd_img = torch.randn(batch_size, 1, H, W)
    
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    sqrt_ac = torch.sqrt(scheduler.alphas_cumprod)
    sqrt_om = torch.sqrt(1 - scheduler.alphas_cumprod)
    
    # Test monotonic interpolation property
    prev_alpha = 1.0
    for t_val in [0, 100, 300, 500, 700, 900, 999]:
        t = torch.full((batch_size,), t_val, dtype=torch.long)
        x_t, target = bridge_forward_diffusion(ld_img, nd_img, t, sqrt_ac, sqrt_om)
        
        current_alpha = scheduler.alphas_cumprod[t_val].item()
        
        # Alpha should decrease over time (more LD influence as t increases)
        assert current_alpha <= prev_alpha + 1e-6, f"Alpha should decrease: {current_alpha} <= {prev_alpha} at t={t_val}"
        prev_alpha = current_alpha
        
        # Target should always be the same
        expected_target = nd_img - ld_img
        assert torch.allclose(target, expected_target, atol=1e-6), f"Target should be constant at t={t_val}"
    
    print("✓ Bridge time consistency test passed")


def run_all_tests():
    """Run all Stage 5 bridge diffusion tests."""
    print("=" * 70)
    print("Running Stage 5 Schrödinger-Bridge Diffusion Tests")
    print("=" * 70)
    
    tests = [
        test_bridge_forward_diffusion,
        test_bridge_forward_diffusion_gradients,
        test_sample_bridge_ddim_initialization,
        test_bridge_vs_standard_diffusion,
        test_bridge_interpolation_properties,
        test_bridge_training_compatibility,
        test_bridge_time_consistency,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("🎉 All Stage 5 tests passed! Bridge diffusion implementation is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
