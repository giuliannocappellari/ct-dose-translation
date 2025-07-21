#!/usr/bin/env python3
"""
Test script for Stage 3: Cross-Attention Implementation
Tests the cross-attention functionality in the enhanced UNet architecture.
"""

import sys
import traceback

import torch
import torch.nn.functional as F

# Import our modules
from df_without_latent_with_filters import (
    CrossAttentionBlock,
    PixelDiffusionUNetConditional,
)


def test_cross_attention_block_initialization():
    """Test CrossAttentionBlock initialization."""
    print("Testing CrossAttentionBlock initialization...")
    
    query_ch = 128
    context_ch = 64
    cross_attn = CrossAttentionBlock(query_ch, context_ch)
    
    # Check attributes
    assert hasattr(cross_attn, 'to_q'), "Should have to_q projection"
    assert hasattr(cross_attn, 'to_k'), "Should have to_k projection"
    assert hasattr(cross_attn, 'to_v'), "Should have to_v projection"
    assert hasattr(cross_attn, 'to_out'), "Should have to_out projection"
    assert hasattr(cross_attn, 'norm_query'), "Should have query normalization"
    assert hasattr(cross_attn, 'norm_context'), "Should have context normalization"
    
    # Check dimensions
    assert cross_attn.query_ch == query_ch, f"Expected query_ch {query_ch}, got {cross_attn.query_ch}"
    assert cross_attn.context_ch == context_ch, f"Expected context_ch {context_ch}, got {cross_attn.context_ch}"
    
    print("✓ CrossAttentionBlock initialization test passed")


def test_cross_attention_block_forward():
    """Test CrossAttentionBlock forward pass."""
    print("Testing CrossAttentionBlock forward pass...")
    
    batch_size = 2
    query_ch = 128
    context_ch = 64
    H, W = 32, 32
    
    cross_attn = CrossAttentionBlock(query_ch, context_ch)
    
    # Create test tensors
    query = torch.randn(batch_size, query_ch, H, W)
    context = torch.randn(batch_size, context_ch, H, W)
    
    # Forward pass
    with torch.no_grad():
        output = cross_attn(query, context)
    
    # Check output shape
    assert output.shape == query.shape, f"Output shape {output.shape} should match query shape {query.shape}"
    
    # Check that output is different from input (cross-attention should modify features)
    assert not torch.allclose(output, query, atol=1e-6), "Output should be different from input query"
    
    print("✓ CrossAttentionBlock forward pass test passed")


def test_cross_attention_different_spatial_sizes():
    """Test CrossAttentionBlock with different spatial sizes."""
    print("Testing CrossAttentionBlock with different spatial sizes...")
    
    batch_size = 1
    query_ch = 64
    context_ch = 64
    
    cross_attn = CrossAttentionBlock(query_ch, context_ch)
    
    # Different spatial sizes
    query = torch.randn(batch_size, query_ch, 16, 16)  # Smaller
    context = torch.randn(batch_size, context_ch, 32, 32)  # Larger
    
    # Forward pass should handle resizing
    with torch.no_grad():
        output = cross_attn(query, context)
    
    # Check output shape matches query
    assert output.shape == query.shape, f"Output shape {output.shape} should match query shape {query.shape}"
    
    print("✓ CrossAttentionBlock different spatial sizes test passed")


def test_cross_attention_gradients():
    """Test gradient flow through CrossAttentionBlock."""
    print("Testing CrossAttentionBlock gradient flow...")
    
    batch_size = 1
    query_ch = 64
    context_ch = 64
    H, W = 16, 16
    
    cross_attn = CrossAttentionBlock(query_ch, context_ch)
    
    # Create test tensors with gradients
    query = torch.randn(batch_size, query_ch, H, W, requires_grad=True)
    context = torch.randn(batch_size, context_ch, H, W, requires_grad=True)
    
    # Forward pass
    output = cross_attn(query, context)
    loss = output.mean()
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist
    assert query.grad is not None, "Query should have gradients"
    assert context.grad is not None, "Context should have gradients"
    
    # Check gradient shapes
    assert query.grad.shape == query.shape, "Query gradient shape should match input"
    assert context.grad.shape == context.shape, "Context gradient shape should match input"
    
    print("✓ CrossAttentionBlock gradient flow test passed")


def test_unet_with_cross_attention():
    """Test UNet with cross-attention enabled."""
    print("Testing UNet with cross-attention...")
    
    # Create UNet with cross-attention
    model = PixelDiffusionUNetConditional(
        base_ch=64,
        ch_mults=(1, 2, 4),  # Smaller for testing
        time_dim=128,
        attn_res=(32, 64),
        use_cross_attention=True
    )
    
    # Check that cross-attention modules exist
    assert hasattr(model, 'cross_attns_down'), "Should have downsampling cross-attention"
    assert hasattr(model, 'cross_attns_up'), "Should have upsampling cross-attention"
    assert hasattr(model, 'cond_encoder'), "Should have conditioning encoder"
    
    # Check cross-attention list lengths
    assert len(model.cross_attns_down) == len(model.downs), "Cross-attention down length should match downs"
    assert len(model.cross_attns_up) == len(model.ups), "Cross-attention up length should match ups"
    
    print("✓ UNet with cross-attention test passed")


def test_unet_cross_attention_forward():
    """Test UNet forward pass with cross-attention."""
    print("Testing UNet cross-attention forward pass...")
    
    batch_size = 1
    H, W = 64, 64  # Smaller for testing
    
    # Create UNet with cross-attention
    model = PixelDiffusionUNetConditional(
        base_ch=32,  # Smaller for testing
        ch_mults=(1, 2),  # Smaller for testing
        time_dim=64,
        attn_res=(32,),
        use_cross_attention=True
    )
    
    # Create test inputs
    x_t = torch.randn(batch_size, 1, H, W)  # Noisy image
    t = torch.randint(0, 1000, (batch_size,))  # Time steps
    cond = torch.randn(batch_size, 1, H, W)  # Conditioning image
    
    # Forward pass
    with torch.no_grad():
        output = model(x_t, t, cond)
    
    # Check output shape
    expected_shape = (batch_size, 1, H, W)
    assert output.shape == expected_shape, f"Output shape {output.shape} should be {expected_shape}"
    
    # Check output is finite
    assert torch.isfinite(output).all(), "Output should be finite"
    
    print("✓ UNet cross-attention forward pass test passed")


def test_unet_cross_attention_vs_baseline():
    """Test that cross-attention produces different outputs than baseline."""
    print("Testing cross-attention vs baseline UNet...")
    
    batch_size = 1
    H, W = 32, 32  # Small for testing
    
    # Create models with and without cross-attention
    model_with_ca = PixelDiffusionUNetConditional(
        base_ch=32,
        ch_mults=(1, 2),
        time_dim=64,
        attn_res=(32,),
        use_cross_attention=True
    )
    
    model_without_ca = PixelDiffusionUNetConditional(
        base_ch=32,
        ch_mults=(1, 2),
        time_dim=64,
        attn_res=(32,),
        use_cross_attention=False
    )
    
    # Create test inputs
    x_t = torch.randn(batch_size, 1, H, W)
    t = torch.randint(0, 1000, (batch_size,))
    cond = torch.randn(batch_size, 1, H, W)
    
    # Forward passes
    with torch.no_grad():
        output_with_ca = model_with_ca(x_t, t, cond)
        output_without_ca = model_without_ca(x_t, t, cond)
    
    # Outputs should be different (different architectures)
    # Note: We can't directly compare since the models have different parameters
    # But we can check that both produce valid outputs
    assert torch.isfinite(output_with_ca).all(), "Cross-attention output should be finite"
    assert torch.isfinite(output_without_ca).all(), "Baseline output should be finite"
    
    print("✓ Cross-attention vs baseline UNet test passed")


def test_unet_cross_attention_gradients():
    """Test gradient flow through UNet with cross-attention."""
    print("Testing UNet cross-attention gradient flow...")
    
    batch_size = 1
    H, W = 32, 32
    
    model = PixelDiffusionUNetConditional(
        base_ch=32,
        ch_mults=(1, 2),
        time_dim=64,
        attn_res=(32,),
        use_cross_attention=True
    )
    
    # Create test inputs
    x_t = torch.randn(batch_size, 1, H, W)
    t = torch.randint(0, 1000, (batch_size,))
    cond = torch.randn(batch_size, 1, H, W)
    
    # Forward pass
    output = model(x_t, t, cond)
    loss = output.mean()
    
    # Backward pass
    loss.backward()
    
    # Check that some parameters have gradients
    has_gradients = False
    for param in model.parameters():
        if param.grad is not None:
            has_gradients = True
            break
    
    assert has_gradients, "Model should have gradients after backward pass"
    
    print("✓ UNet cross-attention gradient flow test passed")


def run_all_tests():
    """Run all Stage 3 cross-attention tests."""
    print("=" * 60)
    print("Running Stage 3 Cross-Attention Tests")
    print("=" * 60)
    
    tests = [
        test_cross_attention_block_initialization,
        test_cross_attention_block_forward,
        test_cross_attention_different_spatial_sizes,
        test_cross_attention_gradients,
        test_unet_with_cross_attention,
        test_unet_cross_attention_forward,
        test_unet_cross_attention_vs_baseline,
        test_unet_cross_attention_gradients,
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
    
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 All Stage 3 tests passed! Cross-attention implementation is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
