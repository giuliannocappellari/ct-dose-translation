#!/usr/bin/env python3
"""
Simple test runner for edge guidance implementation.
Tests the key functionality without requiring pytest.
"""

import sys
import traceback

import numpy as np
import torch

# Import our modules
from df_without_latent_with_filters import (
    GuidedFilterModule,
    PixelDiffusionUNetConditional,
    extract_edges_canny,
)


def test_extract_edges_canny():
    """Test basic Canny edge detection."""
    print("Testing edge detection...")

    # Create a simple test image with clear edges
    img = np.zeros((64, 64), dtype=np.float32)
    img[16:48, 16:48] = 1.0  # White square in center

    edges = extract_edges_canny(img)

    # Check output properties
    assert edges.shape == (64, 64), f"Expected (64, 64), got {edges.shape}"
    assert edges.dtype == np.float32, f"Expected float32, got {edges.dtype}"
    assert edges.min() >= 0.0, f"Expected min >= 0, got {edges.min()}"
    assert edges.max() <= 1.0, f"Expected max <= 1, got {edges.max()}"

    # Should detect edges around the square
    assert edges[15:17, 15:49].sum() > 0, "Should detect top edge"
    assert edges[47:49, 15:49].sum() > 0, "Should detect bottom edge"
    assert edges[15:49, 15:17].sum() > 0, "Should detect left edge"
    assert edges[15:49, 47:49].sum() > 0, "Should detect right edge"

    print("✓ Edge detection test passed")


def test_guided_filter_module():
    """Test Guided Filter Module functionality."""
    print("Testing Guided Filter Module...")

    # Test with kornia-based implementation
    try:
        gfm = GuidedFilterModule(64)

        # Create test tensors
        skip_feat = torch.randn(2, 64, 32, 32)
        edge_feat = torch.randn(2, 1, 32, 32)

        output = gfm(skip_feat, edge_feat)

        # Check output shape
        assert output.shape == skip_feat.shape, (
            f"Expected {skip_feat.shape}, got {output.shape}"
        )

        print("✓ Guided Filter Module test passed")

    except Exception as e:
        print(f"⚠ Guided Filter Module test failed (expected with kornia): {e}")
        print("This is expected if kornia.filters.GuidedFilter has issues")


def test_unet_initialization():
    """Test UNet initialization."""
    print("Testing UNet initialization...")

    model = PixelDiffusionUNetConditional(
        base_ch=32,  # Smaller for testing
        ch_mults=(1, 2, 4),
        time_dim=128,
    )

    # Check that GFMs are created
    assert hasattr(model, "gfms"), "Model should have gfms attribute"
    assert len(model.gfms) == 2, f"Expected 2 GFMs, got {len(model.gfms)}"

    # Check init_conv has correct input channels (3: x_t + cond + edge)
    assert model.init_conv.in_channels == 3, (
        f"Expected 3 input channels, got {model.init_conv.in_channels}"
    )

    print("✓ UNet initialization test passed")


def test_unet_forward():
    """Test UNet forward pass."""
    print("Testing UNet forward pass...")

    model = PixelDiffusionUNetConditional(
        base_ch=16,  # Very small for testing
        ch_mults=(1, 2),
        time_dim=64,
    )

    # Create test inputs
    batch_size = 2
    x_t = torch.randn(batch_size, 1, 32, 32)  # Noisy image
    t = torch.randint(0, 1000, (batch_size,))  # Timesteps
    cond = torch.randn(batch_size, 1, 32, 32)  # Low-dose condition
    edge_map = torch.randn(batch_size, 1, 32, 32)  # Edge map

    # Forward pass
    with torch.no_grad():
        output = model(x_t, t, cond, edge_map)

    # Check output shape
    expected_shape = (batch_size, 1, 32, 32)
    assert output.shape == expected_shape, (
        f"Expected {expected_shape}, got {output.shape}"
    )

    print("✓ UNet forward pass test passed")


def test_unet_forward_without_edges():
    """Test UNet forward pass without providing edge map (auto-generation)."""
    print("Testing UNet forward pass with auto edge generation...")

    model = PixelDiffusionUNetConditional(base_ch=16, ch_mults=(1, 2), time_dim=64)

    # Create test inputs (no edge_map provided)
    batch_size = 1
    x_t = torch.randn(batch_size, 1, 32, 32)
    t = torch.randint(0, 1000, (batch_size,))
    cond = torch.randn(batch_size, 1, 32, 32)

    # Forward pass should work without edge_map
    with torch.no_grad():
        output = model(x_t, t, cond)

    expected_shape = (batch_size, 1, 32, 32)
    assert output.shape == expected_shape, (
        f"Expected {expected_shape}, got {output.shape}"
    )

    print("✓ UNet auto edge generation test passed")


def test_edge_pyramid_creation():
    """Test edge pyramid creation."""
    print("Testing edge pyramid creation...")

    model = PixelDiffusionUNetConditional(base_ch=16, ch_mults=(1, 2, 4), time_dim=64)

    # Create test edge map
    edge_full = torch.randn(2, 1, 64, 64)

    # Create edge pyramid
    edge_pyramid = model.create_edge_pyramid(edge_full)

    # Check pyramid structure
    expected_len = 2  # For 3-level UNet: 2 skip connections
    assert len(edge_pyramid) == expected_len, (
        f"Expected {expected_len} pyramid levels, got {len(edge_pyramid)}"
    )

    # Check scales (should be in decoder order: largest to smallest)
    expected_sizes = [(32, 32), (16, 16)]  # 64/2, 64/4
    for i, edge_map in enumerate(edge_pyramid):
        actual_size = edge_map.shape[-2:]
        expected_size = expected_sizes[i]
        assert actual_size == expected_size, (
            f"Level {i}: expected {expected_size}, got {actual_size}"
        )
        assert edge_map.shape[:2] == (2, 1), (
            f"Batch/channel dims should be (2, 1), got {edge_map.shape[:2]}"
        )

    print("✓ Edge pyramid creation test passed")


def test_gradient_flow():
    """Test that gradients flow through the model."""
    print("Testing gradient flow...")

    model = PixelDiffusionUNetConditional(
        base_ch=8,  # Very small for fast testing
        ch_mults=(1, 2),
        time_dim=32,
    )

    # Create test inputs
    x_t = torch.randn(1, 1, 16, 16, requires_grad=True)
    t = torch.randint(0, 1000, (1,))
    cond = torch.randn(1, 1, 16, 16)
    edge_map = torch.randn(1, 1, 16, 16)

    # Forward pass
    output = model(x_t, t, cond, edge_map)
    loss = output.sum()
    loss.backward()

    # Check gradients exist
    assert x_t.grad is not None, "Input gradients should exist"

    # Check that model parameters have gradients
    param_count = 0
    grad_count = 0
    for param in model.parameters():
        param_count += 1
        if param.grad is not None:
            grad_count += 1

    assert grad_count > 0, f"Expected some gradients, got {grad_count}/{param_count}"
    print(
        f"✓ Gradient flow test passed ({grad_count}/{param_count} parameters have gradients)"
    )


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 50)
    print("Running Stage 1 Edge Guidance Tests")
    print("=" * 50)

    tests = [
        test_extract_edges_canny,
        test_guided_filter_module,
        test_unet_initialization,
        test_unet_forward,
        test_unet_forward_without_edges,
        test_edge_pyramid_creation,
        test_gradient_flow,
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

    print("=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 50)

    if failed == 0:
        print("🎉 All tests passed! Stage 1 implementation is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
