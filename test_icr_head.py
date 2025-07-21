#!/usr/bin/env python3
"""
Test script for Stage 4: ICR Head Implementation
Tests the Image Coordinate Refinement head with coordinate channels.
"""

import sys
import traceback

import torch
import torch.nn.functional as F

# Import our modules
from df_without_latent_with_filters import (
    ICRHead,
    PixelDiffusionUNetConditional,
)


def test_icr_head_initialization():
    """Test ICRHead initialization."""
    print("Testing ICRHead initialization...")
    
    in_ch = 64
    out_ch = 1
    
    # Test with coordinates
    icr_with_coords = ICRHead(in_ch, out_ch, use_coordinates=True)
    assert hasattr(icr_with_coords, 'icr_net'), "Should have ICR network"
    assert icr_with_coords.use_coordinates == True, "Should use coordinates"
    assert icr_with_coords.out_ch == out_ch, f"Expected out_ch {out_ch}, got {icr_with_coords.out_ch}"
    
    # Test without coordinates
    icr_without_coords = ICRHead(in_ch, out_ch, use_coordinates=False)
    assert icr_without_coords.use_coordinates == False, "Should not use coordinates"
    
    print("✓ ICRHead initialization test passed")


def test_icr_head_coordinate_creation():
    """Test coordinate channel creation."""
    print("Testing ICRHead coordinate creation...")
    
    batch_size = 2
    in_ch = 32
    H, W = 64, 64
    
    icr = ICRHead(in_ch, use_coordinates=True)
    
    # Create test input
    x = torch.randn(batch_size, in_ch, H, W)
    
    # Create coordinates
    coords = icr.create_coordinate_channels(x)
    
    # Check coordinate shape
    expected_coord_shape = (batch_size, 2, H, W)
    assert coords.shape == expected_coord_shape, f"Coordinate shape {coords.shape} should be {expected_coord_shape}"
    
    # Check coordinate ranges
    assert coords.min() >= -1.0, f"Coordinates should be >= -1, got min {coords.min()}"
    assert coords.max() <= 1.0, f"Coordinates should be <= 1, got max {coords.max()}"
    
    # Check that x and y coordinates are different
    x_coords = coords[:, 0, :, :]  # [B, H, W]
    y_coords = coords[:, 1, :, :]  # [B, H, W]
    
    # X coordinates should vary along width dimension
    assert not torch.allclose(x_coords[:, :, 0], x_coords[:, :, -1]), "X coordinates should vary along width"
    
    # Y coordinates should vary along height dimension
    assert not torch.allclose(y_coords[:, 0, :], y_coords[:, -1, :]), "Y coordinates should vary along height"
    
    print("✓ ICRHead coordinate creation test passed")


def test_icr_head_forward_with_coordinates():
    """Test ICRHead forward pass with coordinates."""
    print("Testing ICRHead forward pass with coordinates...")
    
    batch_size = 2
    in_ch = 64
    out_ch = 1
    H, W = 32, 32
    
    icr = ICRHead(in_ch, out_ch, use_coordinates=True)
    
    # Create test input
    x = torch.randn(batch_size, in_ch, H, W)
    
    # Forward pass
    with torch.no_grad():
        output = icr(x)
    
    # Check output shape
    expected_shape = (batch_size, out_ch, H, W)
    assert output.shape == expected_shape, f"Output shape {output.shape} should be {expected_shape}"
    
    # Check output is finite
    assert torch.isfinite(output).all(), "Output should be finite"
    
    print("✓ ICRHead forward pass with coordinates test passed")


def test_icr_head_forward_without_coordinates():
    """Test ICRHead forward pass without coordinates."""
    print("Testing ICRHead forward pass without coordinates...")
    
    batch_size = 1
    in_ch = 32
    out_ch = 1
    H, W = 16, 16
    
    icr = ICRHead(in_ch, out_ch, use_coordinates=False)
    
    # Create test input
    x = torch.randn(batch_size, in_ch, H, W)
    
    # Forward pass
    with torch.no_grad():
        output = icr(x)
    
    # Check output shape
    expected_shape = (batch_size, out_ch, H, W)
    assert output.shape == expected_shape, f"Output shape {output.shape} should be {expected_shape}"
    
    # Check output is finite
    assert torch.isfinite(output).all(), "Output should be finite"
    
    print("✓ ICRHead forward pass without coordinates test passed")


def test_icr_head_gradients():
    """Test gradient flow through ICRHead."""
    print("Testing ICRHead gradient flow...")
    
    batch_size = 1
    in_ch = 32
    out_ch = 1
    H, W = 16, 16
    
    icr = ICRHead(in_ch, out_ch, use_coordinates=True)
    
    # Create test input with gradients
    x = torch.randn(batch_size, in_ch, H, W, requires_grad=True)
    
    # Forward pass
    output = icr(x)
    loss = output.mean()
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist
    assert x.grad is not None, "Input should have gradients"
    assert x.grad.shape == x.shape, "Gradient shape should match input"
    
    # Check that ICR parameters have gradients
    has_gradients = False
    for param in icr.parameters():
        if param.grad is not None:
            has_gradients = True
            break
    
    assert has_gradients, "ICR head should have gradients after backward pass"
    
    print("✓ ICRHead gradient flow test passed")


def test_icr_head_coordinate_vs_no_coordinate():
    """Test that ICR head with/without coordinates produces different outputs."""
    print("Testing ICRHead coordinate vs no-coordinate outputs...")
    
    batch_size = 1
    in_ch = 32
    out_ch = 1
    H, W = 16, 16
    
    # Create ICR heads with and without coordinates
    icr_with_coords = ICRHead(in_ch, out_ch, use_coordinates=True)
    icr_without_coords = ICRHead(in_ch, out_ch, use_coordinates=False)
    
    # Create test input
    x = torch.randn(batch_size, in_ch, H, W)
    
    # Forward passes
    with torch.no_grad():
        output_with_coords = icr_with_coords(x)
        output_without_coords = icr_without_coords(x)
    
    # Outputs should have same shape
    assert output_with_coords.shape == output_without_coords.shape, "Outputs should have same shape"
    
    # Both should be finite
    assert torch.isfinite(output_with_coords).all(), "Output with coordinates should be finite"
    assert torch.isfinite(output_without_coords).all(), "Output without coordinates should be finite"
    
    print("✓ ICRHead coordinate vs no-coordinate test passed")


def test_unet_with_icr_head():
    """Test UNet with ICR head enabled."""
    print("Testing UNet with ICR head...")
    
    # Create UNet with ICR head
    model = PixelDiffusionUNetConditional(
        base_ch=32,
        ch_mults=(1, 2),  # Smaller for testing
        time_dim=64,
        attn_res=(32,),
        use_cross_attention=True,
        use_icr_head=True
    )
    
    # Check that ICR head is used
    assert hasattr(model, 'use_icr_head'), "Should have use_icr_head attribute"
    assert model.use_icr_head == True, "Should use ICR head"
    assert isinstance(model.final, ICRHead), "Final layer should be ICRHead"
    
    print("✓ UNet with ICR head test passed")


def test_unet_icr_head_forward():
    """Test UNet forward pass with ICR head."""
    print("Testing UNet ICR head forward pass...")
    
    batch_size = 1
    H, W = 32, 32
    
    # Create UNet with ICR head
    model = PixelDiffusionUNetConditional(
        base_ch=32,
        ch_mults=(1, 2),
        time_dim=64,
        attn_res=(32,),
        use_cross_attention=True,
        use_icr_head=True
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
    
    print("✓ UNet ICR head forward pass test passed")


def test_unet_icr_head_vs_baseline():
    """Test UNet with ICR head vs without ICR head."""
    print("Testing UNet ICR head vs baseline...")
    
    batch_size = 1
    H, W = 32, 32
    
    # Create models with and without ICR head
    model_with_icr = PixelDiffusionUNetConditional(
        base_ch=32,
        ch_mults=(1, 2),
        time_dim=64,
        attn_res=(32,),
        use_cross_attention=True,
        use_icr_head=True
    )
    
    model_without_icr = PixelDiffusionUNetConditional(
        base_ch=32,
        ch_mults=(1, 2),
        time_dim=64,
        attn_res=(32,),
        use_cross_attention=True,
        use_icr_head=False
    )
    
    # Create test inputs
    x_t = torch.randn(batch_size, 1, H, W)
    t = torch.randint(0, 1000, (batch_size,))
    cond = torch.randn(batch_size, 1, H, W)
    
    # Forward passes
    with torch.no_grad():
        output_with_icr = model_with_icr(x_t, t, cond)
        output_without_icr = model_without_icr(x_t, t, cond)
    
    # Both outputs should be valid
    assert torch.isfinite(output_with_icr).all(), "ICR output should be finite"
    assert torch.isfinite(output_without_icr).all(), "Baseline output should be finite"
    
    # Shapes should match
    assert output_with_icr.shape == output_without_icr.shape, "Output shapes should match"
    
    print("✓ UNet ICR head vs baseline test passed")


def test_unet_icr_head_gradients():
    """Test gradient flow through UNet with ICR head."""
    print("Testing UNet ICR head gradient flow...")
    
    batch_size = 1
    H, W = 32, 32
    
    model = PixelDiffusionUNetConditional(
        base_ch=32,
        ch_mults=(1, 2),
        time_dim=64,
        attn_res=(32,),
        use_cross_attention=True,
        use_icr_head=True
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
    
    print("✓ UNet ICR head gradient flow test passed")


def run_all_tests():
    """Run all Stage 4 ICR head tests."""
    print("=" * 60)
    print("Running Stage 4 ICR Head Tests")
    print("=" * 60)
    
    tests = [
        test_icr_head_initialization,
        test_icr_head_coordinate_creation,
        test_icr_head_forward_with_coordinates,
        test_icr_head_forward_without_coordinates,
        test_icr_head_gradients,
        test_icr_head_coordinate_vs_no_coordinate,
        test_unet_with_icr_head,
        test_unet_icr_head_forward,
        test_unet_icr_head_vs_baseline,
        test_unet_icr_head_gradients,
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
        print("🎉 All Stage 4 tests passed! ICR head implementation is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
