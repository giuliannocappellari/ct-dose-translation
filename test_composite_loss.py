#!/usr/bin/env python3
"""
Test script for Stage 2: Composite Loss Functions
Tests the enhanced loss function combining MSE, L1, and LPIPS.
"""

import sys
import traceback

import torch
import torch.nn.functional as F

# Import our modules
from df_without_latent_with_filters import CompositeLoss


def test_composite_loss_initialization():
    """Test CompositeLoss initialization."""
    print("Testing CompositeLoss initialization...")
    
    device = torch.device("cpu")  # Use CPU for testing
    loss_fn = CompositeLoss(device, mse_weight=1.0, l1_weight=0.5, lpips_weight=0.1)
    
    # Check that LPIPS network is initialized
    assert hasattr(loss_fn, 'lpips_net'), "Should have lpips_net attribute"
    assert hasattr(loss_fn, 'mse_weight'), "Should have mse_weight attribute"
    assert hasattr(loss_fn, 'l1_weight'), "Should have l1_weight attribute"
    assert hasattr(loss_fn, 'lpips_weight'), "Should have lpips_weight attribute"
    
    # Check weights
    assert loss_fn.mse_weight == 1.0, f"Expected MSE weight 1.0, got {loss_fn.mse_weight}"
    assert loss_fn.l1_weight == 0.5, f"Expected L1 weight 0.5, got {loss_fn.l1_weight}"
    assert loss_fn.lpips_weight == 0.1, f"Expected LPIPS weight 0.1, got {loss_fn.lpips_weight}"
    
    print("✓ CompositeLoss initialization test passed")


def test_composite_loss_forward():
    """Test CompositeLoss forward pass."""
    print("Testing CompositeLoss forward pass...")
    
    device = torch.device("cpu")
    loss_fn = CompositeLoss(device, mse_weight=1.0, l1_weight=0.5, lpips_weight=0.1)
    
    # Create test tensors (single channel, like CT images)
    batch_size = 2
    pred = torch.randn(batch_size, 1, 64, 64)  # Predicted noise
    target = torch.randn(batch_size, 1, 64, 64)  # Target noise
    
    # Forward pass
    with torch.no_grad():  # No gradients needed for testing
        loss_dict = loss_fn(pred, target)
    
    # Check output structure
    expected_keys = {'total', 'mse', 'l1', 'lpips'}
    assert set(loss_dict.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(loss_dict.keys())}"
    
    # Check that all losses are scalars
    for key, value in loss_dict.items():
        assert isinstance(value, torch.Tensor), f"{key} should be a tensor"
        assert value.dim() == 0, f"{key} should be a scalar tensor, got shape {value.shape}"
        assert value.item() >= 0, f"{key} should be non-negative, got {value.item()}"
    
    # Check that total loss is combination of components
    expected_total = (
        1.0 * loss_dict['mse'] + 
        0.5 * loss_dict['l1'] + 
        0.1 * loss_dict['lpips']
    )
    assert torch.allclose(loss_dict['total'], expected_total, atol=1e-6), \
        f"Total loss mismatch: expected {expected_total.item()}, got {loss_dict['total'].item()}"
    
    print("✓ CompositeLoss forward pass test passed")


def test_composite_loss_gradients():
    """Test that gradients flow through CompositeLoss."""
    print("Testing CompositeLoss gradient flow...")
    
    device = torch.device("cpu")
    loss_fn = CompositeLoss(device, mse_weight=1.0, l1_weight=0.5, lpips_weight=0.1)
    
    # Create test tensors with gradients
    batch_size = 1
    pred = torch.randn(batch_size, 1, 32, 32, requires_grad=True)  # Smaller for faster testing
    target = torch.randn(batch_size, 1, 32, 32)
    
    # Forward pass
    loss_dict = loss_fn(pred, target)
    total_loss = loss_dict['total']
    
    # Backward pass
    total_loss.backward()
    
    # Check that gradients exist
    assert pred.grad is not None, "Gradients should exist for pred"
    assert pred.grad.shape == pred.shape, f"Gradient shape mismatch: {pred.grad.shape} vs {pred.shape}"
    
    # Check that LPIPS network parameters don't have gradients (frozen)
    for param in loss_fn.lpips_net.parameters():
        assert param.grad is None, "LPIPS network parameters should be frozen"
    
    print("✓ CompositeLoss gradient flow test passed")


def test_composite_loss_vs_individual():
    """Test that composite loss components match individual loss computations."""
    print("Testing CompositeLoss component accuracy...")
    
    device = torch.device("cpu")
    loss_fn = CompositeLoss(device, mse_weight=1.0, l1_weight=1.0, lpips_weight=0.0)  # No LPIPS for simpler test
    
    # Create test tensors
    batch_size = 1
    pred = torch.randn(batch_size, 1, 32, 32)
    target = torch.randn(batch_size, 1, 32, 32)
    
    # Compute losses
    with torch.no_grad():
        loss_dict = loss_fn(pred, target)
        
        # Compute individual losses manually
        manual_mse = F.mse_loss(pred, target)
        manual_l1 = F.l1_loss(pred, target)
    
    # Check MSE component
    assert torch.allclose(loss_dict['mse'], manual_mse, atol=1e-6), \
        f"MSE mismatch: {loss_dict['mse'].item()} vs {manual_mse.item()}"
    
    # Check L1 component
    assert torch.allclose(loss_dict['l1'], manual_l1, atol=1e-6), \
        f"L1 mismatch: {loss_dict['l1'].item()} vs {manual_l1.item()}"
    
    print("✓ CompositeLoss component accuracy test passed")


def test_composite_loss_different_weights():
    """Test CompositeLoss with different weight configurations."""
    print("Testing CompositeLoss with different weights...")
    
    device = torch.device("cpu")
    
    # Test different weight configurations
    configs = [
        (1.0, 0.0, 0.0),  # MSE only
        (0.0, 1.0, 0.0),  # L1 only
        (1.0, 1.0, 0.0),  # MSE + L1
        (1.0, 0.5, 0.1),  # Full composite
    ]
    
    pred = torch.randn(1, 1, 32, 32)
    target = torch.randn(1, 1, 32, 32)
    
    for mse_w, l1_w, lpips_w in configs:
        loss_fn = CompositeLoss(device, mse_weight=mse_w, l1_weight=l1_w, lpips_weight=lpips_w)
        
        with torch.no_grad():
            loss_dict = loss_fn(pred, target)
        
        # Check that total loss is correct combination
        expected_total = (
            mse_w * loss_dict['mse'] + 
            l1_w * loss_dict['l1'] + 
            lpips_w * loss_dict['lpips']
        )
        
        assert torch.allclose(loss_dict['total'], expected_total, atol=1e-6), \
            f"Weight config ({mse_w}, {l1_w}, {lpips_w}) failed: {loss_dict['total'].item()} vs {expected_total.item()}"
    
    print("✓ CompositeLoss different weights test passed")


def run_all_tests():
    """Run all Stage 2 composite loss tests."""
    print("=" * 50)
    print("Running Stage 2 Composite Loss Tests")
    print("=" * 50)
    
    tests = [
        test_composite_loss_initialization,
        test_composite_loss_forward,
        test_composite_loss_gradients,
        test_composite_loss_vs_individual,
        test_composite_loss_different_weights,
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
        print("🎉 All Stage 2 tests passed! Composite loss implementation is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
