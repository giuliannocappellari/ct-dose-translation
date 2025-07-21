#!/usr/bin/env python3
"""
Test script for Stage 6: Adversarial GAN Loss Implementation
Tests the PatchGAN discriminator and adversarial loss functionality.
"""

import sys
import traceback

import torch
import torch.nn.functional as F
import torch.optim as optim

# Import our modules
from df_without_latent_with_filters import (
    PatchGANDiscriminator,
    AdversarialLoss,
    CompositeLoss,
    PixelDiffusionUNetConditional,
)


def test_patchgan_discriminator_initialization():
    """Test PatchGANDiscriminator initialization."""
    print("Testing PatchGANDiscriminator initialization...")
    
    # Test with different configurations
    configs = [
        (2, 64, 3, True),   # Standard conditional discriminator
        (1, 32, 2, False),  # Unconditional, smaller
        (2, 128, 4, True),  # Larger discriminator
    ]
    
    for input_ch, base_ch, n_layers, use_spectral_norm in configs:
        disc = PatchGANDiscriminator(input_ch, base_ch, n_layers, use_spectral_norm)
        
        # Check attributes
        assert hasattr(disc, 'model'), "Should have model attribute"
        assert hasattr(disc, 'use_spectral_norm'), "Should have use_spectral_norm attribute"
        assert disc.use_spectral_norm == use_spectral_norm, f"Expected spectral_norm {use_spectral_norm}"
        
        # Check that model is a sequential
        assert isinstance(disc.model, torch.nn.Sequential), "Model should be Sequential"
        
        print(f"  ✓ Config ({input_ch}, {base_ch}, {n_layers}, {use_spectral_norm}) passed")
    
    print("✓ PatchGANDiscriminator initialization test passed")


def test_patchgan_discriminator_forward():
    """Test PatchGANDiscriminator forward pass."""
    print("Testing PatchGANDiscriminator forward pass...")
    
    batch_size = 2
    H, W = 64, 64
    
    # Create discriminator
    disc = PatchGANDiscriminator(input_ch=2, base_ch=64, n_layers=3)
    
    # Test conditional discriminator
    real_img = torch.randn(batch_size, 1, H, W)
    cond_img = torch.randn(batch_size, 1, H, W)
    
    with torch.no_grad():
        output = disc(real_img, cond_img)
    
    # Check output shape (should be patch-based)
    assert len(output.shape) == 4, f"Output should be 4D, got {output.shape}"
    assert output.shape[0] == batch_size, f"Batch size should be {batch_size}, got {output.shape[0]}"
    assert output.shape[1] == 1, f"Output channels should be 1, got {output.shape[1]}"
    
    # Output spatial dimensions should be smaller than input
    assert output.shape[2] < H, f"Output height {output.shape[2]} should be < input height {H}"
    assert output.shape[3] < W, f"Output width {output.shape[3]} should be < input width {W}"
    
    # Test unconditional discriminator
    with torch.no_grad():
        output_uncond = disc(real_img)  # No conditioning
    
    # Should still work but with different input processing
    assert output_uncond.shape[0] == batch_size, "Unconditional output should have correct batch size"
    
    print("✓ PatchGANDiscriminator forward pass test passed")


def test_patchgan_discriminator_gradients():
    """Test gradient flow through PatchGANDiscriminator."""
    print("Testing PatchGANDiscriminator gradient flow...")
    
    batch_size = 1
    H, W = 32, 32
    
    disc = PatchGANDiscriminator(input_ch=2, base_ch=32, n_layers=2)
    
    # Create test inputs with gradients
    real_img = torch.randn(batch_size, 1, H, W, requires_grad=True)
    cond_img = torch.randn(batch_size, 1, H, W, requires_grad=True)
    
    # Forward pass
    output = disc(real_img, cond_img)
    loss = output.mean()
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist
    assert real_img.grad is not None, "Real image should have gradients"
    assert cond_img.grad is not None, "Conditioning image should have gradients"
    
    # Check discriminator parameters have gradients
    has_gradients = False
    for param in disc.parameters():
        if param.grad is not None:
            has_gradients = True
            break
    
    assert has_gradients, "Discriminator should have gradients after backward pass"
    
    print("✓ PatchGANDiscriminator gradient flow test passed")


def test_adversarial_loss_initialization():
    """Test AdversarialLoss initialization."""
    print("Testing AdversarialLoss initialization...")
    
    # Test different loss types
    loss_types = ['lsgan', 'vanilla', 'wgangp']
    
    for loss_type in loss_types:
        adv_loss = AdversarialLoss(loss_type=loss_type)
        
        assert hasattr(adv_loss, 'loss_type'), "Should have loss_type attribute"
        assert adv_loss.loss_type == loss_type, f"Expected loss_type {loss_type}, got {adv_loss.loss_type}"
        
        if loss_type in ['lsgan', 'vanilla']:
            assert hasattr(adv_loss, 'loss_fn'), "Should have loss_fn for LSGAN/vanilla"
        
        print(f"  ✓ Loss type {loss_type} initialized correctly")
    
    print("✓ AdversarialLoss initialization test passed")


def test_adversarial_loss_forward():
    """Test AdversarialLoss forward pass."""
    print("Testing AdversarialLoss forward pass...")
    
    batch_size = 2
    H_patch, W_patch = 8, 8
    
    # Test LSGAN loss
    adv_loss = AdversarialLoss(loss_type='lsgan')
    
    # Create fake discriminator output
    disc_output = torch.randn(batch_size, 1, H_patch, W_patch)
    
    # Test real target
    loss_real = adv_loss(disc_output, target_is_real=True)
    assert isinstance(loss_real, torch.Tensor), "Loss should be a tensor"
    assert loss_real.dim() == 0, "Loss should be scalar"
    assert loss_real.item() >= 0, "Loss should be non-negative"
    
    # Test fake target
    loss_fake = adv_loss(disc_output, target_is_real=False)
    assert isinstance(loss_fake, torch.Tensor), "Loss should be a tensor"
    assert loss_fake.dim() == 0, "Loss should be scalar"
    assert loss_fake.item() >= 0, "Loss should be non-negative"
    
    # Test WGAN-GP loss
    adv_loss_wgan = AdversarialLoss(loss_type='wgangp')
    
    loss_real_wgan = adv_loss_wgan(disc_output, target_is_real=True)
    loss_fake_wgan = adv_loss_wgan(disc_output, target_is_real=False)
    
    # WGAN losses can be negative
    assert isinstance(loss_real_wgan, torch.Tensor), "WGAN loss should be tensor"
    assert isinstance(loss_fake_wgan, torch.Tensor), "WGAN loss should be tensor"
    
    print("✓ AdversarialLoss forward pass test passed")


def test_adversarial_loss_gradients():
    """Test gradient flow through AdversarialLoss."""
    print("Testing AdversarialLoss gradient flow...")
    
    batch_size = 1
    H_patch, W_patch = 4, 4
    
    adv_loss = AdversarialLoss(loss_type='lsgan')
    
    # Create discriminator output with gradients
    disc_output = torch.randn(batch_size, 1, H_patch, W_patch, requires_grad=True)
    
    # Forward pass
    loss = adv_loss(disc_output, target_is_real=True)
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist
    assert disc_output.grad is not None, "Discriminator output should have gradients"
    assert disc_output.grad.shape == disc_output.shape, "Gradient shape should match input"
    
    print("✓ AdversarialLoss gradient flow test passed")


def test_composite_loss_with_adversarial():
    """Test CompositeLoss with adversarial component."""
    print("Testing CompositeLoss with adversarial component...")
    
    device = torch.device("cpu")
    batch_size = 1
    H, W = 32, 32
    
    # Create discriminator
    discriminator = PatchGANDiscriminator(input_ch=2, base_ch=32, n_layers=2)
    
    # Create composite loss with adversarial component
    loss_fn = CompositeLoss(
        device=device,
        mse_weight=1.0,
        l1_weight=0.5,
        lpips_weight=0.0,  # Disable LPIPS for simpler testing
        adv_weight=0.1,
        discriminator=discriminator
    )
    
    # Create test data
    pred = torch.randn(batch_size, 1, H, W)
    target = torch.randn(batch_size, 1, H, W)
    generated_image = torch.randn(batch_size, 1, H, W)
    real_image = torch.randn(batch_size, 1, H, W)
    conditioning = torch.randn(batch_size, 1, H, W)
    
    # Forward pass
    with torch.no_grad():
        loss_dict = loss_fn(pred, target, generated_image=generated_image, 
                          real_image=real_image, conditioning=conditioning)
    
    # Check output structure
    expected_keys = {'total', 'mse', 'l1', 'lpips', 'adv'}
    assert set(loss_dict.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(loss_dict.keys())}"
    
    # Check that all losses are scalars and finite
    for key, value in loss_dict.items():
        assert isinstance(value, torch.Tensor), f"{key} should be a tensor"
        assert value.dim() == 0, f"{key} should be scalar, got shape {value.shape}"
        assert torch.isfinite(value).item(), f"{key} should be finite, got {value.item()}"
    
    # Check that adversarial loss is included in total
    expected_total = (
        1.0 * loss_dict['mse'] + 
        0.5 * loss_dict['l1'] + 
        0.0 * loss_dict['lpips'] +
        0.1 * loss_dict['adv']
    )
    assert torch.allclose(loss_dict['total'], expected_total, atol=1e-6), \
        f"Total loss mismatch: expected {expected_total.item()}, got {loss_dict['total'].item()}"
    
    print("✓ CompositeLoss with adversarial test passed")


def test_gan_training_compatibility():
    """Test that GAN components work together in training scenario."""
    print("Testing GAN training compatibility...")
    
    batch_size = 1
    H, W = 32, 32
    
    # Create models
    generator = PixelDiffusionUNetConditional(
        base_ch=16,
        ch_mults=(1, 2),
        time_dim=32,
        attn_res=(),
        use_cross_attention=False,
        use_icr_head=False
    )
    
    discriminator = PatchGANDiscriminator(input_ch=2, base_ch=16, n_layers=2)
    
    # Create optimizers
    gen_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)
    
    # Create loss functions
    adv_loss_fn = AdversarialLoss(loss_type='lsgan')
    
    # Create test data
    ld_img = torch.randn(batch_size, 1, H, W)
    nd_img = torch.randn(batch_size, 1, H, W)
    t = torch.randint(0, 100, (batch_size,))
    x_t = torch.randn(batch_size, 1, H, W)
    
    # ===== Test Discriminator Training =====
    disc_optimizer.zero_grad()
    
    # Real images
    disc_real = discriminator(nd_img, ld_img)
    disc_loss_real = adv_loss_fn(disc_real, target_is_real=True)
    
    # Fake images
    with torch.no_grad():
        fake_pred = generator(x_t, t, ld_img)
        fake_img = x_t + 0.1 * fake_pred  # Simple reconstruction
        fake_img = torch.clamp(fake_img, -1, 1)
    
    disc_fake = discriminator(fake_img, ld_img)
    disc_loss_fake = adv_loss_fn(disc_fake, target_is_real=False)
    
    disc_loss = (disc_loss_real + disc_loss_fake) * 0.5
    disc_loss.backward()
    disc_optimizer.step()
    
    # ===== Test Generator Training =====
    gen_optimizer.zero_grad()
    
    pred = generator(x_t, t, ld_img)
    
    # Generator adversarial loss
    gen_fake_img = x_t + 0.1 * pred
    gen_fake_img = torch.clamp(gen_fake_img, -1, 1)
    
    disc_gen = discriminator(gen_fake_img, ld_img)
    gen_adv_loss = adv_loss_fn(disc_gen, target_is_real=True)  # Generator wants to fool discriminator
    
    # Add reconstruction loss
    recon_loss = F.mse_loss(pred, nd_img - ld_img)  # Example target
    total_gen_loss = recon_loss + 0.1 * gen_adv_loss
    
    total_gen_loss.backward()
    gen_optimizer.step()
    
    # Check that losses are finite
    assert torch.isfinite(disc_loss).item(), "Discriminator loss should be finite"
    assert torch.isfinite(total_gen_loss).item(), "Generator loss should be finite"
    
    # Check that gradients were computed
    gen_has_grads = any(p.grad is not None for p in generator.parameters())
    disc_has_grads = any(p.grad is not None for p in discriminator.parameters())
    
    assert gen_has_grads, "Generator should have gradients"
    assert disc_has_grads, "Discriminator should have gradients"
    
    print("✓ GAN training compatibility test passed")


def test_spectral_normalization():
    """Test spectral normalization in discriminator."""
    print("Testing spectral normalization...")
    
    # Create discriminators with and without spectral norm
    disc_with_sn = PatchGANDiscriminator(input_ch=2, base_ch=32, n_layers=2, use_spectral_norm=True)
    disc_without_sn = PatchGANDiscriminator(input_ch=2, base_ch=32, n_layers=2, use_spectral_norm=False)
    
    # Check that spectral norm is applied correctly
    def has_spectral_norm(module):
        for name, child in module.named_children():
            if hasattr(child, 'weight_orig'):  # Spectral norm creates weight_orig
                return True
            if has_spectral_norm(child):
                return True
        return False
    
    assert has_spectral_norm(disc_with_sn), "Discriminator with spectral norm should have weight_orig attributes"
    assert not has_spectral_norm(disc_without_sn), "Discriminator without spectral norm should not have weight_orig attributes"
    
    # Test forward pass works for both
    batch_size = 1
    H, W = 32, 32
    x = torch.randn(batch_size, 1, H, W)
    y = torch.randn(batch_size, 1, H, W)
    
    with torch.no_grad():
        out_with_sn = disc_with_sn(x, y)
        out_without_sn = disc_without_sn(x, y)
    
    assert torch.isfinite(out_with_sn).all(), "Output with spectral norm should be finite"
    assert torch.isfinite(out_without_sn).all(), "Output without spectral norm should be finite"
    
    print("✓ Spectral normalization test passed")


def run_all_tests():
    """Run all Stage 6 adversarial GAN loss tests."""
    print("=" * 70)
    print("Running Stage 6 Adversarial GAN Loss Tests")
    print("=" * 70)
    
    tests = [
        test_patchgan_discriminator_initialization,
        test_patchgan_discriminator_forward,
        test_patchgan_discriminator_gradients,
        test_adversarial_loss_initialization,
        test_adversarial_loss_forward,
        test_adversarial_loss_gradients,
        test_composite_loss_with_adversarial,
        test_gan_training_compatibility,
        test_spectral_normalization,
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
        print("🎉 All Stage 6 tests passed! Adversarial GAN loss implementation is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
