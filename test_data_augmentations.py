#!/usr/bin/env python3
"""
Test script for Stage 7: Data Augmentation Implementation
Tests the physics-aware noise injection, spectral intensity jitter, and MixUp/PatchMix augmentations.
"""

import sys
import traceback
import tempfile
import os
from pathlib import Path

import torch
import numpy as np
import pydicom
from PIL import Image
import cv2

# Import our modules
from df_data_augmentation import CTPairsDataset
from torchvision import transforms


def create_mock_dicom_file(pixel_array, filepath):
    """Create a mock DICOM file for testing."""
    # Create a minimal DICOM dataset
    ds = pydicom.Dataset()
    ds.file_meta = pydicom.Dataset()
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    ds.file_meta.MediaStorageSOPInstanceUID = "1.2.3"
    ds.file_meta.ImplementationClassUID = "1.2.3.4"
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    ds.SOPInstanceUID = "1.2.3"
    ds.StudyInstanceUID = "1.2.3.4"
    ds.SeriesInstanceUID = "1.2.3.4.5"
    ds.PatientID = "TEST001"
    ds.PatientName = "Test^Patient"
    ds.Modality = "CT"
    
    # Set pixel data
    ds.Rows, ds.Columns = pixel_array.shape
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = pixel_array.astype(np.uint16).tobytes()
    
    # Save to file
    filepath.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(filepath, write_like_original=False)


def setup_test_dataset():
    """Create a temporary test dataset with mock DICOM files."""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create directory structure for both 3mm and 1mm
    quarter_3mm_dir = temp_dir / "quarter_3mm" / "patient001"
    full_3mm_dir = temp_dir / "full_3mm" / "patient001"
    quarter_1mm_dir = temp_dir / "quarter_1mm" / "patient001"
    full_1mm_dir = temp_dir / "full_1mm" / "patient001"
    
    # Create test images (64x64 for speed)
    H, W = 64, 64
    
    # Create 4 pairs of images
    for i in range(4):
        # Low-dose image (more noise)
        ld_img = np.random.randint(100, 400, (H, W), dtype=np.uint16)
        ld_img += np.random.normal(0, 50, (H, W)).astype(np.uint16)
        
        # Full-dose image (less noise, higher contrast)
        nd_img = np.random.randint(200, 800, (H, W), dtype=np.uint16)
        nd_img += np.random.normal(0, 20, (H, W)).astype(np.uint16)
        
        # Add some structure (simulated anatomy)
        y, x = np.ogrid[:H, :W]
        center_y, center_x = H//2, W//2
        mask = (x - center_x)**2 + (y - center_y)**2 < (H//4)**2
        ld_img[mask] += 200
        nd_img[mask] += 300
        
        # Save as DICOM files for 3mm
        ld_path_3mm = quarter_3mm_dir / f"slice_{i:03d}.IMA"
        nd_path_3mm = full_3mm_dir / f"slice_{i:03d}.IMA"
        
        create_mock_dicom_file(ld_img, ld_path_3mm)
        create_mock_dicom_file(nd_img, nd_path_3mm)
        
        # Save as DICOM files for 1mm (same data, different directory)
        ld_path_1mm = quarter_1mm_dir / f"slice_{i:03d}.IMA"
        nd_path_1mm = full_1mm_dir / f"slice_{i:03d}.IMA"
        
        create_mock_dicom_file(ld_img, ld_path_1mm)
        create_mock_dicom_file(nd_img, nd_path_1mm)
    
    return temp_dir


def test_dataset_initialization():
    """Test CTPairsDataset initialization with augmentations."""
    print("Testing CTPairsDataset initialization with augmentations...")
    
    temp_dir = setup_test_dataset()
    
    try:
        # Test with augmentations enabled
        ds = CTPairsDataset(
            temp_dir,
            enable_augmentations=True,
            noise_injection_prob=0.5,
            spectral_jitter_prob=0.5,
            mixup_prob=0.2,
            patchmix_prob=0.2
        )
        
        assert len(ds) > 0, "Dataset should have pairs"
        assert ds.enable_augmentations == True, "Augmentations should be enabled"
        assert ds.noise_injection_prob == 0.5, "Noise injection probability should be set"
        assert ds.spectral_jitter_prob == 0.5, "Spectral jitter probability should be set"
        assert ds.mixup_prob == 0.2, "MixUp probability should be set"
        assert ds.patchmix_prob == 0.2, "PatchMix probability should be set"
        
        # Test with augmentations disabled
        ds_no_aug = CTPairsDataset(temp_dir, enable_augmentations=False)
        assert ds_no_aug.enable_augmentations == False, "Augmentations should be disabled"
        
        print("✓ Dataset initialization test passed")
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)


def test_physics_noise_injection():
    """Test physics-aware noise injection."""
    print("Testing physics-aware noise injection...")
    
    temp_dir = setup_test_dataset()
    
    try:
        ds = CTPairsDataset(temp_dir, enable_augmentations=False)
        
        # Create test images
        img_q = np.random.randint(50, 200, (64, 64), dtype=np.uint8)
        img_f = np.random.randint(100, 255, (64, 64), dtype=np.uint8)
        
        # Apply noise injection
        noisy_img = ds.apply_physics_noise_injection(img_q, img_f)
        
        # Check output properties
        assert noisy_img.shape == img_q.shape, "Output shape should match input"
        assert noisy_img.dtype == np.uint8, "Output should be uint8"
        assert np.all(noisy_img >= 0) and np.all(noisy_img <= 255), "Output should be in valid range"
        
        # Check that noise was actually added (images should be different)
        assert not np.array_equal(noisy_img, img_f), "Noisy image should differ from original"
        
        # Test multiple applications give different results (stochastic)
        noisy_img2 = ds.apply_physics_noise_injection(img_q, img_f)
        assert not np.array_equal(noisy_img, noisy_img2), "Multiple applications should give different results"
        
        print("✓ Physics noise injection test passed")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_spectral_jitter():
    """Test spectral intensity jitter."""
    print("Testing spectral intensity jitter...")
    
    temp_dir = setup_test_dataset()
    
    try:
        ds = CTPairsDataset(temp_dir, enable_augmentations=False)
        
        # Create test images
        img_q = np.random.randint(50, 200, (64, 64), dtype=np.uint8)
        img_f = np.random.randint(100, 255, (64, 64), dtype=np.uint8)
        
        # Apply spectral jitter
        jittered_q, jittered_f = ds.apply_spectral_jitter(img_q, img_f)
        
        # Check output properties
        assert jittered_q.shape == img_q.shape, "Output Q shape should match input"
        assert jittered_f.shape == img_f.shape, "Output F shape should match input"
        assert jittered_q.dtype == np.uint8, "Output Q should be uint8"
        assert jittered_f.dtype == np.uint8, "Output F should be uint8"
        
        # Check valid range
        assert np.all(jittered_q >= 0) and np.all(jittered_q <= 255), "Jittered Q should be in valid range"
        assert np.all(jittered_f >= 0) and np.all(jittered_f <= 255), "Jittered F should be in valid range"
        
        # Test multiple applications give different results
        jittered_q2, jittered_f2 = ds.apply_spectral_jitter(img_q, img_f)
        assert not np.array_equal(jittered_q, jittered_q2), "Multiple applications should give different Q results"
        
        print("✓ Spectral jitter test passed")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_mixup_loading():
    """Test MixUp sample loading."""
    print("Testing MixUp sample loading...")
    
    temp_dir = setup_test_dataset()
    
    try:
        ds = CTPairsDataset(temp_dir, enable_augmentations=False)
        
        # Test loading second sample
        img_q2, img_f2 = ds.apply_mixup(0)  # Load different sample from index 0
        
        # Check output properties
        assert img_q2.shape == (64, 64), "Loaded Q image should have correct shape"
        assert img_f2.shape == (64, 64), "Loaded F image should have correct shape"
        assert img_q2.dtype == np.uint8, "Loaded Q should be uint8"
        assert img_f2.dtype == np.uint8, "Loaded F should be uint8"
        
        # Check valid range
        assert np.all(img_q2 >= 0) and np.all(img_q2 <= 255), "Loaded Q should be in valid range"
        assert np.all(img_f2 >= 0) and np.all(img_f2 <= 255), "Loaded F should be in valid range"
        
        print("✓ MixUp loading test passed")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_patchmix():
    """Test PatchMix augmentation."""
    print("Testing PatchMix augmentation...")
    
    temp_dir = setup_test_dataset()
    
    try:
        ds = CTPairsDataset(temp_dir, enable_augmentations=False)
        
        # Create test images with distinct patterns
        img_q1 = np.zeros((64, 64), dtype=np.uint8)
        img_f1 = np.zeros((64, 64), dtype=np.uint8)
        img_q2 = np.full((64, 64), 255, dtype=np.uint8)
        img_f2 = np.full((64, 64), 255, dtype=np.uint8)
        
        # Apply PatchMix
        mixed_q, mixed_f = ds.apply_patchmix(img_q1, img_f1, img_q2, img_f2)
        
        # Check output properties
        assert mixed_q.shape == img_q1.shape, "Mixed Q shape should match input"
        assert mixed_f.shape == img_f1.shape, "Mixed F shape should match input"
        assert mixed_q.dtype == np.uint8, "Mixed Q should be uint8"
        assert mixed_f.dtype == np.uint8, "Mixed F should be uint8"
        
        # Check that mixing occurred (should have both 0 and 255 values)
        unique_q = np.unique(mixed_q)
        unique_f = np.unique(mixed_f)
        assert len(unique_q) > 1, "Mixed Q should have multiple values"
        assert len(unique_f) > 1, "Mixed F should have multiple values"
        assert 0 in unique_q and 255 in unique_q, "Mixed Q should contain both original values"
        assert 0 in unique_f and 255 in unique_f, "Mixed F should contain both original values"
        
        print("✓ PatchMix test passed")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_full_augmentation_pipeline():
    """Test the complete augmentation pipeline through __getitem__."""
    print("Testing full augmentation pipeline...")
    
    temp_dir = setup_test_dataset()
    
    try:
        # Create dataset with high augmentation probabilities for testing
        ds = CTPairsDataset(
            temp_dir,
            enable_augmentations=True,
            noise_injection_prob=1.0,  # Always apply for testing
            spectral_jitter_prob=1.0,  # Always apply for testing
            mixup_prob=0.5,  # 50% chance
            patchmix_prob=0.5  # 50% chance (mutually exclusive with mixup)
        )
        
        # Test multiple samples to ensure augmentations work
        for i in range(min(len(ds), 4)):
            ld, nd = ds[i]
            
            # Check output properties
            assert isinstance(ld, torch.Tensor), "LD should be tensor"
            assert isinstance(nd, torch.Tensor), "ND should be tensor"
            assert ld.shape == nd.shape, "LD and ND should have same shape"
            assert len(ld.shape) == 3, "Should be 3D tensor (C, H, W)"
            
            # Check value range (should be normalized to [-1, 1])
            assert ld.min() >= -1.1 and ld.max() <= 1.1, "LD should be roughly in [-1, 1] range"
            assert nd.min() >= -1.1 and nd.max() <= 1.1, "ND should be roughly in [-1, 1] range"
            
            # Check that tensors are finite
            assert torch.isfinite(ld).all(), "LD should be finite"
            assert torch.isfinite(nd).all(), "ND should be finite"
        
        # Test with augmentations disabled
        ds_no_aug = CTPairsDataset(temp_dir, enable_augmentations=False)
        ld_no_aug, nd_no_aug = ds_no_aug[0]
        
        # Should still work without augmentations
        assert isinstance(ld_no_aug, torch.Tensor), "LD without aug should be tensor"
        assert isinstance(nd_no_aug, torch.Tensor), "ND without aug should be tensor"
        
        print("✓ Full augmentation pipeline test passed")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_augmentation_probabilities():
    """Test that augmentation probabilities work correctly."""
    print("Testing augmentation probabilities...")
    
    temp_dir = setup_test_dataset()
    
    try:
        # Test with zero probabilities (no augmentations should be applied)
        ds_zero = CTPairsDataset(
            temp_dir,
            enable_augmentations=True,
            noise_injection_prob=0.0,
            spectral_jitter_prob=0.0,
            mixup_prob=0.0,
            patchmix_prob=0.0
        )
        
        # Get original sample
        ld_orig, nd_orig = ds_zero[0]
        
        # Get sample with augmentations disabled
        ds_disabled = CTPairsDataset(temp_dir, enable_augmentations=False)
        ld_disabled, nd_disabled = ds_disabled[0]
        
        # They should be identical (no augmentations applied)
        assert torch.allclose(ld_orig, ld_disabled, atol=1e-6), "Zero probability should match disabled augmentations"
        assert torch.allclose(nd_orig, nd_disabled, atol=1e-6), "Zero probability should match disabled augmentations"
        
        print("✓ Augmentation probabilities test passed")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_augmentation_determinism():
    """Test that augmentations are properly randomized."""
    print("Testing augmentation randomness...")
    
    temp_dir = setup_test_dataset()
    
    try:
        ds = CTPairsDataset(
            temp_dir,
            enable_augmentations=True,
            noise_injection_prob=1.0,  # Always apply
            spectral_jitter_prob=1.0,  # Always apply
            mixup_prob=0.0,  # Disable for deterministic testing
            patchmix_prob=0.0  # Disable for deterministic testing
        )
        
        # Get same sample multiple times
        samples = []
        for _ in range(5):
            ld, nd = ds[0]  # Same index
            samples.append((ld.clone(), nd.clone()))
        
        # Check that samples are different (due to randomness)
        all_same = True
        for i in range(1, len(samples)):
            if not torch.allclose(samples[0][0], samples[i][0], atol=1e-6):
                all_same = False
                break
        
        assert not all_same, "Augmented samples should be different due to randomness"
        
        print("✓ Augmentation randomness test passed")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def run_all_tests():
    """Run all Stage 7 data augmentation tests."""
    print("=" * 70)
    print("Running Stage 7 Data Augmentation Tests")
    print("=" * 70)
    
    tests = [
        test_dataset_initialization,
        test_physics_noise_injection,
        test_spectral_jitter,
        test_mixup_loading,
        test_patchmix,
        test_full_augmentation_pipeline,
        test_augmentation_probabilities,
        test_augmentation_determinism,
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
        print("🎉 All Stage 7 tests passed! Data augmentation implementation is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
