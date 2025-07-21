import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian

sys.path.append(".")

from df_without_latent_with_filters import (
    CTPairsDataset,
    GuidedFilterModule,
    PixelDiffusionUNetConditional,
    extract_edges_canny,
    get_transform,
)


class TestEdgeDetection:
    """Test edge detection functionality."""

    def test_extract_edges_canny_basic(self):
        """Test basic Canny edge detection."""
        # Create a simple test image with clear edges
        img = np.zeros((64, 64), dtype=np.float32)
        img[16:48, 16:48] = 1.0  # White square in center

        edges = extract_edges_canny(img)

        # Check output properties
        assert edges.shape == (64, 64)
        assert edges.dtype == np.float32
        assert edges.min() >= 0.0
        assert edges.max() <= 1.0

        # Should detect edges around the square
        assert edges[15:17, 15:49].sum() > 0  # Top edge
        assert edges[47:49, 15:49].sum() > 0  # Bottom edge
        assert edges[15:49, 15:17].sum() > 0  # Left edge
        assert edges[15:49, 47:49].sum() > 0  # Right edge

    def test_extract_edges_canny_uint8_input(self):
        """Test edge detection with uint8 input."""
        img = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        edges = extract_edges_canny(img)

        assert edges.shape == (32, 32)
        assert edges.dtype == np.float32
        assert edges.min() >= 0.0
        assert edges.max() <= 1.0

    def test_extract_edges_canny_float_input(self):
        """Test edge detection with float input."""
        img = np.random.rand(32, 32).astype(np.float32)
        edges = extract_edges_canny(img)

        assert edges.shape == (32, 32)
        assert edges.dtype == np.float32
        assert edges.min() >= 0.0
        assert edges.max() <= 1.0


class TestGuidedFilterModule:
    """Test Guided Filter Module functionality."""

    def test_gfm_initialization(self):
        """Test GFM initialization."""
        gfm = GuidedFilterModule(skip_ch=64, edge_ch=1)

        # Check that the module has the expected structure
        assert hasattr(gfm, "conv")
        assert len(gfm.conv) == 3  # Conv2d, GroupNorm, Swish

        # Check input/output channels
        conv_layer = gfm.conv[0]
        assert conv_layer.in_channels == 65  # 64 + 1
        assert conv_layer.out_channels == 64

    def test_gfm_forward_same_size(self):
        """Test GFM forward pass with same spatial dimensions."""
        gfm = GuidedFilterModule(skip_ch=64, edge_ch=1)

        # Create test tensors
        skip_feat = torch.randn(2, 64, 32, 32)
        edge_feat = torch.randn(2, 1, 32, 32)

        output = gfm(skip_feat, edge_feat)

        # Check output shape
        assert output.shape == skip_feat.shape
        assert output.shape == (2, 64, 32, 32)

    def test_gfm_forward_different_size(self):
        """Test GFM forward pass with different spatial dimensions."""
        gfm = GuidedFilterModule(skip_ch=128, edge_ch=1)

        # Create test tensors with different spatial sizes
        skip_feat = torch.randn(2, 128, 64, 64)
        edge_feat = torch.randn(2, 1, 32, 32)  # Smaller edge map

        output = gfm(skip_feat, edge_feat)

        # Check output shape matches skip_feat
        assert output.shape == skip_feat.shape
        assert output.shape == (2, 128, 64, 64)

    def test_gfm_gradient_flow(self):
        """Test that gradients flow through GFM."""
        gfm = GuidedFilterModule(skip_ch=32, edge_ch=1)

        skip_feat = torch.randn(1, 32, 16, 16, requires_grad=True)
        edge_feat = torch.randn(1, 1, 16, 16)

        output = gfm(skip_feat, edge_feat)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert skip_feat.grad is not None
        assert skip_feat.grad.shape == skip_feat.shape


class TestPixelDiffusionUNetConditional:
    """Test the enhanced UNet with edge guidance."""

    def test_unet_initialization(self):
        """Test UNet initialization."""
        model = PixelDiffusionUNetConditional(
            base_ch=32,  # Smaller for testing
            ch_mults=(1, 2, 4),
            time_dim=128,
        )

        # Check that GFMs are created
        assert hasattr(model, "gfms")
        assert len(model.gfms) == 2  # For 3-level UNet: 2 skip connections

        # Check GFM channel dimensions
        expected_channels = [32 * 2, 32 * 1]  # Reversed ch_mults[:-1]
        for i, gfm in enumerate(model.gfms):
            conv_layer = gfm.conv[0]
            assert conv_layer.out_channels == expected_channels[i]

    def test_unet_forward_with_edges(self):
        """Test UNet forward pass with edge guidance."""
        model = PixelDiffusionUNetConditional(
            base_ch=32, ch_mults=(1, 2, 4), time_dim=128
        )

        # Create test inputs
        batch_size = 2
        x_t = torch.randn(batch_size, 1, 64, 64)  # Noisy image
        t = torch.randint(0, 1000, (batch_size,))  # Timesteps
        cond = torch.randn(batch_size, 1, 64, 64)  # Low-dose condition
        edge_map = torch.randn(batch_size, 1, 64, 64)  # Edge map

        # Forward pass
        output = model(x_t, t, cond, edge_map)

        # Check output shape
        assert output.shape == (batch_size, 1, 64, 64)

    def test_unet_forward_without_edges(self):
        """Test UNet forward pass without providing edge map (auto-generation)."""
        model = PixelDiffusionUNetConditional(
            base_ch=32, ch_mults=(1, 2, 4), time_dim=128
        )

        # Create test inputs (no edge_map provided)
        batch_size = 1
        x_t = torch.randn(batch_size, 1, 64, 64)
        t = torch.randint(0, 1000, (batch_size,))
        cond = torch.randn(batch_size, 1, 64, 64)

        # Forward pass should work without edge_map
        output = model(x_t, t, cond)

        assert output.shape == (batch_size, 1, 64, 64)

    def test_edge_pyramid_creation(self):
        """Test edge pyramid creation."""
        model = PixelDiffusionUNetConditional(
            base_ch=32, ch_mults=(1, 2, 4), time_dim=128
        )

        # Create test edge map
        edge_full = torch.randn(2, 1, 64, 64)

        # Create edge pyramid
        edge_pyramid = model.create_edge_pyramid(edge_full)

        # Check pyramid structure
        assert len(edge_pyramid) == 2  # For 3-level UNet: 2 skip connections

        # Check scales (should be in decoder order: largest to smallest)
        expected_sizes = [(32, 32), (16, 16)]  # 64/2, 64/4
        for i, edge_map in enumerate(edge_pyramid):
            assert edge_map.shape[-2:] == expected_sizes[i]
            assert edge_map.shape[:2] == (2, 1)  # Batch and channel dims preserved

    def test_gradient_flow_through_gfm(self):
        """Test that gradients flow through the entire model including GFMs."""
        model = PixelDiffusionUNetConditional(
            base_ch=16,  # Very small for fast testing
            ch_mults=(1, 2),
            time_dim=64,
        )

        # Create test inputs
        x_t = torch.randn(1, 1, 32, 32, requires_grad=True)
        t = torch.randint(0, 1000, (1,))
        cond = torch.randn(1, 1, 32, 32)
        edge_map = torch.randn(1, 1, 32, 32)

        # Forward pass
        output = model(x_t, t, cond, edge_map)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert x_t.grad is not None

        # Check that GFM parameters have gradients
        for gfm in model.gfms:
            for param in gfm.parameters():
                assert param.grad is not None


class TestCTPairsDatasetMock:
    """Test dataset functionality with mock DICOM files."""

    def create_mock_dicom(self, pixel_array, filename):
        """Create a mock DICOM file for testing."""
        # Create a minimal DICOM dataset
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        file_meta.MediaStorageSOPInstanceUID = "1.2.3"
        file_meta.ImplementationClassUID = "1.2.3.4"
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)

        # Set required DICOM attributes
        ds.PatientName = "Test^Patient"
        ds.PatientID = "123456"
        ds.StudyInstanceUID = "1.2.3.4.5"
        ds.SeriesInstanceUID = "1.2.3.4.5.6"
        ds.SOPInstanceUID = "1.2.3.4.5.6.7"
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        ds.StudyDate = "20230101"
        ds.StudyTime = "120000"
        ds.Modality = "CT"
        ds.SeriesNumber = 1
        ds.InstanceNumber = 1

        # Set pixel data
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows, ds.Columns = pixel_array.shape
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.PixelData = pixel_array.tobytes()

        return ds

    def test_dataset_with_mock_data(self):
        """Test dataset functionality with mock DICOM files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create directory structure
            quarter_dir = temp_path / "quarter_3mm" / "patient1"
            full_dir = temp_path / "full_3mm" / "patient1"
            quarter_dir.mkdir(parents=True)
            full_dir.mkdir(parents=True)

            # Create mock pixel arrays
            ld_array = np.random.randint(0, 1000, (64, 64), dtype=np.uint16)
            nd_array = np.random.randint(500, 1500, (64, 64), dtype=np.uint16)

            # Create and save mock DICOM files
            ld_file = quarter_dir / "slice1.dcm"
            nd_file = full_dir / "slice1.dcm"

            ld_ds = self.create_mock_dicom(ld_array, str(ld_file))
            nd_ds = self.create_mock_dicom(nd_array, str(nd_file))

            ld_ds.save_as(str(ld_file))
            nd_ds.save_as(str(nd_file))

            # Test dataset
            dataset = CTPairsDataset(temp_path, transform=get_transform())

            # Check that pairs were found
            assert len(dataset) == 1

            # Test __getitem__
            ld, nd, edge_map = dataset[0]

            # Check output shapes and types
            assert isinstance(ld, torch.Tensor)
            assert isinstance(nd, torch.Tensor)
            assert isinstance(edge_map, torch.Tensor)

            # Check tensor properties
            assert ld.shape == (1, 256, 256)  # CenterCrop to 256x256
            assert nd.shape == (1, 256, 256)
            assert edge_map.shape == (1, 256, 256)

            # Check value ranges (after normalization to [-1, 1])
            assert ld.min() >= -1.0 and ld.max() <= 1.0
            assert nd.min() >= -1.0 and nd.max() <= 1.0
            assert edge_map.min() >= -1.0 and edge_map.max() <= 1.0


class TestIntegration:
    """Integration tests for the complete pipeline."""

    def test_full_pipeline_cpu(self):
        """Test the complete pipeline on CPU."""
        # Create small model for testing
        model = PixelDiffusionUNetConditional(base_ch=16, ch_mults=(1, 2), time_dim=64)

        # Create test batch
        batch_size = 2
        x_t = torch.randn(batch_size, 1, 32, 32)
        t = torch.randint(0, 1000, (batch_size,))
        cond = torch.randn(batch_size, 1, 32, 32)

        # Test forward pass
        with torch.no_grad():
            output = model(x_t, t, cond)

        assert output.shape == (batch_size, 1, 32, 32)

        # Test that model can be set to train/eval mode
        model.train()
        model.eval()

    def test_edge_guidance_improves_features(self):
        """Test that edge guidance actually affects the output."""
        model = PixelDiffusionUNetConditional(base_ch=16, ch_mults=(1, 2), time_dim=64)

        # Create test inputs
        x_t = torch.randn(1, 1, 32, 32)
        t = torch.randint(0, 1000, (1,))
        cond = torch.randn(1, 1, 32, 32)

        # Test with and without edge guidance
        with torch.no_grad():
            # Without edges (auto-generated)
            output_auto = model(x_t, t, cond)

            # With explicit edge map
            edge_map = torch.randn(1, 1, 32, 32)
            output_guided = model(x_t, t, cond, edge_map)

        # Outputs should be different (edge guidance should affect result)
        assert not torch.allclose(output_auto, output_guided, atol=1e-6)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
