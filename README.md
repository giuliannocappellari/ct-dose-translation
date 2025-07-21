# Enhanced CT Pixel-Space Diffusion Model for Low-Dose Denoising

This repository contains an advanced implementation of a pixel-space diffusion model specifically designed for CT image denoising, transforming low-dose (noisy) CT images into high-quality, full-dose equivalents. The model incorporates state-of-the-art techniques including edge-guided filtering, cross-attention mechanisms, adversarial training, and physics-aware data augmentations.

## 🎯 Project Overview

### Problem Statement
Low-dose CT imaging reduces radiation exposure but introduces significant noise that degrades image quality and diagnostic accuracy. This project addresses this challenge by developing a sophisticated diffusion-based denoising model that preserves anatomical structures while effectively removing noise.

### Key Innovations
- **Edge-Guided Filtering**: Multi-scale edge prompts with Guided Filter Modules (GFM)
- **Schrödinger-Bridge Diffusion**: Direct LD→ND transformation instead of noise→ND
- **Cross-Attention Conditioning**: Global attention between noisy input and conditioning features
- **ICR Head**: Image Coordinate Refinement for spatial awareness
- **Adversarial Training**: PatchGAN discriminator for enhanced realism
- **Physics-Aware Augmentations**: CT-specific data augmentations based on imaging physics

## 📁 Repository Structure

```
ct-translation/
├── df_without_latent_with_filters.py    # Core enhanced model (Stages 1-6)
├── df_data_augmentation.py              # Model with data augmentations (Stage 7)
├── changes.md                           # Detailed technical specifications
├── README.md                           # This documentation
├── test_*.py                           # Comprehensive test suites
└── requirements.txt                    # Dependencies
```

## 🏗️ Architecture Overview

### Core Components

#### 1. **Enhanced UNet Architecture** (`PixelDiffusionUNetConditional`)
- **Base Architecture**: U-Net with timestep conditioning
- **Input Channels**: 3 (noisy image + conditioning image + edge map)
- **Adaptive GroupNorm**: Handles varying channel sizes (1-32 groups)
- **Multi-scale Processing**: Downsampling and upsampling paths with skip connections

#### 2. **Edge-Guided Filtering System**
```python
class GuidedFilterModule(nn.Module):
    """Applies edge-guided filtering to skip connections"""
    # Combines skip features with edge information
    # Uses guided filtering for structure preservation
```

#### 3. **Cross-Attention Mechanism**
```python
class CrossAttentionBlock(nn.Module):
    """Global attention between noisy input and conditioning features"""
    # Query: Noisy input features
    # Key/Value: Conditioning image features
    # Enables global context integration
```

#### 4. **ICR Head (Image Coordinate Refinement)**
```python
class ICRHead(nn.Module):
    """Spatial coordinate-aware output refinement"""
    # Adds normalized coordinate channels (x, y) ∈ [-1, 1]
    # Enhances spatial awareness for better reconstruction
```

#### 5. **PatchGAN Discriminator**
```python
class PatchGANDiscriminator(nn.Module):
    """Conditional discriminator for adversarial training"""
    # Patch-based discrimination for texture realism
    # Spectral normalization for training stability
```

## 🚀 Implementation Stages

### Stage 1: Edge-Guided Filtering
**File**: `df_without_latent_with_filters.py`
- **Components**: `GuidedFilterModule`, edge pyramid creation
- **Purpose**: Preserve anatomical structures during denoising
- **Key Features**:
  - Multi-scale edge detection using Canny
  - Guided filter application to skip connections
  - Edge-conditioned feature fusion

### Stage 2: Composite Loss Functions
**Components**: `CompositeLoss` class
- **MSE Loss**: Pixel-wise reconstruction accuracy
- **L1 Loss**: Robust to outliers, preserves edges
- **LPIPS Loss**: Perceptual similarity (optional)
- **Weights**: MSE(1.0) + L1(0.5) + LPIPS(0.1)

### Stage 3: Cross-Attention Integration
**Components**: `CrossAttentionBlock`, `CrossAttentionIdentity`
- **Global Attention**: Between noisy input and conditioning features
- **Multi-Head**: Configurable attention heads
- **Integration**: Applied at key resolutions in U-Net

### Stage 4: ICR Head Implementation
**Components**: `ICRHead` module
- **Coordinate Channels**: Normalized spatial coordinates
- **Refinement**: Final output enhancement with spatial awareness
- **Integration**: Replaces standard final convolution layer

### Stage 5: Schrödinger-Bridge Diffusion
**Components**: `forward_bridge`, `bridge_ddim_sample`
- **Bridge Process**: LD → ND instead of noise → ND
- **Training**: Direction vector targets (ND - LD)
- **Sampling**: Iterative refinement starting from LD images

### Stage 6: Adversarial Training
**Components**: `PatchGANDiscriminator`, `AdversarialLoss`
- **GAN Loss Types**: LSGAN, Vanilla GAN, WGAN-GP
- **Training**: Alternating generator/discriminator optimization
- **Integration**: Added to composite loss with weight 0.01

### Stage 7: Data Augmentations
**File**: `df_data_augmentation.py`
- **Physics-Aware Noise**: Poisson noise modeling
- **Spectral Jitter**: Gamma/brightness/contrast variations
- **MixUp/PatchMix**: Sample mixing for regularization

## 📊 Data Augmentation Details

### 1. Physics-Aware Noise Injection
```python
# Simulates different dose levels using photon statistics
N0 = 10000  # Base photon count
intensity = np.clip(image, 1e-6, 1)
counts = np.random.poisson(intensity * N0)
noisy_image = counts / N0
```

### 2. Spectral Intensity Jitter
```python
# Gamma correction: γ ∈ [0.8, 1.2]
image = image ** gamma
# Brightness/contrast: α ∈ [0.9, 1.1], β ∈ [-0.05, 0.05]
image = image * alpha + beta
# Optional histogram matching (10% probability)
```

### 3. MixUp & PatchMix
```python
# MixUp: Linear blending
lambda_mix = uniform(0.3, 0.7)
mixed = lambda_mix * image1 + (1 - lambda_mix) * image2

# PatchMix: Spatial patch replacement
patch_size = uniform(0.25, 0.5) * image_size
image1[y:y+ph, x:x+pw] = image2[y:y+ph, x:x+pw]
```

## 🔧 Usage

### Basic Training
```python
from df_data_augmentation import PixelDiffusionUNetConditional, CTPairsDataset

# Create model with all enhancements
model = PixelDiffusionUNetConditional(
    base_ch=64,
    ch_mults=(1, 2, 4, 8),
    use_cross_attention=True,
    use_icr_head=True
)

# Create dataset with augmentations
dataset = CTPairsDataset(
    "/path/to/ct/data",
    enable_augmentations=True,
    noise_injection_prob=0.3,
    spectral_jitter_prob=0.4,
    mixup_prob=0.15,
    patchmix_prob=0.15
)

# Train with all features
train_with_eval(
    model=model,
    train_loader=train_loader,
    use_composite_loss=True,
    use_bridge_diffusion=True,
    use_adversarial_loss=True,
    discriminator=discriminator
)
```

### Inference
```python
# Load trained model
model = load_checkpoint(model, "model.pth", device)

# Denoise low-dose CT image
with torch.no_grad():
    denoised = bridge_ddim_sample(
        model, ld_image, scheduler, 
        num_steps=50, device=device
    )
```

## 📈 Performance Features

### Training Optimizations
- **Adaptive Batch Size**: Automatically adjusts for CPU/GPU
- **Mixed Precision**: Optional FP16 training support
- **Gradient Clipping**: Prevents training instability
- **Learning Rate Scheduling**: Cosine annealing with warmup

### Memory Efficiency
- **Gradient Checkpointing**: Reduces memory usage
- **Efficient Attention**: Optimized cross-attention implementation
- **Dynamic Shapes**: Handles variable input sizes

## 🧪 Testing & Validation

### Test Suites
- `test_edge_guidance.py`: Edge filtering validation
- `test_composite_loss.py`: Loss function testing
- `test_cross_attention.py`: Attention mechanism validation
- `test_icr_head.py`: ICR head functionality
- `test_bridge_diffusion.py`: Bridge process validation
- `test_adversarial_loss.py`: GAN training validation
- `test_data_augmentations.py`: Augmentation pipeline testing

### Running Tests
```bash
# Run all tests
uv run test_edge_guidance.py
uv run test_composite_loss.py
uv run test_cross_attention.py
uv run test_icr_head.py
uv run test_bridge_diffusion.py
uv run test_adversarial_loss.py
uv run test_data_augmentations.py
```

## 📋 Requirements

### Core Dependencies
```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
pydicom>=2.3.0
PIL>=9.0.0
opencv-python>=4.5.0
kornia>=0.6.0
diffusers>=0.20.0
tqdm>=4.64.0
```

### Optional Dependencies
```txt
lpips>=0.1.4          # For perceptual loss
scikit-image>=0.19.0  # For histogram matching
```

## 🎛️ Configuration Options

### Model Configuration
```python
model = PixelDiffusionUNetConditional(
    base_ch=64,                    # Base channel count
    ch_mults=(1, 2, 4, 8),        # Channel multipliers
    time_dim=256,                 # Timestep embedding dimension
    attn_res=(32, 64),           # Attention resolutions
    use_cross_attention=True,     # Enable cross-attention
    use_icr_head=True,           # Enable ICR head
    cross_attn_res=(32, 64, 128) # Cross-attention resolutions
)
```

### Training Configuration
```python
train_with_eval(
    epochs=100,
    use_composite_loss=True,      # Enable composite loss
    use_bridge_diffusion=True,    # Enable bridge process
    use_adversarial_loss=True,    # Enable GAN training
    drop_prob=0.1                # Classifier-free guidance dropout
)
```

### Augmentation Configuration
```python
dataset = CTPairsDataset(
    enable_augmentations=True,
    noise_injection_prob=0.3,     # Physics-aware noise
    spectral_jitter_prob=0.4,     # Intensity variations
    mixup_prob=0.15,              # MixUp augmentation
    patchmix_prob=0.15            # PatchMix augmentation
)
```

## 📊 Expected Improvements

### Quantitative Metrics
- **SSIM**: Improved structural similarity
- **PSNR**: Better peak signal-to-noise ratio
- **LPIPS**: Enhanced perceptual quality
- **MSE/L1**: Reduced pixel-wise errors

### Qualitative Benefits
- **Edge Preservation**: Better anatomical structure retention
- **Noise Reduction**: More effective denoising
- **Artifact Reduction**: Fewer reconstruction artifacts
- **Robustness**: Better generalization to unseen data

## 🔬 Technical Details

### Diffusion Process
- **Scheduler**: DDPM with cosine beta schedule
- **Timesteps**: 1000 training steps, 50-100 inference steps
- **Noise Schedule**: β ∈ [1e-4, 2e-2]

### Loss Function Weights
- **MSE**: 1.0 (primary reconstruction loss)
- **L1**: 0.5 (edge preservation)
- **LPIPS**: 0.1 (perceptual quality)
- **Adversarial**: 0.01 (texture realism)

### Training Strategy
- **Optimizer**: AdamW (lr=1e-4, β₁=0.9, β₂=0.95)
- **Weight Decay**: 1e-2
- **Batch Size**: 4-12 (adaptive based on hardware)
- **Gradient Clipping**: 1.0

## 🚨 Known Limitations

1. **Memory Usage**: High memory requirements for large images
2. **Training Time**: Extended training due to multiple loss components
3. **Hyperparameter Sensitivity**: Requires careful tuning of loss weights
4. **Data Requirements**: Benefits from large, diverse training datasets

## 🔮 Future Enhancements

### Stage 8: Self-Supervised Pretraining (Optional)
- **Masked HU Prediction**: Pretrain encoder on self-supervised task
- **Transfer Learning**: Initialize diffusion model with pretrained weights
- **Data Efficiency**: Leverage unlabeled CT data

### Additional Improvements
- **3D Processing**: Extend to volumetric CT data
- **Multi-Scale Training**: Hierarchical resolution training
- **Attention Optimization**: More efficient attention mechanisms
- **Model Compression**: Quantization and pruning for deployment

## 📚 References

1. **Diffusion Models**: Ho et al., "Denoising Diffusion Probabilistic Models"
2. **Guided Filtering**: He et al., "Guided Image Filtering"
3. **Cross-Attention**: Vaswani et al., "Attention Is All You Need"
4. **PatchGAN**: Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks"
5. **CT Physics**: Buzug, "Computed Tomography: From Photon Statistics to Modern Cone-Beam CT"