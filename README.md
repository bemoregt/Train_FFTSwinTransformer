# FFT-Swin Transformer for Semantic Segmentation

A PyTorch implementation of FFT-based Swin Transformer for semantic segmentation on the ADE20K dataset. This project introduces a novel approach by replacing traditional attention mechanisms with Fast Fourier Transform (FFT) operations in the Swin Transformer architecture.

## Overview

This implementation modifies the Swin Transformer architecture by:
- Replacing multi-head attention with FFT-based attention mechanisms
- Implementing FFT operations in both global and window-based attention modules
- Training on the ADE20K dataset for semantic segmentation tasks
- Supporting MPS (Metal Performance Shaders) for Apple Silicon devices

## Key Features

- **FFT-based Attention**: Novel FFT attention mechanism replacing traditional self-attention
- **Window-based FFT**: Localized FFT operations within sliding windows
- **ADE20K Integration**: Automatic dataset download and preprocessing
- **Multi-device Support**: Supports MPS, CUDA, and CPU devices
- **Comprehensive Training Pipeline**: Complete training, validation, and testing framework
- **Visualization Tools**: Built-in segmentation result visualization

## Architecture

### FFTAttention Module
- Applies 2D FFT to value tensors instead of traditional attention computation
- Processes real and imaginary components separately
- Maintains spatial relationships through frequency domain operations

### FFTSwinTransformerBlock
- Integrates FFT attention into Swin Transformer blocks
- Supports window partitioning and cyclic shifting
- Includes skip connections and MLP layers

### Segmentation Decoder
- FPN-style decoder for multi-scale feature fusion
- Progressive upsampling with skip connections
- Final classification head for 150 ADE20K classes

## Requirements

```
torch
torchvision
einops
matplotlib
tqdm
requests
pillow
timm
numpy
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/bemoregt/Train_FFTSwinTransformer.git
cd Train_FFTSwinTransformer
```

2. Install dependencies:
```bash
pip install torch torchvision einops matplotlib tqdm requests pillow timm numpy
```

## Usage

### Training

Run the main training script:
```bash
python main.py
```

The script will automatically:
- Download the ADE20K dataset
- Initialize the FFT-Swin Transformer model
- Train for the specified number of epochs
- Save checkpoints and best models
- Evaluate on validation set

### Model Configuration

Key hyperparameters can be modified in the `main()` function:

```python
img_size = 512          # Input image size
patch_size = 4          # Patch size for tokenization
embed_dim = 96          # Embedding dimension
depths = [2, 2, 6, 2]   # Number of blocks in each stage
num_heads = [3, 6, 12, 24]  # Number of attention heads
window_size = 7         # Window size for local attention
batch_size = 4          # Training batch size
num_epochs = 30         # Number of training epochs
```

### Dataset

The ADE20K dataset is automatically downloaded and preprocessed:
- **Training set**: ~20,000 images with 150 semantic classes
- **Validation set**: ~2,000 images
- **Automatic download**: Script handles dataset acquisition
- **Preprocessing**: Includes data augmentation and normalization

## Model Architecture Details

### FFT-based Attention Mechanism

The core innovation replaces traditional attention with FFT operations:

1. **Input Processing**: Query, Key, Value tensors are generated normally
2. **FFT Application**: 2D FFT is applied to value tensors
3. **Complex Processing**: Real and imaginary components are processed separately
4. **Spatial Reconstruction**: Results are transformed back to spatial domain

### Segmentation Pipeline

1. **Patch Embedding**: Images are divided into patches and embedded
2. **Multi-stage Encoding**: Four-stage hierarchical feature extraction
3. **Feature Pyramid Decoding**: Progressive upsampling with skip connections
4. **Classification Head**: Final pixel-wise classification

## Training Pipeline

### Loss Function
- Cross-entropy loss with ignore index for invalid pixels
- Supports ADE20K's 150-class semantic segmentation

### Optimization
- AdamW optimizer with weight decay
- Cosine annealing learning rate scheduler
- Gradient clipping for stable training

### Evaluation Metrics
- **mIoU (mean Intersection over Union)**: Primary evaluation metric
- **Pixel Accuracy**: Secondary metric for model performance
- **Class-wise IoU**: Detailed per-class performance analysis

## Results and Visualization

The framework provides comprehensive result analysis:

### Training Monitoring
- Real-time loss and metric tracking
- Validation performance evaluation
- Best model checkpointing

### Visualization
- Segmentation mask overlay on original images
- Color-coded semantic maps
- Side-by-side comparison views

## File Structure

```
Train_FFTSwinTransformer/
├── main.py                 # Main training script with full implementation
├── README.md              # This file
└── models/                # Saved model checkpoints (created during training)
    ├── fft_swin_seg_best.pth
    ├── fft_swin_seg_final.pth
    └── fft_swin_seg_epoch_*.pth
```

## Device Support

The implementation automatically detects and uses available devices:

1. **MPS (Apple Silicon)**: Optimized for M1/M2 Mac devices
2. **CUDA**: GPU acceleration for NVIDIA graphics cards
3. **CPU**: Fallback for systems without GPU support

## Performance Notes

- **Memory Usage**: Optimized for efficient memory utilization
- **Training Speed**: FFT operations provide computational efficiency
- **Batch Size**: Recommended batch size of 4 for 512x512 images
- **Dataset Sampling**: Uses subset for faster experimentation

## Citation

If you use this implementation in your research, please consider citing:

```bibtex
@misc{fft-swin-transformer,
  title={FFT-Swin Transformer for Semantic Segmentation},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/bemoregt/Train_FFTSwinTransformer}}
}
```

## Acknowledgments

- Based on the original Swin Transformer architecture
- Built upon PyTorch and timm libraries
- Trained on the ADE20K semantic segmentation dataset

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.