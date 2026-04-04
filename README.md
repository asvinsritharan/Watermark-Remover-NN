# Watermark Remover

A supervised watermark removal pipeline that trains and compares five models via A/B testing, then applies the best-performing model to restore watermarked images.

## Overview

The pipeline:
1. Loads matched pairs of clean and watermarked images
2. Auto-computes watermark masks from pixel differences
3. Trains all models on an 80/20 train/test split
4. Evaluates each model using PSNR and SSIM
5. Runs one-way ANOVA and pairwise Welch's t-tests (Bonferroni corrected) to compare models
6. Selects the best model by mean PSNR and applies it to new images

## Models

| Model | Type | Description |
|---|---|---|
| **TELEA Inpainting** | Classical | Fast marching method inpainting (OpenCV) |
| **NS Inpainting** | Classical | Navier-Stokes based inpainting (OpenCV) |
| **Dictionary Learning** | Classical ML | Sparse patch reconstruction over a learned dictionary |
| **NMF** | Classical ML | Per-channel patch reconstruction via Non-negative Matrix Factorisation |
| **CNN Autoencoder** | Deep Learning | U-Net with skip connections, L1 loss, Adam + ReduceLROnPlateau |

## Project Structure

```
├── run_watermark_removal.py       # Entry point and pipeline orchestration
├── data/
│   ├── clean_images/              # Ground truth unwatermarked images
│   └── watermarked_images/        # Watermarked images (filenames must match clean/)
├── results/                       # Output directory (created on run)
│   ├── saved_models/              # Persisted model weights/configs
│   ├── experiment_results.txt     # Per-model metrics and statistical test results
│   ├── model_comparison.png       # PSNR/SSIM bar charts
│   └── per_model_outputs/         # Per-model restored images
└── src/
    ├── preprocessing/
    │   └── image_processing.py    # Image loading, mask computation, model application
    ├── models/
    │   ├── classical_inpainting.py
    │   ├── dictionary_learning_model.py
    │   ├── nmf_model.py
    │   ├── cnn_autoencoder.py
    │   └── init_models.py
    ├── evaluation/
    │   └── metrics.py             # PSNR and SSIM implementations
    └── ab_testing/
        └── experiment.py          # A/B testing, statistical tests, result plots
```

## Requirements

```
torch
torchvision
opencv-python
numpy
scikit-learn
scipy
matplotlib
joblib
```

Install with:

```bash
pip install torch torchvision opencv-python numpy scikit-learn scipy matplotlib joblib
```

## Data Setup

Place matched image pairs in two directories. Files are matched by stem (name without extension), so `image_0.jpg` in `clean_images/` will be paired with `image_0.jpeg` in `watermarked_images/`. Extensions may differ.

```
data/
├── clean_images/
│   ├── image_0.jpg
│   └── image_1.jpg
└── watermarked_images/
    ├── image_0.jpg
    └── image_1.jpg
```

Watermark masks are computed automatically from the pixel difference between clean and watermarked pairs. If a mask is not available at inference time, the pipeline falls back to auto-detection using brightness and saturation thresholds.

## Usage

### Train, compare, and apply to a test image

```bash
python run_watermark_removal.py \
    --clean_images_path data/clean_images \
    --watermarked_images_path data/watermarked_images \
    --output_path results \
    --test_image path/to/watermarked.jpg \
    --mask path/to/mask.png        # optional
```

If `--mask` is omitted, the watermark region is auto-detected.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--clean_images_path` | `data/clean_images` | Directory of ground truth clean images |
| `--watermarked_images_path` | `data/watermarked_images` | Directory of watermarked images |
| `--output_path` | `results` | Output directory for results and saved models |
| `--test_image` | `None` | Watermarked image to restore using the best model |
| `--mask` | `None` | Binary mask for the test image (white = watermark region) |

### Programmatic usage

```python
from run_watermark_removal import PerformWatermarkRemoval

pipeline = PerformWatermarkRemoval(
    clean_images_path='data/clean_images',
    watermarked_images_path='data/watermarked_images',
    output_path='results'
)

# Apply the best model to a new image
restored = pipeline.remove_watermark('path/to/image.jpg', output_path='results/output.png')

# Apply all models and save each output
results = pipeline.remove_watermark_all_models('path/to/image.jpg', output_dir='results')
```

## Outputs

After running, the `output_path` directory will contain:

- `experiment_results.txt` — mean PSNR/SSIM per model, ANOVA F-statistic and p-value, all pairwise t-test results, and the selected best model
- `model_comparison.png` — side-by-side bar charts of PSNR and SSIM for all models (best model highlighted in orange)
- `saved_models/` — serialised weights for all trained models (`.pt` for CNN, `.pkl` for others)
- `final_output.png` — restored image from the best model (if `--test_image` provided)
- `per_model_outputs/` — restored images from every model (if `--test_image` provided)

## Evaluation

Models are scored on the held-out 20% test split using:

- **PSNR** (Peak Signal-to-Noise Ratio) — higher is better, measures pixel-level reconstruction fidelity
- **SSIM** (Structural Similarity Index) — higher is better, captures luminance, contrast, and structure

Statistical significance is assessed with a one-way ANOVA followed by pairwise Welch's t-tests with Bonferroni correction. The model with the highest mean PSNR is selected automatically.
