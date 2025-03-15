# CT-Denoising-U-Net

## Overview
This project focuses on developing a deep learning model for denoising lung disease images from various datasets, including COVID-19, TB, and lung cancer scans. The model leverages a U-Net architecture with optimizations for handling medical imaging noise and artifacts.

## Features
- **Preprocessing Pipeline**: Converts images to grayscale, resizes them, applies denoising, normalization, CLAHE, and augmentation.
- **Dataset Handling**: Supports structured datasets with separate `Noisy` and `Clean` folders.
- **Denoising Model**: Uses an optimized U-Net for noise reduction in medical images.
- **Inference & Validation**: Loads validation images efficiently, ensuring augmentation consistency.

## Dataset Structure
Ensure your dataset is organized as follows:
```
lung_disease_dataset/
    ├── Train/
    │   ├── Train/
    │   │   ├── Noisy/
    │   │   ├── Clean/
    ├── Test/
        ├── Noisy/
        ├── Clean/
```

## Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
```

## Training the Model
Run the training script:
```bash
python train.py
```

## Running Inference
To denoise an image using the trained model:
```bash
python inference.py --input noisy_image.png --output denoised_image.png
```

## Model Architecture
- **Encoder**: Convolutional layers extract features.
- **Bottleneck**: Captures essential noise patterns.
- **Decoder**: Upsamples and reconstructs clean images.

## Performance Evaluation
- **Metrics Used**: PSNR, SSIM
- **Validation Pipeline**: Ensures consistency between augmented and original images.

## Future Enhancements
- Implement edge enhancement and artifact removal.
- Fine-tune super-resolution techniques for improved image quality.

## License
This project is released under the MIT License.

