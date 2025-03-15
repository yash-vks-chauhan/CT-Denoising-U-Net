# CT-Denoising-U-Net

## Overview
This project focuses on developing a deep learning model for denoising lung disease images, specifically CT scans and X-rays. The model enhances image quality by reducing noise and improving clarity, aiding in better diagnosis.

## Features
✔ U-Net-based model for image denoising  
✔ Supports grayscale CT scans  
✔ Preprocessing pipeline with resizing, normalization, and augmentation  
✔ SSIM & PSNR metrics for quality assessment  
✔ TensorFlow/Keras with GPU acceleration  
✔ Custom data generator for efficient memory management  
✔ Future enhancements: Super-Resolution, Edge Enhancement, Artifact Removal  

### Additional Features
- **Preprocessing Pipeline**: Converts images to grayscale, resizes them, applies denoising, normalization, CLAHE, and augmentation.
- **Dataset Handling**: Supports structured datasets with separate `Noisy` and `Clean` folders.
- **Denoising Model**: Uses an optimized U-Net for noise reduction in medical images.
- **Inference & Validation**: Loads validation images efficiently, ensuring augmentation consistency.

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

## Dataset
The dataset includes images from:
- **COVID-19 Radiography Dataset**
- **Montgomery & Shenzhen TB Dataset**
- **LIDC-IDRI Lung Cancer Dataset**
- **NIH Chest X-ray Dataset**

## Preprocessing Steps
The preprocessing pipeline includes:
1. **Grayscale Conversion** - Ensures uniformity across images.
2. **Resizing** - Standardizes image dimensions.
3. **Denoising** - Removes noise using advanced techniques.
4. **Augmentation** - Applies rotations, flips, and translations.
5. **Normalization** - Scales pixel values between 0 and 1.
6. **CLAHE (Contrast Limited Adaptive Histogram Equalization)** - Enhances contrast.
7. **Super-Resolution Enhancement** - Improves image resolution.
8. **Edge Enhancement & Artifact Removal** - Refines image quality.

## Model Training
The model is trained using paired noisy-clean images. The clean images have 'aug' in their filenames, such as `00007538_001_aug3.png`.

## Image Sample
Below is an example of the lung images used in this project:

![Lung Images](https://github.com/yash-vks-chauhan/CT-Denoising-U-Net/blob/main/are/lung_images.png)

## Results
The model performance is evaluated based on:
- **PSNR (Peak Signal-to-Noise Ratio)**
- **SSIM (Structural Similarity Index)**
- **Qualitative Visual Inspection**

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yash-vks-chauhan/CT-Denoising-U-Net.git
   cd CT-Denoising-U-Net
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the preprocessing script:
   ```bash
   python preprocess.py
   ```
4. Train the model:
   ```bash
   python train.py
   ```

## Future Improvements
- Implement **GAN-based denoising**.
- Explore **Diffusion models** for further enhancement.
- Deploy as a web application for easy access.

## Acknowledgments
Special thanks to the dataset providers and the deep learning research community for their contributions.

## License
This project is released under the MIT License.




