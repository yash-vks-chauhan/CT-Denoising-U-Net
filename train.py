import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import pandas as pd
import seaborn as sns

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Paths to the dataset
clean_img_path = "/kaggle/input/lung-train-model/Train/Train/Clean"
noisy_img_path = "/kaggle/input/lung-train-model/Train/Train/Noisy"

# Image parameters
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 1  # Grayscale images

def create_data_generator(clean_path, noisy_path, batch_size=16):
    """Create a generator that yields batches of images with handling for augmented filenames"""
    clean_files = sorted(os.listdir(clean_path))
    noisy_files = sorted(os.listdir(noisy_path))

    print(f"Found {len(clean_files)} clean image files in '{clean_path}'")
    print(f"Found {len(noisy_files)} noisy image files in '{noisy_path}'")
    
    # Create mappings between clean and noisy files based on base names
    clean_file_map = {}
    for clean_file in clean_files:
        base_name = clean_file.split('_aug')[0] if '_aug' in clean_file else clean_file
        clean_file_map[base_name] = clean_file
    
    # Find matching files between clean and noisy
    matched_pairs = []
    for noisy_file in noisy_files:
        base_name = noisy_file.split('_aug')[0] if '_aug' in noisy_file else noisy_file
        if base_name in clean_file_map:
            matched_pairs.append((clean_file_map[base_name], noisy_file))
    
    print(f"Found {len(matched_pairs)} matched image pairs")
    num_samples = len(matched_pairs)

    indices = np.arange(num_samples)

    while True:
        # Shuffle indices each epoch
        np.random.shuffle(indices)

        for start_idx in range(0, num_samples, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]

            batch_clean = np.zeros((len(batch_indices), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
            batch_noisy = np.zeros((len(batch_indices), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)

            for i, idx in enumerate(batch_indices):
                clean_file, noisy_file = matched_pairs[idx]
                
                # Read and preprocess clean image
                clean_img = cv2.imread(os.path.join(clean_path, clean_file), cv2.IMREAD_GRAYSCALE)
                clean_img = cv2.resize(clean_img, (IMG_WIDTH, IMG_HEIGHT))
                clean_img = clean_img / 255.0  # Normalize to [0,1]

                # Read and preprocess noisy image
                noisy_img = cv2.imread(os.path.join(noisy_path, noisy_file), cv2.IMREAD_GRAYSCALE)
                noisy_img = cv2.resize(noisy_img, (IMG_WIDTH, IMG_HEIGHT))
                noisy_img = noisy_img / 255.0  # Normalize to [0,1]

                batch_clean[i] = clean_img.reshape(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
                batch_noisy[i] = noisy_img.reshape(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

            yield batch_noisy, batch_clean

def load_validation_data(clean_path, noisy_path, validation_split=0.2):
    """Load a smaller validation dataset for evaluation with handling for augmented filenames"""
    clean_files = sorted(os.listdir(clean_path))
    noisy_files = sorted(os.listdir(noisy_path))
    
    print(f"Found {len(clean_files)} clean image files")
    print(f"Found {len(noisy_files)} noisy image files")
    
    # Create mappings between clean and noisy files based on base names
    clean_file_map = {}
    for clean_file in clean_files:
        base_name = clean_file.split('_aug')[0] if '_aug' in clean_file else clean_file
        clean_file_map[base_name] = clean_file
    
    # Find matching files between clean and noisy
    matched_pairs = []
    for noisy_file in noisy_files:
        base_name = noisy_file.split('_aug')[0] if '_aug' in noisy_file else noisy_file
        if base_name in clean_file_map:
            matched_pairs.append((clean_file_map[base_name], noisy_file))
    
    print(f"Found {len(matched_pairs)} matched image pairs")
    
    # Split the data
    train_files, val_files = train_test_split(
        matched_pairs,
        test_size=validation_split,
        random_state=42
    )
    
    # Load only validation data into memory
    val_clean = []
    val_noisy = []
    val_names = []
    
    for clean_file, noisy_file in tqdm(val_files, desc="Loading validation data"):
        # Process clean image
        clean_img = cv2.imread(os.path.join(clean_path, clean_file), cv2.IMREAD_GRAYSCALE)
        clean_img = cv2.resize(clean_img, (IMG_WIDTH, IMG_HEIGHT))
        clean_img = clean_img / 255.0
        
        # Process noisy image
        noisy_img = cv2.imread(os.path.join(noisy_path, noisy_file), cv2.IMREAD_GRAYSCALE)
        noisy_img = cv2.resize(noisy_img, (IMG_WIDTH, IMG_HEIGHT))
        noisy_img = noisy_img / 255.0
        
        val_clean.append(clean_img)
        val_noisy.append(noisy_img)
        val_names.append(clean_file)  # Store clean file name for reference
    
    # Convert to numpy arrays
    val_clean = np.array(val_clean).reshape(-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS).astype(np.float32)
    val_noisy = np.array(val_noisy).reshape(-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS).astype(np.float32)
    
    return val_noisy, val_clean, val_names, len(train_files)

def build_optimized_unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    """Build a more efficient U-Net model for image denoising"""
    inputs = Input(input_shape)

    # Encoder - with fewer filters and efficient blocks
    def encoder_block(x, filters, kernel_size=3, batch_norm=True, pool=True):
        conv = Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
        if batch_norm:
            conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        conv = Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(conv)
        if batch_norm:
            conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)

        if pool:
            pool_layer = MaxPooling2D(pool_size=(2, 2))(conv)
            return conv, pool_layer
        return conv

    # Encoder path
    conv1, pool1 = encoder_block(inputs, 32)
    conv2, pool2 = encoder_block(pool1, 64)
    conv3, pool3 = encoder_block(pool2, 128)
    conv4, pool4 = encoder_block(pool3, 256)

    # Bridge
    conv5 = encoder_block(pool4, 512, pool=False)

    # Decoder path
    def decoder_block(x, skip_connection, filters, kernel_size=3, batch_norm=True):
        up = UpSampling2D(size=(2, 2))(x)
        up = Conv2D(filters, 2, padding='same', kernel_initializer='he_normal')(up)
        up = Activation('relu')(up)

        # Concatenate with skip connection
        merge = Concatenate()([up, skip_connection])

        conv = Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(merge)
        if batch_norm:
            conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)

        conv = Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(conv)
        if batch_norm:
            conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)

        return conv

    # Decoder blocks
    decoder6 = decoder_block(conv5, conv4, 256)
    decoder7 = decoder_block(decoder6, conv3, 128)
    decoder8 = decoder_block(decoder7, conv2, 64)
    decoder9 = decoder_block(decoder8, conv1, 32)

    # Output
    outputs = Conv2D(IMG_CHANNELS, (1, 1), activation='sigmoid')(decoder9)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model

def train_model_with_generator(model_builder, train_gen, val_data, batch_size=16, steps_per_epoch=None, epochs=50):
    """Train the model using a generator for memory efficiency"""
    # Unpack validation data
    X_val, y_val, _, num_train_samples = val_data

    if steps_per_epoch is None:
        steps_per_epoch = num_train_samples // batch_size

    # Build model
    model = model_builder()

    # Print model summary
    print("\nModel Summary:")
    model.summary()
    print("Model summary printed.")

    # Compile model with mixed precision
    optimizer = Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    # Callbacks
    callbacks = [
        ModelCheckpoint('best_denoising_model.keras', save_best_only=True, monitor='val_loss'),
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
    ]

    # Train with generator
    print("Starting model training with model.fit...")
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        workers=4,
        use_multiprocessing=True
    )
    print("model.fit call completed.")

    return model, history

def calculate_improvement_metrics(X_test, y_test, predictions, image_names=None):
    """Calculate comprehensive metrics for model evaluation"""
    metrics_list = []

    for i in range(len(predictions)):
        # Convert to [0,1] format
        noisy_img = X_test[i].reshape(IMG_HEIGHT, IMG_WIDTH)
        clean_img = y_test[i].reshape(IMG_HEIGHT, IMG_WIDTH)
        pred_img = predictions[i].reshape(IMG_HEIGHT, IMG_WIDTH)

        # Ensure all images have the same dtype
        noisy_img = noisy_img.astype(np.float32)
        clean_img = clean_img.astype(np.float32)
        pred_img = pred_img.astype(np.float32)

        # Calculate PSNR with data_range specified
        noisy_psnr_val = psnr(clean_img, noisy_img, data_range=1.0)
        denoised_psnr_val = psnr(clean_img, pred_img, data_range=1.0)
        psnr_improvement = denoised_psnr_val - noisy_psnr_val

        # Calculate SSIM with data_range specified
        noisy_ssim_val = ssim(clean_img, noisy_img, data_range=1.0)
        denoised_ssim_val = ssim(clean_img, pred_img, data_range=1.0)
        ssim_improvement = denoised_ssim_val - noisy_ssim_val

        # Calculate MSE
        noisy_mse = np.mean((clean_img - noisy_img) ** 2)
        denoised_mse = np.mean((clean_img - pred_img) ** 2)
        mse_improvement = noisy_mse - denoised_mse

        # Get image name, handling potential augmented file naming
        if image_names:
            image_id = image_names[i]
            # Extract base name without augmentation suffix if needed
            if '_aug' in image_id:
                base_image_id = image_id.split('_aug')[0]
            else:
                base_image_id = image_id
        else:
            image_id = i
            base_image_id = i

        # Store metrics
        metrics_dict = {
            'image_id': image_id,
            'base_image_id': base_image_id,
            'noisy_psnr': noisy_psnr_val,
            'denoised_psnr': denoised_psnr_val,
            'psnr_improvement': psnr_improvement,
            'psnr_improvement_percent': (psnr_improvement / noisy_psnr_val) * 100 if noisy_psnr_val > 0 else float('inf'),

            'noisy_ssim': noisy_ssim_val,
            'denoised_ssim': denoised_ssim_val,
            'ssim_improvement': ssim_improvement,
            'ssim_improvement_percent': (ssim_improvement / noisy_ssim_val) * 100 if noisy_ssim_val > 0 else float('inf'),

            'noisy_mse': noisy_mse,
            'denoised_mse': denoised_mse,
            'mse_improvement': mse_improvement,
            'mse_reduction_percent': (mse_improvement / noisy_mse) * 100 if noisy_mse > 0 else float('inf'),
        }

        metrics_list.append(metrics_dict)

    # Create DataFrame
    metrics_df = pd.DataFrame(metrics_list)

    return metrics_df

def run_training_pipeline():
    """Execute the full optimized training pipeline"""
    print("Starting optimized training pipeline...")

    # Set batch size
    BATCH_SIZE = 16
    EPOCHS = 50

    # Load validation data (keeping this small to save memory)
    val_data = load_validation_data(clean_img_path, noisy_img_path, validation_split=0.15)
    X_val, y_val, val_names, num_train_samples = val_data

    # Test validation data loading
    print("Testing validation data loading...")
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)
    print("Number of validation samples:", len(X_val))
    print("Validation data loading test successful.")

    # Create training data generator
    train_gen = create_data_generator(clean_img_path, noisy_img_path, batch_size=BATCH_SIZE)

    # Test the generator
    print("Testing data generator...")
    sample_batch_noisy, sample_batch_clean = next(train_gen)
    print("Sample noisy batch shape:", sample_batch_noisy.shape)
    print("Sample clean batch shape:", sample_batch_clean.shape)
    print("Generator test successful, data is being yielded in batches.")

    # Calculate steps per epoch
    steps_per_epoch = num_train_samples // BATCH_SIZE
    print(f"Training with {num_train_samples} samples, {steps_per_epoch} steps per epoch")

    # Train model
    model, history = train_model_with_generator(
        model_builder=build_optimized_unet,
        train_gen=train_gen,
        val_data=val_data,
        batch_size=BATCH_SIZE,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS
    )

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Mean Absolute Error')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

    # Generate predictions on validation set
    print("Generating predictions on validation set...")
    predictions = model.predict(X_val, batch_size=BATCH_SIZE)

    # Evaluate model
    print("Calculating performance metrics...")
    metrics_df = calculate_improvement_metrics(X_val, y_val, predictions, val_names)

    # Display sample results
    sample_indices = np.random.choice(len(X_val), 3, replace=False)

    plt.figure(figsize=(15, 5*len(sample_indices)))

    for i, idx in enumerate(sample_indices):
        # Display original noisy image
        plt.subplot(len(sample_indices), 3, i*3+1)
        plt.imshow(X_val[idx].reshape(IMG_HEIGHT, IMG_WIDTH), cmap='gray')
        plt.title(f"Sample {i+1}\nNoisy Input")
        plt.axis('off')

        # Display ground truth (clean image)
        plt.subplot(len(sample_indices), 3, i*3+2)
        plt.imshow(y_val[idx].reshape(IMG_HEIGHT, IMG_WIDTH), cmap='gray')
        plt.title("Clean (Ground Truth)")
        plt.axis('off')

        # Display model prediction
        plt.subplot(len(sample_indices), 3, i*3+3)
        plt.imshow(predictions[idx].reshape(IMG_HEIGHT, IMG_WIDTH), cmap='gray')
        plt.title("Denoised (Predicted)")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('sample_results.png')
    plt.show()

    # Print summary statistics
    print("\n==== SUMMARY STATISTICS ====")
    print(f"Average PSNR Improvement: {metrics_df['psnr_improvement'].mean():.2f} dB (± {metrics_df['psnr_improvement'].std():.2f})")
    print(f"Average SSIM Improvement: {metrics_df['ssim_improvement'].mean():.4f} (± {metrics_df['ssim_improvement'].std():.4f})")
    print(f"Average MSE Reduction: {metrics_df['mse_reduction_percent'].mean():.2f}% (± {metrics_df['mse_reduction_percent'].std():.2f}%)")

    print("\n==== SUCCESS RATE ====")
    psnr_success = (metrics_df['psnr_improvement'] > 0).mean() * 100
    ssim_success = (metrics_df['ssim_improvement'] > 0).mean() * 100
    mse_success = (metrics_df['mse_improvement'] > 0).mean() * 100
    print(f"PSNR Improvement Success Rate: {psnr_success:.2f}%")
    print(f"SSIM Improvement Success Rate: {ssim_success:.2f}%")
    print(f"MSE Improvement Success Rate: {mse_success:.2f}%")
    
    # Return model and metrics for further analysis if needed
    return model, metrics_df

# Main execution
if __name__ == "__main__":
    # Run training pipeline
    model, metrics_df = run_training_pipeline()
    
    # Optional: Save metrics to CSV
    metrics_df.to_csv('denoising_metrics.csv', index=False)
    
    print("Training and evaluation complete!")

