import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
from dtw import dtw
from numpy.linalg import norm
import seaborn as sns
from scipy.stats import wasserstein_distance

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Global parameters
SEQUENCE_LENGTH = 50  # Length of time sequence to generate
LATENT_DIM = 64  # Dimension of noise input to generator
BATCH_SIZE = 32
EPOCHS = 300
NUM_TRANSFORMER_BLOCKS = 3
NUM_ATTENTION_HEADS = 8
EMBEDDING_DIM = 128
FF_DIM = EMBEDDING_DIM * 4  # Feed-forward dimension
DROPOUT_RATE = 0.1
LEARNING_RATE_GENERATOR = 0.0002
LEARNING_RATE_DISCRIMINATOR = 0.0002

# Positional Encoding
def positional_encoding(length, depth):
    """
    Create positional encoding for transformer inputs
    """
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth
    
    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates
    
    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1
    )
    
    return tf.cast(pos_encoding, dtype=tf.float32)

# Custom Transformer Block for the Generator and Discriminator
# Add these constants at the top of the file
EMBEDDING_DIM = 128
NUM_TRANSFORMER_BLOCKS = 3
NUM_ATTENTION_HEADS = 8
DROPOUT_RATE = 0.1
LEARNING_RATE_GENERATOR = 0.0002
LEARNING_RATE_DISCRIMINATOR = 0.0002

def build_generator(latent_dim, sequence_length, num_features, num_classes):
    """
    Build the Transformer-based Generator model
    """
    # Noise input
    noise_input = keras.Input(shape=(latent_dim,), name="noise_input")
    
    # Label input
    label_input = keras.Input(shape=(num_classes,), name="label_input")
    
    # Concatenate noise and label
    combined_input = layers.Concatenate()([noise_input, label_input])
    
    # Make sure all layers are trainable
    x = layers.Dense(sequence_length * EMBEDDING_DIM, trainable=True)(combined_input)
    x = layers.Reshape((sequence_length, EMBEDDING_DIM))(x)
    
    # Add positional encoding
    pos_enc = positional_encoding(sequence_length, EMBEDDING_DIM)
    pos_enc = pos_enc[:sequence_length, :EMBEDDING_DIM]
    pos_enc = tf.expand_dims(pos_enc, 0)
    x = x + pos_enc
    
    # Transformer encoder blocks
    for _ in range(NUM_TRANSFORMER_BLOCKS):
        x = transformer_encoder_block(
            inputs=x,
            num_heads=NUM_ATTENTION_HEADS,
            key_dim=EMBEDDING_DIM // NUM_ATTENTION_HEADS,
            dropout_rate=DROPOUT_RATE,
            trainable=True  # Make sure blocks are trainable
        )
    
    # Output layer
    x = layers.Dense(num_features, trainable=True)(x)
    
    # Create model
    model = keras.Model([noise_input, label_input], x, name="generator")
    
    # Ensure all layers are trainable
    for layer in model.layers:
        layer.trainable = True
    
    return model

def transformer_encoder_block(inputs, num_heads, key_dim, dropout_rate, trainable=True):
    """
    Transformer encoder block with trainable layers
    """
    # Layer normalization
    x = layers.LayerNormalization(epsilon=1e-6, trainable=trainable)(inputs)
    
    # Multi-head self-attention
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=key_dim,
        dropout=dropout_rate,
        trainable=trainable
    )(x, x)
    
    # Add & Norm
    attention_output = layers.Dropout(dropout_rate)(attention_output)
    x1 = layers.Add()([inputs, attention_output])
    
    # Feed-forward network
    x2 = layers.LayerNormalization(epsilon=1e-6, trainable=trainable)(x1)
    x2 = layers.Dense(EMBEDDING_DIM * 4, activation='gelu', trainable=trainable)(x2)
    x2 = layers.Dense(EMBEDDING_DIM, trainable=trainable)(x2)
    x2 = layers.Dropout(dropout_rate)(x2)
    
    # Add & Norm
    return layers.Add()([x1, x2])

# Discriminator Model
def build_discriminator(sequence_length, num_features, num_classes):
    """
    Build the Transformer-based Discriminator model with dual heads
    """
    # Input sequence
    sequence_input = keras.Input(shape=(sequence_length, num_features), name="sequence_input")
    
    # Add positional encoding
    pos_enc = positional_encoding(sequence_length, num_features)
    x = sequence_input + pos_enc[:, :num_features]
    
    # Project to higher dimension for transformer
    x = layers.Dense(EMBEDDING_DIM)(x)
    
    # Transformer encoder blocks
    for _ in range(NUM_TRANSFORMER_BLOCKS):
        x = transformer_encoder_block(
            inputs=x,
            num_heads=NUM_ATTENTION_HEADS,
            key_dim=EMBEDDING_DIM // NUM_ATTENTION_HEADS,
            dropout_rate=DROPOUT_RATE
        )
    
    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dual heads:
    # 1. Real/Fake classification
    adversarial_output = layers.Dense(1, activation="sigmoid", name="adversarial")(x)
    
    # 2. Class prediction
    classifier_output = layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    
    # Create model
    model = keras.Model(sequence_input, [adversarial_output, classifier_output], name="discriminator")
    return model

# Wasserstein loss with gradient penalty
def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)

def gradient_penalty(discriminator, real_samples, fake_samples):
    """
    Gradient penalty for Wasserstein GAN
    """
    batch_size = tf.shape(real_samples)[0]
    alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
    
    # Interpolation between real and fake samples
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred, _ = discriminator(interpolated, training=True)
    
    # Get gradients
    grads = gp_tape.gradient(pred, interpolated)
    
    # Calculate gradient norm
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
    
    # Apply gradient penalty
    penalty = tf.reduce_mean(tf.square(grad_norm - 1.0))
    
    return penalty

# TTS-CGAN Class
class TTSCGAN:
    def __init__(self, sequence_length, num_features, num_classes, latent_dim):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        # Build models
        self.generator = build_generator(latent_dim, sequence_length, num_features, num_classes)
        self.discriminator = build_discriminator(sequence_length, num_features, num_classes)

        # Ensure models are trainable
        self.generator.trainable = True
        self.discriminator.trainable = True

        # Compile discriminator
        self.discriminator.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_DISCRIMINATOR, beta_1=0.9, beta_2=0.999),
            loss=[wasserstein_loss, "categorical_crossentropy"],
            loss_weights=[1.0, 1.0]
        )

        # Build combined model for generator training
        self.discriminator.trainable = False  # Freeze discriminator weights when training generator
        noise_input = keras.Input(shape=(latent_dim,))
        label_input = keras.Input(shape=(num_classes,))
        generated_sequence = self.generator([noise_input, label_input])
        valid, target_label = self.discriminator(generated_sequence)

        self.combined = keras.Model([noise_input, label_input], [valid, target_label])
        self.combined.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_GENERATOR, beta_1=0.9, beta_2=0.999),
            loss=[wasserstein_loss, "categorical_crossentropy"],
            loss_weights=[1.0, 1.0]
        )

        # Make sure layers are trainable
        for layer in self.generator.layers:
            layer.trainable = True
        for layer in self.discriminator.layers:
            layer.trainable = True
    
    def train(self, x_train, y_train, epochs, batch_size, save_interval=50, output_dir='output'):
        # Add check for trainable weights
        generator_trainable = any(layer.trainable for layer in self.generator.layers)
        discriminator_trainable = any(layer.trainable for layer in self.discriminator.layers)
        
        if not generator_trainable or not discriminator_trainable:
            print("Warning: Some model components are not trainable!")
            print(f"Generator trainable: {generator_trainable}")
            print(f"Discriminator trainable: {discriminator_trainable}")
            
            # Print trainable status of each layer
            print("\nGenerator layers:")
            for layer in self.generator.layers:
                print(f"{layer.name}: trainable = {layer.trainable}")
            
            print("\nDiscriminator layers:")
            for layer in self.discriminator.layers:
                print(f"{layer.name}: trainable = {layer.trainable}")
        ##Train the TTS-CGAN model##
        # Create directories for output
        os.makedirs(output_dir, exist_ok=True)

        # Lists to store loss values
        d_losses = []
        g_losses = []

        # Ensure batch_size doesn't exceed training data size
        batch_size = min(batch_size, x_train.shape[0])

        # Number of critic iterations
        n_critic = 5

        print(f"Starting training for {epochs} epochs with batch size {batch_size}")

        try:
            # Training loop
            for epoch in range(epochs):
                # Train discriminator
                d_loss_epoch = []

                for _ in range(n_critic):
                    # Select a random batch of real sequences
                    idx = np.random.randint(0, x_train.shape[0], batch_size)
                    real_seqs = x_train[idx]
                    real_labels = y_train[idx]

                    # Generate random noise
                    noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                    # Generate fake sequences
                    gen_seqs = self.generator.predict([noise, real_labels], verbose=0)

                    # Train discriminator
                    d_loss_real = self.discriminator.train_on_batch(
                        real_seqs, 
                        [np.ones((batch_size, 1)), real_labels]
                    )

                    d_loss_fake = self.discriminator.train_on_batch(
                        gen_seqs, 
                        [np.zeros((batch_size, 1)), real_labels]
                    )

                    # Calculate discriminator loss
                    d_loss = [
                        0.5 * (d_loss_real[0] + d_loss_fake[0]),  # adversarial loss
                        0.5 * (d_loss_real[1] + d_loss_fake[1])   # classification loss
                    ]
                    d_loss_epoch.append(d_loss)

                # Train generator
                g_loss_epoch = []

                for _ in range(n_critic):
                    # Generate random noise
                    noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                    # Select random labels
                    idx = np.random.randint(0, x_train.shape[0], batch_size)
                    sampled_labels = y_train[idx]

                    # Train generator
                    g_loss = self.combined.train_on_batch(
                        [noise, sampled_labels],
                        [np.ones((batch_size, 1)), sampled_labels]
                    )
                    g_loss_epoch.append(g_loss)

                # Calculate average losses for this epoch
                d_loss_avg = np.mean(d_loss_epoch, axis=0)
                g_loss_avg = np.mean(g_loss_epoch, axis=0)

                # Store losses
                d_losses.append(d_loss_avg)
                g_losses.append(g_loss_avg)

                # Print progress
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}/{epochs}")
                    print(f"D Loss: {d_loss_avg}")
                    print(f"G Loss: {g_loss_avg}")

                # Save sample sequences
                if epoch % save_interval == 0:
                    self.save_sequences(epoch, output_dir)

                    # Plot loss curves
                    plt.figure(figsize=(10, 5))
                    plt.plot(d_losses, label='Discriminator Loss')
                    plt.plot(g_losses, label='Generator Loss')
                    plt.title('Training Losses')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.savefig(f"{output_dir}/loss_history.png")
                    plt.close()

            print("Training completed!")
            return d_losses, g_losses

        except Exception as e:
            print(f"Error during training: {str(e)}")
            return None, None
    
    def save_sequences(self, epoch, output_dir):
        """
        Generate and save sample sequences
        """
        # Generate examples for each class
        num_examples = 3
        num_rows = self.num_classes
        
        plt.figure(figsize=(15, 3 * num_rows))
        
        for class_idx in range(self.num_classes):
            noise = np.random.normal(0, 1, (num_examples, self.latent_dim))
            
            # Create one-hot encoded labels
            labels = np.zeros((num_examples, self.num_classes))
            labels[:, class_idx] = 1
            
            # Generate sequences
            gen_sequences = self.generator.predict([noise, labels])
            
            # Plot sequences
            for i in range(num_examples):
                plt.subplot(num_rows, num_examples, (class_idx * num_examples) + i + 1)
                plt.plot(gen_sequences[i])
                plt.title(f"Class {class_idx}")
                plt.ylim(-1, 1)
            
        plt.tight_layout()
        plt.savefig(f"{output_dir}/generated_sequences_epoch_{epoch}.png")
        plt.close()
    
    def generate_synthetic_sequences(self, num_sequences, class_idx, scaler=None):
        """
        Generate synthetic sequences for a specific class
        """
        noise = np.random.normal(0, 1, (num_sequences, self.latent_dim))

        # Create one-hot encoded labels
        labels = np.zeros((num_sequences, self.num_classes))
        labels[:, class_idx] = 1

        # Generate sequences
        gen_sequences = self.generator.predict([noise, labels], verbose=0)

        # Ensure correct sequence length
        if gen_sequences.shape[1] != self.sequence_length:
            if gen_sequences.shape[1] > self.sequence_length:
                gen_sequences = gen_sequences[:, :self.sequence_length, :]
            else:
                pad_width = ((0, 0), (0, self.sequence_length - gen_sequences.shape[1]), (0, 0))
                gen_sequences = np.pad(gen_sequences, pad_width, mode='constant')

        # If scaler is provided, inverse transform to original scale
        if scaler is not None:
            original_shape = gen_sequences.shape
            gen_sequences_flat = gen_sequences.reshape(-1, 1)
            gen_sequences_orig = scaler.inverse_transform(gen_sequences_flat)
            gen_sequences = gen_sequences_orig.reshape(original_shape)

        print(f"Generated sequences shape: {gen_sequences.shape}")
        return gen_sequences
    
    def save_model(self, filename_prefix):
        """
        Save the TTS-CGAN model
        """
        # Save generator
        self.generator.save(f"{filename_prefix}_generator.h5")
        
        # Save discriminator
        self.discriminator.save(f"{filename_prefix}_discriminator.h5")
        
        print(f"Model saved with prefix: {filename_prefix}")
    
    @classmethod
    def load_model(cls, filename_prefix, sequence_length, num_features, num_classes, latent_dim):
        """
        Load a saved TTS-CGAN model
        """
        # Create a new model instance
        model = cls(sequence_length, num_features, num_classes, latent_dim)
        
        # Load weights for generator and discriminator
        model.generator = keras.models.load_model(f"{filename_prefix}_generator.h5", custom_objects={"wasserstein_loss": wasserstein_loss})
        model.discriminator = keras.models.load_model(f"{filename_prefix}_discriminator.h5", custom_objects={"wasserstein_loss": wasserstein_loss})
        
        return model

# Data preprocessing functions
def load_and_preprocess_data(file_path, sequence_length=SEQUENCE_LENGTH, class_categories=3, step_size=10):
    """
    Load and preprocess the eye openness data with configurable sliding window
    """
    # Load data
    df = pd.read_csv(file_path)
    data = df['Left_eye_openness'].values.reshape(-1, 1)
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences with sliding window
    sequences = []
    for i in range(0, len(scaled_data) - sequence_length + 1, step_size):
        seq = scaled_data[i:i + sequence_length].reshape(sequence_length, 1)
        sequences.append(seq)
    
    sequences = np.array(sequences)
    
    # Create labels
    mean_values = np.mean(sequences, axis=1).flatten()
    labels = np.zeros((len(mean_values), class_categories))
    
    for i, mean in enumerate(mean_values):
        normalized_mean = (mean + 1) / 2
        if normalized_mean > 0.95:
            labels[i, 2] = 1  # Fully open
        elif normalized_mean > 0.8:
            labels[i, 1] = 1  # Partially open
        else:
            labels[i, 0] = 1  # Closed/mostly closed
    
    print(f"Generated {len(sequences)} sequences from {len(data)} original datapoints")
    return sequences, labels, scaler

# Evaluation functions
def evaluate_with_dtw(real_data, synthetic_data, num_samples=10):
    """
    Evaluate the similarity between real and synthetic data using DTW
    """
    dtw_distances = []
    
    # Select random samples from both datasets
    real_indices = np.random.choice(len(real_data), num_samples, replace=False)
    synthetic_indices = np.random.choice(len(synthetic_data), num_samples, replace=False)
    
    for i, j in zip(real_indices, synthetic_indices):
        real_seq = real_data[i].flatten()
        synthetic_seq = synthetic_data[j].flatten()
        
        # Apply Dynamic Time Warping
        distance, path = dtw(real_seq, synthetic_seq, dist=lambda x, y: norm(x - y))
        dtw_distances.append(distance)
    
    return np.mean(dtw_distances), dtw_distances

def calculate_wasserstein_distance(real_data, synthetic_data, num_samples=10):
    """
    Calculate the Wasserstein distance between real and synthetic distributions
    """
    distances = []
    
    # Select random samples from both datasets
    real_indices = np.random.choice(len(real_data), num_samples, replace=False)
    synthetic_indices = np.random.choice(len(synthetic_data), num_samples, replace=False)
    
    for i, j in zip(real_indices, synthetic_indices):
        real_seq = real_data[i].flatten()
        synthetic_seq = synthetic_data[j].flatten()
        
        # Calculate Wasserstein distance
        distance = wasserstein_distance(real_seq, synthetic_seq)
        distances.append(distance)
    
    return np.mean(distances), distances

def compare_real_synthetic_sequences(real_data, synthetic_data, scaler=None, num_samples=5, output_dir='output'):
    """
    Compare and visualize real and synthetic sequences
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Select random samples to visualize
    real_indices = np.random.choice(len(real_data), num_samples, replace=False)
    synthetic_indices = np.random.choice(len(synthetic_data), num_samples, replace=False)
    
    plt.figure(figsize=(15, 10))
    
    # Plot real data
    for i, idx in enumerate(real_indices):
        plt.subplot(2, num_samples, i+1)
        plt.plot(real_data[idx], 'b-', label='Real')
        plt.title(f"Real Sequence {idx}")
        plt.ylim(-1.1, 1.1)
    
    # Plot synthetic data
    for i, idx in enumerate(synthetic_indices):
        plt.subplot(2, num_samples, num_samples+i+1)
        plt.plot(synthetic_data[idx], 'r-', label='Synthetic')
        plt.title(f"Synthetic Sequence {idx}")
        plt.ylim(-1.1, 1.1)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/real_vs_synthetic_comparison.png")
    plt.close()

    # Also create a direct comparison for DTW
    plt.figure(figsize=(15, 10))
    
    for i in range(min(5, num_samples)):
        plt.subplot(2, 3, i+1)
        plt.plot(real_data[real_indices[i]], 'b-', label='Real')
        plt.plot(synthetic_data[synthetic_indices[i]], 'r-', label='Synthetic')
        plt.title(f"Comparison {i+1}")
        plt.legend()
        plt.ylim(-1.1, 1.1)
    
    # Calculate and plot DTW alignment for one example
    real_seq = real_data[real_indices[0]].flatten()
    synthetic_seq = synthetic_data[synthetic_indices[0]].flatten()
    
    distance, path = dtw(real_seq, synthetic_seq, dist=lambda x, y: norm(x - y))
    
    plt.subplot(2, 3, 6)
    plt.plot(path[0], 'b-', label='Path Index Real')
    plt.plot(path[1], 'r-', label='Path Index Synthetic')
    plt.title(f"DTW Alignment\nDistance: {distance:.4f}")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dtw_comparison.png")
    plt.close()
    
    return real_indices, synthetic_indices

def analyze_minimal_data_requirements(dataset_path, min_percentages, sequence_length=SEQUENCE_LENGTH, 
                                    num_classes=3, latent_dim=LATENT_DIM, epochs=EPOCHS, 
                                    batch_size=BATCH_SIZE, output_dir='output'):
    """
    Analyze the minimal amount of real data required for effective data augmentation
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    sequences, labels, scaler = load_and_preprocess_data(dataset_path, sequence_length, num_classes)
    
    # Reserve a test set (20% of the data)
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        sequences, labels, test_size=0.2, random_state=42
    )
    
    # Results tracking
    results = []
    
    # Train models with varying amounts of real data
    for percentage in min_percentages:
        print(f"\nTraining with {percentage}% of real data:")
        
        # Calculate the number of samples to use
        num_samples = int(len(x_train_full) * percentage / 100)
        
        if num_samples < batch_size:
            print(f"Warning: Sample size {num_samples} is less than batch size {batch_size}.")
            print("Using smallest possible batch size.")
            current_batch_size = max(1, num_samples // 2)
        else:
            current_batch_size = batch_size
        
        # Subsample the training data
        indices = np.random.choice(len(x_train_full), num_samples, replace=False)
        x_train = x_train_full[indices]
        y_train = y_train_full[indices]
        
        print(f"Training with {len(x_train)} real samples")
        
        # Initialize and train the TTS-CGAN model
        num_features = x_train.shape[2]
        
        ttscgan = TTSCGAN(
            sequence_length=sequence_length,
            num_features=num_features,
            num_classes=num_classes,
            latent_dim=latent_dim
        )
        
        # Train the model
        d_losses, g_losses = ttscgan.train(
            x_train=x_train, 
            y_train=y_train,
            epochs=epochs,
            batch_size=current_batch_size,
            save_interval=100,
            output_dir=f"{output_dir}/{percentage}_percent"
        )
        
        # Generate synthetic data for evaluation
        synthetic_data_all = []
        
        for class_idx in range(num_classes):
            synthetic_data = ttscgan.generate_synthetic_sequences(
                num_sequences=100,
                class_idx=class_idx
            )
            synthetic_data_all.append(synthetic_data)
        
        synthetic_data_all = np.vstack(synthetic_data_all)
        
        # Evaluate using Wasserstein distance
    ws_mean, ws_distances = calculate_wasserstein_distance(x_test, all_synthetic)
    print(f"Wasserstein Mean Distance: {ws_mean:.4f}")
    
    # Compare and visualize
    compare_real_synthetic_sequences(
        x_test, all_synthetic, 
        scaler=scaler,
        output_dir=output_dir
    )
    
    # Analyze minimal data requirements
    print("\nAnalyzing minimal data requirements...")
    min_percentages = [5, 10, 20, 30, 50, 70]
    results_df = analyze_minimal_data_requirements(
        dataset_path="left_eye_openness.csv",
        min_percentages=min_percentages,
        sequence_length=SEQUENCE_LENGTH,
        num_classes=num_classes,
        latent_dim=LATENT_DIM,
        epochs=300,  # Use fewer epochs for this analysis
        batch_size=BATCH_SIZE,
        output_dir=f"{output_dir}/minimal_data_analysis"
    )
    
    print("\nMinimal data requirements analysis results:")
    print(results_df)
    
    # Plot the minimal data requirements results
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['percentage'], results_df['dtw_mean'], 'o-', label='DTW Distance')
    plt.plot(results_df['percentage'], results_df['ws_mean'], 's-', label='Wasserstein Distance')
    plt.xlabel('Percentage of Real Data')
    plt.ylabel('Distance Metric')
    plt.title('Effect of Real Data Percentage on Synthetic Data Quality')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/minimal_data_requirements_summary.png")
    plt.close()
    
    print("\nTraining and evaluation complete!")

    if __name__ == "__main__":
        main()
        dtw_mean, dtw_distances = evaluate_with_dtw(x_test, synthetic_data_all)
        
        # Evaluate using Wasserstein distance
        ws_mean, ws_distances = calculate_wasserstein_distance(x_test, synthetic_data_all)
        
        # Compare and visualize
        real_indices, synthetic_indices = compare_real_synthetic_sequences(
            x_test, synthetic_data_all, 
            scaler=scaler,
            output_dir=f"{output_dir}/{percentage}_percent"
        )
        
        # Save model
        ttscgan.save_model(f"{output_dir}/{percentage}_percent/ttscgan")
        
        # Record results
        results.append({
            'percentage': percentage,
            'num_samples': num_samples,
            'dtw_mean': dtw_mean,
            'ws_mean': ws_mean
        })
        
        print(f"Results for {percentage}% of data:")
        print(f"  DTW Distance: {dtw_mean:.4f}")
        print(f"  Wasserstein Distance: {ws_mean:.4f}")
    
    # Create summary plot
    plt.figure(figsize=(12, 6))
    
    percentages = [result['percentage'] for result in results]
    dtw_means = [result['dtw_mean'] for result in results]
    ws_means = [result['ws_mean'] for result in results]
    
    plt.subplot(1, 2, 1)
    plt.plot(percentages, dtw_means, 'bo-')
    plt.xlabel('Percentage of Real Data')
    plt.ylabel('DTW Distance')
    plt.title('DTW Distance vs. Real Data Percentage')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(percentages, ws_means, 'ro-')
    plt.xlabel('Percentage of Real Data')
    plt.ylabel('Wasserstein Distance')
    plt.title('Wasserstein Distance vs. Real Data Percentage')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/minimal_data_requirements_analysis.png")
    plt.close()
    
    # Create results table
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/minimal_data_requirements_results.csv", index=False)
    
    return results_df

def main():
    """
    Main execution function
    """
    # Set output directory
    output_dir = 'ttscgan_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    sequences, labels, scaler = load_and_preprocess_data("left_eye_openness.csv", SEQUENCE_LENGTH)
    
    # Split into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)
    
    print(f"Data shape: {sequences.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Number of training samples: {x_train.shape[0]}")
    print(f"Number of testing samples: {x_test.shape[0]}")
    
    # Initialize and build the TTS-CGAN model
    print("Building TTS-CGAN model...")
    num_features = sequences.shape[2]
    num_classes = labels.shape[1]
    
    ttscgan = TTSCGAN(
        sequence_length=SEQUENCE_LENGTH,
        num_features=num_features,
        num_classes=num_classes,
        latent_dim=LATENT_DIM
    )
    
    # Print model summary
    print("\nGenerator Summary:")
    ttscgan.generator.summary()
    
    print("\nDiscriminator Summary:")
    ttscgan.discriminator.summary()
    
    # Train the model
    print("\nTraining the model...")
    d_losses, g_losses = ttscgan.train(
        x_train=x_train, 
        y_train=y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        save_interval=50,
        output_dir=output_dir
    )
    
    # Save the trained model
    print("\nSaving the model...")
    ttscgan.save_model(f"{output_dir}/eye_openness_ttscgan")
    
    # Generate synthetic data for each class and evaluate
    print("\nGenerating and evaluating synthetic data...")
    for class_idx in range(num_classes):
        print(f"Generating synthetic data for class {class_idx}")
        synthetic_sequences = ttscgan.generate_synthetic_sequences(
            num_sequences=50,
            class_idx=class_idx,
            scaler=scaler
        )
        
        # Save to CSV
        df_synthetic = pd.DataFrame()
        for i, seq in enumerate(synthetic_sequences):
            df_synthetic[f'sequence_{i+1}'] = seq.flatten()
        
        df_synthetic.to_csv(f"{output_dir}/synthetic_class_{class_idx}.csv", index=False)
        
        # Plot sample sequences
        plt.figure(figsize=(12, 6))
        for i in range(min(10, len(synthetic_sequences))):
            plt.subplot(2, 5, i+1)
            plt.plot(synthetic_sequences[i])
            plt.title(f"Sequence {i+1}")
            plt.ylim(-1, 1)
        
        plt.suptitle(f"Generated Sequences - Class {class_idx}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/generated_samples_class_{class_idx}.png")
        plt.close()
    
    # Compare real and synthetic data
    all_synthetic = []
    for class_idx in range(num_classes):
        synthetic = ttscgan.generate_synthetic_sequences(
            num_sequences=len(x_test) // num_classes,
            class_idx=class_idx
        )
        all_synthetic.append(synthetic)
    
    all_synthetic = np.vstack(all_synthetic)
    
    # Evaluate using DTW
    dtw_mean, dtw_distances = evaluate_with_dtw(x_test, all_synthetic)
    print(f"\nDTW Mean Distance: {dtw_mean:.4f}")
    
    # Evaluate using Wasserstein distance
    ws_mean, ws_distances = calculate_wasserstein_distance(x_test, all_synthetic)
    print(f"Wasserstein Mean Distance: {ws_mean:.4f}")
    
    # Compare and visualize
    compare_real_synthetic_sequences(
        x_test, all_synthetic, 
        scaler=scaler,
        output_dir=output_dir
    )
    
    # Plot real vs synthetic data for each class
    plt.figure(figsize=(15, 10))
    
    for class_idx in range(num_classes):
        # Get real samples for this class
        class_indices = np.where(y_test[:, class_idx] == 1)[0]
        if len(class_indices) == 0:
            continue
            
        real_samples = x_test[class_indices][:5]  # Just use a few samples
        
        # Generate synthetic samples
        synthetic_samples = ttscgan.generate_synthetic_sequences(
            num_sequences=5,
            class_idx=class_idx
        )
        
        # Plot real data
        plt.subplot(num_classes, 2, 2*class_idx+1)
        for i in range(len(real_samples)):
            plt.plot(real_samples[i], alpha=0.7)
        plt.title(f"Real Data - Class {class_idx}")
        plt.ylim(-1.1, 1.1)
        
        # Plot synthetic data
        plt.subplot(num_classes, 2, 2*class_idx+2)
        for i in range(len(synthetic_samples)):
            plt.plot(synthetic_samples[i], alpha=0.7)
        plt.title(f"Synthetic Data - Class {class_idx}")
        plt.ylim(-1.1, 1.1)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/real_vs_synthetic_by_class.png")
    plt.close()
    
    # Calculate and print statistics for each class
    print("\nClass Statistics:")
    for class_idx in range(num_classes):
        # Get real samples for this class
        class_indices = np.where(y_test[:, class_idx] == 1)[0]
        if len(class_indices) == 0:
            continue
            
        real_samples = x_test[class_indices]
        
        # Generate synthetic samples
        synthetic_samples = ttscgan.generate_synthetic_sequences(
            num_sequences=min(100, len(class_indices)),
            class_idx=class_idx
        )
        
        # Calculate statistics
        real_mean = np.mean(real_samples)
        real_std = np.std(real_samples)
        synthetic_mean = np.mean(synthetic_samples)
        synthetic_std = np.std(synthetic_samples)
        
        # Transform back to original scale if scaler is provided
        if scaler is not None:
            real_orig = scaler.inverse_transform(real_samples.reshape(-1, 1)).reshape(real_samples.shape)
            synthetic_orig = scaler.inverse_transform(synthetic_samples.reshape(-1, 1)).reshape(synthetic_samples.shape)
            
            real_orig_mean = np.mean(real_orig)
            real_orig_std = np.std(real_orig)
            synthetic_orig_mean = np.mean(synthetic_orig)
            synthetic_orig_std = np.std(synthetic_orig)
            
            print(f"Class {class_idx}:")
            print(f"  Real data:      Mean = {real_mean:.4f}, Std = {real_std:.4f}")
            print(f"  Synthetic data: Mean = {synthetic_mean:.4f}, Std = {synthetic_std:.4f}")
            print(f"  Original scale - Real data:      Mean = {real_orig_mean:.4f}, Std = {real_orig_std:.4f}")
            print(f"  Original scale - Synthetic data: Mean = {synthetic_orig_mean:.4f}, Std = {synthetic_orig_std:.4f}")
        else:
            print(f"Class {class_idx}:")
            print(f"  Real data:      Mean = {real_mean:.4f}, Std = {real_std:.4f}")
            print(f"  Synthetic data: Mean = {synthetic_mean:.4f}, Std = {synthetic_std:.4f}")
        print()
    
    print("\nTraining and evaluation complete!")

if __name__ == "__main__":
    main()