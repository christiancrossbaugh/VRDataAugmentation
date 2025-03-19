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
from dtaidistance import dtw
from dtaidistance import preprocessing
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Add dataset prefetching
AUTOTUNE = tf.data.AUTOTUNE

def create_dataset(x_train, y_train, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(buffer_size=len(x_train))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Global parameters
SEQUENCE_LENGTH = 50  # Length of time sequence to generate
LATENT_DIM = 128  # Dimension of noise input to generator
BATCH_SIZE = 64
EPOCHS = 200
NUM_TRANSFORMER_BLOCKS = 4
NUM_ATTENTION_HEADS = 8
EMBEDDING_DIM = 256
FF_DIM = EMBEDDING_DIM * 4  # Feed-forward dimension
DROPOUT_RATE = 0.2
LEARNING_RATE_GENERATOR = 0.0001
LEARNING_RATE_DISCRIMINATOR = 0.0002
GRADIENT_PENALTY_WEIGHT = 15.0
DISCRIMINATOR_UPDATES = 3

LEARNING_RATE_DECAY = 0.95            # Learning rate decay factor
DECAY_STEPS = 1000                    # Steps before decay
WARMUP_EPOCHS = 5                     # Gradual warmup
BATCH_ACCUMULATION = 4                # Effective batch size = 64 * 4

"""
Sample test configuration for the TTS-CGAN model
EMBEDDING_DIM = 128
NUM_TRANSFORMER_BLOCKS = 3
NUM_ATTENTION_HEADS = 8
DROPOUT_RATE = 0.1
LEARNING_RATE_GENERATOR = 0.0002
LEARNING_RATE_DISCRIMINATOR = 0.0002
"""
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

def build_generator(latent_dim, sequence_length, num_features, num_classes):
    """Build the Transformer-based Generator model"""
    # All layers explicitly set as trainable
    noise_input = keras.Input(shape=(latent_dim,), name="noise_input")
    label_input = keras.Input(shape=(num_classes,), name="label_input")
    
    combined_input = layers.Concatenate()([noise_input, label_input])
    
    x = layers.Dense(sequence_length * EMBEDDING_DIM, trainable=True,
                    kernel_initializer='glorot_uniform')(combined_input)
    x = layers.BatchNormalization(trainable=True)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((sequence_length, EMBEDDING_DIM))(x)
    
    # Add positional encoding
    pos_enc = positional_encoding(sequence_length, EMBEDDING_DIM)
    pos_enc = pos_enc[:sequence_length, :EMBEDDING_DIM]
    pos_enc = tf.expand_dims(pos_enc, 0)
    x = x + pos_enc
    
    # Transformer blocks
    for i in range(NUM_TRANSFORMER_BLOCKS):
        x = transformer_encoder_block(
            inputs=x,
            num_heads=NUM_ATTENTION_HEADS,
            key_dim=EMBEDDING_DIM // NUM_ATTENTION_HEADS,
            dropout_rate=DROPOUT_RATE,
            trainable=True,
            name=f'generator_transformer_block_{i}'
        )
    
    output = layers.Dense(num_features, trainable=True,
                         kernel_initializer='glorot_uniform')(x)
    
    model = keras.Model([noise_input, label_input], output, name="generator")
    
    # Explicitly set all layers as trainable
    for layer in model.layers:
        layer.trainable = True
    
    return model

def transformer_encoder_block(inputs, num_heads, key_dim, dropout_rate, trainable=True, name='transformer_block'):
    """
    Transformer encoder block with explicit trainable settings
    """
    # Layer normalization
    x = layers.LayerNormalization(
        epsilon=1e-6, 
        trainable=trainable,
        name=f'{name}_norm1'
    )(inputs)
    
    # Multi-head self-attention
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=key_dim,
        dropout=dropout_rate,
        trainable=trainable,
        name=f'{name}_attention'
    )(x, x)
    
    # Add & Norm with dropout
    attention_output = layers.Dropout(
        dropout_rate,
        name=f'{name}_dropout1'
    )(attention_output)
    x1 = layers.Add(name=f'{name}_add1')([inputs, attention_output])
    
    # Feed-forward network
    x2 = layers.LayerNormalization(
        epsilon=1e-6, 
        trainable=trainable,
        name=f'{name}_norm2'
    )(x1)
    
    # Two dense layers with GELU activation
    x2 = layers.Dense(
        EMBEDDING_DIM * 4, 
        activation='gelu',
        trainable=trainable,
        kernel_initializer='glorot_uniform',
        name=f'{name}_dense1'
    )(x2)
    
    x2 = layers.Dense(
        EMBEDDING_DIM,
        trainable=trainable,
        kernel_initializer='glorot_uniform',
        name=f'{name}_dense2'
    )(x2)
    
    x2 = layers.Dropout(
        dropout_rate,
        name=f'{name}_dropout2'
    )(x2)
    
    # Final Add & Norm
    return layers.Add(name=f'{name}_add2')([x1, x2])

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
    
    # Initial dense projection with explicit trainable setting
    x = layers.Dense(EMBEDDING_DIM, 
                    trainable=True,
                    kernel_initializer='glorot_uniform',
                    name='initial_projection')(x)
    x = layers.BatchNormalization(trainable=True)(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Transformer encoder blocks with explicit trainable settings
    for i in range(NUM_TRANSFORMER_BLOCKS):
        x = transformer_encoder_block(
            inputs=x,
            num_heads=NUM_ATTENTION_HEADS,
            key_dim=EMBEDDING_DIM // NUM_ATTENTION_HEADS,
            dropout_rate=DROPOUT_RATE,
            trainable=True,
            name=f'discriminator_transformer_block_{i}'
        )
        # Add batch normalization after each transformer block
        x = layers.BatchNormalization(trainable=True,
                                    name=f'bn_after_transformer_{i}')(x)
    
    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers before heads with explicit trainable setting
    x = layers.Dense(512, trainable=True,
                    kernel_initializer='glorot_uniform',
                    name='pre_head_dense')(x)
    x = layers.BatchNormalization(trainable=True,
                                name='pre_head_bn')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    
    # Dual heads with explicit trainable settings:
    # 1. Real/Fake classification
    adversarial_head = layers.Dense(256, trainable=True,
                                  kernel_initializer='glorot_uniform',
                                  name='adversarial_dense')(x)
    adversarial_head = layers.LeakyReLU(0.2)(adversarial_head)
    adversarial_head = layers.Dropout(DROPOUT_RATE)(adversarial_head)
    adversarial_output = layers.Dense(1, 
                                    activation="sigmoid",
                                    trainable=True,
                                    kernel_initializer='glorot_uniform',
                                    name="adversarial")(adversarial_head)
    
    # 2. Class prediction
    classifier_head = layers.Dense(256, trainable=True,
                                 kernel_initializer='glorot_uniform',
                                 name='classifier_dense')(x)
    classifier_head = layers.LeakyReLU(0.2)(classifier_head)
    classifier_head = layers.Dropout(DROPOUT_RATE)(classifier_head)
    classifier_output = layers.Dense(num_classes, 
                                   activation="softmax",
                                   trainable=True,
                                   kernel_initializer='glorot_uniform',
                                   name="classifier")(classifier_head)
    
    # Create model
    model = keras.Model(sequence_input, 
                       [adversarial_output, classifier_output], 
                       name="discriminator")
    
    # Explicitly set all layers as trainable
    for layer in model.layers:
        layer.trainable = True
        print(f"Setting {layer.name} as trainable")
    
    # Verify trainable weights
    trainable_count = len(model.trainable_weights)
    print(f"\nDiscriminator has {trainable_count} trainable weights")
    
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

def get_learning_rate_schedule(initial_lr, decay_steps, decay_rate):
    """Create learning rate schedule with warmup"""
    def lr_schedule(epoch):
        # Warmup phase
        if epoch < WARMUP_EPOCHS:
            return initial_lr * ((epoch + 1) / WARMUP_EPOCHS)
        
        # Decay phase
        decay_epoch = epoch - WARMUP_EPOCHS
        return initial_lr * (decay_rate ** (decay_epoch // decay_steps))
    
    return lr_schedule

# TTS-CGAN Class
class TTSCGAN:
    def __init__(self, sequence_length, num_features, num_classes, latent_dim):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        # Build models with explicit trainable settings
        self.generator = build_generator(latent_dim, sequence_length, num_features, num_classes)
        self.discriminator = build_discriminator(sequence_length, num_features, num_classes)

        # Explicitly set layers as trainable
        for layer in self.generator.layers:
            layer.trainable = True
        for layer in self.discriminator.layers:
            layer.trainable = True

        # Compile discriminator with explicit trainable setting
        self.discriminator.trainable = True
        self.discriminator.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=LEARNING_RATE_DISCRIMINATOR, 
                beta_1=0.9, 
                beta_2=0.999
            ),
            loss=[wasserstein_loss, "categorical_crossentropy"],
            loss_weights=[1.0, 1.0]
        )

         # Add learning rate schedulers
        self.g_lr_schedule = get_learning_rate_schedule(
            LEARNING_RATE_GENERATOR, DECAY_STEPS, LEARNING_RATE_DECAY
        )
        self.d_lr_schedule = get_learning_rate_schedule(
            LEARNING_RATE_DISCRIMINATOR, DECAY_STEPS, LEARNING_RATE_DECAY
        )

        # Build combined model for generator training
        self.discriminator.trainable = False  # Freeze discriminator for generator training
        noise_input = keras.Input(shape=(latent_dim,))
        label_input = keras.Input(shape=(num_classes,))
        generated_sequence = self.generator([noise_input, label_input])
        valid, target_label = self.discriminator(generated_sequence)

        self.combined = keras.Model([noise_input, label_input], [valid, target_label])
        self.combined.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=LEARNING_RATE_GENERATOR, 
                beta_1=0.9, 
                beta_2=0.999
            ),
            loss=[wasserstein_loss, "categorical_crossentropy"],
            loss_weights=[1.0, 1.0]
        )

        # Print model information
        print("\nModel Trainable Parameters:")
        print("Generator trainable weights:", len(self.generator.trainable_weights))
        print("Discriminator trainable weights:", len(self.discriminator.trainable_weights))
        print("Combined model trainable weights:", len(self.combined.trainable_weights))

        print("\nGenerator layers:")
        for layer in self.generator.layers:
            print(f"{layer.name}: trainable = {layer.trainable}, weights = {len(layer.trainable_weights)}")

        print("\nDiscriminator layers:")
        for layer in self.discriminator.layers:
            print(f"{layer.name}: trainable = {layer.trainable}, weights = {len(layer.trainable_weights)}")
    
    def train(self, x_train, y_train, epochs, batch_size, save_interval=50, output_dir='output'):
        # Add check for trainable weights
        generator_trainable = any(layer.trainable for layer in self.generator.layers)
        discriminator_trainable = any(layer.trainable for layer in self.discriminator.layers)

        # Print training configuration
        print("\nTraining Configuration:")
        print(f"Generator learning rate: {LEARNING_RATE_GENERATOR}")
        print(f"Discriminator learning rate: {LEARNING_RATE_DISCRIMINATOR}")
        print(f"Batch size: {batch_size}")
        print(f"Batch accumulation steps: {BATCH_ACCUMULATION}")
        print(f"Effective batch size: {batch_size * BATCH_ACCUMULATION}")
        print(f"Epochs: {epochs}")
        print(f"Training samples: {len(x_train)}")

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

                # Batch accumulation loop
                d_loss_accumulated = None
                g_loss_accumulated = None

                for acc_step in range(BATCH_ACCUMULATION):
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

                        # Accumulate losses
                        if d_loss_accumulated is None:
                            d_loss_accumulated = np.array(d_loss) / BATCH_ACCUMULATION
                        else:
                            d_loss_accumulated += np.array(d_loss) / BATCH_ACCUMULATION

                    d_loss_epoch.append(d_loss_accumulated)

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

                        # Accumulate generator losses
                        if g_loss_accumulated is None:
                            g_loss_accumulated = np.array(g_loss) / BATCH_ACCUMULATION
                        else:
                            g_loss_accumulated += np.array(g_loss) / BATCH_ACCUMULATION

                    g_loss_epoch.append(g_loss_accumulated)

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
    
    def gradient_penalty(self, real_samples, fake_samples):
        batch_size = real_samples.shape[0]
        alpha = np.random.random((batch_size, 1, 1))
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            validity, _ = self.discriminator(interpolated)

        gradients = tape.gradient(validity, interpolated)[0]
        gradients_sqr = K.square(gradients)
        gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        gradient_penalty = K.square(1 - gradient_l2_norm)

        return K.mean(gradient_penalty)

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
    try:
        dtw_distances = []
        
        # Select random samples
        real_indices = np.random.choice(len(real_data), num_samples, replace=False)
        synthetic_indices = np.random.choice(len(synthetic_data), num_samples, replace=False)
        
        for i, j in zip(real_indices, synthetic_indices):
            # Flatten and normalize sequences
            real_seq = preprocessing.normalize(real_data[i].flatten())
            synthetic_seq = preprocessing.normalize(synthetic_data[j].flatten())
            
            # Convert to correct dtype
            real_seq = real_seq.astype(np.double)
            synthetic_seq = synthetic_seq.astype(np.double)
            
            # Compute DTW distance
            distance = dtw.distance(real_seq, synthetic_seq)
            dtw_distances.append(distance)
        
        mean_distance = np.mean(dtw_distances)
        return mean_distance, dtw_distances
        
    except Exception as e:
        print(f"Error in DTW evaluation: {str(e)}")
        return None, None
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
    Compare and visualize real vs synthetic sequences
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Select random samples
        real_indices = np.random.choice(len(real_data), num_samples, replace=False)
        synthetic_indices = np.random.choice(len(synthetic_data), num_samples, replace=False)
        
        dtw_distances = []
        
        for idx, (real_idx, syn_idx) in enumerate(zip(real_indices, synthetic_indices)):
            # Get sequences
            real_seq = real_data[real_idx].flatten()
            synthetic_seq = synthetic_data[syn_idx].flatten()
            
            # Compute DTW
            distance, path = fastdtw(real_seq, synthetic_seq, dist=euclidean)
            dtw_distances.append(distance)
            
            # Create visualization
            plt.figure(figsize=(15, 5))
            
            # Plot sequences
            plt.subplot(1, 2, 1)
            plt.plot(real_seq, 'b-', label='Real')
            plt.plot(synthetic_seq, 'r--', label='Synthetic')
            plt.title(f'Sequence Comparison (DTW Distance: {distance:.4f})')
            plt.legend()
            plt.grid(True)
            
            # Plot DTW path
            plt.subplot(1, 2, 2)
            # Create distance matrix for visualization
            n = len(real_seq)
            m = len(synthetic_seq)
            dist_matrix = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    dist_matrix[i, j] = euclidean(real_seq[i], synthetic_seq[j])
            
            plt.imshow(dist_matrix, origin='lower', cmap='viridis', aspect='auto')
            plt.plot([x[1] for x in path], [x[0] for x in path], 'w-', linewidth=2)
            plt.title('DTW Alignment Path')
            plt.colorbar(label='Distance')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/comparison_{idx}.png')
            plt.close()
        
        # Plot DTW distance distribution
        plt.figure(figsize=(8, 5))
        plt.hist(dtw_distances, bins=20, alpha=0.7)
        plt.axvline(np.mean(dtw_distances), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(dtw_distances):.4f}')
        plt.title('DTW Distance Distribution')
        plt.xlabel('DTW Distance')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{output_dir}/dtw_distribution.png')
        plt.close()
        
        return np.mean(dtw_distances), dtw_distances
        
    except Exception as e:
        print(f"Error in sequence comparison: {str(e)}")
        return None, None

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import os

def compare_real_synthetic_sequences(real_data, synthetic_data, scaler=None, num_samples=5, output_dir='output'):
    """
    Compare and visualize real vs synthetic sequences
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Select random samples
        real_indices = np.random.choice(len(real_data), num_samples, replace=False)
        synthetic_indices = np.random.choice(len(synthetic_data), num_samples, replace=False)
        
        dtw_distances = []
        
        for idx, (real_idx, syn_idx) in enumerate(zip(real_indices, synthetic_indices)):
            # Get sequences
            real_seq = real_data[real_idx].flatten()
            synthetic_seq = synthetic_data[syn_idx].flatten()
            
            # Compute DTW
            distance, path = fastdtw(real_seq, synthetic_seq, dist=euclidean)
            dtw_distances.append(distance)
            
            # Create visualization
            plt.figure(figsize=(15, 5))
            
            # Plot sequences
            plt.subplot(1, 2, 1)
            plt.plot(real_seq, 'b-', label='Real')
            plt.plot(synthetic_seq, 'r--', label='Synthetic')
            plt.title(f'Sequence Comparison (DTW Distance: {distance:.4f})')
            plt.legend()
            plt.grid(True)
            
            # Plot DTW path
            plt.subplot(1, 2, 2)
            # Create distance matrix for visualization
            n = len(real_seq)
            m = len(synthetic_seq)
            dist_matrix = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    dist_matrix[i, j] = euclidean(real_seq[i], synthetic_seq[j])
            
            plt.imshow(dist_matrix, origin='lower', cmap='viridis', aspect='auto')
            plt.plot([x[1] for x in path], [x[0] for x in path], 'w-', linewidth=2)
            plt.title('DTW Alignment Path')
            plt.colorbar(label='Distance')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/comparison_{idx}.png')
            plt.close()
        
        # Plot DTW distance distribution
        plt.figure(figsize=(8, 5))
        plt.hist(dtw_distances, bins=20, alpha=0.7)
        plt.axvline(np.mean(dtw_distances), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(dtw_distances):.4f}')
        plt.title('DTW Distance Distribution')
        plt.xlabel('DTW Distance')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{output_dir}/dtw_distribution.png')
        plt.close()
        
        return np.mean(dtw_distances), dtw_distances
        
    except Exception as e:
        print(f"Error in sequence comparison: {str(e)}")
        return None, None

def run_minimal_data_experiment(dataset_path, min_percentages, sequence_length=100,
                              num_classes=3, latent_dim=64, epochs=200,
                              batch_size=32, output_dir='experiment_results',
                              num_trials=3):
    """
    Run experiment with varying amounts of training data
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        
        for percentage in min_percentages:
            print(f"\nTesting with {percentage}% of training data")
            
            for trial in range(num_trials):
                print(f"Trial {trial + 1}/{num_trials}")
                
                # Load and preprocess data
                sequences, labels, scaler = load_and_preprocess_data(
                    dataset_path, sequence_length=sequence_length
                )
                
                # Split data
                train_size = int(len(sequences) * (percentage / 100))
                indices = np.random.permutation(len(sequences))
                train_indices = indices[:train_size]
                test_indices = indices[train_size:]
                
                x_train = sequences[train_indices]
                y_train = labels[train_indices]
                x_test = sequences[test_indices]
                y_test = labels[test_indices]
                
                # Train model
                model = TTSCGAN(
                    sequence_length=sequence_length,
                    num_features=sequences.shape[2],
                    num_classes=num_classes,
                    latent_dim=latent_dim
                )
                
                model.train(
                    x_train=x_train,
                    y_train=y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    output_dir=f"{output_dir}/percentage_{percentage}/trial_{trial}"
                )
                
                # Generate synthetic data
                synthetic_data = []
                samples_per_class = len(x_test) // num_classes
                
                for class_idx in range(num_classes):
                    synthetic = model.generate_synthetic_sequences(
                        num_sequences=samples_per_class,
                        class_idx=class_idx,
                        scaler=scaler
                    )
                    synthetic_data.append(synthetic)
                
                synthetic_data = np.vstack(synthetic_data)
                
                # Evaluate
                dtw_mean, _ = evaluate_with_dtw(x_test, synthetic_data)
                
                results.append({
                    'percentage': percentage,
                    'trial': trial,
                    'dtw_mean': dtw_mean,
                    'train_size': train_size
                })
                
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate aggregated results
        agg_results = results_df.groupby('percentage').agg({
            'dtw_mean': ['mean', 'std'],
            'train_size': 'first'
        }).reset_index()
        
        # Save results
        results_df.to_csv(f'{output_dir}/detailed_results.csv', index=False)
        agg_results.to_csv(f'{output_dir}/aggregated_results.csv', index=False)
        
        return results_df, agg_results
        
    except Exception as e:
        print(f"Error in minimal data experiment: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

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