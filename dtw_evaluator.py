import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
import os
from scipy.spatial.distance import euclidean
from dtaidistance import dtw
from sklearn.preprocessing import MinMaxScaler  # Add this import

class DTWEvaluator:
    """
    A class for evaluating the quality of synthetic data using Dynamic Time Warping (DTW)
    and other metrics as described in the thesis proposal.
    """
    
    def __init__(self, output_dir='dtw_evaluation'):
        """
        Initialize the DTW evaluator
        
        Parameters:
        -----------
        output_dir : str
            Directory to save evaluation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

        def normalize_sequence(self, sequence):
            """
            Normalize a sequence to [-1, 1] range
            """
            seq_reshaped = sequence.reshape(-1, 1)
            return self.scaler.fit_transform(seq_reshaped).ravel()


        # Check if required packages are available
        try:
            import dtaidistance
            from dtaidistance import dtw
        except ImportError:
            raise ImportError("dtaidistance is required. Install it with: pip install dtaidistance")
    def compute_dtw_matrix(self, real_data, synthetic_data, sample_size=None):
        """
        Compute a DTW distance matrix between real and synthetic data samples
        """
        try:
            if sample_size is not None:
                real_indices = np.random.choice(len(real_data), min(sample_size, len(real_data)), replace=False)
                syn_indices = np.random.choice(len(synthetic_data), min(sample_size, len(synthetic_data)), replace=False)
                real_samples = real_data[real_indices]
                syn_samples = synthetic_data[syn_indices]
            else:
                real_samples = real_data
                syn_samples = synthetic_data
            
            # Initialize distance matrix
            dtw_matrix = np.zeros((len(real_samples), len(syn_samples)))
            
            # Compute DTW distance between each pair
            for i, real_seq in enumerate(real_samples):
                for j, syn_seq in enumerate(syn_samples):
                    try:
                        # Flatten and normalize
                        real_flat = self.normalize_sequence(real_seq.flatten())
                        syn_flat = self.normalize_sequence(syn_seq.flatten())
                        
                        # Convert to correct dtype
                        real_flat = real_flat.astype(np.double)
                        syn_flat = syn_flat.astype(np.double)
                        
                        # Compute DTW distance
                        distance = dtw.distance(real_flat, syn_flat)
                        dtw_matrix[i, j] = distance
                    except Exception as e:
                        print(f"Warning: Error computing DTW for pair ({i},{j}): {str(e)}")
                        dtw_matrix[i, j] = np.nan
            
            return dtw_matrix, real_samples, syn_samples
            
        except Exception as e:
            print(f"Error in compute_dtw_matrix: {str(e)}")
            return None, None, None
    
    def compute_wasserstein_matrix(self, real_data, synthetic_data, sample_size=None):
        """
        Compute a Wasserstein distance matrix between real and synthetic data samples
        
        Parameters:
        -----------
        real_data : numpy.ndarray
            Array of real data sequences
        synthetic_data : numpy.ndarray
            Array of synthetic data sequences
        sample_size : int, optional
            Number of samples to use (for large datasets)
            
        Returns:
        --------
        numpy.ndarray
            Wasserstein distance matrix
        """
        if sample_size is not None:
            real_indices = np.random.choice(len(real_data), min(sample_size, len(real_data)), replace=False)
            syn_indices = np.random.choice(len(synthetic_data), min(sample_size, len(synthetic_data)), replace=False)
            real_samples = real_data[real_indices]
            syn_samples = synthetic_data[syn_indices]
        else:
            real_samples = real_data
            syn_samples = synthetic_data
        
        # Initialize distance matrix
        ws_matrix = np.zeros((len(real_samples), len(syn_samples)))
        
        # Compute Wasserstein distance between each pair of real and synthetic sequences
        for i, real_seq in enumerate(real_samples):
            for j, syn_seq in enumerate(syn_samples):
                # Flatten sequences if they are multi-dimensional
                real_flat = real_seq.flatten()
                syn_flat = syn_seq.flatten()
                
                # Compute Wasserstein distance
                distance = wasserstein_distance(real_flat, syn_flat)
                ws_matrix[i, j] = distance
        
        return ws_matrix, real_samples, syn_samples
    
    def visualize_distance_matrix(self, distance_matrix, title, filename):
        """
        Visualize a distance matrix as a heatmap
        
        Parameters:
        -----------
        distance_matrix : numpy.ndarray
            Distance matrix to visualize
        title : str
            Title of the plot
        filename : str
            Filename to save the plot
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(distance_matrix, cmap='viridis', annot=False)
        plt.title(title)
        plt.xlabel('Synthetic Sequences')
        plt.ylabel('Real Sequences')
        plt.savefig(f"{self.output_dir}/{filename}")
        plt.close()
    
    def visualize_dtw_alignment(self, real_seq, synthetic_seq, seq_idx, class_label=None):
        """
        Visualize the DTW alignment between a real and synthetic sequence
        """
        # Flatten and normalize sequences
        real_flat = self.normalize_sequence(real_seq.flatten())
        syn_flat = self.normalize_sequence(synthetic_seq.flatten())
        
        # Convert to correct dtype
        real_flat = real_flat.astype(np.double)
        syn_flat = syn_flat.astype(np.double)
        
        # Compute DTW
        distance = dtw.distance(real_flat, syn_flat)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot sequences
        ax1.plot(real_flat, 'b-', label='Real')
        ax1.plot(syn_flat, 'r-', label='Synthetic')
        title = f"Sequence Comparison"
        if class_label is not None:
            title += f" - Class {class_label}"
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True)
        
        # Plot distance matrix
        try:
            d = dtw.distance_matrix_fast(real_flat.reshape(1, -1), syn_flat.reshape(1, -1))
            ax2.imshow(d, origin='lower', cmap='viridis', 
                      aspect='auto', interpolation='nearest')
        except Exception as e:
            print(f"Warning: Could not compute distance matrix: {str(e)}")
            ax2.text(0.5, 0.5, 'Distance matrix computation failed', 
                    ha='center', va='center')
        
        ax2.set_title(f"DTW Distance Matrix (Distance: {distance:.4f})")
        ax2.set_xlabel("Synthetic Sequence Index")
        ax2.set_ylabel("Real Sequence Index")
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/dtw_alignment_{seq_idx}.png")
        plt.close()
        
        return distance
    
    def evaluate_synthetic_data(self, real_data, synthetic_data, class_labels=None, sample_size=20):
        """
        Evaluate synthetic data quality using DTW and other metrics
        
        Parameters:
        -----------
        real_data : numpy.ndarray
            Array of real data sequences
        synthetic_data : numpy.ndarray
            Array of synthetic data sequences
        class_labels : numpy.ndarray, optional
            Class labels for each sequence
        sample_size : int
            Number of samples to use for evaluation
            
        Returns:
        --------
        dict
            Evaluation results
        """
         # Print shapes for debugging
        print(f"Input shapes - Real: {real_data.shape}, Synthetic: {synthetic_data.shape}")

        # Type checking and validation
        if not isinstance(real_data, np.ndarray) or not isinstance(synthetic_data, np.ndarray):
            raise TypeError("Input data must be numpy arrays")

        # Ensure data has correct shape
        if len(real_data.shape) != 3 or len(synthetic_data.shape) != 3:
            # Reshape if necessary
            try:
                if len(real_data.shape) == 2:
                    real_data = real_data.reshape(real_data.shape[0], -1, 1)
                if len(synthetic_data.shape) == 2:
                    synthetic_data = synthetic_data.reshape(synthetic_data.shape[0], -1, 1)
            except Exception as e:
                raise ValueError(f"Could not reshape data: {str(e)}")

        print(f"Processed shapes - Real: {real_data.shape}, Synthetic: {synthetic_data.shape}")

        # Check shapes match except for first dimension
        if real_data.shape[1:] != synthetic_data.shape[1:]:
            raise ValueError(
                f"Real and synthetic data must have the same shape except for the first dimension. "
                f"Got shapes {real_data.shape} and {synthetic_data.shape}"
            )
    
        # Adjust sample size if necessary
        if sample_size > min(len(real_data), len(synthetic_data)):
            sample_size = min(len(real_data), len(synthetic_data))
            print(f"Adjusted sample size to {sample_size}")
    
            # Compute DTW distance matrix
            dtw_matrix, real_samples, syn_samples = self.compute_dtw_matrix(
                real_data, synthetic_data, sample_size=sample_size
            )
            
            # Compute Wasserstein distance matrix
            ws_matrix, _, _ = self.compute_wasserstein_matrix(
                real_data, synthetic_data, sample_size=sample_size
            )
            
            # Visualize distance matrices
            self.visualize_distance_matrix(
                dtw_matrix, "DTW Distance Matrix", "dtw_distance_matrix.png"
            )
            self.visualize_distance_matrix(
                ws_matrix, "Wasserstein Distance Matrix", "wasserstein_distance_matrix.png"
            )
            
            # Visualize DTW alignments for a few examples
            num_examples = min(5, sample_size)
            dtw_distances = []
            
            for i in range(num_examples):
                real_idx = np.random.randint(0, len(real_samples))
                syn_idx = np.random.randint(0, len(syn_samples))
                
                class_label = None
                if class_labels is not None:
                    class_label = class_labels[real_idx]
                
                distance = self.visualize_dtw_alignment(
                    real_samples[real_idx], syn_samples[syn_idx], i, class_label
                )
                dtw_distances.append(distance)
            
            # Calculate summary statistics
            dtw_mean = np.mean(dtw_matrix)
            dtw_std = np.std(dtw_matrix)
            dtw_min = np.min(dtw_matrix)
            dtw_max = np.max(dtw_matrix)
            
            ws_mean = np.mean(ws_matrix)
            ws_std = np.std(ws_matrix)
            ws_min = np.min(ws_matrix)
            ws_max = np.max(ws_matrix)
            
            # Create histograms of distances
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.hist(dtw_matrix.flatten(), bins=30, alpha=0.7)
            plt.axvline(dtw_mean, color='r', linestyle='--', label=f'Mean: {dtw_mean:.4f}')
            plt.title('DTW Distance Distribution')
            plt.xlabel('DTW Distance')
            plt.ylabel('Frequency')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.hist(ws_matrix.flatten(), bins=30, alpha=0.7)
            plt.axvline(ws_mean, color='r', linestyle='--', label=f'Mean: {ws_mean:.4f}')
            plt.title('Wasserstein Distance Distribution')
            plt.xlabel('Wasserstein Distance')
            plt.ylabel('Frequency')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/distance_distributions.png")
            plt.close()
            
            # Compile results
            results = {
                'dtw_mean': dtw_mean,
                'dtw_std': dtw_std,
                'dtw_min': dtw_min,
                'dtw_max': dtw_max,
                'ws_mean': ws_mean,
                'ws_std': ws_std,
                'ws_min': ws_min,
                'ws_max': ws_max,
                'sample_size': sample_size
            }
            
            # Save results to CSV
            pd.DataFrame([results]).to_csv(f"{self.output_dir}/evaluation_results.csv", index=False)
            
        return results
    
    def evaluate_class_specific(self, real_data, synthetic_data, real_labels, synthetic_labels, num_classes):
        """
        Evaluate synthetic data quality per class
        
        Parameters:
        -----------
        real_data : numpy.ndarray
            Array of real data sequences
        synthetic_data : numpy.ndarray
            Array of synthetic data sequences
        real_labels : numpy.ndarray
            One-hot encoded class labels for real data
        synthetic_labels : numpy.ndarray
            One-hot encoded class labels for synthetic data
        num_classes : int
            Number of classes
            
        Returns:
        --------
        pandas.DataFrame
            Class-specific evaluation results
        """
        class_results = []
        
        for class_idx in range(num_classes):
            # Get indices for this class
            real_class_indices = np.where(real_labels[:, class_idx] == 1)[0]
            syn_class_indices = np.where(synthetic_labels[:, class_idx] == 1)[0]
            
            if len(real_class_indices) == 0 or len(syn_class_indices) == 0:
                print(f"Warning: No samples found for class {class_idx}")
                continue
            
            # Extract class-specific data
            real_class_data = real_data[real_class_indices]
            syn_class_data = synthetic_data[syn_class_indices]
            
            # Create class-specific output directory
            class_dir = f"{self.output_dir}/class_{class_idx}"
            os.makedirs(class_dir, exist_ok=True)
            
            # Compute DTW matrix for this class
            sample_size = min(20, len(real_class_indices), len(syn_class_indices))
            dtw_matrix, _, _ = self.compute_dtw_matrix(
                real_class_data, syn_class_data, sample_size=sample_size
            )
            
            # Compute Wasserstein matrix for this class
            ws_matrix, _, _ = self.compute_wasserstein_matrix(
                real_class_data, syn_class_data, sample_size=sample_size
            )
            
            # Calculate statistics
            dtw_mean = np.mean(dtw_matrix)
            dtw_std = np.std(dtw_matrix)
            ws_mean = np.mean(ws_matrix)
            ws_std = np.std(ws_matrix)
            
            # Record results
            class_results.append({
                'class': class_idx,
                'real_samples': len(real_class_indices),
                'synthetic_samples': len(syn_class_indices),
                'dtw_mean': dtw_mean,
                'dtw_std': dtw_std,
                'ws_mean': ws_mean,
                'ws_std': ws_std
            })
            
            # Visualize for this class
            plt.figure(figsize=(10, 8))
            
            # Plot a few real samples
            plt.subplot(2, 1, 1)
            for i in range(min(5, len(real_class_indices))):
                plt.plot(real_class_data[i].flatten())
            plt.title(f"Real Data Samples - Class {class_idx}")
            plt.grid(True)
            
            # Plot a few synthetic samples
            plt.subplot(2, 1, 2)
            for i in range(min(5, len(syn_class_indices))):
                plt.plot(syn_class_data[i].flatten())
            plt.title(f"Synthetic Data Samples - Class {class_idx}")
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{class_dir}/sample_comparison.png")
            plt.close()
        
        # Create results DataFrame
        results_df = pd.DataFrame(class_results)
        
        # Save results
        results_df.to_csv(f"{self.output_dir}/class_specific_results.csv", index=False)
        
        # Visualize class-specific results
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.bar(results_df['class'], results_df['dtw_mean'], yerr=results_df['dtw_std'], capsize=5)
        plt.title('DTW Distance by Class')
        plt.xlabel('Class')
        plt.ylabel('DTW Distance (Mean)')
        plt.grid(True, axis='y')
        
        plt.subplot(1, 2, 2)
        plt.bar(results_df['class'], results_df['ws_mean'], yerr=results_df['ws_std'], capsize=5)
        plt.title('Wasserstein Distance by Class')
        plt.xlabel('Class')
        plt.ylabel('Wasserstein Distance (Mean)')
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/class_specific_results.png")
        plt.close()
        
        return results_df