import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Import the TTS-CGAN implementation
from ttscgan_implementation import (
    TTSCGAN, load_and_preprocess_data, evaluate_with_dtw,
    calculate_wasserstein_distance, compare_real_synthetic_sequences
)

def run_minimal_data_experiment(
    dataset_path,
    min_percentages=[5, 10, 20, 30, 50, 70],
    sequence_length=100,
    num_classes=3,
    latent_dim=64,
    epochs=200,
    batch_size=32,
    output_dir='experiment_results',
    num_trials=3
):
    """
    Run experiment to determine minimal data requirements for effective augmentation
    
    Parameters:
    -----------
    dataset_path : str
        Path to the dataset CSV file
    min_percentages : list
        List of percentages of real data to use
    sequence_length : int
        Length of sequences to generate
    num_classes : int
        Number of classes in the dataset
    latent_dim : int
        Dimension of latent space for generator
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    output_dir : str
        Directory to save results
    num_trials : int
        Number of trials to run for each percentage
    
    Returns:
    --------
    pandas.DataFrame
        Results of the experiment
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    sequences, labels, scaler = load_and_preprocess_data(dataset_path, sequence_length, num_classes)
    
    # Reserve a test set (20% of the data)
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        sequences, labels, test_size=0.2, random_state=42
    )
    
    # Results storage
    all_results = []
    
    # For each percentage of real data
    for percentage in min_percentages:
        print(f"\n{'='*50}")
        print(f"Running experiments with {percentage}% of real data")
        print(f"{'='*50}")
        
        # Calculate sample size
        num_samples = int(len(x_train_full) * percentage / 100)
        
        if num_samples < batch_size:
            print(f"Warning: Sample size {num_samples} is less than batch size {batch_size}.")
            print("Adjusting batch size.")
            current_batch_size = max(1, num_samples // 2)
        else:
            current_batch_size = batch_size
        
        # Run multiple trials
        for trial in range(num_trials):
            print(f"\nTrial {trial+1}/{num_trials}")
            
            # Create trial output directory
            trial_dir = f"{output_dir}/{percentage}_percent/trial_{trial+1}"
            os.makedirs(trial_dir, exist_ok=True)
            
            # Subsample the training data (different for each trial)
            indices = np.random.choice(len(x_train_full), num_samples, replace=False)
            x_train = x_train_full[indices]
            y_train = y_train_full[indices]
            
            print(f"Training with {len(x_train)} real samples")
            
            # Initialize model
            num_features = x_train.shape[2]
            
            # Set seed for reproducibility, but different for each trial
            tf.random.set_seed(42 + trial)
            np.random.seed(42 + trial)
            
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
                output_dir=trial_dir
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
            
            # Evaluate using DTW
            dtw_mean, dtw_distances = evaluate_with_dtw(x_test, synthetic_data_all)
            
            # Evaluate using Wasserstein distance
            ws_mean, ws_distances = calculate_wasserstein_distance(x_test, synthetic_data_all)
            
            # Compare and visualize
            compare_real_synthetic_sequences(
                x_test, synthetic_data_all, 
                scaler=scaler,
                output_dir=trial_dir
            )
            
            # Save model
            ttscgan.save_model(f"{trial_dir}/ttscgan")
            
            # Record trial results
            all_results.append({
                'percentage': percentage,
                'trial': trial + 1,
                'num_samples': num_samples,
                'dtw_mean': dtw_mean,
                'dtw_std': np.std(dtw_distances),
                'ws_mean': ws_mean,
                'ws_std': np.std(ws_distances),
                'd_loss': d_losses[-1],
                'g_loss': g_losses[-1]
            })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save full results
    results_df.to_csv(f"{output_dir}/experiment_results_full.csv", index=False)
    
    # Create aggregated results
    agg_results = results_df.groupby('percentage').agg({
        'num_samples': 'first',
        'dtw_mean': ['mean', 'std'],
        'ws_mean': ['mean', 'std'],
        'd_loss': ['mean', 'std'],
        'g_loss': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    agg_results.columns = ['_'.join(col).strip('_') for col in agg_results.columns.values]
    
    # Save aggregated results
    agg_results.to_csv(f"{output_dir}/experiment_results_aggregated.csv", index=False)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot DTW distance
    plt.subplot(2, 2, 1)
    plt.errorbar(
        agg_results['percentage'], 
        agg_results['dtw_mean_mean'], 
        yerr=agg_results['dtw_mean_std'],
        fmt='o-', capsize=5
    )
    plt.xlabel('Percentage of Real Data')
    plt.ylabel('DTW Distance')
    plt.title('DTW Distance vs. Real Data Percentage')
    plt.grid(True)
    
    # Plot Wasserstein distance
    plt.subplot(2, 2, 2)
    plt.errorbar(
        agg_results['percentage'], 
        agg_results['ws_mean_mean'], 
        yerr=agg_results['ws_mean_std'],
        fmt='o-', capsize=5
    )
    plt.xlabel('Percentage of Real Data')
    plt.ylabel('Wasserstein Distance')
    plt.title('Wasserstein Distance vs. Real Data Percentage')
    plt.grid(True)
    
    # Plot Generator and Discriminator losses
    plt.subplot(2, 2, 3)
    plt.errorbar(
        agg_results['percentage'], 
        agg_results['d_loss_mean'], 
        yerr=agg_results['d_loss_std'],
        fmt='o-', capsize=5, label='Discriminator Loss'
    )
    plt.xlabel('Percentage of Real Data')
    plt.ylabel('Loss')
    plt.title('Discriminator Loss vs. Real Data Percentage')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.errorbar(
        agg_results['percentage'], 
        agg_results['g_loss_mean'], 
        yerr=agg_results['g_loss_std'],
        fmt='o-', capsize=5, label='Generator Loss'
    )
    plt.xlabel('Percentage of Real Data')
    plt.ylabel('Loss')
    plt.title('Generator Loss vs. Real Data Percentage')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/minimal_data_requirements_results.png")
    plt.close()
    
    # Create thresholds recommendation plot
    plt.figure(figsize=(10, 6))
    
    # Normalize metrics to 0-1 range for comparison
    dtw_norm = (agg_results['dtw_mean_mean'] - agg_results['dtw_mean_mean'].min()) / \
               (agg_results['dtw_mean_mean'].max() - agg_results['dtw_mean_mean'].min())
    ws_norm = (agg_results['ws_mean_mean'] - agg_results['ws_mean_mean'].min()) / \
              (agg_results['ws_mean_mean'].max() - agg_results['ws_mean_mean'].min())
    
    # Invert so higher is better
    dtw_norm = 1 - dtw_norm
    ws_norm = 1 - ws_norm
    
    # Average the normalized metrics
    combined_metric = (dtw_norm + ws_norm) / 2
    
    plt.plot(agg_results['percentage'], combined_metric, 'o-', linewidth=2)
    plt.axhline(y=0.8, color='r', linestyle='--', label='Quality Threshold (80%)')
    
    # Find the first percentage where the combined metric exceeds 0.8
    threshold_idx = np.where(combined_metric >= 0.8)[0]
    if len(threshold_idx) > 0:
        threshold_percentage = agg_results['percentage'].iloc[threshold_idx[0]]
        plt.axvline(x=threshold_percentage, color='g', linestyle='--', 
                   label=f'Recommended Minimum ({threshold_percentage}%)')
    
    plt.xlabel('Percentage of Real Data')
    plt.ylabel('Normalized Quality Metric (Higher is Better)')
    plt.title('Determining Optimal Minimal Real Data Requirement')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/minimal_data_threshold_recommendation.png")
    plt.close()
    
    return results_df, agg_results

def run_comparative_analysis(
    dataset_path,
    sequence_length=100,
    num_classes=3,
    recommended_percentage=30,
    full_percentage=100,
    latent_dim=64,
    epochs=300,
    batch_size=32,
    output_dir='comparative_analysis'
):
    """
    Run a comparative analysis between minimal and full data for model training
    
    Parameters:
    -----------
    dataset_path : str
        Path to the dataset CSV file
    sequence_length : int
        Length of sequences to generate
    num_classes : int
        Number of classes in the dataset
    recommended_percentage : int
        Recommended minimal percentage of real data
    full_percentage : int
        Full percentage of real data (usually 100%)
    latent_dim : int
        Dimension of latent space for generator
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    output_dir : str
        Directory to save results
    
    Returns:
    --------
    dict
        Results of the comparative analysis
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    sequences, labels, scaler = load_and_preprocess_data(dataset_path, sequence_length, num_classes)
    
    # Reserve a test set (20% of the data)
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        sequences, labels, test_size=0.2, random_state=42
    )
    
    # Define models to train
    models_to_train = [
        {'name': f'minimal_{recommended_percentage}_percent', 'percentage': recommended_percentage},
        {'name': 'full_data', 'percentage': full_percentage}
    ]
    
    comparison_results = {}
    
    # Train each model
    for model_config in models_to_train:
        model_name = model_config['name']
        percentage = model_config['percentage']
        
        print(f"\n{'='*50}")
        print(f"Training {model_name} model with {percentage}% of data")
        print(f"{'='*50}")
        
        model_dir = f"{output_dir}/{model_name}"
        os.makedirs(model_dir, exist_ok=True)
        
        # Calculate sample size
        if percentage == 100:
            x_train = x_train_full
            y_train = y_train_full
        else:
            num_samples = int(len(x_train_full) * percentage / 100)
            indices = np.random.choice(len(x_train_full), num_samples, replace=False)
            x_train = x_train_full[indices]
            y_train = y_train_full[indices]
        
        print(f"Training with {len(x_train)} real samples")
        
        # Initialize model
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
            batch_size=batch_size,
            save_interval=100,
            output_dir=model_dir
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
        
        # Evaluate using DTW
        dtw_mean, dtw_distances = evaluate_with_dtw(x_test, synthetic_data_all, num_samples=20)
        
        # Evaluate using Wasserstein distance
        ws_mean, ws_distances = calculate_wasserstein_distance(x_test, synthetic_data_all, num_samples=20)
        
        # Compare and visualize
        compare_real_synthetic_sequences(
            x_test, synthetic_data_all, 
            scaler=scaler,
            num_samples=5,
            output_dir=model_dir
        )
        
        # Save model
        ttscgan.save_model(f"{model_dir}/ttscgan")
        
        # Record results
        comparison_results[model_name] = {
            'percentage': percentage,
            'num_samples': len(x_train),
            'dtw_mean': dtw_mean,
            'dtw_std': np.std(dtw_distances),
            'ws_mean': ws_mean,
            'ws_std': np.std(ws_distances),
            'final_d_loss': d_losses[-1],
            'final_g_loss': g_losses[-1],
            'd_losses': d_losses,
            'g_losses': g_losses
        }
    
    # Create comparison visualization
    plt.figure(figsize=(15, 10))
    
    # Training loss comparison
    plt.subplot(2, 2, 1)
    for model_name, results in comparison_results.items():
        plt.plot(results['d_losses'], label=f"{model_name} - D Loss")
    plt.xlabel('Iteration')
    plt.ylabel('Discriminator Loss')
    plt.title('Discriminator Loss During Training')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    for model_name, results in comparison_results.items():
        plt.plot(results['g_losses'], label=f"{model_name} - G Loss")
    plt.xlabel('Iteration')
    plt.ylabel('Generator Loss')
    plt.title('Generator Loss During Training')
    plt.legend()
    plt.grid(True)
    
    # Metrics comparison
    model_names = list(comparison_results.keys())
    dtw_means = [comparison_results[model]['dtw_mean'] for model in model_names]
    dtw_stds = [comparison_results[model]['dtw_std'] for model in model_names]
    ws_means = [comparison_results[model]['ws_mean'] for model in model_names]
    ws_stds = [comparison_results[model]['ws_std'] for model in model_names]
    
    plt.subplot(2, 2, 3)
    plt.bar(model_names, dtw_means, yerr=dtw_stds, capsize=5)
    plt.ylabel('DTW Distance')
    plt.title('DTW Distance Comparison')
    plt.grid(True, axis='y')
    
    plt.subplot(2, 2, 4)
    plt.bar(model_names, ws_means, yerr=ws_stds, capsize=5)
    plt.ylabel('Wasserstein Distance')
    plt.title('Wasserstein Distance Comparison')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparative_analysis_results.png")
    plt.close()
    
    # Save detailed results
    results_df = pd.DataFrame([{
        'model_name': model_name,
        'percentage': results['percentage'],
        'num_samples': results['num_samples'],
        'dtw_mean': results['dtw_mean'],
        'dtw_std': results['dtw_std'],
        'ws_mean': results['ws_mean'],
        'ws_std': results['ws_std'],
        'final_d_loss': results['final_d_loss'],
        'final_g_loss': results['final_g_loss']
    } for model_name, results in comparison_results.items()])
    
    results_df.to_csv(f"{output_dir}/comparative_analysis_results.csv", index=False)
    
    # Calculate improvement ratio
    minimal_model = f'minimal_{recommended_percentage}_percent'
    full_model = 'full_data'
    
    dtw_ratio = comparison_results[minimal_model]['dtw_mean'] / comparison_results[full_model]['dtw_mean']
    ws_ratio = comparison_results[minimal_model]['ws_mean'] / comparison_results[full_model]['ws_mean']
    
    # Calculate data efficiency
    data_efficiency = (full_percentage - recommended_percentage) / full_percentage * 100
    
    efficiency_results = {
        'recommended_percentage': recommended_percentage,
        'data_reduction': data_efficiency,
        'dtw_ratio': dtw_ratio,
        'ws_ratio': ws_ratio,
        'summary': f"Using {recommended_percentage}% of data achieves {dtw_ratio:.2f}x the DTW distance "
                   f"and {ws_ratio:.2f}x the Wasserstein distance compared to using 100% of data, "
                   f"resulting in a {data_efficiency:.1f}% reduction in data requirements."
    }
    
    # Save efficiency results
    pd.DataFrame([efficiency_results]).to_csv(f"{output_dir}/efficiency_results.csv", index=False)
    
    print("\nComparative Analysis Results:")
    print(f"Using {recommended_percentage}% of real data:")
    print(f"  - Reduces data requirements by {data_efficiency:.1f}%")
    print(f"  - DTW distance ratio: {dtw_ratio:.2f}x")
    print(f"  - Wasserstein distance ratio: {ws_ratio:.2f}x")
    
    return comparison_results, efficiency_results

if __name__ == "__main__":
    # Run minimal data requirements experiment
    print("Running minimal data requirements experiment...")
    results_df, agg_results = run_minimal_data_experiment(
        dataset_path="left_eye_openness.csv",
        min_percentages=[5, 10, 20, 30, 50, 70],
        sequence_length=100,
        num_classes=3,
        latent_dim=64,
        epochs=200,
        batch_size=32,
        output_dir="experiment_results",
        num_trials=3
    )
    
    # Determine recommended minimum percentage
    # Using a simple approach: find where normalized quality metric exceeds 0.8
    dtw_norm = 1 - (agg_results['dtw_mean_mean'] - agg_results['dtw_mean_mean'].min()) / \
               (agg_results['dtw_mean_mean'].max() - agg_results['dtw_mean_mean'].min())
    ws_norm = 1 - (agg_results['ws_mean_mean'] - agg_results['ws_mean_mean'].min()) / \
              (agg_results['ws_mean_mean'].max() - agg_results['ws_mean_mean'].min())
    combined_metric = (dtw_norm + ws_norm) / 2
    
    threshold_idx = np.where(combined_metric >= 0.8)[0]
    if len(threshold_idx) > 0:
        recommended_percentage = int(agg_results['percentage'].iloc[threshold_idx[0]])
    else:
        # Default to a moderate percentage if no clear threshold is found
        recommended_percentage = 30
    
    print(f"\nRecommended minimum real data percentage: {recommended_percentage}%")
    
    # Run comparative analysis
    print("\nRunning comparative analysis between minimal and full data...")
    comparison_results, efficiency_results = run_comparative_analysis(
        dataset_path="left_eye_openness.csv",
        sequence_length=100,
        num_classes=3,
        recommended_percentage=recommended_percentage,
        full_percentage=100,
        latent_dim=64,
        epochs=300,
        batch_size=32,
        output_dir="comparative_analysis"
    )
    
    print("\nExperiments completed successfully!")