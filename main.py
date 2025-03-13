import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import argparse
from datetime import datetime

# Import modules
from ttscgan_implementation import (
    TTSCGAN, load_and_preprocess_data, 
    evaluate_with_dtw, calculate_wasserstein_distance,
    compare_real_synthetic_sequences
)
from experiment_runner import run_minimal_data_experiment, run_comparative_analysis
from dtw_evaluator import DTWEvaluator

SEQUENCE_LENGTH = 50

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='TTS-CGAN for Eye Openness Data with Minimal Data Requirements Analysis')
    parser.add_argument('--data_file', type=str, default='left_eye_openness.csv', help='Path to the eye openness CSV file')
    parser.add_argument('--output_dir', type=str, default=f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}', help='Output directory')
    parser.add_argument('--sequence_length', type=int, default=100, help='Length of sequences to generate')
    parser.add_argument('--latent_dim', type=int, default=64, help='Dimension of latent space')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes in the dataset')
    parser.add_argument('--min_percentages', type=str, default='5,10,20,30,50,70', help='Comma-separated list of percentages to test')
    parser.add_argument('--num_trials', type=int, default=3, help='Number of trials for each percentage')
    parser.add_argument('--mode', type=str, choices=['full', 'minimal', 'comparative'], default='full', 
                        help='Experiment mode: full (all experiments), minimal (just minimal data analysis), or comparative (just comparative analysis)')
    
    args = parser.parse_args()
    
    # Validate num_classes
    if args.num_classes < 1:
        raise ValueError("num_classes must be at least 1")

    # Ensure sequence_length is consistent
    SEQUENCE_LENGTH = args.sequence_length
    
    # Create output directory and evaluation directory
    os.makedirs(args.output_dir, exist_ok=True)
    eval_dir = os.path.join(args.output_dir, 'dtw_evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    
    # Initialize DTW evaluator
    dtw_evaluator = DTWEvaluator(output_dir=eval_dir)
    
    # Save arguments
    with open(f"{args.output_dir}/experiment_config.txt", 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # Parse min_percentages
    min_percentages = [int(p) for p in args.min_percentages.split(',')]
    
    print(f"Starting experiments with mode: {args.mode}")
    print(f"Output directory: {args.output_dir}")
    
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Stage 1: Basic TTS-CGAN Training (for full or comparative modes)
    if args.mode in ['full', 'comparative']:
        print("\n{0} STAGE 1: Basic TTS-CGAN Training {0}".format('='*20))
        
        # Create directory for basic results
        basic_dir = f"{args.output_dir}/basic_training"
        os.makedirs(basic_dir, exist_ok=True)
        
        # Load and preprocess data
        sequences, labels, scaler = load_and_preprocess_data(
            "left_eye_openness.csv", 
            sequence_length=SEQUENCE_LENGTH,
            step_size=10
        )
        
        # Split into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(
            sequences, labels, test_size=0.2, random_state=42
        )
        
        print(f"Data shape: {sequences.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Number of training samples: {x_train.shape[0]}")
        print(f"Number of testing samples: {x_test.shape[0]}")
        
        # Initialize and train TTS-CGAN model
        print("Building and training TTS-CGAN model...")
        num_features = sequences.shape[2]
        num_classes = args.num_classes
        
        ttscgan = TTSCGAN(
            sequence_length=SEQUENCE_LENGTH,
            num_features=num_features,
            num_classes=num_classes,
            latent_dim=args.latent_dim
        )
        
        # Train the model
        try:
            d_losses, g_losses = ttscgan.train(
                x_train=x_train, 
                y_train=y_train,
                epochs=args.epochs,
                batch_size=args.batch_size,
                save_interval=50,
                output_dir=basic_dir
            )
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return

        # Generate synthetic data
        print("\nGenerating synthetic data for evaluation...")
        print(f"x_test shape: {x_test.shape}")
        
        all_synthetic = []
        samples_per_class = len(x_test) // num_classes
        
        for class_idx in range(num_classes):
            print(f"\nGenerating class {class_idx}")
            print(f"x_test shape: {x_test.shape}")
            print(f"Generating {samples_per_class} samples")
            
            synthetic = ttscgan.generate_synthetic_sequences(
                num_sequences=samples_per_class,
                class_idx=class_idx,
                scaler=scaler
            )
            print(f"Generated shape: {synthetic.shape}")
            
            # Ensure correct sequence length
            if synthetic.shape[1] != SEQUENCE_LENGTH:
                print(f"Reshaping synthetic data to match sequence length {SEQUENCE_LENGTH}")
                # Either truncate or pad the sequences
                if synthetic.shape[1] > SEQUENCE_LENGTH:
                    synthetic = synthetic[:, :SEQUENCE_LENGTH, :]
                else:
                    pad_width = ((0, 0), (0, SEQUENCE_LENGTH - synthetic.shape[1]), (0, 0))
                    synthetic = np.pad(synthetic, pad_width, mode='constant')
                print(f"After reshaping: {synthetic.shape}")
            
            all_synthetic.append(synthetic)
        
        # Stack and verify shapes
        all_synthetic = np.vstack(all_synthetic)
        print(f"\nFinal shapes:")
        print(f"x_test: {x_test.shape}")
        print(f"all_synthetic: {all_synthetic.shape}")
        
        # Verify shapes match
        assert all_synthetic.shape[1:] == x_test.shape[1:], \
            f"Shape mismatch: {all_synthetic.shape} vs {x_test.shape}"

        # Now evaluate
        try:
            print("\nEvaluating synthetic data quality...")
            dtw_results = dtw_evaluator.evaluate_synthetic_data(x_test, all_synthetic)
            print("\nDTW Evaluation Results:")
            for metric, value in dtw_results.items():
                print(f"{metric}: {value}")
        except Exception as e:
            print(f"Error during DTW evaluation: {str(e)}")
            dtw_results = None
        
        # Create synthetic labels for class-specific evaluation
        synthetic_labels = np.zeros((len(all_synthetic), args.num_classes))
        samples_per_class = len(all_synthetic) // args.num_classes
        
        for class_idx in range(args.num_classes):
            start_idx = class_idx * samples_per_class
            end_idx = (class_idx + 1) * samples_per_class
            synthetic_labels[start_idx:end_idx, class_idx] = 1
        
        # Evaluate per class
        dtw_evaluator.evaluate_class_specific(
            x_test, all_synthetic, y_test, synthetic_labels, args.num_classes
        )
        
        # Save the model
        ttscgan.save_model(f"{basic_dir}/ttscgan_model")
        
        print("Basic TTS-CGAN training and evaluation complete!")
    
    # Stage 2: Minimal Data Requirements Analysis
    if args.mode in ['full', 'minimal']:
        print("\n{0} STAGE 2: Minimal Data Requirements Analysis {0}".format('='*20))
        
        # Create directory for minimal data analysis
        minimal_dir = f"{args.output_dir}/minimal_data_analysis"
        os.makedirs(minimal_dir, exist_ok=True)
        
        # Run minimal data requirements experiment
        results_df, agg_results = run_minimal_data_experiment(
            dataset_path=args.data_file,
            min_percentages=min_percentages,
            sequence_length=args.sequence_length,
            num_classes=args.num_classes,
            latent_dim=args.latent_dim,
            epochs=200,  # Use fewer epochs for this analysis
            batch_size=args.batch_size,
            output_dir=minimal_dir,
            num_trials=args.num_trials
        )
        
        # Determine recommended minimum percentage
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
        
        # Save recommendation
        with open(f"{minimal_dir}/recommendation.txt", 'w') as f:
            f.write(f"Recommended minimum real data percentage: {recommended_percentage}%\n")
            f.write(f"This recommendation is based on achieving at least 80% of the maximum quality metric.\n")
            f.write(f"DTW normalization: {dtw_norm.tolist()}\n")
            f.write(f"Wasserstein normalization: {ws_norm.tolist()}\n")
            f.write(f"Combined metric: {combined_metric.tolist()}\n")
        
        print("Minimal data requirements analysis complete!")
    
    # Stage 3: Comparative Analysis
    if args.mode in ['full', 'comparative']:
        print("\n{0} STAGE 3: Comparative Analysis {0}".format('='*20))
        
        # Create directory for comparative analysis
        comparative_dir = f"{args.output_dir}/comparative_analysis"
        os.makedirs(comparative_dir, exist_ok=True)
        
        # Determine recommended percentage if not already done
        if args.mode == 'comparative':
            # Use a default value
            recommended_percentage = 30
        
        # Run comparative analysis
        comparison_results, efficiency_results = run_comparative_analysis(
            dataset_path=args.data_file,
            sequence_length=args.sequence_length,
            num_classes=args.num_classes,
            recommended_percentage=recommended_percentage,
            full_percentage=100,
            latent_dim=args.latent_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=comparative_dir
        )
        
        print("\nComparative Analysis Results:")
        print(efficiency_results['summary'])
        
        print("Comparative analysis complete!")
    
    print("\nAll experiments completed successfully!")

if __name__ == "__main__":
    main()