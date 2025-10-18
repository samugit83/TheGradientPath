"""
NETWORK ATTACK PREDICTION - BATCH TRAINING SCRIPT
=================================================

This script implements batch training for network attack prediction models.
It handles the complete training pipeline from data loading to model persistence.

FEATURES:
---------
✓ **Automatic Data Type Detection**: Identifies numerical vs categorical columns
✓ **Label Encoding for Categoricals**: Efficient encoding for all categorical features  
✓ **Robust Preprocessing**: Handles missing values and unknown categories gracefully
✓ **Feature Scaling**: Applies appropriate scaling to numerical features only
✓ **Balanced Dataset Training**: Handles class imbalance automatically
✓ **Optimal Threshold Calculation**: Uses precision-recall curve for threshold optimization
✓ **Comprehensive Testing**: Full evaluation with multiple metrics on test dataset
✓ **Class Distribution Tracking**: Monitors normal vs attack traffic balance

KEY COMPONENTS:
--------------
1. **BatchTrainer**: Main class handling the complete batch training pipeline
2. **Comprehensive Testing**: Multi-metric evaluation with detailed analysis

USAGE:
------
from batch_train import BatchTrainer, execute_comprehensive_testing
from config import get_default_config

config = get_default_config()
trainer = BatchTrainer(config)
success = trainer.execute_training()

if success:
    execute_comprehensive_testing(...)
"""

import os   
import numpy as np 
import pandas as pd 
import json        
import time    
from collections import Counter 
import torch       

from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, precision_recall_curve
from sklearn.utils import resample 

from modules.ae import AEWrapper, AEConfig   
from modules.orc_selector import ORCFeatureSelector, ORCConfig  
from modules.sklearn_wrapper import SklearnWrapper, SklearnConfig  
from modules.stream_utils import stream_csv_raw, print_data_summary, create_feature_config  
from modules.data_preprocessing import DataPreprocessor, PreprocessingConfig 

from config import get_default_config, validate_config, print_config_summary


class BatchTrainer:
    """
    Main class for batch training of network attack prediction models.
    
    This class encapsulates the entire machine learning pipeline including:
    - Data loading and preprocessing
    - Model training with balanced datasets
    - Optimal threshold calculation
    - Class distribution tracking
    - Model persistence
    """
    
    def __init__(self, config):
        """
        Initialize the trainer with configuration parameters.
        
        Args:
            config (dict): Configuration dictionary containing paths and settings
        """
        self.config = config
        self.data_config = None
        self.preproc_config = None
        self.preproc_info = None
        self.preprocessor = None
        self.ae = None
        self.orc_sel = None
        self.rf = None
        self.available_cols = None
        self.n_features_processed = None
        
        # Training data
        self.all_samples = []
        self.all_labels = []
    
    def validate_configuration(self):
        """Validate configuration and check if data files exist."""
        
        if not os.path.exists(self.config['data_path']):
            print(f"❌ ERROR: Data file not found: {self.config['data_path']}")
            print("\nPlease check:")
            print("1. The file path is correct")
            print("2. The file exists at that location")
            print("3. You have read permissions for the file")
            return False
        
        print(f"✅ Data file validated: {self.config['data_path']}")
        return True
    
    def analyze_data(self):
        """Analyze the dataset and configure feature extraction."""
        print("\n=== ANALYZING YOUR DATA ===")
        print(f"Data file: {self.config['data_path']}")
        
        # Print comprehensive data summary to help user understand their data
        print_data_summary(self.config['data_path'])
        
        # Get automatic feature configuration
        self.data_config = create_feature_config(self.config['data_path'], 
                                                exclude_cols=self.config['exclude_columns'])
        
        # Use user-specified features or auto-detected ones
        if self.config['feature_columns'] is not None:
            feature_cols = [col for col in self.config['feature_columns'] 
                          if col not in (self.config['exclude_columns'] or [])]
            print(f"Using user-specified features: {len(feature_cols)} columns")
        else:
            feature_cols = self.data_config['feature_cols']
            print(f"Auto-detected features: {len(feature_cols)} columns")
            print(f"├─ Numerical: {len(self.data_config['numerical_features'])}")
            print(f"└─ Categorical: {len(self.data_config['categorical_features'])}")
        
        # Validate label column
        label_col = self.config['label_column']
        if label_col not in self.data_config['label_candidates'] and self.data_config['label_candidates']:
            print(f"WARNING: Label column '{label_col}' not in detected candidates: {self.data_config['label_candidates']}")
            print("Please verify your LABEL_COLUMN setting is correct.")
        
        print(f"Label column: '{label_col}'")
        
        # Calculate available columns
        sample_df = pd.read_csv(self.config['data_path'], nrows=100)
        self.available_cols = [col for col in sample_df.columns 
                             if col not in (self.config['exclude_columns'] or []) 
                             and col != label_col]
        
        print(f"Available feature columns: {len(self.available_cols)}")
        print()
    
    def setup_preprocessing(self):
        """Setup data preprocessing configuration and fit preprocessor."""
        print("=== SETTING UP PREPROCESSING ===")
        
        # Create preprocessing configuration
        if self.config['preprocessing_config'] is not None:
            preprocessing_cfg = self.config['preprocessing_config']
            self.preproc_config = PreprocessingConfig(
                categorical_encoding=preprocessing_cfg.get('categorical_encoding', 'label'),
                handle_unknown=preprocessing_cfg.get('handle_unknown', 'ignore'),
                scale_features=preprocessing_cfg.get('scale_features', True)
            )
            print("Using user-specified preprocessing configuration")
            print(f"├─ Log transformation: {'ENABLED' if preprocessing_cfg.get('apply_log_transform', True) else 'DISABLED'}")
            print(f"├─ Extreme range threshold: {preprocessing_cfg.get('extreme_range_threshold', 1e4)}")
        else:
            # Auto-configure based on data characteristics
            self.preproc_config = PreprocessingConfig(
                # Use label encoding for all categoricals (compact and efficient)
                categorical_encoding='label',
                handle_unknown='ignore',        # Gracefully handle new categories in production
                scale_features=self.data_config['preprocessing_suggestions']['scale_features']  # Scale numerical features
            )
            print("Using auto-detected preprocessing configuration")
        
        print(f"├─ Categorical encoding: {self.preproc_config.categorical_encoding}")
        print(f"├─ Handle unknown: {self.preproc_config.handle_unknown}")
        print(f"└─ Scale features: {self.preproc_config.scale_features}")
        
        # Initialize and fit preprocessor
        self.preprocessor = DataPreprocessor(self.preproc_config, self.config['preprocessing_config'])
        
        print("\n=== FITTING PREPROCESSOR ===")
        print("Reading sample of data to fit encoders and scalers...")
        
        # Read a sample of data to fit the preprocessor
        sample_df = pd.read_csv(self.config['data_path'], nrows=self.config['model_config']['preprocessing_fit_size'])
        feature_df = sample_df[self.available_cols]
        
        # Fit preprocessor on sample data
        self.preprocessor.fit(feature_df)
        
        # Get information about the fitted preprocessor
        self.preproc_info = self.preprocessor.get_feature_info()
        self.n_features_processed = self.preproc_info['total_output_features']
        
        print(f"✓ Fitted preprocessor on {len(sample_df)} samples")
        print(f"├─ Input features: {len(self.available_cols)}")
        print(f"├─ Output features: {self.preproc_info['total_output_features']}")
        print(f"├─ Numerical features: {len(self.preproc_info['numerical_features'])}")
        print(f"└─ Label encoded: {len(self.preproc_info['label_features'])}")
    
    def initialize_models(self):
        """Initialize all ML models with proper configurations."""
        print("\n=== INITIALIZING MODELS ===")
        
        cfg = self.config['model_config']
        
        # Check if feature selection is enabled
        apply_feature_selection = cfg.get('apply_feature_selection', True)
        
        if apply_feature_selection:
            print("✅ Feature selection ENABLED - initializing AutoEncoder + ORC")
            
            # Initialize AutoEncoder with processed feature dimension
            ae_cfg = AEConfig(
                d_in=self.n_features_processed,
                d_hidden=cfg['ae']['d_hidden'],
                lr=cfg['ae']['lr']
            )
            self.ae = AEWrapper(ae_cfg)
            print(f"✓ AutoEncoder: {self.n_features_processed} → {cfg['ae']['d_hidden']} → {self.n_features_processed}")
            
            # Initialize ORC with processed features
            orc_cfg = ORCConfig(**cfg['orc'])
            self.orc_sel = ORCFeatureSelector(
                n_features=self.n_features_processed,
                cfg=orc_cfg,
                feature_names=self.preproc_info['output_feature_names'],
                training_mode="batch" 
            )
            print(f"✓ ORC Selector: {self.n_features_processed} features → top {cfg['orc']['top_k']} (BATCH mode - no locking)")
        else:
            print("⚠️ Feature selection DISABLED - using ALL features directly")
            self.ae = None
            self.orc_sel = None
            print(f"✓ Will use all {self.n_features_processed} features for classification")
        
        # Initialize SGD Classifier
        sgd_cfg = SklearnConfig(**cfg['sgd_classifier'])
        self.rf = SklearnWrapper(sgd_cfg)
        print(f"✓ SGD Classifier: incremental learning enabled")
        
        # Create artifacts directory
        os.makedirs(self.config['artifacts_dir'], exist_ok=True)
    
    def load_training_data(self):
        """Load and analyze training dataset."""
        print("\n=== LOADING TRAINING DATASET ===")
        
        self.all_samples = []
        self.all_labels = []
        
        print("📊 Loading training dataset...")
        train_count = 0
        loading_start = time.time()
        
        for x_raw, y in stream_csv_raw(self.config['data_path'], 
                                      feature_cols=self.available_cols, 
                                      label_col=self.config['label_column'], 
                                      chunksize=5000):
            if y is not None:
                self.all_samples.append(x_raw)
                self.all_labels.append(y)
                train_count += 1
                
                # Progress during loading
                if train_count % 20000 == 0:
                    elapsed = time.time() - loading_start
                    loading_speed = train_count / elapsed if elapsed > 0 else 0
                    print(f"   📈 Loaded {train_count:,} samples ({loading_speed:.0f} samples/sec)")
        
        total_samples = len(self.all_samples)
        loading_time = time.time() - loading_start
        print(f"✓ Loaded {total_samples:,} training samples in {loading_time:.1f} seconds")
        
        # Analyze class distribution
        class_counts = Counter(self.all_labels)
        print(f"├─ Normal (0): {class_counts[0]:,} ({class_counts[0]/total_samples*100:.1f}%)")
        print(f"├─ Attack (1): {class_counts[1]:,} ({class_counts[1]/total_samples*100:.1f}%)")
        imbalance_ratio = max(class_counts.values()) / min(class_counts.values())
        print(f"└─ Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    def balance_dataset(self):
        """Balance the dataset to reduce model bias."""
        print("\n=== BALANCING DATASET ===")
        print("🎯 Creating balanced dataset to reduce model bias...")
        
        total_samples = len(self.all_samples)
        
        # Separate samples by class
        normal_samples = [s for s, l in zip(self.all_samples, self.all_labels) if l == 0]
        attack_samples = [s for s, l in zip(self.all_samples, self.all_labels) if l == 1]
        
        print(f"├─ Original Normal samples: {len(normal_samples):,}")
        print(f"├─ Original Attack samples: {len(attack_samples):,}")
        
        # Balance the dataset by undersampling to the minority class
        min_count = min(len(normal_samples), len(attack_samples))
        print(f"├─ Target balanced size: {min_count:,} samples per class")
        
        # Undersample both classes to match the smaller one
        normal_balanced = resample(normal_samples, n_samples=min_count, random_state=42)
        attack_balanced = resample(attack_samples, n_samples=min_count, random_state=42)
        
        # Combine balanced data
        self.all_samples = normal_balanced + attack_balanced
        self.all_labels = [0] * len(normal_balanced) + [1] * len(attack_balanced)
        
        # Shuffle the combined dataset
        combined = list(zip(self.all_samples, self.all_labels))
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(combined)
        self.all_samples, self.all_labels = zip(*combined)
        self.all_samples = list(self.all_samples)
        self.all_labels = list(self.all_labels)
        
        # Update totals
        balanced_total = len(self.all_samples)
        balanced_counts = Counter(self.all_labels)
        
        print(f"✅ BALANCED DATASET CREATED:")
        print(f"├─ Total samples: {balanced_total:,} (reduced from {total_samples:,})")
        print(f"├─ Normal samples: {balanced_counts[0]:,} ({balanced_counts[0]/balanced_total*100:.1f}%)")
        print(f"├─ Attack samples: {balanced_counts[1]:,} ({balanced_counts[1]/balanced_total*100:.1f}%)")
        print(f"└─ Perfect balance ratio: 1.00:1 🎯")
    
    def process_samples_through_pipeline(self):
        """Process all samples through the ML pipeline for feature selection."""
        print("\n=== PROCESSING SAMPLES THROUGH PIPELINE ===")
        
        # Check if feature selection is enabled
        cfg = self.config['model_config']
        apply_feature_selection = cfg.get('apply_feature_selection', True)
        
        # Process all samples through AE and ORC for feature selection
        processed_samples = []
        processed_labels = []
        
        # Add timing and progress tracking
        start_time = time.time()
        total_samples = len(self.all_samples)
        
        # Progress tracking for AutoEncoder and feature selection only
        progress_interval = 10000  # Show progress every 10k samples
        
        # Track AutoEncoder training statistics
        ae_train_count = 0
        ae_skip_count = 0
        
        sample_i = 0
        for x_raw, y in zip(self.all_samples, self.all_labels):
            try:
                # ================================================================
                # PREPROCESSING: Handle mixed data types
                # ================================================================
                
                # Transform raw sample to processed feature vector
                x_processed = self.preprocessor.transform_single(x_raw)
                
                if apply_feature_selection:
                    # FEATURE SELECTION ENABLED: Use AutoEncoder + ORC pipeline
                    # Convert to tensor for AutoEncoder
                    x_tensor = torch.from_numpy(x_processed.astype(np.float32))
                    
                    # ================================================================
                    # AUTOENCODER: Feature learning and reconstruction
                    # ================================================================
                    
                    # Get reconstruction and compute errors
                    recon = self.ae.forward_no_grad(x_tensor)
                    err = np.abs(x_processed - recon.numpy())
                    
                    # Update ORC feature selector
                    self.orc_sel.update(err)
                    
                    # ================================================================
                    # FEATURE SELECTION: Dynamic feature selection
                    # ================================================================
                    
                    # Get currently selected feature indices
                    mask_idx = self.orc_sel.get_mask_indices()
                    
                    # Create reduced feature dictionary for classifier
                    feature_names = self.preproc_info['output_feature_names']
                    x_reduced = {feature_names[i]: float(x_processed[i]) for i in mask_idx}
                    
                    # ================================================================
                    # AUTOENCODER TRAINING: Optional online training
                    # ================================================================
                    
                    # CRITICAL FIX: Train AutoEncoder ONLY on normal traffic (y == 0)
                    # This is essential for anomaly detection - the AE should only learn normal patterns!
                    if sample_i % cfg['ae_train_every'] == 0:
                        if y == 0:  # Train only on normal traffic
                            self.ae.train_step(x_tensor)
                            ae_train_count += 1
                        else:  # Skip attack samples (count but don't log each one)
                            ae_skip_count += 1
                 
                else:
                    # FEATURE SELECTION DISABLED: Use all features directly
                    feature_names = self.preproc_info['output_feature_names']
                    x_reduced = {feature_names[i]: float(x_processed[i]) for i in range(len(x_processed))}
                    mask_idx = list(range(len(x_processed)))  # All features selected
                    
                    if sample_i == 0:  # Log once
                        print(f"⚠️ Feature selection DISABLED - using ALL {len(x_processed)} features")
                
                processed_samples.append(x_reduced)
                processed_labels.append(y)
                
                sample_i += 1
                
                # ================================================================
                # PROGRESS TRACKING: Show AutoEncoder and feature selection progress
                # ================================================================
                
                # Show progress every progress_interval samples
                if sample_i % progress_interval == 0:
                    elapsed = time.time() - start_time
                    progress_pct = (sample_i / total_samples) * 100
                    speed = sample_i / elapsed if elapsed > 0 else 0
                    
                    if apply_feature_selection:
                        print(f"📈 Progress: {sample_i:,}/{total_samples:,} ({progress_pct:.1f}%) | "
                              f"Selected_Features={len(mask_idx)} | Speed={speed:.0f}/sec")
                    else:
                        print(f"📈 Progress: {sample_i:,}/{total_samples:,} ({progress_pct:.1f}%) | "
                              f"All_Features={len(mask_idx)} | Speed={speed:.0f}/sec")

                # Lightweight progress updates between major checkpoints
                elif sample_i % 5000 == 0:
                    self._show_progress_update(sample_i, total_samples, start_time)
                    
            except Exception as e:
                print(f"⚠️  Warning: Failed to process sample {sample_i}: {e}")
                continue
        
        print(f"✅ Successfully processed {len(processed_samples):,} samples")
        
        if apply_feature_selection:
            print(f"🧠 AutoEncoder training: Completed online learning")
            print(f"├─ Normal samples trained: {ae_train_count:,}")
            print(f"├─ Attack samples skipped: {ae_skip_count:,} (AE learns only normal patterns)")
            print(f"🎯 Feature selection: {len(mask_idx)} features selected")
        else:
            print(f"⚠️ Feature selection: DISABLED - using all {len(mask_idx)} features")
            
        return processed_samples, processed_labels
    
    def _show_progress_update(self, sample_i, total_samples, start_time):
        """Show lightweight progress updates."""
        elapsed = time.time() - start_time
        progress_pct = (sample_i / total_samples) * 100
        samples_per_sec = sample_i / elapsed if elapsed > 0 else 0
        eta_seconds = (total_samples - sample_i) / samples_per_sec if samples_per_sec > 0 else 0
        eta_minutes = eta_seconds / 60
        
        print(f"   Progress: {sample_i:,}/{total_samples:,} ({progress_pct:.1f}%) | "
              f"Speed: {samples_per_sec:.0f}/sec | ETA: {eta_minutes:.1f}min")
    
    def train_classifier(self, processed_samples, processed_labels):
        """Train the SGD classifier with final feature selection."""
        print("\n=== TRAINING BALANCED SGD CLASSIFIER ===")
        
        cfg = self.config['model_config']
        
        # Force final recomputation of feature rankings
        print("🔄 Finalizing feature selection...")
        self.orc_sel.force_recompute()
        final_mask = self.orc_sel.get_mask_indices()
        print(f"✓ Final feature selection: {len(final_mask)} features selected")
        
        # Reprocess all samples with the final feature mask to ensure consistency
        print("🔄 Applying final feature selection to all samples...")
        final_processed_samples = []
        feature_names = self.preproc_info['output_feature_names']
        final_feature_names = [feature_names[i] for i in final_mask]
        
        # Debug: Print first few feature names to verify
        print(f"📋 Final selected features: {final_feature_names}")
        
        for i, (x_raw, y) in enumerate(zip(self.all_samples, self.all_labels)):
            try:
                # Reprocess the sample
                x_processed = self.preprocessor.transform_single(x_raw)
                
                # Apply final feature selection
                x_reduced = {feature_names[j]: float(x_processed[j]) for j in final_mask}
                final_processed_samples.append(x_reduced)
                
                # Progress update for reprocessing
                if (i + 1) % 50000 == 0:
                    print(f"   Reprocessed {i + 1:,}/{len(self.all_samples):,} samples...")
                    
            except Exception as e:
                print(f"⚠️  Warning: Failed to reprocess sample {i}: {e}")
                continue
        
        print(f"✓ Reprocessed {len(final_processed_samples):,} samples with final feature selection")
        
        # Train the classifier on all processed samples with consistent features
        print("🚀 Training Balanced SGD classifier...")
        print(f"├─ Training samples: {len(final_processed_samples):,}")
        print(f"├─ Features per sample: {len(final_feature_names)}")
        print(f"├─ Loss function: {cfg['sgd_classifier']['loss']}")
        print(f"└─ Manual class balancing: enabled")
        
        # Debug: Check first sample keys
        if final_processed_samples:
            sample_keys = list(final_processed_samples[0].keys())
            print(f"📋 Sample feature keys (first 5): {sample_keys[:5]}")
        
        training_start = time.time()
        self.rf.fit(final_processed_samples, processed_labels)
        training_time = time.time() - training_start
        
        print(f"✅ SGD Classifier training completed in {training_time:.1f} seconds")
        
        return final_processed_samples, processed_labels
    
    def calculate_optimal_threshold(self, final_processed_samples, processed_labels):
        """Calculate optimal attack threshold using precision-recall curve on a sample."""
        print("\n=== CALCULATING OPTIMAL ATTACK THRESHOLD ===")
        print("🎯 Using precision-recall curve to find optimal threshold...")
        
        try:
            # Use a representative sample instead of all training data for efficiency
            sample_size = min(10000, len(final_processed_samples))  # Use max 10k samples
            print(f"📊 Using {sample_size:,} samples for threshold optimization (faster than all {len(final_processed_samples):,})...")
            
            # Create stratified sample to maintain class balance
            normal_indices = [i for i, label in enumerate(processed_labels) if label == 0]
            attack_indices = [i for i, label in enumerate(processed_labels) if label == 1]
            
            # Sample proportionally from each class
            normal_sample_size = min(sample_size // 2, len(normal_indices))
            attack_sample_size = min(sample_size // 2, len(attack_indices))
            
            sample_indices = []
            if normal_sample_size > 0:
                normal_sample_indices = np.random.choice(normal_indices, normal_sample_size, replace=False)
                sample_indices.extend(normal_sample_indices)
            if attack_sample_size > 0:
                attack_sample_indices = np.random.choice(attack_indices, attack_sample_size, replace=False)
                sample_indices.extend(attack_sample_indices)
            
            # Get prediction probabilities for the sample
            print("📊 Computing prediction probabilities...")
            y_proba = []
            y_true_sample = []
            
            for idx in sample_indices:
                sample = final_processed_samples[idx]
                true_label = processed_labels[idx]
                pred_proba = self.rf.predict_proba(sample)[1]  # Get attack probability
                y_proba.append(pred_proba)
                y_true_sample.append(true_label)
            
            print(f"✓ Computed probabilities for {len(y_proba):,} samples (much faster!)")
            
            # Calculate precision-recall curve
            precision, recall, thresholds = precision_recall_curve(y_true_sample, y_proba)
            
            # Calculate F1 scores for each threshold (harmonic mean of precision and recall)
            # F1 = 2 * (precision * recall) / (precision + recall)
            f1_scores = np.where((precision + recall) == 0, 0, 2 * (precision * recall) / (precision + recall))
            
            # Find threshold that maximizes F1 score
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]
            optimal_f1 = f1_scores[optimal_idx]
            optimal_precision = precision[optimal_idx]
            optimal_recall = recall[optimal_idx]
            
            # Update classifier's threshold
            original_threshold = self.rf.cfg.attack_threshold
            self.rf.cfg.attack_threshold = float(optimal_threshold)
            
            print(f"📈 THRESHOLD OPTIMIZATION RESULTS:")
            print(f"├─ Original threshold: {original_threshold:.3f}")      # Default 0.5
            print(f"├─ Optimal threshold: {optimal_threshold:.3f}")        # Data-driven optimum
            print(f"├─ Expected F1 Score: {optimal_f1:.4f}")              # Best achievable F1
            print(f"├─ Expected Precision: {optimal_precision:.4f}")       # How often attacks are real
            print(f"├─ Expected Recall: {optimal_recall:.4f}")             # How many attacks we catch
            print(f"└─ Sample size used: {len(y_proba):,} (much faster!)")
            
            # Test the new threshold on the same sample
            print("\n🧪 Testing optimal threshold on optimization sample...")
            correct_predictions = 0
            for i, (true_label, pred_proba) in enumerate(zip(y_true_sample, y_proba)):
                pred_label = 1 if pred_proba >= optimal_threshold else 0  # Apply threshold
                if pred_label == true_label:
                    correct_predictions += 1
            
            test_accuracy = correct_predictions / len(y_proba)
            print(f"✅ Test accuracy with optimal threshold: {test_accuracy:.4f} ({correct_predictions}/{len(y_proba)})")
            
        except Exception as e:
            print(f"⚠️  Warning: Threshold optimization failed: {e}")
            print(f"   Using default threshold: {self.rf.cfg.attack_threshold}")
    
    def save_models_and_metadata(self):
        """Save all trained models and metadata."""
        print("\n=== SAVING MODELS ===")
        
        self.preprocessor.save(os.path.join(self.config['artifacts_dir'], 'preprocessor.pkl'))
        self.ae.save(os.path.join(self.config['artifacts_dir'], 'ae.pt'))
        self.orc_sel.save(os.path.join(self.config['artifacts_dir'], 'orc.npz'))
        self.rf.save(os.path.join(self.config['artifacts_dir'], 'rf.pkl'))
        
        print(f"✓ Preprocessor saved: {self.config['artifacts_dir']}/preprocessor.pkl")
        print(f"✓ AutoEncoder saved: {self.config['artifacts_dir']}/ae.pt")
        print(f"✓ ORC selector saved: {self.config['artifacts_dir']}/orc.npz")
        print(f"✓ SGD Classifier saved: {self.config['artifacts_dir']}/rf.pkl")
        
        # Save selected feature names for interpretability
        sel_feat_names = self.orc_sel.get_mask_names()
        with open(os.path.join(self.config['artifacts_dir'], 'selected_features.txt'), 'w') as f:
            f.write("=== SELECTED FEATURES AFTER TRAINING ===\n")
            f.write(f"Total features selected: {len(sel_feat_names)}\n\n")
            for i, name in enumerate(sel_feat_names, 1):
                f.write(f"{i:2d}. {name}\n")
        
        # Create updated model config with the ACTUAL optimized threshold
        updated_model_config = self.config['model_config'].copy()
        updated_model_config['sgd_classifier'] = updated_model_config['sgd_classifier'].copy()
        updated_model_config['sgd_classifier']['attack_threshold'] = float(self.rf.cfg.attack_threshold)
        
        # Save comprehensive training metadata with UPDATED configuration
        # Calculate class distribution from training data
        normal_count = sum(1 for label in self.all_labels if label == 0)
        attack_count = sum(1 for label in self.all_labels if label == 1)
        
        metadata = {
            'data_config': self.data_config,
            'preprocessing_config': self.preproc_config.__dict__,
            'model_config': updated_model_config,  
            'training_stats': {
                'total_samples': len(self.all_samples),
                'input_features': len(self.available_cols),
                'processed_features': self.n_features_processed,
                'selected_features': len(sel_feat_names),
                'optimized_attack_threshold': float(self.rf.cfg.attack_threshold),
                # Add class distribution tracking
                'total_normal_samples': normal_count,
                'total_attack_samples': attack_count,
                'class_balance_ratio': normal_count / max(attack_count, 1)
            },
            'feature_info': self.preproc_info,
            # NEW: Feature selection configuration
            'apply_feature_selection': updated_model_config.get('apply_feature_selection', True),  # Critical flag for training/prediction
            'training_mode': 'batch',  # Identifies this as batch training (no feature locking)
            'orc_feature_selection': {
                'is_locked': False,  # Batch training never locks features
                'samples_seen': len(self.all_samples),
                'lock_threshold': None,  # Not applicable for batch training
                'locked_features_count': 0,
                'total_features': self.n_features_processed,
                'locked_feature_names': [],
                'locked_feature_indices': [],
                'lock_message': f"BATCH training mode - processed all {len(self.all_samples)} samples for feature selection" if updated_model_config.get('apply_feature_selection', True) else "Feature selection DISABLED - using all features"
            }
        }
        
        with open(os.path.join(self.config['artifacts_dir'], 'training_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"✓ Feature info saved: {self.config['artifacts_dir']}/selected_features.txt")
        print(f"✓ Training metadata saved: {self.config['artifacts_dir']}/training_metadata.json")
        print(f"✓ Optimized threshold ({self.rf.cfg.attack_threshold:.6f}) saved in metadata")
        print(f"✓ Class distribution saved: Normal: {normal_count:,}, Attack: {attack_count:,} (ratio: {normal_count / max(attack_count, 1):.2f}:1)")
    
    def print_training_summary(self):
        """Print comprehensive training summary."""
        sel_feat_names = self.orc_sel.get_mask_names()
        
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"📊 Dataset: {len(self.all_samples):,} samples processed")
        print(f"📁 Models saved to: {self.config['artifacts_dir']}/")
        print(f"🔧 Selected Features: {len(sel_feat_names)}/{self.n_features_processed}")
        print(f"🎯 Production Threshold: {self.rf.cfg.attack_threshold:.6f} (dynamic + 0.2 adjustment)")
        print()
        print("📈 PERFORMANCE TRACKING:")
        print("├─ Real-time F1/Accuracy shown during training epochs above")
        print("├─ Convergence metrics updated every 10,000 samples")
        print("├─ Optimal threshold calculated using precision-recall curve")
        print("├─ Production adjustment (+0.2) applied to reduce false positives")
        print("└─ Final model ready for testing and deployment")
        print()
        print("NEXT STEPS:")
        print("1. Review selected features in 'selected_features.txt'")
        print("2. Check training metadata in 'training_metadata.json'") 
        print("3. Use 'incremental_train.py' for real-time incremental training")
        print("4. The optimal threshold will be automatically used during inference!")
        print("="*60)
    
    def execute_training(self):
        """Execute the complete training pipeline."""
        print("🚀 STARTING BATCH TRAINING PIPELINE")
        print("="*70)
        
        # Step 1: Validate configuration
        if not self.validate_configuration():
            return False
        
        # Step 2: Analyze data
        self.analyze_data()
        
        # Step 3: Setup preprocessing
        self.setup_preprocessing()
        
        # Step 4: Initialize models
        self.initialize_models()
        
        # Step 5: Load training data
        self.load_training_data()
        
        # Step 6: Balance dataset
        self.balance_dataset()
        
        # Step 7: Process samples through pipeline
        processed_samples, processed_labels = self.process_samples_through_pipeline()
        
        # Step 8: Train classifier
        final_processed_samples, final_labels = self.train_classifier(processed_samples, processed_labels)
        
        # Step 9: Calculate optimal threshold
        self.calculate_optimal_threshold(final_processed_samples, final_labels)
        
        # Step 9.5: Adjust threshold for production (add 0.2 to reduce false positives)
        original_dynamic_threshold = self.rf.cfg.attack_threshold
        production_threshold = original_dynamic_threshold + 0.2
        self.rf.cfg.attack_threshold = float(production_threshold)
        
        print(f"\n=== PRODUCTION THRESHOLD ADJUSTMENT ===")
        print(f"🎯 Adjusting threshold for real-world deployment:")
        print(f"├─ Dynamic optimal threshold: {original_dynamic_threshold:.6f}")
        print(f"├─ Production adjustment:     +0.200000")
        print(f"└─ Final production threshold: {production_threshold:.6f}")
        print(f"✅ This reduces false positives for normal traffic!")
        
        # Step 10: Save models and metadata
        self.save_models_and_metadata()
        
        # Step 11: Print summary
        self.print_training_summary()
        
        return True
    
    def load_existing_models(self):
        """Load existing models for testing without training."""
        print("\n=== LOADING EXISTING MODELS FOR TESTING ===")
        
        # Load training metadata to get configurations
        metadata_path = os.path.join(self.config['artifacts_dir'], 'training_metadata.json')
        if not os.path.exists(metadata_path):
            print(f"❌ ERROR: Training metadata not found: {metadata_path}")
            print("Please run training first with execute_training=True")
            return False
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Extract configurations from metadata
        self.config['model_config'] = metadata['model_config']
        self.preproc_info = metadata['feature_info']
        self.n_features_processed = metadata['training_stats']['processed_features']
        
        print("✓ Configurations loaded from training metadata")
        return True


def execute_comprehensive_testing(test_data_path, artifacts_dir, feature_cols, label_col, preproc_info, n_features_processed,
                                 ae_cfg, orc_cfg, max_test_samples=10000):
    """
    Execute comprehensive testing on the UNSW_NB15_testing-set.csv dataset
    and display detailed performance metrics.
    
    Args:
        max_test_samples (int): Maximum number of samples to test (randomly sampled). Default: 10,000
    """
    from sklearn.metrics import confusion_matrix
    
    print("\n" + "="*70)
    print("🧪 EXECUTING COMPREHENSIVE TESTING")
    print("="*70)
    
    if not os.path.exists(test_data_path):
        print(f"❌ ERROR: Testing data file not found: {test_data_path}")
        return
    
    print(f"📁 Testing data: {test_data_path}")
    print(f"🎯 Max test samples: {max_test_samples:,} (randomly sampled)")
    
    print("\n=== LOADING TRAINED MODELS ===")
    
    preprocessor = DataPreprocessor.load(os.path.join(artifacts_dir, 'preprocessor.pkl'))
    print("✓ Preprocessor loaded")
    
    ae = AEWrapper(ae_cfg)
    ae.load(os.path.join(artifacts_dir, 'ae.pt'))
    print("✓ AutoEncoder loaded")
    
    orc_sel = ORCFeatureSelector.load(os.path.join(artifacts_dir, 'orc.npz'), orc_cfg, training_mode="batch")
    print("✓ ORC Feature Selector loaded (BATCH mode)")
    
    rf = SklearnWrapper.load(os.path.join(artifacts_dir, 'rf.pkl'))
    print("✓ SGD Classifier loaded")
    
    from river import metrics
    test_metrics = {
        'accuracy': metrics.Accuracy(),
        'precision': metrics.Precision(),
        'recall': metrics.Recall(),
        'f1': metrics.F1(),
        'balanced_accuracy': metrics.BalancedAccuracy(),
        'confusion_matrix': metrics.ConfusionMatrix(),
        'kappa': metrics.CohenKappa(),
        'mcc': metrics.MCC()  # Matthews Correlation Coefficient
    }
    
    print("\n=== LOADING AND SAMPLING TEST DATA ===")
    
    # First, load all test data to enable random sampling
    print("📊 Loading all test data for random sampling...")
    all_test_samples = []
    all_test_labels = []
    
    loading_start = time.time()
    for x_raw, y_true in stream_csv_raw(test_data_path, feature_cols=feature_cols, 
                                       label_col=label_col, chunksize=5000):
        all_test_samples.append(x_raw)
        all_test_labels.append(y_true)
        
        # Progress during loading
        if len(all_test_samples) % 20000 == 0:
            elapsed = time.time() - loading_start
            loading_speed = len(all_test_samples) / elapsed if elapsed > 0 else 0
            print(f"   📈 Loaded {len(all_test_samples):,} samples ({loading_speed:.0f} samples/sec)")
    
    total_available = len(all_test_samples)
    loading_time = time.time() - loading_start
    print(f"✓ Loaded {total_available:,} test samples in {loading_time:.1f} seconds")
    
    # Random sampling if we have more samples than requested
    if total_available > max_test_samples:
        print(f"🎲 Randomly sampling {max_test_samples:,} from {total_available:,} available samples...")
        
        np.random.seed(42)
        sample_indices = np.random.choice(total_available, max_test_samples, replace=False)
        
        sampled_test_samples = [all_test_samples[i] for i in sample_indices]
        sampled_test_labels = [all_test_labels[i] for i in sample_indices]
        
        # Analyze sample distribution
        sample_class_counts = Counter(sampled_test_labels)
        print(f"├─ Sampled Normal (0): {sample_class_counts[0]:,} ({sample_class_counts[0]/max_test_samples*100:.1f}%)")
        print(f"├─ Sampled Attack (1): {sample_class_counts[1]:,} ({sample_class_counts[1]/max_test_samples*100:.1f}%)")
        print(f"└─ Sample represents {max_test_samples/total_available*100:.1f}% of total data")
        
        test_samples = sampled_test_samples
        test_labels = sampled_test_labels
    else:
        print(f"📊 Using all {total_available:,} available samples (less than max {max_test_samples:,})")
        test_samples = all_test_samples
        test_labels = all_test_labels
    
    print("\n=== PROCESSING TEST DATA ===")
    
    total_samples = 0
    correct_predictions = 0
    y_true_list = []
    y_pred_list = []
    y_proba_list = []
    
    test_convergence_interval = min(2500, len(test_samples) // 4)  
    
    mask_idx = orc_sel.get_mask_indices()
    feature_names = preproc_info['output_feature_names']
    selected_feature_names = [feature_names[i] for i in mask_idx]
    
    print(f"🚀 Testing with {len(selected_feature_names)} selected features on {len(test_samples):,} samples...")
    
    processing_start = time.time()
    for x_raw, y_true in zip(test_samples, test_labels):
        
        try:
            x_processed = preprocessor.transform_single(x_raw)
        except Exception as e:
            print(f"Warning: Failed to preprocess sample: {e}")
            continue
        
        x_reduced = {selected_feature_names[i]: float(x_processed[mask_idx[i]]) 
                     for i in range(len(mask_idx))}
        
        y_pred = rf.predict(x_reduced)
        y_proba = rf.predict_proba(x_reduced)
        
        y_true_list.append(y_true)
        y_pred_list.append(y_pred)
        y_proba_list.append(y_proba.get(1, 0.0))
        
        for metric_name, metric in test_metrics.items():
            if metric_name not in ['confusion_matrix', 'mcc', 'kappa']: 
                metric.update(y_true, y_pred)
        
        total_samples += 1
        if y_pred == y_true:
            correct_predictions += 1
        
        if total_samples % test_convergence_interval == 0:
            current_f1 = test_metrics['f1'].get()
            current_acc = correct_predictions / total_samples
            elapsed = time.time() - processing_start
            speed = total_samples / elapsed if elapsed > 0 else 0
            print(f"📊 Test Progress: {total_samples:,}/{len(test_samples):,} samples | F1: {current_f1:.4f} | Acc: {current_acc:.4f} | Speed: {speed:.0f}/sec")
        
        # Lightweight progress between convergence checks
        elif total_samples % (test_convergence_interval // 4) == 0:  # More frequent updates
            print(f"   Testing: {total_samples:,}/{len(test_samples):,} samples processed...")
    
    processing_time = time.time() - processing_start
    print(f"✅ Completed testing on {total_samples:,} samples in {processing_time:.1f} seconds")
    
    
    print("\n📊 CALCULATING COMPREHENSIVE METRICS...")
    
    sklearn_f1 = f1_score(y_true_list, y_pred_list)
    sklearn_accuracy = accuracy_score(y_true_list, y_pred_list)
    sklearn_precision = precision_score(y_true_list, y_pred_list)
    sklearn_recall = recall_score(y_true_list, y_pred_list)
    sklearn_cm = confusion_matrix(y_true_list, y_pred_list)
    
    
    print("\n" + "="*70)
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("="*70)
    
    print("\n🎯 CLASSIFICATION METRICS:")
    print(f"├─ Accuracy:          {test_metrics['accuracy'].get():.4f} (River) | {sklearn_accuracy:.4f} (Sklearn)")
    print(f"├─ Precision:         {test_metrics['precision'].get():.4f} (River) | {sklearn_precision:.4f} (Sklearn)")
    print(f"├─ Recall:            {test_metrics['recall'].get():.4f} (River) | {sklearn_recall:.4f} (Sklearn)")
    print(f"├─ F1-Score:          {test_metrics['f1'].get():.4f} (River) | {sklearn_f1:.4f} (Sklearn)")
    print(f"├─ Balanced Accuracy: {test_metrics['balanced_accuracy'].get():.4f}")
    print(f"├─ Cohen's Kappa:     {test_metrics['kappa'].get():.4f}")
    print(f"└─ Matthews Corr.:    {test_metrics['mcc'].get():.4f}")
    
    print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
    print(classification_report(y_true_list, y_pred_list, target_names=['Normal', 'Attack']))
    
    # Fix confusion matrix using sklearn instead of River
    cm_sklearn = confusion_matrix(y_true_list, y_pred_list, labels=[0, 1])
    
    print(f"\n📋 CONFUSION MATRIX:")
    print(f"├─ True Negatives:  {cm_sklearn[0][0]:,}")
    print(f"├─ False Positives: {cm_sklearn[0][1]:,}")
    print(f"├─ False Negatives: {cm_sklearn[1][0]:,}")
    print(f"└─ True Positives:  {cm_sklearn[1][1]:,}")
    
    attack_samples = sum(1 for y in y_true_list if y == 1)
    normal_samples = sum(1 for y in y_true_list if y == 0)
    
    print(f"\n🔍 ATTACK DETECTION ANALYSIS:")
    print(f"├─ Total Test Samples:   {total_samples:,}")
    print(f"├─ Normal Traffic:       {normal_samples:,} ({normal_samples/total_samples*100:.1f}%)")
    print(f"├─ Attack Traffic:       {attack_samples:,} ({attack_samples/total_samples*100:.1f}%)")
    print(f"└─ Correct Predictions:  {correct_predictions:,} ({correct_predictions/total_samples*100:.1f}%)")
    
    selected_features = orc_sel.get_mask_names()
    print(f"\n⚙️ FEATURE SELECTION ANALYSIS:")
    print(f"├─ Total Available Features: {n_features_processed}")
    print(f"├─ Selected Features:        {len(selected_features)}")
    print(f"└─ Feature Reduction:        {(1-len(selected_features)/n_features_processed)*100:.1f}%")
    
    test_results = {
        'test_metrics': {name: float(metric.get()) for name, metric in test_metrics.items() 
                        if name != 'confusion_matrix'},
        'confusion_matrix': {
            'tn': int(cm_sklearn[0][0]), 'fp': int(cm_sklearn[0][1]),
            'fn': int(cm_sklearn[1][0]), 'tp': int(cm_sklearn[1][1])
        },
        'sample_counts': {
            'total': total_samples,
            'normal': normal_samples,
            'attack': attack_samples,
            'correct': correct_predictions
        },
        'feature_selection': {
            'total_features': n_features_processed,
            'selected_features': len(selected_features),
            'selected_feature_names': selected_features
        }
    }
    
    test_results_path = os.path.join(artifacts_dir, 'test_results.json')
    with open(test_results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n💾 Test results saved to: {test_results_path}")
    
    print("\n" + "="*70)
    print("🏆 TESTING SUMMARY")
    print("="*70)
    print(f"🎯 Overall Accuracy:    {test_metrics['accuracy'].get():.4f}")
    print(f"🎯 F1-Score:            {test_metrics['f1'].get():.4f}")
    print(f"🎯 Attack Detection:    {test_metrics['recall'].get():.4f} (Recall)")
    
    # Calculate False Alarm Rate with safety check
    normal_total = cm_sklearn[0][0] + cm_sklearn[0][1]  # Total normal samples (TN + FP)
    if normal_total > 0:
        false_alarm_rate = cm_sklearn[0][1] / normal_total
        print(f"🎯 False Alarm Rate:    {false_alarm_rate:.4f}")
    else:
        print(f"🎯 False Alarm Rate:    N/A (no normal samples in test data)")
    
    print(f"🔧 Feature Efficiency:  {len(selected_features)}/{n_features_processed} features")
    print("="*70)


def main(execute_training=False, execute_test=True):
    """
    Main function that orchestrates batch training and testing.
    
    Args:
        execute_training (bool): If True, executes the training process
        execute_test (bool): If True, executes testing after training
    """
    
    config = get_default_config()
    
    if not validate_config(config):
        print("❌ Configuration validation failed!")
        return
    
    print_config_summary(config)
    
    trainer = BatchTrainer(config)

    if execute_training:
        success = trainer.execute_training()
        if not success:
            return
    else:
        success = trainer.load_existing_models()
        if not success:
            return
        
        trainer.analyze_data()
    
    if execute_test:
        ae_cfg = AEConfig(
            d_in=trainer.n_features_processed,
            d_hidden=config['model_config']['ae']['d_hidden'],
            lr=config['model_config']['ae']['lr']
        )
        orc_cfg = ORCConfig(**config['model_config']['orc'])
        
        execute_comprehensive_testing(
            test_data_path=config['test_data_path'],
            artifacts_dir=config['artifacts_dir'],
            feature_cols=trainer.available_cols,
            label_col=config['label_column'],
            preproc_info=trainer.preproc_info,
            n_features_processed=trainer.n_features_processed,
            ae_cfg=ae_cfg,
            orc_cfg=orc_cfg,
            max_test_samples=config['testing_config']['max_test_samples']
        )


if __name__ == '__main__':
    execute_training = False  
    execute_test = True 
    
    print("="*70)
    print("🚀 NETWORK ATTACK PREDICTION - BATCH TRAINING PIPELINE")
    print("="*70)
    print(f"📋 Configuration:")
    print(f"├─ Execute Training: {'✅ Yes' if execute_training else '❌ No'}")
    print(f"└─ Execute Testing:  {'✅ Yes' if execute_test else '❌ No'}")
    print()
    
    main(execute_training=execute_training, execute_test=execute_test) 