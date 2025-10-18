"""
NETWORK ATTACK PREDICTION - CONFIGURATION FILE
===============================================

This file contains all configuration parameters for the network attack prediction system.
All hardcoded values have been moved here for centralized configuration management.

Configuration sections:
- Data configuration: File paths and data handling
- Preprocessing configuration: Data preprocessing settings  
- Model configuration: ML model parameters
- Training configuration: Training process settings
- Testing configuration: Testing and evaluation settings
"""

# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    # ========================================================================
    # FEATURE SELECTION CONFIGURATION
    # ========================================================================
    'apply_feature_selection': False,  # Enable/disable AutoEncoder + ORC feature selection
    
    # ========================================================================
    # DATA CONFIGURATION
    # ========================================================================
    'data_path': "dataset/UNSW_NB15_train.csv",
    'test_data_path': "dataset/UNSW_NB15_test.csv", 
    'label_column': "label",
    'feature_columns': None,  # Auto-detect features
    'exclude_columns': ["id", "attack_cat"],
    
    # ========================================================================
    # OUTPUT CONFIGURATION
    # ========================================================================
    'artifacts_dir': "artifacts",
    
    # ========================================================================
    # PREPROCESSING CONFIGURATION
    # ========================================================================
    'preprocessing_config': {
        'apply_log_transform': True,        # Enable/disable logarithmic transformation for extreme ranges
        'extreme_range_threshold': 1e4,     # Features with range > this value get log transformed
        'categorical_encoding': 'label',    # Encoding method for categorical features
        'handle_unknown': 'ignore',         # How to handle unknown categories
        'scale_features': True              # Enable/disable feature scaling
    },
    
    # ========================================================================
    # MODEL CONFIGURATION
    # ========================================================================
    'model_config': {
        # AutoEncoder (AE) configuration
        'ae': {
            'd_hidden': 32,    # Size of compressed representation
            'lr': 1e-4         # Learning rate
        },
        
        # Online Reconstruction Control (ORC) configuration
        'orc': {
            'beta': 0.8,           # Smoothing factor for error averaging
            'top_k': 30,           # Number of top features to select
            'update_every': 500,   # Frequency of feature ranking updates
            'lock_after_samples': 100  # NEW: Lock feature selection after this many samples (incremental mode only)
        },
        
        # SGD Classifier configuration (replaces Random Forest)
        'sgd_classifier': {
            # SGDClassifier parameters for incremental learning
            'loss': 'log_loss',              # For probability estimates
            'learning_rate': 'adaptive',     # Adaptive learning rate
            'eta0': 0.01,                    # Initial learning rate
            'alpha': 0.0001,                 # Regularization strength
            'random_state': 42,              # Reproducibility
            'max_iter': 1000,                # Max iterations per fit
            'tol': 1e-3,                     # Tolerance for stopping
            'attack_threshold': 0.5,         # Threshold for attack classification (will be optimized)
            
            # Probability calibration configuration
            'enable_calibration': True,      # Enable/disable Platt scaling calibration
            'calibration_method': 'sigmoid', # Calibration method: 'sigmoid' or 'isotonic'
            'calibration_threshold': 1000    # Minimum samples before applying calibration
        },
        
        # Training control
        'ae_train_every': 1,        # AE training frequency
        'preprocessing_fit_size': 10000,  # Samples to fit preprocessor
        'combine_datasets': True,   # Combine training and testing datasets
        'train_test_split': 0.8     # 80% for training, 20% for testing
    },
    
    # ========================================================================
    # TRAINING CONFIGURATION
    # ========================================================================
    'training_config': {
        'balance_dataset': True,        # Balance classes during training
        'batch_size': 5000,            # Batch size for data streaming
        'progress_interval': 10000,    # Show progress every N samples
        'convergence_interval': 2500,  # Check convergence every N samples
        'threshold_optimization': {
            'enabled': True,           # Enable optimal threshold calculation
            'sample_size': 10000,      # Max samples for threshold optimization
            'production_adjustment': 0.2  # Add to threshold for production (reduce false positives)
        }
    },
    
    # ========================================================================
    # TESTING CONFIGURATION
    # ========================================================================
    'testing_config': {
        'max_test_samples': 10000,     # Maximum samples for testing
        'random_sampling': True,       # Use random sampling for large test sets
        'random_seed': 42,            # Seed for reproducible sampling
        'save_results': True,         # Save detailed test results
        'metrics_to_calculate': [     # Metrics to calculate during testing
            'accuracy', 'precision', 'recall', 'f1', 
            'balanced_accuracy', 'confusion_matrix', 
            'kappa', 'mcc'
        ]
    }
}


def get_default_config():
    """
    Get the default configuration dictionary.
    
    Returns:
        dict: Complete default configuration
    """
    return DEFAULT_CONFIG.copy()


def validate_config(config):
    """
    Validate configuration parameters.
    
    Args:
        config (dict): Configuration to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    required_keys = [
        'data_path', 'label_column', 'artifacts_dir', 'model_config'
    ]
    
    for key in required_keys:
        if key not in config:
            print(f"❌ Missing required configuration key: {key}")
            return False
    
    # Validate model config structure
    model_config = config['model_config']
    required_model_keys = ['ae', 'orc', 'sgd_classifier']
    
    for key in required_model_keys:
        if key not in model_config:
            print(f"❌ Missing required model configuration key: {key}")
            return False
    
    print("✅ Configuration validation passed")
    return True



def print_config_summary(config):
    """
    Print a summary of the configuration.
    
    Args:
        config (dict): Configuration to summarize
    """
    print("📋 CONFIGURATION SUMMARY:")
    print(f"├─ Data path: {config.get('data_path', 'Not set')}")
    print(f"├─ Label column: {config.get('label_column', 'Not set')}")
    print(f"├─ Artifacts directory: {config.get('artifacts_dir', 'Not set')}")
    
    if 'model_config' in config:
        model_cfg = config['model_config']
        print(f"├─ AutoEncoder hidden size: {model_cfg.get('ae', {}).get('d_hidden', 'Not set')}")
        print(f"├─ ORC top features: {model_cfg.get('orc', {}).get('top_k', 'Not set')}")
        print(f"└─ SGD Classifier loss: {model_cfg.get('sgd_classifier', {}).get('loss', 'Not set')}")
    else:
        print(f"└─ Model config: Not set")
