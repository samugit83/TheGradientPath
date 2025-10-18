"""
NETWORK ATTACK PREDICTION - INCREMENTAL TRAINING SCRIPT
=======================================================

This script implements incremental/online training for network attack prediction models.
It handles streaming data and updates models in real-time without retraining from scratch.

FEATURES:
---------
✓ **Streaming Data Processing**: Handles batches of network traffic data
✓ **Incremental Learning**: Updates models without full retraining
✓ **Class Distribution Tracking**: Progressive monitoring of normal vs attack traffic
✓ **Model Persistence**: Saves updated models after each batch
✓ **Robust Error Handling**: Gracefully handles preprocessing and model errors
✓ **Memory Efficient**: Processes data in batches without storing entire dataset
✓ **Production Ready**: Designed for real-time deployment scenarios

KEY COMPONENTS:
--------------
1. **IncrementalTrainer**: Main class handling streaming training pipeline
2. **Progressive Class Tracking**: Monitors training balance over time
3. **Model State Management**: Loads existing models or initializes new ones

USAGE:
------
from incremental_train import IncrementalTrainer

trainer = IncrementalTrainer("artifacts")
if trainer.initialize_or_load_models():
    # Process streaming batches
    flows_data = [{'features': {...}, 'label': 0}, ...]
    result = trainer.process_streaming_batch(flows_data)
    
    # Check training status
    status = trainer.get_training_status()
"""

import os
import numpy as np
import pandas as pd
import json
import time
import torch
from typing import List, Dict, Any

from modules.ae import AEWrapper, AEConfig
from modules.orc_selector import ORCFeatureSelector, ORCConfig
from modules.sklearn_wrapper import SklearnWrapper, SklearnConfig
from modules.data_preprocessing import DataPreprocessor


class IncrementalTrainer:
    """
    Handles incremental training on streaming network traffic data.
    
    This class manages:
    1. Loading existing models or creating new ones
    2. Processing streaming data batches
    3. Updating models incrementally
    4. Progressive class distribution tracking
    5. Saving updated artifacts
    """
    
    def __init__(self, artifacts_dir: str = "artifacts", config: Dict[str, Any] = None):
        """
        Initialize the incremental trainer.
        
        Args:
            artifacts_dir: Directory containing model artifacts
            config: Optional model configuration dict. If None, will use default config.
        """
        self.artifacts_dir = artifacts_dir
        self.preprocessor = None
        self.ae = None
        self.orc_sel = None
        self.rf = None
        
        if config is None:
            # Load full default config and extract model_config + root flags
            from config import get_default_config
            full_config = get_default_config()
            self.config = full_config['model_config'].copy()
            self.config['apply_feature_selection'] = full_config.get('apply_feature_selection', True)
            print(f"🔄 Loaded default config with apply_feature_selection: {self.config['apply_feature_selection']}")
        else:
            self.config = config
            if 'apply_feature_selection' not in self.config:
                self.config['apply_feature_selection'] = True
                print(f"🔄 Added missing apply_feature_selection: {self.config['apply_feature_selection']}")
        
        self.is_initialized = False
        
        # Track training stats
        self.total_samples_processed = 0
        self.batches_processed = 0
        
        # Track class distribution progressively
        self.total_normal_samples = 0  # Label 0 (normal traffic)
        self.total_attack_samples = 0  # Label 1 (attack traffic)
        

    def initialize_or_load_models(self) -> bool:
        """
        Load existing models from artifacts or initialize new ones.
        
        Returns:
            True if models were loaded/initialized successfully
        """
        self.is_initialized = True
        metadata_path = os.path.join(self.artifacts_dir, 'training_metadata.json')
        
        if os.path.exists(metadata_path):
            return self._load_existing_models()
        else:
            print("🆕 No existing models found - will initialize on first training batch")
            return True


    def _load_existing_models(self) -> bool:
        """Load existing trained models from artifacts."""
        try:
            print("🔄 Loading existing models for incremental training...")
            
            # Load training metadata
            metadata_path = os.path.join(self.artifacts_dir, 'training_metadata.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Get current config (either provided or default)
            if self.config is None:
                from config import get_default_config
                current_config = get_default_config()['model_config'].copy()
                # IMPORTANT: Include root-level flags
                default_full_config = get_default_config()
                current_config['apply_feature_selection'] = default_full_config.get('apply_feature_selection', True)
                print("🔄 Using default configuration from config.py")
                print(f"🔄 Default apply_feature_selection: {current_config['apply_feature_selection']}")
            else:
                current_config = self.config
                print("🔄 Using provided configuration")
                # Ensure apply_feature_selection is included
                if 'apply_feature_selection' not in current_config:
                    current_config['apply_feature_selection'] = True
                    print(f"🔄 Added missing apply_feature_selection: {current_config['apply_feature_selection']}")
            
            # Load the saved model config but merge with current config for new flags
            loaded_model_config = metadata['model_config']
            
            # Merge current config into loaded config to ensure new flags are available
            for key, value in current_config.items():
                if key not in loaded_model_config:
                    loaded_model_config[key] = value
                    print(f"🔄 Added new config flag: {key} = {value}")
            
            self.config = loaded_model_config
            n_features_processed = metadata['training_stats']['processed_features']
            
            # Load existing class distribution counts if available
            training_stats = metadata.get('training_stats', {})
            self.total_normal_samples = training_stats.get('total_normal_samples', 0)
            self.total_attack_samples = training_stats.get('total_attack_samples', 0)
            self.total_samples_processed = training_stats.get('total_samples', 0)
            self.batches_processed = training_stats.get('batches_processed', 0)
            
            print(f"✓ Loaded existing class distribution: Normal: {self.total_normal_samples:,}, Attack: {self.total_attack_samples:,}")
            
            # Load preprocessor
            preprocessor_path = os.path.join(self.artifacts_dir, 'preprocessor.pkl')
            self.preprocessor = DataPreprocessor.load(preprocessor_path)
            print("✓ Preprocessor loaded")
            
            # Check if feature selection is enabled in the loaded config
            apply_feature_selection = self.config.get('apply_feature_selection', True)
            
            if apply_feature_selection:
                print("✅ Feature selection ENABLED - loading AutoEncoder + ORC")
                
                # Load AutoEncoder
                ae_cfg = AEConfig(
                    d_in=n_features_processed,
                    d_hidden=self.config['ae']['d_hidden'],
                    lr=self.config['ae']['lr']
                )
                self.ae = AEWrapper(ae_cfg)
                ae_path = os.path.join(self.artifacts_dir, 'ae.pt')
                self.ae.load(ae_path)
                print("✓ AutoEncoder loaded")
                
                # Load ORC Feature Selector
                orc_cfg = ORCConfig(**self.config['orc'])
                # Override config to match production metadata exactly
                selected_features_from_metadata = metadata['training_stats']['selected_features']
                orc_cfg.top_k = selected_features_from_metadata  # Use exact count from metadata
                orc_path = os.path.join(self.artifacts_dir, 'orc.npz')
                self.orc_sel = ORCFeatureSelector.load(orc_path, orc_cfg, training_mode="incremental")
                print(f"✓ ORC Feature Selector loaded (using {selected_features_from_metadata} features from metadata)")
                
                # NEW: Check and restore ORC feature locking from training metadata
                orc_metadata = metadata.get('orc_feature_selection', {})
                if orc_metadata.get('is_locked', False):
                    locked_indices = orc_metadata.get('locked_feature_indices', [])
                    samples_seen = orc_metadata.get('samples_seen', 0)
                    
                    if locked_indices:
                        # Restore locked features
                        self.orc_sel.set_locked_features(np.array(locked_indices), samples_seen)
                        print(f"🔒 Restored locked feature selection from training_metadata.json")
                        print(f"├─ Locked after: {samples_seen} samples")
                        print(f"├─ Features locked: {len(locked_indices)}")
                        print(f"└─ Lock threshold: {orc_metadata.get('lock_threshold', 100)}")
                    else:
                        print("⚠️ Feature selection marked as locked but no locked indices found")
                else:
                    print(f"🔓 Feature selection is still active (seen: {orc_metadata.get('samples_seen', 0)}/{orc_metadata.get('lock_threshold', 100)} samples)")
            else:
                print("⚠️ Feature selection DISABLED - using ALL features directly")
                self.ae = None
                self.orc_sel = None
            
            # Load SGD Classifier
            rf_path = os.path.join(self.artifacts_dir, 'rf.pkl')
            self.rf = SklearnWrapper.load(rf_path)
            
            # Synchronize SGD classifier's class counts with metadata
            # This ensures consistency between saved metadata and classifier state
            if hasattr(self.rf, 'class_counts'):
                # Update classifier's internal counts to match metadata
                self.rf.class_counts[0] = self.total_normal_samples
                self.rf.class_counts[1] = self.total_attack_samples
                print(f"✓ Synchronized SGD classifier counts: Normal: {self.total_normal_samples:,}, Attack: {self.total_attack_samples:,}")
            
            print("✓ SGD Classifier loaded")
            
            self.is_initialized = True
            print("✅ All existing models loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load existing models: {e}")
            return False


    def process_streaming_batch(self, flows_data: List[Dict]) -> Dict[str, Any]:
        """
        Process a batch of streaming network traffic data for incremental training.
        
        Args:
            flows_data: List of flow dictionaries with 'features' and 'label'
            
        Returns:
            Dictionary with processing results and metrics
        """
        if not self.is_initialized:
            raise ValueError("Models must be initialized first")
        
        try:
            print(f"\n🔄 Processing streaming batch: {len(flows_data)} samples")
            
            features_list = []
            labels_list = []
            
            for flow in flows_data:
                if 'features' in flow and 'label' in flow:
                    features_list.append(flow['features'])
                    labels_list.append(flow['label'])
            
            if not features_list:
                return {'error': 'No valid samples in batch'}
            
            features_df = pd.DataFrame(features_list)
            
            # Handle first batch - initialize models
            if self.preprocessor is None:
                return self._initialize_models_from_first_batch(features_df, labels_list)
            
            # Update preprocessor with new categorical values
            update_info = self.preprocessor.update_with_new_data(features_df)
            
            # Update scaler with new numerical data (for adaptive scalers)
            if features_df is not None and len(features_df) > 0:
                self.preprocessor.partial_fit_scaler(features_df)
                print(f"📊 Scaler stats: {self.preprocessor.get_scaler_stats()}")
            
            # Process samples through the pipeline
            processed_samples = []
            processed_labels = []
            
            # Check if feature selection is enabled
            apply_feature_selection = self.config.get('apply_feature_selection', True)
            
            # Initialize mask_idx for tracking selected features
            mask_idx = []  # Initialize to avoid reference error
            
            for i, (features, label) in enumerate(zip(features_list, labels_list)):
                try:
                    # Transform sample
                    x_processed = self.preprocessor.transform_single(features)
                    
                    if apply_feature_selection:
                        # FEATURE SELECTION ENABLED: Use AutoEncoder + ORC pipeline
                        x_tensor = torch.from_numpy(x_processed.astype(np.float32))
                        
                        # AutoEncoder forward pass and error computation
                        recon = self.ae.forward_no_grad(x_tensor)
                        err = np.abs(x_processed - recon.numpy())
                        
                        # Update ORC
                        self.orc_sel.update(err)
                        
                        # Get selected features
                        mask_idx = self.orc_sel.get_mask_indices()
                        feature_names = self.preprocessor.get_feature_names()
                        x_reduced = {feature_names[j]: float(x_processed[j]) for j in mask_idx}
                        
                        if i % 5 == 0 and label == 0:  # Train every 5th sample, but ONLY if it's normal traffic
                            self.ae.train_step(x_tensor)
                            print(f"🔧 AutoEncoder trained on normal sample {i} (label={label})")
                        elif label == 1:
                            print(f"⚠️ Skipping AE training on attack sample {i} (label={label}) - AE should only learn normal patterns!")
                    else:
                        # FEATURE SELECTION DISABLED: Use all features directly
                        feature_names = self.preprocessor.get_feature_names()
                        x_reduced = {feature_names[j]: float(x_processed[j]) for j in range(len(x_processed))}
                        # Set mask_idx to all features when feature selection is disabled
                        mask_idx = list(range(len(x_processed)))
                        
                        if i == 0:  # Log once per batch
                            print(f"⚠️ Using ALL {len(x_processed)} features (feature selection disabled)")
                    
                    processed_samples.append(x_reduced)
                    processed_labels.append(label)
                    
                except Exception as e:
                    print(f"⚠️ Warning: Failed to process sample {i}: {e}")
                    continue
            
            # Two-phase incremental training approach
            # Phase 1: Feature selection stabilization (don't train classifier yet)
            # Phase 2: Stable classifier training (after features are locked)
            if len(processed_samples) > 0:
                if apply_feature_selection:
                    # Check if feature selection is locked
                    if self.orc_sel.is_locked():
                        # Phase 2: Features are stable - safe to train classifier
                        print(f"🔄 Phase 2: Features LOCKED - training SGD classifier with {len(processed_samples)} samples")
                        self.rf.fit(processed_samples, processed_labels)
                        print(f"✅ SGD classifier training completed")
                    else:
                        # Phase 1: Still doing feature selection - don't train classifier yet
                        samples_seen = self.orc_sel._samples_seen
                        lock_threshold = self.orc_sel.cfg.lock_after_samples
                        print(f"🔄 Phase 1: Feature selection ACTIVE - skipping SGD training")
                        print(f"├─ Progress: {samples_seen}/{lock_threshold} samples seen")
                        print(f"└─ SGD training will start after feature selection locks")
                else:
                    # No feature selection - can train classifier immediately
                    print(f"🔄 No feature selection - training SGD classifier with {len(processed_samples)} samples")
                    self.rf.fit(processed_samples, processed_labels)
                    print(f"✅ SGD classifier training completed")

            # Update counters
            self.total_samples_processed += len(processed_samples)
            self.batches_processed += 1
            
            # Calculate batch metrics
            if processed_labels:
                normal_count = sum(1 for label in processed_labels if label == 0)
                attack_count = sum(1 for label in processed_labels if label == 1)
            else:
                normal_count = attack_count = 0
            
            # Update cumulative class distribution counters
            self.total_normal_samples += normal_count
            self.total_attack_samples += attack_count
            
            # Save updated models
            self._save_updated_models()
            
            result = {
                'success': True,
                'processed_samples': len(processed_samples),
                'normal_samples': normal_count,
                'attack_samples': attack_count,
                'total_processed': self.total_samples_processed,
                'batches_processed': self.batches_processed,
                'selected_features': len(mask_idx),
                'preprocessing_updates': update_info,
                # Add cumulative class distribution tracking
                'total_normal_samples': self.total_normal_samples,
                'total_attack_samples': self.total_attack_samples,
                'class_balance_ratio': self.total_normal_samples / max(self.total_attack_samples, 1)  # Avoid division by zero
            }
            
            print(f"✅ Batch processed: {len(processed_samples)} samples")
            print(f"├─ Batch: Normal: {normal_count}, Attack: {attack_count}")
            print(f"├─ Cumulative: Normal: {self.total_normal_samples:,}, Attack: {self.total_attack_samples:,}")
            print(f"├─ Class balance ratio: {self.total_normal_samples / max(self.total_attack_samples, 1):.2f}:1")
            print(f"├─ Selected features: {len(mask_idx)}")
            print(f"└─ Total processed: {self.total_samples_processed:,}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing batch: {e}")
            return {'error': str(e), 'success': False}

    def _initialize_models_from_first_batch(self, features_df: pd.DataFrame, labels_list: List[int]) -> Dict[str, Any]:
        """Initialize all models from the first batch of data."""
        print("🆕 Initializing models from first batch...")
        
        self.preprocessor = DataPreprocessor()
        self.preprocessor.fit(features_df)
        
        sample_processed = self.preprocessor.transform(features_df[:1])
        n_features_processed = sample_processed.shape[1]
        
        apply_feature_selection = self.config.get('apply_feature_selection', True)
        
        if apply_feature_selection:
            print("✅ Feature selection ENABLED - using AutoEncoder + ORC")
            
            ae_cfg = AEConfig(
                d_in=n_features_processed,
                d_hidden=self.config['ae']['d_hidden'],
                lr=self.config['ae']['lr']
            )
            self.ae = AEWrapper(ae_cfg)
            
            orc_cfg = ORCConfig(**self.config['orc'])
            feature_names = self.preprocessor.get_feature_names()
            self.orc_sel = ORCFeatureSelector(
                n_features=n_features_processed,
                cfg=orc_cfg,
                feature_names=feature_names,
                training_mode="incremental" 
            )
        else:
            print("⚠️ Feature selection DISABLED - using ALL features directly")
            self.ae = None
            self.orc_sel = None
        
        sgd_cfg = SklearnConfig(**self.config['sgd_classifier'])
        self.rf = SklearnWrapper(sgd_cfg)
        
        print(f"✅ Models initialized with {n_features_processed} features")
        
        return self.process_streaming_batch([
            {'features': features_df.iloc[i].to_dict(), 'label': labels_list[i]}
            for i in range(len(features_df))
        ])

    def _save_updated_models(self):
        """Save all updated models to artifacts."""
        try:
            os.makedirs(self.artifacts_dir, exist_ok=True)
            
            # Always save preprocessor and classifier
            self.preprocessor.save(os.path.join(self.artifacts_dir, 'preprocessor.pkl'))
            self.rf.save(os.path.join(self.artifacts_dir, 'rf.pkl'))
            
            # Check if feature selection is enabled
            apply_feature_selection = self.config.get('apply_feature_selection', True)
            
            if apply_feature_selection:
                # Save AutoEncoder and ORC selector only if feature selection is enabled
                self.ae.save(os.path.join(self.artifacts_dir, 'ae.pt'))
                self.orc_sel.save(os.path.join(self.artifacts_dir, 'orc.npz'))
                
                # Get ORC locking status
                orc_lock_status = self.orc_sel.get_lock_status()
                selected_features = self.orc_sel.get_mask_names()
            else:
                # When feature selection is disabled, use all features
                feature_names = self.preprocessor.get_feature_names()
                selected_features = feature_names
                orc_lock_status = {
                    'is_locked': False,
                    'samples_seen': 0,
                    'lock_threshold': None,
                    'locked_features_count': 0,
                    'total_features': len(feature_names),
                    'locked_feature_names': []
                }
            
            # Update metadata
            feature_info = self.preprocessor.get_feature_info()
            
            metadata = {
                'model_config': self.config,
                'training_stats': {
                    'total_samples': self.total_samples_processed,
                    'processed_features': feature_info['total_output_features'],
                    'selected_features': len(selected_features),
                    'batches_processed': self.batches_processed,
                    'optimized_attack_threshold': float(self.rf.cfg.attack_threshold),
                    # Add cumulative class distribution tracking
                    'total_normal_samples': self.total_normal_samples,
                    'total_attack_samples': self.total_attack_samples,
                    'class_balance_ratio': self.total_normal_samples / max(self.total_attack_samples, 1)
                },
                'feature_info': feature_info,
                # NEW: Feature selection configuration
                'apply_feature_selection': apply_feature_selection,  # Critical flag for training/prediction
                'feature_selection_enabled': apply_feature_selection,  # Backward compatibility
                'orc_feature_selection': {
                    'is_locked': orc_lock_status['is_locked'],
                    'samples_seen': orc_lock_status['samples_seen'],
                    'lock_threshold': orc_lock_status['lock_threshold'],
                    'locked_features_count': orc_lock_status['locked_features_count'],
                    'total_features': orc_lock_status['total_features'],
                    'locked_feature_names': orc_lock_status['locked_feature_names'],
                    'locked_feature_indices': self.orc_sel.get_mask_indices().tolist() if apply_feature_selection and orc_lock_status['is_locked'] else [],
                    'lock_message': f"Feature selection {'LOCKED' if orc_lock_status['is_locked'] else 'ACTIVE'} after {orc_lock_status['samples_seen']} samples" if apply_feature_selection else "Feature selection DISABLED - using all features"
                },
                # Training mode information
                'training_mode': 'incremental',  # Identifies this as incremental training
                'last_updated': time.time()
            }
            
            metadata_path = os.path.join(self.artifacts_dir, 'training_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to save models: {e}")

    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status and metrics."""
        if not self.is_initialized:
            return {'initialized': False}
        
        status = {
            'initialized': True,
            'total_samples_processed': self.total_samples_processed,
            'batches_processed': self.batches_processed,
            'models_loaded': {
                'preprocessor': self.preprocessor is not None,
                'autoencoder': self.ae is not None,
                'orc_selector': self.orc_sel is not None,
                'sgd_classifier': self.rf is not None
            },
            # Add class distribution tracking
            'class_distribution': {
                'total_normal_samples': self.total_normal_samples,
                'total_attack_samples': self.total_attack_samples,
                'class_balance_ratio': self.total_normal_samples / max(self.total_attack_samples, 1),
                'normal_percentage': (self.total_normal_samples / max(self.total_samples_processed, 1)) * 100,
                'attack_percentage': (self.total_attack_samples / max(self.total_samples_processed, 1)) * 100
            }
        }
        
        if self.orc_sel:
            status['selected_features'] = len(self.orc_sel.get_mask_indices())
        
        # Add SGDClassifier status if available
        if self.rf and hasattr(self.rf, 'classifier') and hasattr(self.rf.classifier, 'classes_'):
            status['sgd_classifier_classes'] = list(self.rf.classifier.classes_)
            status['can_predict_attacks'] = 1 in self.rf.classifier.classes_
        
        return status

    def reset_training_state(self):
        """Reset all training counters and statistics."""
        print("🔄 Resetting training state...")
        self.total_samples_processed = 0
        self.batches_processed = 0
        self.total_normal_samples = 0
        self.total_attack_samples = 0
        print("✅ Training state reset")

    def get_class_distribution_summary(self) -> Dict[str, Any]:
        """Get a detailed summary of class distribution."""
        total_samples = max(self.total_samples_processed, 1)  # Avoid division by zero
        
        return {
            'total_samples': self.total_samples_processed,
            'normal_samples': self.total_normal_samples,
            'attack_samples': self.total_attack_samples,
            'normal_percentage': (self.total_normal_samples / total_samples) * 100,
            'attack_percentage': (self.total_attack_samples / total_samples) * 100,
            'class_balance_ratio': self.total_normal_samples / max(self.total_attack_samples, 1),
            'is_balanced': abs(self.total_normal_samples - self.total_attack_samples) / total_samples < 0.1,  # Within 10%
            'batches_processed': self.batches_processed
        }

    def export_training_history(self, filepath: str):
        """Export training history and statistics to a JSON file."""
        try:
            history = {
                'training_summary': {
                    'total_samples_processed': self.total_samples_processed,
                    'batches_processed': self.batches_processed,
                    'artifacts_directory': self.artifacts_dir,
                    'last_updated': time.time()
                },
                'class_distribution': self.get_class_distribution_summary(),
                'model_status': self.get_training_status(),
                'configuration': self.config
            }
            
            with open(filepath, 'w') as f:
                json.dump(history, f, indent=2, default=str)
            
            print(f"✅ Training history exported to: {filepath}")
            
        except Exception as e:
            print(f"❌ Failed to export training history: {e}")

