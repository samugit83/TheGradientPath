"""
SKLEARN CLASSIFIER WRAPPER - INCREMENTAL LEARNING IMPLEMENTATION
================================================================

This module uses sklearn's SGDClassifier for true incremental learning.
SGDClassifier supports partial_fit() which allows true online learning.

Key features:
- True incremental learning with partial_fit()
- Handles imbalanced data with manual class weighting
- Works with mixed numerical/categorical features
- Fast inference for real-time network monitoring
- Built-in feature importance approximation
"""

import pickle
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.calibration import CalibratedClassifierCV
from collections import deque


@dataclass
class SklearnConfig:
    """
    Configuration class for the sklearn-based incremental classifier.
    
    This contains SGDClassifier-specific parameters optimized
    for imbalanced network attack detection with incremental learning.
    """
    # SGDClassifier parameters for incremental learning
    loss: str = 'log_loss'              # For probability estimates
    learning_rate: str = 'adaptive'     # Adaptive learning rate
    eta0: float = 0.01                  # Initial learning rate
    alpha: float = 0.0001               # Regularization strength
    random_state: int = 42              # Reproducibility
    max_iter: int = 1000                # Max iterations per fit
    tol: float = 1e-3                   # Tolerance for stopping
    
    # Attack prediction parameters
    attack_threshold: float = 0.5       # Threshold for attack classification
    
    # Probability calibration parameters
    enable_calibration: bool = True      # Enable/disable Platt scaling calibration
    calibration_method: str = 'sigmoid'  # Calibration method: 'sigmoid' or 'isotonic'
    calibration_threshold: int = 1000    # Minimum samples before applying calibration
    
    # Incremental learning parameters
    batch_size: int = 32                # Batch size for partial_fit
    adaptation_rate: float = 0.1        # How quickly to adapt to new data


class SklearnWrapper:
    """
    Sklearn-based classifier wrapper using SGDClassifier for incremental learning.
    
    This class provides true incremental learning capabilities:
    1. Uses partial_fit() for incremental updates
    2. Maintains model state across batches
    3. Handles class imbalance dynamically
    4. Provides probability estimates
    5. Adapts to concept drift over time
    """
    
    def __init__(self, cfg: SklearnConfig):
        """
        Initialize the incremental classifier wrapper.
        
        Args:
            cfg: Configuration object with hyperparameters
        """
        self.cfg = cfg
        
        # Initialize the SGD classifier for incremental learning
        self.base_classifier = SGDClassifier(
            loss=cfg.loss,
            learning_rate=cfg.learning_rate,
            eta0=cfg.eta0,
            alpha=cfg.alpha,
            random_state=cfg.random_state,
            max_iter=cfg.max_iter,
            tol=cfg.tol
        )
        
        # We'll use the base classifier directly for incremental learning
        # and apply calibration only after sufficient training data
        self.classifier = self.base_classifier
        self.calibrated_classifier = None
        
        # Calibration configuration from cfg
        self.enable_calibration = getattr(cfg, 'enable_calibration', True)
        self.calibration_method = getattr(cfg, 'calibration_method', 'sigmoid')
        self.calibration_threshold = getattr(cfg, 'calibration_threshold', 1000)
        self.needs_calibration = self.enable_calibration
        
        
        # Track training state
        self.is_trained = False
        self.classes_seen = set()
        self.sample_count = 0
        
        # Performance tracking
        self.predictions = deque(maxlen=1000)
        self.true_labels = deque(maxlen=1000)
        
        # Feature importance approximation (SGD doesn't have built-in importance)
        self.feature_names = []
        
        # Manual class weight tracking for balanced learning
        self.class_counts = {0: 0, 1: 0}

    def fit(self, X_dict_list, y_list):
        """
        Fit the classifier using incremental learning.
        
        This method uses partial_fit for true incremental learning,
        allowing the model to learn from new data without forgetting old data.
        
        Args:
            X_dict_list: List of feature dictionaries
            y_list: List of labels (0=Normal, 1=Attack)
        """
        if len(X_dict_list) == 0:
            return
        
        # Get feature names from first sample
        self.feature_names = list(X_dict_list[0].keys())
        
        # Convert to numpy array
        X = np.array([[sample[fname] for fname in self.feature_names] 
                     for sample in X_dict_list])
        y = np.array(y_list)
        
        
        # Track classes and counts for balanced learning
        unique_classes, counts = np.unique(y, return_counts=True)
        print(f"🔍 DEBUG: Updating class counts - batch classes: {list(unique_classes)}, counts: {list(counts)}")
        for cls, count in zip(unique_classes, counts):
            self.classes_seen.add(cls)
            old_count = self.class_counts.get(cls, 0)
            self.class_counts[cls] = old_count + count
            print(f"🔍 DEBUG: Class {cls}: {old_count} + {count} = {self.class_counts[cls]}")
        
        # Compute sample weights for balanced learning
        if len(self.classes_seen) > 1:
            # Use cumulative class counts for balanced learning, not just current batch
            total_normal = self.class_counts.get(0, 0)
            total_attack = self.class_counts.get(1, 0)
            
            # Compute class weights based on cumulative counts
            if total_normal > 0 and total_attack > 0:
                # Inverse frequency weighting for balance
                total_samples = total_normal + total_attack
                weight_normal = total_samples / (2 * total_normal)
                weight_attack = total_samples / (2 * total_attack)
                class_weight_dict = {0: weight_normal, 1: weight_attack}
            else:
                # Fallback to equal weights if one class has no samples
                class_weight_dict = {0: 1.0, 1: 1.0}
            
            # Create sample weights based on class weights
            sample_weights = np.array([class_weight_dict[label] for label in y])
        else:
            # If only one class seen so far, use equal weights
            sample_weights = np.ones(len(y))
        
        # Use partial_fit for incremental learning
        if not self.is_trained:
            # First training - need to specify all possible classes
            all_classes = [0, 1]  # Normal and Attack classes
            self.classifier.partial_fit(X, y, classes=all_classes, sample_weight=sample_weights)
            self.is_trained = True
            print(f"✅ Initial incremental training on {len(X_dict_list)} samples")
        else:
            # Incremental update - just add new data
            self.classifier.partial_fit(X, y, sample_weight=sample_weights)
            print(f"🔄 Incremental update with {len(X_dict_list)} samples")
        
        self.sample_count += len(X_dict_list)
        
        # Check if we need to perform calibration
        if (self.enable_calibration and 
            self.needs_calibration and 
            self.sample_count >= self.calibration_threshold and 
            len(self.classes_seen) > 1):
            self.perform_calibration(X, y)
        
        # Debug information
        print(f"├─ Current batch: Normal: {sum(1 for label in y if label == 0)}, Attack: {sum(1 for label in y if label == 1)}")
        print(f"├─ Cumulative counts: Normal: {self.class_counts.get(0, 0)}, Attack: {self.class_counts.get(1, 0)}")
        print(f"├─ Total samples processed: {self.sample_count}")
        print(f"├─ Classes seen: {sorted(list(self.classes_seen))}")
        if len(self.classes_seen) > 1:
            total_normal = self.class_counts.get(0, 0)
            total_attack = self.class_counts.get(1, 0)
            if total_normal > 0 and total_attack > 0:
                print(f"├─ Class weights: Normal: {(total_normal + total_attack) / (2 * total_normal):.3f}, Attack: {(total_normal + total_attack) / (2 * total_attack):.3f}")
        if self.enable_calibration:
            calibration_status = "✅ Calibrated" if not self.needs_calibration else f"⏳ Need {self.calibration_threshold - self.sample_count} more samples"
            print(f"├─ Calibration: ENABLED ({self.calibration_method}) - {calibration_status}")
        else:
            print(f"├─ Calibration: DISABLED")
        print(f"└─ Classifier classes: {list(self.classifier.classes_) if hasattr(self.classifier, 'classes_') else 'Not available'}")

    def perform_calibration(self, X_recent, y_recent):
        """
        Perform probability calibration using recent training data.
        
        Args:
            X_recent: Recent training features
            y_recent: Recent training labels
        """
        try:
            print("🎯 PERFORMING PROBABILITY CALIBRATION...")
            
            # Create calibrated classifier using recent data for calibration
            self.calibrated_classifier = CalibratedClassifierCV(
                self.base_classifier,
                method=self.calibration_method,  # Use configured calibration method
                cv='prefit'  # Use pre-fitted classifier
            )
            
            # Fit calibration using recent data
            self.calibrated_classifier.fit(X_recent, y_recent)
            
            # Switch to using calibrated classifier for predictions
            self.classifier = self.calibrated_classifier
            self.needs_calibration = False
            
            print("✅ Probability calibration completed!")
            print(f"├─ Method: {self.calibration_method}")
            print("├─ Calibration data: Recent batch")
            print("└─ Calibrated probabilities now available")
            
        except Exception as e:
            print(f"⚠️ WARNING: Calibration failed: {e}")
            print("├─ Continuing with uncalibrated classifier")
            print("└─ This may affect threshold optimization")

    def predict(self, x: Dict[str, float]) -> int:
        """
        Make a prediction for a single sample.
        
        Args:
            x: Dictionary containing feature names as keys and values as floats
            
        Returns:
            Predicted class label (0 for normal, 1 for attack)
        """
        if not self.is_trained:
            print("⚠️ WARNING: Classifier not trained, returning default prediction")
            return 0
        
        try:
            # Convert dict to array
            x_array = np.array([[x[fname] for fname in self.feature_names]])
            
            
            # Make prediction
            prediction = self.classifier.predict(x_array)[0]
            return int(prediction)
            
        except Exception as e:
            print(f"❌ ERROR: Prediction failed: {e}")
            print(f"├─ Classifier trained: {self.is_trained}")
            print(f"├─ Feature names: {self.feature_names}")
            print(f"└─ Input keys: {list(x.keys())}")
            return 0  # Default to normal on error

    def predict_proba(self, x: Dict[str, float]) -> Dict[Any, float]:
        """
        Get prediction probabilities for each class.
        
        Args:
            x: Dictionary containing feature names as keys and values as floats
            
        Returns:
            Dictionary with class labels as keys and probabilities as values
            Example: {0: 0.3, 1: 0.7} means 30% normal, 70% attack
        """
        if not self.is_trained:
            print("⚠️ WARNING: Classifier not trained, returning default probabilities")
            return {0: 0.9, 1: 0.1}
        
        try:
            # Convert dict to array
            x_array = np.array([[x[fname] for fname in self.feature_names]])
            
            # Scale features
            # x_scaled = self.scaler.transform(x_array) # REMOVED: Use scaler from DataPreprocessor
            
            # Get probabilities
            probabilities = self.classifier.predict_proba(x_array)[0]
            
            # Convert to dict format
            classes = self.classifier.classes_
            result = {}
            for i, prob in enumerate(probabilities):
                result[classes[i]] = float(prob)
            
            # Ensure both classes are present
            if 0 not in result:
                result[0] = 0.0
            if 1 not in result:
                result[1] = 0.0
                
            return result
            
        except Exception as e:
            print(f"❌ ERROR: Probability prediction failed: {e}")
            print(f"├─ Classifier trained: {self.is_trained}")
            if hasattr(self.classifier, 'classes_'):
                print(f"├─ Classifier classes: {list(self.classifier.classes_)}")
            print(f"└─ Feature keys: {list(x.keys())}")
            return {0: 0.9, 1: 0.1}  # Default to normal on error

    def step(self, x_raw: Dict[str, float], x_reduced: Dict[str, float], y_true=None):
        """
        Perform one step of incremental learning.
        
        Args:
            x_raw: Raw feature dictionary (for compatibility)
            x_reduced: Preprocessed/selected features (used for prediction)
            y_true: True label (None if not available)
            
        Returns:
            y_pred: The predicted class label
        """
        # Make prediction
        y_pred = self.predict(x_reduced)
        
        # Incremental learning if we have the true label
        if y_true is not None:
            # Learn from this single sample
            self.fit([x_reduced], [y_true])
            
            # Track performance
            self.predictions.append(y_pred)
            self.true_labels.append(y_true)
        
        return y_pred

    def get_f1_score(self) -> float:
        """
        Get current F1 score based on recent predictions.
        
        Returns:
            F1 score (0.0 if insufficient data)
        """
        if len(self.predictions) < 10 or len(self.true_labels) < 10:
            return 0.0
        
        try:
            return f1_score(list(self.true_labels), list(self.predictions))
        except:
            return 0.0

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance approximation for SGD classifier.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained or not hasattr(self.classifier, 'coef_'):
            return {}
        
        try:
            # Use absolute coefficients as importance approximation
            coefficients = np.abs(self.classifier.coef_[0])
            
            # Normalize to sum to 1
            if np.sum(coefficients) > 0:
                coefficients = coefficients / np.sum(coefficients)
            
            return {name: float(coef) for name, coef in zip(self.feature_names, coefficients)}
        except:
            return {}

    def save(self, path: str):
        """
        Save the entire model state to disk.
        
        Args:
            path: File path where to save the model
        """
        save_data = {
            'cfg': self.cfg,
            'classifier': self.classifier,
            'base_classifier': self.base_classifier,
            'calibrated_classifier': self.calibrated_classifier,
            'enable_calibration': self.enable_calibration,
            'calibration_method': self.calibration_method,
            'needs_calibration': self.needs_calibration,
            'calibration_threshold': self.calibration_threshold,


            'is_trained': self.is_trained,
            'classes_seen': self.classes_seen,
            'sample_count': self.sample_count,
            'feature_names': self.feature_names,
            'class_counts': self.class_counts
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)

    @classmethod
    def load(cls, path: str):
        """
        Load a previously saved model from disk.
        
        Args:
            path: File path where the model was saved
            
        Returns:
            SklearnWrapper: Fully restored model ready for incremental learning
        """
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        # Create new instance
        wrapper = cls(save_data['cfg'])
        
        # Restore state
        wrapper.classifier = save_data['classifier']
        wrapper.base_classifier = save_data.get('base_classifier', save_data['classifier'])
        wrapper.calibrated_classifier = save_data.get('calibrated_classifier', None)
        wrapper.enable_calibration = save_data.get('enable_calibration', True)
        wrapper.calibration_method = save_data.get('calibration_method', 'sigmoid')
        wrapper.needs_calibration = save_data.get('needs_calibration', True)
        wrapper.calibration_threshold = save_data.get('calibration_threshold', 1000)

        wrapper.is_trained = save_data['is_trained']
        wrapper.classes_seen = save_data.get('classes_seen', set())
        wrapper.sample_count = save_data.get('sample_count', 0)
        wrapper.feature_names = save_data.get('feature_names', [])
        wrapper.class_counts = save_data.get('class_counts', {0: 0, 1: 0})
        
        return wrapper

    @property
    def metric(self):
        """
        Compatibility property that mimics online learning metric interface.
        
        Returns:
            Mock metric object with get() method
        """
        class MockMetric:
            def __init__(self, wrapper):
                self.wrapper = wrapper
            
            def get(self):
                return self.wrapper.get_f1_score()
        
        return MockMetric(self)
