"""
DATA PREPROCESSING MODULE FOR MIXED DATA TYPES
==============================================

This module handles preprocessing of network traffic data that contains both:
1. Numerical features (e.g., packet counts, byte counts, durations)  
2. Categorical features (e.g., protocol types, service names, connection states)

KEY COMPONENTS:
--------------
1. **DataPreprocessor**: Main class that handles all preprocessing operations
2. **Automatic Type Detection**: Identifies categorical vs numerical columns
3. **Label Encoding Strategy**: Efficient encoding for all categorical variables
4. **Consistent Pipeline**: Ensures same preprocessing during training and inference

ENCODING STRATEGY:
------------------
**Label Encoding**: Maps categories to integers  
   - Best for: All categorical features (compact and efficient)
   - Example: protocol="tcp" → 0, service="http" → 15, flag="SF" → 2
   - Benefits: Compact representation, no partial selection issues, memory efficient
   - Unknown handling: New categories → -1 (graceful degradation)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder
import pickle

# Import our robust incremental scaler
from modules.incremental_scaler import RobustIncrementalScaler


@dataclass 
class PreprocessingConfig:
    """
    Configuration for data preprocessing pipeline.
    
    Parameters:
    -----------
    categorical_encoding : str
        Strategy for encoding categorical variables: 'label' only
        - 'label': Label encode all categoricals (compact and efficient)
        
    handle_unknown : str
        How to handle unknown categories during inference: 'ignore' or 'error'
        - 'ignore': Unknown categories get encoded as -1 (recommended for production)
        - 'error': Raise error if unknown category encountered
        
    scale_features : bool
        Whether to apply robust scaling to numerical features
        Uses RobustIncrementalScaler (adaptive + outlier-resistant)
        Recommended: True for neural networks and attack detection
        
    scaler_alpha : float
        Learning rate for incremental scaler adaptation (0 < alpha < 1)
        Lower values = more stable, higher values = faster adaptation
        Default: 0.01 (conservative adaptation)
    """
    categorical_encoding: str = 'label'      # Only label encoding supported
    handle_unknown: str = 'ignore'           # Production-safe default
    scale_features: bool = True              # Whether to scale numerical features (always robust)
    scaler_alpha: float = 0.01               # Conservative adaptation rate


class DataPreprocessor:
    """
    Comprehensive data preprocessor for mixed numerical/categorical data.
    
    This class automatically:
    1. Detects column types (numerical vs categorical)
    2. Applies appropriate encoding to categorical features
    3. Scales numerical features (optional)
    4. Maintains consistent feature ordering
    5. Handles unknown categories during inference
    6. Supports incremental updates for streaming data
    
    WORKFLOW:
    --------
    Training Phase:
    1. fit() - Learn encoders and scalers from training data
    2. transform() - Apply learned transformations
    
    Inference Phase:  
    1. transform() - Apply previously learned transformations to new data
    2. Handles unknown categories gracefully
    
    Streaming Phase:
    1. update_with_new_data() - Incrementally update encoders with new categories
    2. check_dimension_changes() - Detect if model dimensions need updating
    """
    
    def __init__(self, config: PreprocessingConfig = None, preprocessing_config: Dict = None):
        """
        Initialize the data preprocessor.
        
        Args:
            config: Configuration object with preprocessing parameters
            preprocessing_config: Dictionary with preprocessing settings (log transform, etc.)
        """
        self.config = config or PreprocessingConfig()
        self.preprocessing_config = preprocessing_config or {}
        
        # Storage for fitted encoders and scalers
        self.numerical_features: List[str] = []
        self.categorical_features: List[str] = []
        
        # Fitted transformers
        self.scaler: Optional[RobustIncrementalScaler] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        
        # Feature name mappings for consistency
        self.output_feature_names: List[str] = []
        self.is_fitted = False
        
        # Track original dimensions for streaming updates
        self._initial_categorical_counts: Dict[str, int] = {}
        
        # Log transformation configuration from config
        self.log_transform_features: List[str] = []
        self.apply_log_transform = self.preprocessing_config.get('apply_log_transform', True)
        self.extreme_range_threshold = self.preprocessing_config.get('extreme_range_threshold', 1e4)

    def _detect_feature_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Automatically detect numerical vs categorical features.
        
        Heuristics used:
        1. dtype 'object' or 'category' → categorical
        2. dtype numeric but few unique values → potentially categorical  
        3. dtype numeric with many unique values → numerical
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Tuple of (numerical_features, categorical_features)
        """
        numerical = []
        categorical = []
        
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                # String/object columns are definitely categorical
                categorical.append(col)
            elif df[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                # Numeric columns - check cardinality to decide
                unique_vals = df[col].nunique()
                total_vals = len(df[col])
                
                # If very few unique values relative to total, might be categorical
                if unique_vals <= 20 and unique_vals / total_vals < 0.05:
                    # Low cardinality numeric might be categorical (e.g., 0/1 flags)
                    # But let's be conservative and treat as numerical unless obvious
                    if unique_vals <= 5:
                        categorical.append(col)
                    else:
                        numerical.append(col)
                else:
                    numerical.append(col)
            else:
                # Fallback: treat unknown types as numerical
                numerical.append(col)
                
        return numerical, categorical

    def _detect_extreme_ranges(self, df: pd.DataFrame):
        """Detect features that need log transformation due to extreme ranges."""
        self.log_transform_features = []
        
        if not self.apply_log_transform:
            print("📊 Log transformation DISABLED by configuration")
            return
        
        for col in self.numerical_features:
            if col in df.columns:
                values = df[col].values
                non_zero_values = values[values > 0]  # Only positive values for log transform
                
                if len(non_zero_values) > 0:
                    value_range = np.max(non_zero_values) - np.min(non_zero_values)
                    if value_range > self.extreme_range_threshold:
                        self.log_transform_features.append(col)
                        print(f"📊 Will log-transform '{col}' (range: {value_range:,.2f})")

    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit all encoders and scalers on training data.
        
        This method:
        1. Detects feature types automatically
        2. Determines encoding strategy for categoricals
        3. Fits appropriate encoders and scalers
        4. Builds output feature name mapping
        
        Args:
            df: Training DataFrame
            
        Returns:
            self (for method chaining)
        """
        # Step 1: Detect feature types
        self.numerical_features, self.categorical_features = self._detect_feature_types(df)
        
        self._detect_extreme_ranges(df)
        
        # Step 2: Determine encoding strategy for categoricals
        # Since we only use label encoding, this step is simplified
        self.label_features = self.categorical_features
        
        df_transformed = df.copy()
        for col in self.log_transform_features:
            if col in df_transformed.columns:
                # Add small constant to avoid log(0), then apply log
                df_transformed[col] = np.log(df_transformed[col] + 1e-6)
                print(f"✓ Log-transformed '{col}': range [{df_transformed[col].min():.3f}, {df_transformed[col].max():.3f}]")
        
        # Step 3: Fit numerical feature scaler (on transformed data)
        if self.config.scale_features and self.numerical_features:
            self.scaler = RobustIncrementalScaler(alpha=self.config.scaler_alpha)
            # Use transformed data for fitting scaler
            numerical_data = df_transformed[self.numerical_features].values
            self.scaler.fit(numerical_data, feature_names=self.numerical_features)
            print("✓ Using RobustIncrementalScaler on log-transformed data")
        else:
            print("✓ Feature scaling disabled")
        
        # Step 4: Fit categorical encoders
        for col in self.label_features:
            encoder = LabelEncoder()
            encoder.fit(df[col])
            self.label_encoders[col] = encoder
        
        # Step 5: Build output feature names
        self._build_output_feature_names()
        
        self.is_fitted = True
        return self

    def _build_output_feature_names(self):
        """
        Build the list of output feature names after all transformations.
        
        This ensures consistent feature ordering between training and inference.
        """
        self.output_feature_names = []
        
        # Add numerical features (unchanged names)
        self.output_feature_names.extend(self.numerical_features)
        
        # Add label encoded features  
        for col in self.label_features:
            self.output_feature_names.append(f"{col}_encoded")

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted encoders and scalers.
        
        This method applies all learned transformations consistently:
        1. Apply log transformation to extreme features
        2. Scale numerical features (if configured)
        3. Label encode designated categorical features
        4. Concatenate all features in consistent order
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed feature matrix as numpy array
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform. Call fit() first.")
        
        # Step 1: Apply log transformation to extreme features
        df_transformed = df.copy()
        for col in self.log_transform_features:
            if col in df_transformed.columns:
                df_transformed[col] = np.log(df_transformed[col] + 1e-6)
        
        feature_arrays = []
        
        # Step 2: Transform numerical features (using log-transformed data)
        if self.numerical_features:
            if self.config.scale_features and self.scaler is not None:
                numerical_data = self.scaler.transform(df_transformed[self.numerical_features].values)
            else:
                numerical_data = df_transformed[self.numerical_features].values
            feature_arrays.append(numerical_data)
        
        # Step 3: Transform label encoded categoricals (using original data)
        for col in self.label_features:
            if col in df.columns:  # Use original df for categorical data
                # Handle unknown categories
                if self.config.handle_unknown == 'ignore':
                    # Map unknown categories to -1
                    encoded = []
                    for val in df[col]:
                        try:
                            encoded.append(self.label_encoders[col].transform([val])[0])
                        except ValueError:
                            encoded.append(-1)  # Unknown category
                    encoded = np.array(encoded, dtype=np.float32).reshape(-1, 1)
                else:
                    # Let encoder raise error for unknown categories
                    encoded = self.label_encoders[col].transform(df[col]).reshape(-1, 1)
                feature_arrays.append(encoded)
            else:
                # Handle missing column
                encoded = np.full((len(df), 1), -1, dtype=np.float32)
                feature_arrays.append(encoded)
        
        # Concatenate all features
        if feature_arrays:
            return np.concatenate(feature_arrays, axis=1)
        else:
            return np.empty((len(df), 0), dtype=np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Convenience method to fit and transform in one step.
        
        Args:
            df: Training DataFrame
            
        Returns:
            Transformed feature matrix
        """
        return self.fit(df).transform(df)

    def transform_single(self, sample: Dict[str, Any]) -> np.ndarray:
        """
        Transform a single sample (dictionary) to feature vector.
        
        This is useful for online/streaming scenarios where you receive
        one sample at a time as a dictionary.
        
        Args:
            sample: Dictionary with feature names as keys
            
        Returns:
            Transformed feature vector as 1D numpy array
        """
        # Apply log transformation to extreme features first
        sample_transformed = sample.copy()
        for col in self.log_transform_features:
            if col in sample_transformed:
                sample_transformed[col] = np.log(sample_transformed[col] + 1e-6)
        
        # Convert to DataFrame (use original sample for categorical, transformed for numerical)
        df_original = pd.DataFrame([sample])  # For categorical features
        df_transformed = pd.DataFrame([sample_transformed])  # For numerical features
        
        # Use the main transform method but with mixed data
        feature_arrays = []
        
        # Transform numerical features (using log-transformed data)
        if self.numerical_features:
            if self.config.scale_features and self.scaler is not None:
                numerical_data = self.scaler.transform(df_transformed[self.numerical_features].values)
            else:
                numerical_data = df_transformed[self.numerical_features].values
            feature_arrays.append(numerical_data)
        
        # Transform categorical features (using original data)
        for col in self.label_features:
            if col in df_original.columns:
                if self.config.handle_unknown == 'ignore':
                    encoded = []
                    for val in df_original[col]:
                        try:
                            encoded.append(self.label_encoders[col].transform([val])[0])
                        except ValueError:
                            encoded.append(-1)
                    encoded = np.array(encoded, dtype=np.float32).reshape(-1, 1)
                else:
                    encoded = self.label_encoders[col].transform(df_original[col]).reshape(-1, 1)
                feature_arrays.append(encoded)
            else:
                encoded = np.full((1, 1), -1, dtype=np.float32)
                feature_arrays.append(encoded)
        
        # Concatenate and return first row
        if feature_arrays:
            result = np.concatenate(feature_arrays, axis=1)
            return result[0]  # Return first (and only) row
        else:
            return np.empty((0,), dtype=np.float32)

    def get_feature_names(self) -> List[str]:
        """
        Get names of output features after transformation.
        
        Returns:
            List of feature names in the order they appear in transformed arrays
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        return self.output_feature_names.copy()

    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about feature transformations.
        
        Returns:
            Dictionary with preprocessing information for debugging/analysis
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
            
        return {
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'label_features': self.label_features,
            'output_feature_names': self.output_feature_names,
            'total_output_features': len(self.output_feature_names),
            'config': self.config
        }

    def save(self, path: str):
        """
        Save the fitted preprocessor to disk.
        
        Args:
            path: File path to save the preprocessor
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor")
            
        save_data = {
            'config': self.config,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'label_features': self.label_features,
            'output_feature_names': self.output_feature_names,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'is_fitted': self.is_fitted,
            'log_transform_features': self.log_transform_features,
            'extreme_range_threshold': self.extreme_range_threshold
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)

    @classmethod
    def load(cls, path: str) -> 'DataPreprocessor':
        """
        Load a previously saved preprocessor.
        
        Args:
            path: File path to load from
            
        Returns:
            Loaded DataPreprocessor instance
        """
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        # Create new instance
        preprocessor = cls(save_data['config'])
        
        # Restore all attributes
        preprocessor.numerical_features = save_data['numerical_features']
        preprocessor.categorical_features = save_data['categorical_features']
        preprocessor.label_features = save_data['label_features']
        preprocessor.output_feature_names = save_data['output_feature_names']
        preprocessor.scaler = save_data['scaler']
        preprocessor.label_encoders = save_data['label_encoders']
        preprocessor.is_fitted = save_data['is_fitted']
        
        preprocessor.log_transform_features = save_data.get('log_transform_features', [])
        preprocessor.extreme_range_threshold = save_data.get('extreme_range_threshold', 10000.0)
        
        return preprocessor

    def update_with_new_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Update encoders incrementally with new streaming data.
        
        This method:
        1. Identifies new categorical values not seen during initial training
        2. Updates label encoders to include these new categories
        3. Tracks dimensional changes that might require model updates
        
        Args:
            df: New streaming data DataFrame
            
        Returns:
            Dictionary with update information including dimension changes
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before incremental updates")
        
        update_info = {
            'new_categories_found': False,
            'dimension_changes': {},
            'updated_encoders': []
        }
        
        # Check for new categorical values
        for col in self.categorical_features:
            if col in df.columns:
                # Get current categories in encoder
                current_categories = set(self.label_encoders[col].classes_)
                
                # Get new categories in streaming data
                new_categories = set(df[col].dropna().astype(str)) - current_categories
                
                if new_categories:
                    update_info['new_categories_found'] = True
                    update_info['updated_encoders'].append(col)
                    
                    # Create updated encoder with new categories
                    old_categories = list(current_categories)
                    all_categories = old_categories + list(new_categories)
                    
                    # Refit encoder with expanded categories
                    new_encoder = LabelEncoder()
                    new_encoder.fit(all_categories)
                    
                    # Track dimension change
                    old_dim = len(current_categories)
                    new_dim = len(all_categories)
                    update_info['dimension_changes'][col] = {
                        'old_count': old_dim,
                        'new_count': new_dim,
                        'added_categories': list(new_categories)
                    }
                    
                    # Update the encoder
                    self.label_encoders[col] = new_encoder
        
        return update_info

    def get_dimension_info(self) -> Dict[str, Any]:
        """
        Get current dimension information for model compatibility checks.
        
        Returns:
            Dictionary with current feature dimensions
        """
        return {
            'total_features': len(self.output_feature_names),
            'numerical_features': len(self.numerical_features),
            'categorical_features': len(self.categorical_features),
            'categorical_dimensions': {
                col: len(self.label_encoders[col].classes_) 
                for col in self.categorical_features if col in self.label_encoders
            }
        }

    def partial_fit_scaler(self, df: pd.DataFrame):
        """
        Update scaler with new numerical data (for streaming scenarios).
        
        This method now supports true incremental scaling for adaptive scalers.
        
        Args:
            df: New data to update scaler with
        """
        if self.config.scale_features and self.numerical_features and self.scaler is not None:
            if len(self.numerical_features) > 0:
                numerical_data = df[self.numerical_features].values
                self.scaler.partial_fit(numerical_data)
                print(f"✓ Scaler updated with {len(df)} new samples")
            else:
                print("⚠️ No numerical features to update scaler.")
    
    def get_scaler_stats(self) -> Dict[str, Any]:
        """Get current scaler statistics for debugging."""
        if not self.config.scale_features or self.scaler is None:
            return {"scaling_enabled": False}
        
        stats = self.scaler.get_stats()
        stats["scaler_type"] = "robust" # Always robust
        return stats


