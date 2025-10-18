import numpy as np
from dataclasses import dataclass
from typing import List, Sequence, Optional


@dataclass
class ORCConfig:
    """
    Configuration class that holds hyperparameters for the ORC feature selection algorithm.
    
    ORC uses reconstruction errors from an autoencoder to identify the most important features
    for detecting network attacks. Features that are harder to reconstruct (higher error)
    are considered more informative.
    
    All values should be provided from config.py - no defaults are set here.
    """
    beta: float                    # Smoothing factor for error averaging
    top_k: int                     # Number of top features to select  
    update_every: int              # Frequency of feature ranking updates
    lock_after_samples: int        # Lock feature selection after this many samples (incremental mode only)


class ORCFeatureSelector:
    """
    Online Reconstruction-based Feature Selector (ORC).
    
    This class implements an online feature selection algorithm that:
    1. Maintains an Exponential Moving Average (EMA) of reconstruction errors for each feature
    2. Selects the top-k features with highest reconstruction errors
    3. Updates the feature selection periodically based on recent data
    
    The intuition is that features with higher reconstruction errors from an autoencoder
    are more informative for detecting anomalies/attacks, as they capture patterns
    that are harder to compress/reconstruct.
    """
    
    def __init__(self, n_features: int, cfg: ORCConfig, feature_names: Optional[Sequence[str]] = None, training_mode: str = "incremental"):
        """
        Initialize the ORC feature selector.
        
        Args:
            n_features: Total number of input features
            cfg: Configuration object with hyperparameters
            feature_names: Optional list of feature names (if None, will create generic names)
            training_mode: "batch" or "incremental" - controls feature locking behavior
        """
        self.cfg = cfg  # Store configuration for later use
        
        # Initialize the ORC array to store EMA of reconstruction errors for each feature
        # Start with zeros - will be updated as we see reconstruction errors
        self.orc = np.zeros(n_features, dtype=np.float32)
        
        # Store feature names for interpretability
        # If not provided, create generic names like "f0", "f1", etc.
        self.feature_names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(n_features)]
        
        # Counter to track how many samples we've processed
        self._counter = 0  
        
        # Current mask of selected feature indices
        # Initially select all features, will be refined as we get reconstruction errors
        self._current_mask_idx = np.arange(n_features)  # [0, 1, 2, ..., n_features-1]
        
        # NEW: Stable feature locking functionality
        self.training_mode = training_mode  # "batch" or "incremental"
        self._is_locked = False  # Whether feature selection is locked
        self._samples_seen = 0   # Count of individual samples processed
        self._locked_features = None  # Locked feature indices (None until locked)
        
        # Log the training mode
        if training_mode == "batch":
            print(f"🔄 ORC initialized for BATCH training - feature locking DISABLED")
        else:
            print(f"🔄 ORC initialized for INCREMENTAL training - will lock after {cfg.lock_after_samples} samples")

    def update(self, abs_err: np.ndarray):
        """
        Update the ORC scores with new reconstruction errors from the autoencoder.
        
        This method implements Exponential Moving Average (EMA) to combine:
        - Historical reconstruction errors (weighted by beta)
        - New reconstruction errors (weighted by 1-beta)
        
        EMA formula: new_orc = beta * old_orc + (1-beta) * new_error
        
        Args:
            abs_err: Array of absolute reconstruction errors for each feature
                    Shape: (n_features,) where each value is |original - reconstructed|
        """
        # Increment sample counter for stable locking (only for incremental mode)
        if self.training_mode == "incremental":
            self._samples_seen += 1
        
        # If already locked, don't update ORC scores or recompute mask (only for incremental)
        if self.training_mode == "incremental" and self._is_locked:
            return
        
        beta = self.cfg.beta  # Get the EMA decay factor from configuration
        
        # Update ORC using exponential moving average
        # This gives more weight to recent errors while keeping some memory of past errors
        self.orc = beta * self.orc + (1 - beta) * abs_err
        
        # Increment the sample counter
        self._counter += 1
        
        # Check if we should lock the feature selection (only for incremental mode)
        if self.training_mode == "incremental" and self._samples_seen >= self.cfg.lock_after_samples:
            self._lock_feature_selection()
            return
        
        # Check if it's time to recompute the feature selection mask
        if self._counter % self.cfg.update_every == 0:
            self._recompute_mask()  # Update which features are selected

    def _lock_feature_selection(self):
        """
        Lock feature selection after sufficient samples have been processed.
        
        This method:
        1. Calculates final ORC scores based on accumulated reconstruction errors
        2. Selects the top-k features with highest scores
        3. Locks these features permanently for stable training
        4. Prints the selected features for interpretability
        """
        if self._is_locked:
            return  # Already locked
        
        print(f"🔒 LOCKING FEATURE SELECTION after {self._samples_seen} samples")
        
        # Perform final feature selection based on current ORC scores
        self._recompute_mask()
        
        # Lock the current selection
        self._locked_features = self._current_mask_idx.copy()
        self._is_locked = True
        
        # Print locked features for interpretability
        print(f"✅ LOCKED {len(self._locked_features)} features:")
        selected_names = self.get_mask_names()
        for i, (idx, name) in enumerate(zip(self._locked_features, selected_names)):
            orc_score = self.orc[idx]
            print(f"  {i+1:2d}. {name} (idx={idx}): orc_score = {orc_score:.6f}")
        
        print(f"🎯 These features will be used for ALL future training and predictions")
        print(f"💾 Feature lock will be saved to training_metadata.json")

    def _recompute_mask(self):
        """
        Recompute the feature selection mask by selecting top-k features with highest ORC scores.
        
        This method:
        1. Sorts all features by their ORC scores (reconstruction errors) in descending order
        2. Selects the top-k features with highest errors
        3. Keeps the indices sorted for stable feature ordering
        
        Features with higher reconstruction errors are considered more informative
        because they capture patterns that are difficult for the autoencoder to compress,
        often indicating anomalous or attack-related behavior.
        """
        # Get indices of features sorted by ORC score in descending order
        # np.argsort(-self.orc) gives indices that would sort -orc in ascending order
        # which is equivalent to sorting orc in descending order
        top_idx = np.argsort(-self.orc)[: self.cfg.top_k]  # Take only top-k indices
        
        # Sort the selected indices to maintain consistent feature ordering
        # This ensures that feature [i] always refers to the same original feature
        self._current_mask_idx = np.sort(top_idx)

    def get_mask_indices(self) -> np.ndarray:
        """
        Get the indices of currently selected features.
        
        Returns:
            Array of feature indices that are currently selected based on ORC scores.
            These indices can be used to extract the selected features from input data.
            Example: if returns [2, 5, 7], then features at positions 2, 5, and 7
            are the most informative according to reconstruction errors.
        """
        # If locked, return the locked features; otherwise return current mask
        if self._is_locked and self._locked_features is not None:
            return self._locked_features
        return self._current_mask_idx

    def is_locked(self) -> bool:
        """
        Check if feature selection is locked.
        
        Returns:
            True if feature selection is locked and stable, False otherwise.
        """
        return self._is_locked

    def get_lock_status(self) -> dict:
        """
        Get detailed information about the locking status.
        
        Returns:
            Dictionary containing lock status information.
        """
        return {
            "is_locked": self._is_locked,
            "samples_seen": self._samples_seen,
            "lock_threshold": self.cfg.lock_after_samples,
            "locked_features_count": len(self._locked_features) if self._locked_features is not None else 0,
            "total_features": len(self.feature_names),
            "locked_feature_names": [self.feature_names[i] for i in self._locked_features] if self._locked_features is not None else []
        }

    def set_locked_features(self, locked_indices: np.ndarray, samples_seen: int = None):
        """
        Manually set locked features (used when loading from training_metadata).
        
        Args:
            locked_indices: Array of feature indices to lock
            samples_seen: Number of samples seen (optional, for consistency)
        """
        self._locked_features = np.array(locked_indices)
        self._current_mask_idx = self._locked_features.copy()
        self._is_locked = True
        if samples_seen is not None:
            self._samples_seen = samples_seen
        
        print(f"🔓➡️🔒 Restored locked feature selection:")
        selected_names = self.get_mask_names()
        for i, (idx, name) in enumerate(zip(self._locked_features, selected_names)):
            print(f"  {i+1:2d}. {name} (idx={idx})")
        print(f"🎯 Using {len(self._locked_features)} locked features for all operations")

    def get_mask_names(self) -> List[str]:
        """
        Get the names of currently selected features.
        
        Returns:
            List of feature names corresponding to the selected indices.
            Useful for interpretability and understanding which features are important.
            Example: ["packet_size", "protocol_type", "connection_duration"]
        """
        # Use the selected indices to get corresponding feature names
        return [self.feature_names[i] for i in self._current_mask_idx]

    def force_recompute(self):
        """
        Force an immediate recomputation of the feature selection mask.
        
        This bypasses the normal update_every schedule and immediately updates
        which features are selected based on current ORC scores.
        Useful when you want to get the most up-to-date feature selection
        without waiting for the next scheduled update.
        """
        self._recompute_mask()

    def save(self, path: str):
        """
        Save the current state of the feature selector to disk.
        
        This saves:
        - ORC scores (reconstruction error averages) for all features
        - Current feature selection mask (which features are selected)
        - Feature names for interpretability
        - Locking state (NEW: for stable feature selection)
        - Training mode (NEW: batch vs incremental)
        
        Args:
            path: File path where to save the selector state
        """
        # Use numpy's savez to save multiple arrays in a single compressed file
        np.savez(path, 
                orc=self.orc,                          # EMA reconstruction errors
                mask=self._current_mask_idx,           # Currently selected feature indices  
                feature_names=self.feature_names,      # Feature names for interpretability
                is_locked=self._is_locked,             # NEW: Locking state
                samples_seen=self._samples_seen,       # NEW: Sample count
                locked_features=self._locked_features if self._locked_features is not None else np.array([]),  # NEW: Locked indices
                training_mode=self.training_mode       # NEW: Training mode
                )

    @classmethod
    def load(cls, path: str, cfg: ORCConfig, training_mode: str = None):
        """
        Load a previously saved feature selector from disk.
        
        This is a class method that creates a new ORCFeatureSelector instance
        and restores all saved components to their previous state, including
        locking state for stable feature selection.
        
        Args:
            path: File path where the selector was saved
            cfg: Configuration object (may be different from original)
            training_mode: Override training mode ("batch" or "incremental"). 
                          If None, will use saved mode or default to "incremental"
            
        Returns:
            ORCFeatureSelector: Fully restored selector ready for use
        """
        # Load the saved numpy arrays
        data = np.load(path, allow_pickle=True)
        
        # Determine training mode (with backward compatibility)
        if training_mode is not None:
            # Use provided override
            final_training_mode = training_mode
        elif 'training_mode' in data:
            # Use saved training mode
            final_training_mode = str(data['training_mode'])
        else:
            # Backward compatibility - default to incremental
            final_training_mode = "incremental"
            print("⚠️ No training mode found in saved data, defaulting to 'incremental'")
        
        # Create a new selector instance with the determined training mode
        obj = cls(n_features=len(data['orc']),        # Number of features from saved ORC array
                 cfg=cfg,                              # Use provided configuration
                 feature_names=data['feature_names'].tolist(),  # Restore feature names
                 training_mode=final_training_mode)    # Use determined training mode
        
        # Restore the saved state
        obj.orc = data['orc']                    # Restore ORC scores
        obj._current_mask_idx = data['mask']     # Restore selected feature indices
        
        # NEW: Restore locking state if available (only relevant for incremental mode)
        if 'is_locked' in data and final_training_mode == "incremental":
            obj._is_locked = bool(data['is_locked'])
            obj._samples_seen = int(data.get('samples_seen', 0))
            
            # Restore locked features if they exist
            locked_features = data.get('locked_features', np.array([]))
            if len(locked_features) > 0:
                obj._locked_features = locked_features
                obj._current_mask_idx = locked_features.copy()  # Use locked features as current mask
                
                print(f"🔓➡️🔒 Loaded locked feature selection:")
                selected_names = obj.get_mask_names()
                for i, (idx, name) in enumerate(zip(obj._locked_features, selected_names)):
                    print(f"  {i+1:2d}. {name} (idx={idx})")
                print(f"🎯 Using {len(obj._locked_features)} locked features")
        elif final_training_mode == "batch":
            print(f"🔄 Loaded for BATCH training - feature locking disabled")
        
        return obj
