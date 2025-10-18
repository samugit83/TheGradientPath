"""
ROBUST INCREMENTAL SCALER FOR STREAMING DATA
=============================================

This module implements a robust incremental scaler that can adapt to changing
data distributions in streaming scenarios while being resistant to outliers.

Perfect for network attack detection where attacks are outliers that shouldn't
skew the scaling of normal traffic.
"""

import numpy as np
from typing import Optional, Dict, Any
import pickle


class RobustIncrementalScaler:
    """
    Incremental version of RobustScaler using median and IQR.
    
    More robust to outliers than standard scaling, which is important
    for network attack detection where attacks may be outliers.
    """
    
    def __init__(self, alpha: float = 0.01, min_samples: int = 100):
        """
        Initialize incremental robust scaler.
        
        Args:
            alpha: Learning rate for exponential moving average
            min_samples: Minimum samples before starting updates
        """
        self.alpha = alpha
        self.min_samples = min_samples
        
        # Statistics tracking
        self.median_ = None
        self.iqr_ = None
        self.scale_ = None
        self.n_samples_ = 0
        self.is_fitted = False
        
        # Keep recent samples for robust statistics
        self.recent_samples_ = []
        self.max_recent_samples_ = 1000
        
    def fit(self, X: np.ndarray, feature_names: Optional[list] = None):
        """Fit robust scaler on initial data."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # Calculate robust statistics
        self.median_ = np.median(X, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        self.iqr_ = q75 - q25
        
        # Use IQR as scale, fallback to 1 if IQR is 0
        self.scale_ = self.iqr_.copy()
        self.scale_[self.scale_ == 0] = 1.0
        
        self.n_samples_ = X.shape[0]
        self.recent_samples_ = X.tolist()[-self.max_recent_samples_:]
        self.is_fitted = True
        
        print(f"✓ RobustIncrementalScaler fitted on {self.n_samples_} samples")
        print(f"├─ Median range: [{np.min(self.median_):.3f}, {np.max(self.median_):.3f}]")
        print(f"└─ IQR range: [{np.min(self.iqr_):.3f}, {np.max(self.iqr_):.3f}]")
        
        return self
    
    def partial_fit(self, X: np.ndarray):
        """Incrementally update robust statistics."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before partial_fit")
            
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # Add to recent samples
        self.recent_samples_.extend(X.tolist())
        self.recent_samples_ = self.recent_samples_[-self.max_recent_samples_:]
        
        if self.n_samples_ < self.min_samples:
            return self
            
        # Recalculate robust statistics on recent samples
        recent_array = np.array(self.recent_samples_)
        
        batch_median = np.median(recent_array, axis=0)
        q75 = np.percentile(recent_array, 75, axis=0)
        q25 = np.percentile(recent_array, 25, axis=0)
        batch_iqr = q75 - q25
        
        # Update using exponential moving average
        old_median = self.median_.copy()
        self.median_ = (1 - self.alpha) * self.median_ + self.alpha * batch_median
        self.iqr_ = (1 - self.alpha) * self.iqr_ + self.alpha * batch_iqr
        self.scale_ = self.iqr_.copy()
        self.scale_[self.scale_ == 0] = 1.0
        
        self.n_samples_ += X.shape[0]
        
        # Check for significant changes
        median_change = np.mean(np.abs(self.median_ - old_median))
        if median_change > 0.1:
            print(f"📊 RobustScaler adapted: median change = {median_change:.3f}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform using robust scaling."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
            
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        return (X - self.median_) / self.scale_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current scaler statistics for debugging."""
        if not self.is_fitted:
            return {"fitted": False}
            
        return {
            "fitted": True,
            "n_samples": self.n_samples_,
            "n_features": len(self.median_),
            "median_range": [float(np.min(self.median_)), float(np.max(self.median_))],
            "iqr_range": [float(np.min(self.iqr_)), float(np.max(self.iqr_))],
            "alpha": self.alpha,
            "recent_samples_count": len(self.recent_samples_)
        }
    
    def save(self, path: str):
        """Save scaler state to file."""
        save_data = {
            'median_': self.median_,
            'iqr_': self.iqr_,
            'scale_': self.scale_,
            'n_samples_': self.n_samples_,
            'alpha': self.alpha,
            'min_samples': self.min_samples,
            'recent_samples_': self.recent_samples_,
            'max_recent_samples_': self.max_recent_samples_,
            'is_fitted': self.is_fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
    
    @classmethod
    def load(cls, path: str):
        """Load scaler state from file."""
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        scaler = cls(alpha=save_data['alpha'], min_samples=save_data['min_samples'])
        scaler.median_ = save_data['median_']
        scaler.iqr_ = save_data['iqr_']
        scaler.scale_ = save_data['scale_']
        scaler.n_samples_ = save_data['n_samples_']
        scaler.recent_samples_ = save_data['recent_samples_']
        scaler.max_recent_samples_ = save_data['max_recent_samples_']
        scaler.is_fitted = save_data['is_fitted']
        
        return scaler 