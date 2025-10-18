import pandas as pd
import numpy as np
from typing import Iterator, Tuple, Dict, Optional, List, Any


def stream_csv_raw(path: str, feature_cols: List[str], label_col: Optional[str] = None,
                   chunksize: int = 1_000) -> Iterator[Tuple[Dict[str, Any], Optional[int]]]:
    """
    Stream a large CSV in chunks and yield one row at a time as dict with raw values.
    
    This version preserves original data types (strings, numbers) without forcing 
    conversion to float. Use this when you have mixed data types that need preprocessing.
    
    Args:
        path: Path to CSV file
        feature_cols: List of feature column names to include
        label_col: Optional label column name  
        chunksize: Number of rows to read at once
        
    Yields:
        Tuple of (feature_dict, label_value) where:
        - feature_dict contains raw values (strings, numbers) as they appear in CSV
        - label_value is converted to int (0/1) or None if no label column
    """
    # Include label column in reading if specified
    cols_to_read = feature_cols + ([label_col] if label_col else [])
    
    for chunk in pd.read_csv(path, usecols=cols_to_read, chunksize=chunksize):
        for _, row in chunk.iterrows():
            # Extract features as raw values (preserve original types)
            x = {col: row[col] for col in feature_cols}
            
            # Process label if available
            y = None
            if label_col is not None:
                val = row[label_col]
                # Customize mapping: e.g., 'BENIGN' -> 0, others -> 1
                if isinstance(val, str):
                    y = 0 if val.lower() in ("benign", "normal") else 1
                else:
                    y = int(val)
                    
            yield x, y


def stream_csv(path: str, feature_cols: List[str], label_col: Optional[str] = None,
               chunksize: int = 1_000) -> Iterator[Tuple[Dict[str, float], Optional[int]]]:
    """
    Stream a large CSV in chunks and yield one row at a time as dict.
    
    DEPRECATED: This function assumes all features are numerical.
    Use stream_csv_raw() for mixed data types with proper preprocessing.
    
    Args:
        path: Path to CSV file
        feature_cols: List of numerical feature column names
        label_col: Optional label column name
        chunksize: Number of rows to read at once
        
    Yields:
        Tuple of (feature_dict, label_value) where:
        - feature_dict contains float values for all features
        - label_value is 0/1 or None if no label column
    """
    for x_raw, y in stream_csv_raw(path, feature_cols, label_col, chunksize):
        # Convert all features to float (old behavior)
        try:
            x = {col: float(x_raw[col]) for col in feature_cols}
        except (ValueError, TypeError) as e:
            # If conversion fails, this data has categorical features
            raise ValueError(
                f"Cannot convert feature '{col}' with value '{x_raw[col]}' to float. "
                f"Your data contains categorical features. "
                f"Use stream_csv_raw() with DataPreprocessor instead."
            ) from e
            
        yield x, y


def detect_feature_types(csv_path: str, sample_size: int = 1000) -> Tuple[List[str], List[str], List[str]]:
    """
    Analyze a CSV file to detect column types automatically.
    
    This function reads a sample of the CSV and categorizes columns as:
    - Numerical: Can be converted to float and have many unique values
    - Categorical: String values or low-cardinality numeric
    - Label candidates: Likely target variables based on name patterns
    
    Args:
        csv_path: Path to the CSV file
        sample_size: Number of rows to analyze (more = better detection)
        
    Returns:
        Tuple of (numerical_features, categorical_features, label_candidates)
    """
    # Read a sample of the data
    sample_df = pd.read_csv(csv_path, nrows=sample_size)
    
    numerical = []
    categorical = []
    label_candidates = []
    
    for col in sample_df.columns:
        col_lower = col.lower()
        
        # Check if this looks like a label column
        if any(keyword in col_lower for keyword in ['label', 'target', 'class', 'attack', 'malware']):
            label_candidates.append(col)
            continue
        
        # Check data type and cardinality
        if sample_df[col].dtype == 'object':
            # String columns are categorical
            categorical.append(col)
        elif sample_df[col].dtype in ['int64', 'int32', 'float64', 'float32']:
            # Numeric columns - check cardinality
            unique_vals = sample_df[col].nunique()
            total_vals = len(sample_df[col])
            
            # Heuristic: if very few unique values, might be categorical
            if unique_vals <= 10 and unique_vals / total_vals < 0.1:
                categorical.append(col)
            else:
                numerical.append(col)
        else:
            # Unknown type - default to numerical
            numerical.append(col)
    
    return numerical, categorical, label_candidates


def create_feature_config(csv_path: str, exclude_cols: List[str] = None) -> Dict[str, Any]:
    """
    Create a configuration dictionary for features by analyzing CSV structure.
    
    This is a convenience function that analyzes your CSV and suggests
    how to configure the preprocessing pipeline.
    
    Args:
        csv_path: Path to your CSV file
        exclude_cols: Column names to exclude from analysis (e.g., ID columns)
        
    Returns:
        Dictionary with suggested configuration for your data
    """
    exclude_cols = exclude_cols or []
    
    # Detect feature types
    numerical, categorical, label_candidates = detect_feature_types(csv_path)
    
    # Remove excluded columns
    numerical = [col for col in numerical if col not in exclude_cols]
    categorical = [col for col in categorical if col not in exclude_cols]
    
    # All features (numerical + categorical)
    feature_cols = numerical + categorical
    
    # Suggest label column (first label candidate or ask user to specify)
    suggested_label = label_candidates[0] if label_candidates else None
    
    config = {
        'csv_path': csv_path,
        'feature_cols': feature_cols,
        'numerical_features': numerical,
        'categorical_features': categorical,
        'label_col': suggested_label,
        'label_candidates': label_candidates,
        'total_features': len(feature_cols),
        'preprocessing_suggestions': {
            'categorical_encoding': 'label',  # Use label encoding for all categoricals
            'scale_features': True if numerical else False,
            'handle_unknown': 'ignore'  # Safe default for production
        }
    }
    
    return config


def print_data_summary(csv_path: str, max_examples: int = 5):
    """
    Print a comprehensive summary of the CSV data structure.
    
    This helps you understand your data before setting up preprocessing.
    
    Args:
        csv_path: Path to your CSV file
        max_examples: Maximum number of example values to show per column
    """
    print(f"=== DATA SUMMARY FOR {csv_path} ===\n")
    
    # Read sample for analysis
    sample_df = pd.read_csv(csv_path, nrows=1000)
    
    print(f"Dataset shape: {sample_df.shape[0]}+ rows × {sample_df.shape[1]} columns")
    print(f"(showing analysis of first 1000 rows)\n")
    
    # Detect feature types
    numerical, categorical, label_candidates = detect_feature_types(csv_path)
    
    print("COLUMN TYPE ANALYSIS:")
    print(f"├─ Numerical features: {len(numerical)}")
    print(f"├─ Categorical features: {len(categorical)}")
    print(f"└─ Label candidates: {len(label_candidates)}")
    print()
    
    # Show details for each column type
    if numerical:
        print("NUMERICAL FEATURES:")
        for col in numerical[:10]:  # Show first 10
            stats = sample_df[col].describe()
            print(f"├─ {col}: range [{stats['min']:.2f}, {stats['max']:.2f}], "
                  f"mean={stats['mean']:.2f}, unique={sample_df[col].nunique()}")
        if len(numerical) > 10:
            print(f"└─ ... and {len(numerical) - 10} more numerical features")
        print()
    
    if categorical:
        print("CATEGORICAL FEATURES:")
        for col in categorical[:10]:  # Show first 10
            unique_vals = sample_df[col].unique()[:max_examples]
            unique_count = sample_df[col].nunique()
            print(f"├─ {col}: {unique_count} unique values, "
                  f"examples: {list(unique_vals)}")
        if len(categorical) > 10:
            print(f"└─ ... and {len(categorical) - 10} more categorical features")
        print()
    
    if label_candidates:
        print("POTENTIAL LABEL COLUMNS:")
        for col in label_candidates:
            unique_vals = sample_df[col].unique()
            print(f"├─ {col}: values = {list(unique_vals)}")
        print()
    
    # Show first few rows as example
    print("SAMPLE DATA (first 3 rows):")
    print(sample_df.head(3).to_string())
    print()
    
    # Preprocessing recommendations
    config = create_feature_config(csv_path)
    print("RECOMMENDED CONFIGURATION:")
    print(f"├─ Feature columns: {config['total_features']} total")
    print(f"├─ Label column: '{config['label_col']}' (verify this is correct)")
    print(f"├─ Categorical encoding: {config['preprocessing_suggestions']['categorical_encoding']}")
    print(f"├─ Scale features: {config['preprocessing_suggestions']['scale_features']}")
    print(f"└─ Handle unknown: {config['preprocessing_suggestions']['handle_unknown']}")
    print()
