"""
Data Loading Utilities
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from pathlib import Path


def load_data(file_path: str, sample_size: Optional[int] = None, 
              random_state: int = 42) -> pd.DataFrame:
    """
    Load CSV data file
    
    Args:
        file_path: Path to CSV file
        sample_size: If provided, sample this many rows randomly
        random_state: Random state for sampling
        
    Returns:
        DataFrame with loaded data
    """
    print(f"Loading data from {file_path}...")
    
    # Read CSV
    df = pd.read_csv(file_path)
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        print(f"Sampled {len(df)} rows")
    
    return df


def prepare_data(df: pd.DataFrame, text_column: str = 'Tweet') -> pd.DataFrame:
    """
    Prepare data for analysis
    
    Args:
        df: Input DataFrame
        text_column: Name of the text column
        
    Returns:
        Prepared DataFrame
    """
    # Create a copy
    df_prep = df.copy()
    
    # Ensure text column exists
    if text_column not in df_prep.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")
    
    # Remove rows with missing text
    initial_len = len(df_prep)
    df_prep = df_prep.dropna(subset=[text_column])
    df_prep = df_prep[df_prep[text_column].str.strip() != '']
    
    removed = initial_len - len(df_prep)
    if removed > 0:
        print(f"Removed {removed} rows with missing or empty text")
    
    # Reset index
    df_prep = df_prep.reset_index(drop=True)
    
    print(f"Final dataset size: {len(df_prep)} rows")
    
    return df_prep


def split_data(df: pd.DataFrame, test_size: float = 0.2, 
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets
    
    Args:
        df: Input DataFrame
        test_size: Proportion of data for testing
        random_state: Random state
        
    Returns:
        Tuple of (train_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    print(f"Train set: {len(train_df)} rows")
    print(f"Test set: {len(test_df)} rows")
    
    return train_df, test_df
