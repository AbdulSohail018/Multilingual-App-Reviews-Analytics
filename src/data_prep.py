#!/usr/bin/env python3
"""
Data Preprocessing Script for Multilingual App Reviews Analytics

This script loads the raw CSV dataset, performs data cleaning and normalization,
and saves the processed data as Parquet format for efficient analysis.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import langid
from typing import Optional

def setup_paths():
    """Setup and validate data directory paths."""
    project_root = Path(__file__).parent.parent
    raw_data_path = project_root / "data" / "raw"
    processed_data_path = project_root / "data" / "processed"
    
    # Create directories if they don't exist
    processed_data_path.mkdir(parents=True, exist_ok=True)
    
    return raw_data_path, processed_data_path

def load_dataset(raw_data_path: Path, filename: str = "multilingual_mobile_app_reviews_2025.csv") -> pd.DataFrame:
    """Load the raw dataset from CSV file."""
    file_path = raw_data_path / filename
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {file_path}. "
            f"Please download the 'multilingual_mobile_app_reviews_2025.csv' file "
            f"from Kaggle and place it in the data/raw/ directory."
        )
    
    print(f"Loading dataset from {file_path}")
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        # Try alternative encodings if UTF-8 fails
        print("UTF-8 encoding failed, trying latin-1...")
        df = pd.read_csv(file_path, encoding='latin-1')
    
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df

def clean_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize data types."""
    df_clean = df.copy()
    
    # Convert string IDs to integers where possible
    for col in df_clean.columns:
        if 'id' in col.lower() and df_clean[col].dtype == 'object':
            try:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').astype('Int64')
                print(f"Converted {col} to integer type")
            except:
                print(f"Could not convert {col} to integer, keeping as string")
    
    # Convert ratings to float
    rating_columns = [col for col in df_clean.columns if 'rating' in col.lower() or 'score' in col.lower()]
    for col in rating_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            print(f"Converted {col} to numeric type")
    
    # Convert date columns
    date_columns = [col for col in df_clean.columns if 'date' in col.lower() or 'time' in col.lower()]
    for col in date_columns:
        if col in df_clean.columns:
            try:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                print(f"Converted {col} to datetime type")
            except:
                print(f"Could not convert {col} to datetime")
    
    return df_clean

def detect_languages(df: pd.DataFrame, text_column: str = 'review_text') -> pd.DataFrame:
    """Detect and impute missing languages using langid."""
    df_clean = df.copy()
    
    # Find language column (flexible naming)
    language_col = None
    for col in df_clean.columns:
        if 'lang' in col.lower():
            language_col = col
            break
    
    if language_col is None:
        print("No language column found, creating one...")
        language_col = 'language'
        df_clean[language_col] = None
    
    # Check if text column exists
    if text_column not in df_clean.columns:
        # Try to find a text column
        text_cols = [col for col in df_clean.columns if 'text' in col.lower() or 'review' in col.lower() or 'comment' in col.lower()]
        if text_cols:
            text_column = text_cols[0]
            print(f"Using {text_column} as text column for language detection")
        else:
            print("No text column found for language detection")
            return df_clean
    
    # Detect languages for missing values
    missing_lang_mask = df_clean[language_col].isnull() | (df_clean[language_col] == '')
    missing_count = missing_lang_mask.sum()
    
    if missing_count > 0:
        print(f"Detecting languages for {missing_count} reviews with missing language info...")
        
        for idx in df_clean[missing_lang_mask].index:
            text = str(df_clean.loc[idx, text_column])
            if text and text != 'nan' and len(text.strip()) > 0:
                try:
                    detected_lang, confidence = langid.classify(text)
                    if confidence > 0.5:  # Only use high-confidence predictions
                        df_clean.loc[idx, language_col] = detected_lang
                except:
                    continue
        
        detected_count = (~(df_clean[language_col].isnull() | (df_clean[language_col] == ''))).sum() - (len(df_clean) - missing_count)
        print(f"Successfully detected languages for {detected_count} reviews")
    
    return df_clean

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset."""
    df_clean = df.copy()
    
    print("\nHandling missing values...")
    
    # For numeric columns, fill with appropriate values
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            if 'rating' in col.lower() or 'score' in col.lower():
                # Fill rating missing values with median
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                print(f"Filled missing {col} values with median: {median_val}")
            else:
                # Fill other numeric columns with 0 or median based on context
                df_clean[col].fillna(0, inplace=True)
                print(f"Filled missing {col} values with 0")
    
    # For text columns, fill with appropriate defaults
    text_cols = df_clean.select_dtypes(include=['object']).columns
    for col in text_cols:
        if df_clean[col].isnull().sum() > 0:
            if 'text' in col.lower() or 'review' in col.lower() or 'comment' in col.lower():
                df_clean[col].fillna('', inplace=True)
                print(f"Filled missing {col} values with empty string")
            else:
                df_clean[col].fillna('unknown', inplace=True)
                print(f"Filled missing {col} values with 'unknown'")
    
    return df_clean

def deduplicate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate records based on review_id or similar unique identifier."""
    df_clean = df.copy()
    
    # Find potential ID columns for deduplication
    id_columns = [col for col in df_clean.columns if 'id' in col.lower()]
    
    if id_columns:
        id_col = id_columns[0]  # Use the first ID column found
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=[id_col], keep='first')
        final_count = len(df_clean)
        duplicates_removed = initial_count - final_count
        print(f"Removed {duplicates_removed} duplicate records based on {id_col}")
    else:
        # Fallback: remove exact duplicates across all columns
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates(keep='first')
        final_count = len(df_clean)
        duplicates_removed = initial_count - final_count
        print(f"Removed {duplicates_removed} exact duplicate records")
    
    return df_clean

def save_processed_data(df: pd.DataFrame, processed_data_path: Path, filename: str = "cleaned_reviews.parquet"):
    """Save the cleaned dataset as Parquet format."""
    output_path = processed_data_path / filename
    
    print(f"Saving cleaned dataset to {output_path}")
    df.to_parquet(output_path, index=False, engine='pyarrow')
    print(f"Saved {len(df)} rows to {output_path}")
    
    # Also save a CSV version for easy inspection
    csv_path = processed_data_path / filename.replace('.parquet', '.csv')
    df.to_csv(csv_path, index=False)
    print(f"Also saved CSV version to {csv_path}")

def print_data_summary(df: pd.DataFrame, title: str):
    """Print a summary of the dataset."""
    print(f"\n{title}")
    print("=" * len(title))
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

def main():
    """Main data preprocessing pipeline."""
    parser = argparse.ArgumentParser(description="Preprocess multilingual app reviews dataset")
    parser.add_argument("--input", "-i", type=str, default="multilingual_mobile_app_reviews_2025.csv",
                        help="Input CSV filename (default: multilingual_mobile_app_reviews_2025.csv)")
    parser.add_argument("--output", "-o", type=str, default="cleaned_reviews.parquet",
                        help="Output Parquet filename (default: cleaned_reviews.parquet)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    try:
        # Setup paths
        raw_data_path, processed_data_path = setup_paths()
        
        # Load dataset
        df = load_dataset(raw_data_path, args.input)
        
        if args.verbose:
            print_data_summary(df, "Original Dataset Summary")
        
        # Data cleaning pipeline
        print("\n" + "="*50)
        print("Starting data cleaning pipeline...")
        print("="*50)
        
        # Step 1: Clean data types
        print("\n1. Cleaning data types...")
        df = clean_data_types(df)
        
        # Step 2: Handle missing values
        print("\n2. Handling missing values...")
        df = handle_missing_values(df)
        
        # Step 3: Detect languages
        print("\n3. Detecting languages...")
        df = detect_languages(df)
        
        # Step 4: Deduplicate
        print("\n4. Removing duplicates...")
        df = deduplicate_data(df)
        
        if args.verbose:
            print_data_summary(df, "Cleaned Dataset Summary")
        
        # Save processed data
        print("\n5. Saving processed data...")
        save_processed_data(df, processed_data_path, args.output)
        
        print("\n" + "="*50)
        print("Data preprocessing completed successfully!")
        print("="*50)
        print(f"Final dataset shape: {df.shape}")
        print(f"Cleaned data saved to: data/processed/{args.output}")
        
    except Exception as e:
        print(f"Error during data preprocessing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()