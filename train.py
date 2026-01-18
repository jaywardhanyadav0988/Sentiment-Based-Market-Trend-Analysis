"""
Training Script for Sentiment Analysis Models
"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.utils.data_loader import load_data, prepare_data
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.models.sentiment_analyzer import SentimentAnalyzer
from src.evaluation.metrics import ModelEvaluator


def main():
    parser = argparse.ArgumentParser(description='Train sentiment analysis models')
    parser.add_argument('--data_path', type=str, default='data/stock_tweets.csv',
                       help='Path to CSV data file')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Sample size (for faster testing)')
    parser.add_argument('--use_bert', action='store_true',
                       help='Use BERT model (requires transformers library)')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Output directory for results')
    parser.add_argument('--preprocess', action='store_true',
                       help='Apply text preprocessing')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("SENTIMENT-BASED MARKET TREND ANALYSIS - TRAINING")
    print("="*80)
    
    # Load data
    print("\n1. Loading data...")
    df = load_data(args.data_path, sample_size=args.sample_size)
    df = prepare_data(df, text_column='Tweet')
    
    # Preprocessing
    if args.preprocess:
        print("\n2. Preprocessing text...")
        preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
        df['processed_text'] = preprocessor.preprocess_batch(df['Tweet'])
        text_column = 'processed_text'
    else:
        text_column = 'Tweet'
    
    # Initialize sentiment analyzer
    print("\n3. Initializing sentiment analyzers...")
    analyzer = SentimentAnalyzer(use_bert=args.use_bert)
    
    # Determine which models to use
    models_to_use = ['vader', 'textblob']
    if args.use_bert and analyzer.use_bert:
        models_to_use.append('bert')
    
    print(f"Using models: {', '.join(models_to_use)}")
    
    # Analyze sentiment
    print("\n4. Analyzing sentiment...")
    print("This may take a while for large datasets...")
    
    # Sample for faster processing if dataset is very large
    if len(df) > 10000:
        print(f"Dataset is large ({len(df)} rows). Processing first 10000 rows for demonstration...")
        df_sample = df.head(10000).copy()
    else:
        df_sample = df.copy()
    
    sentiment_results = analyzer.analyze_batch(df_sample[text_column], models=models_to_use)
    
    # Combine with original data
    df_results = pd.concat([df_sample.reset_index(drop=True), 
                           sentiment_results.reset_index(drop=True)], axis=1)
    
    # Save results
    output_file = os.path.join(args.output_dir, 'sentiment_results.csv')
    df_results.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Generate summary statistics
    print("\n5. Generating summary statistics...")
    
    summary_stats = []
    
    for model in models_to_use:
        if model == 'vader':
            label_col = 'vader_label'
        elif model == 'textblob':
            label_col = 'textblob_label'
        elif model == 'bert':
            label_col = 'bert_label'
        else:
            continue
        
        if label_col in df_results.columns:
            label_counts = df_results[label_col].value_counts()
            summary_stats.append({
                'Model': model.upper(),
                'Positive': label_counts.get('positive', 0),
                'Negative': label_counts.get('negative', 0),
                'Neutral': label_counts.get('neutral', 0),
                'Total': len(df_results)
            })
    
    df_summary = pd.DataFrame(summary_stats)
    print("\nSentiment Distribution:")
    print(df_summary.to_string(index=False))
    
    # Save summary
    summary_file = os.path.join(args.output_dir, 'summary_statistics.csv')
    df_summary.to_csv(summary_file, index=False)
    
    # Stock-wise analysis
    if 'Stock Name' in df_results.columns:
        print("\n6. Stock-wise sentiment analysis...")
        stock_sentiment = []
        
        for stock in df_results['Stock Name'].unique():
            stock_df = df_results[df_results['Stock Name'] == stock]
            
            stock_data = {'Stock': stock}
            
            for model in models_to_use:
                if model == 'vader':
                    label_col = 'vader_label'
                elif model == 'textblob':
                    label_col = 'textblob_label'
                elif model == 'bert':
                    label_col = 'bert_label'
                else:
                    continue
                
                if label_col in stock_df.columns:
                    label_counts = stock_df[label_col].value_counts()
                    stock_data[f'{model}_positive'] = label_counts.get('positive', 0)
                    stock_data[f'{model}_negative'] = label_counts.get('negative', 0)
                    stock_data[f'{model}_neutral'] = label_counts.get('neutral', 0)
                    stock_data[f'{model}_total'] = len(stock_df)
            
            stock_sentiment.append(stock_data)
        
        df_stock = pd.DataFrame(stock_sentiment)
        stock_file = os.path.join(args.output_dir, 'stock_sentiment.csv')
        df_stock.to_csv(stock_file, index=False)
        print(f"Stock-wise analysis saved to {stock_file}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nResults saved in: {args.output_dir}")
    print(f"  - sentiment_results.csv: Full results with predictions")
    print(f"  - summary_statistics.csv: Overall statistics")
    if 'Stock Name' in df_results.columns:
        print(f"  - stock_sentiment.csv: Stock-wise analysis")


if __name__ == "__main__":
    main()
