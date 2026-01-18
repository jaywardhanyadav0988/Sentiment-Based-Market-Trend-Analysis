"""
Inference Script for Sentiment Analysis
Quick script to analyze sentiment of new text
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.preprocessing.text_preprocessor import TextPreprocessor
from src.models.sentiment_analyzer import SentimentAnalyzer


def main():
    parser = argparse.ArgumentParser(description='Analyze sentiment of text')
    parser.add_argument('--text', type=str, required=True,
                       help='Text to analyze')
    parser.add_argument('--preprocess', action='store_true',
                       help='Apply text preprocessing')
    parser.add_argument('--use_bert', action='store_true',
                       help='Use BERT model')
    parser.add_argument('--models', nargs='+', 
                       default=['vader', 'textblob'],
                       choices=['vader', 'textblob', 'bert'],
                       help='Models to use')
    
    args = parser.parse_args()
    
    # Preprocess if requested
    text = args.text
    if args.preprocess:
        preprocessor = TextPreprocessor()
        text = preprocessor.preprocess(args.text)
        print(f"Original: {args.text}")
        print(f"Preprocessed: {text}\n")
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer(use_bert=args.use_bert)
    
    # Analyze
    print("Analyzing sentiment...\n")
    results = analyzer.analyze(text, models=args.models)
    
    # Display results
    print("="*60)
    print("SENTIMENT ANALYSIS RESULTS")
    print("="*60)
    
    for model_name, model_results in results.items():
        print(f"\n{model_name.upper()}:")
        print("-" * 40)
        
        if 'label' in model_results:
            label = model_results['label']
            emoji = {'positive': '🟢', 'negative': '🔴', 'neutral': '🟡'}.get(label, '⚪')
            print(f"Sentiment: {emoji} {label.capitalize()}")
        
        for key, value in model_results.items():
            if key != 'label' and isinstance(value, (int, float)):
                print(f"{key.capitalize()}: {value:.4f}")
    
    # Ensemble result
    ensemble_label, confidence = analyzer.get_ensemble_label(text)
    emoji = {'positive': '🟢', 'negative': '🔴', 'neutral': '🟡'}.get(ensemble_label, '⚪')
    print(f"\n{'='*60}")
    print(f"ENSEMBLE RESULT: {emoji} {ensemble_label.capitalize()} (confidence: {confidence:.4f})")
    print("="*60)


if __name__ == "__main__":
    main()
