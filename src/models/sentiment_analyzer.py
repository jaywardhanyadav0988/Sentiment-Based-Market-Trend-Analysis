"""
Sentiment Analysis Models
Implements VADER, TextBlob, and BERT-based sentiment analysis
"""

import pandas as pd
import numpy as np
import os  # <--- ADDED: To handle directory checking
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# VADER
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# TextBlob
from textblob import TextBlob

# BERT
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("Warning: transformers library not available. BERT model will not be used.")


class SentimentAnalyzer:
    """
    Unified sentiment analyzer supporting multiple models
    """
    
    def __init__(self, use_bert: bool = True):
        """
        Initialize sentiment analyzers
        
        Args:
            use_bert: Whether to initialize BERT model (requires transformers library)
        """
        # Initialize VADER
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize BERT if available
        self.use_bert = use_bert and BERT_AVAILABLE
        self.bert_tokenizer = None
        self.bert_model = None
        self.device = None
        
        if self.use_bert:
            try:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # --- CHANGED: Caching Logic Start ---
                model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
                cache_dir = "models/bert_cache"  # Local folder to store the model
                
                # Check if we have the model saved locally
                if os.path.exists(cache_dir) and os.listdir(cache_dir):
                    print(f"Loading BERT model from local cache: {cache_dir}...")
                    self.bert_tokenizer = AutoTokenizer.from_pretrained(cache_dir)
                    self.bert_model = AutoModelForSequenceClassification.from_pretrained(cache_dir)
                else:
                    print(f"Downloading BERT model: {model_name}...")
                    self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.bert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                    
                    # Save locally for next time
                    print(f"Saving BERT model to local cache: {cache_dir}...")
                    os.makedirs(cache_dir, exist_ok=True)
                    self.bert_tokenizer.save_pretrained(cache_dir)
                    self.bert_model.save_pretrained(cache_dir)
                # --- CHANGED: Caching Logic End ---

                self.bert_model.to(self.device)
                self.bert_model.eval()
                print(f"BERT model loaded on {self.device}")
            except Exception as e:
                print(f"Warning: Could not load BERT model: {e}")
                self.use_bert = False
    
    def vader_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with sentiment scores
        """
        if pd.isna(text) or not isinstance(text, str):
            return {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0, 'label': 'neutral'}
        
        scores = self.vader_analyzer.polarity_scores(text)
        
        # Determine label
        if scores['compound'] >= 0.05:
            label = 'positive'
        elif scores['compound'] <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        
        scores['label'] = label
        return scores
    
    def textblob_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using TextBlob
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with sentiment scores
        """
        if pd.isna(text) or not isinstance(text, str):
            return {'polarity': 0.0, 'subjectivity': 0.0, 'label': 'neutral'}
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Determine label
        if polarity > 0:
            label = 'positive'
        elif polarity < 0:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'label': label
        }
    
    def bert_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using BERT (RoBERTa-based)
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with sentiment scores
        """
        if not self.use_bert or self.bert_model is None:
            return {'label': 'neutral', 'score': 0.0}
        
        if pd.isna(text) or not isinstance(text, str):
            return {'label': 'neutral', 'score': 0.0}
        
        try:
            # Tokenize and encode
            inputs = self.bert_tokenizer(text, return_tensors='pt', truncation=True, 
                                        max_length=512, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Map labels (0: negative, 1: neutral, 2: positive)
            labels = ['negative', 'neutral', 'positive']
            scores = predictions[0].cpu().numpy()
            
            # Get predicted label and score
            predicted_idx = np.argmax(scores)
            predicted_label = labels[predicted_idx]
            predicted_score = float(scores[predicted_idx])
            
            return {
                'label': predicted_label,
                'score': predicted_score,
                'negative': float(scores[0]),
                'neutral': float(scores[1]),
                'positive': float(scores[2])
            }
        except Exception as e:
            print(f"Error in BERT sentiment analysis: {e}")
            return {'label': 'neutral', 'score': 0.0}
    
    def analyze(self, text: str, models: List[str] = ['vader', 'textblob', 'bert']) -> Dict:
        """
        Analyze sentiment using multiple models
        
        Args:
            text: Input text string
            models: List of models to use ['vader', 'textblob', 'bert']
            
        Returns:
            Dictionary with results from all models
        """
        results = {}
        
        if 'vader' in models:
            results['vader'] = self.vader_sentiment(text)
        
        if 'textblob' in models:
            results['textblob'] = self.textblob_sentiment(text)
        
        if 'bert' in models and self.use_bert:
            results['bert'] = self.bert_sentiment(text)
        
        return results
    
    def analyze_batch(self, texts: pd.Series, models: List[str] = ['vader', 'textblob', 'bert']) -> pd.DataFrame:
        """
        Analyze sentiment for a batch of texts
        
        Args:
            texts: Pandas Series of text strings
            models: List of models to use
            
        Returns:
            DataFrame with sentiment scores for each model
        """
        results = []
        
        for idx, text in enumerate(texts):
            if idx % 1000 == 0:
                print(f"Processing {idx}/{len(texts)} texts...")
            
            analysis = self.analyze(text, models=models)
            row = {'text': text}
            
            # Extract scores
            if 'vader' in analysis:
                row['vader_compound'] = analysis['vader']['compound']
                row['vader_pos'] = analysis['vader']['pos']
                row['vader_neu'] = analysis['vader']['neu']
                row['vader_neg'] = analysis['vader']['neg']
                row['vader_label'] = analysis['vader']['label']
            
            if 'textblob' in analysis:
                row['textblob_polarity'] = analysis['textblob']['polarity']
                row['textblob_subjectivity'] = analysis['textblob']['subjectivity']
                row['textblob_label'] = analysis['textblob']['label']
            
            if 'bert' in analysis:
                row['bert_label'] = analysis['bert']['label']
                row['bert_score'] = analysis['bert']['score']
                row['bert_negative'] = analysis['bert'].get('negative', 0.0)
                row['bert_neutral'] = analysis['bert'].get('neutral', 0.0)
                row['bert_positive'] = analysis['bert'].get('positive', 0.0)
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def get_ensemble_label(self, text: str) -> Tuple[str, float]:
        """
        Get ensemble sentiment label from all models
        
        Args:
            text: Input text string
            
        Returns:
            Tuple of (label, confidence)
        """
        analysis = self.analyze(text)
        
        labels = []
        scores = []
        
        if 'vader' in analysis:
            vader_label = analysis['vader']['label']
            vader_score = abs(analysis['vader']['compound'])
            labels.append(vader_label)
            scores.append(vader_score)
        
        if 'textblob' in analysis:
            textblob_label = analysis['textblob']['label']
            textblob_score = abs(analysis['textblob']['polarity'])
            labels.append(textblob_label)
            scores.append(textblob_score)
        
        if 'bert' in analysis:
            bert_label = analysis['bert']['label']
            bert_score = analysis['bert']['score']
            labels.append(bert_label)
            scores.append(bert_score)
        
        if not labels:
            return ('neutral', 0.0)
        
        # Weighted voting
        positive_count = labels.count('positive')
        negative_count = labels.count('negative')
        neutral_count = labels.count('neutral')
        
        if positive_count > negative_count and positive_count > neutral_count:
            return ('positive', np.mean(scores))
        elif negative_count > positive_count and negative_count > neutral_count:
            return ('negative', np.mean(scores))
        else:
            return ('neutral', np.mean(scores))


if __name__ == "__main__":
    # Example usage
    analyzer = SentimentAnalyzer(use_bert=False)  # Set to True if transformers installed
    
    sample_text = "Tesla is revolutionizing the electric vehicle industry!"
    print("Sample text:", sample_text)
    print("\nVADER:", analyzer.vader_sentiment(sample_text))
    print("TextBlob:", analyzer.textblob_sentiment(sample_text))
    if analyzer.use_bert:
        print("BERT:", analyzer.bert_sentiment(sample_text))
    print("\nEnsemble:", analyzer.get_ensemble_label(sample_text))