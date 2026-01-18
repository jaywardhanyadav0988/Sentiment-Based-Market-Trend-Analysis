"""
Text Preprocessing Module
Handles tokenization, stopword removal, lemmatization, and normalization
"""

import re
import string
import pandas as pd
from typing import List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)


class TextPreprocessor:
    """
    Comprehensive text preprocessing class for social media text
    """
    
    def __init__(self, remove_stopwords: bool = True, lemmatize: bool = True):
        """
        Initialize the preprocessor
        
        Args:
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize words
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tweet_tokenizer = TweetTokenizer()
        
        # Add common social media stopwords
        self.stop_words.update(['rt', 'http', 'https', 'www', 'com', 'amp', 'quot'])
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text: lowercase, remove URLs, mentions, hashtags symbols
        
        Args:
            text: Input text string
            
        Returns:
            Normalized text string
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions but keep the username
        text = re.sub(r'@(\w+)', r'\1', text)
        
        # Remove hashtag symbol but keep the word
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove HTML entities
        text = re.sub(r'&amp;', 'and', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&quot;', '"', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using TweetTokenizer (better for social media)
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        # Use TweetTokenizer for better handling of social media text
        tokens = self.tweet_tokenizer.tokenize(text)
        
        # Remove punctuation-only tokens
        tokens = [token for token in tokens if token not in string.punctuation]
        
        return tokens
    
    def remove_stopwords_func(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of tokens without stopwords
        """
        if not self.remove_stopwords:
            return tokens
        
        return [token for token in tokens if token not in self.stop_words]
    
    def get_wordnet_pos(self, tag: str) -> str:
        """
        Map POS tag to WordNet POS tag
        
        Args:
            tag: NLTK POS tag
            
        Returns:
            WordNet POS tag
        """
        if tag.startswith('J'):
            return 'a'  # Adjective
        elif tag.startswith('V'):
            return 'v'  # Verb
        elif tag.startswith('N'):
            return 'n'  # Noun
        elif tag.startswith('R'):
            return 'r'  # Adverb
        else:
            return 'n'  # Default to noun
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens using POS tagging
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of lemmatized tokens
        """
        if not self.lemmatize:
            return tokens
        
        # POS tag the tokens
        pos_tags = pos_tag(tokens)
        
        # Lemmatize with POS tags
        lemmatized = []
        for token, tag in pos_tags:
            pos = self.get_wordnet_pos(tag)
            lemmatized_token = self.lemmatizer.lemmatize(token, pos)
            lemmatized.append(lemmatized_token)
        
        return lemmatized
    
    def preprocess(self, text: str, return_tokens: bool = False) -> str:
        """
        Complete preprocessing pipeline
        
        Args:
            text: Input text string
            return_tokens: If True, return list of tokens; if False, return string
            
        Returns:
            Preprocessed text (string or list of tokens)
        """
        # Normalize
        normalized = self.normalize_text(text)
        
        # Tokenize
        tokens = self.tokenize(normalized)
        
        # Remove stopwords
        tokens = self.remove_stopwords_func(tokens)
        
        # Lemmatize
        tokens = self.lemmatize_tokens(tokens)
        
        # Filter out empty tokens
        tokens = [token for token in tokens if len(token) > 1]
        
        if return_tokens:
            return tokens
        else:
            return ' '.join(tokens)
    
    def preprocess_batch(self, texts: pd.Series, return_tokens: bool = False) -> pd.Series:
        """
        Preprocess a batch of texts
        
        Args:
            texts: Pandas Series of text strings
            return_tokens: If True, return list of tokens; if False, return string
            
        Returns:
            Pandas Series of preprocessed texts
        """
        return texts.apply(lambda x: self.preprocess(x, return_tokens=return_tokens))


if __name__ == "__main__":
    # Example usage
    preprocessor = TextPreprocessor()
    
    sample_text = "@Tesla $TSLA is amazing! Check out https://tesla.com #EV #innovation"
    print("Original:", sample_text)
    print("Preprocessed:", preprocessor.preprocess(sample_text))
    print("Tokens:", preprocessor.preprocess(sample_text, return_tokens=True))
