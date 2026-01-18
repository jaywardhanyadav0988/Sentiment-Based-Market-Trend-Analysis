# Sentiment-Based Market Trend Analysis

A comprehensive AI/ML project for analyzing market trends using sentiment analysis on social media data (Twitter/X tweets about stocks).

## 🎯 Project Overview

This project performs NLP-based sentiment analysis on large-scale social media text data to predict market trends. It implements multiple sentiment analysis models (VADER, TextBlob, BERT) and provides comprehensive evaluation metrics.

## ✨ Features

- **Text Preprocessing**: Tokenization, stopword removal, lemmatization, and normalization
- **Multiple Sentiment Models**: 
  - VADER (Valence Aware Dictionary and sEntiment Reasoner)
  - TextBlob
  - BERT/RoBERTa (Twitter-specific model)
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Interactive Web Application**: Streamlit-based UI for easy analysis
- **Stock-wise Analysis**: Analyze sentiment trends by stock/company
- **Batch Processing**: Efficient processing of large datasets

## 📁 Project Structure

```
Sentiment Based Anlysis/
├── data/                    # Data files
│   └── stock_tweets.csv     # Input dataset
├── src/                     # Source code
│   ├── preprocessing/       # Text preprocessing modules
│   │   └── text_preprocessor.py
│   ├── models/              # Sentiment analysis models
│   │   └── sentiment_analyzer.py
│   ├── evaluation/          # Model evaluation
│   │   └── metrics.py
│   └── utils/               # Utility functions
│       └── data_loader.py
├── models/                  # Trained models and results
├── notebooks/               # Jupyter notebooks for exploration
├── app/                     # Streamlit application
│   └── app.py
├── config/                  # Configuration files
├── train.py                 # Training script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Installation

1. **Clone or download the project**

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download NLTK data** (if not automatically downloaded):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

## 📊 Dataset

The project uses a CSV file (`stock_tweets.csv`) with the following columns:
- `Date`: Timestamp of the tweet
- `Tweet`: Text content of the tweet
- `Stock Name`: Stock ticker symbol (e.g., TSLA)
- `Company Name`: Full company name

## 💻 Usage

### 1. Training/Processing Script

Run the training script to analyze the dataset:

```bash
# Basic usage
python train.py

# With options
python train.py --data_path data/stock_tweets.csv --sample_size 5000 --use_bert --preprocess

# Options:
#   --data_path: Path to CSV file (default: data/stock_tweets.csv)
#   --sample_size: Number of samples to process (default: None, processes all)
#   --use_bert: Enable BERT model (requires transformers library)
#   --preprocess: Apply text preprocessing
#   --output_dir: Output directory (default: models)
```

### 2. Streamlit Web Application

Launch the interactive web application:

```bash
streamlit run app/app.py
```

The app provides:
- **Overview**: Dataset statistics and preview
- **Analyze**: Single text and batch sentiment analysis
- **Trends**: Visualizations of sentiment trends by stock
- **Results**: Detailed results table with download option

### 3. Python API

Use the modules directly in your code:

```python
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.models.sentiment_analyzer import SentimentAnalyzer
from src.evaluation.metrics import ModelEvaluator

# Preprocess text
preprocessor = TextPreprocessor()
processed_text = preprocessor.preprocess("Tesla is amazing! $TSLA")

# Analyze sentiment
analyzer = SentimentAnalyzer(use_bert=True)
results = analyzer.analyze(processed_text, models=['vader', 'textblob', 'bert'])

# Evaluate model
evaluator = ModelEvaluator()
metrics = evaluator.calculate_metrics(y_true, y_pred)
```

## 🔧 Technologies Used

- **Python 3.8+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **NLTK**: Natural Language Toolkit for text processing
- **VADER**: Sentiment analysis model
- **TextBlob**: Text processing and sentiment analysis
- **BERT/RoBERTa**: Deep learning-based sentiment analysis (via transformers)
- **Scikit-learn**: Machine learning utilities
- **Streamlit**: Web application framework
- **Plotly/Matplotlib**: Data visualization

## 📈 Model Evaluation

The project evaluates models using:
- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value
- **Recall**: Sensitivity/True positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

## 🎓 Key Features Explained

### Text Preprocessing
- **Tokenization**: Splitting text into words/tokens
- **Stopword Removal**: Removing common words (the, is, etc.)
- **Lemmatization**: Converting words to their base form
- **Normalization**: Lowercasing, URL removal, handling special characters

### Sentiment Analysis Models
- **VADER**: Rule-based model optimized for social media
- **TextBlob**: Simple API for common NLP tasks
- **BERT**: Transformer-based model fine-tuned on Twitter data

## 📝 Example Output

After running the training script, you'll get:
- `sentiment_results.csv`: Full dataset with sentiment predictions
- `summary_statistics.csv`: Overall sentiment distribution
- `stock_sentiment.csv`: Stock-wise sentiment analysis

## 🤝 Contributing

Feel free to fork the project and submit pull requests for improvements.

## 📄 License

This project is for educational purposes.

## 👤 Author

Created as part of an AI/ML project demonstrating sentiment analysis capabilities.

## 🙏 Acknowledgments

- VADER: Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text
- BERT Model: Cardiff NLP Twitter-RoBERTa-base-sentiment-latest
- TextBlob: Python library for processing textual data

---

**Note**: For BERT model usage, ensure you have sufficient computational resources (GPU recommended for large datasets).
