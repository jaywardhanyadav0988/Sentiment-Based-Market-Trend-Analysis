# Project Structure

## 📁 Complete Project Organization

```
Sentiment Based Anlysis/
│
├── 📊 data/                          # Data directory
│   └── stock_tweets.csv              # Main dataset (moved from root)
│
├── 🧠 src/                           # Source code
│   ├── __init__.py                   # Package initialization
│   │
│   ├── preprocessing/                # Text preprocessing modules
│   │   ├── __init__.py
│   │   └── text_preprocessor.py     # Tokenization, stopwords, lemmatization
│   │
│   ├── models/                       # Sentiment analysis models
│   │   ├── __init__.py
│   │   └── sentiment_analyzer.py    # VADER, TextBlob, BERT implementations
│   │
│   ├── evaluation/                   # Model evaluation
│   │   ├── __init__.py
│   │   └── metrics.py               # Accuracy, Precision, Recall, F1, CM
│   │
│   └── utils/                        # Utility functions
│       ├── __init__.py
│       └── data_loader.py            # Data loading and preparation
│
├── 🎯 models/                        # Output directory for trained models/results
│   ├── sentiment_results.csv        # Generated after training
│   ├── summary_statistics.csv       # Generated after training
│   └── stock_sentiment.csv          # Generated after training
│
├── 📓 notebooks/                     # Jupyter notebooks
│   └── exploratory_analysis.ipynb    # EDA notebook
│
├── 🌐 app/                           # Streamlit web application
│   └── app.py                        # Main Streamlit app
│
├── ⚙️ config/                        # Configuration files
│   └── config.yaml                   # Project configuration
│
├── 🚀 train.py                       # Training script
├── 🔍 inference.py                   # Inference script for single texts
├── 📦 setup.py                       # Package setup script
├── 📋 requirements.txt               # Python dependencies
├── 📖 README.md                      # Main documentation
├── 🏃 QUICKSTART.md                  # Quick start guide
├── 📝 PROJECT_STRUCTURE.md           # This file
└── .gitignore                        # Git ignore rules
```

## 🎯 Key Components

### 1. Preprocessing (`src/preprocessing/`)
- **TextPreprocessor**: Complete NLP preprocessing pipeline
  - Normalization (lowercase, URL removal, HTML entities)
  - Tokenization (TweetTokenizer for social media)
  - Stopword removal
  - Lemmatization with POS tagging

### 2. Models (`src/models/`)
- **SentimentAnalyzer**: Unified sentiment analysis interface
  - VADER: Rule-based, optimized for social media
  - TextBlob: Simple polarity and subjectivity
  - BERT: RoBERTa-based transformer model
  - Ensemble: Combines all models

### 3. Evaluation (`src/evaluation/`)
- **ModelEvaluator**: Comprehensive evaluation metrics
  - Accuracy, Precision, Recall, F1-Score
  - Per-class metrics
  - Confusion matrix visualization
  - Model comparison

### 4. Application (`app/`)
- **Streamlit App**: Interactive web interface
  - Dataset overview
  - Single text analysis
  - Batch processing
  - Trend visualization
  - Results export

### 5. Scripts
- **train.py**: Full pipeline training script
- **inference.py**: Quick sentiment analysis for single texts

## 📊 Data Flow

```
CSV Data → Data Loader → Preprocessing → Sentiment Analysis → Evaluation → Results
```

## 🔧 Technology Stack

- **Core**: Python 3.8+, Pandas, NumPy
- **NLP**: NLTK, VADER, TextBlob
- **ML/DL**: Scikit-learn, PyTorch, Transformers
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Web**: Streamlit
- **Notebooks**: Jupyter

## 📈 Output Files

After running `train.py`:
1. `models/sentiment_results.csv` - Full results with predictions
2. `models/summary_statistics.csv` - Overall statistics
3. `models/stock_sentiment.csv` - Stock-wise analysis

## 🎓 Usage Patterns

1. **Quick Analysis**: `python inference.py --text "Your text here"`
2. **Full Training**: `python train.py --preprocess --use_bert`
3. **Web App**: `streamlit run app/app.py`
4. **Exploration**: Open `notebooks/exploratory_analysis.ipynb`

## ✅ Project Checklist

- [x] Proper folder structure
- [x] Text preprocessing module
- [x] Multiple sentiment models (VADER, TextBlob, BERT)
- [x] Evaluation metrics (Accuracy, Precision, Recall, F1, CM)
- [x] Training script
- [x] Inference script
- [x] Web application (Streamlit)
- [x] Configuration files
- [x] Documentation (README, Quick Start)
- [x] Requirements file
- [x] Jupyter notebook for exploration
- [x] Git ignore file
- [x] Setup script

## 🚀 Next Steps

1. Install dependencies: `pip install -r requirements.txt`
2. Run quick test: `python train.py --sample_size 500`
3. Launch app: `streamlit run app/app.py`
4. Explore notebook: Open `notebooks/exploratory_analysis.ipynb`
