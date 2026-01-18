# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Verify Data File

Make sure `data/stock_tweets.csv` exists in your project directory.

### Step 3: Run Quick Analysis

**Option A: Using the Training Script**
```bash
python train.py --sample_size 1000 --preprocess
```

**Option B: Using the Web App**
```bash
streamlit run app/app.py
```

**Option C: Analyze Single Text**
```bash
python inference.py --text "Tesla is revolutionizing the EV industry!" --preprocess
```

## 📋 Common Use Cases

### 1. Analyze Full Dataset
```bash
python train.py --data_path data/stock_tweets.csv --preprocess --output_dir models
```

### 2. Quick Test with Sample
```bash
python train.py --sample_size 500 --preprocess
```

### 3. Use BERT Model (requires GPU recommended)
```bash
python train.py --sample_size 1000 --use_bert --preprocess
```

### 4. Launch Interactive App
```bash
streamlit run app/app.py
```
Then:
- Upload your CSV file or use the default one
- Adjust sample size in sidebar
- Click "Analyze Dataset" in the Analyze tab
- View results in Trends and Results tabs

## 🔍 Example Python Usage

```python
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.models.sentiment_analyzer import SentimentAnalyzer

# Preprocess text
preprocessor = TextPreprocessor()
text = preprocessor.preprocess("@Tesla $TSLA is amazing!")

# Analyze sentiment
analyzer = SentimentAnalyzer(use_bert=False)
results = analyzer.analyze(text, models=['vader', 'textblob'])

print(results)
```

## 📊 Understanding Output

After running `train.py`, you'll get:

1. **sentiment_results.csv**: Full dataset with sentiment predictions
2. **summary_statistics.csv**: Overall sentiment distribution
3. **stock_sentiment.csv**: Stock-wise analysis (if Stock Name column exists)

## ⚠️ Troubleshooting

### Issue: NLTK data not found
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

### Issue: BERT model too slow
- Use `--sample_size` to limit dataset size
- Set `use_bert=False` in code or don't use `--use_bert` flag
- Consider using GPU for faster BERT inference

### Issue: Memory errors with large dataset
- Reduce `sample_size` parameter
- Process in batches
- Use preprocessing to reduce text size

## 📚 Next Steps

1. Explore the Jupyter notebook: `notebooks/exploratory_analysis.ipynb`
2. Customize preprocessing in `src/preprocessing/text_preprocessor.py`
3. Add your own models in `src/models/sentiment_analyzer.py`
4. Modify evaluation metrics in `src/evaluation/metrics.py`

## 💡 Tips

- Start with small sample sizes (500-1000) for testing
- Use preprocessing for better results on social media text
- VADER works best for social media without preprocessing
- BERT provides best accuracy but is slower
- Combine multiple models for ensemble predictions
