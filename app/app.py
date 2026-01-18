"""
Streamlit Application for Sentiment-Based Market Trend Analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.data_loader import load_data, prepare_data
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.models.sentiment_analyzer import SentimentAnalyzer


# Page configuration
st.set_page_config(
    page_title="Sentiment-Based Market Trend Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_dataset(file_path: str, sample_size: int = None):
    """Load and cache dataset"""
    df = load_data(file_path, sample_size=sample_size)
    df = prepare_data(df, text_column='Tweet')
    return df


@st.cache_resource
def initialize_analyzer(use_bert: bool = False):
    """Initialize and cache sentiment analyzer"""
    return SentimentAnalyzer(use_bert=use_bert)


def main():
    # Header
    st.title("📈 Sentiment-Based Market Trend Analysis")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    
    # Check if default file exists
    default_path = Path(__file__).parent.parent / "data" / "stock_tweets.csv"
    has_default_file = default_path.exists()
    
    # Options
    sample_size = st.sidebar.slider("Sample Size (for faster processing)", 
                                     min_value=100, max_value=60000, 
                                     value=1000, step=100)
    
    use_bert = st.sidebar.checkbox("Use BERT Model (slower but more accurate)", value=False)
    apply_preprocessing = st.sidebar.checkbox("Apply Text Preprocessing", value=True)
    
    # Main content
    if uploaded_file is not None or has_default_file:
        try:
            # Load data
            with st.spinner("Loading data..."):
                if uploaded_file is not None:
                    # Handle uploaded file
                    df = pd.read_csv(uploaded_file)
                    df = prepare_data(df, text_column='Tweet')
                    if sample_size and sample_size < len(df):
                        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
                else:
                    # Use default file
                    df = load_dataset(str(default_path), sample_size=sample_size)
            
            st.success(f"Loaded {len(df)} tweets")
            
            # Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Analyze", "📈 Trends", "📋 Results"])
            
            # Tab 1: Overview
            with tab1:
                st.header("Dataset Overview")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Tweets", len(df))
                
                with col2:
                    if 'Stock Name' in df.columns:
                        st.metric("Unique Stocks", df['Stock Name'].nunique())
                
                with col3:
                    if 'Company Name' in df.columns:
                        st.metric("Unique Companies", df['Company Name'].nunique())
                
                with col4:
                    avg_length = df['Tweet'].str.len().mean()
                    st.metric("Avg Tweet Length", f"{avg_length:.0f} chars")
                
                # Data preview
                st.subheader("Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Stock distribution
                if 'Stock Name' in df.columns:
                    st.subheader("Stock Distribution")
                    stock_counts = df['Stock Name'].value_counts().head(10)
                    fig = px.bar(x=stock_counts.index, y=stock_counts.values,
                                labels={'x': 'Stock', 'y': 'Count'},
                                title="Top 10 Stocks by Tweet Count")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Tab 2: Analyze
            with tab2:
                st.header("Sentiment Analysis")
                
                # Initialize analyzer
                with st.spinner("Initializing models..."):
                    analyzer = initialize_analyzer(use_bert=use_bert)
                
                # Single text analysis
                st.subheader("Analyze Single Text")
                input_text = st.text_area("Enter text to analyze:", 
                                         "Tesla is revolutionizing the electric vehicle industry!")
                
                if st.button("Analyze"):
                    # Preprocess if needed
                    if apply_preprocessing:
                        preprocessor = TextPreprocessor()
                        processed_text = preprocessor.preprocess(input_text)
                    else:
                        processed_text = input_text
                    
                    # Analyze
                    results = analyzer.analyze(processed_text, 
                                             models=['vader', 'textblob'] + (['bert'] if use_bert else []))
                    
                    # Display results
                    cols = st.columns(len(results))
                    for idx, (model_name, model_results) in enumerate(results.items()):
                        with cols[idx]:
                            st.markdown(f"### {model_name.upper()}")
                            
                            if 'label' in model_results:
                                label = model_results['label']
                                st.markdown(f"**Sentiment:** {label.capitalize()}")
                            
                            if 'compound' in model_results:
                                st.metric("Compound Score", f"{model_results['compound']:.3f}")
                                st.metric("Positive", f"{model_results['pos']:.3f}")
                                st.metric("Neutral", f"{model_results['neu']:.3f}")
                                st.metric("Negative", f"{model_results['neg']:.3f}")
                            
                            if 'polarity' in model_results:
                                st.metric("Polarity", f"{model_results['polarity']:.3f}")
                                st.metric("Subjectivity", f"{model_results['subjectivity']:.3f}")
                            
                            if 'score' in model_results:
                                st.metric("Confidence", f"{model_results['score']:.3f}")
                
                # Batch analysis
                st.subheader("Batch Analysis")
                if st.button("Analyze Dataset"):
                    with st.spinner("Analyzing dataset... This may take a while..."):
                        # Preprocess
                        if apply_preprocessing:
                            preprocessor = TextPreprocessor()
                            df['processed_text'] = preprocessor.preprocess_batch(df['Tweet'])
                            text_col = 'processed_text'
                        else:
                            text_col = 'Tweet'
                        
                        # Analyze (sample for performance)
                        sample_df = df.head(min(500, len(df)))
                        models_to_use = ['vader', 'textblob'] + (['bert'] if use_bert else [])
                        sentiment_results = analyzer.analyze_batch(sample_df[text_col], 
                                                                  models=models_to_use)
                        
                        # Combine
                        df_analyzed = pd.concat([sample_df.reset_index(drop=True),
                                               sentiment_results.reset_index(drop=True)], axis=1)
                        
                        # Store in session state
                        st.session_state['df_analyzed'] = df_analyzed
                        st.session_state['analyzed'] = True
                    
                    st.success("Analysis complete!")
            
            # Tab 3: Trends
            with tab3:
                st.header("Market Trends")
                
                if 'analyzed' in st.session_state and st.session_state['analyzed']:
                    df_analyzed = st.session_state['df_analyzed']
                    
                    # Overall sentiment distribution
                    st.subheader("Overall Sentiment Distribution")
                    
                    model_choice = st.selectbox("Select Model:", 
                                              ['vader', 'textblob'] + (['bert'] if use_bert else []))
                    
                    if f'{model_choice}_label' in df_analyzed.columns:
                        sentiment_counts = df_analyzed[f'{model_choice}_label'].value_counts()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig_pie = px.pie(values=sentiment_counts.values,
                                           names=sentiment_counts.index,
                                           title=f"Sentiment Distribution ({model_choice.upper()})")
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with col2:
                            fig_bar = px.bar(x=sentiment_counts.index,
                                           y=sentiment_counts.values,
                                           labels={'x': 'Sentiment', 'y': 'Count'},
                                           title=f"Sentiment Counts ({model_choice.upper()})")
                            st.plotly_chart(fig_bar, use_container_width=True)
                        
                        # Stock-wise trends
                        if 'Stock Name' in df_analyzed.columns:
                            st.subheader("Stock-wise Sentiment")
                            
                            selected_stocks = st.multiselect("Select Stocks:",
                                                            df_analyzed['Stock Name'].unique(),
                                                            default=df_analyzed['Stock Name'].unique()[:5])
                            
                            if selected_stocks:
                                stock_df = df_analyzed[df_analyzed['Stock Name'].isin(selected_stocks)]
                                stock_sentiment = stock_df.groupby(['Stock Name', 
                                                                   f'{model_choice}_label']).size().reset_index(name='count')
                                
                                fig = px.bar(stock_sentiment,
                                           x='Stock Name', y='count',
                                           color=f'{model_choice}_label',
                                           title="Sentiment by Stock",
                                           labels={'count': 'Number of Tweets'})
                                st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Please run batch analysis first in the 'Analyze' tab.")
            
            # Tab 4: Results
            with tab4:
                st.header("Detailed Results")
                
                if 'analyzed' in st.session_state and st.session_state['analyzed']:
                    df_analyzed = st.session_state['df_analyzed']
                    
                    # Download button
                    csv = df_analyzed.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv"
                    )
                    
                    # Display results
                    st.dataframe(df_analyzed, use_container_width=True)
                else:
                    st.info("Please run batch analysis first in the 'Analyze' tab.")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)
    else:
        st.warning("Please upload a CSV file or ensure data/stock_tweets.csv exists")


# Run the app
main()
