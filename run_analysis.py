import pandas as pd
from pathlib import Path
from src.data_loader import load_data, load_config
from src.preprocessor import preprocess_df
from src.sentiment_analyzer import add_sentiment
from src.event_correlator import add_event_flags, analyze_event_impact
from src.engagement_predictor import train_engagement_model, predict_controversy
from src.trend_analyzer import plot_sentiment_trend, plot_seasonal_patterns
import warnings
from src.visualizer import ClimateVisualizer
warnings.filterwarnings('ignore')

def main():
    config = load_config()
    Path(config['viz']['save_path']).mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess
    print("Loading data...")
    df = load_data(config['data']['raw_path'])
    df = preprocess_df(df)
    
    # FIXED SENTIMENT LINE 👇
    print("Analyzing sentiment...")
    df = add_sentiment(df, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest")
    
    # Event correlation
    print("Correlating with events...")
    df = add_event_flags(df)
    event_impact = analyze_event_impact(df)
    
    # Train engagement model
    print("Training engagement model...")
    model_path = "models/engagement_model.pkl"
    Path("models").mkdir(exist_ok=True)
    model = train_engagement_model(df, model_path)

if __name__ == "__main__":
    main()
