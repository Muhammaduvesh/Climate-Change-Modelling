from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from typing import Dict, List
from pathlib import Path
import torch
import logging

# Fix logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest", 
                 max_length=512, batch_size=16, device=None):
       
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading sentiment model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        try:
            # Load tokenizer and model separately for better control
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Legacy pipeline for compatibility
            self.analyzer = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == 'cuda' else -1,
                max_length=max_length,
                truncation=True,
                return_overflowing_tokens=False
            )
            logger.info("Sentiment analyzer loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def analyze_single(self, text: str) -> Dict:
        """Process single text with proper truncation"""
        if pd.isna(text) or not text.strip():
            return {'label': 'NEUTRAL', 'score': 0.0}
        
        # Truncate text to avoid token length issues
        if len(text) > 1000:
            text = text[:1000]
        
        try:
            # Tokenize manually for safety
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=self.max_length
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = probs.argmax().item()
                score = probs.max().item()
                
                # Map to labels (model-specific)
                label = self.model.config.id2label[predicted_class] if hasattr(self.model.config, 'id2label') else 'POSITIVE'
                return {'label': label.upper(), 'score': float(score)}
                
        except Exception as e:
            logger.warning(f"Single analysis failed: {e}")
            return {'label': 'NEUTRAL', 'score': 0.0}
    
    def analyze_batch(self, texts: List[str], batch_size: int = None) -> List[Dict]:
        """Safe batch processing with fallback to single"""
        batch_size = batch_size or self.batch_size
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            try:
                # Try pipeline first
                batch_results = self.analyzer(batch, truncation=True, max_length=self.max_length)
                
                # Handle pipeline output format
                if isinstance(batch_results, dict):
                    batch_results = [batch_results]
                
                results.extend(batch_results)
                
            except Exception as e:
                logger.warning(f"Batch failed with pipeline, using single mode: {e}")
                # Fallback to single processing
                for text in batch:
                    results.append(self.analyze_single(text))
        
        return results
    
    def get_sentiment_score(self, result: Dict) -> float:
        """Convert sentiment result to numeric score"""
        label = result.get('label', 'NEUTRAL').upper()
        score = result.get('score', 0.0)
        return score if label in ['POSITIVE', 'LABEL_0'] else -score

def add_sentiment(df: pd.DataFrame, model_name: str = None, save_cache: bool = True) -> pd.DataFrame:
    """
    Add sentiment scores to dataframe with caching
    """
    cache_path = Path("data/processed/sentiment_cache.pkl")
    
    # Load cache if exists
    if save_cache and cache_path.exists():
        logger.info("Loading sentiment cache...")
        cached_df = pd.read_pickle(cache_path)
        if len(cached_df) == len(df):
            logger.info("Using cached sentiment scores")
            return cached_df
    
    logger.info("Analyzing sentiment for all texts...")
    analyzer = SentimentAnalyzer(model_name=model_name)
    
    # Remove empty texts
    valid_texts = df['text_clean'].fillna('').astype(str).tolist()
    sentiments = analyzer.analyze_batch(valid_texts)
    
    df = df.copy()
    df['sentiment_label'] = [r['label'] for r in sentiments]
    df['sentiment_score'] = [analyzer.get_sentiment_score(r) for r in sentiments]
    
    # Save cache
    if save_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache_path)
        logger.info(f"Cached sentiment results: {cache_path}")
    
    logger.info(f"Sentiment stats - POS: {sum(1 for x in df['sentiment_label'] if 'POS' in str(x))}, "
                f"NEG: {sum(1 for x in df['sentiment_label'] if 'NEG' in str(x))}")
    
    return df
