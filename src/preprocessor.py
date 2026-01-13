import pandas as pd
import re
from typing import List

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    return text

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['text_clean'] = df['text'].apply(clean_text)
    df['engagement'] = df['likesCount'].fillna(0) + df['commentsCount'].fillna(0)
    df['year_month'] = df['date'].dt.to_period('M')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['season'] = pd.cut(df['month'], bins=[0,3,6,9,12], labels=['Winter','Spring','Summer','Fall'])
    return df
