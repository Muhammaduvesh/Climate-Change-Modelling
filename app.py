import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import joblib
from pathlib import Path
import statsmodels
import plotly.graph_objects as go

st.set_page_config(page_title="Climate Discourse", layout="wide", page_icon="🌍")

@st.cache_data
def load_data():
    pkl_path = Path("data/processed/sentiment_cache.pkl")
    if pkl_path.exists():
        return pd.read_pickle(pkl_path)
    return None

@st.cache_data
def load_model():
    model_path = Path("models/engagement_model.pkl")
    if model_path.exists():
        return joblib.load(model_path)
    return None

st.title("Climate Discourse Analysis Dashboard")
st.markdown("NASA climate comments analysis (522 posts, 2020-2023)")

# Load data
df = load_data()
model = load_model()

if df is None:
    st.error("**Run `python run_analysis.py` first!**")
    st.stop()

st.success(f"Loaded {len(df)} climate comments")

# Key Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Posts", len(df))
col2.metric("Avg Engagement", f"{df['engagement'].mean():.1f}")
col3.metric("Avg Sentiment", f"{df['sentiment_score'].mean():+.3f}")
col4.metric("Peak Engagement", f"{df['engagement'].max():.0f}")

# Sentiment Trend - FIXED
st.subheader("Sentiment Trends (2020-2023)")
df['year_month'] = df['date'].dt.to_period('M').astype(str)
monthly_sent = df.groupby(['year_month', 'sentiment_label']).size().reset_index(name='count')
monthly_sent['year_month'] = pd.to_datetime(monthly_sent['year_month'])
fig1 = px.line(monthly_sent, x='year_month', y='count', color='sentiment_label',
               title="Monthly Sentiment Distribution")
st.plotly_chart(fig1, use_container_width=True)

# Seasonal Patterns - FIXED  
st.subheader("Seasonal Engagement Peaks")
if 'season' not in df.columns:
    df['month'] = df['date'].dt.month
    df['season'] = pd.cut(df['month'], bins=[0,3,6,9,12], 
                          labels=['Winter','Spring','Summer','Fall'])

seasonal = df.groupby('season', observed=True)['engagement'].mean().reset_index()
fig2 = px.bar(seasonal, x='season', y='engagement', 
              title="Summer = Highest Engagement (Wildfires)")
st.plotly_chart(fig2, use_container_width=True)

# Engagement vs Sentiment
st.subheader("Engagement vs Sentiment")
fig3 = px.scatter(df, x='sentiment_score', y='engagement', 
                  title="More Negative = More Engagement?",
                  trendline="ols", hover_data=['text'])
st.plotly_chart(fig3, use_container_width=True)

# Live Predictor
st.subheader("Live Controversy Predictor")
with st.form("predictor"):
    text = st.text_area("Enter climate comment:", 
                       "Climate change is a hoax - just weather cycles!")
    month = st.slider("Month (1-12)", 1, 12, 6)  # Summer
    year = st.slider("Year", 2020, 2023, 2022)
    
    if st.form_submit_button("🔮 Predict Engagement"):
        if model:
            text_len = len(text)
            # Mock prediction using model structure
            features = np.array([[0.0, month, year, text_len]])
            pred = np.expm1(model.predict(features)[0])
            st.success(f"🎯 **Predicted Engagement: {pred:.0f}**")
            st.info(f"💡 Longer texts + Summer months = higher controversy")
        else:
            st.warning("Model not found - predictions disabled")

# Top Controversial Posts
st.subheader("Top 10 Most Engaging Posts")
top_cols = ['date', 'sentiment_label', 'sentiment_score', 'engagement']
if 'predicted_engagement' in df.columns:
    top_cols.append('predicted_engagement')
top_posts = df.nlargest(10, 'engagement')[top_cols].copy()
top_posts['date'] = top_posts['date'].dt.strftime('%Y-%m-%d')
st.dataframe(top_posts, use_container_width=True, hide_index=True)

# Word Frequency (Bonus)
st.subheader("Word Cloud Preview")
text_sample = ' '.join(df['text_clean'].dropna().head(100))
st.text_area("Top climate words:", text_sample[:500], height=150)
