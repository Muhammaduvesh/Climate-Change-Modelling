import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_sentiment_trend(df: pd.DataFrame, save_path: Path):
    monthly_sent = df.groupby(['year_month', 'sentiment_label']).size().unstack(fill_value=0)
    monthly_sent['total'] = monthly_sent.sum(axis=1)
    monthly_sent['pos_ratio'] = monthly_sent['POSITIVE'] / monthly_sent['total']
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    monthly_sent['pos_ratio'].plot(ax=axes[0], title='Positive Sentiment Trend (2020-2023)')
    axes[0].set_ylabel('Positive Ratio')
    
    df.groupby('year_month')['sentiment_score'].mean().plot(ax=axes[1], title='Average Sentiment Score')
    axes[1].set_ylabel('Mean Sentiment Score')
    
    plt.tight_layout()
    plt.savefig(save_path / 'sentiment_trend.png')
    plt.close()

def plot_seasonal_patterns(df: pd.DataFrame, save_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    sns.boxplot(data=df, x='season', y='sentiment_score', ax=axes[0,0])
    axes[0,0].set_title('Sentiment by Season')
    
    engagement_season = df.groupby('season')['engagement'].mean()
    engagement_season.plot(kind='bar', ax=axes[0,1])
    axes[0,1].set_title('Avg Engagement by Season')
    
    monthly_engagement = df.groupby('year_month')['engagement'].mean()
    monthly_engagement.plot(ax=axes[1,0], title='Monthly Engagement')
    
    yearly_sent = df.groupby('year')['sentiment_score'].mean()
    yearly_sent.plot(kind='bar', ax=axes[1,1])
    axes[1,1].set_title('Yearly Avg Sentiment')
    
    plt.tight_layout()
    plt.savefig(save_path / 'seasonal_patterns.png')
    plt.close()
