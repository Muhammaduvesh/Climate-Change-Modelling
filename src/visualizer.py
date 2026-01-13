import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import numpy as np
from typing import Dict, List
from datetime import datetime

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ClimateVisualizer:
    def __init__(self, save_path: str = "reports/figures"):
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
    
    def sentiment_trend_plot(self, df: pd.DataFrame, period: str = 'M'):
        """Interactive sentiment trend over time"""
        monthly_sent = df.groupby([df['date'].dt.to_period(period), 'sentiment_label']).size().unstack(fill_value=0)
        monthly_sent['total'] = monthly_sent.sum(axis=1)
        monthly_sent['pos_ratio'] = monthly_sent.get('POSITIVE', 0) / monthly_sent['total']
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=monthly_sent.index, y=monthly_sent['pos_ratio'], 
                      name='Positive Ratio', line=dict(color='green')),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Bar(x=monthly_sent.index, y=monthly_sent['total'], 
                   name='Total Posts', marker_color='lightblue'),
            secondary_y=True
        )
        
        fig.update_layout(
            title="Sentiment Trend Analysis (2020-2023)",
            xaxis_title="Time Period",
            hovermode='x unified'
        )
        
        fig.write_html(self.save_path / 'sentiment_trend_interactive.html')
        fig.write_image(self.save_path / 'sentiment_trend.png', scale=2)
        fig.show()
        
        return monthly_sent
    
    def seasonal_heatmap(self, df: pd.DataFrame):
        """Seasonal engagement and sentiment heatmap"""
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['season'] = pd.cut(df['month'], bins=[0,3,6,9,12], 
                             labels=['Winter','Spring','Summer','Fall'])
        
        pivot_engagement = df.pivot_table(
            values='engagement', index='season', columns='year', aggfunc='mean'
        )
        pivot_sentiment = df.pivot_table(
            values='sentiment_score', index='season', columns='year', aggfunc='mean'
        )
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        sns.heatmap(pivot_engagement, annot=True, cmap='YlOrRd', ax=ax1)
        ax1.set_title('Average Engagement by Season & Year')
        
        sns.heatmap(pivot_sentiment, annot=True, cmap='RdBu_r', center=0, ax=ax2)
        ax2.set_title('Average Sentiment by Season & Year')
        
        plt.tight_layout()
        plt.savefig(self.save_path / 'seasonal_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def engagement_distribution(self, df: pd.DataFrame):
        """Interactive engagement distribution with sentiment overlay"""
        fig = px.histogram(df, x='engagement', color='sentiment_label',
                          marginal='box', hover_data=['text_length', 'date'],
                          title='Engagement Distribution by Sentiment')
        
        fig.update_layout(bargap=0.1)
        fig.write_html(self.save_path / 'engagement_distribution.html')
        fig.show()
    
    def event_impact_timeline(self, df: pd.DataFrame, events: Dict[str, str]):
        """Timeline showing event windows and engagement spikes"""
        df_event = df.copy()
        for event, date_str in events.items():
            event_date = pd.to_datetime(date_str)
            window_start = event_date - pd.Timedelta(days=7)
            window_end = event_date + pd.Timedelta(days=30)
            df_event[f'{event}_window'] = df_event['date'].between(window_start, window_end)
        
        # Plot engagement around events
        fig, ax = plt.subplots(figsize=(15, 8))
        df['rolling_engagement'] = df['engagement'].rolling(window=14, center=True).mean()
        df['rolling_sentiment'] = df['sentiment_score'].rolling(window=14, center=True).mean()
        
        ax.plot(df['date'], df['rolling_engagement'], label='Engagement (14d rolling)', linewidth=2)
        ax_twin = ax.twinx()
        ax_twin.plot(df['date'], df['rolling_sentiment'], color='orange', 
                    label='Sentiment (14d rolling)', alpha=0.7)
        
        # Mark events
        for event, date_str in events.items():
            event_date = pd.to_datetime(date_str)
            ax.axvline(event_date, color='red', linestyle='--', alpha=0.5, 
                      label=event if list(events.keys()).index(event) == 0 else "")
        
        ax.set_title('Engagement & Sentiment Timeline with Climate Events')
        ax.set_ylabel('Engagement', color='blue')
        ax_twin.set_ylabel('Sentiment Score', color='orange')
        fig.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(self.save_path / 'event_timeline.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def controversy_radar(self, df: pd.DataFrame, top_n: int = 10):
        """Radar chart of controversy drivers"""
        controversy_posts = df.nlargest(top_n, 'predicted_engagement')
        
        metrics = ['sentiment_score', 'text_length', 'likesCount', 'commentsCount']
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for idx, (_, row) in enumerate(controversy_posts.iterrows()):
            values = [row[m] for m in metrics] + [row[metrics[0]]]
            ax.plot(angles, values, 'o-', linewidth=2, label=f'Post {idx+1}')
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, max(df[metrics].max()))
        ax.set_title('Top 10 Controversial Posts - Feature Radar', size=16, pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.savefig(self.save_path / 'controversy_radar.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def wordcloud_sentiment(self, df: pd.DataFrame):
        """Word clouds for positive vs negative sentiment"""
        try:
            from wordcloud import WordCloud
        except ImportError:
            print("Install wordcloud: pip install wordcloud")
            return
        
        pos_text = ' '.join(df[df['sentiment_label'] == 'POSITIVE']['text_clean'].dropna())
        neg_text = ' '.join(df[df['sentiment_label'] == 'NEGATIVE']['text_clean'].dropna())
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        if pos_text:
            wc_pos = WordCloud(width=800, height=400, background_color='white').generate(pos_text)
            axes[0].imshow(wc_pos, interpolation='bilinear')
            axes[0].set_title('Positive Sentiment Words')
            axes[0].axis('off')
        
        if neg_text:
            wc_neg = WordCloud(width=800, height=400, background_color='white', 
                             colormap='Reds').generate(neg_text)
            axes[1].imshow(wc_neg, interpolation='bilinear')
            axes[1].set_title('Negative Sentiment Words')
            axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.save_path / 'sentiment_wordclouds.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def full_dashboard(self, df: pd.DataFrame, events: Dict[str, str]):
        """Generate complete visualization suite"""
        print("Generating full visualization dashboard...")
        
        # Key climate events
        CLIMATE_EVENTS = events or {
            'COP26': '2021-10-31', 'COP27': '2022-11-06',
            'Canada Wildfires': '2022-07-01', 'Pakistan Floods': '2022-06-15'
        }
        
        self.sentiment_trend_plot(df)
        self.seasonal_heatmap(df)
        self.engagement_distribution(df)
        self.event_impact_timeline(df, CLIMATE_EVENTS)
        self.wordcloud_sentiment(df)
        
        print(f"All visualizations saved to {self.save_path}")
        print("Interactive HTML files ready for stakeholder review")

# Usage example
if __name__ == "__main__":
    from data_loader import load_data  # Adjust import path
    from preprocessor import preprocess_df
    from sentiment_analyzer import add_sentiment
    
    viz = ClimateVisualizer()
    
    # Load and prepare data (demo)
    df = load_data('data/raw/climate_nasa.csv')
    df = preprocess_df(df)
    df = add_sentiment(df)  # Add sentiment scores
    
    viz.full_dashboard(df, None)
