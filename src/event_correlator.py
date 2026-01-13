import pandas as pd
from datetime import datetime

# Key climate events 2020-2023 (UTC)
CLIMATE_EVENTS = {
    '2020 Australian Bushfires': '2020-01-01',
    '2021 Pacific NW Heat Dome': '2021-06-25', 
    '2021 European Floods': '2021-07-12',
    '2022 Pakistan Floods': '2022-06-15',
    '2022 Canada Wildfires': '2022-05-01',
    'COP26': '2021-10-31',
    'COP27': '2022-11-06',
    'COP28': '2023-11-30'
}

def add_event_flags(df: pd.DataFrame) -> pd.DataFrame:
    """FIXED: Handle timezone-aware dates"""
    df = df.copy()
    
    # Normalize ALL dates to naive (no timezone)
    df['date_naive'] = df['date'].dt.tz_localize(None)
    
    for event, date_str in CLIMATE_EVENTS.items():
        # Parse as naive datetime
        event_date = pd.to_datetime(date_str).tz_localize(None)
        window_start = event_date - pd.Timedelta(days=7)
        window_end = event_date + pd.Timedelta(days=30)
        
        # Use naive date column
        df[f'{event}_window'] = df['date_naive'].between(window_start, window_end)
    
    return df

def analyze_event_impact(df: pd.DataFrame):
    """Analyze engagement during event windows"""
    event_cols = [col for col in df.columns if col.endswith('_window')]
    impacts = {}
    
    for col in event_cols:
        event_data = df[df[col]]
        if len(event_data) > 5:  # Only events with enough data
            impacts[col] = {
                'n_comments': len(event_data),
                'avg_engagement': event_data['engagement'].mean(),
                'avg_sentiment': event_data['sentiment_score'].mean()
            }
    
    if impacts:
        impact_df = pd.DataFrame(impacts).T.round(2)
        print("\nEVENT IMPACT ANALYSIS:")
        print(impact_df.sort_values('avg_engagement', ascending=False))
        return impact_df
    else:
        print("No significant event overlaps found")
        return pd.DataFrame()
