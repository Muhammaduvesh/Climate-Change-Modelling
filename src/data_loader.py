import pandas as pd
import yaml
from pathlib import Path

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data(raw_path):
    df = pd.read_csv(raw_path, parse_dates=['date'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['text', 'date'])
    return df

if __name__ == "__main__":
    config = load_config()
    df = load_data(config['data']['raw_path'])
    print(f"Loaded {len(df)} records from {df['date'].min()} to {df['date'].max()}")
