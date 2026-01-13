from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import pandas as pd
import numpy as np

def prepare_features(df: pd.DataFrame) -> tuple:
    features = ['sentiment_score', 'month', 'year', 'text_length']
    df['text_length'] = df['text_clean'].str.len()
    
    X = df[features].fillna(0)
    y = np.log1p(df['engagement'])
    
    return X, y

def train_engagement_model(df: pd.DataFrame, model_path: str):
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
    
    joblib.dump(model, model_path)
    return model

def predict_controversy(df: pd.DataFrame, model_path: str):
    model = joblib.load(model_path)
    X, _ = prepare_features(df)
    df['predicted_engagement'] = np.expm1(model.predict(X))
    high_engagement = df.nlargest(10, 'predicted_engagement')
    return high_engagement
