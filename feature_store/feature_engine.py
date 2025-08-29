# feature_store/feature_engine.py
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

class FeatureEngine:
    def __init__(self):
        self.encoders = {}
        self.feature_store = pd.DataFrame()
    
    def enrich_logs(self, log_data):
        """Enrich raw logs with additional features"""
        if not log_data:
            return pd.DataFrame()
            
        df = pd.DataFrame(log_data)
        
        # Extract features from timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Encode categorical features
        categorical_features = ['system', 'level']
        for feature in categorical_features:
            if feature in df.columns:
                if feature not in self.encoders:
                    self.encoders[feature] = LabelEncoder()
                    # Handle new categories by fitting on all known values plus 'unknown'
                    known_values = list(df[feature].unique()) + ['unknown']
                    self.encoders[feature].fit(known_values)
                
                # Transform, handling unseen categories
                df[feature] = df[feature].apply(
                    lambda x: x if x in self.encoders[feature].classes_ else 'unknown'
                )
                df[feature] = self.encoders[feature].transform(df[feature])
        
        # Add numerical features (simplified)
        if 'response_time' not in df.columns:
            df['response_time'] = np.random.exponential(150, len(df))
        
        # Add error indicator
        df['is_error'] = (df['level'] == 'ERROR').astype(int)
        df['is_warning'] = (df['level'] == 'WARN').astype(int)
        
        # Add synthetic metrics based on system
        for system in ['java_app', 'kubernetes', 'cobol_mainframe']:
            system_mask = df['system'] == system
            if system_mask.any():
                df.loc[system_mask, 'cpu_usage'] = np.random.uniform(20, 80, system_mask.sum())
                df.loc[system_mask, 'memory_usage'] = np.random.uniform(30, 85, system_mask.sum())
        
        # Store features
        self.feature_store = pd.concat([self.feature_store, df], ignore_index=True)
        
        return df
    
    def get_timeseries_features(self, window_size=10):
        """Create time-series features from feature store"""
        if len(self.feature_store) < window_size:
            return pd.DataFrame()
        
        return self.feature_store