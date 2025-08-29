# feedback_loop/model_retrainer.py
import pandas as pd
import numpy as np
import json
import os
import pickle
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRetrainer:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.performance_history = {}
        self.feedback_data = []
    
    def save_feedback(self, anomaly_info, resolution_effective, feedback_notes=""):
        """Save feedback for model retraining"""
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'anomaly_info': anomaly_info,
            'resolution_effective': resolution_effective,
            'feedback_notes': feedback_notes,
            'system': anomaly_info.get('system', 'unknown')
        }
        
        self.feedback_data.append(feedback_entry)
        
        # Save to file
        feedback_file = os.path.join(self.models_dir, "feedback_data.jsonl")
        with open(feedback_file, 'a') as f:
            f.write(json.dumps(feedback_entry) + '\n')
        
        logger.info(f"Feedback saved: {resolution_effective} for {anomaly_info.get('type', 'unknown')}")
        
        # Check if we have enough data to retrain
        if len(self.feedback_data) >= 10:  # Retrain after 10 feedback entries
            self.retrain_models()
    
    def prepare_training_data(self, features_df):
        """Prepare training data from features and feedback"""
        if not self.feedback_data or features_df.empty:
            return None, None
        
        # Create labeled data based on feedback
        X = []
        y = []
        
        for feedback in self.feedback_data:
            system = feedback['system']
            anomaly_type = feedback['anomaly_info'].get('type', '')
            is_effective = feedback['resolution_effective']
            
            # Get features for this system around the anomaly time
            anomaly_time = datetime.fromisoformat(feedback['anomaly_info'].get('timestamp', ''))
            time_window_start = anomaly_time - timedelta(minutes=30)
            time_window_end = anomaly_time + timedelta(minutes=30)
            
            # Filter features for the time window and system
            system_features = features_df[
                (features_df['system'] == system) & 
                (features_df['timestamp'] >= time_window_start.isoformat()) & 
                (features_df['timestamp'] <= time_window_end.isoformat())
            ]
            
            if not system_features.empty:
                # Use the feature vector closest to the anomaly time
                closest_idx = (pd.to_datetime(system_features['timestamp']) - anomaly_time).abs().idxmin()
                feature_vector = system_features.loc[closest_idx]
                
                # Extract numerical features
                numerical_features = []
                for col in feature_vector.index:
                    if pd.api.types.is_numeric_dtype(feature_vector[col]):
                        numerical_features.append(feature_vector[col])
                
                if numerical_features:
                    X.append(numerical_features)
                    y.append(1 if is_effective else 0)  # 1 = effective, 0 = not effective
        
        return np.array(X), np.array(y)
    
    def save_model(self, model, model_name, system, version=None):
        """Save model with versioning"""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = os.path.join(self.models_dir, system, model_name)
        os.makedirs(model_path, exist_ok=True)
        
        # Save model
        model_file = os.path.join(model_path, f"{version}.pkl")
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'system': system,
            'model_name': model_name,
            'performance': self.performance_history.get(system, {}).get(model_name, {})
        }
        
        with open(os.path.join(model_path, f"{version}_metadata.json"), 'w') as f:
            json.dump(metadata, f)
        
        logger.info(f"Model saved: {model_file}")
    
    def load_model(self, system, model_name, version="latest"):
        """Load a trained model"""
        model_path = os.path.join(self.models_dir, system, model_name)
        
        if version == "latest":
            # Find the latest version
            versions = [f for f in os.listdir(model_path) if f.endswith('.pkl')]
            if not versions:
                return None
            versions.sort(reverse=True)
            version = versions[0].replace('.pkl', '')
        
        model_file = os.path.join(model_path, f"{version}.pkl")
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            return model
        except FileNotFoundError:
            logger.warning(f"Model not found: {model_file}")
            return None
    
    def evaluate_model(self, y_true, y_pred, model_name, system):
        """Evaluate model performance"""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'timestamp': datetime.now().isoformat()
        }
        
        if system not in self.performance_history:
            self.performance_history[system] = {}
        
        if model_name not in self.performance_history[system]:
            self.performance_history[system][model_name] = []
        
        self.performance_history[system][model_name].append(metrics)
        
        # Keep only last 100 evaluations
        if len(self.performance_history[system][model_name]) > 100:
            self.performance_history[system][model_name] = self.performance_history[system][model_name][-100:]
        
        return metrics
    
    def check_retrain_need(self, system, model_name, threshold=0.1):
        """Check if model needs retraining based on performance degradation"""
        if system not in self.performance_history or model_name not in self.performance_history[system]:
            return False
        
        history = self.performance_history[system][model_name]
        if len(history) < 10:  # Need enough history
            return False
        
        # Check if recent performance is significantly worse than historical
        recent_perf = np.mean([h['accuracy'] for h in history[-5:]])
        historical_perf = np.mean([h['accuracy'] for h in history[:-5]])
        
        return (historical_perf - recent_perf) > threshold
    
    def retrain_models(self, features_df=None):
        """Retrain models based on feedback data"""
        if not self.feedback_data:
            logger.warning("No feedback data available for retraining")
            return {"status": "no_data", "samples": 0}
        
        # Prepare training data
        X, y = self.prepare_training_data(features_df)
        
        if X is None or len(X) < 10:  # Need minimum samples
            logger.warning(f"Insufficient training data: {len(X) if X is not None else 0} samples")
            return {"status": "insufficient_data", "samples": len(X) if X is not None else 0}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train a simple model (in practice, you'd use your actual model)
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        metrics = self.evaluate_model(y_test, y_pred, "effectiveness_predictor", "global")
        
        # Save model
        self.save_model(model, "effectiveness_predictor", "global")
        
        logger.info(f"Model retrained with {len(X)} samples. Accuracy: {metrics['accuracy']:.3f}")
        
        return {"status": "retrained", "samples": len(X), "accuracy": metrics['accuracy']}