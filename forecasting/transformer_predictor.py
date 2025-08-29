# forecasting/transformer_predictor.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=3, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(5000, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, 3)  # risk_score, failure_prob, confidence
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.positional_encoding[:seq_len, :, :]
        
        # Transformer expects (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)
        
        # Use the last time step for prediction
        x = x[:, -1, :]
        return self.output_layer(x)

class AdvancedPredictiveAnalyzer:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.scalers = {}
        self.sequence_length = 24  # 24 hours of historical data
        self.training_data = {system: [] for system in ['java_app', 'kubernetes', 'cobol_mainframe']}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize predictive models for each system"""
        systems = ['java_app', 'kubernetes', 'cobol_mainframe']
        
        for system in systems:
            # Transformer-based risk and failure predictor
            input_dim = 10  # Number of features
            self.models[f'{system}_transformer'] = TimeSeriesTransformer(input_dim=input_dim).to(self.device)
            
            # Scaler for features
            self.scalers[system] = StandardScaler()
            
            # Load pre-trained models if available
            self._load_model(system)
    
    def _load_model(self, system):
        """Load pre-trained model if exists"""
        try:
            model_path = f"{self.models_dir}/{system}_transformer.pth"
            self.models[f'{system}_transformer'].load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            print(f"Loaded pre-trained model for {system}")
            
            # Load scaler if available
            scaler_path = f"{self.models_dir}/{system}_scaler.pkl"
            self.scalers[system] = joblib.load(scaler_path)
            print(f"Loaded scaler for {system}")
        except FileNotFoundError:
            print(f"No pre-trained model found for {system}, will train when data is available")
    
    def add_training_data(self, system, features):
        """Add features to training data for a system"""
        if system not in self.training_data:
            self.training_data[system] = []
        
        self.training_data[system].append(features)
        
        # Keep only the most recent data (2x sequence length for training)
        if len(self.training_data[system]) > self.sequence_length * 2:
            self.training_data[system] = self.training_data[system][-self.sequence_length * 2:]
    
    def prepare_training_data(self, system):
        """Prepare training data for transformer model from stored features"""
        if len(self.training_data[system]) < self.sequence_length * 2:
            return None, None
        
        # Extract features
        features = np.array(self.training_data[system])
        
        # Create sequences
        X, y = self._create_sequences(features)
        
        return X, y
    
    def _extract_features(self, system_data):
        """Extract relevant features for prediction from system data"""
        features = []
        
        for _, row in system_data.iterrows():
            feature_vector = [
                row.get('error_count', 0),
                row.get('warning_count', 0),
                row.get('cpu_usage', 0),
                row.get('memory_usage', 0),
                row.get('response_time', 0),
                row.get('is_error', 0),
                row.get('is_warning', 0),
                row.get('hour', 0) / 24.0,  # Normalized hour
                row.get('day_of_week', 0) / 7.0,  # Normalized day of week
                datetime.now().hour / 24.0  # Current hour
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def _create_sequences(self, features):
        """Create input sequences and targets for transformer"""
        X, y = [], []
        
        for i in range(len(features) - self.sequence_length):
            X.append(features[i:i+self.sequence_length])
            
            # Target: next time step's risk score, failure probability, and confidence
            next_features = features[i+self.sequence_length]
            # Simple risk calculation based on features
            risk_score = min(
                next_features[0] * 2.5 +  # error count
                next_features[1] * 1.0 +  # warning count
                (max(0, next_features[2] - 70) * 0.1) +  # high cpu
                (max(0, next_features[3] - 75) * 0.1),  # high memory
                10.0
            )
            
            # Simple failure probability calculation
            failure_prob = min(
                next_features[0] * 0.15 + 
                (max(0, next_features[2] - 80) * 0.01) + 
                (max(0, next_features[3] - 85) * 0.01),
                0.95
            )
            
            confidence = 0.8  # Base confidence
            
            y.append([risk_score, failure_prob, confidence])
        
        return np.array(X), np.array(y)
    
    def train_model(self, system, historical_df=None, *args, **kwargs):
        """Train the transformer model for a specific system with available data"""
        if historical_df is not None:
            pass

        X, y = self.prepare_training_data(system)
        
        if X is None or len(X) == 0:
            return {"status": "skipped", "reason": "not_enough_data"}
        
        # Scale features
        X_reshaped = X.reshape(-1, X.shape[-1])
        self.scalers[system].fit(X_reshaped)
        X_scaled = self.scalers[system].transform(X_reshaped).reshape(X.shape)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Training setup
        model = self.models[f'{system}_transformer']
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Training loop
        model.train()
        for epoch in range(50):  # Reduced epochs for faster training
            optimizer.zero_grad()
            predictions = model(X_tensor)
            loss = criterion(predictions, y_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"System {system}, Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Save model and scaler
        self._save_model(system)
        return True
    
    def predict_risk_and_failure(self, recent_features, system):
        """Predict risk and failure probability using transformer with live data"""
        if len(recent_features) < 2:  # Need at least 2 data points
            return {
                "risk_score": 0.0,
                "failure_probability": 0.0,
                "confidence": 0.0,
                "timeframe": "N/A",
                "critical_metrics": []
            }
        
        # Prepare input sequence from recent features
        features_array = np.array([self._extract_features_row(row) for row in recent_features])
        
        # If we don't have enough data for a full sequence, pad with zeros
        if len(features_array) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(features_array), features_array.shape[1]))
            features_array = np.vstack([padding, features_array])
        else:
            # Use only the most recent data
            features_array = features_array[-self.sequence_length:]
        
        # Add to training data for future training
        for feature_row in features_array:
            self.add_training_data(system, feature_row)
        
        # If we have enough training data, train the model
        if len(self.training_data[system]) >= self.sequence_length * 2:
            self.train_model(system)
        
        # Scale features for prediction
        try:
            features_scaled = self.scalers[system].transform(features_array)
        except:
            # If scaler isn't fitted yet, use raw features with simple normalization
            features_scaled = features_array / np.maximum(1, np.max(np.abs(features_array), axis=0))
        
        # Convert to tensor and predict
        model = self.models[f'{system}_transformer']
        model.eval()
        
        with torch.no_grad():
            input_tensor = torch.FloatTensor(features_scaled).unsqueeze(0).to(self.device)
            prediction = model(input_tensor).cpu().numpy()[0]
        
        risk_score, failure_prob, confidence = prediction
        
        # Ensure values are within valid ranges
        risk_score = max(0, min(10, risk_score))
        failure_prob = max(0, min(1, failure_prob))
        confidence = max(0, min(1, confidence))
        
        # Determine timeframe based on failure probability
        if failure_prob > 0.7:
            timeframe = "1-2 hours"
        elif failure_prob > 0.5:
            timeframe = "4-6 hours"
        elif failure_prob > 0.3:
            timeframe = "12-24 hours"
        else:
            timeframe = "24+ hours"
        
        return {
            "risk_score": risk_score,
            "failure_probability": failure_prob,
            "confidence": confidence,
            "timeframe": timeframe,
            "critical_metrics": self._identify_critical_metrics(recent_features[-1] if recent_features else {})
        }
    
    def _extract_features_row(self, row):
        """Extract features from a single row of system data"""
        return [
            row.get('error_count', 0),
            row.get('warning_count', 0),
            row.get('cpu_usage', 0),
            row.get('memory_usage', 0),
            row.get('response_time', 0),
            row.get('is_error', 0),
            row.get('is_warning', 0),
            datetime.now().hour / 24.0,  # Current hour
            datetime.now().weekday() / 7.0,  # Current day of week
            np.random.random()  # Random noise for variability
        ]
    
    def _identify_critical_metrics(self, current_state):
        """Identify metrics approaching critical thresholds"""
        critical_metrics = []
        
        if current_state.get('cpu_usage', 0) > 85:
            critical_metrics.append({
                "metric": "cpu_usage", 
                "value": current_state['cpu_usage'], 
                "threshold": 85
            })
        
        if current_state.get('memory_usage', 0) > 90:
            critical_metrics.append({
                "metric": "memory_usage", 
                "value": current_state['memory_usage'], 
                "threshold": 90
            })
        
        if current_state.get('error_count', 0) > 5:
            critical_metrics.append({
                "metric": "error_rate", 
                "value": current_state['error_count'], 
                "threshold": 5
            })
        
        return critical_metrics
    
    def _save_model(self, system):
        """Save trained model and scaler"""
        import os
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Save model
        model_path = f"{self.models_dir}/{system}_transformer.pth"
        torch.save(self.models[f'{system}_transformer'].state_dict(), model_path)
        
        # Save scaler
        scaler_path = f"{self.models_dir}/{system}_scaler.pkl"
        joblib.dump(self.scalers[system], scaler_path)
        
        print(f"Model and scaler saved for {system}")
    
    def generate_intelligent_recommendations(self, system, predictions, current_state):
        """Generate AI-driven recommendations using enhanced logic"""
        recommendations = []
        
        # Risk-based recommendations
        if predictions.get('risk_score', 0) > 7.0:
            recommendations.extend(self._get_risk_mitigation_recommendations(system))
        
        # Failure probability recommendations
        if predictions.get('failure_probability', 0) > 0.6:
            recommendations.extend(self._get_failure_prevention_recommendations(system))
        
        # Current state-based recommendations
        recommendations.extend(self._get_state_based_recommendations(system, current_state))
        
        # Add transformer-specific insights
        recommendations.extend(self._get_transformer_insights(system, predictions))
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _get_transformer_insights(self, system, predictions):
        """Get insights based on transformer model predictions"""
        insights = []
        
        if predictions.get('failure_probability', 0) > 0.7:
            insights.append(f"🚨 CRITICAL: Transformer model predicts {predictions['failure_probability']*100:.1f}% chance of failure within {predictions['timeframe']}")
        
        if predictions.get('risk_score', 0) > 8.0:
            insights.append(f"⚠️ HIGH RISK: Transformer model predicts risk score of {predictions['risk_score']:.1f}/10")
        
        return insights
    
    def _get_risk_mitigation_recommendations(self, system):
        """Get risk mitigation recommendations"""
        recommendations = []
        
        base_recommendations = [
            "Implement additional monitoring for critical metrics",
            "Create automated remediation runbooks",
            "Increase alerting thresholds for early detection",
            "Schedule immediate system health check",
            "Prepare rollback procedures for quick recovery"
        ]
        
        system_specific = {
            'java_app': [
                "Increase JVM monitoring frequency",
                "Add circuit breakers for external dependencies",
                "Implement graceful degradation"
            ],
            'kubernetes': [
                "Add pod disruption budgets",
                "Implement node affinity/anti-affinity rules",
                "Set up cluster autoscaling"
            ],
            'cobol_mainframe': [
                "Increase storage monitoring frequency",
                "Implement job queue management",
                "Add batch job prioritization"
            ]
        }
        
        recommendations.extend(base_recommendations)
        recommendations.extend(system_specific.get(system, []))
        
        return recommendations
    
    def _get_failure_prevention_recommendations(self, system):
        """Get failure prevention recommendations"""
        recommendations = []
        
        system_specific = {
            'java_app': [
                "Increase JVM heap size to prevent OutOfMemory errors",
                "Restart application to clear memory leaks",
                "Scale out application instances to distribute load",
                "Check database connection pool configuration",
                "Review recent deployment for potential issues"
            ],
            'kubernetes': [
                "Scale deployment to add more replicas",
                "Increase resource limits for pods",
                "Check node resource availability",
                "Restart failing pods",
                "Review horizontal pod autoscaler configuration"
            ],
            'cobol_mainframe': [
                "Increase storage allocation for batch jobs",
                "Optimize job scheduling to reduce contention",
                "Check tape drive availability and performance",
                "Review dataset compression and cleanup procedures",
                "Verify mainframe system resource availability"
            ]
        }
        
        return system_specific.get(system, [
            "Investigate root cause of predicted failure",
            "Implement immediate monitoring enhancements",
            "Prepare failover procedures"
        ])
    
    def _get_state_based_recommendations(self, system, current_state):
        """Get recommendations based on current system state"""
        recommendations = []
        
        if current_state.get('high_cpu_usage', False):
            recommendations.append(f"Optimize CPU usage for {system} - current usage: {current_state.get('cpu_usage', 0)}%")
        
        if current_state.get('high_memory_usage', False):
            recommendations.append(f"Optimize memory usage for {system} - current usage: {current_state.get('memory_usage', 0)}%")
        
        if current_state.get('high_error_rate', False):
            recommendations.append(f"Investigate error rate spike for {system} - current errors: {current_state.get('error_count', 0)}")
        
        return recommendations