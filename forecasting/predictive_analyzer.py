# forecasting/predictive_analyzer.py
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PredictiveAnalyzer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.forecast_results = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize predictive models for each system"""
        systems = ['java_app', 'kubernetes', 'cobol_mainframe']
        
        for system in systems:
            # Time series forecasting model
            self.models[f'{system}_prophet'] = Prophet(
                yearly_seasonality=False,
                weekly_seasonality=True,
                daily_seasonality=True,
                changepoint_prior_scale=0.05
            )
            
            # Failure prediction model (classification)
            self.models[f'{system}_failure_predictor'] = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
            
            # Risk forecasting model
            self.models[f'{system}_risk_predictor'] = RandomForestClassifier(
                n_estimators=50,
                random_state=42
            )
            
            self.scalers[system] = StandardScaler()
    
    def prepare_timeseries_data(self, historical_data, target_metric, system):
        """Prepare time series data for forecasting"""
        if historical_data.empty:
            return pd.DataFrame()
            
        system_data = historical_data[historical_data['system'] == system]
        if len(system_data) < 24:  # Minimum data points
            return pd.DataFrame()
            
        ts_data = system_data[['timestamp', target_metric]].copy()
        ts_data = ts_data.set_index('timestamp').resample('1H').mean().ffill()
        
        return ts_data
    
    def train_prophet_forecast(self, ts_data, periods=24):
        """Train Facebook Prophet model for forecasting"""
        prophet_df = ts_data.reset_index()
        prophet_df.columns = ['ds', 'y']
        
        try:
            model = Prophet(
                yearly_seasonality=False,
                weekly_seasonality=True,
                daily_seasonality=True
            )
            model.fit(prophet_df)
            
            future = model.make_future_dataframe(periods=periods, freq='H')
            forecast = model.predict(future)
            
            return model, forecast
        except Exception as e:
            print(f"Prophet training failed: {e}")
            return None, None
    
    def predict_failures(self, features, system):
        """Predict potential failures for a system"""
        if features.empty:
            return {"probability": 0.0, "confidence": 0.0, "timeframe": "N/A"}
        
        # Prepare features for prediction
        prediction_features = self._extract_prediction_features(features, system)
        
        if prediction_features.empty:
            return {"probability": 0.0, "confidence": 0.0, "timeframe": "N/A"}
        
        # Scale features
        scaled_features = self.scalers[system].transform(prediction_features)
        
        # Predict failure probability
        model_key = f'{system}_failure_predictor'
        if model_key in self.models:
            probabilities = self.models[model_key].predict_proba(scaled_features)
            failure_prob = probabilities[0][1]  # Probability of failure class
            
            confidence = min(failure_prob * 1.5, 0.95)  # Adjust confidence
            
            # Determine timeframe based on probability
            if failure_prob > 0.7:
                timeframe = "1-2 hours"
            elif failure_prob > 0.5:
                timeframe = "4-6 hours" 
            elif failure_prob > 0.3:
                timeframe = "12-24 hours"
            else:
                timeframe = "24+ hours"
            
            return {
                "probability": round(failure_prob, 3),
                "confidence": round(confidence, 3),
                "timeframe": timeframe,
                "critical_metrics": self._identify_critical_metrics(features, system)
            }
        
        return {"probability": 0.0, "confidence": 0.0, "timeframe": "N/A"}
    
    def forecast_risk(self, features, system):
        """Forecast risk score for the next period"""
        if features.empty:
            return {"risk_score": 0.0, "trend": "stable", "factors": []}
        
        prediction_features = self._extract_prediction_features(features, system)
        
        if prediction_features.empty:
            return {"risk_score": 0.0, "trend": "stable", "factors": []}
        
        # Scale features
        scaled_features = self.scalers[system].transform(prediction_features)
        
        # Predict risk score (0-10 scale)
        model_key = f'{system}_risk_predictor'
        if model_key in self.models:
            risk_score = self.models[model_key].predict(scaled_features)[0]
            risk_score = max(0, min(10, risk_score))  # Clamp to 0-10
            
            # Determine trend
            current_risk = self._calculate_current_risk(features, system)
            trend = "increasing" if risk_score > current_risk else "decreasing" if risk_score < current_risk else "stable"
            
            return {
                "risk_score": round(risk_score, 2),
                "trend": trend,
                "factors": self._identify_risk_factors(features, system),
                "confidence": 0.85  # Base confidence
            }
        
        return {"risk_score": 0.0, "trend": "stable", "factors": []}
    
    def generate_recommendations(self, system, current_state, predictions):
        """Generate AI-driven recommendations"""
        recommendations = []
        
        # High failure probability recommendations
        if predictions.get('failure_probability', 0) > 0.6:
            recommendations.extend(self._get_failure_prevention_recommendations(system))
        
        # High risk score recommendations
        if predictions.get('risk_score', 0) > 7.0:
            recommendations.extend(self._get_risk_mitigation_recommendations(system))
        
        # System-specific recommendations
        recommendations.extend(self._get_system_specific_recommendations(system, current_state))
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _extract_prediction_features(self, features, system):
        """Extract features for prediction models"""
        system_data = features[features['system'] == system]
        
        if system_data.empty:
            return pd.DataFrame()
        
        # Calculate features for prediction
        prediction_features = pd.DataFrame()
        
        # Error rates and patterns
        prediction_features['error_count_1h'] = len(system_data[system_data['level'] == 'ERROR'])
        prediction_features['warning_count_1h'] = len(system_data[system_data['level'] == 'WARN'])
        
        # Resource utilization trends
        if 'cpu_usage' in system_data.columns:
            prediction_features['cpu_mean'] = system_data['cpu_usage'].mean()
            prediction_features['cpu_std'] = system_data['cpu_usage'].std()
        
        if 'memory_usage' in system_data.columns:
            prediction_features['memory_mean'] = system_data['memory_usage'].mean()
            prediction_features['memory_std'] = system_data['memory_usage'].std()
        
        # Time-based features
        prediction_features['hour'] = pd.to_datetime(system_data['timestamp'].iloc[0]).hour
        prediction_features['day_of_week'] = pd.to_datetime(system_data['timestamp'].iloc[0]).dayofweek
        
        return prediction_features
    
    def _calculate_current_risk(self, features, system):
        """Calculate current risk score based on features"""
        system_data = features[features['system'] == system]
        
        if system_data.empty:
            return 0.0
        
        error_count = len(system_data[system_data['level'] == 'ERROR'])
        warning_count = len(system_data[system_data['level'] == 'WARN'])
        
        # Simple risk calculation (can be enhanced)
        risk_score = min(error_count * 2 + warning_count * 1, 10)
        return risk_score
    
    def _identify_risk_factors(self, features, system):
        """Identify main factors contributing to risk"""
        system_data = features[features['system'] == system]
        factors = []
        
        if system_data.empty:
            return factors
        
        # Check for high error rates
        error_count = len(system_data[system_data['level'] == 'ERROR'])
        if error_count > 3:
            factors.append(f"High error rate ({error_count} errors in last hour)")
        
        # Check resource utilization
        if 'cpu_usage' in system_data.columns and system_data['cpu_usage'].mean() > 80:
            factors.append("High CPU utilization")
        
        if 'memory_usage' in system_data.columns and system_data['memory_usage'].mean() > 85:
            factors.append("High memory utilization")
        
        return factors
    
    def _identify_critical_metrics(self, features, system):
        """Identify metrics that are approaching critical thresholds"""
        system_data = features[features['system'] == system]
        critical_metrics = []
        
        if system_data.empty:
            return critical_metrics
        
        # CPU critical check
        if 'cpu_usage' in system_data.columns:
            cpu_avg = system_data['cpu_usage'].mean()
            if cpu_avg > 85:
                critical_metrics.append({"metric": "cpu_usage", "value": cpu_avg, "threshold": 85})
        
        # Memory critical check
        if 'memory_usage' in system_data.columns:
            memory_avg = system_data['memory_usage'].mean()
            if memory_avg > 90:
                critical_metrics.append({"metric": "memory_usage", "value": memory_avg, "threshold": 90})
        
        # Error rate critical check
        error_rate = len(system_data[system_data['level'] == 'ERROR']) / len(system_data)
        if error_rate > 0.1:  # 10% error rate
            critical_metrics.append({"metric": "error_rate", "value": error_rate*100, "threshold": 10})
        
        return critical_metrics
    
    def _get_failure_prevention_recommendations(self, system):
        """Get recommendations for failure prevention"""
        recommendations = []
        
        if system == 'java_app':
            recommendations = [
                "Increase JVM heap size to prevent OutOfMemory errors",
                "Restart application to clear memory leaks",
                "Scale out application instances to distribute load",
                "Check database connection pool configuration",
                "Review recent deployment for potential issues"
            ]
        elif system == 'kubernetes':
            recommendations = [
                "Scale deployment to add more replicas",
                "Increase resource limits for pods",
                "Check node resource availability",
                "Restart failing pods",
                "Review horizontal pod autoscaler configuration"
            ]
        elif system == 'cobol_mainframe':
            recommendations = [
                "Increase storage allocation for batch jobs",
                "Optimize job scheduling to reduce contention",
                "Check tape drive availability and performance",
                "Review dataset compression and cleanup procedures",
                "Verify mainframe system resource availability"
            ]
        
        return recommendations
    
    def _get_risk_mitigation_recommendations(self, system):
        """Get recommendations for risk mitigation"""
        recommendations = []
        
        base_recommendations = [
            "Implement additional monitoring for critical metrics",
            "Create automated remediation runbooks",
            "Increase alerting thresholds for early detection",
            "Schedule immediate system health check",
            "Prepare rollback procedures for quick recovery"
        ]
        
        return base_recommendations
    
    def _get_system_specific_recommendations(self, system, current_state):
        """Get system-specific recommendations based on current state"""
        recommendations = []
        
        # Add system-specific logic here based on current_state
        if system == 'java_app' and current_state.get('high_memory_usage', False):
            recommendations.append("Trigger garbage collection and analyze heap dump")
        
        if system == 'kubernetes' and current_state.get('pod_restarts', 0) > 5:
            recommendations.append("Investigate pod crash logs and resource constraints")
        
        return recommendations
    
    def save_models(self, models_dir="models"):
        """Save trained models to disk"""
        import os
        os.makedirs(models_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            joblib.dump(model, f"{models_dir}/{model_name}.joblib")
        
        for scaler_name, scaler in self.scalers.items():
            joblib.dump(scaler, f"{models_dir}/{scaler_name}_scaler.joblib")
    
    def load_models(self, models_dir="models"):
        """Load trained models from disk"""
        import os
        import glob
        
        model_files = glob.glob(f"{models_dir}/*.joblib")
        
        for model_file in model_files:
            model_name = os.path.basename(model_file).replace('.joblib', '')
            if 'scaler' in model_name:
                self.scalers[model_name.replace('_scaler', '')] = joblib.load(model_file)
            else:
                self.models[model_name] = joblib.load(model_file)