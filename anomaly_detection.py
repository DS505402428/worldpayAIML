import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
import json
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

class AnomalyDetector:
    def __init__(self):
        self.prometheus_url = "http://localhost:9090/api/v1/query"
        self.ollama_url = "http://localhost:11434/api/generate"
        self.models = {}
        self.metric_baselines = {}  # Store baseline stats for each metric
        self.check_count = 0
        
    def query_prometheus(self, query):
        """Query Prometheus for metrics"""
        try:
            response = requests.get(self.prometheus_url, params={'query': query})
            response.raise_for_status()
            data = response.json()
            return data['data']['result']
        except Exception as e:
            print(f"Error querying Prometheus: {e}")
            return []
    
    def analyze_with_llm(self, system_name, metric_name, timestamp, value, normal_range):
        """Call LLM to analyze logs when an anomaly is detected"""
        prompt = f"""
        At {timestamp}, our monitoring system detected a potential anomaly in {system_name}. 
        The metric '{metric_name}' had a value of {value:.2f}, which deviates from its normal range of {normal_range}.
        
        Please act as an expert SRE and provide a concise analysis:
        1. What kind of problem could this indicate for a {system_name}?
        2. What would be a likely root cause?
        3. Suggest one immediate remediation step.
        
        Please provide a very concise analysis in the following format:
        PROBLEM: [One sentence]
        ROOT CAUSE: [One sentence] 
        ACTION: [One concrete action]
        """
        
        try:
            data = {
                "model": "llama3.1",
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(self.ollama_url, json=data)
            response.raise_for_status()
            result = response.json()
            
            print(f"\n🔍 **LLM Analysis for {system_name} anomaly:**")
            print(result['response'])
            print("-" * 60)
            
        except Exception as e:
            print(f"Could not get LLM analysis: {e}")
    
    def analyze_predicted_anomaly(self, metric_name, anomaly_info):
        """Get LLM analysis for predicted future anomaly"""
        prompt = f"""
        Our forecasting model predicts that at {anomaly_info['time']}, the metric '{metric_name}' 
        will reach {anomaly_info['predicted_value']:.2f}, which exceeds the normal threshold of {anomaly_info['threshold']:.2f}.
        
        As an expert SRE, please provide:
        1. What this predicted anomaly might indicate
        2. Potential preemptive actions to prevent this
        3. How urgently we should act
        
        Format your response as:
        PREDICTED ISSUE: [One sentence]
        PREVENTATIVE ACTION: [One sentence]
        URGENCY: [Low/Medium/High]
        """
        
        try:
            data = {
                "model": "llama3.1",
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(self.ollama_url, json=data)
            response.raise_for_status()
            result = response.json()
            
            print(f"\n🔮 **Predictive Analysis for {metric_name}:**")
            print(result['response'])
            print("-" * 60)
            
        except Exception as e:
            print(f"Could not get predictive analysis: {e}")
    
    def calculate_baseline_stats(self, values):
        """Calculate baseline statistics for a metric"""
        if len(values) < 10:  # Need enough data for meaningful stats
            return "insufficient data"
        
        mean = np.mean(values)
        std = np.std(values)
        return f"{mean-2*std:.1f}-{mean+2*std:.1f} (mean: {mean:.1f})"
    
    def detect_anomalies(self, metric_name, lookback_minutes=15):
        """Detect anomalies in a specific metric"""
        # Query data from Prometheus
        query = f'{metric_name}[{lookback_minutes}m]'
        data = self.query_prometheus(query)
        
        if not data:
            return []
        
        # Extract values and timestamps
        values = []
        timestamps = []
        for result in data:
            for value in result['values']:
                timestamps.append(datetime.fromtimestamp(value[0]))
                values.append(float(value[1]))
        
        # Update baseline statistics
        if metric_name not in self.metric_baselines or len(values) > 20:
            self.metric_baselines[metric_name] = self.calculate_baseline_stats(values)
        
        # Prepare data for anomaly detection
        df = pd.DataFrame({
            'timestamp': timestamps,
            'value': values
        })
        df = df.set_index('timestamp')
        
        # Skip detection if not enough data
        if len(df) < 20:
            return []
        
        # Train anomaly detection model (less sensitive)
        X = df['value'].values.reshape(-1, 1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if metric_name not in self.models:
            self.models[metric_name] = IsolationForest(
                contamination=0.05,  # More conservative (5% anomalies)
                random_state=42,
                n_estimators=100
            )
        
        model = self.models[metric_name]
        model.fit(X_scaled)
        
        # Detect anomalies
        df['anomaly_score'] = model.decision_function(X_scaled)
        df['anomaly'] = model.predict(X_scaled)
        df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})  # 0=normal, 1=anomaly
        
        # Only consider strong anomalies (very negative scores)
        strong_anomalies = df[(df['anomaly'] == 1) & (df['anomaly_score'] < -0.2)]
        
        # Print results
        if len(strong_anomalies) > 0:
            print(f"\n=== Anomaly Detection Results for {metric_name} ===")
            print(f"Total data points: {len(df)}")
            print(f"Normal range: {self.metric_baselines[metric_name]}")
            print(f"Strong anomalies detected: {len(strong_anomalies)}")
            
            print("Anomaly details:")
            for idx, row in strong_anomalies.iterrows():
                print(f"  Time: {idx}, Value: {row['value']:.2f}, Score: {row['anomaly_score']:.3f}")
                
                # Determine which system this metric belongs to
                if 'system1' in metric_name:
                    system_name = 'Java Application (System 1)'
                elif 'system2' in metric_name:
                    system_name = 'Kubernetes Platform (System 2)'
                elif 'system3' in metric_name:
                    system_name = 'COBOL Mainframe (System 3)'
                else:
                    system_name = 'Unknown System'
                
                # Get LLM analysis for this anomaly
                self.analyze_with_llm(
                    system_name, 
                    metric_name, 
                    idx, 
                    row['value'],
                    self.metric_baselines[metric_name]
                )
        
        return strong_anomalies
    
    def predict_future_anomalies(self, metric_name, forecast_minutes=30):
        """Predict potential future anomalies using time-series forecasting"""
        # Query historical data
        query = f'{metric_name}[60m]'  # 60 minutes of historical data
        data = self.query_prometheus(query)
        
        if not data or len(data[0]['values']) < 30:
            return []  # Not enough data
        
        # Extract values and timestamps
        values = []
        timestamps = []
        for result in data:
            for value in result['values']:
                timestamps.append(datetime.fromtimestamp(value[0]))
                values.append(float(value[1]))
        
        # Create time series dataframe
        df = pd.DataFrame({
            'timestamp': timestamps,
            'value': values
        })
        df = df.set_index('timestamp')
        
        try:
            # Fit model
            model = SimpleExpSmoothing(df['value']).fit()
            
            # Forecast future values
            forecast = model.forecast(forecast_minutes)
            
            # Check if forecasted values exceed anomaly thresholds
            current_mean = df['value'].mean()
            current_std = df['value'].std()
            
            # Define threshold (2 standard deviations from mean)
            upper_threshold = current_mean + (2 * current_std)
            lower_threshold = current_mean - (2 * current_std)
            
            future_anomalies = []
            for i, value in enumerate(forecast):
                if value > upper_threshold or value < lower_threshold:
                    future_time = datetime.now() + timedelta(minutes=i+1)
                    future_anomalies.append({
                        'time': future_time,
                        'predicted_value': value,
                        'threshold': upper_threshold if value > upper_threshold else lower_threshold
                    })
            
            return future_anomalies
            
        except Exception as e:
            print(f"Forecasting error for {metric_name}: {e}")
            return []
    
    def run_continuous_detection(self):
        """Continuously monitor for anomalies"""
        metrics_to_monitor = [
            'system1_cpu_usage',
            'system1_memory_usage',
            'system2_cpu_usage',
            'system2_pod_count',
            'system3_cpu_usage',
            'system3_jobs_completed'
        ]
        
        print("🤖 AI Anomaly Detection System Initializing...")
        print("📊 Building baseline models (this may take a few minutes)...")
        
        # Wait to collect baseline data
        baseline_minutes = 5
        print(f"⏳ Collecting {baseline_minutes} minutes of baseline data...")
        for i in range(baseline_minutes):
            print(f"   {baseline_minutes - i} minutes remaining...")
            time.sleep(60)  # Wait 1 minute between checks
        
        print("✅ Baseline data collected. Starting continuous monitoring...")
        print("Monitoring metrics:", metrics_to_monitor)
        print("\n" + "="*70)
        
        self.check_count = 0
        
        while True:
            self.check_count += 1
            print(f"\n🕒 Check #{self.check_count} at {datetime.now()}")
            print("="*70)
            
            anomaly_count = 0
            
            # 1. Detect current anomalies
            for metric in metrics_to_monitor:
                anomalies = self.detect_anomalies(metric)
                anomaly_count += len(anomalies)
            
            # 2. PREDICT future anomalies (every 5 checks)
            if self.check_count % 5 == 0:  # Run prediction every 5 minutes
                print("\n🔮 PREDICTIVE ANALYSIS: Forecasting next 30 minutes...")
                for metric in metrics_to_monitor:
                    future_anomalies = self.predict_future_anomalies(metric)
                    if future_anomalies:
                        print(f"\n⚠️  PREDICTED FUTURE ANOMALIES for {metric}:")
                        for anomaly in future_anomalies:
                            print(f"   Time: {anomaly['time']}, Predicted Value: {anomaly['predicted_value']:.2f}")
                            
                            # Get LLM analysis for predicted anomaly
                            self.analyze_predicted_anomaly(metric, anomaly)
            
            if anomaly_count == 0 and self.check_count % 5 != 0:
                print("✅ All systems normal - no anomalies detected")
            
            time.sleep(60)  # Check every 60 seconds

if __name__ == "__main__":
    detector = AnomalyDetector()
    detector.run_continuous_detection()