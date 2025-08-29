# visualization/dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

class ObservabilityDashboard:
    def __init__(self):
        st.set_page_config(page_title="AI Observability Platform", layout="wide")
    
    def display_overview(self, risk_scores, anomalies):
        """Display system overview"""
        st.title("AI Observability Platform Dashboard")
        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Systems", len(risk_scores))
        
        with col2:
            total_anomalies = sum(len(anoms) for anoms in anomalies.values() if anoms is not None)
            st.metric("Active Anomalies", total_anomalies)
        
        with col3:
            avg_risk = sum(risk_scores.values()) / len(risk_scores) if risk_scores else 0
            st.metric("Average Risk Score", f"{avg_risk:.2f}/10")
        
        with col4:
            st.metric("Last Updated", datetime.now().strftime("%H:%M:%S"))
        
        # Display risk scores
        st.subheader("System Risk Scores")
        risk_df = pd.DataFrame.from_dict(risk_scores, orient='index', columns=['Risk Score'])
        st.bar_chart(risk_df)
    
    def display_anomalies(self, anomalies):
        """Display anomalies table"""
        st.subheader("Recent Anomalies")
        
        anomaly_list = []
        for system, anomaly_data in anomalies.items():
            if anomaly_data and 'scores' in anomaly_data:
                for i, score in enumerate(anomaly_data['scores']):
                    if score > 1.0:  # Only show actual anomalies
                        anomaly_list.append({
                            'System': system,
                            'Score': score,
                            'Timestamp': datetime.now() - timedelta(minutes=len(anomaly_data['scores'])-i)
                        })
        
        if anomaly_list:
            anomaly_df = pd.DataFrame(anomaly_list)
            st.dataframe(anomaly_df.sort_values('Score', ascending=False))
        else:
            st.info("No anomalies detected in the current data")
    
    def display_forecasts(self, forecast_results):
        """Display forecasting results"""
        st.subheader("System Forecasts")
        
        for system, metrics in forecast_results.items():
            with st.expander(f"Forecasts for {system}"):
                for metric, data in metrics.items():
                    if 'prophet_forecast' in data:
                        # Create forecast plot
                        forecast_df = data['prophet_forecast'][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(48)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], 
                                                name='Forecast', line=dict(color='blue')))
                        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], 
                                                fill=None, mode='lines', line_color='lightblue', 
                                                name='Upper Bound'))
                        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], 
                                                fill='tonexty', mode='lines', line_color='lightblue', 
                                                name='Lower Bound'))
                        fig.update_layout(title=f"{metric} Forecast for {system}")
                        st.plotly_chart(fig)
    
    def display_rca_results(self, rca_results):
        """Display RCA results"""
        st.subheader("Root Cause Analysis")
        
        for i, result in enumerate(rca_results):
            with st.expander(f"RCA Result {i+1} - {result['timestamp']}"):
                st.json(result['rca_result'])
    
    def run(self, risk_scores, anomalies, forecast_results, rca_results):
        """Run the dashboard"""
        self.display_overview(risk_scores, anomalies)
        self.display_anomalies(anomalies)
        self.display_forecasts(forecast_results)
        self.display_rca_results(rca_results)