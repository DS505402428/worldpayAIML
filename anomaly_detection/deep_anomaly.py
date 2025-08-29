# anomaly_detection/deep_anomaly.py
import pandas as pd
import numpy as np
import re
from datetime import datetime

class EnhancedAnomalyDetector:
    def __init__(self):
        self.known_patterns = {
            'java_app': {
                'NullPointerException': r'NullPointerException',
                'OutOfMemoryError': r'OutOfMemoryError',
                'DatabaseConnectionTimeout': r'Database connection timeout',
                'HighMemoryUsage': r'High memory usage detected',
                'SlowQueries': r'Slow database queries'
            },
            'kubernetes': {
                'PodCrashOOM': r'Pod crashed due to OOM',
                'CPUThrottling': r'CPU throttling detected',
                'ImagePullFailed': r'Image pull failed',
                'NodeNotReady': r'Node not ready',
                'NetworkLatency': r'Network latency increased'
            },
            'cobol_mainframe': {
                'JobAbend': r'ABEND:.*',
                'StorageWarning': r'Storage allocation nearing limit',
                'TapeDriveSlow': r'Tape drive response slow',
                'DatasetMountFailed': r'Dataset mount failed',
                'CICSAbend': r'CICS transaction abended'
            }
        }
    
    def detect_specific_anomalies(self, logs_df, metric_anomaly=False):
        """Detect specific anomalies from logs and metrics"""
        anomalies = {}
        
        # Process each system's logs
        for system in ['java_app', 'kubernetes', 'cobol_mainframe']:
            system_logs = logs_df[logs_df['system'] == system]
            if len(system_logs) == 0:
                continue
                
            # Check for specific error patterns
            for _, log in system_logs.iterrows():
                if log['level'] in ['ERROR', 'WARN']:
                    detected_issue = self._classify_issue(log['message'], system)
                    if detected_issue and system not in anomalies:
                        anomalies[system] = {
                            'type': detected_issue,
                            'message': log['message'],
                            'timestamp': log['timestamp'],
                            'score': self._calculate_severity_score(detected_issue, log['level']),
                            'system': system
                        }
        
        # Handle metric anomalies with specific context
        if metric_anomaly:
            # Determine which system has the metric anomaly
            metric_context = self._get_metric_context(logs_df)
            if metric_context:
                anomalies['metrics'] = {
                    'type': f'Metric_Anomaly_{metric_context}',
                    'message': f'Abnormal metric values detected in {metric_context}',
                    'timestamp': datetime.now().isoformat(),
                    'score': 8.0,
                    'system': metric_context
                }
            else:
                anomalies['metrics'] = {
                    'type': 'Metric_Anomaly_General',
                    'message': 'Abnormal metric values detected across systems',
                    'timestamp': datetime.now().isoformat(),
                    'score': 7.0,
                    'system': 'general'
                }
        
        return anomalies
    
    def _classify_issue(self, message, system):
        """Classify the specific issue from the log message"""
        if system not in self.known_patterns:
            return None
            
        for issue_type, pattern in self.known_patterns[system].items():
            if re.search(pattern, message, re.IGNORECASE):
                return issue_type
                
        return None
    
    def _calculate_severity_score(self, issue_type, log_level):
        """Calculate severity score based on issue type and log level"""
        base_scores = {
            'NullPointerException': 8.0,
            'OutOfMemoryError': 9.0,
            'DatabaseConnectionTimeout': 7.5,
            'PodCrashOOM': 9.0,
            'CPUThrottling': 7.0,
            'JobAbend': 8.5,
            'StorageWarning': 6.5
        }
        
        # Default score if issue type not in base_scores
        score = base_scores.get(issue_type, 7.0)
        
        # Adjust based on log level
        if log_level == 'WARN':
            score *= 0.7  # Reduce score for warnings
        
        return min(score, 10.0)  # Cap at 10.0
    
    def _get_metric_context(self, logs_df):
        """Determine which system is most likely causing metric anomalies"""
        # Look for recent errors to provide context
        recent_errors = {}
        
        for system in ['java_app', 'kubernetes', 'cobol_mainframe']:
            system_logs = logs_df[logs_df['system'] == system]
            error_count = len(system_logs[system_logs['level'] == 'ERROR'])
            recent_errors[system] = error_count
        
        # Return the system with the most recent errors
        if recent_errors:
            return max(recent_errors.items(), key=lambda x: x[1])[0]
        
        return None