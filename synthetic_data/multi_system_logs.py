# synthetic_data/multi_system_logs.py
import random
import time
import json
from datetime import datetime
from faker import Faker
import numpy as np
from prometheus_client import start_http_server, Gauge, Counter, Histogram

fake = Faker()

class SyntheticSystemLogs:
    def __init__(self, verbose=True):
        self.verbose = verbose
        # Initialize metrics
        self.metrics = {
            'java_app_cpu_usage': Gauge('java_app_cpu_usage', 'CPU usage for Java application'),
            'java_app_memory_usage': Gauge('java_app_memory_usage', 'Memory usage for Java application'),
            'java_app_error_count': Counter('java_app_error_count', 'Error count for Java application'),
            'k8s_cpu_usage': Gauge('k8s_cpu_usage', 'CPU usage for Kubernetes cluster'),
            'k8s_memory_usage': Gauge('k8s_memory_usage', 'Memory usage for Kubernetes cluster'),
            'k8s_pod_count': Gauge('k8s_pod_count', 'Pod count in Kubernetes cluster'),
            'cobol_cpu_usage': Gauge('cobol_cpu_usage', 'CPU usage for COBOL mainframe'),
            'cobol_jobs_processed': Counter('cobol_jobs_processed', 'Jobs processed by COBOL mainframe'),
            'cobol_error_count': Counter('cobol_error_count', 'Error count for COBOL mainframe')
        }
    
    def _generate_java_logs(self):
        level = random.choices(['INFO', 'WARN', 'ERROR'], weights=[0.7, 0.2, 0.1])[0]
        message = ''
        
        if level == 'ERROR':
            message = random.choice([
                'Database connection timeout',
                'OutOfMemoryError: Java heap space',
                'NullPointerException in service layer'
            ])
            self.metrics['java_app_error_count'].inc()
        elif level == 'WARN':
            message = random.choice([
                'High memory usage detected',
                'Slow database queries',
                'Connection pool at 80% capacity'
            ])
        else:
            message = f'Request processed successfully for user {fake.random_int(1000, 9999)}'
            
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'system': 'java_app',
            'level': level,
            'message': message,
            'trace_id': fake.uuid4(),
            'user_id': fake.random_int(1000, 9999),
            'response_time': random.expovariate(1/150)
        }
        
        if self.verbose and level in ['ERROR', 'WARN']:
            print(f"JAVA: {level} - {message}")
            
        return log_entry
    
    def _generate_k8s_logs(self):
        level = random.choices(['INFO', 'WARN', 'ERROR'], weights=[0.75, 0.15, 0.1])[0]
        message = ''
        
        if level == 'ERROR':
            message = random.choice([
                'Pod crashed due to OOM',
                'Image pull failed',
                'Node not ready'
            ])
        elif level == 'WARN':
            message = random.choice([
                'High memory pressure on node',
                'CPU throttling detected',
                'Network latency increased'
            ])
        else:
            message = 'Pod started successfully'
            
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'system': 'kubernetes',
            'level': level,
            'message': message,
            'pod_name': f'pod-{fake.random_int(1, 100)}',
            'namespace': 'production'
        }
        
        if self.verbose and level in ['ERROR', 'WARN']:
            print(f"K8S: {level} - {message}")
            
        return log_entry
    
    def _generate_cobol_logs(self):
        level = random.choices(['INFO', 'WARN', 'ERROR'], weights=[0.8, 0.15, 0.05])[0]
        message = ''
        
        if level == 'ERROR':
            message = random.choice([
                'ABEND: U4038 System error',
                'Dataset mount failed',
                'CICS transaction abended'
            ])
            self.metrics['cobol_error_count'].inc()
        elif level == 'WARN':
            message = random.choice([
                'Batch job running longer than expected',
                'Storage allocation nearing limit',
                'Tape drive response slow'
            ])
        else:
            message = 'Batch job completed successfully'
            self.metrics['cobol_jobs_processed'].inc()
            
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'system': 'cobol_mainframe',
            'level': level,
            'message': message,
            'job_id': f'JOB{fake.random_int(10000, 99999)}',
            'region': 'MAINFRAME_PROD'
        }
        
        if self.verbose and level in ['ERROR', 'WARN']:
            print(f"COBOL: {level} - {message}")
            
        return log_entry
    
    def generate_metrics(self):
        # Update system metrics with occasional anomalies
        anomaly_detected = False
        
        if random.random() < 0.1:  # 10% chance of anomaly
            cpu = random.uniform(80, 99)
            memory = random.uniform(85, 99)
            self.metrics['java_app_cpu_usage'].set(cpu)
            self.metrics['java_app_memory_usage'].set(memory)
            if self.verbose:
                print(f"JAVA METRIC ANOMALY: CPU={cpu:.1f}%, Memory={memory:.1f}%")
            anomaly_detected = True
        else:
            self.metrics['java_app_cpu_usage'].set(random.uniform(20, 75))
            self.metrics['java_app_memory_usage'].set(random.uniform(30, 70))
            
        if random.random() < 0.08:  # 8% chance of anomaly
            cpu = random.uniform(5, 20)
            memory = random.uniform(10, 30)
            pods = random.randint(1, 3)
            self.metrics['k8s_cpu_usage'].set(cpu)
            self.metrics['k8s_memory_usage'].set(memory)
            self.metrics['k8s_pod_count'].set(pods)
            if self.verbose:
                print(f"K8S METRIC ANOMALY: CPU={cpu:.1f}%, Memory={memory:.1f}%, Pods={pods}")
            anomaly_detected = True
        else:
            self.metrics['k8s_cpu_usage'].set(random.uniform(15, 65))
            self.metrics['k8s_memory_usage'].set(random.uniform(25, 70))
            self.metrics['k8s_pod_count'].set(random.randint(5, 20))
            
        if random.random() < 0.12:  # 12% chance of anomaly
            cpu = random.uniform(80, 95)
            self.metrics['cobol_cpu_usage'].set(cpu)
            if self.verbose:
                print(f"COBOL METRIC ANOMALY: CPU={cpu:.1f}%")
            anomaly_detected = True
        else:
            self.metrics['cobol_cpu_usage'].set(random.uniform(10, 60))
            
        return anomaly_detected
    
    def collect_data(self):
        """Collect logs from all systems"""
        logs = []
        logs.append(self._generate_java_logs())
        logs.append(self._generate_k8s_logs())
        logs.append(self._generate_cobol_logs())
        
        # Update metrics and check for anomalies
        metric_anomaly = self.generate_metrics()
        
        return logs, metric_anomaly
    
    def run(self):
        start_http_server(8000)
        print("Starting enhanced synthetic data generation...")
        
        while True:
            logs, metric_anomaly = self.collect_data()
            for log in logs:
                print(json.dumps(log))
            time.sleep(2)

if __name__ == "__main__":
    generator = SyntheticSystemLogs()
    generator.run()