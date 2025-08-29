# main.py - Enhanced with proper Prometheus metrics & consistent RCA printing
import time
import json
import pandas as pd
import numpy as np
import random
import re
import requests
import logging
import threading
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
import torch
from forecasting.transformer_predictor import AdvancedPredictiveAnalyzer
from rca.rag_llm_analyzer import RAGLLMAnalyzer
# Add to imports in main.py
from slo_manager.slo_definitions import SLO_REGISTRY, SystemType
from prometheus_client import Gauge

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Prometheus client metrics
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Gauge, Counter

ANOMALIES_TOTAL = Counter('anomalies_total', 'Total number of anomalies detected')
FEEDBACK_QUEUE_SIZE = Gauge('feedback_queue_size', 'Current size of feedback queue')
MONITORING_ACTIVE = Gauge('monitoring_active', 'Whether monitoring is active (1) or not (0)')
SYSTEMS_MONITORED = Gauge('systems_monitored', 'Number of systems being monitored')
FEATURE_HISTORY_SIZE = Gauge('feature_history_size', 'Size of feature history')
KNOWLEDGE_BASE_ENTRIES = Gauge('knowledge_base_entries', 'Number of knowledge base entries')
SLO_COMPLIANCE = Gauge('slo_compliance', 'SLO compliance status', ['system', 'slo_name'])
SLO_ERROR_BUDGET = Gauge('slo_error_budget', 'SLO error budget remaining', ['system', 'slo_name'])


class SyntheticSystemLogs:
    def __init__(self, verbose=True):
        self.verbose = verbose

    def _generate_java_logs(self):
        level = random.choices(['INFO', 'WARN', 'ERROR'], weights=[0.7, 0.2, 0.1])[0]
        if level == 'ERROR':
            message = random.choice([
                'Database connection timeout',
                'OutOfMemoryError: Java heap space',
                'NullPointerException in service layer',
                'StackOverflowError in recursive function',
                'ClassNotFoundException for service implementation'
            ])
        elif level == 'WARN':
            message = random.choice([
                'High memory usage detected',
                'Slow database queries',
                'Connection pool at 80% capacity',
                'GC overhead limit approaching',
                'Thread pool exhaustion warning'
            ])
        else:
            message = f'Request processed successfully for user {random.randint(1000, 9999)}'
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'system': 'java_app',
            'level': level,
            'message': message,
            'response_time': random.expovariate(1/150)
        }

    def _generate_k8s_logs(self):
        level = random.choices(['INFO', 'WARN', 'ERROR'], weights=[0.75, 0.15, 0.1])[0]
        if level == 'ERROR':
            message = random.choice([
                'Pod crashed due to OOM',
                'Image pull failed',
                'Node not ready',
                'Persistent volume claim failed',
                'Network policy conflict'
            ])
        elif level == 'WARN':
            message = random.choice([
                'High memory pressure on node',
                'CPU throttling detected',
                'Network latency increased',
                'Pod scheduled on over-utilized node',
                'Volume storage approaching limit'
            ])
        else:
            message = 'Pod started successfully'
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'system': 'kubernetes',
            'level': level,
            'message': message
        }

    def _generate_cobol_logs(self):
        level = random.choices(['INFO', 'WARN', 'ERROR'], weights=[0.8, 0.15, 0.05])[0]
        if level == 'ERROR':
            message = random.choice([
                'ABEND: U4038 System error',
                'Dataset mount failed',
                'CICS transaction abended',
                'VSAM file access violation',
                'DB2 connection timeout'
            ])
        elif level == 'WARN':
            message = random.choice([
                'Batch job running longer than expected',
                'Storage allocation nearing limit',
                'Tape drive response slow',
                'Database tablespace approaching maximum',
                'Transaction volume exceeding threshold'
            ])
        else:
            message = 'Batch job completed successfully'
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'system': 'cobol_mainframe',
            'level': level,
            'message': message
        }

    def collect_data(self):
        logs = [
            self._generate_java_logs(),
            self._generate_k8s_logs(),
            self._generate_cobol_logs(),
        ]
        # 20% chance of a general metric anomaly
        metric_anomaly = random.random() < 0.2
        return logs, metric_anomaly


class AIObservabilityPlatform:
    def __init__(self):
        self.synthetic_generator = SyntheticSystemLogs(verbose=True)
        self.systems = ['java_app', 'kubernetes', 'cobol_mainframe']
        self.anomaly_history = []
        self.feedback_queue = []
        self.is_monitoring = False
        self.monitoring_thread = None
        self.feature_history = []

        # Advanced components
        self.predictive_analyzer = AdvancedPredictiveAnalyzer()
        self.rag_analyzer = RAGLLMAnalyzer(ollama_host="http://ollama:11434")

        # Start monitoring automatically
        self.start_monitoring()

    def start_monitoring(self):
        if self.is_monitoring:
            logger.info("Monitoring is already running")
            return

        def run_monitoring():
            self.is_monitoring = True
            MONITORING_ACTIVE.set(1)
            try:
                self.run_continuous_monitoring()
            except Exception as e:
                logger.error(f"Monitoring failed: {e}")
            finally:
                self.is_monitoring = False
                MONITORING_ACTIVE.set(0)

        self.monitoring_thread = threading.Thread(target=run_monitoring, daemon=True)
        self.monitoring_thread.start()
        logger.info("Monitoring started successfully")

    def collect_and_process_data(self):
        logger.info("Collecting and processing data...")
        logs, metric_anomaly = self.synthetic_generator.collect_data()
        df = pd.DataFrame(logs)
        df = self._enhance_features(df)
        self._update_feature_history(df)
        return df, metric_anomaly

    def _enhance_features(self, df):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_error'] = (df['level'] == 'ERROR').astype(int)
        df['is_warning'] = (df['level'] == 'WARN').astype(int)

        # Synthetic metrics
        for system in self.systems:
            mask = df['system'] == system
            if mask.any():
                df.loc[mask, 'cpu_usage'] = np.random.uniform(20, 80, mask.sum())
                df.loc[mask, 'memory_usage'] = np.random.uniform(30, 85, mask.sum())
                df.loc[mask, 'response_time'] = np.random.exponential(150, mask.sum())
                df.loc[mask, 'network_latency'] = np.random.uniform(5, 50, mask.sum())
        return df

    def _update_feature_history(self, df):
        current_features = {'timestamp': datetime.now(), 'systems': {}}
        for system in self.systems:
            sd = df[df['system'] == system]
            current_features['systems'][system] = {
                'error_count': int((sd['level'] == 'ERROR').sum()),
                'warning_count': int((sd['level'] == 'WARN').sum()),
                'cpu_usage': float(sd['cpu_usage'].mean() if 'cpu_usage' in sd else 0),
                'memory_usage': float(sd['memory_usage'].mean() if 'memory_usage' in sd else 0),
                'response_time': float(sd['response_time'].mean() if 'response_time' in sd else 0),
                'network_latency': float(sd['network_latency'].mean() if 'network_latency' in sd else 0),
                'total_logs': int(len(sd)),
            }
        self.feature_history.append(current_features)
        if len(self.feature_history) > 48:
            self.feature_history = self.feature_history[-48:]
        FEATURE_HISTORY_SIZE.set(len(self.feature_history))

    def run_predictive_analysis(self):
        logger.info("Running advanced predictive analysis...")
        if not self.feature_history:
            return {}, {}
        risk_predictions = {}
        failure_predictions = {}
        for system in self.systems:
            recent = [f['systems'].get(system, {}) for f in self.feature_history[-24:]]
            pred = self.predictive_analyzer.predict_risk_and_failure(recent, system)
            risk_predictions[system] = pred['risk_score']
            failure_predictions[system] = {
                'probability': pred['failure_probability'],
                'timeframe': pred['timeframe'],
                'confidence': pred['confidence'],
                'critical_metrics': pred['critical_metrics'],
            }
            if len(self.feature_history) % 24 == 0:
                historical = self._prepare_training_data()
                self.predictive_analyzer.train_model(system)
        return risk_predictions, failure_predictions

    def _prepare_training_data(self):
        rows = []
        for feat in self.feature_history:
            for system, metrics in feat['systems'].items():
                rows.append({'timestamp': feat['timestamp'], 'system': system, **metrics})
        return pd.DataFrame(rows)

    def detect_anomalies(self, features, metric_anomaly):
        logger.info("Detecting anomalies...")
        anomalies: Dict[str, Dict[str, Any]] = {}

        # General metric anomaly
        if metric_anomaly:
            anomalies['metrics'] = {
                'type': 'Metric_Anomaly_General',
                'message': 'Abnormal metric values detected across systems',
                'timestamp': datetime.now().isoformat(),
                'score': 7.0,
                'system': 'general'
            }

        # Log-driven anomalies
        for system in self.systems:
            sd = features[features['system'] == system]
            if sd.empty:
                continue
            for _, log in sd.iterrows():
                if log['level'] in ['ERROR', 'WARN']:
                    detected = self._classify_issue(str(log['message']), system)
                    if detected and system not in anomalies:
                        anomalies[system] = {
                            'type': detected,
                            'message': str(log['message']),
                            'timestamp': log['timestamp'].isoformat() if hasattr(log['timestamp'], 'isoformat') else str(log['timestamp']),
                            'score': self._calculate_severity_score(detected, log['level']),
                            'system': system
                        }

        if anomalies:
            ANOMALIES_TOTAL.inc(len(anomalies))
        return anomalies

    def _classify_issue(self, message, system):
        patterns = {
            'java_app': {
                'NullPointerException': r'NullPointerException',
                'OutOfMemoryError': r'OutOfMemoryError',
                'DatabaseConnectionTimeout': r'Database connection timeout',
                'StackOverflowError': r'StackOverflowError',
                'ClassNotFoundException': r'ClassNotFoundException',
                'ConnectionPoolExhaustion': r'Connection pool.*exhaust(ed|ion)'
            },
            'kubernetes': {
                'PodCrashOOM': r'Pod crashed due to OOM',
                'CPUThrottling': r'CPU throttling detected',
                'ImagePullFailed': r'Image pull failed',
                'NodeNotReady': r'Node not ready',
                'NetworkPolicyConflict': r'Network policy conflict',
                'VolumeClaimFailed': r'Persistent volume claim failed'
            },
            'cobol_mainframe': {
                'JobAbend': r'ABEND:',
                'StorageWarning': r'Storage allocation nearing limit',
                'TapeDriveSlow': r'Tape drive response slow',
                'DatasetMountFailed': r'Dataset mount failed',
                'CICSAbend': r'CICS transaction abended',
                'DB2Timeout': r'DB2 connection timeout'
            }
        }
        if system not in patterns:
            return None
        for issue_type, regex in patterns[system].items():
            if re.search(regex, message, re.IGNORECASE):
                return issue_type
        return None

    def _calculate_severity_score(self, issue_type, log_level):
        base = {
            'NullPointerException': 8.0,
            'OutOfMemoryError': 9.0,
            'DatabaseConnectionTimeout': 7.5,
            'StackOverflowError': 8.5,
            'ClassNotFoundException': 7.0,
            'ConnectionPoolExhaustion': 8.0,
            'PodCrashOOM': 9.0,
            'CPUThrottling': 7.0,
            'ImagePullFailed': 7.5,
            'NodeNotReady': 8.5,
            'NetworkPolicyConflict': 6.5,
            'VolumeClaimFailed': 7.0,
            'JobAbend': 8.5,
            'StorageWarning': 6.5,
            'TapeDriveSlow': 5.5,
            'DatasetMountFailed': 7.0,
            'CICSAbend': 8.0,
            'DB2Timeout': 7.5
        }
        score = base.get(issue_type, 7.0)
        if log_level == 'WARN':
            score *= 0.7
        return min(score, 10.0)

    def _get_current_system_state(self, system):
        if not self.feature_history:
            return {}
        s = self.feature_history[-1]['systems'].get(system, {})
        return {
            'high_cpu_usage': s.get('cpu_usage', 0) > 80,
            'high_memory_usage': s.get('memory_usage', 0) > 85,
            'high_error_rate': s.get('error_count', 0) > 3,
            'high_latency': s.get('response_time', 0) > 200,
            'current_cpu': s.get('cpu_usage', 0),
            'current_memory': s.get('memory_usage', 0),
            'current_errors': s.get('error_count', 0)
        }

    def run_continuous_monitoring(self):
        print("Starting AI Observability Platform...")
        print("Systems monitored:", self.systems)
        print("Press Ctrl+C to stop\n")

        iteration = 0
        self.is_monitoring = True
        MONITORING_ACTIVE.set(1)
        SYSTEMS_MONITORED.set(len(self.systems))

        while self.is_monitoring:
            iteration += 1
            print(f"\n=== Iteration {iteration} - {datetime.now()} ===")
            try:
                # 1) Collect
                features, metric_anomaly = self.collect_and_process_data()
                if features.empty:
                    print("No features collected, skipping iteration")
                    time.sleep(5)
                    continue

                # 2) Predict
                risk_predictions, failure_predictions = self.run_predictive_analysis()
                self.monitor_slo_compliance()

                # 3) Detect anomalies
                anomalies = self.detect_anomalies(features, metric_anomaly)

                # 4) RCA + output
                for system_key, anomaly_info in anomalies.items():
                    anomaly_info['system'] = system_key  # ensure consistency
                    self._handle_anomaly(system_key, anomaly_info, risk_predictions, failure_predictions)

                # 5) Summary
                print("\n📊 CURRENT RISK PREDICTIONS:")
                for system, score in risk_predictions.items():
                    print(f"   {system}: {score:.1f}/10")

                print("\n🔮 FAILURE PREDICTIONS:")
                for system, prediction in failure_predictions.items():
                    if prediction['probability'] > 0.3:
                        print(f"   {system}: {prediction['probability']*100:.1f}% within {prediction['timeframe']}")

                print(f"\n{'🚨 Anomalies detected: '+str(len(anomalies)) if anomalies else '✅ No anomalies detected'}")

                time.sleep(10)

            except KeyboardInterrupt:
                print("\nStopping AI Observability Platform...")
                self.is_monitoring = False
                MONITORING_ACTIVE.set(0)
                break
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(10)
            

    def _handle_anomaly(self, system_key, anomaly_info, risk_predictions, failure_predictions):
        logger.info(f"🔍 Handling anomaly for {system_key}: {anomaly_info['type']}")

        # Enhanced RCA with RAG
        rca_result = self.rag_analyzer.perform_enhanced_rca(anomaly_info)

        # Intelligent recommendations
        current_state = self._get_current_system_state(system_key if system_key in self.systems else self.systems[0])
        recommendations = self.predictive_analyzer.generate_intelligent_recommendations(
            system_key,
            {
                'risk_score': risk_predictions.get(system_key, 0),
                'failure_probability': failure_predictions.get(system_key, {}).get('probability', 0)
            },
            current_state
        )

        # Record
        anomaly_record = {
            'info': anomaly_info,
            'rca_result': rca_result,
            'recommendations': recommendations,
            'timestamp': datetime.now(),
            'resolved': False,
            'feedback_provided': False
        }
        self.anomaly_history.append(anomaly_record)

        # Pretty print
        display_system = "general" if system_key == "metrics" else system_key
        print(f"\n🚨 {anomaly_info['type'].replace('_', ' ').upper()} DETECTED in {display_system.upper()}")
        print(f"   Message: {anomaly_info['message']}")
        print(f"   Time: {anomaly_info['timestamp']}")
        print(f"   Severity Score: {anomaly_info['score']:.2f}/10")

        print("\n🔍 ENHANCED RCA WITH RAG:")
        analysis_preview = rca_result['analysis'] or "(no analysis)"
        print(analysis_preview[:500] + ("..." if len(analysis_preview) > 500 else ""))

        print("\n🤖 INTELLIGENT RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations[:3]):
            print(f"   {i+1}. {rec}")

        print("\n📊 PREDICTIVE INSIGHTS:")
        if system_key in risk_predictions:
            print(f"   Predicted Risk: {risk_predictions[system_key]:.1f}/10")
        if system_key in failure_predictions:
            fail_pred = failure_predictions[system_key]
            print(f"   Failure Probability: {fail_pred['probability']*100:.1f}% within {fail_pred['timeframe']}")
            if fail_pred['critical_metrics']:
                print(f"   Critical Metrics: {[m['metric'] for m in fail_pred['critical_metrics']]}")
        print("-" * 60)

        # Queue for feedback
        self.feedback_queue.append(anomaly_record)
        FEEDBACK_QUEUE_SIZE.set(len(self.feedback_queue))
    def monitor_slo_compliance(self):
        """Monitor SLO compliance based on current system state"""
        if not self.feature_history:
            return
            
        current_state = self.feature_history[-1]['systems']
        slo_status = {}
        
        for slo_id, slo in SLO_REGISTRY.items():
            system = slo.system
            if system not in current_state:
                continue
                
            # Simplified compliance checking - in a real implementation,
            # you would execute the Prometheus queries
            system_state = current_state[system]
            
            # Example: Check if error count is within SLO limits
            if "error" in slo_id.lower() and "error_count" in system_state:
                error_count = system_state['error_count']
                # This is a simplified check - real implementation would use the actual SLO query
                is_compliant = error_count < 5  # Example threshold
                
                # Set Prometheus metrics
                SLO_COMPLIANCE.labels(system=system, slo_name=slo.name).set(1 if is_compliant else 0)
                SLO_ERROR_BUDGET.labels(system=system, slo_name=slo.name).set(
                    max(0, 100 - (error_count * 20))  # Simplified error budget calculation
                )
        if slo_status:
            print("\n📈 SLO COMPLIANCE STATUS:")
            for slo_key, status in slo_status.items():
                compliance_icon = "✅" if status["compliant"] else "❌"
                print(f"   {compliance_icon} {slo_key}: Compliant={status['compliant']}, Error Budget={status['error_budget']}%, Errors={status['error_count']}")


# Initialize the platform
platform = AIObservabilityPlatform()

@app.get("/health")
async def health_check():
    try:
        try:
            ollama_response = requests.get("http://ollama:11434/api/tags", timeout=5)
            ollama_healthy = ollama_response.status_code == 200
        except Exception:
            ollama_healthy = False

        torch_healthy = torch.cuda.is_available() if hasattr(torch, 'cuda') else False
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "services": {
                    "ollama": ollama_healthy,
                    "pytorch_gpu": torch_healthy,
                    "platform": True,
                    "monitoring_active": platform.is_monitoring,
                    "transformer_models_loaded": len(getattr(platform.predictive_analyzer, "models", {})) > 0
                }
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

@app.get("/metrics")
async def metrics():
    FEEDBACK_QUEUE_SIZE.set(len(platform.feedback_queue))
    MONITORING_ACTIVE.set(1 if platform.is_monitoring else 0)
    SYSTEMS_MONITORED.set(len(platform.systems))
    FEATURE_HISTORY_SIZE.set(len(platform.feature_history))

    try:
        kb_count = len(platform.rag_analyzer.knowledge_base)
        KNOWLEDGE_BASE_ENTRIES.set(kb_count)
    except Exception:
        KNOWLEDGE_BASE_ENTRIES.set(0)

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/status")
async def platform_status():
    risk_predictions, failure_predictions = platform.run_predictive_analysis()
    return JSONResponse(
        status_code=200,
        content={
            "systems": platform.systems,
            "anomaly_count": len(platform.anomaly_history),
            "feedback_queue_size": len(platform.feedback_queue),
            "monitoring_active": platform.is_monitoring,
            "risk_predictions": risk_predictions,
            "failure_predictions": failure_predictions,
            "feature_history_size": len(platform.feature_history)
        }
    )

@app.get("/anomalies")
async def get_anomalies(limit: int = 10):
    recent = []
    for anomaly in platform.anomaly_history[-limit:]:
        recent.append({
            "system": anomaly['info'].get('system', 'unknown'),
            "type": anomaly['info'].get('type', 'unknown'),
            "message": anomaly['info'].get('message', ''),
            "timestamp": anomaly['info'].get('timestamp', ''),
            "score": anomaly['info'].get('score', 0),
            "resolved": anomaly.get('resolved', False),
            "has_recommendations": len(anomaly.get('recommendations', [])) > 0
        })
    return JSONResponse(status_code=200, content={"anomalies": recent})

@app.get("/predictions")
async def get_predictions():
    risk_predictions, failure_predictions = platform.run_predictive_analysis()
    return JSONResponse(
        status_code=200,
        content={
            "risk_predictions": risk_predictions,
            "failure_predictions": failure_predictions
        }
    )

@app.post("/monitoring/start")
async def start_monitoring():
    if platform.is_monitoring:
        return JSONResponse(status_code=400, content={"message": "Monitoring is already running"})
    platform.start_monitoring()
    return JSONResponse(status_code=200, content={"message": "Monitoring started successfully"})

@app.post("/monitoring/stop")
async def stop_monitoring():
    platform.is_monitoring = False
    MONITORING_ACTIVE.set(0)
    return JSONResponse(status_code=200, content={"message": "Monitoring stop signal sent"})

@app.post("/feedback/{anomaly_index}")
async def provide_feedback(anomaly_index: int, effective: bool, notes: str = ""):
    try:
        if anomaly_index < 0 or anomaly_index >= len(platform.anomaly_history):
            raise HTTPException(status_code=404, detail="Anomaly not found")

        anomaly = platform.anomaly_history[anomaly_index]
        anomaly['feedback_provided'] = True
        anomaly['resolution_effective'] = effective
        anomaly['feedback_notes'] = notes

        platform.rag_analyzer.add_to_knowledge_base(
            anomaly['info'],
            anomaly['rca_result']['analysis'],
            effectiveness=1.0 if effective else 0.0
        )

        return JSONResponse(status_code=200, content={"message": "Feedback recorded successfully"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "AI Observability Platform API",
        "version": "2.0 - Enhanced with Transformer & RAG",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "status": "/status",
            "anomalies": "/anomalies?limit=10",
            "predictions": "/predictions",
            "start_monitoring": "/monitoring/start (POST)",
            "stop_monitoring": "/monitoring/stop (POST)",
            "provide_feedback": "/feedback/{index} (POST)"
        },
        "features": [
            "Transformer-based risk prediction",
            "LLM with RAG for root cause analysis",
            "Intelligent recommendations",
            "Continuous learning feedback loop",
            "Multi-system monitoring"
        ]
    }

if __name__ == "__main__":
    import subprocess
    import sys
    try:
        # Try to generate documentation, but don't fail if it doesn't work
        result = subprocess.run([sys.executable, "generate_docs.py"], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ SLO documentation generated successfully")
        else:
            print(f"⚠️ Documentation generation had issues: {result.stderr}")
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"⚠️ Could not generate documentation: {e}")
    import uvicorn
    # Run unbuffered when using docker: set PYTHONUNBUFFERED=1 or run `python -u`
    uvicorn.run(app, host="0.0.0.0", port=8000)
