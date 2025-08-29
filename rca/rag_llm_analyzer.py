# rca/rag_llm_analyzer.py
import os
import time
import json
import faiss
import pickle
import logging
import requests
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
from sentence_transformers import SentenceTransformer
import ollama

# -------------------------
# Config (env-overridable)
# -------------------------
DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")            # aligns with compose/init
LLM_READ_TIMEOUT = int(os.getenv("LLM_READ_TIMEOUT", "300"))  # seconds
LLM_CONNECT_TIMEOUT = int(os.getenv("LLM_CONNECT_TIMEOUT", "10"))
LLM_NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT", "160"))    # smaller = faster on CPU
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGLLMAnalyzer:
    def __init__(self, ollama_host: str = DEFAULT_OLLAMA_HOST):
        self.ollama_host = ollama_host
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_db = None
        self.knowledge_base: List[Dict[str, Any]] = []

        # Ollama client & model
        self.available_model = LLM_MODEL
        self.ollama_available = False

        # The `ollama` python client uses httpx internally
        # Some versions accept `timeout` in Client; if not, per-call timeouts still apply
        try:
            self.ollama_client = ollama.Client(host=self.ollama_host, timeout=LLM_READ_TIMEOUT)
        except TypeError:
            # Fallback if this ollama lib version doesn't accept timeout in ctor
            self.ollama_client = ollama.Client(host=self.ollama_host)

        self._initialize_ollama()
        self._initialize_vector_db()

    # -------------------------
    # Ollama bootstrap
    # -------------------------
    def _initialize_ollama(self, max_retries: int = 10, retry_delay: int = 5):
        """Initialize Ollama connection, ensure model exists, and prewarm."""
        for attempt in range(1, max_retries + 1):
            try:
                r = requests.get(
                    f"{self.ollama_host}/api/version",
                    timeout=(LLM_CONNECT_TIMEOUT, 5),
                )
                if r.status_code == 200:
                    self.ollama_available = True
                    logger.info("✅ Ollama connected successfully")
                    break
            except Exception as e:
                logger.warning(f"Ollama not available (attempt {attempt}/{max_retries}): {e}")
            time.sleep(retry_delay)

        if not self.ollama_available:
            logger.warning("⚠️ Ollama not available after retries. Using fallback analysis only.")
            return

        # Ensure required model exists
        try:
            tags = requests.get(
                f"{self.ollama_host}/api/tags",
                timeout=(LLM_CONNECT_TIMEOUT, 15),
            )
            tags.raise_for_status()
            models = [m.get("name") for m in tags.json().get("models", [])]
            if self.available_model not in models:
                logger.warning(f"Llama model '{self.available_model}' not found, pulling...")
                self._pull_model(self.available_model)
                logger.info("✅ Llama model pulled successfully")
        except Exception as e:
            logger.error(f"Could not verify/pull model '{self.available_model}': {e}")

        # Prewarm so first real call is snappy
        self._prewarm()

    def _pull_model(self, name: str):
        """Pull a model by name using the HTTP API (longer timeout for downloads)."""
        try:
            resp = requests.post(
                f"{self.ollama_host}/api/pull",
                json={"name": name},
                timeout=(LLM_CONNECT_TIMEOUT, max(LLM_READ_TIMEOUT, 600)),
            )
            if resp.status_code != 200:
                logger.warning(f"Failed to pull model '{name}': {resp.text}")
        except Exception as e:
            logger.error(f"Error pulling model '{name}': {e}")

    def _prewarm(self):
        """Generate 1 token to load the model into memory; ignore errors."""
        try:
            _ = self.ollama_client.generate(
                model=self.available_model,
                prompt="ok",
                options={"num_predict": 1, "temperature": 0.0},
                stream=False,
            )
        except Exception:
            pass  # best-effort

    # -------------------------
    # Vector DB (FAISS)
    # -------------------------
    def _initialize_vector_db(self):
        """Initialize FAISS vector DB and attempt to load existing KB."""
        self.vector_db = faiss.IndexFlatL2(384)  # all-MiniLM-L6-v2 dimension
        self.knowledge_base = []

        os.makedirs("knowledge", exist_ok=True)
        try:
            self.load_knowledge_base("knowledge/rag_knowledge.pkl")
        except FileNotFoundError:
            logger.info("No existing knowledge base found, starting fresh")

    def add_to_knowledge_base(self, incident_data: Dict, resolution: str, effectiveness: float):
        """Add incident + resolution, embed, index, and persist."""
        entry = {
            'incident': incident_data,
            'resolution': resolution,
            'effectiveness': effectiveness,
            'timestamp': datetime.now().isoformat(),
            'embedding': None,
        }
        text = self._create_incident_text(incident_data)
        emb = self.embedding_model.encode([text])[0]
        entry['embedding'] = emb

        self.vector_db.add(np.array([emb]).astype('float32'))
        self.knowledge_base.append(entry)
        logger.info(f"Added incident to knowledge base: {incident_data.get('type', 'unknown')}")
        self.save_knowledge_base("knowledge/rag_knowledge.pkl")

    def search_similar_incidents(self, query_incident: Dict, k: int = 5) -> List[Dict]:
        """Return k most similar incidents by vector distance."""
        if not self.knowledge_base:
            return []
        text = self._create_incident_text(query_incident)
        q = self.embedding_model.encode([text])[0]
        distances, indices = self.vector_db.search(np.array([q]).astype('float32'), k)

        out: List[Dict[str, Any]] = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.knowledge_base):
                inc = self.knowledge_base[idx]
                out.append({
                    'incident': inc['incident'],
                    'resolution': inc['resolution'],
                    'effectiveness': inc['effectiveness'],
                    'similarity_score': float(1 / (1 + distances[0][i])),
                    'timestamp': inc['timestamp'],
                })
        return out

    # -------------------------
    # LLM
    # -------------------------
    def _create_incident_text(self, incident_data: Dict) -> str:
        return (
            f"System: {incident_data.get('system', 'unknown')}\n"
            f"Type: {incident_data.get('type', 'unknown')}\n"
            f"Message: {incident_data.get('message', 'unknown')}\n"
            f"Severity: {incident_data.get('score', 0)}\n"
            f"Timestamp: {incident_data.get('timestamp', 'unknown')}\n"
        )

    def _create_rag_prompt(self, incident_data: Dict, similar_incidents: List[Dict]) -> str:
        incident_context = (
            "CURRENT INCIDENT:\n"
            f"- System: {incident_data.get('system', 'unknown')}\n"
            f"- Type: {incident_data.get('type', 'unknown')}\n"
            f"- Message: {incident_data.get('message', 'unknown')}\n"
            f"- Severity: {incident_data.get('score', 0)}/10\n"
            f"- Timestamp: {incident_data.get('timestamp', 'unknown')}\n"
        )

        if similar_incidents:
            rag_context = "SIMILAR HISTORICAL INCIDENTS AND RESOLUTIONS:\n"
            for i, inc in enumerate(similar_incidents[:3]):
                rag_context += (
                    f"Incident {i+1} (Similarity: {inc['similarity_score']:.2f}, "
                    f"Effectiveness: {inc['effectiveness']:.2f}):\n"
                    f"- System: {inc['incident'].get('system', 'unknown')}\n"
                    f"- Type: {inc['incident'].get('type', 'unknown')}\n"
                    f"- Resolution: {inc['resolution'][:300]}...\n"
                )
        else:
            rag_context = "No similar historical incidents found."

        prompt = f"""You are an AI observability expert. Analyze the current incident and provide:

1. ROOT CAUSE ANALYSIS: Identify the most likely root cause based on the incident details and similar historical incidents
2. IMPACT ASSESSMENT: Evaluate the potential impact on system performance and user experience
3. RESOLUTION RECOMMENDATIONS: Provide specific, actionable steps to resolve the issue
4. PREVENTIVE MEASURES: Suggest how to prevent similar incidents in the future

{incident_context}

{rag_context}

Please provide a comprehensive analysis in clear, technical language:"""
        return prompt

    def generate_llm_analysis_with_rag(self, incident_data: Dict, similar_incidents: List[Dict]) -> str:
        """Use Ollama with retries; fall back to heuristic analysis on failure."""
        if not self.ollama_available:
            return self._generate_fallback_analysis(incident_data, similar_incidents)

        prompt = self._create_rag_prompt(incident_data, similar_incidents)

        backoffs = [0, 5, 10]  # seconds
        for attempt, delay in enumerate(backoffs, start=1):
            if delay:
                time.sleep(delay)
            try:
                resp = self.ollama_client.generate(
                    model=self.available_model,
                    prompt=prompt,
                    options={
                        "temperature": LLM_TEMPERATURE,
                        "top_p": LLM_TOP_P,
                        "num_predict": LLM_NUM_PREDICT,
                    },
                    stream=False,
                )
                return resp.get("response", "").strip()
            except Exception as e:
                logger.error(f"LLM analysis failed (attempt {attempt}/{len(backoffs)}): {e}")

        return self._generate_fallback_analysis(incident_data, similar_incidents)

    # -------------------------
    # Fallback (heuristics)
    # -------------------------
    def _generate_fallback_analysis(self, incident_data: Dict, similar_incidents: List[Dict]) -> str:
        system = incident_data.get('system', 'unknown')
        incident_type = incident_data.get('type', 'unknown')

        analysis_templates = {
            'metrics': {
                'Metric_Anomaly_General': (
                    "Broad metric deviations across systems. Investigate spikes/drops in CPU, memory, "
                    "latency, and error rates. Correlate with recent deployments or infrastructure events."
                ),
            },
            'java_app': {
                'NullPointerException': "Root cause likely uninitialized object reference. Check recent code changes and null checks.",
                'OutOfMemoryError': "Memory leak or insufficient heap allocation. Analyze heap dump and monitor garbage collection.",
                'DatabaseConnectionTimeout': "Network latency or database overload. Check connection pool settings and database performance.",
                'StackOverflowError': "Infinite recursion or deep call stack. Review recursive algorithms and stack size configuration.",
                'ClassNotFoundException': "Classpath issue or missing dependency. Verify deployment artifacts and dependency management.",
                'ConnectionPoolExhaustion': "Connection leaks or high demand. Monitor connection usage and implement proper connection closing."
            },
            'kubernetes': {
                'PodCrashOOM': "Container memory limits too low or memory leak. Adjust resource limits and monitor memory usage patterns.",
                'CPUThrottling': "CPU requests/limits misconfigured. Optimize resource allocation and implement horizontal pod autoscaling.",
                'ImagePullFailed': "Registry authentication or network issues. Check image pull secrets and network connectivity.",
                'NodeNotReady': "Node resource exhaustion or kubelet issues. Check node status and resource utilization.",
                'NetworkPolicyConflict': "Network policy misconfiguration. Review network policies and namespace labels.",
                'VolumeClaimFailed': "Storage class issues or quota limits. Verify storage class configuration and resource quotas."
            },
            'cobol_mainframe': {
                'JobAbend': "Program logic error or resource contention. Check job control language and system resources.",
                'StorageWarning': "Dataset space allocation nearing limits. Monitor storage usage and consider archiving old data.",
                'TapeDriveSlow': "Hardware performance degradation or contention. Check tape drive status and scheduling.",
                'DatasetMountFailed': "Dataset not available or permission issues. Verify dataset availability and security settings.",
                'CICSAbend': "Transaction processing error. Review CICS region configuration and transaction programs.",
                'DB2Timeout': "Database contention or SQL performance issues. Analyze SQL queries and database locks."
            }
        }

        specific = analysis_templates.get(system, {}).get(incident_type, "")
        if specific:
            analysis = f"ROOT CAUSE ANALYSIS: {specific}"
        else:
            analysis = f"ROOT CAUSE ANALYSIS: {incident_type} detected in {system}. Requires investigation of system logs and metrics."

        recommendations = [
            "Check recent system changes and deployments",
            "Monitor system metrics for abnormal patterns",
            "Review application and system logs for additional context",
            "Verify resource allocation and configuration settings",
        ]
        if similar_incidents:
            best = max(similar_incidents, key=lambda x: x['similarity_score'])
            recommendations.insert(0, f"Consider solution from similar incident: {best['resolution'][:100]}...")

        analysis += "\n\nRECOMMENDATIONS:\n" + "\n".join(f"- {r}" for r in recommendations)
        analysis += "\n\nIMPACT: Service degradation likely. User experience may be affected until resolved."
        return analysis

    # -------------------------
    # Public RCA entrypoint
    # -------------------------
    def perform_enhanced_rca(self, incident_data: Dict) -> Dict[str, Any]:
        """Perform enhanced RCA using similarity (RAG) + LLM, with graceful fallback."""
        logger.info(f"Performing enhanced RCA for {incident_data.get('type', 'unknown')}")
        similar_incidents = self.search_similar_incidents(incident_data)
        analysis = self.generate_llm_analysis_with_rag(incident_data, similar_incidents)
        return {
            'analysis': analysis,
            'similar_incidents_found': len(similar_incidents),
            'most_similar_incident': similar_incidents[0] if similar_incidents else None,
            'analysis_timestamp': datetime.now().isoformat(),
            'rag_enhanced': len(similar_incidents) > 0,
        }

    # -------------------------
    # Persistence
    # -------------------------
    def save_knowledge_base(self, filename: str):
        try:
            with open(filename, 'wb') as f:
                pickle.dump({
                    'knowledge_base': self.knowledge_base,
                    'vector_db_state': faiss.serialize_index(self.vector_db),
                }, f)
            logger.info(f"Knowledge base saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")

    def load_knowledge_base(self, filename: str):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.knowledge_base = data['knowledge_base']
            self.vector_db = faiss.deserialize_index(data['vector_db_state'])
        logger.info(f"Knowledge base loaded from {filename} with {len(self.knowledge_base)} entries")


# Manual test
if __name__ == "__main__":
    analyzer = RAGLLMAnalyzer()
    test_incident = {
        'system': 'java_app',
        'type': 'OutOfMemoryError',
        'message': 'OutOfMemoryError: Java heap space',
        'score': 9.0,
        'timestamp': datetime.now().isoformat()
    }
    result = analyzer.perform_enhanced_rca(test_incident)
    print("RCA Result:\n", result['analysis'])
