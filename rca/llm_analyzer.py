# rca/llm_analyzer.py
import requests
import json
import numpy as np
from datetime import datetime, timedelta
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMAnalyzer:
    def __init__(self, ollama_url="http://ollama:11434/api/generate"):
        self.ollama_url = ollama_url
        self.log_db = []
        self.ollama_available = False
        self._initialize_ollama()
        
    def _initialize_ollama(self, max_retries=5, retry_delay=5):
        """Initialize Ollama connection with retry logic"""
        for attempt in range(max_retries):
            try:
                response = requests.get("http://ollama:11434/api/tags", timeout=10)
                if response.status_code == 200:
                    self.ollama_available = True
                    logger.info("✅ Ollama connected successfully")
                    # Ensure we have the required model
                    models = response.json().get('models', [])
                    if not any('llama' in model['name'] for model in models):
                        logger.warning("Llama model not found, pulling...")
                        self._pull_model()
                    return
            except requests.exceptions.ConnectionError:
                logger.warning(f"Ollama not available (attempt {attempt+1}/{max_retries}), retrying...")
                time.sleep(retry_delay)
            except Exception as e:
                logger.error(f"Error checking Ollama: {e}")
                time.sleep(retry_delay)
        
        logger.warning("⚠️ Ollama not available after retries. Using fallback analysis.")
        self.ollama_available = False
    
    def _pull_model(self):
        """Pull the required model if not available"""
        try:
            pull_response = requests.post(
                "http://ollama:11434/api/pull",
                json={"name": "llama3.1"},
                timeout=120
            )
            if pull_response.status_code == 200:
                logger.info("✅ Llama model pulled successfully")
            else:
                logger.warning(f"Failed to pull model: {pull_response.text}")
        except Exception as e:
            logger.error(f"Error pulling model: {e}")
    
    def add_to_vector_db(self, log_entry):
        """Add log entry to database"""
        self.log_db.append(log_entry)
    
    def search_similar_logs(self, query, k=5):
        """Search for similar logs in database using semantic similarity"""
        # Simple keyword-based search for demo
        similar_logs = []
        query_words = query.lower().split()
        
        for log in self.log_db:
            score = 0
            message = log.get('message', '').lower()
            system = log.get('system', '').lower()
            
            # Score based on keyword matches
            for word in query_words:
                if word in message:
                    score += 2
                if word in system:
                    score += 1
            
            if score > 0:
                similar_logs.append((score, log))
        
        # Sort by score and return top k
        similar_logs.sort(key=lambda x: x[0], reverse=True)
        return [log for score, log in similar_logs[:k]]
    
    def generate_llm_analysis(self, prompt, max_retries=3):
        """Generate analysis using LLM with improved retry logic"""
        if not self.ollama_available:
            return self._generate_fallback_analysis(prompt)
            
        for attempt in range(max_retries):
            try:
                data = {
                    "model": "llama3.1",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_predict": 500  # Increased response length
                    }
                }
                
                response = requests.post(self.ollama_url, json=data, timeout=30)
                response.raise_for_status()
                result = response.json()
                
                return result['response']
            except requests.exceptions.Timeout:
                logger.warning(f"LLM timeout (attempt {attempt + 1}/{max_retries}), retrying...")
                time.sleep(2)
            except Exception as e:
                logger.error(f"LLM error: {e}")
                # Check if Ollama is still available
                self._initialize_ollama(max_retries=1)
                if not self.ollama_available:
                    return self._generate_fallback_analysis(prompt)
                
        logger.warning("LLM timeout after retries, using fallback")
        return self._generate_fallback_analysis(prompt)
    
    def _generate_fallback_analysis(self, prompt):
        """Generate fallback analysis when LLM is unavailable"""
        # Extract key information from the prompt for fallback analysis
        prompt_lower = prompt.lower()
        
        if "java" in prompt_lower:
            return """FALLBACK RCA FOR JAVA APPLICATION:
ROOT CAUSE: NullPointerException typically indicates missing object initialization or null references.
IMMEDIATE ACTION: Check recent code deployments, review stack traces for the exact location of the NPE.
LONG-TERM: Implement better null checking and use Optional types where appropriate."""
        
        elif "kubernetes" in prompt_lower or "k8s" in prompt_lower:
            return """FALLBACK RCA FOR KUBERNETES:
ROOT CAUSE: Pod crashes often relate to resource constraints (CPU/Memory) or misconfigured deployments.
IMMEDIATE ACTION: Check pod logs with kubectl, verify resource requests/limits.
LONG-TERM: Implement resource monitoring and autoscaling."""
        
        elif "cobol" in prompt_lower:
            return """FALLBACK RCA FOR COBOL SYSTEM:
ROOT CAUSE: Storage allocation issues typically indicate dataset space exhaustion or inefficient job processing.
IMMEDIATE ACTION: Check dataset allocations, verify job scheduling.
LONG-TERM: Implement storage monitoring and cleanup procedures."""
        
        elif "metric" in prompt_lower:
            return """FALLBACK RCA FOR METRIC ANOMALY:
ROOT CAUSE: Abnormal metric values often indicate resource exhaustion, performance degradation, or system overload.
IMMEDIATE ACTION: Check system resources, review recent changes or deployments.
LONG-TERM: Implement better capacity planning and auto-scaling."""
        
        else:
            return """FALLBACK RCA:
ROOT CAUSE: Unknown system issue detected.
IMMEDIATE ACTION: Review logs and metrics for unusual patterns.
LONG-TERM: Enhance monitoring and alerting for similar issues."""
    
    def perform_rca(self, anomaly_info, related_logs, metrics_data):
        """Perform root cause analysis using LLM with fallback"""
        # Create context from similar logs
        context = "Related historical incidents:\n"
        for i, log in enumerate(related_logs[:3]):  # Top 3 similar logs
            context += f"{i+1}. {log.get('timestamp', '')} - {log.get('system', '')} - {log.get('level', '')}: {log.get('message', '')}\n"
        
        # Create a more structured prompt with clear instructions
        prompt = f"""
        You are an expert Site Reliability Engineer (SRE) performing root cause analysis.
        
        Analyze this anomaly and provide a structured response with:
        1. ROOT CAUSE: [Brief explanation of the likely root cause]
        2. IMMEDIATE ACTION: [One specific action to take immediately]
        3. LONG-TERM FIX: [One recommendation to prevent recurrence]
        4. INVESTIGATION STEPS: [2-3 specific steps to investigate further]
        
        ANOMALY DETAILS:
        - System: {anomaly_info.get('system', 'unknown')}
        - Time: {anomaly_info.get('timestamp', 'unknown')}
        - Type: {anomaly_info.get('type', 'unknown')}
        - Message: {anomaly_info.get('message', 'unknown')}
        - Severity Score: {anomaly_info.get('score', 'unknown')}/10
        
        CONTEXT FROM SIMILAR INCIDENTS:
        {context}
        
        Please provide a concise, structured response.
        """
        
        return self.generate_llm_analysis(prompt)