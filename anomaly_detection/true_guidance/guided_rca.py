# true_guidance/guided_rca.py
import requests
import json
from typing import Dict, List
from knowledge_graphs.kg_rca import KnowledgeGraphRCA
from reinforcement_rla.rl_rca import RCAReinforcementLearner

class GuidedRCA:
    def __init__(self, ollama_url="http://localhost:11434/api/generate"):
        self.ollama_url = ollama_url
        self.knowledge_graph = KnowledgeGraphRCA()
        self.rl_agent = RCAReinforcementLearner()
        self.case_history = []
    
    def analyze_incident(self, incident_data: Dict) -> Dict:
        """Perform guided RCA for an incident"""
        # Extract key information from the incident
        system = incident_data.get('system', 'unknown')
        error_message = incident_data.get('message', '')
        metrics = incident_data.get('metrics', {})
        
        # Step 1: Use knowledge graph to find potential resolutions
        kg_resolutions = self.knowledge_graph.find_relevant_resolutions(
            self._classify_issue(error_message), 
            incident_data
        )
        
        # Step 2: Use reinforcement learning to recommend actions
        rl_state = self._create_state_representation(incident_data)
        rl_recommendations = self.rl_agent.get_recommended_actions(rl_state)
        
        # Step 3: Use LLM to generate comprehensive guidance
        comprehensive_guidance = self._generate_comprehensive_guidance(
            incident_data, kg_resolutions, rl_recommendations
        )
        
        # Step 4: Learn from this incident to improve future recommendations
        self._learn_from_incident(incident_data, comprehensive_guidance)
        
        return {
            "incident": incident_data,
            "knowledge_graph_recommendations": kg_resolutions,
            "reinforcement_learning_recommendations": rl_recommendations,
            "comprehensive_guidance": comprehensive_guidance
        }
    
    def _classify_issue(self, error_message: str) -> str:
        """Classify the issue based on error message"""
        error_lower = error_message.lower()
        
        if "nullpointer" in error_lower or "npe" in error_lower:
            return "NullPointerException"
        elif "timeout" in error_lower and "connection" in error_lower:
            return "DatabaseConnectionTimeout"
        elif "outofmemory" in error_lower or "oom" in error_lower:
            return "OutOfMemoryError"
        elif "connection refused" in error_lower:
            return "ConnectionRefused"
        
        return "UnknownError"
    
    def _create_state_representation(self, incident_data: Dict) -> str:
        """Create a state representation for reinforcement learning"""
        system = incident_data.get('system', 'unknown')
        error_type = self._classify_issue(incident_data.get('message', ''))
        severity = incident_data.get('severity', 'medium')
        
        return f"{system}_{error_type}_{severity}"
    
    def _generate_comprehensive_guidance(self, incident_data: Dict, 
                                       kg_resolutions: List[Dict], 
                                       rl_recommendations: List[Dict]) -> str:
        """Generate comprehensive guidance using LLM"""
        prompt = self._create_guidance_prompt(incident_data, kg_resolutions, rl_recommendations)
        
        try:
            data = {
                "model": "llama3.1",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 500
                }
            }
            
            response = requests.post(self.ollama_url, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            return result['response']
        except Exception as e:
            return f"Error generating guidance: {e}"
    
    def _create_guidance_prompt(self, incident_data: Dict, 
                               kg_resolutions: List[Dict], 
                               rl_recommendations: List[Dict]) -> str:
        """Create a prompt for comprehensive guidance generation"""
        incident_desc = f"""
        Incident Details:
        - System: {incident_data.get('system', 'unknown')}
        - Error: {incident_data.get('message', 'unknown')}
        - Time: {incident_data.get('timestamp', 'unknown')}
        - Severity: {incident_data.get('severity', 'medium')}
        """
        
        kg_desc = "Knowledge Graph Recommendations:\n"
        for i, res in enumerate(kg_resolutions[:3]):  # Top 3 recommendations
            kg_desc += f"{i+1}. {res['resolution']} (confidence: {res['confidence']:.2f})\n"
            for step in res['steps'][:2]:  # First 2 steps
                kg_desc += f"   {step}\n"
        
        rl_desc = "Reinforcement Learning Recommendations:\n"
        for i, rec in enumerate(rl_recommendations[:3]):  # Top 3 recommendations
            rl_desc += f"{i+1}. {rec['action']} (confidence: {rec['confidence']:.2f})\n"
            for step in rec['steps'][:2]:  # First 2 steps
                rl_desc += f"   {step}\n"
        
        prompt = f"""
        You are an expert Site Reliability Engineer providing guided root cause analysis.
        
        Based on the following incident information and recommendations from our AI systems,
        provide comprehensive guidance that includes:
        
        1. IMMEDIATE ACTIONS: Specific, actionable steps to resolve the issue now
        2. ROOT CAUSE ANALYSIS: Explanation of what likely caused the issue
        3. PREVENTION STRATEGIES: How to prevent similar issues in the future
        4. MONITORING RECOMMENDATIONS: What to monitor to detect similar issues early
        
        {incident_desc}
        
        {kg_desc}
        
        {rl_desc}
        
        Please provide structured, actionable guidance that an SRE could follow directly.
        """
        
        return prompt
    
    def _learn_from_incident(self, incident_data: Dict, guidance: str):
        """Learn from this incident to improve future recommendations"""
        # Store the incident and guidance for future learning
        self.case_history.append({
            "incident": incident_data,
            "guidance": guidance,
            "timestamp": incident_data.get('timestamp', '')
        })
        
        # Here you would implement mechanisms to:
        # 1. Update the knowledge graph based on successful resolutions
        # 2. Update reinforcement learning rewards based on outcomes
        # 3. Refine the guidance generation based on feedback
        
        # For now, we'll just store the history
        print(f"Learned from incident: {incident_data.get('message', 'unknown')}")