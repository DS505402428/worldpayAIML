# true_guidance/guided_rca.py
import requests
import json
from typing import Dict, List
import random
import networkx as nx
from collections import defaultdict
import numpy as np
import re

class KnowledgeGraphRCA:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_enhanced_knowledge_graph()
    
    def _build_enhanced_knowledge_graph(self):
        """Build a comprehensive knowledge graph with specific guidance"""
        
        # Java Application Issues
        self._add_java_issues()
        
        # Kubernetes Issues
        self._add_kubernetes_issues()
        
        # COBOL Mainframe Issues
        self._add_cobol_issues()
        
        # Metric Anomalies
        self._add_metric_issues()
    
    def _add_java_issues(self):
        """Add Java-specific issues and resolutions"""
        # NullPointerException
        self.graph.add_node("Java_NullPointerException", type="issue",
                           description="Null pointer exception in Java application")
        self.graph.add_node("Java_NullPointer_Resolution", type="resolution",
                           description="Comprehensive NullPointerException resolution")
        
        self.graph.add_edge("Java_NullPointerException", "Java_NullPointer_Resolution",
                           relationship="resolved_by", confidence=0.9)
        
        # OutOfMemoryError
        self.graph.add_node("Java_OutOfMemoryError", type="issue",
                           description="Java heap space out of memory error")
        self.graph.add_node("Java_Memory_Resolution", type="resolution",
                           description="Memory issue resolution for Java")
        
        self.graph.add_edge("Java_OutOfMemoryError", "Java_Memory_Resolution",
                           relationship="resolved_by", confidence=0.85)
        
        # Database Connection Timeout
        self.graph.add_node("Java_DB_Timeout", type="issue",
                           description="Database connection timeout in Java application")
        self.graph.add_node("Java_DB_Resolution", type="resolution",
                           description="Database connection issue resolution for Java")
        
        self.graph.add_edge("Java_DB_Timeout", "Java_DB_Resolution",
                           relationship="resolved_by", confidence=0.82)
    
    def _add_kubernetes_issues(self):
        """Add Kubernetes-specific issues and resolutions"""
        # CPU Throttling
        self.graph.add_node("K8S_CPU_Throttling", type="issue",
                           description="Kubernetes CPU throttling detected")
        self.graph.add_node("K8S_CPU_Resolution", type="resolution",
                           description="CPU throttling resolution for Kubernetes")
        
        self.graph.add_edge("K8S_CPU_Throttling", "K8S_CPU_Resolution",
                           relationship="resolved_by", confidence=0.88)
        
        # Pod Crashes
        self.graph.add_node("K8S_Pod_Crash", type="issue",
                           description="Kubernetes pod crash due to OOM")
        self.graph.add_node("K8S_Pod_Resolution", type="resolution",
                           description="Pod crash resolution for Kubernetes")
        
        self.graph.add_edge("K8S_Pod_Crash", "K8S_Pod_Resolution",
                           relationship="resolved_by", confidence=0.87)
        
        # Image Pull Failures
        self.graph.add_node("K8S_Image_Pull_Failed", type="issue",
                           description="Kubernetes image pull failed")
        self.graph.add_node("K8S_Image_Resolution", type="resolution",
                           description="Image pull failure resolution for Kubernetes")
        
        self.graph.add_edge("K8S_Image_Pull_Failed", "K8S_Image_Resolution",
                           relationship="resolved_by", confidence=0.85)
        
        # Node Not Ready
        self.graph.add_node("K8S_Node_Not_Ready", type="issue",
                           description="Kubernetes node not ready")
        self.graph.add_node("K8S_Node_Resolution", type="resolution",
                           description="Node not ready resolution for Kubernetes")
        
        self.graph.add_edge("K8S_Node_Not_Ready", "K8S_Node_Resolution",
                           relationship="resolved_by", confidence=0.83)
    
    def _add_cobol_issues(self):
        """Add COBOL-specific issues and resolutions"""
        # Storage Issues
        self.graph.add_node("COBOL_Storage_Warning", type="issue",
                           description="COBOL storage allocation nearing limit")
        self.graph.add_node("COBOL_Storage_Resolution", type="resolution",
                           description="Storage issue resolution for COBOL")
        
        self.graph.add_edge("COBOL_Storage_Warning", "COBOL_Storage_Resolution",
                           relationship="resolved_by", confidence=0.82)
        
        # Job Abends
        self.graph.add_node("COBOL_Job_Abend", type="issue",
                           description="COBOL job abended with system error")
        self.graph.add_node("COBOL_Job_Resolution", type="resolution",
                           description="Job abend resolution for COBOL")
        
        self.graph.add_edge("COBOL_Job_Abend", "COBOL_Job_Resolution",
                           relationship="resolved_by", confidence=0.8)
        
        # CICS Transaction Issues
        self.graph.add_node("COBOL_CICS_Abend", type="issue",
                           description="CICS transaction abended")
        self.graph.add_node("COBOL_CICS_Resolution", type="resolution",
                           description="CICS transaction resolution for COBOL")
        
        self.graph.add_edge("COBOL_CICS_Abend", "COBOL_CICS_Resolution",
                           relationship="resolved_by", confidence=0.79)
    
    def _add_metric_issues(self):
        """Add metric-specific issues and resolutions"""
        # Generic metric anomaly
        self.graph.add_node("Metric_Anomaly", type="issue",
                           description="Abnormal metric values detected")
        self.graph.add_node("Metric_Investigation", type="resolution",
                           description="Metric anomaly investigation procedure")
        
        self.graph.add_edge("Metric_Anomaly", "Metric_Investigation",
                           relationship="resolved_by", confidence=0.75)
        
        # System-specific metric anomalies
        self.graph.add_node("Java_Metric_Anomaly", type="issue",
                           description="Abnormal Java metric values detected")
        self.graph.add_node("Java_Metric_Resolution", type="resolution",
                           description="Java metric anomaly resolution")
        
        self.graph.add_edge("Java_Metric_Anomaly", "Java_Metric_Resolution",
                           relationship="resolved_by", confidence=0.78)
        
        self.graph.add_node("K8S_Metric_Anomaly", type="issue",
                           description="Abnormal Kubernetes metric values detected")
        self.graph.add_node("K8S_Metric_Resolution", type="resolution",
                           description="Kubernetes metric anomaly resolution")
        
        self.graph.add_edge("K8S_Metric_Anomaly", "K8S_Metric_Resolution",
                           relationship="resolved_by", confidence=0.77)
        
        self.graph.add_node("COBOL_Metric_Anomaly", type="issue",
                           description="Abnormal COBOL metric values detected")
        self.graph.add_node("COBOL_Metric_Resolution", type="resolution",
                           description="COBOL metric anomaly resolution")
        
        self.graph.add_edge("COBOL_Metric_Anomaly", "COBOL_Metric_Resolution",
                           relationship="resolved_by", confidence=0.76)
    
    def find_relevant_resolutions(self, issue: str, context: Dict) -> List[Dict]:
        """Find relevant resolutions for an issue using the knowledge graph"""
        resolutions = []
        
        # First try exact match
        if issue in self.graph:
            for resolution in self.graph.successors(issue):
                if self.graph.nodes[resolution].get('type') == 'resolution':
                    edge_data = self.graph[issue][resolution]
                    resolutions.append({
                        'resolution': resolution,
                        'description': self.graph.nodes[resolution].get('description', ''),
                        'confidence': edge_data.get('confidence', 0.5),
                        'steps': self._generate_resolution_steps(resolution, context)
                    })
        
        # If no exact match, try pattern matching
        if not resolutions:
            resolved_issue = self._pattern_match_issue(issue)
            if resolved_issue and resolved_issue in self.graph:
                for resolution in self.graph.successors(resolved_issue):
                    if self.graph.nodes[resolution].get('type') == 'resolution':
                        edge_data = self.graph[resolved_issue][resolution]
                        resolutions.append({
                            'resolution': resolution,
                            'description': self.graph.nodes[resolution].get('description', ''),
                            'confidence': edge_data.get('confidence', 0.5) * 0.8,  # Lower confidence for pattern match
                            'steps': self._generate_resolution_steps(resolution, context)
                        })
        
        # Sort by confidence
        resolutions.sort(key=lambda x: x['confidence'], reverse=True)
        return resolutions
    
    def _pattern_match_issue(self, issue_message: str) -> str:
        """Pattern match issue message to known issue types"""
        issue_lower = issue_message.lower()
        
        # Java patterns
        if "nullpointer" in issue_lower or "npe" in issue_lower:
            return "Java_NullPointerException"
        elif "outofmemory" in issue_lower or "oom" in issue_lower:
            return "Java_OutOfMemoryError"
        elif "timeout" in issue_lower and ("database" in issue_lower or "connection" in issue_lower):
            return "Java_DB_Timeout"
        
        # Kubernetes patterns
        elif "cpu throttling" in issue_lower or "cpu usage" in issue_lower:
            return "K8S_CPU_Throttling"
        elif "pod crash" in issue_lower or "oom" in issue_lower:
            return "K8S_Pod_Crash"
        elif "image pull" in issue_lower and "fail" in issue_lower:
            return "K8S_Image_Pull_Failed"
        elif "node not ready" in issue_lower or "node notready" in issue_lower:
            return "K8S_Node_Not_Ready"
        
        # COBOL patterns
        elif "storage" in issue_lower and ("limit" in issue_lower or "nearing" in issue_lower):
            return "COBOL_Storage_Warning"
        elif "abend" in issue_lower or "system error" in issue_lower:
            return "COBOL_Job_Abend"
        elif "cics" in issue_lower and "abend" in issue_lower:
            return "COBOL_CICS_Abend"
        
        # Metric patterns with system context
        elif "metric" in issue_lower and "anomal" in issue_lower:
            if "java" in issue_lower:
                return "Java_Metric_Anomaly"
            elif "k8s" in issue_lower or "kubernetes" in issue_lower:
                return "K8S_Metric_Anomaly"
            elif "cobol" in issue_lower:
                return "COBOL_Metric_Anomaly"
            else:
                return "Metric_Anomaly"
        
        return None
    
    def _generate_resolution_steps(self, resolution: str, context: Dict) -> List[str]:
        """Generate specific resolution steps based on the context"""
        system = context.get('system', 'unknown')
        
        resolution_steps = {
            "Java_NullPointer_Resolution": [
                "🔍 IMMEDIATE DIAGNOSIS:",
                "1. Check stack trace to identify exact line causing NPE",
                "2. Review recent code changes around that location",
                "3. Verify object initialization before usage",
                "",
                "🚀 IMMEDIATE ACTIONS:",
                "1. Add null check: if (object != null) { object.method(); }",
                "2. Use Optional.ofNullable(object).ifPresent(obj -> obj.method())",
                "3. Add defensive programming: Objects.requireNonNull(object, 'message')",
                "",
                "🛠️ DEBUGGING TOOLS:",
                "1. Use jstack to get thread dumps",
                "2. Enable -XX:+ShowCodeDetailsInExceptionMessages (Java 14+)",
                "3. Use IDE debugger to step through the code",
                "",
                "🛡️ PREVENTION:",
                "1. Enable @NonNull annotations in IDE",
                "2. Add static code analysis to catch potential NPEs",
                "3. Implement comprehensive unit tests for null scenarios"
            ],
            
            "Java_Memory_Resolution": [
                "🔍 IMMEDIATE DIAGNOSIS:",
                "1. Check JVM memory usage with jstat or similar tools",
                "2. Analyze heap dump with MAT or similar tool",
                "3. Identify memory leaks with profiling tools",
                "",
                "🚀 IMMEDIATE ACTIONS:",
                "1. Increase JVM heap size: -Xmx4g",
                "2. Optimize garbage collection settings",
                "3. Restart application if memory leak is suspected",
                "",
                "🛠️ DEBUGGING TOOLS:",
                "1. jmap -heap <pid> to check heap configuration",
                "2. jstat -gc <pid> to monitor garbage collection",
                "3. Eclipse MAT to analyze heap dumps",
                "",
                "🛡️ PREVENTION:",
                "1. Implement memory usage monitoring",
                "2. Regular performance profiling",
                "3. Code reviews focused on memory management"
            ],
            
            "Java_DB_Resolution": [
                "🔍 IMMEDIATE DIAGNOSIS:",
                "1. Check database connection pool status",
                "2. Verify database server availability",
                "3. Review network connectivity between app and DB",
                "",
                "🚀 IMMEDIATE ACTIONS:",
                "1. Increase connection timeout settings",
                "2. Optimize connection pool configuration",
                "3. Verify database user permissions and limits",
                "",
                "🛠️ DEBUGGING TOOLS:",
                "1. Check database logs for connection errors",
                "2. Use network monitoring tools to check connectivity",
                "3. Test database connection with standalone tool",
                "",
                "🛡️ PREVENTION:",
                "1. Implement connection pool monitoring",
                "2. Add circuit breakers for database calls",
                "3. Regular database performance tuning"
            ],
            
            "K8S_CPU_Resolution": [
                "🔍 IMMEDIATE DIAGNOSIS:",
                "1. Check kubectl top pods to identify CPU usage",
                "2. Review pod CPU limits and requests",
                "3. Examine application performance metrics",
                "",
                "🚀 IMMEDIATE ACTIONS:",
                "1. Increase CPU limits: kubectl set resources deployment <name> --limits=cpu=1000m",
                "2. Add horizontal pod autoscaling: kubectl autoscale deployment <name> --cpu-percent=80 --min=1 --max=10",
                "3. Optimize application CPU usage",
                "",
                "🛠️ DEBUGGING TOOLS:",
                "1. kubectl describe pod <pod-name> to see resource limits",
                "2. kubectl top nodes to check node CPU capacity",
                "3. Application performance monitoring tools",
                "",
                "🛡️ PREVENTION:",
                "1. Implement proper resource profiling",
                "2. Set realistic CPU requests and limits",
                "3. Add CPU monitoring and alerts"
            ],
            
            "K8S_Pod_Resolution": [
                "🔍 IMMEDIATE DIAGNOSIS:",
                "1. Check pod logs for error messages",
                "2. Review pod events with kubectl describe pod",
                "3. Examine resource limits and usage",
                "",
                "🚀 IMMEDIATE ACTIONS:",
                "1. Increase memory limits if OOM: kubectl set resources deployment <name> --limits=memory=2Gi",
                "2. Restart failing pods: kubectl delete pod <pod-name>",
                "3. Check node resources and availability",
                "",
                "🛠️ DEBUGGING TOOLS:",
                "1. kubectl logs <pod-name> --previous to see previous pod logs",
                "2. kubectl get events --all-namespaces to see cluster events",
                "3. kubectl describe node <node-name> to check node resources",
                "",
                "🛡️ PREVENTION:",
                "1. Implement proper resource profiling",
                "2. Add liveness and readiness probes",
                "3. Set up pod disruption budgets"
            ],
            
            "K8S_Image_Resolution": [
                "🔍 IMMEDIATE DIAGNOSIS:",
                "1. Check image pull secrets configuration",
                "2. Verify container registry accessibility",
                "3. Review network connectivity to registry",
                "",
                "🚀 IMMEDIATE ACTIONS:",
                "1. Update image pull secrets: kubectl create secret docker-registry",
                "2. Check registry authentication credentials",
                "3. Verify image tag exists in registry",
                "",
                "🛠️ DEBUGGING TOOLS:",
                "1. kubectl describe pod <pod-name> to see image pull errors",
                "2. Test registry access with docker pull command",
                "3. Check network connectivity to registry",
                "",
                "🛡️ PREVENTION:",
                "1. Implement image cache strategies",
                "2. Add registry health monitoring",
                "3. Use image availability checks in CI/CD"
            ],
            
            "K8S_Node_Resolution": [
                "🔍 IMMEDIATE DIAGNOSIS:",
                "1. Check node status: kubectl get nodes",
                "2. Review node events: kubectl describe node <node-name>",
                "3. Check node resource usage: kubectl top nodes",
                "",
                "🚀 IMMEDIATE ACTIONS:",
                "1. Restart kubelet on affected node: systemctl restart kubelet",
                "2. Drain and cordon node: kubectl drain <node-name> --ignore-daemonsets",
                "3. Check node hardware and OS issues",
                "",
                "🛠️ DEBUGGING TOOLS:",
                "1. Check kubelet logs: journalctl -u kubelet",
                "2. Verify node network connectivity",
                "3. Check node hardware status (memory, disk, CPU)",
                "",
                "🛡️ PREVENTION:",
                "1. Implement node health monitoring",
                "2. Set up automatic node repair where possible",
                "3. Regular node maintenance and updates"
            ],
            
            "COBOL_Storage_Resolution": [
                "🔍 IMMEDIATE DIAGNOSIS:",
                "1. Check current storage allocation usage",
                "2. Review dataset sizes and growth patterns",
                "3. Examine job processing metrics",
                "",
                "🚀 IMMEDIATE ACTIONS:",
                "1. Clean up temporary datasets: IDCAMS DELETE",
                "2. Compress existing datasets: IEHLIST or similar",
                "3. Request temporary storage increase if needed",
                "",
                "🛠️ DEBUGGING TOOLS:",
                "1. Use DFSMSdss to analyze storage usage",
                "2. Check SMF records for storage-related events",
                "3. Use ISPF utilities to examine dataset attributes",
                "",
                "🛡️ PREVENTION:",
                "1. Implement automated dataset cleanup",
                "2. Add storage monitoring and alerts",
                "3. Optimize batch job storage usage"
            ],
            
            "COBOL_Job_Resolution": [
                "🔍 IMMEDIATE DIAGNOSIS:",
                "1. Check job output for error codes and messages",
                "2. Review system logs for related events",
                "3. Verify program compilation status",
                "",
                "🚀 IMMEDIATE ACTIONS:",
                "1. Check JCL parameters and dataset allocations",
                "2. Verify program dependencies and linkages",
                "3. Review recent program changes",
                "",
                "🛠️ DEBUGGING TOOLS:",
                "1. Use SDSF to examine job output and syslog",
                "2. Check abend codes in system manuals",
                "3. Use File Manager to examine dataset contents",
                "",
                "🛡️ PREVENTION:",
                "1. Implement comprehensive job monitoring",
                "2. Add automated job validation",
                "3. Maintain program change documentation"
            ],
            
            "COBOL_CICS_Resolution": [
                "🔍 IMMEDIATE DIAGNOSIS:",
                "1. Check CICS region logs for error messages",
                "2. Review transaction dump output",
                "3. Verify program and map availability",
                "",
                "🚀 IMMEDIATE ACTIONS:",
                "1. Restart affected CICS transaction",
                "2. Check CICS system definitions (CSD)",
                "3. Verify resource availability (files, databases)",
                "",
                "🛠️ DEBUGGING TOOLS:",
                "1. Use CICS transaction tracer",
                "2. Examine CICS monitoring facility (CMF) data",
                "3. Check CICS system dump for details",
                "",
                "🛡️ PREVENTION:",
                "1. Implement CICS transaction monitoring",
                "2. Add automated CICS health checks",
                "3. Regular CICS region maintenance"
            ],
            
            "Metric_Investigation": [
                "🔍 IMMEDIATE DIAGNOSIS:",
                "1. Check which specific metrics are anomalous",
                "2. Review correlation between different metric anomalies",
                "3. Examine recent system changes or deployments",
                "",
                "🚀 IMMEDIATE ACTIONS:",
                "1. Identify root cause system among Java/K8S/COBOL",
                "2. Check application-specific metrics for that system",
                "3. Review logs for correlated error messages",
                "",
                "🛠️ DEBUGGING TOOLS:",
                "1. Use Grafana to visualize metric trends",
                "2. Check Prometheus queries for metric anomalies",
                "3. Use correlation analysis tools",
                "",
                "🛡️ PREVENTION:",
                "1. Implement cross-system correlation alerts",
                "2. Add automated root cause analysis",
                "3. Create runbooks for common metric anomaly patterns"
            ],
            
            "Java_Metric_Resolution": [
                "🔍 IMMEDIATE DIAGNOSIS:",
                "1. Check Java-specific metrics: heap usage, GC activity, thread count",
                "2. Review application performance metrics",
                "3. Examine JVM configuration and settings",
                "",
                "🚀 IMMEDIATE ACTIONS:",
                "1. Adjust JVM memory settings if needed",
                "2. Check for memory leaks or inefficient code",
                "3. Review recent application deployments",
                "",
                "🛠️ DEBUGGING TOOLS:",
                "1. Use JMX to monitor JVM metrics",
                "2. Analyze garbage collection logs",
                "3. Use APM tools for application performance",
                "",
                "🛡️ PREVENTION:",
                "1. Implement comprehensive JVM monitoring",
                "2. Set up alerts for key Java metrics",
                "3. Regular performance tuning and optimization"
            ],
            
            "K8S_Metric_Resolution": [
                "🔍 IMMEDIATE DIAGNOSIS:",
                "1. Check Kubernetes-specific metrics: node CPU/memory, pod restarts",
                "2. Review cluster capacity and resource usage",
                "3. Examine deployment and replica configurations",
                "",
                "🚀 IMMEDIATE ACTIONS:",
                "1. Adjust resource requests and limits if needed",
                "2. Scale deployments based on resource usage",
                "3. Check node health and capacity",
                "",
                "🛠️ DEBUGGING TOOLS:",
                "1. Use kubectl top to check resource usage",
                "2. Review Kubernetes dashboard metrics",
                "3. Check cluster autoscaler logs if enabled",
                "",
                "🛡️ PREVENTION:",
                "1. Implement comprehensive Kubernetes monitoring",
                "2. Set up horizontal pod autoscaling",
                "3. Regular cluster capacity planning"
            ],
            
            "COBOL_Metric_Resolution": [
                "🔍 IMMEDIATE DIAGNOSIS:",
                "1. Check COBOL-specific metrics: job duration, CPU usage, storage",
                "2. Review batch job scheduling and performance",
                "3. Examine mainframe system metrics",
                "",
                "🚀 IMMEDIATE ACTIONS:",
                "1. Optimize batch job scheduling if needed",
                "2. Check for resource contention issues",
                "3. Review recent job configuration changes",
                "",
                "🛠️ DEBUGGING TOOLS:",
                "1. Use SMF records to analyze job performance",
                "2. Check mainframe monitoring tools (RMF, CMF)",
                "3. Review job control language (JCL) parameters",
                "",
                "🛡️ PREVENTION:",
                "1. Implement comprehensive mainframe monitoring",
                "2. Set up alerts for key COBOL metrics",
                "3. Regular performance tuning of batch jobs"
            ]
        }
        
        return resolution_steps.get(resolution, [
            "1. Investigate the specific error message",
            "2. Check system metrics for correlated anomalies",
            "3. Review recent deployments or configuration changes"
        ])

class RCAReinforcementLearner:
    def __init__(self):
        self.q_table = defaultdict(lambda: np.zeros(3))  # State -> action values
        self.actions = [
            "check_logs", 
            "restart_service", 
            "rollback_deployment"
        ]
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.3
        
        # Pre-defined rewards for different outcomes
        self.rewards = {
            "success": 10,
            "partial_success": 5,
            "no_effect": 0,
            "worsened": -5,
            "catastrophic": -10
        }
    
    def choose_action(self, state: str) -> str:
        """Choose an action based on current state using epsilon-greedy policy"""
        if np.random.random() < self.exploration_rate:
            # Explore: choose a random action
            return np.random.choice(self.actions)
        else:
            # Exploit: choose the best known action
            action_values = self.q_table[state]
            return self.actions[np.argmax(action_values)]
    
    def update_q_value(self, state: str, action: str, reward: float, next_state: str):
        """Update Q-value based on the observed reward"""
        action_index = self.actions.index(action)
        old_value = self.q_table[state][action_index]
        
        # Q-learning formula
        next_max = np.max(self.q_table[next_state])
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_max - old_value)
        
        self.q_table[state][action_index] = new_value
    
    def get_recommended_actions(self, state: str, top_n: int = 3) -> List[Dict]:
        """Get recommended actions for a given state"""
        if state not in self.q_table:
            return []
        
        action_values = self.q_table[state]
        # Get indices of top N actions
        top_indices = np.argsort(action_values)[-top_n:][::-1]
        
        recommendations = []
        for idx in top_indices:
            recommendations.append({
                'action': self.actions[idx],
                'confidence': action_values[idx],
                'steps': self._get_action_steps(self.actions[idx])
            })
        
        return recommendations
    
    def _get_action_steps(self, action: str) -> List[str]:
        """Get detailed steps for an action"""
        action_steps = {
            "check_logs": [
                "1. Identify relevant log files based on the issue",
                "2. Search for error patterns and exceptions",
                "3. Correlate timestamps with incident reports",
                "4. Document findings for future reference"
            ],
            "restart_service": [
                "1. Identify the affected service",
                "2. Check if there are any dependencies",
                "3. Execute graceful restart command",
                "4. Verify service health after restart"
            ],
            "rollback_deployment": [
                "1. Identify the recent deployment",
                "2. Check rollback compatibility",
                "3. Execute rollback procedure",
                "4. Verify system stability post-rollback"
            ]
        }
        
        return action_steps.get(action, ["No specific steps available"])

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
        
        # Step 3: Use LLM to generate comprehensive guidance (if available)
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
            return "Java_NullPointerException"
        elif "outofmemory" in error_lower or "oom" in error_lower:
            return "Java_OutOfMemoryError"
        elif "timeout" in error_lower and ("database" in error_lower or "connection" in error_lower):
            return "Java_DB_Timeout"
        elif "cpu throttling" in error_lower or "cpu usage" in error_lower:
            return "K8S_CPU_Throttling"
        elif "pod crash" in error_lower or "oom" in error_lower:
            return "K8S_Pod_Crash"
        elif "image pull" in error_lower and "fail" in error_lower:
            return "K8S_Image_Pull_Failed"
        elif "node not ready" in error_lower or "node notready" in error_lower:
            return "K8S_Node_Not_Ready"
        elif "storage" in error_lower and ("limit" in error_lower or "nearing" in error_lower):
            return "COBOL_Storage_Warning"
        elif "abend" in error_lower or "system error" in error_lower:
            return "COBOL_Job_Abend"
        elif "cics" in error_lower and "abend" in error_lower:
            return "COBOL_CICS_Abend"
        elif "metric" in error_lower and "anomal" in error_lower:
            if "java" in error_lower:
                return "Java_Metric_Anomaly"
            elif "k8s" in error_lower or "kubernetes" in error_lower:
                return "K8S_Metric_Anomaly"
            elif "cobol" in error_lower:
                return "COBOL_Metric_Anomaly"
            else:
                return "Metric_Anomaly"
        
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
        """Generate comprehensive guidance with specific, actionable steps"""
        
        system = incident_data.get('system', 'unknown')
        error_type = incident_data.get('type', '')
        error_message = incident_data.get('message', '')
        
        # Use system-specific guidance templates
        if system in ['java_app', 'kubernetes', 'cobol_mainframe']:
            guidance = self._generate_system_specific_guidance(system, error_type, error_message)
        else:
            # Fallback to generic guidance
            guidance = self._generate_generic_guidance(incident_data)
        
        # Add knowledge graph recommendations if available
        if kg_resolutions:
            guidance += "\n\n🎯 KNOWLEDGE-BASED RECOMMENDATIONS:\n"
            for i, res in enumerate(kg_resolutions[:3]):
                guidance += f"{i+1}. {res['resolution']} (confidence: {res['confidence']:.2f})\n"
                for step in res['steps']:
                    guidance += f"   {step}\n"
        
        # Add reinforcement learning recommendations if available
        if rl_recommendations:
            guidance += "\n\n🤖 AI-RECOMMENDED ACTIONS:\n"
            for i, rec in enumerate(rl_recommendations[:3]):
                guidance += f"{i+1}. {rec['action']} (confidence: {rec['confidence']:.2f})\n"
                for step in rec['steps']:
                    guidance += f"   {step}\n"
        
        return guidance

    def _generate_system_specific_guidance(self, system: str, error_type: str, error_message: str) -> str:
        """Generate system-specific guidance based on error type"""
        
        guidance_templates = {
            'java_app': {
                'NullPointerException': """
🔍 JAVA NULLPOINTEREXCEPTION DETECTED:

🚀 IMMEDIATE ACTIONS:
1. Locate the exact line causing the NPE using stack trace
2. Check if the object was properly initialized before use
3. Add null checks: if (object != null) { object.method(); }

🛠️ DEBUGGING STEPS:
1. Use jstack to get thread dump
2. Check recent code changes around the affected area
3. Verify dependency injections are working correctly

🛡️ PREVENTION:
1. Use Optional<> for potentially null values
2. Add @NonNull annotations
3. Implement comprehensive unit tests for null scenarios
""",
                'OutOfMemoryError': """
🔍 JAVA OUTOFMEMORYERROR DETECTED:

🚀 IMMEDIATE ACTIONS:
1. Increase JVM heap size: -Xmx4g
2. Restart the application
3. Check for memory leaks with jmap or similar tools

🛠️ DEBUGGING STEPS:
1. Analyze heap dump with MAT or similar tool
2. Check for large object allocations
3. Review garbage collection logs

🛡️ PREVENTION:
1. Implement memory usage monitoring
2. Set up alerting for memory thresholds
3. Regular performance profiling
""",
                'Java_DB_Timeout': """
🔍 JAVA DATABASE CONNECTION TIMEOUT DETECTED:

🚀 IMMEDIATE ACTIONS:
1. Increase database connection timeout settings
2. Check database server availability and performance
3. Verify network connectivity between app and DB

🛠️ DEBUGGING STEPS:
1. Check database connection pool status
2. Review database server logs for issues
3. Test database connectivity with standalone tool

🛡️ PREVENTION:
1. Implement connection pool monitoring
2. Add circuit breakers for database calls
3. Regular database performance tuning
"""
            },
            'kubernetes': {
                'K8S_CPU_Throttling': """
🔍 KUBERNETES CPU THROTTLING DETECTED:

🚀 IMMEDIATE ACTIONS:
1. Increase CPU limits: kubectl set resources deployment <name> --limits=cpu=2
2. Add more replicas to distribute load
3. Check node CPU capacity

🛠️ DEBUGGING STEPS:
1. Check pod CPU usage: kubectl top pods
2. Review application performance metrics
3. Examine node resource allocation

🛡️ PREVENTION:
1. Implement proper CPU profiling
2. Set up HPA with CPU-based scaling
3. Add CPU usage alerts
""",
                'K8S_Pod_Crash': """
🔍 KUBERNETES POD CRASH (OOM) DETECTED:

🚀 IMMEDIATE ACTIONS:
1. Increase memory limits: kubectl set resources deployment <name> --limits=memory=2Gi
2. Restart the pod: kubectl delete pod <pod-name>
3. Check node memory pressure

🛠️ DEBUGGING STEPS:
1. Check pod logs: kubectl logs <pod-name> --previous
2. Review resource usage: kubectl top pods
3. Examine deployment configuration

🛡️ PREVENTION:
1. Implement proper resource profiling
2. Set up HPA with memory-based scaling
3. Add memory usage alerts
""",
                'K8S_Node_Not_Ready': """
🔍 KUBERNETES NODE NOT READY DETECTED:

🚀 IMMEDIATE ACTIONS:
1. Restart kubelet on affected node: systemctl restart kubelet
2. Drain and cordon node: kubectl drain <node-name> --ignore-daemonsets
3. Check node hardware and OS issues

🛠️ DEBUGGING STEPS:
1. Check kubelet logs: journalctl -u kubelet
2. Verify node network connectivity
3. Check node hardware status (memory, disk, CPU)

🛡️ PREVENTION:
1. Implement node health monitoring
2. Set up automatic node repair where possible
3. Regular node maintenance and updates
"""
            },
            'cobol_mainframe': {
                'COBOL_Storage_Warning': """
🔍 COBOL STORAGE WARNING DETECTED:

🚀 IMMEDIATE ACTIONS:
1. Clean up temporary datasets: IDCAMS DELETE
2. Compress existing datasets: IEHLIST or similar
3. Request temporary storage increase if needed

🛠️ DEBUGGING STEPS:
1. Use DFSMSdss to analyze storage usage
2. Check SMF records for storage-related events
3. Use ISPF utilities to examine dataset attributes

🛡️ PREVENTION:
1. Implement automated dataset cleanup
2. Add storage monitoring and alerts
3. Optimize batch job storage usage
""",
                'COBOL_Job_Abend': """
🔍 COBOL JOB ABEND DETECTED:

🚀 IMMEDIATE ACTIONS:
1. Check JCL parameters and dataset allocations
2. Verify program dependencies and linkages
3. Review recent program changes

🛠️ DEBUGGING STEPS:
1. Use SDSF to examine job output and syslog
2. Check abend codes in system manuals
3. Use File Manager to examine dataset contents

🛡️ PREVENTION:
1. Implement comprehensive job monitoring
2. Add automated job validation
3. Maintain program change documentation
"""
            }
        }
        
        # Get the appropriate guidance template
        system_templates = guidance_templates.get(system, {})
        specific_guidance = system_templates.get(error_type, "")
        
        if specific_guidance:
            return specific_guidance
        else:
            return self._generate_generic_guidance({
                'system': system,
                'type': error_type,
                'message': error_message
            })

    def _generate_generic_guidance(self, incident_data: Dict) -> str:
        """Generate generic guidance when no specific template is available"""
        system = incident_data.get('system', 'unknown')
        error_type = incident_data.get('type', 'unknown')
        error_message = incident_data.get('message', 'unknown')
        
        return f"""
🔍 {error_type.upper()} DETECTED IN {system.upper()}:

🚀 IMMEDIATE ACTIONS:
1. Investigate the specific error message: {error_message}
2. Check system metrics for correlated anomalies
3. Review recent deployments or configuration changes
4. Verify system connectivity and dependencies

🛠️ DEBUGGING STEPS:
1. Check all related system logs
2. Review performance metrics trends
3. Verify resource utilization
4. Check for correlated events across systems

🛡️ PREVENTION:
1. Implement comprehensive monitoring
2. Add automated alerting for anomalies
3. Regular system health checks
4. Maintain incident runbooks
"""

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
    
    def _generate_fallback_guidance(self, incident_data: Dict, 
                                  kg_resolutions: List[Dict], 
                                  rl_recommendations: List[Dict]) -> str:
        """Generate fallback guidance when LLM is not available"""
        system = incident_data.get('system', 'unknown')
        error_message = incident_data.get('message', 'unknown')
        
        guidance = f"""
GUIDED RCA FOR INCIDENT:
System: {system}
Error: {error_message}

KNOWLEDGE GRAPH RECOMMENDATIONS:
"""
        
        for i, res in enumerate(kg_resolutions[:3]):
            guidance += f"{i+1}. {res['resolution']} (confidence: {res['confidence']:.2f})\n"
            for step in res['steps']:
                guidance += f"   {step}\n"
        
        guidance += "\nREINFORCEMENT LEARNING RECOMMENDATIONS:\n"
        for i, rec in enumerate(rl_recommendations[:3]):
            guidance += f"{i+1}. {rec['action']} (confidence: {rec['confidence']:.2f})\n"
            for step in rec['steps']:
                guidance += f"   {step}\n"
        
        guidance += """
        
IMMEDIATE ACTIONS:
1. Investigate the specific error message in the context of recent changes
2. Check system metrics for any correlated anomalies
3. Review recent deployments or configuration changes

LONG-TERM PREVENTION:
1. Implement additional monitoring for this type of error
2. Add automated remediation steps where possible
3. Document this incident and resolution for future reference
"""
        
        return guidance
    
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