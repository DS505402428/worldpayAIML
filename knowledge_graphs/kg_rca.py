# knowledge_graphs/kg_rca.py
import networkx as nx
from typing import Dict, List
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
    
    def _add_cobol_issues(self):
        """Add COBOL-specific issues and resolutions"""
        # Storage Issues
        self.graph.add_node("COBOL_Storage_Warning", type="issue",
                           description="COBOL storage allocation nearing limit")
        self.graph.add_node("COBOL_Storage_Resolution", type="resolution",
                           description="Storage issue resolution for COBOL")
        
        self.graph.add_edge("COBOL_Storage_Warning", "COBOL_Storage_Resolution",
                           relationship="resolved_by", confidence=0.82)
    
    def _add_metric_issues(self):
        """Add metric-specific issues and resolutions"""
        # Generic metric anomaly
        self.graph.add_node("Metric_Anomaly", type="issue",
                           description="Abnormal metric values detected")
        self.graph.add_node("Metric_Investigation", type="resolution",
                           description="Metric anomaly investigation procedure")
        
        self.graph.add_edge("Metric_Anomaly", "Metric_Investigation",
                           relationship="resolved_by", confidence=0.75)
    
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
        
        # Kubernetes patterns
        elif "cpu throttling" in issue_lower:
            return "K8S_CPU_Throttling"
        elif "pod crash" in issue_lower or "oom" in issue_lower:
            return "K8S_Pod_Crash"
        
        # COBOL patterns
        elif "storage" in issue_lower and ("limit" in issue_lower or "nearing" in issue_lower):
            return "COBOL_Storage_Warning"
        
        # Metric patterns
        elif "metric" in issue_lower and "anomal" in issue_lower:
            return "Metric_Anomaly"
        
        return None
        
    def export_runbook_documentation(self) -> Dict:
        """Exports all resolutions and steps for documentation."""
        runbooks = {}
        for issue_node in self.graph.nodes:
            if self.graph.nodes[issue_node].get('type') == 'issue':
                # Use a simple context for export
                resolutions = self.find_relevant_resolutions(issue_node, {'system': 'unknown'})
                runbooks[issue_node] = [
                    {
                        'resolution': r['resolution'],
                        'description': self.graph.nodes[r['resolution']].get('description', ''),
                        'steps': r['steps'],
                        'confidence': r.get('confidence', 0.5)
                    } for r in resolutions
                ]
        return runbooks
    
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
                "🛡️ PREVENTION:",
                "1. Enable @NonNull annotations in IDE",
                "2. Add static code analysis to catch potential NPEs",
                "3. Implement comprehensive unit tests for null scenarios"
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
                "🛡️ PREVENTION:",
                "1. Implement proper resource profiling",
                "2. Set realistic CPU requests and limits",
                "3. Add CPU monitoring and alerts"
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
                "🛡️ PREVENTION:",
                "1. Implement automated dataset cleanup",
                "2. Add storage monitoring and alerts",
                "3. Optimize batch job storage usage"
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
                "🛡️ PREVENTION:",
                "1. Implement cross-system correlation alerts",
                "2. Add automated root cause analysis",
                "3. Create runbooks for common metric anomaly patterns"
            ]
        }
        
        return resolution_steps.get(resolution, [
            "1. Investigate the specific error message",
            "2. Check system metrics for correlated anomalies",
            "3. Review recent deployments or configuration changes"
        ])