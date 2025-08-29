# slo_manager/slo_definitions.py
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from enum import Enum
from datetime import datetime

class SystemType(str, Enum):
    JAVA_APP = "java_app"
    KUBERNETES = "kubernetes"
    COBOL_MAINFRAME = "cobol_mainframe"

class SLODefinition(BaseModel):
    system: SystemType
    name: str
    description: str
    sli_query: str
    target: float
    rolling_period_days: int = 30
    alert_burn_rate_threshold: float = 10.0
    alert_short_window: str = "1h"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

# Define all SLOs for all systems
SLO_REGISTRY: Dict[str, SLODefinition] = {
    "java_availability": SLODefinition(
        system=SystemType.JAVA_APP,
        name="Java Application Availability",
        description="Successful HTTP requests",
        sli_query="""
        (sum(rate(http_requests_total{job="java_app", status!~"5.."}[5m]))
        /
        (sum(rate(http_requests_total{job="java_app"}[5m])))
        """,
        target=0.9995,
    ),
    "java_latency": SLODefinition(
        system=SystemType.JAVA_APP,
        name="Java Application Latency",
        description="95th percentile latency under 500ms",
        sli_query="""
        histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{job="java_app"}[5m])) by (le)) < 0.5
        """,
        target=0.95,
    ),
    "k8s_pod_availability": SLODefinition(
        system=SystemType.KUBERNETES,
        name="Kubernetes Pod Availability",
        description="Pods running successfully",
        sli_query="""
        sum(kube_pod_status_ready{condition="true"}) / sum(kube_pod_info)
        """,
        target=0.9999,
    ),
    "cobol_error_rate": SLODefinition(
        system=SystemType.COBOL_MAINFRAME,
        name="COBOL Job Success Rate",
        description="Batch jobs that do not abend",
        sli_query="1 - (increase(cobol_error_count_total[5m]) / increase(cobol_jobs_processed_total[5m]))",
        target=0.995,
    ),
}