# slo_manager/dashboard_generator.py
import json
from pathlib import Path
from typing import Dict
from .slo_definitions import SLODefinition, SLO_REGISTRY

def generate_grafana_dashboard_json(slo_registry: Dict[str, SLODefinition]):
    """Generate a Grafana dashboard JSON for SLO monitoring"""
    dashboard = {
        "title": "SLO Overview (Auto-Generated)",
        "panels": [],
        "templating": {
            "list": [
                {
                    "name": "system",
                    "type": "query",
                    "datasource": "Prometheus",
                    "query": "label_values(java_app_cpu_usage, system)",
                    "refresh": 1,
                    "includeAll": True,
                    "multi": True,
                    "sort": 1
                }
            ]
        },
        "time": {
            "from": "now-6h",
            "to": "now"
        },
        "timezone": "browser",
        "uid": "slo-overview-auto"
    }
    
    # Create a row for each system
    systems = set(slo.system for slo in slo_registry.values())
    row_height = 8
    current_y = 0
    
    for system in systems:
        # Add a row for the system
        system_slos = [slo for slo in slo_registry.values() if slo.system == system]
        
        # Add a row panel
        row_panel = {
            "collapsed": False,
            "gridPos": {"h": 1, "w": 24, "x": 0, "y": current_y},
            "id": 1000 + current_y,
            "panels": [],
            "title": f"{system.upper()} System SLOs",
            "type": "row"
        }
        dashboard["panels"].append(row_panel)
        current_y += 1
        
        # Add panels for each SLO in this system
        for i, slo in enumerate(system_slos):
            panel = {
                "id": i + 1 + (current_y * 10),
                "title": slo.name,
                "type": "graph",
                "gridPos": {"h": row_height, "w": 12, "x": (i % 2) * 12, "y": current_y + (i // 2) * row_height},
                "targets": [{
                    "expr": slo.sli_query,
                    "format": "time_series",
                    "refId": "A"
                }],
                "datasource": "Prometheus",
                "description": slo.description
            }
            dashboard["panels"].append(panel)
        
        current_y += (len(system_slos) + 1) // 2 * row_height
    
    return dashboard

def save_grafana_dashboard(slo_registry: Dict[str, SLODefinition], output_path: Path):
    """Generate and save Grafana dashboard JSON"""
    dashboard = generate_grafana_dashboard_json(slo_registry)
    with open(output_path, 'w') as f:
        json.dump(dashboard, f, indent=2)
    print(f"✅ Grafana dashboard generated at {output_path}")

if __name__ == "__main__":
    save_grafana_dashboard(SLO_REGISTRY, Path("grafana_slo_dashboard.json"))