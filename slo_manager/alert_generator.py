# slo_manager/alert_generator.py
import yaml
from pathlib import Path
from typing import Dict
from .slo_definitions import SLODefinition, SLO_REGISTRY

def generate_prometheus_alert_rules(slo_registry: Dict[str, SLODefinition], output_path: Path):
    """Generates Prometheus alerting rules for SLO error budget burn rates."""
    groups = [{
        'name': 'slo_error_budget_alerts',
        'rules': []
    }]

    for slo_id, slo in slo_registry.items():
        # Create burn rate alert
        alert_rule = {
            'alert': f"ErrorBudgetBurnRateTooHigh_{slo_id}",
            'expr': f"""
                1 - ({slo.sli_query}) > (1 - {slo.target}) * {slo.alert_burn_rate_threshold}
                """.strip(),
            'for': '5m',
            'labels': {
                'severity': 'critical',
                'system': slo.system,
                'slo_id': slo_id
            },
            'annotations': {
                'summary': f"{slo.name} is burning error budget too fast.",
                'description': f"Error budget burn rate for {slo.name} is exceeding {slo.alert_burn_rate_threshold}. Immediate investigation required.",
                'runbook_url': f"https://wiki.mycompany.com/runbooks/{slo.system}/error_budget_burn"
            }
        }
        groups[0]['rules'].append(alert_rule)

    # Write the rules to a file that Prometheus can load
    rules_yaml = {"groups": groups}
    with open(output_path, 'w') as f:
        yaml.dump(rules_yaml, f, sort_keys=False, default_flow_style=False)

    print(f"✅ Alert rules generated at {output_path}")

if __name__ == "__main__":
    generate_prometheus_alert_rules(SLO_REGISTRY, Path("prometheus_alerts.yml"))