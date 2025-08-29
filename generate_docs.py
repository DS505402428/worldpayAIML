#!/usr/bin/env python3
# generate_docs.py
"""
Generate SLO documentation, Prometheus alert rules, Grafana dashboard JSON,
and (optionally) runbook docs using the knowledge graph RCA module.

This script is safe to run inside the container or locally. It normalizes
paths so outputs land under the repo root in ./generated_docs and
./generated_configs by default. You can override output files via env vars:

- SLO_DOC_OUT                (default: generated_docs/SLO_DOCUMENTATION.md)
- PROM_ALERTS_OUT            (default: generated_configs/prometheus_slo_alerts.yml)
- GRAFANA_DASHBOARD_OUT      (default: generated_configs/grafana_slo_dashboard.json)
- RUNBOOKS_OUT               (default: generated_docs/RUNBOOKS.md)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import os
import sys
import traceback

# --- Resolve repo root and ensure imports work no matter the CWD ---
ROOT = Path(__file__).resolve().parent  # repo root (where this file lives)
os.chdir(ROOT)  # ensure relative outputs land in the repo
# Make sure our package imports (slo_manager, knowledge_graphs, etc.) resolve
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Imports from the repo ---
from slo_manager.doc_generator import (  # type: ignore
    generate_slo_markdown_docs,
    generate_runbooks_markdown,
)
from slo_manager.alert_generator import generate_prometheus_alert_rules  # type: ignore
from slo_manager.dashboard_generator import save_grafana_dashboard  # type: ignore
from slo_manager.slo_definitions import SLO_REGISTRY  # type: ignore
from knowledge_graphs.kg_rca import KnowledgeGraphRCA  # type: ignore


def _ts() -> str:
    """UTC timestamp for logs."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_parents(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print(f"[{_ts()}] Generating SLO documentation and configuration...")

    # Resolve output targets (env overrides supported)
    slo_doc_out = Path(os.environ.get("SLO_DOC_OUT", "generated_docs/SLO_DOCUMENTATION.md"))
    prom_alerts_out = Path(os.environ.get("PROM_ALERTS_OUT", "generated_configs/prometheus_slo_alerts.yml"))
    grafana_dashboard_out = Path(os.environ.get("GRAFANA_DASHBOARD_OUT", "generated_configs/grafana_slo_dashboard.json"))
    runbooks_out = Path(os.environ.get("RUNBOOKS_OUT", "generated_docs/RUNBOOKS.md"))

    # Ensure output directories exist
    _ensure_parents(slo_doc_out)
    _ensure_parents(prom_alerts_out)
    _ensure_parents(grafana_dashboard_out)
    _ensure_parents(runbooks_out)

    failures = []

    # 1) SLO Markdown Documentation
    try:
        generate_slo_markdown_docs(SLO_REGISTRY, slo_doc_out)
        print(f"[{_ts()}] ✅ SLO documentation written to {slo_doc_out}")
    except Exception as e:
        failures.append("SLO documentation")
        print(f"[{_ts()}] ❌ Failed to generate SLO documentation: {e}")
        traceback.print_exc()

    # 2) Prometheus Alert Rules
    try:
        generate_prometheus_alert_rules(SLO_REGISTRY, prom_alerts_out)
        print(f"[{_ts()}] ✅ Prometheus alert rules written to {prom_alerts_out}")
    except Exception as e:
        failures.append("Prometheus alert rules")
        print(f"[{_ts()}] ❌ Failed to generate Prometheus alert rules: {e}")
        traceback.print_exc()

    # 3) Grafana Dashboard JSON
    try:
        save_grafana_dashboard(SLO_REGISTRY, grafana_dashboard_out)
        print(f"[{_ts()}] ✅ Grafana dashboard written to {grafana_dashboard_out}")
    except Exception as e:
        failures.append("Grafana dashboard")
        print(f"[{_ts()}] ❌ Failed to generate Grafana dashboard: {e}")
        traceback.print_exc()

    # 4) Runbooks (optional; depends on KnowledgeGraphRCA)
    try:
        kg_rca = KnowledgeGraphRCA()
        generate_runbooks_markdown(kg_rca, runbooks_out)
        print(f"[{_ts()}] ✅ Runbook documentation written to {runbooks_out}")
    except Exception as e:
        # Keep it as a warning so the rest still succeeds.
        print(f"[{_ts()}] ⚠️ Could not generate runbook documentation: {e}")
        # Uncomment if you want this to be a "failure" in CI:
        # failures.append("Runbook documentation")
        # traceback can be noisy in CI; print if you need details:
        traceback.print_exc()

    if failures:
        print(f"[{_ts()}] Completed with issues: {', '.join(failures)}")
    else:
        print(f"[{_ts()}] ✅ All documentation and configuration generated successfully!")


if __name__ == "__main__":
    main()
