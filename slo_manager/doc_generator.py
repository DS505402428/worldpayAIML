# slo_manager/doc_generator.py
import json
from pathlib import Path
from typing import Dict
from .slo_definitions import SLODefinition, SLO_REGISTRY
from knowledge_graphs.kg_rca import KnowledgeGraphRCA

def generate_slo_markdown_docs(slo_registry: Dict[str, SLODefinition], output_path: Path):
    """Generates a Markdown file for the company wiki from the SLO registry."""
    docs_content = "# Service Level Objectives (SLOs)\n\n"
    docs_content += "*Automatically generated from source. Do not edit manually.*\n\n"
    docs_content += f"*Last Updated: {datetime.now().isoformat()}*\n\n"

    # Group SLOs by system
    slos_by_system = {}
    for slo_id, slo in slo_registry.items():
        slos_by_system.setdefault(slo.system, []).append(slo)

    for system, slos in slos_by_system.items():
        docs_content += f"## {system.upper()} System\n\n"
        for slo in slos:
            docs_content += f"### {slo.name}\n"
            docs_content += f"**Target:** {slo.target * 100}%\n"
            docs_content += f"**Description:** {slo.description}\n"
            docs_content += "**SLI Query:**\n```promql\n" + slo.sli_query.strip() + "\n```\n\n"
            docs_content += "---\n\n"

    output_path.write_text(docs_content)
    print(f"✅ SLO Documentation generated at {output_path}")

def generate_runbooks_markdown(rca_analyzer: KnowledgeGraphRCA, output_path: Path):
    """Generate runbook documentation from the knowledge graph"""
    runbooks = rca_analyzer.export_runbook_documentation()
    docs_content = "# Automated Runbooks\n\n"
    docs_content += "*Generated from the AI Knowledge Graph. The AI uses these steps for RCA.*\n\n"
    docs_content += f"*Last Updated: {datetime.now().isoformat()}*\n\n"

    for issue, resolutions in runbooks.items():
        docs_content += f"## {issue}\n"
        for res in resolutions:
            docs_content += f"### {res['resolution']} (Confidence: {res['confidence']:.2f})\n"
            docs_content += f"{res['description']}\n\n"
            docs_content += "**Steps:**\n"
            for step in res['steps']:
                docs_content += f"1. {step}\n"
            docs_content += "\n---\n\n"

    output_path.write_text(docs_content)
    print(f"✅ Runbook documentation generated at {output_path}")

if __name__ == "__main__":
    # Generate SLO docs
    generate_slo_markdown_docs(SLO_REGISTRY, Path("SLO_DOCUMENTATION.md"))
    
    # Generate runbook docs (requires a KnowledgeGraphRCA instance)
    # kg_rca = KnowledgeGraphRCA()
    # generate_runbooks_markdown(kg_rca, Path("RUNBOOKS.md"))