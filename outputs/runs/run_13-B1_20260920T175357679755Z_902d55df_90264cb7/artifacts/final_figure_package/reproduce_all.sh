#!/usr/bin/env bash
set -euo pipefail
mkdir -p reproduced
for f in FIG-Q1-01 FIG-Q1-02 FIG-Q1-03 FIG-Q2-01 FIG-Q2-02 FIG-Q2-03 FIG-Q2-04 FIG-Q3-01 FIG-Q3-02 FIG-Q3-03 FIG-Q3-04 FIG-Q4-01 FIG-Q4-02 FIG-Q4-03 FIG-Q4-04; do
  python scripts/step13b1_final_figure_renderer.py --figure-id "$f" --output-dir reproduced --blueprint-dir blueprint --source-dir source_data --dpi 450
done
