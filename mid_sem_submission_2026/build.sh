#!/usr/bin/env bash
set -euo pipefail

SUBMISSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SUBMISSION_DIR"

mkdir -p build/report build/presentation
conda run -n soccer tectonic --keep-logs --outdir build/report report/main.tex
conda run -n soccer tectonic --keep-logs --outdir build/presentation presentation/presentation.tex

cp build/report/main.pdf build/CPG44_Mid_Semester_Report_2026.pdf
cp build/presentation/presentation.pdf build/CPG44_Mid_Semester_Presentation_2026.pdf

echo "Report: build/CPG44_Mid_Semester_Report_2026.pdf"
echo "Slides: build/CPG44_Mid_Semester_Presentation_2026.pdf"
