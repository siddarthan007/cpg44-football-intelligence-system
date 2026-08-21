#!/usr/bin/env bash
set -euo pipefail

SUBMISSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JAR_PATH="${PLANTUML_JAR:-/tmp/plantuml.jar}"

if [[ ! -f "$JAR_PATH" ]]; then
  echo "PlantUML jar was not found at $JAR_PATH"
  echo "Download it from https://github.com/plantuml/plantuml/releases/latest/download/plantuml.jar"
  echo "or set PLANTUML_JAR to its path."
  exit 1
fi

cd "$SUBMISSION_DIR"
java -jar "$JAR_PATH" -tpng -o ../figures diagrams/*.puml
echo "Rendered PlantUML diagrams into figures/."
