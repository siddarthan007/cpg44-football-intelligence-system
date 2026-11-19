"""Regenerate the active-source inventory without reading credentials or datasets."""

import ast
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def main():
    sources = []
    for source_root in (ROOT / "soccer_analytics", ROOT / "backend/src", ROOT / "scripts", ROOT / "tests"):
        for path in sorted(source_root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8-sig")
            try:
                tree = ast.parse(text)
            except SyntaxError as error:
                sources.append({"path": path.relative_to(ROOT).as_posix(), "parse_error": str(error)})
                continue
            symbols, imports, endpoints = [], [], []
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append({"name": node.name, "line": node.lineno, "kind": type(node).__name__})
                    for dec in getattr(node, "decorator_list", []):
                        if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.args \
                                and isinstance(dec.args[0], ast.Constant) and dec.func.attr in {
                                    "get", "post", "put", "delete", "patch", "websocket"}:
                            endpoints.append({"method": dec.func.attr, "path": dec.args[0].value,
                                              "handler": node.name, "line": node.lineno})
                elif isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append("." * node.level + (node.module or "") + ":" + ",".join(a.name for a in node.names))
            sources.append({"path": path.relative_to(ROOT).as_posix(), "language": "python",
                            "sha256": hashlib.sha256(text.encode()).hexdigest(), "lines": len(text.splitlines()),
                            "imports": sorted(set(imports)), "symbols": sorted(symbols, key=lambda s: s["line"]),
                            "endpoints": endpoints})
    for path in sorted((ROOT / "frontend/src").rglob("*")):
        if path.suffix not in {".ts", ".tsx", ".css"}:
            continue
        text = path.read_text(encoding="utf-8")
        sources.append({"path": path.relative_to(ROOT).as_posix(), "language": path.suffix[1:],
                        "sha256": hashlib.sha256(text.encode()).hexdigest(), "lines": len(text.splitlines()),
                        "imports": re.findall(r"from\s+['\"]([^'\"]+)['\"]", text),
                        "routes": re.findall(r'<Route\s+path=[\'\"]([^\'\"]+)', text)})
    result = {"scope": "Active source, scripts, tests and frontend. Reference trees, model binaries, datasets and report artifacts are excluded.",
              "root_directories": sorted(p.name for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith(".")),
              "source_files": sources}
    out = ROOT / "docs/research/2026-11-11/project-map.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Mapped {len(sources)} source files, {sum(len(s.get('endpoints', [])) for s in sources)} API routes: {out}")


if __name__ == "__main__":
    main()
