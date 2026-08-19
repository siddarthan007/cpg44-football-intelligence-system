Product API for the CPG44 dashboard.

```bash
cd ../..   # repo root
PYTHONPATH=backend/src:. python -m uvicorn cpg44_api.main:app --host 0.0.0.0 --port 8000
```

Docs: http://127.0.0.1:8000/docs
