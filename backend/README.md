Product API for the CPG44 dashboard.

```bash
cd ../..   # repo root
export CPG44_RELAY_URL="https://cpg44.nivaspms.com"
export CPG44_RELAY_TOKEN="<same token as the VPS>"
PYTHONPATH=backend/src:. python -m uvicorn cpg44_api.main:app --host 0.0.0.0 --port 8000
```

Docs: http://127.0.0.1:8000/docs

Run `scripts/run_hub.sh` first. It receives the public relay stream and exposes
processed wearable values to this API.
