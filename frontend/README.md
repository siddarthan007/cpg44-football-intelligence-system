# CPG44 dashboard

React + Vite UI for the local match workspace, tactical pitch, measured
wearable data and live WebSocket/JPEG updates. The default design is light,
responsive and uses the repository's own design tokens.

```bash
# API
PYTHONPATH=backend/src:. python -m uvicorn cpg44_api.main:app --port 8000

# UI
npm install
npm run dev
```

Open http://127.0.0.1:5173/

The browser connects only to the local API and its WebSockets. Relay secrets
remain in the backend and firmware provisioning path.
