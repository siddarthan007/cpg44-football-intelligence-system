# CPG44 dashboard

React + Vite UI inspired by `Downloads/Capstone/Capstone/frontend`
(dark pitch-side layout, match workspace, tactical pitch, live WebSocket).

```bash
# API
PYTHONPATH=backend/src:. python -m uvicorn cpg44_api.main:app --port 8000

# UI
npm install
npm run dev
```

Open http://127.0.0.1:5173/
