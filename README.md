# vertex-rag-mcp

MCP server exposing Vertex AI RAG mini_rag.

## Setup
- Create `fastagent.secrets.yaml` in the repo root (see `fastagent.secrets.yaml.example`).
- Ensure Google credentials are available (for example `GOOGLE_APPLICATION_CREDENTIALS`).
- Grant the service account access to the Google Drive folder you query.

## Run
```bash
python -m vertex_rag_mcp.server
```

If installed as a package:
```bash
vertex-rag-mcp
```
