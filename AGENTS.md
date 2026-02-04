# Repository Guidelines

## Project Overview
- MCP server for Vertex AI RAG and Google Drive content access.
- Tool implementations live in `src/vertex_rag_mcp/vertex_rag_tool.py`.
- Server entrypoint is `src/vertex_rag_mcp/server.py`.
- Run locally with `uv run --directory /home/tools/vertex-rag-mcp --active python -m vertex_rag_mcp.server`.

## Documentation
- Keep `README.md` and `CHANGELOG.md` current with tool behavior and defaults.
- Document protocol-facing changes (tool parameters, defaults, output shapes).

## Testing
- No automated tests are defined. Run targeted manual checks when changing Drive or RAG behavior.

## Versioning
- Bump `pyproject.toml` version on every change.
