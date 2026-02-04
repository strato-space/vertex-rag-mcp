# vertex-rag-mcp

MCP server exposing Vertex AI RAG mini_rag.

## Tools
- `mini_rag(query, drive_id, top_k=50)` — query Vertex RAG against a Drive folder.
- `list_drive_files(drive_id, recursive=True, include_folders=False, limit=0)` — list Drive files in a folder with sizes and timestamps.
- `read_multiple_text_files(drive_id, file_ids=None, include_types=None, all_files=False, recursive=True, output_format="markdown", max_files=0, max_total_chars=0, max_file_chars=0, max_pages=0, max_table_rows=50, max_table_cols=12, chunk_size=0)` — extract text from documents and return Markdown/text. CSV is rendered as a Markdown table (limited). Each file starts with a YAML-style metadata block (`url`, `name`, `created`, `lastModified`).
- `read_multiple_md_files(drive_id, file_ids=None, include_types=None, all_files=False, recursive=True, convert_pdf_to_gdoc=True, cleanup_converted=True, strip_images=True, max_files=0, max_total_chars=0, max_file_chars=0, max_pages=0, max_table_rows=50, max_table_cols=12, chunk_size=0)` — extract text as Markdown, preferring Google Docs export. PDFs can be converted to Docs first. `strip_images` removes embedded data:image blocks.
- `read_multiple_binary_files(drive_id, file_ids=None, include_types=None, all_files=False, recursive=True, tool_result="resource_link", response_format="url", max_files=0, max_total_bytes=0, max_file_bytes=0, chunk_size=0)` — list Drive documents **with content** for indexing (defaults exclude media). `tool_result` controls content[] blocks (`resource_link` vs `resource`), `response_format` controls structuredContent (`url` vs `b64_json`). Limits are optional; use 0 to disable.
- `read_single_file_raw(file_id, export_mime=None, chunk_size=0)` — return **raw base64** content for a single file (content only, no structuredContent).
- `incremental_update_corpus(drive_id, delete_removed=False, dry_run=False)` — update a Drive-backed corpus by file modified times.
- `full_refresh_corpus(drive_id, delete_old=True)` — rebuild a corpus from scratch and optionally delete previous versions.

## Default indexable MIME types (no media)
- `application/pdf`
- `application/rtf`
- `application/vnd.google-apps.document`
- `application/vnd.google-apps.presentation`
- `application/vnd.google-apps.spreadsheet`
- `application/vnd.oasis.opendocument.text`
- `application/vnd.openxmlformats-officedocument.presentationml.presentation`
- `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`
- `application/vnd.openxmlformats-officedocument.wordprocessingml.document`
- `text/csv`
- `text/html`
- `text/markdown`
- `text/plain`

## Setup
- Create `fastagent.secrets.yaml` in the repo root (see `fastagent.secrets.yaml.example`).
- Ensure Google credentials are available (for example `GOOGLE_APPLICATION_CREDENTIALS`).
- Grant the service account access to the Google Drive folder you query.
- `read_multiple_md_files` with `convert_pdf_to_gdoc=True` requires Drive write access to create/delete temporary Google Docs.
- Optional: set `VERTEX_RAG_MCP_CONVERT_SHARED_DRIVE_ID` to a Shared Drive ID to store temporary conversion Docs there (recommended for service accounts).

## Run
```bash
python -m vertex_rag_mcp.server
```

If installed as a package:
```bash
vertex-rag-mcp
```
