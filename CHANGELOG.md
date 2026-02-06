# Changelog

## 2026-02-05
### PROBLEM SOLVED
- None.

### FEATURE IMPLEMENTED
- **13:05** Added `id`, `owner`, and `size` fields to export metadata headers; introduced `VERTEX_RAG_MCP_SKIP_OWNERS` for skipping files by owner.
- **13:12** Skip-owner filter now matches substrings (e.g., `task-syncer` matches service account email).
- **13:15** `read_multiple_text_files` now separates documents with blank lines instead of `---` to avoid delimiter collisions in downstream parsing.

### CHANGES
- **13:05** `list_drive_files` now includes `owner` in outputs and fetches Drive ownership metadata from the API.
- **13:05** Added `output/` to `.gitignore`.
- **13:05** Bumped the package version to `0.0.27` and refreshed `uv.lock`.

## 2026-02-04
### PROBLEM SOLVED
- **09:30** RAG corpora could be created without any text files, causing `mini_rag` to return empty contexts; corpus selection now skips empty corpora.

### FEATURE IMPLEMENTED
- **09:30** Raised the default `mini_rag` result size to `top_k=50` for broader recall.
- **09:30** Each text extraction output now starts with a metadata block (`url`, `name`, `created`, `lastModified`).
- **10:05** Added a Drive markdown export tool that writes outputs to `./output/{drive_id}-vertex-rag.md`.

### CHANGES
- **09:30** Filtered corpus imports to text-capable MIME types only.
- **09:30** Updated server defaults and README to match new `mini_rag` behavior.
- **09:30** Added repository guidelines in `AGENTS.md`.
- **10:05** Exposed the export tool in the MCP server and documented it; bumped the package version and refreshed `uv.lock`.
