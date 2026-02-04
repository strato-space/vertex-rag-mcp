# Changelog

## 2026-02-04
### PROBLEM SOLVED
- **09:30** RAG corpora could be created without any text files, causing `mini_rag` to return empty contexts; corpus selection now skips empty corpora.

### FEATURE IMPLEMENTED
- **09:30** Raised the default `mini_rag` result size to `top_k=50` for broader recall.
- **09:30** Each text extraction output now starts with a metadata block (`url`, `name`, `created`, `lastModified`).

### CHANGES
- **09:30** Filtered corpus imports to text-capable MIME types only.
- **09:30** Updated server defaults and README to match new `mini_rag` behavior.
- **09:30** Added repository guidelines in `AGENTS.md`.
