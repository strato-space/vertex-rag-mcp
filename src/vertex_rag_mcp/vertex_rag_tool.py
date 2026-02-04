from __future__ import annotations

import base64
import csv
import datetime
import io
import logging
import os
import re
import time
from pathlib import Path
from collections import deque

import google.auth
import vertexai
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ResourceLink,
    TextContent,
    TextResourceContents,
)
from vertexai import rag

from fast_agent.config import get_settings

# RAG quickstart: Required roles, Prepare your Google Cloud console, Run Vertex AI RAG Engine
# https://docs.cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/rag-quickstart
#
# Vertex AI RAG Engine overview: Overview, Supported regions, ...
# https://docs.cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/rag-overview
#
# Install the Vertex AI SDK for Python
# https://docs.cloud.google.com/vertex-ai/docs/start/install-sdk
#
# Admin console
# https://console.cloud.google.com/vertex-ai/rag

CONFIG_PATH = "fastagent.secrets.yaml"
EMBEDDING_MODEL = "text-embedding-005"
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
SCOPES_READWRITE = ["https://www.googleapis.com/auth/drive"]
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
DEFAULT_MAX_FILES = 100
DEFAULT_MAX_TOTAL_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_FILE_BYTES = 10 * 1024 * 1024
DEFAULT_INDEXABLE_MIME_TYPES = [
    "application/pdf",
    "application/rtf",
    "application/vnd.google-apps.document",
    "application/vnd.google-apps.presentation",
    "application/vnd.google-apps.spreadsheet",
    "application/vnd.oasis.opendocument.text",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/csv",
    "text/html",
    "text/markdown",
    "text/plain",
]
MARKDOWN_EXPORTABLE_MIME_TYPES = [
    "application/pdf",
    "application/vnd.google-apps.document",
    "application/vnd.google-apps.presentation",
    "application/vnd.google-apps.spreadsheet",
    "text/csv",
    "text/html",
    "text/markdown",
    "text/plain",
    "application/json",
    "application/xml",
    "application/xhtml+xml",
]
GOOGLE_EXPORT_MIME_TYPES = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
}
CONVERT_APP_PROP_KEY = "vertex_rag_mcp_source_id"

_vertex_initialized = False
logger = logging.getLogger(__name__)


def _ensure_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    level_name = os.getenv("VERTEX_RAG_MCP_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _load_vertex_settings() -> tuple[str, str]:
    settings = get_settings(CONFIG_PATH)
    vertex_cfg = getattr(settings.google, "vertex_ai", {}) if settings.google else {}
    project_id = vertex_cfg.get("project_id")
    location = vertex_cfg.get("location")
    if not project_id or not location:
        raise ValueError(
            "Missing google.vertex_ai.project_id/location in fastagent.secrets.yaml"
        )
    return project_id, location


def _ensure_vertexai_init(project_id: str, location: str) -> None:
    global _vertex_initialized
    if not _vertex_initialized:
        vertexai.init(project=project_id, location=location)
        _vertex_initialized = True


def _drive_folder_name(folder_id: str) -> str:
    credentials, _ = google.auth.default(scopes=SCOPES)
    drive_service = build("drive", "v3", credentials=credentials)
    payload = (
        drive_service.files()
        .get(
            fileId=folder_id,
            fields="id,name,mimeType",
            supportsAllDrives=True,
        )
        .execute()
    )
    return payload["name"]


def _ensure_vertex_ready() -> None:
    project_id, location = _load_vertex_settings()
    _ensure_vertexai_init(project_id, location)


def _timestamp_to_utc(ts: object) -> datetime.datetime | None:
    seconds = getattr(ts, "seconds", None)
    if seconds is None:
        return None
    nanos = getattr(ts, "nanos", 0) or 0
    return datetime.datetime.fromtimestamp(
        float(seconds) + (float(nanos) / 1_000_000_000.0), tz=datetime.UTC
    )


def _parse_rfc3339_utc(value: str | None) -> datetime.datetime | None:
    if not value:
        return None
    try:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        dt = datetime.datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.UTC)
        return dt.astimezone(datetime.UTC)
    except ValueError:
        return None


def _find_latest_corpus_for_drive(drive_id: str) -> object | None:
    candidates: list[object] = []
    for corpus in rag.list_corpora():
        display_name = getattr(corpus, "display_name", None)
        if display_name and drive_id in display_name:
            candidates.append(corpus)

    if not candidates:
        return None

    def corpus_sort_key(c: object) -> tuple[float, float]:
        updated = _timestamp_to_utc(getattr(c, "update_time", None))
        created = _timestamp_to_utc(getattr(c, "create_time", None))
        return (
            updated.timestamp() if updated else 0.0,
            created.timestamp() if created else 0.0,
        )

    return max(candidates, key=corpus_sort_key)


def _parse_drive_file_id(url: str) -> str | None:
    if not url:
        return None
    match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    return None


def _parse_drive_folder_id(url: str) -> str | None:
    if not url:
        return None
    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    return None


def _corpus_has_files(corpus_name: str) -> bool:
    try:
        return any(True for _ in rag.list_files(corpus_name))
    except Exception as exc:
        logger.warning("corpus_has_files failed corpus=%s error=%s", corpus_name, exc)
        return False


def _import_paths(corpus_name: str, paths: list[str]) -> object:
    drive_paths = [p for p in paths if "drive.google.com" in p]
    non_drive_paths = [p for p in paths if p not in drive_paths]
    import_paths: list[str] = list(non_drive_paths)

    if drive_paths:
        credentials, _ = google.auth.default(scopes=SCOPES)
        drive_service = build("drive", "v3", credentials=credentials)

        for path in drive_paths:
            folder_id = _parse_drive_folder_id(path)
            if folder_id:
                items = list_drive_files(
                    folder_id, recursive=True, include_folders=False
                )
                for item in items:
                    if item.get("mime_type") in MARKDOWN_EXPORTABLE_MIME_TYPES:
                        fid = item.get("id")
                        if fid:
                            import_paths.append(
                                f"https://drive.google.com/file/d/{fid}/view"
                            )
                continue

            file_id = _parse_drive_file_id(path)
            if file_id:
                try:
                    meta = _get_drive_metadata(drive_service, file_id)
                    if meta.get("mimeType") in MARKDOWN_EXPORTABLE_MIME_TYPES:
                        import_paths.append(path)
                except Exception as exc:
                    logger.warning(
                        "import_paths failed to fetch metadata file_id=%s error=%s",
                        file_id,
                        exc,
                    )
                continue

    # De-duplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for p in import_paths:
        if p in seen:
            continue
        seen.add(p)
        deduped.append(p)

    if not deduped:
        raise ValueError("No importable text files found for corpus import.")

    return rag.import_files(
        corpus_name,
        deduped,
        transformation_config=rag.TransformationConfig(
            chunking_config=rag.ChunkingConfig(
                chunk_size=512,
                chunk_overlap=100,
            ),
        ),
        max_embedding_requests_per_min=1000,
    )


def list_drive_files(
    drive_id: str,
    *,
    recursive: bool = True,
    include_folders: bool = False,
    limit: int | None = None,
) -> list[dict]:
    """List Google Drive files under a folder ID.

    Args:
        drive_id: Google Drive folder ID to list.
        recursive: If True, traverses nested folders.
        include_folders: If True, include folder entries in the result.
        limit: Optional max number of returned entries (None/unset = no limit).

    Returns:
        List of dicts with file metadata: id, name, path, size, created_time, modified_time.
        size is in bytes when available (Google Docs may not have a size).
    """
    if not drive_id:
        raise ValueError("drive_id must be a non-empty Google Drive ID.")

    credentials, _ = google.auth.default(scopes=SCOPES)
    drive_service = build("drive", "v3", credentials=credentials)

    def list_children(folder_id: str) -> list[dict]:
        query = f"'{folder_id}' in parents and trashed = false"
        fields = "nextPageToken, files(id,name,mimeType,size,createdTime,modifiedTime,webViewLink)"
        items: list[dict] = []
        page_token: str | None = None
        while True:
            resp = (
                drive_service.files()
                .list(
                    q=query,
                    pageSize=1000,
                    pageToken=page_token,
                    fields=fields,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            items.extend(resp.get("files", []))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        return items

    results: list[dict] = []
    visited_folders: set[str] = set()
    queue: deque[tuple[str, str]] = deque([(drive_id, "")])

    while queue:
        current_folder_id, current_path = queue.popleft()
        if current_folder_id in visited_folders:
            continue
        visited_folders.add(current_folder_id)

        for item in list_children(current_folder_id):
            name = item.get("name") or ""
            item_path = f"{current_path}/{name}" if current_path else name
            mime_type = item.get("mimeType")
            is_folder = mime_type == FOLDER_MIME_TYPE

            size_raw = item.get("size")
            size: int | None = None
            if size_raw is not None:
                try:
                    size = int(size_raw)
                except (TypeError, ValueError):
                    size = None

            row = {
                "id": item.get("id"),
                "name": name,
                "path": item_path,
                "mime_type": mime_type,
                "size": size,
                "created_time": item.get("createdTime"),
                "modified_time": item.get("modifiedTime"),
                "web_view_link": item.get("webViewLink"),
            }

            if is_folder:
                if include_folders:
                    results.append(row)
                if recursive and item.get("id"):
                    queue.append((item["id"], item_path))
            else:
                results.append(row)

            if limit and len(results) >= limit:
                return results

    return results


def _download_drive_bytes(drive_service, file_id: str, *, chunk_size: int | None = None) -> bytes:
    request = drive_service.files().get_media(fileId=file_id, supportsAllDrives=True)
    buffer = io.BytesIO()
    if chunk_size:
        downloader = MediaIoBaseDownload(buffer, request, chunksize=chunk_size)
    else:
        downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buffer.getvalue()


def _export_drive_bytes(
    drive_service, file_id: str, export_mime: str, *, chunk_size: int | None = None
) -> bytes:
    request = drive_service.files().export_media(fileId=file_id, mimeType=export_mime)
    buffer = io.BytesIO()
    if chunk_size:
        downloader = MediaIoBaseDownload(buffer, request, chunksize=chunk_size)
    else:
        downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buffer.getvalue()


def _get_drive_metadata(drive_service, file_id: str) -> dict:
    return (
        drive_service.files()
        .get(
            fileId=file_id,
            fields="id,name,mimeType,size,createdTime,modifiedTime,webViewLink",
            supportsAllDrives=True,
        )
        .execute()
    )


def _bytes_to_text_or_b64(payload: bytes) -> tuple[str, bool]:
    try:
        return payload.decode("utf-8"), False
    except UnicodeDecodeError:
        return base64.b64encode(payload).decode("ascii"), True


def read_multiple_binary_files(
    drive_id: str,
    *,
    file_ids: list[str] | None = None,
    include_types: list[str] | None = None,
    all_files: bool = False,
    recursive: bool = True,
    tool_result: str = "resource_link",
    response_format: str = "url",
    max_files: int | None = None,
    max_total_bytes: int | None = None,
    max_file_bytes: int | None = None,
    chunk_size: int | None = None,
) -> CallToolResult:
    """Return Drive files with content for indexing (no media by default).

    Args:
        drive_id: Google Drive folder ID to list.
        file_ids: Optional list of specific file IDs to include.
        include_types: Optional list of MIME types to include (None = defaults).
        all_files: If True, ignore type filters and return all non-folder files.
        recursive: If True, traverses nested folders.
        max_files: Optional max number of files to download (None = no limit).
        max_total_bytes: Optional max total bytes to download (None = no limit).
        max_file_bytes: Optional max per-file bytes to download (None = no limit).
        chunk_size: Optional download chunk size (bytes).
    """
    if not drive_id:
        raise ValueError("drive_id must be a non-empty Google Drive ID.")

    _ensure_logging()

    if tool_result not in {"resource_link", "resource"}:
        tool_result = "resource_link"

    def normalize_limit(value: int | None) -> int | None:
        if value is None:
            return None
        try:
            value = int(value)
        except (TypeError, ValueError):
            return None
        return None if value <= 0 else value

    max_files = normalize_limit(max_files)
    max_total_bytes = normalize_limit(max_total_bytes)
    max_file_bytes = normalize_limit(max_file_bytes)

    if max_files is None and (all_files or tool_result == "resource" or response_format == "b64_json"):
        max_files = DEFAULT_MAX_FILES
    if max_total_bytes is None and (all_files or tool_result == "resource" or response_format == "b64_json"):
        max_total_bytes = DEFAULT_MAX_TOTAL_BYTES
    if max_file_bytes is None and (all_files or tool_result == "resource" or response_format == "b64_json"):
        max_file_bytes = DEFAULT_MAX_FILE_BYTES

    logger.info(
        "read_multiple_binary_files start drive_id=%s recursive=%s all_files=%s tool_result=%s response_format=%s max_files=%s max_total_bytes=%s max_file_bytes=%s",
        drive_id,
        recursive,
        all_files,
        tool_result,
        response_format,
        max_files,
        max_total_bytes,
        max_file_bytes,
    )

    items = list_drive_files(
        drive_id,
        recursive=recursive,
        include_folders=False,
        limit=max_files if max_files and all_files and not file_ids else None,
    )

    if file_ids:
        id_set = {f for f in file_ids if f}
        items = [i for i in items if i.get("id") in id_set]

    if all_files:
        allowed_items = items
    else:
        allowed_types = (
            include_types if include_types is not None else DEFAULT_INDEXABLE_MIME_TYPES
        )
        allowed = set(allowed_types)
        allowed_items = [i for i in items if i.get("mime_type") in allowed]

    pre_limit_count = len(allowed_items)
    if max_files and pre_limit_count > max_files:
        allowed_items = allowed_items[:max_files]

    credentials, _ = google.auth.default(scopes=SCOPES)
    drive_service = build("drive", "v3", credentials=credentials)

    structured_items: list[dict] = []
    content_blocks: list[ResourceLink | EmbeddedResource | TextContent] = []
    warnings: list[str] = []
    if max_files and pre_limit_count > max_files:
        warnings.append(
            f"Trimmed file list from {pre_limit_count} to max_files {max_files}"
        )
    total_bytes = 0
    downloaded = 0
    skipped = 0
    started_at = time.monotonic()
    for item in allowed_items:
        file_id = item.get("id")
        mime_type = item.get("mime_type") or ""
        web_url = item.get("web_view_link") or (
            f"https://drive.google.com/file/d/{file_id}/view" if file_id else None
        )
        if not file_id:
            continue

        size = item.get("size")
        if max_file_bytes and size and size > max_file_bytes:
            skipped += 1
            warnings.append(
                f"Skipped {item.get('name') or file_id}: size {size} > max_file_bytes {max_file_bytes}"
            )
            structured_items.append({**item, "content": None, "content_error": "file_too_large"})
            continue

        if max_total_bytes and size and (total_bytes + size) > max_total_bytes:
            warnings.append(
                f"Stopped before {item.get('name') or file_id}: total {total_bytes} + size {size} > max_total_bytes {max_total_bytes}"
            )
            break

        file_started = time.monotonic()
        try:
            if mime_type in GOOGLE_EXPORT_MIME_TYPES:
                export_mime = GOOGLE_EXPORT_MIME_TYPES[mime_type]
                raw = _export_drive_bytes(
                    drive_service, file_id, export_mime, chunk_size=chunk_size
                )
            else:
                raw = _download_drive_bytes(
                    drive_service, file_id, chunk_size=chunk_size
                )
        except Exception as exc:
            skipped += 1
            logger.warning("read_multiple_binary_files download failed file_id=%s error=%s", file_id, exc)
            structured_items.append({**item, "content": None, "content_error": str(exc)})
            continue
        finally:
            logger.info(
                "read_multiple_binary_files download finished file_id=%s elapsed=%.3fs",
                file_id,
                time.monotonic() - file_started,
            )

        content, is_base64 = _bytes_to_text_or_b64(raw)
        raw_size = len(raw)
        total_bytes += raw_size
        downloaded += 1
        if max_file_bytes and raw_size > max_file_bytes:
            skipped += 1
            warnings.append(
                f"Skipped {item.get('name') or file_id}: downloaded size {raw_size} > max_file_bytes {max_file_bytes}"
            )
            structured_items.append({**item, "content": None, "content_error": "file_too_large"})
            if max_total_bytes and total_bytes >= max_total_bytes:
                warnings.append(
                    f"Stopped after {item.get('name') or file_id}: total_bytes {total_bytes} >= max_total_bytes {max_total_bytes}"
                )
                break
            continue

        structured_item = {
            **item,
            "web_url": web_url,
            "content": content,
            "content_is_base64": is_base64,
        }
        structured_items.append(structured_item)

        if max_total_bytes and total_bytes >= max_total_bytes:
            warnings.append(
                f"Stopped after {item.get('name') or file_id}: total_bytes {total_bytes} >= max_total_bytes {max_total_bytes}"
            )
            break

        if tool_result == "resource":
            if is_base64:
                content_blocks.append(
                    EmbeddedResource(
                        type="resource",
                        resource=BlobResourceContents(
                            uri=web_url or f"https://drive.google.com/file/d/{file_id}/view",
                            mimeType=mime_type or "application/octet-stream",
                            blob=content,
                        ),
                    )
                )
            else:
                content_blocks.append(
                    EmbeddedResource(
                        type="resource",
                        resource=TextResourceContents(
                            uri=web_url or f"https://drive.google.com/file/d/{file_id}/view",
                            mimeType=mime_type or "text/plain",
                            text=content,
                        ),
                    )
                )
        else:
            if web_url:
                content_blocks.append(
                    ResourceLink(
                        type="resource_link",
                        name=item.get("name") or file_id,
                        uri=web_url,
                        mimeType=mime_type,
                        size=item.get("size"),
                        description=item.get("path"),
                    )
                )

    if response_format == "b64_json":
        structured_content = {
            "data": [
                {
                    "b64_json": i.get("content"),
                    "content_is_base64": i.get("content_is_base64"),
                }
                for i in structured_items
            ]
        }
    else:
        structured_content = {
            "data": [{"url": i.get("web_view_link") or i.get("web_url")} for i in structured_items]
        }

    if warnings:
        structured_content["warnings"] = warnings
    structured_content["stats"] = {
        "downloaded": downloaded,
        "skipped": skipped,
        "total_bytes": total_bytes,
        "elapsed_sec": round(time.monotonic() - started_at, 3),
    }

    if not content_blocks:
        content_blocks.append(TextContent(type="text", text="No files matched the request."))

    logger.info(
        "read_multiple_binary_files done drive_id=%s downloaded=%s skipped=%s total_bytes=%s elapsed=%.3fs",
        drive_id,
        downloaded,
        skipped,
        total_bytes,
        time.monotonic() - started_at,
    )

    return CallToolResult(content=content_blocks, structuredContent=structured_content)


def _decode_text_bytes(payload: bytes) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload.decode("utf-8", errors="replace")


def _csv_to_markdown(text: str, *, max_rows: int = 50, max_cols: int = 12) -> str:
    reader = csv.reader(io.StringIO(text))
    rows = []
    for idx, row in enumerate(reader):
        if idx >= max_rows:
            break
        rows.append(row[:max_cols])
    if not rows:
        return text
    header = rows[0]
    body = rows[1:]
    # Escape pipes
    def esc(cell: str) -> str:
        return cell.replace("|", "\\|")

    header_line = "| " + " | ".join(esc(c) for c in header) + " |"
    sep_line = "| " + " | ".join("---" for _ in header) + " |"
    body_lines = ["| " + " | ".join(esc(c) for c in row) + " |" for row in body]
    return "\n".join([header_line, sep_line] + body_lines)


def _extract_pdf_text(raw: bytes, *, max_pages: int | None = None) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"pypdf not available: {exc}") from exc

    reader = PdfReader(io.BytesIO(raw))
    pages = reader.pages
    if max_pages and max_pages > 0:
        pages = pages[:max_pages]
    extracted = []
    for page in pages:
        extracted.append(page.extract_text() or "")
    return "\n".join(extracted).strip()


def _export_doc_markdown(drive_service, file_id: str, *, chunk_size: int | None = None) -> str:
    """Export a Google Doc as markdown when supported; fallback to plain text."""
    try:
        raw = _export_drive_bytes(
            drive_service, file_id, "text/markdown", chunk_size=chunk_size
        )
        return _decode_text_bytes(raw)
    except HttpError:
        raw = _export_drive_bytes(
            drive_service, file_id, "text/plain", chunk_size=chunk_size
        )
        return _decode_text_bytes(raw)


def _copy_to_google_doc(
    drive_service, file_id: str, *, name: str | None = None
) -> str:
    """Create a temporary Google Doc copy for conversion (returns new file id)."""
    shared_drive_id = os.getenv("VERTEX_RAG_MCP_CONVERT_SHARED_DRIVE_ID")
    body = {
        "mimeType": "application/vnd.google-apps.document",
        "name": name or f"{file_id} (md-convert)",
    }
    if shared_drive_id:
        body["parents"] = [shared_drive_id]
        body["appProperties"] = {CONVERT_APP_PROP_KEY: file_id}
    payload = (
        drive_service.files()
        .copy(
            fileId=file_id,
            body=body,
            fields="id",
            supportsAllDrives=True,
        )
        .execute()
    )
    return payload["id"]


def _delete_drive_file(drive_service, file_id: str) -> None:
    try:
        drive_service.files().delete(fileId=file_id, supportsAllDrives=True).execute()
    except Exception as exc:
        logger.warning("failed to delete temp file_id=%s error=%s", file_id, exc)


def _cleanup_legacy_conversions(drive_service, shared_drive_id: str) -> None:
    query = (
        "mimeType='application/vnd.google-apps.document' and trashed=false "
        f"and '{shared_drive_id}' in parents"
    )
    page_token: str | None = None
    while True:
        resp = (
            drive_service.files()
            .list(
                q=query,
                fields="nextPageToken, files(id, name, appProperties)",
                pageToken=page_token,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
                corpora="drive",
                driveId=shared_drive_id,
                pageSize=200,
            )
            .execute()
        )
        for item in resp.get("files", []):
            app_props = item.get("appProperties") or {}
            if CONVERT_APP_PROP_KEY not in app_props:
                _delete_drive_file(drive_service, item.get("id"))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break


def _find_recent_converted_doc(
    drive_service,
    *,
    shared_drive_id: str,
    source_file_id: str,
    source_modified_time: str | None,
) -> str | None:
    query = (
        "mimeType='application/vnd.google-apps.document' and trashed=false "
        f"and '{shared_drive_id}' in parents "
        f"and appProperties has {{ key='{CONVERT_APP_PROP_KEY}' and value='{source_file_id}' }}"
    )
    resp = (
        drive_service.files()
        .list(
            q=query,
            fields="files(id, modifiedTime, appProperties)",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            corpora="drive",
            driveId=shared_drive_id,
            pageSize=1,
        )
        .execute()
    )
    files = resp.get("files", [])
    if not files:
        return None
    converted = files[0]
    src_dt = _parse_rfc3339_utc(source_modified_time)
    conv_dt = _parse_rfc3339_utc(converted.get("modifiedTime"))
    if src_dt and conv_dt and conv_dt >= src_dt:
        return converted.get("id")
    return None


def _strip_markdown_images(text: str) -> tuple[str, int]:
    cleaned = []
    removed = 0
    for line in text.splitlines():
        if "data:image/" in line:
            removed += 1
            continue
        if line.lstrip().startswith("[image") and "data:image/" in line:
            removed += 1
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip(), removed


def _count_markdown_image_refs(text: str) -> int:
    count = 0
    for line in text.splitlines():
        if "![](" in line or "![" in line:
            count += line.count("![](") + line.count("![")
    return count


def read_multiple_text_files(
    drive_id: str,
    *,
    file_ids: list[str] | None = None,
    include_types: list[str] | None = None,
    all_files: bool = False,
    recursive: bool = True,
    output_format: str = "markdown",
    strip_images: bool = False,
    convert_pdf_to_gdoc: bool = False,
    cleanup_converted: bool = True,
    max_files: int | None = None,
    max_total_chars: int | None = None,
    max_file_chars: int | None = None,
    max_pages: int | None = None,
    max_table_rows: int = 50,
    max_table_cols: int = 12,
    chunk_size: int | None = None,
) -> CallToolResult:
    """Extract text from Drive documents and return as Markdown (default)."""
    if not drive_id:
        raise ValueError("drive_id must be a non-empty Google Drive ID.")

    _ensure_logging()
    output_format = (output_format or "markdown").lower()
    if output_format not in {"markdown", "md", "text", "plain"}:
        output_format = "markdown"

    def normalize_limit(value: int | None) -> int | None:
        if value is None:
            return None
        try:
            value = int(value)
        except (TypeError, ValueError):
            return None
        return None if value <= 0 else value

    max_files = normalize_limit(max_files)
    max_total_chars = normalize_limit(max_total_chars)
    max_file_chars = normalize_limit(max_file_chars)
    max_pages = normalize_limit(max_pages)
    max_table_rows = int(max_table_rows) if max_table_rows and max_table_rows > 0 else 50
    max_table_cols = int(max_table_cols) if max_table_cols and max_table_cols > 0 else 12

    logger.info(
        "read_multiple_text_files start drive_id=%s recursive=%s all_files=%s output_format=%s max_files=%s max_total_chars=%s max_file_chars=%s max_pages=%s",
        drive_id,
        recursive,
        all_files,
        output_format,
        max_files,
        max_total_chars,
        max_file_chars,
        max_pages,
    )

    items = list_drive_files(
        drive_id,
        recursive=recursive,
        include_folders=False,
        limit=max_files if max_files and all_files and not file_ids else None,
    )

    if file_ids:
        id_set = {f for f in file_ids if f}
        items = [i for i in items if i.get("id") in id_set]

    if all_files:
        allowed_items = items
    else:
        allowed_types = (
            include_types if include_types is not None else DEFAULT_INDEXABLE_MIME_TYPES
        )
        allowed = set(allowed_types)
        allowed_items = [i for i in items if i.get("mime_type") in allowed]

    if max_files and len(allowed_items) > max_files:
        allowed_items = allowed_items[:max_files]

    credentials, _ = google.auth.default(scopes=SCOPES)
    drive_service = build("drive", "v3", credentials=credentials)
    drive_service_rw = None
    shared_drive_id = os.getenv("VERTEX_RAG_MCP_CONVERT_SHARED_DRIVE_ID")
    legacy_cleanup_done = False

    total_chars = 0
    chunks = []
    for item in allowed_items:
        file_id = item.get("id")
        mime_type = item.get("mime_type") or ""
        path = item.get("path") or item.get("name") or file_id or "unknown"
        if not file_id:
            continue
        if output_format in {"markdown", "md"} and mime_type.startswith(
            ("image/", "video/", "audio/")
        ):
            logger.info(
                "read_multiple_text_files skip media file_id=%s mime_type=%s",
                file_id,
                mime_type,
            )
            continue

        file_started = time.monotonic()
        converted_doc_id: str | None = None
        try:
            if mime_type in GOOGLE_EXPORT_MIME_TYPES:
                export_mime = GOOGLE_EXPORT_MIME_TYPES[mime_type]
                if (
                    mime_type == "application/vnd.google-apps.document"
                    and output_format in {"markdown", "md"}
                ):
                    text = _export_doc_markdown(
                        drive_service, file_id, chunk_size=chunk_size
                    )
                else:
                    export_mime = GOOGLE_EXPORT_MIME_TYPES[mime_type]
                    raw = _export_drive_bytes(
                        drive_service, file_id, export_mime, chunk_size=chunk_size
                    )
                    text = _decode_text_bytes(raw)
                if export_mime == "text/csv" and output_format in {"markdown", "md"}:
                    text = _csv_to_markdown(
                        text, max_rows=max_table_rows, max_cols=max_table_cols
                    )
            elif mime_type == "application/pdf":
                if convert_pdf_to_gdoc and output_format in {"markdown", "md"}:
                    try:
                        if drive_service_rw is None:
                            credentials_rw, _ = google.auth.default(
                                scopes=SCOPES_READWRITE
                            )
                            drive_service_rw = build(
                                "drive", "v3", credentials=credentials_rw
                            )
                        if shared_drive_id and not legacy_cleanup_done:
                            _cleanup_legacy_conversions(
                                drive_service_rw, shared_drive_id
                            )
                            legacy_cleanup_done = True
                        converted_doc_id = None
                        if shared_drive_id:
                            converted_doc_id = _find_recent_converted_doc(
                                drive_service_rw,
                                shared_drive_id=shared_drive_id,
                                source_file_id=file_id,
                                source_modified_time=item.get("modified_time"),
                            )
                        if converted_doc_id is None:
                            converted_doc_id = _copy_to_google_doc(
                                drive_service_rw, file_id, name=item.get("name")
                            )
                            logger.info(
                                "read_multiple_text_files created_conversion file_id=%s converted_id=%s",
                                file_id,
                                converted_doc_id,
                            )
                            if shared_drive_id:
                                drive_service_rw.files().update(
                                    fileId=converted_doc_id,
                                    body={
                                        "appProperties": {
                                            CONVERT_APP_PROP_KEY: file_id
                                        }
                                    },
                                    supportsAllDrives=True,
                                ).execute()
                        else:
                            logger.info(
                                "read_multiple_text_files reused_conversion file_id=%s converted_id=%s",
                                file_id,
                                converted_doc_id,
                            )
                        text = _export_doc_markdown(
                            drive_service_rw, converted_doc_id, chunk_size=chunk_size
                        )
                    except Exception as exc:
                        logger.warning(
                            "read_multiple_text_files pdf->gdoc failed file_id=%s error=%s",
                            file_id,
                            exc,
                        )
                        raw = _download_drive_bytes(
                            drive_service, file_id, chunk_size=chunk_size
                        )
                        text = _extract_pdf_text(raw, max_pages=max_pages)
                else:
                    raw = _download_drive_bytes(
                        drive_service, file_id, chunk_size=chunk_size
                    )
                    text = _extract_pdf_text(raw, max_pages=max_pages)
            elif mime_type.startswith("text/") or mime_type in {
                "application/json",
                "application/xml",
                "application/xhtml+xml",
                "text/csv",
                "text/markdown",
            }:
                raw = _download_drive_bytes(drive_service, file_id, chunk_size=chunk_size)
                text = _decode_text_bytes(raw)
                if mime_type == "text/csv" and output_format in {"markdown", "md"}:
                    text = _csv_to_markdown(
                        text, max_rows=max_table_rows, max_cols=max_table_cols
                    )
            else:
                text = f"[unsupported mime for text extraction: {mime_type}]"
        except Exception as exc:
            logger.warning("read_multiple_text_files failed file_id=%s error=%s", file_id, exc)
            text = f"[error extracting text: {exc}]"
        finally:
            if converted_doc_id and cleanup_converted and drive_service_rw is not None:
                _delete_drive_file(drive_service_rw, converted_doc_id)
            logger.info(
                "read_multiple_text_files finished file_id=%s elapsed=%.3fs",
                file_id,
                time.monotonic() - file_started,
            )

            if max_file_chars and len(text) > max_file_chars:
                text = text[:max_file_chars] + "\n\n...[truncated]..."
        if strip_images and output_format in {"markdown", "md"}:
            text, removed_images = _strip_markdown_images(text)
            if removed_images:
                logger.info(
                    "read_multiple_text_files stripped_images file_id=%s removed=%s",
                    file_id,
                    removed_images,
                )
        if output_format in {"markdown", "md"}:
            image_refs = _count_markdown_image_refs(text)
            if image_refs:
                logger.info(
                    "read_multiple_text_files image_refs file_id=%s refs=%s",
                    file_id,
                    image_refs,
                )

        web_url = item.get("web_view_link") or (
            f"https://drive.google.com/file/d/{file_id}/view" if file_id else ""
        )
        meta_block = "\n".join(
            [
                "---",
                f"url: {web_url or ''}",
                f"name: {item.get('name') or ''}",
                f"created: {item.get('created_time') or ''}",
                f"lastModified: {item.get('modified_time') or ''}",
                "---",
            ]
        )

        if output_format in {"markdown", "md"}:
            chunk = f"{meta_block}\n\n## {path}\n\n{text.strip()}\n"
        else:
            chunk = f"{meta_block}\n{path}:\n{text.strip()}\n"

        if max_total_chars and (total_chars + len(chunk)) > max_total_chars:
            chunks.append("\n...[truncated: max_total_chars reached]...\n")
            break

        total_chars += len(chunk)
        chunks.append(chunk)

    output_text = "\n---\n".join(chunks).strip() if chunks else "No files matched the request."

    logger.info(
        "read_multiple_text_files done drive_id=%s files=%s total_chars=%s",
        drive_id,
        len(chunks),
        total_chars,
    )

    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        structuredContent={"content": output_text},
    )


def read_multiple_md_files(
    drive_id: str,
    *,
    file_ids: list[str] | None = None,
    include_types: list[str] | None = None,
    all_files: bool = False,
    recursive: bool = True,
    convert_pdf_to_gdoc: bool = True,
    cleanup_converted: bool = True,
    strip_images: bool = True,
    max_files: int | None = None,
    max_total_chars: int | None = None,
    max_file_chars: int | None = None,
    max_pages: int | None = None,
    max_table_rows: int = 50,
    max_table_cols: int = 12,
    chunk_size: int | None = None,
) -> CallToolResult:
    """Extract text from Drive documents and return Markdown (prefer Google Docs export)."""
    return read_multiple_text_files(
        drive_id,
        file_ids=file_ids,
        include_types=include_types or MARKDOWN_EXPORTABLE_MIME_TYPES,
        all_files=all_files,
        recursive=recursive,
        output_format="markdown",
        convert_pdf_to_gdoc=convert_pdf_to_gdoc,
        cleanup_converted=cleanup_converted,
        strip_images=strip_images,
        max_files=max_files,
        max_total_chars=max_total_chars,
        max_file_chars=max_file_chars,
        max_pages=max_pages,
        max_table_rows=max_table_rows,
        max_table_cols=max_table_cols,
        chunk_size=chunk_size,
    )


def export(drive_id: str) -> CallToolResult:
    """Export Drive markdown to ./output/{drive_id}-vertex-rag.md."""
    result = read_multiple_md_files(drive_id)

    output_text = ""
    if isinstance(result.structuredContent, dict):
        output_text = str(result.structuredContent.get("content") or "")
    if not output_text:
        for block in result.content or []:
            if isinstance(block, TextContent) and block.text:
                output_text = block.text
                break
    if not output_text:
        output_text = "No files matched the request."

    out_dir = Path(__file__).resolve().parents[2] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{drive_id}-vertex-rag.md"
    out_path.write_text(output_text, encoding="utf-8")
    uri = out_path.resolve().as_uri()

    return CallToolResult(
        content=[
            ResourceLink(
                type="resource_link",
                uri=uri,
                name=out_path.name,
                mimeType="text/markdown",
            )
        ],
        structuredContent={"path": str(out_path)},
    )


def read_single_file_raw(
    file_id: str,
    *,
    export_mime: str | None = None,
    chunk_size: int | None = None,
) -> CallToolResult:
    """Return raw base64 content for a single Drive file (no structuredContent wrapper)."""
    if not file_id:
        raise ValueError("file_id must be a non-empty Google Drive ID.")

    _ensure_logging()

    credentials, _ = google.auth.default(scopes=SCOPES)
    drive_service = build("drive", "v3", credentials=credentials)
    meta = _get_drive_metadata(drive_service, file_id)
    mime_type = meta.get("mimeType") or "application/octet-stream"
    web_url = meta.get("webViewLink") or f"https://drive.google.com/file/d/{file_id}/view"

    file_started = time.monotonic()
    if mime_type in GOOGLE_EXPORT_MIME_TYPES:
        export_mime_type = export_mime or GOOGLE_EXPORT_MIME_TYPES[mime_type]
        raw = _export_drive_bytes(
            drive_service, file_id, export_mime_type, chunk_size=chunk_size
        )
        effective_mime = export_mime_type
    else:
        raw = _download_drive_bytes(drive_service, file_id, chunk_size=chunk_size)
        effective_mime = mime_type

    b64_payload = base64.b64encode(raw).decode("ascii")
    logger.info(
        "read_single_file_raw file_id=%s bytes=%s elapsed=%.3fs",
        file_id,
        len(raw),
        time.monotonic() - file_started,
    )

    content_blocks: list[EmbeddedResource] = [
        EmbeddedResource(
            type="resource",
            resource=BlobResourceContents(
                uri=web_url,
                mimeType=effective_mime or "application/octet-stream",
                blob=b64_payload,
            ),
        )
    ]
    return CallToolResult(content=content_blocks)


def _create_and_import_corpus(display_name: str, paths: list[str]) -> tuple[rag.RagCorpus, object]:
    embedding_model_config = rag.RagEmbeddingModelConfig(
        vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
            publisher_model=f"publishers/google/models/{EMBEDDING_MODEL}"
        )
    )
    rag_corpus = rag.create_corpus(
        display_name=display_name,
        backend_config=rag.RagVectorDbConfig(
            rag_embedding_model_config=embedding_model_config
        ),
    )
    resp = _import_paths(rag_corpus.name, paths)
    return rag_corpus, resp


def mini_rag(query: str, drive_id: str, top_k: int = 50) -> object:
    """Query Vertex RAG using a Google Drive folder ID and return retrieval results.

    Args:
        query: Natural language query to search against the RAG corpus.
        drive_id: Google Drive folder ID to index/search. The RAG corpus is created
            on first use for this drive_id and reused on subsequent calls.
        top_k: Number of top matches to return.
    """
    _ensure_vertex_ready()
    if not drive_id:
        raise ValueError("drive_id must be a non-empty Google Drive ID.")

    paths = [f"https://drive.google.com/drive/folders/{drive_id}"]
    folder_name = _drive_folder_name(drive_id)
    key = drive_id
    display_name = f"{folder_name} | {key}"

    existing_corpus = _find_latest_corpus_for_drive(drive_id=drive_id)

    if existing_corpus and _corpus_has_files(existing_corpus.name):
        rag_corpus = existing_corpus
    else:
        rag_corpus, _ = _create_and_import_corpus(display_name, paths)

    rag_retrieval_config = rag.RagRetrievalConfig(
        top_k=top_k,
        filter=rag.Filter(vector_distance_threshold=0.5),
    )
    return rag.retrieval_query(
        rag_resources=[
            rag.RagResource(
                rag_corpus=rag_corpus.name,
            )
        ],
        text=query,
        rag_retrieval_config=rag_retrieval_config,
    )


def full_refresh_corpus(drive_id: str, *, delete_old: bool = True) -> dict:
    """Create a fresh corpus for a Drive folder and (optionally) delete old versions.

    The new corpus uses a timestamped display name to avoid collisions.
    """
    _ensure_vertex_ready()
    if not drive_id:
        raise ValueError("drive_id must be a non-empty Google Drive ID.")

    folder_name = _drive_folder_name(drive_id)
    stable_prefix = f"{folder_name} | {drive_id}"
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    display_name = f"{stable_prefix} | {timestamp}"

    old_corpora = [
        c
        for c in rag.list_corpora()
        if getattr(c, "display_name", None) and drive_id in c.display_name
    ]

    paths = [f"https://drive.google.com/drive/folders/{drive_id}"]
    rag_corpus = None
    import_resp = None
    try:
        rag_corpus, resp = _create_and_import_corpus(display_name, paths)
        import_resp = {
            "imported_rag_files_count": getattr(resp, "imported_rag_files_count", None),
            "skipped_rag_files_count": getattr(resp, "skipped_rag_files_count", None),
            "failed_rag_files_count": getattr(resp, "failed_rag_files_count", None),
        }
    except Exception:
        if rag_corpus and getattr(rag_corpus, "name", None):
            try:
                rag.delete_corpus(rag_corpus.name)
            except Exception:
                pass
        raise

    deleted: list[str] = []
    if delete_old:
        for c in old_corpora:
            name = getattr(c, "name", None)
            if not name:
                continue
            try:
                rag.delete_corpus(name)
                deleted.append(name)
            except Exception:
                continue

    return {
        "drive_id": drive_id,
        "folder_name": folder_name,
        "new_corpus_name": getattr(rag_corpus, "name", None),
        "new_corpus_display_name": display_name,
        "deleted_old_corpora": deleted,
        "import_response": import_resp,
    }


def incremental_update_corpus(
    drive_id: str,
    *,
    delete_removed: bool = False,
    dry_run: bool = False,
) -> dict:
    """Incrementally update an existing corpus for a Drive folder.

    Strategy:
    - List Drive files with modifiedTime.
    - List corpus files and map them to Drive file IDs via google_drive_source.resource_ids.
    - Re-import files whose Drive modifiedTime is newer than the corpus file updateTime.
    - Optionally delete corpus files that no longer exist in Drive.
    """
    _ensure_vertex_ready()
    if not drive_id:
        raise ValueError("drive_id must be a non-empty Google Drive ID.")

    folder_name = _drive_folder_name(drive_id)

    corpus = _find_latest_corpus_for_drive(drive_id=drive_id)
    if corpus is None:
        if dry_run:
            return {
                "drive_id": drive_id,
                "folder_name": folder_name,
                "corpus_name": None,
                "plan": {
                    "create_corpus": True,
                    "import_folder": True,
                },
            }
        created, _ = _create_and_import_corpus(
            f"{folder_name} | {drive_id}",
            [f"https://drive.google.com/drive/folders/{drive_id}"],
        )
        corpus_name = created.name
    else:
        corpus_name = corpus.name

    drive_items = list_drive_files(drive_id, recursive=True, include_folders=False)
    drive_by_id: dict[str, dict] = {
        i["id"]: i for i in drive_items if i.get("id")  # type: ignore[index]
    }

    corpus_by_drive_id: dict[str, dict] = {}
    for f in rag.list_files(corpus_name):
        gds = getattr(f, "google_drive_source", None)
        resource_ids = getattr(gds, "resource_ids", None) if gds else None
        if not resource_ids:
            continue
        drive_file_id = resource_ids[0].resource_id
        corpus_by_drive_id[drive_file_id] = {
            "rag_file_name": f.name,
            "update_time": _timestamp_to_utc(getattr(f, "update_time", None)),
            "display_name": getattr(f, "display_name", None),
        }

    to_import: list[str] = []
    for drive_file_id, item in drive_by_id.items():
        drive_modified = _parse_rfc3339_utc(item.get("modified_time"))
        existing = corpus_by_drive_id.get(drive_file_id)
        if not existing:
            to_import.append(drive_file_id)
            continue
        corpus_updated = existing.get("update_time")
        if drive_modified and corpus_updated and drive_modified > corpus_updated:
            to_import.append(drive_file_id)

    to_delete: list[str] = []
    if delete_removed:
        for drive_file_id, meta in corpus_by_drive_id.items():
            if drive_file_id not in drive_by_id:
                rag_file_name = meta.get("rag_file_name")
                if rag_file_name:
                    to_delete.append(rag_file_name)

    result: dict = {
        "drive_id": drive_id,
        "folder_name": folder_name,
        "corpus_name": corpus_name,
        "drive_files": len(drive_by_id),
        "corpus_files": len(corpus_by_drive_id),
        "plan": {
            "import": len(to_import),
            "delete": len(to_delete),
        },
    }

    if dry_run:
        result["planned_import_file_ids"] = to_import
        result["planned_delete_rag_file_names"] = to_delete
        return result

    import_response = None
    if to_import:
        paths = [f"https://drive.google.com/file/d/{fid}/view" for fid in to_import]
        resp = _import_paths(corpus_name, paths)
        import_response = {
            "imported_rag_files_count": getattr(resp, "imported_rag_files_count", None),
            "skipped_rag_files_count": getattr(resp, "skipped_rag_files_count", None),
            "failed_rag_files_count": getattr(resp, "failed_rag_files_count", None),
        }

    deleted = 0
    for rag_file_name in to_delete:
        try:
            rag.delete_file(rag_file_name, corpus_name=corpus_name)
            deleted += 1
        except Exception:
            continue

    result["import_response"] = import_response
    result["deleted_files_count"] = deleted
    return result
