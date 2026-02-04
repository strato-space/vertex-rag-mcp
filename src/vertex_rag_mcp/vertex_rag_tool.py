from __future__ import annotations

import google.auth
import vertexai
from googleapiclient.discovery import build
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
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"

_vertex_initialized = False


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
        fields = "nextPageToken, files(id,name,mimeType,size,createdTime,modifiedTime)"
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
    queue: list[tuple[str, str]] = [(drive_id, "")]

    while queue:
        current_folder_id, current_path = queue.pop(0)
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


def _create_and_import_corpus(display_name: str, paths: list[str]) -> rag.RagCorpus:
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
    rag.import_files(
        rag_corpus.name,
        paths,
        transformation_config=rag.TransformationConfig(
            chunking_config=rag.ChunkingConfig(
                chunk_size=512,
                chunk_overlap=100,
            ),
        ),
        max_embedding_requests_per_min=1000,
    )
    return rag_corpus


def mini_rag(query: str, drive_id: str, top_k: int = 5) -> object:
    """Query Vertex RAG using a Google Drive folder ID and return retrieval results.

    Args:
        query: Natural language query to search against the RAG corpus.
        drive_id: Google Drive folder ID to index/search. The RAG corpus is created
            on first use for this drive_id and reused on subsequent calls.
        top_k: Number of top matches to return.
    """
    project_id, location = _load_vertex_settings()
    _ensure_vertexai_init(project_id, location)
    if not drive_id:
        raise ValueError("drive_id must be a non-empty Google Drive ID.")

    paths = [f"https://drive.google.com/drive/folders/{drive_id}"]
    folder_name = _drive_folder_name(drive_id)
    key = drive_id
    display_name = f"{folder_name} | {key}"

    existing_corpus = None
    for corpus in rag.list_corpora():
        if corpus.display_name and key in corpus.display_name:
            existing_corpus = corpus
            break

    if existing_corpus:
        rag_corpus = existing_corpus
    else:
        rag_corpus = _create_and_import_corpus(display_name, paths)

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
