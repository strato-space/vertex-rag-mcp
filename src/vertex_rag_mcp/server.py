from __future__ import annotations

from typing import Annotated

try:
    # FastMCP SDK
    from mcp.server.fastmcp import FastMCP
except Exception as e:  # pragma: no cover
    raise SystemExit(f"FastMCP not available: {e}")

from mcp.types import ToolAnnotations

from . import vertex_rag_tool

mcp = FastMCP("vertex-rag-mcp")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def mini_rag(
    query: Annotated[str, "Natural language query to search against the RAG corpus."],
    drive_id: Annotated[str, "Google Drive folder ID to index/search."],
    top_k: Annotated[int, "Number of top matches to return."] = 50,
) -> object:
    """Query Vertex RAG using a Google Drive folder ID and return retrieval results."""
    return vertex_rag_tool.mini_rag(query=query, drive_id=drive_id, top_k=top_k)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def list_drive_files(
    drive_id: Annotated[str, "Google Drive folder ID to list files from."],
    recursive: Annotated[bool, "Recursively traverse nested folders."] = True,
    include_folders: Annotated[bool, "Include folder entries in results."] = False,
    limit: Annotated[int, "Optional max number of returned items (0 = no limit)."] = 0,
) -> object:
    """List Drive files in a folder with sizes and timestamps."""
    return vertex_rag_tool.list_drive_files(
        drive_id=drive_id,
        recursive=recursive,
        include_folders=include_folders,
        limit=limit or None,
    )


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def read_multiple_binary_files(
    drive_id: Annotated[str, "Google Drive folder ID to list documents from."],
    file_ids: Annotated[
        list[str] | None, "Optional list of specific Drive file IDs to include."
    ] = None,
    include_types: Annotated[
        list[str] | None, "Optional list of MIME types to include (None = defaults)."
    ] = None,
    all_files: Annotated[bool, "If True, ignore type filters and return all files."] = False,
    recursive: Annotated[bool, "Recursively traverse nested folders."] = True,
    tool_result: Annotated[
        str, "Controls content[] shape: 'resource_link' (default) or 'resource'."
    ] = "resource_link",
    response_format: Annotated[
        str, "Controls structuredContent: 'url' (default) or 'b64_json'."
    ] = "url",
    max_files: Annotated[int, "Max number of files to download (0 = no limit)."] = 0,
    max_total_bytes: Annotated[int, "Max total bytes to download (0 = no limit)."] = 0,
    max_file_bytes: Annotated[int, "Max per-file bytes to download (0 = no limit)."] = 0,
    chunk_size: Annotated[int, "Download chunk size in bytes (0 = default)."] = 0,
) -> object:
    """List Drive documents with content for indexing (no media by default)."""
    return vertex_rag_tool.read_multiple_binary_files(
        drive_id=drive_id,
        file_ids=file_ids,
        include_types=include_types,
        all_files=all_files,
        recursive=recursive,
        tool_result=tool_result,
        response_format=response_format,
        max_files=max_files or None,
        max_total_bytes=max_total_bytes or None,
        max_file_bytes=max_file_bytes or None,
        chunk_size=chunk_size or None,
    )


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def read_multiple_text_files(
    drive_id: Annotated[str, "Google Drive folder ID to list documents from."],
    file_ids: Annotated[
        list[str] | None, "Optional list of specific Drive file IDs to include."
    ] = None,
    include_types: Annotated[
        list[str] | None, "Optional list of MIME types to include (None = defaults)."
    ] = None,
    all_files: Annotated[bool, "If True, ignore type filters and return all files."] = False,
    recursive: Annotated[bool, "Recursively traverse nested folders."] = True,
    output_format: Annotated[str, "Output format: markdown (default) or text."] = "markdown",
    max_files: Annotated[int, "Max number of files to download (0 = no limit)."] = 0,
    max_total_chars: Annotated[int, "Max total chars to return (0 = no limit)."] = 0,
    max_file_chars: Annotated[int, "Max chars per file (0 = no limit)."] = 0,
    max_pages: Annotated[int, "Max PDF pages to read (0 = no limit)."] = 0,
    max_table_rows: Annotated[int, "Max CSV rows to render (default 50)."] = 50,
    max_table_cols: Annotated[int, "Max CSV columns to render (default 12)."] = 12,
    chunk_size: Annotated[int, "Download chunk size in bytes (0 = default)."] = 0,
) -> object:
    """Extract text from Drive documents and return Markdown/text."""
    return vertex_rag_tool.read_multiple_text_files(
        drive_id=drive_id,
        file_ids=file_ids,
        include_types=include_types,
        all_files=all_files,
        recursive=recursive,
        output_format=output_format,
        max_files=max_files or None,
        max_total_chars=max_total_chars or None,
        max_file_chars=max_file_chars or None,
        max_pages=max_pages or None,
        max_table_rows=max_table_rows,
        max_table_cols=max_table_cols,
        chunk_size=chunk_size or None,
    )


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def read_multiple_md_files(
    drive_id: Annotated[str, "Google Drive folder ID to list documents from."],
    file_ids: Annotated[
        list[str] | None, "Optional list of specific Drive file IDs to include."
    ] = None,
    include_types: Annotated[
        list[str] | None, "Optional list of MIME types to include (None = defaults)."
    ] = None,
    all_files: Annotated[bool, "If True, ignore type filters and return all files."] = False,
    recursive: Annotated[bool, "Recursively traverse nested folders."] = True,
    convert_pdf_to_gdoc: Annotated[
        bool, "Convert PDFs to Google Docs before exporting Markdown."
    ] = True,
    cleanup_converted: Annotated[
        bool, "Delete temporary converted Docs after export."
    ] = True,
    max_files: Annotated[int, "Max number of files to download (0 = no limit)."] = 0,
    max_total_chars: Annotated[int, "Max total chars to return (0 = no limit)."] = 0,
    max_file_chars: Annotated[int, "Max chars per file (0 = no limit)."] = 0,
    max_pages: Annotated[int, "Max PDF pages to read when not converting (0 = no limit)."] = 0,
    max_table_rows: Annotated[int, "Max CSV rows to render (default 50)."] = 50,
    max_table_cols: Annotated[int, "Max CSV columns to render (default 12)."] = 12,
    chunk_size: Annotated[int, "Download chunk size in bytes (0 = default)."] = 0,
) -> object:
    """Extract text from Drive documents and return Markdown (prefers Google Docs export)."""
    return vertex_rag_tool.read_multiple_md_files(
        drive_id=drive_id,
        file_ids=file_ids,
        include_types=include_types,
        all_files=all_files,
        recursive=recursive,
        convert_pdf_to_gdoc=convert_pdf_to_gdoc,
        cleanup_converted=cleanup_converted,
        max_files=max_files or None,
        max_total_chars=max_total_chars or None,
        max_file_chars=max_file_chars or None,
        max_pages=max_pages or None,
        max_table_rows=max_table_rows,
        max_table_cols=max_table_cols,
        chunk_size=chunk_size or None,
    )


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def read_single_file_raw(
    file_id: Annotated[str, "Google Drive file ID to download."],
    export_mime: Annotated[
        str | None, "Optional export MIME for Google Docs types (defaults to text/plain/csv)."
    ] = None,
    chunk_size: Annotated[int, "Download chunk size in bytes (0 = default)."] = 0,
) -> object:
    """Return raw base64 content for a single Drive file (no structuredContent wrapper)."""
    return vertex_rag_tool.read_single_file_raw(
        file_id=file_id,
        export_mime=export_mime,
        chunk_size=chunk_size or None,
    )


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False))
def full_refresh_corpus(
    drive_id: Annotated[str, "Google Drive folder ID to (re)index."],
    delete_old: Annotated[bool, "Delete previous corpus versions after success."] = True,
) -> object:
    """Create a fresh corpus for a Drive folder and optionally delete old versions."""
    return vertex_rag_tool.full_refresh_corpus(drive_id=drive_id, delete_old=delete_old)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False))
def incremental_update_corpus(
    drive_id: Annotated[str, "Google Drive folder ID to update."],
    delete_removed: Annotated[
        bool, "Delete corpus files that no longer exist in Drive."
    ] = False,
    dry_run: Annotated[bool, "Plan changes without applying them."] = False,
) -> object:
    """Incrementally update a corpus based on Drive modified times."""
    return vertex_rag_tool.incremental_update_corpus(
        drive_id=drive_id,
        delete_removed=delete_removed,
        dry_run=dry_run,
    )


async def main():
    # Run the server
    await mcp.run_stdio_async()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
