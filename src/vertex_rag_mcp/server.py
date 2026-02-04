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
    top_k: Annotated[int, "Number of top matches to return."] = 5,
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


async def main():
    # Run the server
    await mcp.run_stdio_async()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
