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


async def main():
    # Run the server
    await mcp.run_stdio_async()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
