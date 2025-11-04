#!/usr/bin/env python3
"""
Test file processor via Llama Stack API (when running server).
"""

import asyncio
import aiohttp
import json
from pathlib import Path


async def test_via_api():
    """Test file processor through the API endpoint."""
    print("🌐 Testing via Llama Stack API")
    print("=" * 40)

    # Assuming server is running on default port
    base_url = "http://localhost:8321"
    api_url = f"{base_url}/file-processors/process"

    print(f"📡 API endpoint: {api_url}")
    print("ℹ️  Make sure Llama Stack server is running with file processors enabled!")
    print("   Run: llama stack run <distribution-with-file-processors>")

    # Example request - you'd replace this with actual file data
    print("\n💡 Example API usage:")
    print("POST /file-processors/process")
    print("Content-Type: application/json")
    print("""
{
    "file_data": "<base64-encoded-file-data>",
    "filename": "document.pdf",
    "options": {"format": "markdown"},
    "chunking_strategy": {
        "type": "static",
        "static": {
            "max_chunk_size_tokens": 300,
            "chunk_overlap_tokens": 50
        }
    },
    "include_embeddings": false
}
""")

    print("\n📤 Response format:")
    print("""
{
    "content": "Extracted text content...",
    "chunks": ["chunk 1 text...", "chunk 2 text..."],
    "embeddings": null,
    "metadata": {
        "pages": 5,
        "processor": "pypdf",
        "chunk_count": 12,
        "processing_time_seconds": 0.15
    }
}
""")


if __name__ == "__main__":
    asyncio.run(test_via_api())