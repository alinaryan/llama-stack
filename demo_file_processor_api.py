#!/usr/bin/env python3
"""
Demo script showing how to use the File Processors API.

This demonstrates the flexible file processing capabilities including
conversion, chunking, and the extensible provider architecture.
"""

import asyncio
from typing import Any

from llama_stack.apis.file_processors import ProcessedContent
from llama_stack.providers.inline.file_processors.placeholder import PlaceholderFileProcessorsImpl
from llama_stack.providers.inline.file_processors.placeholder.config import PlaceholderConfig


async def demo_file_processing():
    """Demonstrate file processing capabilities."""

    # Configure the file processor
    config = PlaceholderConfig(
        max_file_size_mb=100,
        timeout_seconds=30,
        enable_chunking=True,
        chunk_size=500
    )

    # Initialize the processor (in real usage, this would be managed by the stack)
    processor = PlaceholderFileProcessorsImpl(config)

    # Example 1: Basic text file processing
    print("=== Example 1: Basic Text Processing ===")
    text_content = "This is a sample document with multiple sentences. " * 10
    text_data = text_content.encode('utf-8')

    result = await processor.process_file(
        file_data=text_data,
        filename="sample.txt",
        options={"enable_chunking": True, "chunk_size": 100}
    )

    print(f"Processed content length: {len(result.content)}")
    print(f"Number of chunks: {len(result.chunks) if result.chunks else 0}")
    print(f"Metadata: {result.metadata}")
    print()

    # Example 2: Processing with different options
    print("=== Example 2: Processing Without Chunking ===")
    result_no_chunks = await processor.process_file(
        file_data=text_data,
        filename="sample.txt",
        options={"enable_chunking": False}
    )

    print(f"Processed content length: {len(result_no_chunks.content)}")
    print(f"Chunks: {result_no_chunks.chunks}")
    print()

    # Example 3: Simulate different file types
    print("=== Example 3: Different File Types ===")

    # PDF simulation
    pdf_result = await processor.process_file(
        file_data=b"Mock PDF content",
        filename="document.pdf",
        options={"chunking_strategy": "paragraph"}
    )
    print(f"PDF processing result: {pdf_result.metadata}")

    # Image simulation
    image_result = await processor.process_file(
        file_data=b"\x89PNG\r\n\x1a\n...",  # Mock binary data
        filename="image.png",
        options={"enable_ocr": True}
    )
    print(f"Image processing result: {image_result.metadata}")
    print()

    print("=== Demo Complete ===")
    print("This demonstrates the flexible architecture where:")
    print("- Multiple file types can be processed")
    print("- Different chunking strategies can be applied")
    print("- Provider-specific options can be configured")
    print("- Metadata is preserved for downstream processing")


if __name__ == "__main__":
    asyncio.run(demo_file_processing())