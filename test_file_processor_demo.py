#!/usr/bin/env python3
"""
Simple demo of the enhanced file processor API.
"""

import asyncio
import io
from reportlab.lib.pagesizes import letter  # type: ignore
from reportlab.pdfgen import canvas  # type: ignore

from llama_stack.apis.vector_io.vector_io import (
    VectorStoreChunkingStrategyStatic,
    VectorStoreChunkingStrategyStaticConfig,
)
from llama_stack.providers.inline.file_processors.pypdf.config import PyPDFConfig
from llama_stack.providers.inline.file_processors.pypdf.pypdf import PyPDFFileProcessorImpl


def create_sample_pdf() -> bytes:
    """Create a simple test PDF."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)

    pdf.drawString(100, 750, "Test Document")
    pdf.drawString(100, 700, "This is a sample document for testing the file processor API.")
    pdf.drawString(100, 650, "It demonstrates chunking and processing capabilities.")

    for i in range(10):
        pdf.drawString(100, 600 - i * 20, f"Line {i + 1}: Additional content for chunking test.")

    pdf.save()
    buffer.seek(0)
    return buffer.read()


async def demo():
    """Demonstrate the file processor API."""
    print("🧪 File Processor API Demo")
    print("=" * 40)

    # Initialize processor
    config = PyPDFConfig()
    processor = PyPDFFileProcessorImpl(config)

    # Create test PDF
    pdf_data = create_sample_pdf()
    print(f"📄 Created test PDF: {len(pdf_data)} bytes")

    # Test basic processing
    print("\n📝 Basic Processing (no chunking):")
    result = await processor.process_file(pdf_data, "demo.pdf")
    print(f"   Content length: {len(result.content)} characters")
    print(f"   Pages: {result.metadata.get('pages', 0)}")
    print(f"   Chunks: {result.chunks}")

    # Test with chunking
    print("\n✂️  Processing with Chunking:")
    chunking_strategy = VectorStoreChunkingStrategyStatic(
        type="static",
        static=VectorStoreChunkingStrategyStaticConfig(
            max_chunk_size_tokens=100,
            chunk_overlap_tokens=20
        )
    )

    result_chunked = await processor.process_file(
        pdf_data,
        "demo.pdf",
        chunking_strategy=chunking_strategy
    )
    print(f"   Content length: {len(result_chunked.content)} characters")
    print(f"   Number of chunks: {len(result_chunked.chunks) if result_chunked.chunks else 0}")

    if result_chunked.chunks:
        print(f"   First chunk: {result_chunked.chunks[0][:100]}...")
        print(f"   Last chunk: {result_chunked.chunks[-1][:100]}...")

    print("\n✅ Demo complete! The chunks are returned in the ProcessedContent object.")
    print("📋 You can access them via result.chunks and use them for vector store ingestion.")


if __name__ == "__main__":
    asyncio.run(demo())