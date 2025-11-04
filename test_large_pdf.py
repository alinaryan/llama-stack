#!/usr/bin/env python3
"""
Test with a larger PDF to show chunking capabilities.
"""

import asyncio
from pathlib import Path

from llama_stack.apis.vector_io.vector_io import (
    VectorStoreChunkingStrategyStatic,
    VectorStoreChunkingStrategyStaticConfig,
)
from llama_stack.providers.inline.file_processors.pypdf.config import PyPDFConfig
from llama_stack.providers.inline.file_processors.pypdf.pypdf import PyPDFFileProcessorImpl


async def test_large_pdf():
    """Test with a larger PDF document."""
    print("📚 Testing with Larger PDF")
    print("=" * 40)

    # Test with the travel policy PDF (should be more complex)
    pdf_path = Path("test_data/pdfs/US-Youth-Soccer-Travel-Policy.pdf")

    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path}")
        return

    # Initialize processor
    config = PyPDFConfig()
    processor = PyPDFFileProcessorImpl(config)

    # Read the PDF
    file_data = pdf_path.read_bytes()
    print(f"📄 Processing: {pdf_path.name}")
    print(f"📊 File size: {len(file_data):,} bytes")

    # Process with moderate chunking
    chunking_strategy = VectorStoreChunkingStrategyStatic(
        type="static",
        static=VectorStoreChunkingStrategyStaticConfig(
            max_chunk_size_tokens=400,
            chunk_overlap_tokens=80
        )
    )

    result = await processor.process_file(
        file_data,
        pdf_path.name,
        chunking_strategy=chunking_strategy
    )

    print(f"✅ Processing complete!")
    print(f"📝 Content length: {len(result.content):,} characters")
    print(f"📖 Pages: {result.metadata.get('pages', 0)}")
    print(f"✂️  Chunks created: {len(result.chunks) if result.chunks else 0}")
    print(f"⏱️  Processing time: {result.metadata.get('processing_time_seconds', 0):.3f}s")

    if result.chunks:
        print(f"\n📋 Chunk Analysis:")
        chunk_sizes = [len(chunk) for chunk in result.chunks]
        print(f"   Average chunk size: {sum(chunk_sizes) / len(chunk_sizes):.0f} characters")
        print(f"   Smallest chunk: {min(chunk_sizes)} characters")
        print(f"   Largest chunk: {max(chunk_sizes)} characters")

        print(f"\n📝 Sample chunks:")
        for i, chunk in enumerate(result.chunks[:3]):
            print(f"   Chunk {i+1}: {chunk[:150]}...")

        print(f"\n💡 How to use these chunks:")
        print(f"   • Insert into vector database for semantic search")
        print(f"   • Generate embeddings for each chunk")
        print(f"   • Use for RAG (Retrieval Augmented Generation)")
        print(f"   • Index for document Q&A systems")

    return result


async def main():
    """Run the large PDF test."""
    try:
        result = await test_large_pdf()

        if result and result.chunks:
            print(f"\n🔧 Example: Accessing chunks programmatically")
            print("```python")
            print("# After processing:")
            print("for i, chunk in enumerate(result.chunks):")
            print("    print(f'Chunk {i}: {len(chunk)} chars')")
            print("    # Send to vector store, generate embeddings, etc.")
            print("```")

            print(f"\n🎯 Output locations in your code:")
            print("   • result.content    -> Full extracted text")
            print("   • result.chunks     -> List of chunk strings")
            print("   • result.metadata   -> Processing information")
            print("   • result.embeddings -> Embedding vectors (if requested)")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())