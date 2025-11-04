#!/usr/bin/env python3
"""
Test file processor with an actual PDF file.
"""

import asyncio
from pathlib import Path

from llama_stack.apis.vector_io.vector_io import (
    VectorStoreChunkingStrategyStatic,
    VectorStoreChunkingStrategyStaticConfig,
)
from llama_stack.providers.inline.file_processors.pypdf.config import PyPDFConfig
from llama_stack.providers.inline.file_processors.pypdf.pypdf import PyPDFFileProcessorImpl


async def test_real_file():
    """Test with a real PDF file if available."""
    print("📂 Testing with Real PDF Files")
    print("=" * 40)

    # Initialize processor
    config = PyPDFConfig()
    processor = PyPDFFileProcessorImpl(config)

    # Look for any PDF files in current directory
    pdf_files = list(Path(".").glob("*.pdf"))

    if not pdf_files:
        print("❌ No PDF files found in current directory.")
        print("💡 Try putting a PDF file here and running again!")
        print("   Or specify a path below:")

        # Example with custom path
        custom_path = input("Enter path to PDF file (or press Enter to skip): ").strip()
        if custom_path and Path(custom_path).exists():
            pdf_files = [Path(custom_path)]

    if pdf_files:
        pdf_file = pdf_files[0]
        print(f"📄 Testing with: {pdf_file.name}")

        # Read the file
        file_data = pdf_file.read_bytes()

        # Process with chunking
        chunking_strategy = VectorStoreChunkingStrategyStatic(
            type="static",
            static=VectorStoreChunkingStrategyStaticConfig(
                max_chunk_size_tokens=300,
                chunk_overlap_tokens=50
            )
        )

        result = await processor.process_file(
            file_data,
            pdf_file.name,
            chunking_strategy=chunking_strategy
        )

        print(f"✅ Processed successfully!")
        print(f"   File size: {len(file_data)} bytes")
        print(f"   Content length: {len(result.content)} characters")
        print(f"   Pages: {result.metadata.get('pages', 0)}")
        print(f"   Chunks created: {len(result.chunks) if result.chunks else 0}")

        if result.chunks:
            print(f"\n📝 First chunk preview:")
            print(f"   {result.chunks[0][:200]}...")

        # Save output to files for inspection
        output_dir = Path("processor_output")
        output_dir.mkdir(exist_ok=True)

        # Save full content
        (output_dir / f"{pdf_file.stem}_content.txt").write_text(result.content)
        print(f"💾 Full content saved to: {output_dir}/{pdf_file.stem}_content.txt")

        # Save chunks if available
        if result.chunks:
            for i, chunk in enumerate(result.chunks):
                (output_dir / f"{pdf_file.stem}_chunk_{i:03d}.txt").write_text(chunk)
            print(f"💾 {len(result.chunks)} chunks saved to: {output_dir}/{pdf_file.stem}_chunk_*.txt")

        print(f"📊 Metadata saved to: {output_dir}/{pdf_file.stem}_metadata.txt")
        (output_dir / f"{pdf_file.stem}_metadata.txt").write_text(str(result.metadata))

    else:
        print("❌ No PDF files to test with.")


if __name__ == "__main__":
    asyncio.run(test_real_file())