#!/usr/bin/env python3
"""
Test the enhanced file processor API with real PDF files.
This demo shows how to process PDFs with different chunking strategies
and where to access the output.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

from llama_stack.apis.vector_io.vector_io import (
    VectorStoreChunkingStrategyStatic,
    VectorStoreChunkingStrategyStaticConfig,
)
from llama_stack.providers.inline.file_processors.pypdf.config import PyPDFConfig
from llama_stack.providers.inline.file_processors.pypdf.pypdf import PyPDFFileProcessorImpl


def format_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def save_results(pdf_file: Path, result: Any, output_dir: Path) -> None:
    """Save processing results to files for inspection."""
    file_stem = pdf_file.stem.replace(" ", "_").replace("(", "").replace(")", "")

    # Save full extracted text
    content_file = output_dir / f"{file_stem}_content.txt"
    content_file.write_text(result.content, encoding='utf-8')

    # Save metadata as JSON
    metadata_file = output_dir / f"{file_stem}_metadata.json"
    metadata_file.write_text(json.dumps(result.metadata, indent=2), encoding='utf-8')

    # Save chunks if available
    if result.chunks:
        chunks_dir = output_dir / f"{file_stem}_chunks"
        chunks_dir.mkdir(exist_ok=True)

        for i, chunk in enumerate(result.chunks):
            chunk_file = chunks_dir / f"chunk_{i:03d}.txt"
            chunk_file.write_text(chunk, encoding='utf-8')

        # Save chunk summary
        chunk_summary = {
            "total_chunks": len(result.chunks),
            "chunk_sizes": [len(chunk) for chunk in result.chunks],
            "average_chunk_size": sum(len(chunk) for chunk in result.chunks) / len(result.chunks),
            "chunks_preview": [chunk[:100] + "..." if len(chunk) > 100 else chunk for chunk in result.chunks[:3]]
        }

        summary_file = output_dir / f"{file_stem}_chunk_summary.json"
        summary_file.write_text(json.dumps(chunk_summary, indent=2), encoding='utf-8')

    return file_stem


async def process_pdf(processor: PyPDFFileProcessorImpl, pdf_file: Path, chunking_strategy=None) -> Any:
    """Process a single PDF file."""
    print(f"\n📄 Processing: {pdf_file.name}")
    print(f"   File size: {format_size(pdf_file.stat().st_size)}")

    # Read the PDF
    file_data = pdf_file.read_bytes()

    # Process the file
    result = await processor.process_file(
        file_data,
        pdf_file.name,
        chunking_strategy=chunking_strategy,
        include_embeddings=False  # Set to True if you want to test embedding requests
    )

    print(f"   ✅ Processed successfully!")
    print(f"   📝 Content length: {len(result.content):,} characters")
    print(f"   📖 Pages: {result.metadata.get('pages', 'unknown')}")
    print(f"   ⏱️  Processing time: {result.metadata.get('processing_time_seconds', 0):.3f}s")

    if result.chunks:
        print(f"   ✂️  Chunks created: {len(result.chunks)}")
        avg_chunk_size = sum(len(chunk) for chunk in result.chunks) / len(result.chunks)
        print(f"   📏 Average chunk size: {avg_chunk_size:.0f} characters")
    else:
        print(f"   ✂️  No chunking applied")

    return result


async def main():
    """Test file processor with real PDFs."""
    print("🧪 Enhanced File Processor API - Real PDF Test")
    print("=" * 60)

    # Set up paths
    pdf_dir = Path("/home/alina/dev/pdfs/just-text")
    output_dir = Path("./processor_results")
    output_dir.mkdir(exist_ok=True)

    print(f"📂 PDF directory: {pdf_dir}")
    print(f"💾 Output directory: {output_dir}")

    # Initialize processor
    config = PyPDFConfig()
    processor = PyPDFFileProcessorImpl(config)
    print(f"🔧 Initialized PyPDF processor")

    # Get PDF files
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in {pdf_dir}")
        return

    print(f"📋 Found {len(pdf_files)} PDF files")

    # Test different scenarios
    test_scenarios = [
        {
            "name": "Basic Processing (No Chunking)",
            "chunking_strategy": None,
            "description": "Extract text without chunking"
        },
        {
            "name": "Small Chunks (200 tokens)",
            "chunking_strategy": VectorStoreChunkingStrategyStatic(
                type="static",
                static=VectorStoreChunkingStrategyStaticConfig(
                    max_chunk_size_tokens=200,
                    chunk_overlap_tokens=50
                )
            ),
            "description": "Create small chunks for detailed analysis"
        },
        {
            "name": "Large Chunks (800 tokens)",
            "chunking_strategy": VectorStoreChunkingStrategyStatic(
                type="static",
                static=VectorStoreChunkingStrategyStaticConfig(
                    max_chunk_size_tokens=800,
                    chunk_overlap_tokens=100
                )
            ),
            "description": "Create larger chunks for context preservation"
        }
    ]

    # Process a smaller PDF first (let's use the smallest one)
    pdf_files_by_size = sorted(pdf_files, key=lambda p: p.stat().st_size)
    test_file = pdf_files_by_size[0]  # Start with smallest file

    print(f"\n🎯 Testing with: {test_file.name} ({format_size(test_file.stat().st_size)})")

    results = {}

    for scenario in test_scenarios:
        print(f"\n{'='*20} {scenario['name']} {'='*20}")
        print(f"📝 {scenario['description']}")

        try:
            result = await process_pdf(processor, test_file, scenario['chunking_strategy'])

            # Save results
            file_stem = save_results(test_file, result, output_dir)
            scenario_name = scenario['name'].lower().replace(' ', '_').replace('(', '').replace(')', '')

            # Move files to scenario-specific subdirectory
            scenario_dir = output_dir / f"{file_stem}_{scenario_name}"
            scenario_dir.mkdir(exist_ok=True)

            # Move the generated files
            for file_path in output_dir.glob(f"{file_stem}_*"):
                if file_path.is_file():
                    file_path.rename(scenario_dir / file_path.name)
                elif file_path.is_dir() and f"{file_stem}_chunks" in file_path.name:
                    import shutil
                    shutil.move(str(file_path), str(scenario_dir / file_path.name))

            results[scenario['name']] = {
                'content_length': len(result.content),
                'chunk_count': len(result.chunks) if result.chunks else 0,
                'pages': result.metadata.get('pages', 0),
                'processing_time': result.metadata.get('processing_time_seconds', 0),
                'output_dir': str(scenario_dir)
            }

        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[scenario['name']] = {'error': str(e)}

    # Summary
    print(f"\n{'='*60}")
    print("📊 PROCESSING SUMMARY")
    print(f"{'='*60}")

    for scenario_name, result in results.items():
        print(f"\n🔸 {scenario_name}:")
        if 'error' in result:
            print(f"   ❌ Failed: {result['error']}")
        else:
            print(f"   📝 Content: {result['content_length']:,} characters")
            print(f"   ✂️  Chunks: {result['chunk_count']}")
            print(f"   📖 Pages: {result['pages']}")
            print(f"   ⏱️  Time: {result['processing_time']:.3f}s")
            print(f"   💾 Output: {result['output_dir']}")

    print(f"\n🎯 HOW TO ACCESS THE OUTPUT:")
    print(f"   • Full text content: result.content")
    print(f"   • Text chunks: result.chunks (list of strings)")
    print(f"   • Processing metadata: result.metadata")
    print(f"   • Embeddings: result.embeddings (if requested)")

    print(f"\n📁 All results saved in: {output_dir.absolute()}")
    print(f"💡 You can now inspect the extracted text and chunks in the output files!")

    # Show how to use the results programmatically
    print(f"\n🔧 PROGRAMMATIC ACCESS EXAMPLE:")
    print("""
    # After processing:
    result = await processor.process_file(file_data, filename, chunking_strategy=strategy)

    # Access full text
    full_text = result.content

    # Access chunks for vector store ingestion
    if result.chunks:
        for i, chunk in enumerate(result.chunks):
            print(f"Chunk {i}: {chunk[:100]}...")
            # Insert into vector store, generate embeddings, etc.

    # Access metadata
    pages = result.metadata.get('pages', 0)
    processing_time = result.metadata.get('processing_time_seconds', 0)
    """)


if __name__ == "__main__":
    asyncio.run(main())