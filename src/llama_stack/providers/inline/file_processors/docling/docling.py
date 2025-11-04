# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import tempfile
import time
from pathlib import Path
from typing import Any

from llama_stack.apis.file_processors import FileProcessors, ProcessedContent
from llama_stack.apis.vector_io.vector_io import VectorStoreChunkingStrategy
from llama_stack.log import get_logger

from .config import DoclingConfig

logger = get_logger(__name__)


class DoclingFileProcessorImpl(FileProcessors):
    def __init__(self, config: DoclingConfig):
        self.config = config
        self.converter: Any = None
        self._initialize_docling()
        logger.info("Docling processor initialized with DocumentConverter")

    def _initialize_docling(self):
        """Initialize Docling document converter and chunker"""
        try:
            from docling.document_converter import DocumentConverter  # type: ignore

            # Initialize DocumentConverter with configuration
            self.converter = DocumentConverter()

            # Initialize HybridChunker for intelligent document-aware chunking
            try:
                from docling_core.transforms.chunker.hybrid_chunker import HybridChunker  # type: ignore

                self.chunker = HybridChunker()
                logger.info("Docling DocumentConverter and HybridChunker initialized successfully")
            except ImportError:
                logger.warning("docling-core[chunking] not installed. Chunking will fall back to simple approach.")
                self.chunker = None

        except ImportError as e:
            logger.error("Docling not installed. Run: pip install docling")
            raise ImportError("Docling not installed. Run: pip install docling") from e
        except Exception as e:
            logger.error(f"Failed to initialize Docling DocumentConverter: {e}")
            raise RuntimeError(f"Failed to initialize Docling DocumentConverter: {e}") from e

    def _parse_docling_options(self, options: dict[str, Any]) -> dict[str, Any]:
        """Parse and validate Docling-specific options"""
        if not options:
            return {}

        # ConvertManager supports these options
        docling_options = {}

        # Output format options
        if "format" in options:
            docling_options["output_format"] = options["format"]

        # Processing options that ConvertManager handles
        if "extract_tables" in options:
            docling_options["extract_tables"] = bool(options["extract_tables"])
        if "extract_figures" in options:
            docling_options["extract_figures"] = bool(options["extract_figures"])
        if "ocr_enabled" in options:
            docling_options["ocr_enabled"] = bool(options["ocr_enabled"])
        if "ocr_languages" in options and isinstance(options["ocr_languages"], list):
            docling_options["ocr_languages"] = options["ocr_languages"]
        if "preserve_layout" in options:
            docling_options["preserve_layout"] = bool(options["preserve_layout"])

        return docling_options

    def _create_docling_chunks(
        self, document: Any, chunking_strategy: VectorStoreChunkingStrategy, format_type: str
    ) -> tuple[list[str], int]:
        """Create chunks using Docling's native chunking capabilities"""
        if not self.chunker:
            # Fall back to simple text-based chunking if HybridChunker not available
            return self._fallback_text_chunking(document, chunking_strategy, format_type)

        try:
            from llama_stack.apis.vector_io.vector_io import VectorStoreChunkingStrategyStatic

            # Configure chunker based on strategy
            chunker_kwargs = {}

            if isinstance(chunking_strategy, VectorStoreChunkingStrategyStatic):
                # Use static strategy parameters for Docling chunker
                max_tokens = chunking_strategy.static.max_chunk_size_tokens
                # Note: Docling chunker doesn't have direct overlap control,
                # but HybridChunker has its own sophisticated overlap strategy
                chunker_kwargs["max_tokens"] = max_tokens
            else:
                # Default for auto strategy - use Docling's defaults
                chunker_kwargs["max_tokens"] = 800

            # Create chunks using Docling's HybridChunker
            # This respects document structure (sections, paragraphs, tables, etc.)
            chunk_iter = self.chunker.chunk(dl_doc=document, **chunker_kwargs)

            # Convert chunks to text using contextualize method
            chunks = []
            for chunk in chunk_iter:
                # contextualize() returns the metadata-enriched serialization
                chunk_text = self.chunker.contextualize(chunk)
                chunks.append(chunk_text)

            logger.info(f"Docling native chunking created {len(chunks)} chunks")
            return chunks, len(chunks)

        except Exception as e:
            logger.warning(f"Docling native chunking failed, falling back to text chunking: {e}")
            return self._fallback_text_chunking(document, chunking_strategy, format_type)

    def _fallback_text_chunking(
        self, document: Any, chunking_strategy: VectorStoreChunkingStrategy, format_type: str
    ) -> tuple[list[str], int]:
        """Fallback chunking using simple text approach when native chunking fails"""
        try:
            from llama_stack.apis.vector_io.vector_io import VectorStoreChunkingStrategyStatic
            from llama_stack.providers.utils.inference.prompt_adapter import interleaved_content_as_str
            from llama_stack.providers.utils.memory.vector_store import make_overlapped_chunks

            # Export document as text for fallback chunking
            if format_type == "markdown":
                text = document.export_to_markdown()
            else:
                text = document.export_to_text()

            if isinstance(chunking_strategy, VectorStoreChunkingStrategyStatic):
                max_chunk_size = chunking_strategy.static.max_chunk_size_tokens
                chunk_overlap = chunking_strategy.static.chunk_overlap_tokens
            else:
                # Default for auto strategy
                max_chunk_size = 800
                chunk_overlap = 400

            # Create document metadata for chunking
            document_metadata = {
                "processor": "docling_fallback",
                "pages": len(document.pages) if hasattr(document, "pages") else 0,
                "tables": len(document.tables) if hasattr(document, "tables") else 0,
                "figures": len(document.figures) if hasattr(document, "figures") else 0,
            }

            # Generate chunks using existing utility
            chunk_objects = make_overlapped_chunks(
                document_id="docling_document",
                text=text,
                window_len=max_chunk_size,
                overlap_len=chunk_overlap,
                metadata=document_metadata,
            )

            # Extract text content from chunk objects and convert to strings
            chunks = [interleaved_content_as_str(chunk.content) for chunk in chunk_objects]
            logger.info(f"Docling fallback chunking created {len(chunks)} chunks")
            return chunks, len(chunks)

        except Exception as e:
            logger.error(f"Fallback chunking also failed: {e}")
            return [], 0

    async def process_file(
        self,
        file_data: bytes,
        filename: str,
        options: dict[str, Any] | None = None,
        chunking_strategy: VectorStoreChunkingStrategy | None = None,
        include_embeddings: bool = False,
    ) -> ProcessedContent:
        start_time = time.time()
        options = options or {}

        logger.info(f"Processing file with Docling DocumentConverter: {filename}, size: {len(file_data)} bytes")
        logger.debug(f"Docling options: {options}")

        try:
            # Parse options for DocumentConverter
            docling_options = self._parse_docling_options(options)

            # Process file using temporary file (Docling requirement)
            with tempfile.NamedTemporaryFile(suffix=f"_{filename}", delete=False) as tmp:
                tmp.write(file_data)
                tmp.flush()
                tmp_path = Path(tmp.name)

                try:
                    # Convert using the DocumentConverter
                    result = self.converter.convert(tmp_path)

                    # Determine output format
                    format_type = options.get("format", "markdown")

                    # Export content based on requested format
                    if format_type == "markdown":
                        content = result.document.export_to_markdown()
                    elif format_type == "html":
                        content = result.document.export_to_html()
                    elif format_type == "json":
                        content = result.document.export_to_json()
                    else:
                        content = result.document.export_to_text()

                    processing_time = time.time() - start_time

                    # Handle chunking if strategy provided
                    chunks = None
                    embeddings = None
                    chunk_count = 0

                    if chunking_strategy:
                        chunks, chunk_count = self._create_docling_chunks(
                            result.document, chunking_strategy, format_type
                        )

                        # Generate embeddings if requested
                        if include_embeddings and chunks:
                            # TODO: Implement embedding generation
                            # This would require access to an embedding model
                            logger.warning("Embedding generation not yet implemented for Docling provider")

                    # Extract metadata from Docling result
                    processed = ProcessedContent(
                        content=content,
                        chunks=chunks,
                        embeddings=embeddings,
                        metadata={
                            "pages": len(result.document.pages) if hasattr(result.document, "pages") else 0,
                            "tables": len(result.document.tables) if hasattr(result.document, "tables") else 0,
                            "figures": len(result.document.figures) if hasattr(result.document, "figures") else 0,
                            "format": format_type,
                            "processor": "docling",
                            "processing_time_seconds": processing_time,
                            "content_length": len(content),
                            "filename": filename,
                            "file_size_bytes": len(file_data),
                            "converter_options": docling_options,
                            "chunking_strategy": chunking_strategy.model_dump() if chunking_strategy else None,
                            "chunk_count": chunk_count,
                            "include_embeddings": include_embeddings,
                            "docling_version": getattr(result, "version", "unknown"),
                        },
                    )

                    logger.info(
                        f"Docling processing completed: {processed.metadata.get('pages', 0)} pages, "
                        f"{processed.metadata.get('tables', 0)} tables, {chunk_count} chunks, {processing_time:.2f}s"
                    )
                    return processed

                finally:
                    # Clean up temporary file
                    try:
                        tmp_path.unlink()
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to cleanup temp file {tmp_path}: {cleanup_error}")

        except Exception as e:
            logger.error(f"Docling processing failed for {filename}: {str(e)}")
            raise RuntimeError(f"Docling processing failed: {str(e)}") from e
