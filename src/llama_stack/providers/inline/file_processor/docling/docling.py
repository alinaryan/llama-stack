# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
import tempfile
import time
from pathlib import Path
from typing import Any

from llama_stack.apis.file_processor import FileProcessor, ProcessedContent
from llama_stack.apis.vector_io.vector_io import Chunk, VectorStoreChunkingStrategy
from llama_stack.log import get_logger
from llama_stack.providers.utils.memory.vector_store import extract_chunk_params_from_strategy

from .config import DoclingConfig

logger = get_logger(__name__)


class DoclingFileProcessorImpl(FileProcessor):
    def __init__(self, config: DoclingConfig):
        self.config = config
        self.converter: Any = None
        self._initialize_docling()
        logger.info("Docling processor initialized with DocumentConverter")

    def _initialize_docling(self) -> None:
        """Initialize Docling document converter and chunker"""
        # Detect optional deps without a try first
        import importlib.util

        if importlib.util.find_spec("docling.document_converter") is None:
            raise ImportError("Docling not installed. Run: pip install docling")

        from docling.document_converter import DocumentConverter  # type: ignore[import-not-found]

        self.converter = DocumentConverter()

        # Chunker is optional: no need to fail the whole init if it's missing
        if importlib.util.find_spec("docling_core.transforms.chunker.hybrid_chunker") is None:
            logger.info("docling-core[chunking] not installed; falling back to simple chunking.")
            self.chunker = None
            return

        from docling_core.transforms.chunker.hybrid_chunker import HybridChunker  # type: ignore[import-not-found]

        self.chunker = HybridChunker()
        logger.info("Initialized Docling DocumentConverter + HybridChunker")

    def _extract_document_stats(self, document: Any, attributes: list[str]) -> dict[str, int]:
        """Safely extract statistics from document attributes (pages, tables, figures, etc.)"""
        stats = {}
        for attr in attributes:
            if hasattr(document, attr):
                try:
                    stats[attr] = len(getattr(document, attr))
                except (TypeError, AttributeError):
                    stats[attr] = 0
            else:
                stats[attr] = 0
        return stats

    def _export_document_content(self, document: Any, format_type: str) -> str:
        """Export document content in the specified format using dictionary mapping"""
        export_methods = {
            "markdown": "export_to_markdown",
            "html": "export_to_html",
            "json": "export_to_json",
        }

        # Get the method name, default to text if not found
        method_name = export_methods.get(format_type, "export_to_text")

        # Check if document has the method
        if not hasattr(document, method_name):
            raise AttributeError(f"Document does not support {format_type} export")

        # Get the method and call it
        try:
            export_method = getattr(document, method_name)
            result = export_method()
            return str(result)  # Ensure we return a string
        except RuntimeError as e:
            raise RuntimeError(f"Failed to export document as {format_type}: {e}") from e

    def _parse_docling_options(self, options: dict[str, Any]) -> dict[str, Any]:
        """Parse and validate Docling-specific options"""
        if not options:
            return {}

        # Dictionary mapping input options to output format with transformations
        option_mappings = {
            # source_key: (target_key, transform_func, validation_func)
            "format": ("output_format", str, None),
            "extract_tables": ("extract_tables", bool, None),
            "extract_figures": ("extract_figures", bool, None),
            "ocr_enabled": ("ocr_enabled", bool, None),
            "ocr_languages": ("ocr_languages", None, lambda x: isinstance(x, list)),
            "preserve_layout": ("preserve_layout", bool, None),
        }

        docling_options = {}

        for source_key, (target_key, transform_func, validation_func) in option_mappings.items():
            if source_key in options:
                value = options[source_key]

                # Apply validation if provided
                if validation_func and not validation_func(value):
                    continue  # Skip invalid values

                # Apply transformation if provided
                if transform_func:
                    value = transform_func(value)

                docling_options[target_key] = value

        return docling_options

    def _create_docling_chunks(
        self, document: Any, chunking_strategy: VectorStoreChunkingStrategy, format_type: str
    ) -> tuple[list[Chunk], int]:
        """Create chunks using Docling's native chunking capabilities"""
        if not self.chunker:
            # Fall back to simple text-based chunking if HybridChunker not available
            return self._fallback_text_chunking(document, chunking_strategy, format_type)

        # Configure chunker based on strategy
        chunker_kwargs = {}

        # Extract chunk parameters using utility function
        max_tokens, _ = extract_chunk_params_from_strategy(
            chunking_strategy,
            self.config.default_chunk_size_tokens,
            self.config.default_chunk_overlap_tokens,
        )
        # Note: Docling chunker doesn't have direct overlap control,
        # but HybridChunker has its own sophisticated overlap strategy
        chunker_kwargs["max_tokens"] = max_tokens

        try:
            # Create chunks using Docling's HybridChunker
            # This respects document structure (sections, paragraphs, tables, etc.)
            chunk_iter = self.chunker.chunk(dl_doc=document, **chunker_kwargs)
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(f"Docling chunker.chunk() failed: {e}")
            return self._fallback_text_chunking(document, chunking_strategy, format_type)

        try:
            # Convert chunks to Chunk objects using contextualize method
            chunks = []
            for i, chunk in enumerate(chunk_iter):
                # contextualize() returns the metadata-enriched serialization
                chunk_text = self.chunker.contextualize(chunk)
                chunk_obj = Chunk(
                    content=chunk_text,
                    metadata={
                        "chunk_index": i,
                        "processor": "docling_hybrid",
                        "format": format_type,
                    },
                )
                chunks.append(chunk_obj)

            logger.info(f"Docling native chunking created {len(chunks)} chunks")
            return chunks, len(chunks)

        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(f"Docling chunk processing failed: {e}")
            return self._fallback_text_chunking(document, chunking_strategy, format_type)

    def _fallback_text_chunking(
        self, document: Any, chunking_strategy: VectorStoreChunkingStrategy, format_type: str
    ) -> tuple[list[Chunk], int]:
        """Fallback chunking using simple text approach when native chunking fails"""
        from llama_stack.providers.utils.memory.vector_store import make_overlapped_chunks

        # Export document as text for fallback chunking using helper method
        try:
            text = self._export_document_content(document, format_type)
        except (AttributeError, RuntimeError) as e:
            logger.error(f"Failed to export document for fallback chunking: {e}")
            return [], 0

        # Extract chunk parameters using utility function
        max_chunk_size, chunk_overlap = extract_chunk_params_from_strategy(
            chunking_strategy,
            self.config.default_chunk_size_tokens,
            self.config.default_chunk_overlap_tokens,
        )

        # Create document metadata for chunking
        document_metadata: dict[str, Any] = {"processor": "docling_fallback"}

        # Safely extract document statistics using helper function
        document_stats = self._extract_document_stats(document, ["pages", "tables", "figures"])
        document_metadata.update(document_stats)

        # Generate chunks using existing utility
        try:
            chunk_objects = make_overlapped_chunks(
                document_id="docling_document",
                text=text,
                window_len=max_chunk_size,
                overlap_len=chunk_overlap,
                metadata=document_metadata,
            )
        except ValueError as e:
            logger.error(f"Invalid parameters for chunk creation: {e}")
            return [], 0
        except TypeError as e:
            logger.error(f"Type error in chunk creation: {e}")
            return [], 0

        # Return the chunk objects directly (already in proper Chunk format)
        logger.info(f"Docling fallback chunking created {len(chunk_objects)} chunks")
        return chunk_objects, len(chunk_objects)

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

        # Handle base64 encoded file data from JSON requests
        if isinstance(file_data, str):
            if not file_data:
                raise ValueError("Empty file data provided")
            file_data = base64.b64decode(file_data)
            logger.debug(f"Decoded base64 file data: {len(file_data)} bytes")
        elif not isinstance(file_data, bytes):
            raise TypeError(f"file_data must be bytes or base64 string, got {type(file_data)}")

        if len(file_data) == 0:
            raise ValueError("Empty file data after processing")

        logger.info(f"Processing file with Docling DocumentConverter: {filename}, size: {len(file_data)} bytes")
        logger.debug(f"Docling options: {options}")

        # Parse options for DocumentConverter
        docling_options = self._parse_docling_options(options)

        # Process file using temporary file (Docling requirement)
        try:
            with tempfile.NamedTemporaryFile(suffix=f"_{filename}", delete=False) as tmp:
                tmp.write(file_data)
                tmp.flush()
                tmp_path = Path(tmp.name)
        except OSError as e:
            logger.error(f"Failed to create temporary file for {filename}: {e}")
            raise RuntimeError(f"Failed to create temporary file: {e}") from e

        try:
            # Convert using the DocumentConverter - be specific about conversion errors
            if not self.converter:
                raise RuntimeError("DocumentConverter not initialized")

            result = self.converter.convert(tmp_path)
            if not result or not hasattr(result, "document"):
                raise RuntimeError(f"Invalid conversion result for {filename}")

            # Determine output format
            format_type = options.get("format", "markdown")

            # Export content based on requested format using helper method
            try:
                content = self._export_document_content(result.document, format_type)
            except (AttributeError, RuntimeError) as e:
                logger.error(f"Export failed for {filename}: {e}")
                raise RuntimeError(f"Export failed for {filename}: {e}") from e

            processing_time = time.time() - start_time

            # Handle chunking if strategy provided
            chunks = None
            embeddings = None
            chunk_count = 0

            if chunking_strategy:
                chunks, chunk_count = self._create_docling_chunks(result.document, chunking_strategy, format_type)

                # Generate embeddings if requested
                if include_embeddings and chunks:
                    # TODO: Implement embedding generation
                    # This would require access to an embedding model
                    logger.warning("Embedding generation not yet implemented for Docling provider")

            # Extract metadata from Docling result - handle each piece safely
            # Safely extract document statistics using helper function
            document_stats = self._extract_document_stats(result.document, ["pages", "tables", "figures"])

            # Build metadata dict safely
            metadata = {
                **document_stats,  # Unpack pages, tables, figures counts
                "format": format_type,
                "processor": "docling",
                "processing_time_seconds": processing_time,
                "content_length": len(content),
                "filename": filename,
                "file_size_bytes": len(file_data),
                "converter_options": docling_options,
                "chunk_count": chunk_count,
                "include_embeddings": include_embeddings,
                "docling_version": getattr(result, "version", "unknown"),
            }

            # Handle optional metadata that might fail
            if chunking_strategy:
                try:
                    metadata["chunking_strategy"] = chunking_strategy.model_dump()
                except (AttributeError, TypeError):
                    logger.warning(f"Failed to serialize chunking strategy for {filename}")
                    metadata["chunking_strategy"] = None
            else:
                metadata["chunking_strategy"] = None

            # Create ProcessedContent with validated inputs
            processed = ProcessedContent(
                content=content,
                chunks=chunks,
                embeddings=embeddings,
                metadata=metadata,
            )

            logger.info(
                f"Docling processing completed: {document_stats['pages']} pages, "
                f"{document_stats['tables']} tables, {chunk_count} chunks, {processing_time:.2f}s"
            )
            return processed

        except ImportError as e:
            logger.error(f"Missing dependencies for processing {filename}: {e}")
            raise RuntimeError(f"Missing dependencies for processing: {e}") from e
        except OSError as e:
            logger.error(f"File I/O error processing {filename}: {e}")
            raise RuntimeError(f"File I/O error: {e}") from e
        except RuntimeError:
            # Re-raise RuntimeError as-is (already has good error messages)
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing {filename}: {e}")
            raise RuntimeError(f"Unexpected error processing file: {e}") from e

        finally:
            # Clean up temporary file
            try:
                tmp_path.unlink()
            except OSError as cleanup_error:
                logger.warning(f"Failed to cleanup temp file {tmp_path}: {cleanup_error}")
