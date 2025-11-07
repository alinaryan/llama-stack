# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import io
import time
from typing import Any

from llama_stack.apis.file_processors import FileProcessors, ProcessedContent
from llama_stack.apis.vector_io.vector_io import VectorStoreChunkingStrategy
from llama_stack.log import get_logger
from llama_stack.providers.utils.memory.vector_store import make_overlapped_chunks

from .config import PyPDFConfig

logger = get_logger(__name__)


class PyPDFFileProcessorImpl(FileProcessors):
    def __init__(self, config: PyPDFConfig):
        self.config = config
        logger.info("PyPDF processor initialized")

    async def process_file(
        self,
        file_data: bytes,
        filename: str,
        options: dict[str, Any] | None = None,
        chunking_strategy: VectorStoreChunkingStrategy | None = None,
        include_embeddings: bool = False,
    ) -> ProcessedContent:
        start_time = time.time()
        logger.info(f"Processing PDF file: {filename}, size: {len(file_data)} bytes")

        try:
            # Import here to avoid dependency issues if pypdf not installed
            from pypdf import PdfReader  # type: ignore[import-not-found] # pypdf is optional dependency

            # Migrate existing 3-line logic from vector_store.py
            pdf_reader = PdfReader(io.BytesIO(file_data))
            text = "\n".join([page.extract_text() for page in pdf_reader.pages])

            processing_time = time.time() - start_time

            # Handle chunking if strategy provided
            chunks = None
            embeddings = None
            chunk_count = 0

            if chunking_strategy:
                # Use existing chunking logic from vector store
                from llama_stack.apis.vector_io.vector_io import VectorStoreChunkingStrategyStatic

                if isinstance(chunking_strategy, VectorStoreChunkingStrategyStatic):
                    max_chunk_size = chunking_strategy.static.max_chunk_size_tokens
                    chunk_overlap = chunking_strategy.static.chunk_overlap_tokens
                else:
                    # Default for auto strategy
                    max_chunk_size = 800
                    chunk_overlap = 400

                # Create document metadata for chunking
                document_metadata = {
                    "filename": filename,
                    "processor": "pypdf",
                    "pages": len(pdf_reader.pages),
                }

                # Generate chunks using existing utility
                chunk_objects = make_overlapped_chunks(
                    document_id=filename,
                    text=text,
                    window_len=max_chunk_size,
                    overlap_len=chunk_overlap,
                    metadata=document_metadata,
                )

                # Use the chunk objects directly (they are already in the correct Chunk format)
                chunks = chunk_objects
                chunk_count = len(chunks)

                # Generate embeddings if requested
                if include_embeddings and chunks:
                    # TODO: Implement embedding generation
                    # This would require access to an embedding model
                    logger.warning("Embedding generation not yet implemented for PyPDF provider")

            result = ProcessedContent(
                content=text,
                chunks=chunks,
                embeddings=embeddings,
                metadata={
                    "pages": len(pdf_reader.pages),
                    "processor": "pypdf",
                    "processing_time_seconds": processing_time,
                    "content_length": len(text),
                    "filename": filename,
                    "file_size_bytes": len(file_data),
                    "chunking_strategy": chunking_strategy.model_dump() if chunking_strategy else None,
                    "chunk_count": chunk_count,
                    "include_embeddings": include_embeddings,
                },
            )

            logger.info(
                f"PyPDF processing completed: {len(pdf_reader.pages)} pages, {len(text)} chars, "
                f"{chunk_count} chunks, {processing_time:.2f}s"
            )
            return result

        except ImportError as e:
            logger.error("PyPDF not installed. Run: pip install pypdf")
            raise RuntimeError("PyPDF not installed. Run: pip install pypdf") from e
        except Exception as e:
            logger.error(f"PyPDF processing failed for {filename}: {str(e)}")
            raise RuntimeError(f"PyPDF processing failed: {str(e)}") from e
