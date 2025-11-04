# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from llama_stack.apis.vector_io.vector_io import VectorStoreChunkingStrategy
from llama_stack.apis.version import LLAMA_STACK_API_V1
from llama_stack.core.telemetry.trace_protocol import trace_protocol
from llama_stack.schema_utils import json_schema_type, webmethod


@json_schema_type
class ProcessedContent(BaseModel):
    """
    Result of file processing containing extracted content, optional chunks, and metadata.

    :param content: Extracted text content from the file
    :param chunks: Optional text chunks when chunking strategy is applied
    :param embeddings: Optional embeddings for chunks when include_embeddings=True
    :param metadata: Processing metadata including processor info, timing, chunking details, etc.
    """

    content: str = Field(..., description="Extracted text content from file")
    chunks: list[str] | None = Field(default=None, description="Optional text chunks when chunking is applied")
    embeddings: list[list[float]] | None = Field(default=None, description="Optional embeddings for chunks")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Processing metadata including chunking details")


@runtime_checkable
@trace_protocol
class FileProcessors(Protocol):
    """File Processors

    This API provides document processing capabilities for extracting text content
    from various file formats, with optional chunking and embedding generation.

    The API supports:
    - File format conversion (PDFs, Word docs, images, etc.)
    - Provider-specific chunking strategies leveraging document understanding
    - Optional embedding generation for chunks
    - Multiple processing providers that can be configured and swapped
    - Integration with vector store ingestion pipelines
    """

    @webmethod(route="/file-processors/process", method="POST", level=LLAMA_STACK_API_V1)
    async def process_file(
        self,
        file_data: bytes,
        filename: str,
        options: dict[str, Any] | None = None,
        chunking_strategy: VectorStoreChunkingStrategy | None = None,
        include_embeddings: bool = False,
    ) -> ProcessedContent:
        """Process a file with optional provider-specific chunking and embedding generation.

        Each provider can implement intelligent chunking based on their document understanding:
        - PyPDF: Token-based chunking with overlap
        - Docling: Section-aware chunking respecting document structure
        - Unstructured.io: Element-based chunking (paragraphs, tables, etc.)

        :param file_data: The raw file data as bytes
        :param filename: Name of the file (used for format detection and processing hints)
        :param options: Optional processing options including:
                       - format: Output format ("markdown", "html", "json", "text")
                       - extract_tables: Enable table extraction (bool)
                       - extract_figures: Enable figure extraction (bool)
                       - ocr_enabled: Enable OCR for images (bool)
                       - processor_specific options
        :param chunking_strategy: Optional chunking strategy (VectorStoreChunkingStrategy)
                                 If provided, provider will chunk content according to strategy
        :param include_embeddings: Whether to generate embeddings for chunks (requires chunking_strategy)
        :returns: ProcessedContent with converted content, optional chunks, embeddings, and processing metadata
        """
