# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, Field

from llama_stack.apis.vector_io.vector_io import VectorStoreChunkingStrategyStaticConfig


class PyPDFConfig(BaseModel):
    timeout_seconds: int = Field(default=30, ge=1, le=300, description="Processing timeout in seconds")
    max_file_size_mb: int = Field(default=50, ge=1, le=500, description="Maximum file size in MB")

    # Chunking configuration for RAG/vector store integration
    default_chunk_size_tokens: int = Field(
        default=VectorStoreChunkingStrategyStaticConfig.model_fields["max_chunk_size_tokens"].default,
        ge=100,
        le=4096,
        description="Default chunk size in tokens when no chunking strategy is specified",
    )
    default_chunk_overlap_tokens: int = Field(
        default=VectorStoreChunkingStrategyStaticConfig.model_fields["chunk_overlap_tokens"].default,
        ge=0,
        le=2048,
        description="Default chunk overlap in tokens when no chunking strategy is specified",
    )

    # PDF processing options
    extract_metadata: bool = Field(
        default=True, description="Extract PDF metadata (title, author, creation date, etc.)"
    )

    @staticmethod
    def sample_run_config(**kwargs):
        return {
            "timeout_seconds": 30,
            "max_file_size_mb": 50,
            "default_chunk_size_tokens": VectorStoreChunkingStrategyStaticConfig.model_fields[
                "max_chunk_size_tokens"
            ].default,
            "default_chunk_overlap_tokens": VectorStoreChunkingStrategyStaticConfig.model_fields[
                "chunk_overlap_tokens"
            ].default,
            "extract_metadata": True,
        }
