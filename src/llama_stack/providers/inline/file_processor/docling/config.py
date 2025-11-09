# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, Field

from llama_stack.apis.vector_io.vector_io import VectorStoreChunkingStrategyStaticConfig


class DoclingConfig(BaseModel):
    timeout_seconds: int = Field(default=120, ge=1, le=600, description="Processing timeout in seconds")
    max_file_size_mb: int = Field(default=100, ge=1, le=1000, description="Maximum file size in MB")
    model_cache_dir: str | None = Field(default=None, description="Directory to cache Docling models")
    enable_gpu: bool = Field(default=False, description="Enable GPU acceleration if available")
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

    @staticmethod
    def sample_run_config(**kwargs):
        return {
            "timeout_seconds": 120,
            "max_file_size_mb": 100,
            "model_cache_dir": None,
            "enable_gpu": False,
            "default_chunk_size_tokens": VectorStoreChunkingStrategyStaticConfig.model_fields[
                "max_chunk_size_tokens"
            ].default,
            "default_chunk_overlap_tokens": VectorStoreChunkingStrategyStaticConfig.model_fields[
                "chunk_overlap_tokens"
            ].default,
        }
