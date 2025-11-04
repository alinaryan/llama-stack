# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.providers.datatypes import Api, InlineProviderSpec, ProviderSpec


def available_providers() -> list[ProviderSpec]:
    return [
        # PyPDF - Default provider for backward compatibility
        InlineProviderSpec(
            api=Api.file_processors,
            provider_type="inline::pypdf",
            pip_packages=["pypdf"],
            module="llama_stack.providers.inline.file_processors.pypdf",
            config_class="llama_stack.providers.inline.file_processors.pypdf.PyPDFConfig",
            description="Simple PDF text extraction using PyPDF library. Default processor for backward compatibility.",
        ),
        # Docling - Advanced inline processing with native chunking
        InlineProviderSpec(
            api=Api.file_processors,
            provider_type="inline::docling",
            pip_packages=["docling", "torch", "torchvision", "docling-core[chunking]"],
            module="llama_stack.providers.inline.file_processors.docling",
            config_class="llama_stack.providers.inline.file_processors.docling.DoclingConfig",
            description="Advanced document processing using Docling with table/figure extraction and native chunking",
        ),
    ]
