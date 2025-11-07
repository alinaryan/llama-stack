# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import cast

from llama_stack.providers.datatypes import Api, InlineProviderSpec, ProviderSpec

# We provide two versions of the Docling provider so that distributions can package the appropriate version of torch.
# The CPU version is used for distributions that don't have GPU support -- they result in smaller container images.
docling_def = dict(
    api=Api.file_processors,
    pip_packages=["docling", "docling-core[chunking]"],
    module="llama_stack.providers.inline.file_processors.docling",
    config_class="llama_stack.providers.inline.file_processors.docling.DoclingConfig",
    description="Advanced document processing using Docling with table/figure extraction and native chunking",
)


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
        # Docling CPU - Advanced inline processing optimized for CPU-only environments
        InlineProviderSpec(
            **{  # type: ignore
                **docling_def,
                "provider_type": "inline::docling-cpu",
                "pip_packages": (
                    cast(list[str], docling_def["pip_packages"])
                    + ["torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu"]
                ),
                "description": "Advanced document processing using Docling with CPU-optimized PyTorch for smaller deployments",
            },
        ),
        # Docling GPU - Advanced inline processing with GPU acceleration support
        InlineProviderSpec(
            **{  # type: ignore
                **docling_def,
                "provider_type": "inline::docling-gpu",
                "pip_packages": (cast(list[str], docling_def["pip_packages"]) + ["torch", "torchvision"]),
                "description": "Advanced document processing using Docling with GPU acceleration for high-performance processing",
            },
        ),
    ]
