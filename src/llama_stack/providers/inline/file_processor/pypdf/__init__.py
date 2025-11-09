# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.providers.datatypes import Api

from .config import PyPDFConfig
from .pypdf import PyPDFFileProcessorImpl

__all__ = ["PyPDFConfig", "PyPDFFileProcessorImpl"]


async def get_provider_impl(config: PyPDFConfig, deps: dict[Api, Any]):
    return PyPDFFileProcessorImpl(config)
