# AUTOGENERATED! DO NOT EDIT! File to edit: ../../../nbs/01_DataPipes/02a_pipes.iter.cacheholder.ipynb.

# %% auto 0
__all__ = ['T_co', 'PickleableInMemoryCacheHolderIterDataPipe']

# %% ../../../nbs/01_DataPipes/02a_pipes.iter.cacheholder.ipynb 2
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Python native modules
import hashlib
import inspect
import os.path
import sys
import time
import uuid
import warnings
from enum import IntEnum

from collections import deque
from functools import partial
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Tuple, TypeVar
# Third party libs
try:
    import portalocker
except ImportError:
    portalocker = None

from torch.utils.data.datapipes.utils.common import _check_unpickable_fn, DILL_AVAILABLE

from torch.utils.data.graph import traverse_dps
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe
# Local modules


# %% ../../../nbs/01_DataPipes/02a_pipes.iter.cacheholder.ipynb 4
if DILL_AVAILABLE:
    import dill

    dill.extend(use_dill=False)

T_co = TypeVar("T_co", covariant=True)

@functional_datapipe("pickleable_in_memory_cache")
class PickleableInMemoryCacheHolderIterDataPipe(IterDataPipe[T_co]):
    r"""
    Stores elements from the source DataPipe in memory, up to a size limit
    if specified (functional name: ``in_memory_cache``). This cache is FIFO - once the cache is full,
    further elements will not be added to the cache until the previous ones are yielded and popped off from the cache.

    Args:
        source_dp: source DataPipe from which elements are read and stored in memory
        size: The maximum size (in megabytes) that this DataPipe can hold in memory. This defaults to unlimited.

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> source_dp = IterableWrapper(range(10))
        >>> cache_dp = source_dp.pickleable_in_memory_cache(size=5)
        >>> list(cache_dp)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    size: Optional[int] = None
    idx: int

    def __init__(self, source_dp: IterDataPipe[T_co], size: Optional[int] = None) -> None:
        self.source_dp: IterDataPipe[T_co] = source_dp
        # cache size in MB
        if size is not None:
            self.size = size * 1024 * 1024
        self.cache: Optional[Deque] = None
        self.idx: int = 0

    def __getstate__(self):
        state = (
            self.source_dp,
            self.size
        )
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        (
            self.source_dp,
            self.size
        ) = state
        self.cache: Optional[Deque] = None
        self.idx: int = 0

    def __iter__(self) -> Iterator[T_co]:
        if self.cache:
            if self.idx > 0:
                for idx, data in enumerate(self.source_dp):
                    if idx < self.idx:
                        yield data
                    else:
                        break
            yield from self.cache
        else:
            # Local cache
            cache: Deque = deque()
            idx = 0
            for data in self.source_dp:
                cache.append(data)
                # Cache reaches limit
                if self.size is not None and sys.getsizeof(cache) > self.size:
                    cache.popleft()
                    idx += 1
                yield data
            self.cache = cache
            self.idx = idx

    def __len__(self) -> int:
        try:
            return len(self.source_dp)
        except TypeError:
            if self.cache:
                return self.idx + len(self.cache)
            else:
                raise TypeError(f"{type(self).__name__} instance doesn't have valid length until the cache is loaded.")

