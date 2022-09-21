# AUTOGENERATED! DO NOT EDIT! File to edit: ../../../nbs/01_DataPipes/01c_pipes.map.mux.ipynb.

# %% auto 0
__all__ = ['T_co', 'MultiplexerMapDataPipe']

# %% ../../../nbs/01_DataPipes/01c_pipes.map.mux.ipynb 3
# Python native modules
import os
from inspect import isfunction,ismethod
from itertools import chain, zip_longest
from typing import Callable, Dict, Iterable, Optional, TypeVar
# Third party libs
from fastcore.all import *
# from fastrl.torch_core import *
# from torch.utils.data.dataloader import DataLoader as OrgDataLoader
import torchdata.datapipes as dp
from torchdata.datapipes import functional_datapipe
from torchdata.dataloader2.graph import find_dps,DataPipeGraph,Type,DataPipe,MapDataPipe,IterDataPipe
from torchdata.dataloader2.dataloader2 import DataLoader2
# Local modules

# %% ../../../nbs/01_DataPipes/01c_pipes.map.mux.ipynb 5
T_co = TypeVar("T_co", covariant=True)

@functional_datapipe("mux")
class MultiplexerMapDataPipe(MapDataPipe[T_co]):
    def __init__(self, *datapipes, dp_index_map: Optional[Dict[MapDataPipe, Iterable]] = None):
        self.datapipes = datapipes
        self.dp_index_map = dp_index_map if dp_index_map else {}
        self.length: Optional[int] = None
        self.index_map = {}
        # Create a generator that yields (index, (dp_num, old_index)) in sequentially order.
        indices = (self._add_dp_num(i, dp) for i, dp in enumerate(datapipes))
        dp_id_and_key_tuples = chain.from_iterable(zip_longest(*indices))
        self.key_gen = enumerate(e for e in dp_id_and_key_tuples if e is not None)

    def _add_dp_num(self, dp_num: int, dp: MapDataPipe):
        # Assume 0-index for all DataPipes unless alternate indices are defined in `self.dp_index_map`
        dp_indices = self.dp_index_map[dp] if dp in self.dp_index_map else range(len(dp))
        for idx in dp_indices:
            yield dp_num, idx

    def __getitem__(self, index):
        if 0 <= index < len(self):
            if index in self.index_map:
                dp_num, old_key = self.index_map[index]
            else:
                curr_key = -1
                while curr_key < index:
                    curr_key, dp_num_key_tuple = next(self.key_gen)
                    dp_num, old_key = dp_num_key_tuple
                self.index_map[index] = dp_num, old_key
            try:
                return self.datapipes[dp_num][old_key]
            except KeyError:
                raise RuntimeError(
                    f"Incorrect key is given to MapDataPipe {dp_num} in Multiplexer, likely because"
                    f"that DataPipe is not 0-index but alternate indices are not given."
                )
        raise RuntimeError(f"Index {index} is out of bound for Multiplexer.")

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        if self.length is None:
            self.length = 0
            for dp in self.datapipes:
                self.length += len(dp)
        return self.length
    
MultiplexerMapDataPipe.__doc__ = """Yields one element at a time from each of the input Iterable DataPipes (functional name: ``mux``). As in,
    one element from the 1st input DataPipe, then one element from the 2nd DataPipe in the next iteration,
    and so on. It ends when the shortest input DataPipe is exhausted.
"""
