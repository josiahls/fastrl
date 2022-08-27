# AUTOGENERATED! DO NOT EDIT! File to edit: ../../../nbs/01_DataPipes/01b_pipes.map.demux.ipynb.

# %% auto 0
__all__ = ['T_co', 'DemultiplexerMapDataPipe']

# %% ../../../nbs/01_DataPipes/01b_pipes.map.demux.ipynb 3
# Python native modules
import os
from inspect import isfunction,ismethod
from typing import Callable, Dict, Iterable, Optional, TypeVar
# Third party libs
from fastcore.all import *
from fastai.torch_basics import *
from torch.utils.data.datapipes.utils.common import _check_unpickable_fn
import torchdata.datapipes as dp
from torchdata.datapipes import functional_datapipe
from torchdata.dataloader2.graph import find_dps,DataPipeGraph,Type,DataPipe,MapDataPipe,IterDataPipe
from torchdata.dataloader2.dataloader2 import DataLoader2
# Local modules

# %% ../../../nbs/01_DataPipes/01b_pipes.map.demux.ipynb 5
T_co = TypeVar("T_co", covariant=True)

@functional_datapipe("demux")
class DemultiplexerMapDataPipe(MapDataPipe[T_co]):
    def __new__(cls, datapipe: MapDataPipe, num_instances: int, classifier_fn: Callable, drop_none: bool = False,
                source_index: Optional[Iterable] = None):
        if num_instances < 1:
            raise ValueError(f"Expected `num_instances` larger than 0, but {num_instances} is found")
        _check_unpickable_fn(classifier_fn)
        container = _DemultiplexerMapDataPipe(datapipe, num_instances, classifier_fn, drop_none, source_index)
        return [_DemultiplexerChildMapDataPipe(container, i) for i in range(num_instances)]

class _DemultiplexerMapDataPipe(MapDataPipe[T_co]):
    def __init__(
        self,
        datapipe: MapDataPipe[T_co],
        num_instances: int,
        classifier_fn: Callable[[T_co], Optional[int]],
        drop_none: bool,
        source_index: Optional[Iterable],
    ):
        self.main_datapipe = datapipe
        self.num_instances = num_instances
        self.classifier_fn = classifier_fn
        self.drop_none = drop_none
        self.iterator = None
        self.exhausted = False  # Once we iterate through `main_datapipe` once, we know all the index mapping
        self.index_mapping = [[] for _ in range(num_instances)]
        self.source_index = source_index  # if None, assume `main_datapipe` 0-index

    def _classify_next(self):
        if self.source_index is None:
            self.source_index = range(len(self.main_datapipe))
        if self.iterator is None:
            self.iterator = iter(self.source_index)
        try:
            next_source_idx = next(self.iterator)
        except StopIteration:
            self.exhausted = True
            return
        value = self.main_datapipe[next_source_idx]
        classification = self.classifier_fn(value)
        if classification is None and self.drop_none:
            self._classify_next()
        else:
            self.index_mapping[classification].append(value)

    def classify_all(self):
        while not self.exhausted:
            self._classify_next()

    def get_value(self, instance_id: int, index: int) -> T_co:
        while not self.exhausted and len(self.index_mapping[instance_id]) <= index:
            self._classify_next()
        if len(self.index_mapping[instance_id]) > index:
            return self.index_mapping[instance_id][index]
        raise RuntimeError("Index is out of bound.")

    def __len__(self):
        return len(self.main_datapipe)

class _DemultiplexerChildMapDataPipe(MapDataPipe[T_co]):
    def __init__(self, main_datapipe: _DemultiplexerMapDataPipe, instance_id: int):
        self.main_datapipe: _DemultiplexerMapDataPipe = main_datapipe
        self.instance_id = instance_id

    def __getitem__(self, index: int):
        return self.main_datapipe.get_value(self.instance_id, index)

    def __len__(self):
        self.main_datapipe.classify_all()  # You have to read through the entirety of main_datapipe to know `len`
        return len(self.main_datapipe.index_mapping[self.instance_id])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
            
DemultiplexerMapDataPipe.__doc__ = """Splits the input DataPipe into multiple child DataPipes, using the given
    classification function (functional name: ``demux``). A list of the child DataPipes is returned from this operation.
"""
