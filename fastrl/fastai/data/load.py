# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02d_fastai.data.load.ipynb (unless otherwise specified).

__all__ = ['TypeTransformLoop', 'ItemTransformLoop', 'BatchTransformLoop']

# Cell
# Python native modules
import os
from typing import Callable
# Third party libs
from fastcore.all import *
import torchdata.datapipes as dp
from torch.utils.data.dataloader_experimental import DataLoader2
from torch.utils.data.graph import traverse
# Local modules
from ...pipes.core import *
from ...pipes.map.mux import *
from ...pipes.map.demux import *

# Cell
class TypeTransformLoop(dp.map.MapDataPipe):
    def __init__(self,datapipe, type_tfms):
        self.type_tfms,self.datapipe = Pipeline(type_tfms),datapipe

    def __getitem__(self, index):
        data = self.datapipe[index]
        return self.type_tfms(data)

    def __len__(self): return len(self.datapipe)

class ItemTransformLoop(dp.iter.IterDataPipe):
    def __init__(self,source_datapipe, item_tfms:List[Callable]):
        self.item_tfms,self.source_datapipe = Pipeline(item_tfms),source_datapipe

    def __iter__(self):
        for data in self.source_datapipe:
            yield self.item_tfms(data)

class BatchTransformLoop(dp.iter.IterDataPipe):
    def __init__(self,source_datapipe, batch_tfms):
        self.batch_tfms,self.source_datapipe = Pipeline(batch_tfms),source_datapipe

    def __iter__(self):
        for data in self.source_datapipe:
            yield self.batch_tfms(data)