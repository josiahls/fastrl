# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/02_DataLoading/02g_data.block.ipynb.

# %% auto 0
__all__ = ['TransformBlock', 'DataBlock']

# %% ../../nbs/02_DataLoading/02g_data.block.ipynb 3
# Python native modules
import os
from typing import Any,Callable,Generator
from inspect import isfunction,ismethod
# Third party libs
from fastcore.all import *
from torch.utils.data.dataloader_experimental import DataLoader2
from torch.utils.data.datapipes._typing import _DataPipeMeta, _IterDataPipeMeta
from fastai.torch_core import *
from fastai.data.transforms import *
import torchdata.datapipes as dp
from collections import deque
from fastai.imports import *
# Local modules
from ..pipes.core import *
from ..core import *

# %% ../../nbs/02_DataLoading/02g_data.block.ipynb 7
class TransformBlock():
    "A basic wrapper that links defaults transforms for the data block API"
    def __init__(self, 
        # A function that initializes a datapipeline and returns a datapipe.
        # Minimum must support:
        #
        #     `pipe_fn(source, bs, n)`
        #
        # Where:
        #   - `source` is the data to be input into the datapipes
        #   - `bs` is the batch size of the returned data
        #   - `n` is the number of iterations to make through the datapipes per epoch                 
        pipe_fn:Callable[[Any,int,int],_DataPipeMeta]=None, 
        # One or more `Transform`s for converting types. These will be re-called if workers!=0 for the dataloader.
        type_tfms:list=None, 
        item_tfms:list=None, # `ItemTransform`s, applied per peice of data (not batch)
        batch_tfms:list=None, # `Transform`s applied over a batch of data
        # `Callback`s for use in dataloaders. These usually augment a preexisting pipeline in some way
        cbs:list=None,
        pipe_fn_kwargs:dict=None, # Additional arguments to be passed to `pipe_fn`
        dl_type:DataLoader2=None, # Task specific `TfmdDL`, defaults to `TfmdDL`
        dls_kwargs:dict=None, # Additional arguments to be passed to `DataLoaders`
    ):
        self.type_tfms                   = L(type_tfms)
        self.item_tfms                   = L(item_tfms)
        self.batch_tfms                  = L(batch_tfms)
        self.pipe_fn,self.pipe_fn_kwargs = pipe_fn,ifnone(pipe_fn_kwargs,{})
        self.cbs                         = L(cbs)
        self.dl_type,self.dls_kwargs     = dl_type,ifnone(dls_kwargs,{})

# %% ../../nbs/02_DataLoading/02g_data.block.ipynb 9
class DataBlock(object):
    def __init__(
        self,
        # Each transform block will have its own dataloader. 
        blocks:List[TransformBlock]=None, 
    ):
        store_attr(but='blocks')
        self.blocks = L(blocks)

    def datapipes(
        self,
        source:Any,
        bs=1,
        n=None,
        return_blocks:bool=False
    ) -> Generator[Union[Tuple[_DataPipeMeta,TransformBlock],_DataPipeMeta],None,None]:
        for b in self.blocks:
            pipe = b.pipe_fn(source,bs=bs,n=n,**b.pipe_fn_kwargs)
            yield (pipe,b) if return_blocks else pipe
        
    def dataloaders(
        self,
        source:Any,
        bs=1,
        n=None,
        num_workers=0,
        **kwargs
    ) -> Generator[DataLoader2,None,None]:
        for pipe,block in self.datapipes(source,bs=bs,n=n,return_blocks=True,**kwargs):
            yield block.dl_type(pipe,num_workers=num_workers,**merge(kwargs,block.dls_kwargs))
