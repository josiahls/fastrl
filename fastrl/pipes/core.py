# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/01_DataPipes/01a_pipes.core.ipynb.

# %% auto 0
__all__ = ['find_dp', 'insert_dp', 'PassThroughIterPipe', 'TypeTransformLoop', 'ItemTransformLoop', 'BatchTransformLoop']

# %% ../../nbs/01_DataPipes/01a_pipes.core.ipynb 3
# Python native modules
import os
import logging
import inspect
from typing import Callable,Union
# Third party libs
from fastcore.all import *
import torchdata.datapipes as dp
from torchdata.datapipes import functional_datapipe
from torchdata.dataloader2.graph import find_dps,DataPipeGraph,Type,DataPipe,traverse,_assign_attr,replace_dp
# Local modules

# %% ../../nbs/01_DataPipes/01a_pipes.core.ipynb 5
def find_dp(graph: DataPipeGraph, dp_type: Type[DataPipe]) -> DataPipe:
    pipes = find_dps(graph,dp_type)
    if len(pipes)==1: return pipes[0]
    elif len(pipes)>1:
        found_ids = set([id(pipe) for pipe in pipes])
        if len(found_ids)>1:
            warn(f"""There are {len(pipes)} pipes of type {dp_type}. If this is intended, 
                     please use `find_dps` directly. Returning first instance.""")
        return pipes[0]
    else:
        raise LookupError(f'Unable to find {dp_type} starting at {graph}')
    
find_dp.__doc__ = "Returns a single `DataPipe` as opposed to `find_dps`.\n"+find_dps.__doc__

# %% ../../nbs/01_DataPipes/01a_pipes.core.ipynb 6
def _insert_dp(recv_dp, send_graph: DataPipeGraph, old_dp: DataPipe, new_dp: DataPipe) -> None:
    old_dp_id = id(old_dp)
    for send_id in send_graph:
        if send_id == old_dp_id:
            _assign_attr(recv_dp, old_dp, new_dp, inner_dp=True)
            
            # Replace the last datapipe in new_dp with the old_dp
            final_datapipe = find_dp(traverse(new_dp),PassThroughIterPipe)
            
            _assign_attr(new_dp, final_datapipe, old_dp, inner_dp=True)
            # new_dp.source_datapipe
        else:
            send_dp, sub_send_graph = send_graph[send_id]
            _insert_dp(send_dp, sub_send_graph, old_dp, new_dp)

# %% ../../nbs/01_DataPipes/01a_pipes.core.ipynb 7
def insert_dp(graph: DataPipeGraph, on_datapipe: DataPipe, insert_datapipe: DataPipe) -> DataPipeGraph:
    r"""
    Given the graph of DataPipe generated by ``traverse`` function and the ``on_datapipe`` DataPipe to be reconnected and
    the new ``insert_datapipe`` DataPipe to be inserted after ``on_datapipe``, 
    return the new graph of DataPipe.
    """
    assert len(graph) == 1

    # Check if `on_datapipe` is that the head of the graph
    # If so, we `insert_datapipe`
    if id(on_datapipe) in graph: 
        graph = traverse(insert_datapipe, only_datapipe=True)

    final_datapipe = list(graph.values())[0][0]
    
    for recv_dp, send_graph in graph.values():
        _insert_dp(recv_dp, send_graph, on_datapipe, insert_datapipe)

    return traverse(final_datapipe, only_datapipe=True)


# %% ../../nbs/01_DataPipes/01a_pipes.core.ipynb 9
class PassThroughIterPipe(dp.iter.IterDataPipe):
    def __init__(self,source_datapipe): self.source_datapipe = source_datapipe
    def __iter__(self): return (o for o in self.source_datapipe)

# %% ../../nbs/01_DataPipes/01a_pipes.core.ipynb 14
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
