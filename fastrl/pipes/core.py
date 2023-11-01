# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/01_DataPipes/01a_pipes.core.ipynb.

# %% auto 0
__all__ = ['find_dps', 'find_dp', 'DataPipeAugmentationFn', 'apply_dp_augmentation_fns']

# %% ../../nbs/01_DataPipes/01a_pipes.core.ipynb 2
# Python native modules
import os
import logging
import inspect
from typing import Callable,Union,TypeVar,Optional,Type,List,Tuple
# Third party libs
import torchdata.datapipes as dp
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.map import MapDataPipe
from torchdata.dataloader2.graph import DataPipe, DataPipeGraph,find_dps,traverse_dps,list_dps
# Local modules

# %% ../../nbs/01_DataPipes/01a_pipes.core.ipynb 5
def find_dps(
        graph: DataPipeGraph, 
        dp_type: Type[DataPipe],
        include_subclasses:bool=False
    ) -> List[DataPipe]:
    r"""
    Given the graph of DataPipe generated by ``traverse`` function, return DataPipe
    instances with the provided DataPipe type.
    """
    dps: List[DataPipe] = []

    def helper(g) -> None:  # pyre-ignore
        for _, (dp, src_graph) in g.items():
            if include_subclasses and issubclass(type(dp),dp_type):
                dps.append(dp)
            elif type(dp) is dp_type:  # Please not use `isinstance`, there is a bug.
                dps.append(dp)
            helper(src_graph)

    helper(graph)

    return dps

# %% ../../nbs/01_DataPipes/01a_pipes.core.ipynb 6
def find_dp(
        # A graph created from the `traverse` function
        graph: DataPipeGraph, 
        # 
        dp_type: Type[DataPipe],
        include_subclasses:bool=False
    ) -> DataPipe:
    pipes = find_dps(graph,dp_type,include_subclasses)
    if len(pipes)==1: return pipes[0]
    elif len(pipes)>1:
        found_ids = set([id(pipe) for pipe in pipes])
        if len(found_ids)>1:
            logging.warn("""There are %s pipes of type %s. If this is intended, 
                     please use `find_dps` directly. Returning first instance.""",len(pipes),dp_type)
        return pipes[0]
    else:
        raise LookupError(f'Unable to find {dp_type} starting at {graph}')
    
find_dp.__doc__ = "Returns a single `DataPipe` as opposed to `find_dps`.\n"+find_dps.__doc__

# %% ../../nbs/01_DataPipes/01a_pipes.core.ipynb 19
class DataPipeAugmentationFn(Callable[[DataPipe],Optional[DataPipe]]):...

DataPipeAugmentationFn.__doc__ = f"""`DataPipeAugmentationFn` must take in a `DataPipe` and either output a `DataPipe` or `None`. This function should perform some operation on the graph
such as replacing, removing, inserting `DataPipe`'s and `DataGraph`s. Below is an example that replaces a `dp.iter.Batcher` datapipe with a `dp.iter.Filter`"""

# %% ../../nbs/01_DataPipes/01a_pipes.core.ipynb 23
def apply_dp_augmentation_fns(
        pipe:DataPipe,
        dp_augmentation_fns:Optional[Tuple[DataPipeAugmentationFn]],
        debug:bool=False
    ) -> DataPipe:
    "Given a `pipe`, run `dp_augmentation_fns` other the pipeline"
    if dp_augmentation_fns is None: return pipe
    for fn in dp_augmentation_fns:
        if debug: print(f'Running fn: {fn} given current pipe: \n\t{traverse_dps(pipe)}')
        result = fn(pipe)
        if result is not None: pipe = result
    return pipe
