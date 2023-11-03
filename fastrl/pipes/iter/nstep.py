# AUTOGENERATED! DO NOT EDIT! File to edit: ../../../nbs/01_DataPipes/01c_pipes.iter.nstep.ipynb.

# %% auto 0
__all__ = ['NStepper', 'NStepFlattener', 'n_steps_expected']

# %% ../../../nbs/01_DataPipes/01c_pipes.iter.nstep.ipynb 2
# Python native modules
import os
from typing import Type, Dict, Union, Tuple
import typing
import warnings
# Third party libs
from fastcore.all import add_docs
import torchdata.datapipes as dp
from torchdata.dataloader2.graph import find_dps,DataPipeGraph,DataPipe
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.map import MapDataPipe
# Local modules
from ...core import StepTypes

# %% ../../../nbs/01_DataPipes/01c_pipes.iter.nstep.ipynb 4
class NStepper(IterDataPipe):
    def __init__(
            self, 
            # The datapipe we are extracting from must produce `StepType.types`
            source_datapipe:IterDataPipe[Union[StepTypes.types]], 
            # Maximum number of steps to produce per yield as a tuple. This is the *max* number
            # and may be less if for example we are yielding terminal states.
            # Default produces single steps
            n:int=1
        ) -> None:
        self.source_datapipe:IterDataPipe[StepTypes.types] = source_datapipe
        self.n:int = n
        self.env_buffer:Dict = {}
        
    def __iter__(self) -> StepTypes.types:
        self.env_buffer = {}
        for step in self.source_datapipe:
            if not issubclass(step.__class__,StepTypes.types):
                raise Exception(f'Expected typing.NamedTuple object got {type(step)}\n{step}')
    
            env_id,terminated = int(step.env_id),bool(step.terminated)
        
            if env_id in self.env_buffer:
                self.env_buffer[env_id].append(step)
            else:
                self.env_buffer[env_id] = [step]
                
            if not terminated and len(self.env_buffer[env_id])<self.n: continue
            
            while terminated and len(self.env_buffer[env_id])!=0:
                yield tuple(self.env_buffer[env_id])
                self.env_buffer[env_id].pop(0)
                
            if not terminated:
                yield tuple(self.env_buffer[env_id])
                self.env_buffer[env_id].pop(0)
add_docs(
NStepper,
"""Accepts a `source_datapipe` or iterable whose `next()` produces a `StepType.types` of 
max size `n` that will contain steps from a single environment with 
a subset of fields from `SimpleStep`, namely `terminated` and `env_id`.""",
)

# %% ../../../nbs/01_DataPipes/01c_pipes.iter.nstep.ipynb 5
class NStepFlattener(IterDataPipe):
    def __init__(
            self, 
            # The datapipe we are extracting from must produce `StepType.types` or `Tuple[StepType.types]`
            source_datapipe:IterDataPipe[Union[StepTypes.types]], 
        ) -> None:
        self.source_datapipe:IterDataPipe[[StepTypes.types]] = source_datapipe
        
    def __iter__(self) -> StepTypes.types:
        for step in self.source_datapipe:
            if issubclass(step.__class__,StepTypes.types):
                # print(step)
                yield step
            elif isinstance(step,tuple):
                # print('got step: ',step)
                yield from step 
            else:
                raise Exception(f'Expected {StepTypes.types} or tuple object got {type(step)}\n{step}')

            
add_docs(
NStepFlattener,
"""Handles unwrapping `StepType.typess` in tuples better than `dp.iter.UnBatcher` and `dp.iter.Flattener`""",
)

# %% ../../../nbs/01_DataPipes/01c_pipes.iter.nstep.ipynb 20
def n_steps_expected(
    default_steps:int, # The number of steps the episode would run without n_steps
    n:int # The n-step value that we are planning ot use
):
    return (default_steps * n) - sum(range(n))
    
n_steps_expected.__doc__=r"""
Produces the expected number of steps, assuming a fully deterministic episode based on `default_steps` and `n`

Given `n=2`, given 1 envs, knowing that `CartPole-v1` when `seed=0` will always run 18 steps, the total 
steps will be:

$$
18 * n - \sum_{0}^{n - 1}(i)
$$
"""    
