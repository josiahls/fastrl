# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/07_Agents/02_Continuous/12t_agents.trpo.ipynb.

# %% auto 0
__all__ = ['AdvantageBuffer', 'A']

# %% ../../nbs/07_Agents/02_Continuous/12t_agents.trpo.ipynb 3
# Python native modules
from typing import Callable
# Third party libs
import numpy as np
import torch
import torchdata.datapipes as dp 
from torchdata.dataloader2.graph import DataPipe,traverse,replace_dp
# Local modules
from ..core import *
from ..pipes.core import *
from ..torch_core import *
from ..layers import *
from ..data.block import *
from ..envs.gym import *

# %% ../../nbs/07_Agents/02_Continuous/12t_agents.trpo.ipynb 7
class AdvantageBuffer(dp.iter.IterDataPipe):
    debug=False
    def __init__(self,
            source_datapipe,
            # Will accumulate up to `bs` or when the episode has terminated.
            bs=1,
            # If the `self.device` is not cpu, and `store_as_cpu=True`, then
            # calls to `sample()` will dynamically move them to `self.device`, and
            # next `sample()` will move them back to cpu before producing new samples.
            # This can be slower, but can save vram.
            # If `store_as_cpu=False`, then samples stay on `self.device`
            #
            # If being run with n_workers>0, shared_memory, and fork, this MUST be true. This is needed because
            # otherwise the tensors in the memory will remain shared with the tensors created in the 
            # dataloader.
            store_as_cpu:bool=True
        ):
        self.source_datapipe = source_datapipe
        self.bs = bs
        self.store_as_cpu = store_as_cpu
        self.device = None

    def to(self,*args,**kwargs):
        self.device = kwargs.get('device',None)

    def __repr__(self):
        return str({k:v if k!='memory' else f'{len(self)} elements' for k,v in self.__dict__.items()})

    def __len__(self): return self._sz_tracker
    
    def __iter__(self):
        for step in self.source_datapipe:
            print('GAE step')
            # if self.debug: print('Adding to advantage buffer: ',b)
            
            # if not issubclass(b.__class__,(StepType,list,tuple)):
            #     raise Exception(f'Expected typing.NamedTuple,list,tuple object got {type(step)}\n{step}')
            
            # if issubclass(b.__class__,StepType):   self.add(b)
            # elif issubclass(b.__class__,(list,tuple)): 
            #     for step in b: self.add(step)
            # else:
            #     raise Exception(f'This should not have occured: {self.__dict__}')
        
            # if self._sz_tracker<self.bs: continue
            yield step 

    @classmethod
    def insert_dp(cls,old_dp=GymStepper) -> Callable[[DataPipe],DataPipe]:
        def _insert_dp(pipe):
            v = replace_dp(
                traverse(pipe,only_datapipe=True),
                find_dp(traverse(pipe,only_datapipe=True),old_dp),
                cls(find_dp(traverse(pipe,only_datapipe=True),old_dp))
            )
            return list(v.values())[0][0]
        return _insert_dp

add_docs(
AdvantageBuffer,
"""Samples entire trajectories instead of individual time steps.""",
to=torch.Tensor.to.__doc__
)

# %% ../../nbs/07_Agents/02_Continuous/12t_agents.trpo.ipynb 10
class A:pass
