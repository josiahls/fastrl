# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/07_Agents/02_Continuous/12t_agents.trpo.ipynb.

# %% auto 0
__all__ = ['AdvantageStep', 'AdvantageBuffer']

# %% ../../nbs/07_Agents/02_Continuous/12t_agents.trpo.ipynb 3
# Python native modules
from typing import *
from typing_extensions import Literal
import typing 
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
class AdvantageStep(typing.NamedTuple):
    state:       torch.FloatTensor=torch.FloatTensor([0])
    action:      torch.FloatTensor=torch.FloatTensor([0])
    next_state:  torch.FloatTensor=torch.FloatTensor([0])
    terminated:  torch.BoolTensor=torch.BoolTensor([1])
    truncated:   torch.BoolTensor=torch.BoolTensor([1])
    reward:      torch.FloatTensor=torch.LongTensor([0])
    total_reward:torch.FloatTensor=torch.FloatTensor([0])
    advantage:   torch.FloatTensor=torch.FloatTensor([0])
    env_id:      torch.LongTensor=torch.LongTensor([0])
    proc_id:     torch.LongTensor=torch.LongTensor([0])
    step_n:      torch.LongTensor=torch.LongTensor([0])
    episode_n:   torch.LongTensor=torch.LongTensor([0])
    image:       torch.FloatTensor=torch.FloatTensor([0])
    
    def clone(self):
        return self.__class__(
            **{fld:getattr(self,fld).clone() for fld in self.__class__._fields}
        )
    
    def detach(self):
        return self.__class__(
            **{fld:getattr(self,fld).detach() for fld in self.__class__._fields}
        )
    
    def device(self,device='cpu'):
        return self.__class__(
            **{fld:getattr(self,fld).to(device=device) for fld in self.__class__._fields}
        )

    def to(self,*args,**kwargs):
        return self.__class__(
            **{fld:getattr(self,fld).to(*args,**kwargs) for fld in self.__class__._fields}
        )
    
    @classmethod
    def random(cls,seed=None,**flds):
        _flds,_annos = cls._fields,cls.__annotations__

        def _random_annos(anno):
            t = anno(1)
            if anno==torch.BoolTensor: t.random_(2) 
            else:                      t.random_(100)
            return t

        return cls(
            *(flds.get(
                f,_random_annos(_annos[f])
            ) for f in _flds)
        )

add_namedtuple_doc(
AdvantageStep,
"""Represents a single step in an environment similar to `SimpleStep` however has
an addition field called `advantage`.""",
advantage="""Generally characterized as $A(s,a) = Q(s,a) - V(s)$""",
**{f:getattr(SimpleStep,f).__doc__ for f in SimpleStep._fields}
)

# %% ../../nbs/07_Agents/02_Continuous/12t_agents.trpo.ipynb 10
class AdvantageBuffer(dp.iter.IterDataPipe):
    debug=False
    def __init__(self,
            source_datapipe:DataPipe,
            # Will accumulate up to `bs` or when the episode has terminated.
            bs=1000,
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
    
    def __iter__(self) -> AdvantageStep:
        self.env_advantage_buffer:Dict[Literal['env'],list] = {}
        for step in self.source_datapipe:
            if self.debug: print('Adding to advantage buffer: ',step)
            env_id = int(step.env_id.detach().cpu())
            if env_id not in self.env_advantage_buffer: 
                self.env_advantage_buffer[env_id] = []
            self.env_advantage_buffer[env_id].append(step)

            if any((step.truncated,step.terminated,len(self.env_advantage_buffer)>self.bs)):
                yield AdvantageStep()

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
