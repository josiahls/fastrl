# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/07_Agents/02_Continuous/12t_agents.trpo.ipynb.

# %% auto 0
__all__ = ['AdvantageStep', 'discounted_cumsum_', 'AdvantageBuffer', 'Actor']

# %% ../../nbs/07_Agents/02_Continuous/12t_agents.trpo.ipynb 3
# Python native modules
from typing import *
from typing_extensions import Literal
import typing 
# Third party libs
import numpy as np
import torch
from torch import nn
import torchdata.datapipes as dp 
from torchdata.dataloader2.graph import DataPipe,traverse,replace_dp
from fastcore.all import test_eq,test_ne
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
@torch.jit.script
def discounted_cumsum_(t:torch.Tensor,gamma:float,reverse:bool=False):
    """Performs a cumulative sum on `t` where `gamma` is applied for each index
    >1."""
    if reverse:
        # We do +2 because +1 is needed to avoid out of index t[idx], and +2 is needed
        # to avoid out of index for t[idx+1].
        for idx in range(t.size(0)-2,-1,-1):
            t[idx] = t[idx] + t[idx+1] * gamma
    else:
        for idx in range(1,t.size(0)):
            t[idx] = t[idx] + t[idx-1] * gamma

# %% ../../nbs/07_Agents/02_Continuous/12t_agents.trpo.ipynb 12
class AdvantageBuffer(dp.iter.IterDataPipe):
    debug=False
    def __init__(self,
            # A datapipe that produces `StepType`s.
            source_datapipe:DataPipe,
            # A model that takes in a `state` and outputs a single value 
            # representing $V$, where as $Q$ is $V + reward$
            critic:nn.Module,
            # Will accumulate up to `bs` or when the episode has terminated.
            bs=1000,
            # The discount factor, otherwise known as $\gamma$, is defined in 
            # (Shulman et al., 2016) as '... $\gamma$ introduces bias into
            # the policy gradient estimate...'.
            discount:float=0.99,
            # $\lambda$ is unqiue to GAE and manages importance to values when 
            # they are in accurate is defined in (Shulman et al., 2016) as '... $\lambda$ < 1
            # introduces bias only when the value function is inaccurate....'.
            gamma:float=0.99
        ):
        self.source_datapipe = source_datapipe
        self.bs = bs
        self.critic = critic
        self.device = None
        self.discount = discount
        self.gamma = gamma
        self.env_advantage_buffer:Dict[Literal['env'],list] = {}

    def to(self,*args,**kwargs):
        self.device = kwargs.get('device',None)

    def __repr__(self):
        return str({k:v if k!='env_advantage_buffer' else f'{len(self)} elements' 
                    for k,v in self.__dict__.items()})

    def __len__(self): return self._sz_tracker

    def update_advantage_buffer(self,step:StepType) -> int:
        if self.debug: 
            print('Adding to advantage buffer: ',step)
        env_id = int(step.env_id.detach().cpu())
        if env_id not in self.env_advantage_buffer: 
            self.env_advantage_buffer[env_id] = []
        self.env_advantage_buffer[env_id].append(step)
        return env_id
        
    def zip_steps(
            self,
            steps:List[StepType]
        ) -> Tuple[torch.FloatTensor,torch.FloatTensor,torch.BoolTensor]:
            step_subset = [(o.reward,o.state,o.truncated or o.terminated) for o in steps]
            zipped_fields = zip(*step_subset)
            return L(zipped_fields).map(torch.vstack)

    def delta_calc(self,reward,v,v_next,done):
        return reward + (self.gamma * v * done) - v_next

    def __iter__(self) -> AdvantageStep:
        self.env_advantage_buffer:Dict[Literal['env'],list] = {}
        for step in self.source_datapipe:
            env_id = self.update_advantage_buffer(step)
            done = step.truncated or step.terminated
            if done or len(self.env_advantage_buffer[env_id])>self.bs:
                steps = self.env_advantage_buffer[env_id]
                rewards,states,dones = self.zip_steps(steps)
                # We vstack the final next_state so we have a complete picture
                # of the state transitions and matching reward/done shapes.
                values = self.critic(torch.vstack((states,steps[-1].next_state)))
                delta = self.delta_calc(rewards,values[:-1],values[1:],dones)
                discounted_cumsum_(delta,self.discount*self.gamma,reverse=True)

                for _step,gae_advantage in zip(*(steps,delta)):
                    yield AdvantageStep(
                        advantage=gae_advantage,
                        **{f:getattr(_step,f) for f in _step._fields}
                    )

    @classmethod
    def insert_dp(cls,critic,old_dp=GymStepper) -> Callable[[DataPipe],DataPipe]:
        def _insert_dp(pipe):
            v = replace_dp(
                traverse(pipe,only_datapipe=True),
                find_dp(traverse(pipe,only_datapipe=True),old_dp),
                cls(find_dp(traverse(pipe,only_datapipe=True),old_dp),critic=critic)
            )
            return list(v.values())[0][0]
        return _insert_dp

add_docs(
AdvantageBuffer,
"""Collects an entire episode, calculates the advantage for each step, then
yields that episode's `AdvantageStep`s.

This is described in the original paper `(Shulman et al., 2016) High-Dimensional 
Continuous Control Usin Generalized Advantage Estimation`.

This algorithm is based on the concept of advantage:

$A_{\pi}(s,a) = Q_{\pi}(s,a) - V_{\pi}(s)$

Where (Shulman et al., 2016) pg 5 calculates it as:

$\hat{A}_{t}^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}^V$

Where (Shulman et al., 2016) pg 4 defines $\delta$ as:

$\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_{t})$
""",
to=torch.Tensor.to.__doc__,
update_advantage_buffer="Adds `step` to `env_advantage_buffer` based on the environment id.",
zip_steps="""Given `steps`, strip out the `Tuple[reward,state,truncated or terminated]` fields,
and `torch.vstack` them.""",
delta_calc="""Calculates $\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_{t})$ which 
is the advantage difference between state transitions."""
)

# %% ../../nbs/07_Agents/02_Continuous/12t_agents.trpo.ipynb 16
class Actor(Module):
    def __init__(            
            self,
            state_sz:int,   # The input dim of the state / flattened conv output
            action_sz:int,  # The output dim of the actions
            hidden:int=400, # Number of neurons connected between the 2 input/output layers
        ):
        self.mu = nn.Sequential(
            nn.Linear(state_sz, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_sz),
            nn.Tanh(),
        )
        self.logstd = nn.Parameter(torch.zeros(action_sz))

    def forward(self, x):
        return self.mu(x.float())

add_docs(
Actor,
"""Produces continuous outputs from mean of a Gaussian distribution.""",
forward="Mean outputs from a parameterized Gaussian distribution."
)
