# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/10d_agents.dqn.dueling.ipynb.

# %% auto 0
__all__ = ['DuelingBlock', 'DuelingDQN']

# %% ../nbs/10d_agents.dqn.dueling.ipynb 3
# Python native modules
# Third party libs
from torch.nn import *
from fastcore.all import *
from fastai.learner import *
from fastai.torch_basics import *
from fastai.torch_core import *
from fastai.callback.all import *
# Local modules
from ...fastai.data.block_simple import *
from ...fastai.data.gym import *
from ...agent import *
from ...core import *
from .core import *
from .targets import *
from .double import *
from ...memory.experience_replay import *

# %% ../nbs/10d_agents.dqn.dueling.ipynb 5
class DuelingBlock(nn.Module):
    def __init__(self,n_actions,hidden=512,lin_cls=nn.Linear):
        super().__init__()
        self.val=lin_cls(hidden,1)
        self.adv=lin_cls(hidden,n_actions)

    def forward(self,xi):
        val,adv=self.val(xi),self.adv(xi)
        xi=val.expand_as(adv)+(adv-adv.mean()).squeeze(0)
        return xi
    
class DuelingDQN(DQN):
    def __init__(self,state_sz:int,n_actions,hidden=512):
        super(DQN,self).__init__()
        self.layers=nn.Sequential(
            nn.Linear(state_sz,hidden),
            nn.ReLU(),
            DuelingBlock(n_actions,hidden)
        )
