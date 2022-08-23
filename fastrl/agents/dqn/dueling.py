# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/12n_agents.dqn.dueling.ipynb.

# %% auto 0
__all__ = ['DuelingBlock', 'DuelingDQN']

# %% ../nbs/12n_agents.dqn.dueling.ipynb 3
# Python native modules
import os
from collections import deque
# Third party libs
from fastcore.all import *
import torchdata.datapipes as dp
from torch.utils.data.dataloader_experimental import DataLoader2
from torch.utils.data.datapipes._typing import _DataPipeMeta, _IterDataPipeMeta
# Local modules
import torch
from torch.nn import *
import torch.nn.functional as F
from torch.optim import *
from fastai.torch_basics import *
from fastai.torch_core import *

from ...core import *
from ..core import *
from ...pipes.core import *
from ...fastai.data.block import *
from ...memory.experience_replay import *
from ..core import *
from ..discrete import *
from ...loggers.core import *
from ...loggers.jupyter_visualizers import *
from ...learner.core import *
from .basic import *
from .target import *

# %% ../nbs/12n_agents.dqn.dueling.ipynb 6
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
