# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/20f_qlearning.dqn_dueling.ipynb (unless otherwise specified).

__all__ = ['DuelingBlock', 'DuelingDQN']

# Cell
import torch.nn.utils as nn_utils
from fastai.torch_basics import *
from fastai.data.all import *
from fastai.basics import *
from dataclasses import field,asdict
from typing import List,Any,Dict,Callable
from collections import deque
import gym
import torch.multiprocessing as mp
from torch.optim import *

from ..data import *
from ..async_data import *
from ..basic_agents import *
from ..learner import *
from ..metrics import *
from ..ptan_extension import *
from .dqn import *
from .dqn_target import *

if IN_NOTEBOOK:
    from IPython import display
    import PIL.Image

# Cell
class DuelingBlock(nn.Module):
    def __init__(self,h,ao,lin_cls=nn.Linear):
        super().__init__()

        self.val = lin_cls(h, 1)
        self.adv = lin_cls(h, ao)

    def forward(self, xi):
        val, adv = self.val(xi), self.adv(xi)
        xi = val.expand_as(adv) + (adv - adv.mean()).squeeze(0)
        return xi

class DuelingDQN(LinearDQN):
    def __init__(self, input_shape, n_actions):
        super(LinearDQN, self).__init__()

        self.policy = nn.Sequential(
            nn.Linear(input_shape[0], 512),
            nn.ReLU(),
            DuelingBlock(512, n_actions)
        )
