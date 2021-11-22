# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/10b_agents.dqn.targets.ipynb (unless otherwise specified).

__all__ = ['DQNTargetTrainer']

# Cell
# Python native modules
import os
from collections import deque
from typing import *
from warnings import warn
import logging
# Third party libs
import torch
from torch.nn import *
from torch import optim
from fastcore.all import *
from fastai.learner import *
from fastai.torch_basics import *
from fastai.torch_core import *
from fastai.optimizer import OptimWrapper
from fastai.callback.all import *
# Local modules
from ...data.block import *
from ...data.gym import *
from ...agent import *
from ...core import *
from .core import *
from ...memory.experience_replay import *

_logger=logging.getLogger()

# Cell
class DQNTargetTrainer(Callback):

    def __init__(self,n_batch=0,target_sync=300,discount=0.99,n_steps=1):
        store_attr()
        self._xb=None

    def before_fit(self):
        self.learn.target_model=deepcopy(self.learn.model.model)
        self.n_batch=0

    def after_pred(self):
        self._xb=self.yb
        self.learn.yb=[]

        self.learn.opt.zero_grad()
        with torch.no_grad():
            s=self.learn.xb['state']
            a=self.learn.xb['action']
            ns=self.xb['next_state']
            r=self.xb['reward']
            d=self.xb['done']

            # Lets try this
            # ns[d.squeeze(-1)]=s[d.squeeze(-1)]

        self.learn.state_action_values = self.learn.model.model(s)

        self.learn.selected_state_action_values = self.learn.state_action_values.gather(1,a).squeeze(-1)
        # with torch.no_grad():

        self.learn.next_state_values = self.target_model(ns).max(1)[0]
        # self.learn.next_state_values[d.squeeze(-1)]=0
        r[d.squeeze(-1)]=0
        self.learn.expected_state_action_values = self.learn.next_state_values.detach() * (self.discount**self.n_steps) + r.squeeze(-1)

        self.learn.loss= nn.MSELoss()(self.learn.selected_state_action_values,self.learn.expected_state_action_values)

        self.learn.loss.backward()
        self.learn.opt.step()

        with torch.no_grad():
            self.learn.expected_reward=self.learn.state_action_values.cpu()
            self.learn.retrospective_action=self.learn.state_action_values.cpu().argmax(dim=1).reshape(-1,1).float()
            self.learn.td_error=(self.learn.selected_state_action_values.cpu()-self.learn.expected_state_action_values.cpu()).reshape(-1,1)**2

    def before_backward(self): self.learn.yb=self._xb

    def after_batch(self):
        if self.n_batch%self.target_sync==0:
            self.target_model.load_state_dict(self.learn.model.state_dict())
            # if self.n_batch>1:raise Exception
        self.n_batch+=1