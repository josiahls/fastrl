# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/10a_agents.dqn.core.ipynb (unless otherwise specified).

__all__ = ['DQN', 'ArgMaxFeed', 'DiscreteEpsilonRandomSelect', 'Epsilon', 'DQNTrainer']

# Cell
# Python native modules
import os
from collections import deque
# Third party libs
import torch
from torch.nn import *
from fastcore.all import *
from fastai.learner import *
from fastai.torch_basics import *
from fastai.torch_core import *
from fastai.callback.all import *
# Local modules
from ...data.block import *
from ...data.gym import *
from ...agent import *
from ...core import *
from ...memory.experience_replay import *

# Cell
class DQN(Module):
    def __init__(self,state_sz:int,action_sz:int,hidden=512):
        self.layers=Sequential(
            Linear(state_sz,hidden),
            ReLU(),
            Linear(hidden,action_sz),
        )
    def forward(self,x): return self.layers(x)

# Cell
class ArgMaxFeed(AgentCallback):
    def before_action(self):
        raw_action=self.agent.model(self.experience['state'].to(default_device()))
        self.agent.raw_action_shape=raw_action.shape
        self.agent.action=torch.argmax(raw_action,dim=1).reshape(-1,1)

class DiscreteEpsilonRandomSelect(AgentCallback):

    def __init__(self,epsilon=0.5,idx=0,min_epsilon=0.2,max_epsilon=1,max_steps=5000):
        store_attr()

    def before_noise(self):
        self.mask=torch.randn(size=(self.agent.action.shape[0],))<self.epsilon
        self.experience['randomly_selected']=self.mask.reshape(-1,1)
        self.experience['epsilon']=torch.full(self.agent.action.shape,self.epsilon)
        self.experience['orignal_actions']=self.agent.action.detach().clone()
        self.agent.action[self.mask]=self.agent.action[self.mask].random_(0,self.agent.raw_action_shape[1])
        self.agent.action=self.agent.action.detach().cpu().numpy()

        if self.agent.model.training:
            self.idx+=1
            self.epsilon=max(self.min_epsilon,self.max_epsilon-self.idx/self.max_steps)

# Cell
class Epsilon(Metric):
    order=30
    epsilon=0

    @property
    def value(self): return self.epsilon
    def reset(self): self.epsilon=0
    def accumulate(self,learn):
        for cb in learn.model.cbs:
            if type(cb)==DiscreteEpsilonRandomSelect:
                self.epsilon=cb.epsilon

# Cell
class DQNTrainer(Callback):
    "Performs traditional training on `next_q`. Requires a callback such as `RegularNextQ`"
    def __init__(self,discount=0.99,n_steps=1):
        store_attr()
        self._xb=None
        self.n_batch=0

    def after_pred(self):
        self.learn.yb=self.xb
        # self.learn.xb=self.xb
        self._xb=(self.xb,)
        self.learn.done_mask=self.xb['done'].reshape(-1,)
        self.learn.next_q=self.learn.model.model(self.xb['next_state']).max(dim=1).values.reshape(-1,1)
        self.learn.next_q[self.done_mask]=0 #xb[done_mask]['reward']
        self.learn.targets=self.xb['reward']+self.learn.next_q*(self.discount**self.n_steps)
        self.learn.pred=self.learn.model.model(self.xb['state'])

        t_q=self.pred.clone()
        t_q.scatter_(1,self.xb['action'],self.targets)
        # finalize the xb and yb
        self.learn.yb=(t_q,)

        if (self.n_batch-1)%500==0:
            print('The loss should be practically zero: ',self.loss)
            print(self.learn.pred-t_q)


        with torch.no_grad():
            self.learn.td_error=(self.pred-self.yb[0]).mean(dim=1).reshape(-1,1)**2

    def before_backward(self):
        self.n_batch+=1
        self.learn.xb=self._xb