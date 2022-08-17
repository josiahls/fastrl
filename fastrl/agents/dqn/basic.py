# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/12g_agents.dqn.basic.ipynb.

# %% auto 0
__all__ = ['DQN', 'QCalc', 'ModelLearnCalc', 'StepBatcher', 'EpisodeCollector', 'LossCollector',
           'RollingTerminatedRewardCollector']

# %% ../nbs/12g_agents.dqn.basic.ipynb 3
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

# %% ../nbs/12g_agents.dqn.basic.ipynb 5
class DQN(Module):
    def __init__(self,state_sz:int,action_sz:int,hidden=512):
        self.layers=Sequential(
            Linear(state_sz,hidden),
            ReLU(),
            Linear(hidden,action_sz),
        )
    def forward(self,x): return self.layers(x)


# %% ../nbs/12g_agents.dqn.basic.ipynb 12
class QCalc(dp.iter.IterDataPipe):
    def __init__(self,source_datapipe,discount=0.99,nsteps=1):
        self.source_datapipe = source_datapipe
        self.discount = discount
        self.nsteps = nsteps
        self.learner = find_pipe_instance(self,LearnerBase)
        
    def __iter__(self):
        for batch in self.source_datapipe:
            self.learner.done_mask = batch.terminated.reshape(-1,)
            
            self.learner.next_q = self.learner.model(batch.next_state)
            # print(self.learner.next_q,self.learner.done_mask)
            self.learner.next_q = self.learner.next_q.max(dim=1).values.reshape(-1,1)
            self.learner.next_q[self.learner.done_mask] = 0 #xb[done_mask]['reward']
            self.learner.targets = batch.reward+self.learner.next_q*(self.discount**self.nsteps)
            self.learner.pred = self.learner.model(batch.state)
            
            t_q=self.learner.pred.clone()
            t_q.scatter_(1,batch.action.long(),self.learner.targets)
            
            self.learner.loss_grad = self.learner.loss_func(self.learner.pred, t_q)
            yield batch
            
class ModelLearnCalc(dp.iter.IterDataPipe):
    def __init__(self,source_datapipe):
        self.source_datapipe = source_datapipe
        self.learner = find_pipe_instance(self,LearnerBase)
        
    def __iter__(self):
        for batch in self.source_datapipe:
            self.learner.loss_grad.backward()
            self.learner.opt.step()
            self.learner.opt.zero_grad()
            self.learner.loss = self.learner.loss_grad.clone()
            yield self.learner.loss

# %% ../nbs/12g_agents.dqn.basic.ipynb 13
class StepBatcher(dp.iter.IterDataPipe):
    def __init__(self,source_datapipe):
        "Converts multiple `StepType` into a single `StepType` with the fields concated."
        self.source_datapipe = source_datapipe
        
    def __iter__(self):
        for batch in self.source_datapipe:
            # print(batch)
            cls = batch[0].__class__
            yield cls(
                **{
                    fld:torch.vstack(tuple(getattr(step,fld) for step in batch)) for fld in cls._fields
                }
            )

# %% ../nbs/12g_agents.dqn.basic.ipynb 14
class EpisodeCollector(LogCollector):
    def __iter__(self):
        for q in self.main_queues: q.put(Record('episode',None))
        for steps in self.source_datapipe:
            if isinstance(steps,dp.DataChunk):
                for step in steps:
                    for q in self.main_queues: q.put(Record('episode',step.episode_n.detach().numpy()[0]))
            else:
                for q in self.main_queues: q.put(Record('episode',steps.episode_n.detach().numpy()[0]))
            yield steps

# %% ../nbs/12g_agents.dqn.basic.ipynb 15
class LossCollector(LogCollector):
    def __init__(self,
         source_datapipe, # The parent datapipe, likely the one to collect metrics from
         logger_bases:List[LoggerBase] # `LoggerBase`s that we want to send metrics to
        ):
        self.source_datapipe = source_datapipe
        self.main_queues = [o.main_queue for o in logger_bases]
        self.learner = find_pipe_instance(self,LearnerBase)
        
    def __iter__(self):
        for q in self.main_queues: q.put(Record('loss',None))
        for steps in self.source_datapipe:
            for q in self.main_queues: q.put(Record('loss',self.learner.loss.detach().numpy()))
            yield steps

# %% ../nbs/12g_agents.dqn.basic.ipynb 16
class RollingTerminatedRewardCollector(LogCollector):
    def __init__(self,
         source_datapipe, # The parent datapipe, likely the one to collect metrics from
         logger_bases:List[LoggerBase], # `LoggerBase`s that we want to send metrics to
         rolling_length:int=100
        ):
        self.source_datapipe = source_datapipe
        self.main_queues = [o.main_queue for o in logger_bases]
        self.rolling_rewards = deque([],maxlen=rolling_length)
        
    def step2terminated(self,step): return bool(step.terminated)
    def __iter__(self):
        for q in self.main_queues: q.put(Record('rolling_reward',None))
        for steps in self.source_datapipe:
            if isinstance(steps,dp.DataChunk):
                for step in steps:
                    if self.step2terminated(step):
                        self.rolling_rewards.append(step.total_reward.detach().numpy()[0])
                        for q in self.main_queues: q.put(Record('rolling_reward',np.average(self.rolling_rewards)))
            elif self.step2terminated(steps):
                self.rolling_rewards.append(steps.total_reward.detach().numpy()[0])
                for q in self.main_queues: q.put(Record('rolling_reward',np.average(self.rolling_rewards)))
            yield steps
