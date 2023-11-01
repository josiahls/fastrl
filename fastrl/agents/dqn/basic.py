# AUTOGENERATED! DO NOT EDIT! File to edit: ../../../nbs/07_Agents/01_Discrete/12g_agents.dqn.basic.ipynb.

# %% auto 0
__all__ = ['DataPipeAugmentationFn', 'DQN', 'DQNAgent', 'QCalc', 'TargetCalc', 'LossCalc', 'ModelLearnCalc', 'LossCollector',
           'DQNLearner']

# %% ../../../nbs/07_Agents/01_Discrete/12g_agents.dqn.basic.ipynb 2
# Python native modules
import os
from collections import deque
from typing import Callable,Optional,List
# Third party libs
from fastcore.all import ifnone
import torchdata.datapipes as dp
from torchdata.dataloader2 import DataLoader2
from torchdata.dataloader2.graph import traverse_dps,DataPipe
import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
import numpy as np
# Local modules
from ..core import AgentHead,AgentBase
from ...pipes.core import find_dp
from ...memory.experience_replay import ExperienceReplay
from ..core import StepFieldSelector,SimpleModelRunner,NumpyConverter
from ..discrete import EpsilonCollector,PyPrimativeConverter,ArgMaxer,EpsilonSelector
from fastrl.loggers.core import (
    LogCollector,Record,BatchCollector,EpochCollector,RollingTerminatedRewardCollector,EpisodeCollector,is_record
)
from ...learner.core import LearnerBase,LearnerHead,StepBatcher
from ...torch_core import Module

# %% ../../../nbs/07_Agents/01_Discrete/12g_agents.dqn.basic.ipynb 5
class DQN(Module):
    def __init__(
            self,
            state_sz:int,  # The input dim of the state
            action_sz:int, # The output dim of the actions
            hidden=512,    # Number of neurons connected between the 2 input/output layers
            head_layer:Module=nn.Linear, # DQN extensions such as Dueling DQNs have custom heads
            activition_fn:Module=nn.ReLU # The activiation fn used by `DQN`
        ):
        self.layers=nn.Sequential(
            nn.Linear(state_sz,hidden),
            activition_fn(),
            head_layer(hidden,action_sz),
        )
    def forward(self,x): return self.layers(x)


# %% ../../../nbs/07_Agents/01_Discrete/12g_agents.dqn.basic.ipynb 7
DataPipeAugmentationFn = Callable[[DataPipe],Optional[DataPipe]]

def DQNAgent(
    model,
    min_epsilon=0.02,
    max_epsilon=1,
    max_steps=1000,
    device='cpu',
    do_logging:bool=False
)->AgentHead:
    agent_base = AgentBase(model)
    agent_base = StepFieldSelector(agent_base,field='next_state')
    agent_base = SimpleModelRunner(agent_base).to(device=device)
    agent,raw_agent = agent_base.fork(2)
    agent = agent.map(torch.clone)
    agent = ArgMaxer(agent)
    agent = EpsilonSelector(agent,min_epsilon=min_epsilon,max_epsilon=max_epsilon,max_steps=max_steps,device=device)
    if do_logging: 
        agent = EpsilonCollector(agent).catch_records()
    agent = ArgMaxer(agent,only_idx=True)
    agent = NumpyConverter(agent)
    agent = PyPrimativeConverter(agent)
    agent = agent.zip(raw_agent)
    agent = AgentHead(agent)
    return agent

# %% ../../../nbs/07_Agents/01_Discrete/12g_agents.dqn.basic.ipynb 14
class QCalc(dp.iter.IterDataPipe):
    def __init__(self,source_datapipe):
        self.source_datapipe = source_datapipe
        
    def __iter__(self):
        self.learner = find_dp(traverse_dps(self),LearnerBase)
        for batch in self.source_datapipe:
            self.learner.done_mask = batch.terminated.reshape(-1,)
            self.learner.next_q = self.learner.model(batch.next_state)
            self.learner.next_q = self.learner.next_q.max(dim=1).values.reshape(-1,1)
            self.learner.next_q[self.learner.done_mask] = 0 
            yield batch

# %% ../../../nbs/07_Agents/01_Discrete/12g_agents.dqn.basic.ipynb 15
class TargetCalc(dp.iter.IterDataPipe):
    def __init__(self,source_datapipe,discount=0.99,nsteps=1):
        self.source_datapipe = source_datapipe
        self.discount = discount
        self.nsteps = nsteps
        self.learner = None
        
    def __iter__(self):
        self.learner = find_dp(traverse_dps(self),LearnerBase)
        for batch in self.source_datapipe:
            self.learner.targets = batch.reward+self.learner.next_q*(self.discount**self.nsteps)
            self.learner.pred = self.learner.model(batch.state)
            self.learner.target_qs = self.learner.pred.clone().float()
            self.learner.target_qs.scatter_(1,batch.action.long(),self.learner.targets.float())
            yield batch

# %% ../../../nbs/07_Agents/01_Discrete/12g_agents.dqn.basic.ipynb 16
class LossCalc(dp.iter.IterDataPipe):
    def __init__(self,source_datapipe,loss_func):
        self.source_datapipe = source_datapipe
        self.loss_func = loss_func
        
    def __iter__(self):
        self.learner = find_dp(traverse_dps(self),LearnerBase)
        for batch in self.source_datapipe:
            self.learner.loss_grad = self.loss_func(self.learner.pred, self.learner.target_qs)
            yield batch

# %% ../../../nbs/07_Agents/01_Discrete/12g_agents.dqn.basic.ipynb 17
class ModelLearnCalc(dp.iter.IterDataPipe):
    def __init__(self,source_datapipe, opt):
        self.source_datapipe = source_datapipe
        self.opt = opt
        
    def __iter__(self):
        self.learner = find_dp(traverse_dps(self),LearnerBase)
        for batch in self.source_datapipe:
            self.learner.loss_grad.backward()
            self.opt.step()
            self.opt.zero_grad()
            self.learner.loss = self.learner.loss_grad.clone()
            yield self.learner.loss

# %% ../../../nbs/07_Agents/01_Discrete/12g_agents.dqn.basic.ipynb 18
class LossCollector(dp.iter.IterDataPipe):
    title:str='loss'

    def __init__(self,
            source_datapipe, # The parent datapipe, likely the one to collect metrics from
        ):
        self.source_datapipe = source_datapipe
        self.main_buffers = None
        
        
    def __iter__(self):
        self.learner = find_dp(traverse_dps(self),LearnerBase)
        yield Record(self.title,None)
        for i,steps in enumerate(self.source_datapipe):
            yield Record('loss',self.learner.loss.cpu().detach().numpy())
            yield steps

# %% ../../../nbs/07_Agents/01_Discrete/12g_agents.dqn.basic.ipynb 19
def DQNLearner(
    model,
    dls,
    logger_bases:Optional[Callable]=None,
    loss_func=nn.MSELoss(),
    opt=optim.AdamW,
    lr=0.005,
    bs=128,
    max_sz=10000,
    nsteps=1,
    device=None,
    batches=None
) -> LearnerHead:
    learner = LearnerBase(model,dls[0])
    learner = BatchCollector(learner,batches=batches)
    learner = EpochCollector(learner)
    if logger_bases: 
        learner = logger_bases(learner) 
        learner = RollingTerminatedRewardCollector(learner)
        learner = EpisodeCollector(learner)
    learner = learner.catch_records()
    learner = ExperienceReplay(learner,bs=bs,max_sz=max_sz,freeze_memory=True)
    learner = StepBatcher(learner,device=device)
    learner = QCalc(learner)
    learner = TargetCalc(learner,nsteps=nsteps)
    learner = LossCalc(learner,loss_func=loss_func)
    learner = ModelLearnCalc(learner,opt=opt(model.parameters(),lr=lr))
    if logger_bases: 
        learner = LossCollector(learner).catch_records()

    if len(dls)==2:
        val_learner = LearnerBase(model,dls[1])
        val_learner = BatchCollector(val_learner,batches=batches)
        val_learner = EpochCollector(val_learner).dump_records()
        learner = LearnerHead((learner,val_learner),model)
    else:
        learner = LearnerHead(learner,model)
    return learner
