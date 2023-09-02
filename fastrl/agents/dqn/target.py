# AUTOGENERATED! DO NOT EDIT! File to edit: ../../../nbs/07_Agents/01_Discrete/12h_agents.dqn.target.ipynb.

# %% auto 0
__all__ = ['TargetModelUpdater', 'TargetModelQCalc', 'DQNTargetLearner']

# %% ../../../nbs/07_Agents/01_Discrete/12h_agents.dqn.target.ipynb 2
# Python native modules
from copy import deepcopy
from typing import Optional,Callable
# Third party libs
import torchdata.datapipes as dp
from torchdata.dataloader2.graph import traverse_dps,DataPipe
import torch
from torch import nn,optim
# Local modules
from ...pipes.core import find_dp
from ...memory.experience_replay import ExperienceReplay
from ...loggers.core import BatchCollector,EpochCollector
from ...learner.core import LearnerBase,LearnerHead
from fastrl.agents.dqn.basic import (
    LossCollector,
    RollingTerminatedRewardCollector,
    EpisodeCollector,
    StepBatcher,
    TargetCalc,
    LossCalc,
    ModelLearnCalc,
    DQN,
    DQNAgent
)

# %% ../../../nbs/07_Agents/01_Discrete/12h_agents.dqn.target.ipynb 5
class TargetModelUpdater(dp.iter.IterDataPipe):
    def __init__(self,source_datapipe=None,target_sync=300):
        self.source_datapipe = source_datapipe
        if source_datapipe is not None:
            self.learner = find_dp(traverse_dps(self),LearnerBase)
            self.learner.target_model = deepcopy(self.learner.model)
        self.target_sync = target_sync
        self.n_batch = 0
        
    def reset(self):
        self.learner = find_dp(traverse_dps(self),LearnerBase)
        self.learner.target_model = deepcopy(self.learner.model)
        
    def __iter__(self):
        if self._snapshot_state.NotStarted: 
            self.reset()
        for batch in self.source_datapipe:
            if self.n_batch%self.target_sync==0:
                self.learner.target_model.load_state_dict(self.learner.model.state_dict())
            self.n_batch+=1
            yield batch

# %% ../../../nbs/07_Agents/01_Discrete/12h_agents.dqn.target.ipynb 6
class TargetModelQCalc(dp.iter.IterDataPipe):
    def __init__(self,source_datapipe=None):
        self.source_datapipe = source_datapipe
        if source_datapipe is not None: self.learner = find_dp(traverse_dps(self),LearnerBase)
        
    def reset(self):
        self.learner = find_dp(traverse_dps(self),LearnerBase)
        
    def __iter__(self):
        for batch in self.source_datapipe:
            self.learner.done_mask = batch.terminated.reshape(-1,)
            with torch.no_grad():
                self.learner.next_q = self.learner.target_model(batch.next_state)
            self.learner.next_q = self.learner.next_q.max(dim=1).values.reshape(-1,1)
            self.learner.next_q[self.learner.done_mask] = 0 
            yield batch

# %% ../../../nbs/07_Agents/01_Discrete/12h_agents.dqn.target.ipynb 7
def DQNTargetLearner(
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
    learner = LearnerBase(
        model,
        dls,
        batches=batches,
        loss_func=loss_func,
        opt=opt(model.parameters(),lr=lr)
    )
    learner = BatchCollector(learner,batch_on_pipe=LearnerBase)
    learner = EpochCollector(learner).catch_records()
    if logger_bases: 
        learner = logger_bases(learner)
        learner = RollingTerminatedRewardCollector(learner).catch_records()
        learner = EpisodeCollector(learner).catch_records()
    learner = ExperienceReplay(learner,bs=bs,max_sz=max_sz)
    learner = StepBatcher(learner,device=device)
    learner = TargetModelQCalc(learner)
    learner = TargetCalc(learner,nsteps=nsteps)
    learner = LossCalc(learner)
    learner = ModelLearnCalc(learner)
    learner = TargetModelUpdater(learner)
    if logger_bases: 
        learner = LossCollector(learner).catch_records()
    learner = LearnerHead(learner)
    return learner
