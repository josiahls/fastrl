# AUTOGENERATED! DO NOT EDIT! File to edit: ../../../nbs/07_Agents/01_Discrete/12h_agents.dqn.target.ipynb.

# %% auto 0
__all__ = ['TargetModelUpdater', 'TargetModelQCalc', 'DQNTargetLearner']

# %% ../../../nbs/07_Agents/01_Discrete/12h_agents.dqn.target.ipynb 2
# Python native modules
from copy import deepcopy
from typing import Optional,Callable,Tuple
# Third party libs
import torchdata.datapipes as dp
from torchdata.dataloader2.graph import traverse_dps,DataPipe
import torch
from torch import nn,optim
# Local modulesf
from ...pipes.core import find_dp
from ...memory.experience_replay import ExperienceReplay
from ...loggers.core import BatchCollector,EpochCollector
from ...learner.core import LearnerBase,LearnerHead
from ...loggers.vscode_visualizers import VSCodeDataPipe
from ...loggers.core import ProgressBarLogger
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

# %% ../../../nbs/07_Agents/01_Discrete/12h_agents.dqn.target.ipynb 7
class TargetModelUpdater(dp.iter.IterDataPipe):
    def __init__(self,source_datapipe,target_sync=300):
        self.source_datapipe = source_datapipe
        self.target_sync = target_sync
        self.n_batch = 0
        self.learner = find_dp(traverse_dps(self),LearnerBase)
        with torch.no_grad():
            self.learner.target_model = deepcopy(self.learner.model)
        
    def reset(self):
        self.learner = find_dp(traverse_dps(self),LearnerBase)
        with torch.no_grad():
            self.learner.target_model = deepcopy(self.learner.model)
        
    def __iter__(self):
        if self._snapshot_state.NotStarted: 
            self.reset()
        for batch in self.source_datapipe:
            if self.n_batch%self.target_sync==0:
                with torch.no_grad():
                    self.learner.target_model.load_state_dict(self.learner.model.state_dict())
            self.n_batch+=1
            yield batch

# %% ../../../nbs/07_Agents/01_Discrete/12h_agents.dqn.target.ipynb 8
class TargetModelQCalc(dp.iter.IterDataPipe):
    def __init__(self,source_datapipe=None):
        self.source_datapipe = source_datapipe
        
    def __iter__(self):
        self.learner = find_dp(traverse_dps(self),LearnerBase)
        for batch in self.source_datapipe:
            self.learner.done_mask = batch.terminated.reshape(-1,)
            with torch.no_grad():
                self.learner.next_q = self.learner.target_model(batch.next_state)
            self.learner.next_q = self.learner.next_q.max(dim=1).values.reshape(-1,1)
            self.learner.next_q[self.learner.done_mask] = 0 
            yield batch

# %% ../../../nbs/07_Agents/01_Discrete/12h_agents.dqn.target.ipynb 9
def DQNTargetLearner(
    model,
    dls,
    do_logging:bool=True,
    loss_func=nn.MSELoss(),
    opt=optim.AdamW,
    lr=0.005,
    bs=128,
    max_sz=10000,
    nsteps=1,
    device=None,
    batches=None,
    target_sync=300
) -> LearnerHead:
    learner = LearnerBase(model,dls=dls[0])
    learner = BatchCollector(learner,batches=batches)
    learner = EpochCollector(learner)
    if do_logging: 
        learner = learner.dump_records()
        learner = ProgressBarLogger(learner)
        learner = RollingTerminatedRewardCollector(learner)
        learner = EpisodeCollector(learner).catch_records()
    learner = ExperienceReplay(learner,bs=bs,max_sz=max_sz)
    learner = StepBatcher(learner,device=device)
    learner = TargetModelQCalc(learner)
    learner = TargetCalc(learner,nsteps=nsteps)
    learner = LossCalc(learner,loss_func=loss_func)
    learner = ModelLearnCalc(learner,opt=opt(model.parameters(),lr=lr))
    learner = TargetModelUpdater(learner,target_sync=target_sync)
    if do_logging: 
        learner = LossCollector(learner).catch_records()

    if len(dls)==2:
        val_learner = LearnerBase(model,dls[1]).visualize_vscode()
        val_learner = BatchCollector(val_learner,batches=batches)
        val_learner = EpochCollector(val_learner).catch_records(drop=True)
        return LearnerHead((learner,val_learner))
    else:
        return LearnerHead(learner)
