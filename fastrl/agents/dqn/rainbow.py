# AUTOGENERATED! DO NOT EDIT! File to edit: ../../../nbs/07_Agents/01_Discrete/12r_agents.dqn.rainbow.ipynb.

# %% auto 0
__all__ = ['DQNRainbowLearner']

# %% ../../../nbs/07_Agents/01_Discrete/12r_agents.dqn.rainbow.ipynb 2
# Python native modules
from copy import deepcopy
from typing import Optional,Callable,Tuple
# Third party libs
import torchdata.datapipes as dp
from torchdata.dataloader2.graph import traverse_dps,DataPipe
import torch
from torch import nn,optim
from fastcore.all import store_attr,ifnone
import numpy as np
import torch.nn.functional as F
# Local modulesf
from ...torch_core import default_device,to_detach,evaluating
from ...pipes.core import find_dp
from ..core import StepFieldSelector,SimpleModelRunner,NumpyConverter
from ..discrete import EpsilonCollector,PyPrimativeConverter,ArgMaxer,EpsilonSelector
from ...memory.experience_replay import ExperienceReplay
from ...loggers.core import BatchCollector,EpochCollector
from ...learner.core import LearnerBase,LearnerHead
from ...loggers.vscode_visualizers import VSCodeDataPipe
from ..core import AgentHead,AgentBase
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
from fastrl.agents.dqn.target import (
    TargetModelUpdater,
    TargetModelQCalc
)
from .dueling import DuelingHead 
from fastrl.agents.dqn.categorical import (
    CategoricalDQNAgent,
    CategoricalDQN,
    PartialCrossEntropy
) 
from ..categorical import CategoricalTargetQCalc

# %% ../../../nbs/07_Agents/01_Discrete/12r_agents.dqn.rainbow.ipynb 4
def DQNRainbowLearner(
    model,
    dls,
    logger_bases:Optional[Callable]=None,
    loss_func=PartialCrossEntropy,
    opt=optim.AdamW,
    lr=0.005,
    bs=128,
    max_sz=10000,
    nsteps=1,
    device=None,
    batches=None,
    target_sync=300,
    double_dqn_strategy=True
) -> LearnerHead:
    learner = LearnerBase(model,dls=dls[0])
    learner = BatchCollector(learner,batches=batches)
    learner = EpochCollector(learner)
    if logger_bases: 
        learner = logger_bases(learner)
        learner = RollingTerminatedRewardCollector(learner)
        learner = EpisodeCollector(learner)
    learner = learner.catch_records()
    learner = ExperienceReplay(learner,bs=bs,max_sz=max_sz)
    learner = StepBatcher(learner,device=device)
    learner = CategoricalTargetQCalc(learner,nsteps=nsteps,double_dqn_strategy=double_dqn_strategy).to(device=device)
    # learner = TargetCalc(learner,nsteps=nsteps)
    learner = LossCalc(learner,loss_func=loss_func)
    learner = ModelLearnCalc(learner,opt=opt(model.parameters(),lr=lr))
    learner = TargetModelUpdater(learner,target_sync=target_sync)
    if logger_bases: 
        learner = LossCollector(learner).catch_records()

    if len(dls)==2:
        val_learner = LearnerBase(model,dls[1])
        val_learner = BatchCollector(val_learner,batches=batches)
        val_learner = EpochCollector(val_learner).catch_records(drop=True)
        val_learner = VSCodeDataPipe(val_learner)
        return LearnerHead((learner,val_learner),model)
    else:
        return LearnerHead(learner,model)