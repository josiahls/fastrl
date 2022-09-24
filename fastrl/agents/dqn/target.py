# AUTOGENERATED! DO NOT EDIT! File to edit: ../../../nbs/07_Agents/01_Discrete/12h_agents.dqn.target.ipynb.

# %% auto 0
__all__ = ['TargetModelUpdater', 'TargetModelQCalc']

# %% ../../../nbs/07_Agents/01_Discrete/12h_agents.dqn.target.ipynb 3
# Python native modules
import os
from collections import deque
from copy import deepcopy
from typing import Callable
# Third party libs
from fastcore.all import *
import torchdata.datapipes as dp
from torch.utils.data.dataloader_experimental import DataLoader2
from torch.utils.data.datapipes._typing import _DataPipeMeta, _IterDataPipeMeta
from torchdata.dataloader2.graph import find_dps,traverse,DataPipe,replace_dp,remove_dp
import torch
from torch.nn import *
import torch.nn.functional as F
from torch.optim import *

from ...torch_core import *
# Local modules

from ...core import *
from ..core import *
from ...pipes.core import *
from ...data.block import *
from ...memory.experience_replay import *
from ..core import *
from ..discrete import *
from ...loggers.core import *
from ...loggers.vscode_visualizers import *
from ...learner.core import *
from .basic import *

# %% ../../../nbs/07_Agents/01_Discrete/12h_agents.dqn.target.ipynb 6
class TargetModelUpdater(dp.iter.IterDataPipe):
    def __init__(self,source_datapipe=None,target_sync=300):
        self.source_datapipe = source_datapipe
        if source_datapipe is not None:
            self.learner = find_dp(traverse(self),LearnerBase)
            self.learner.target_model = deepcopy(self.learner.model)
        self.target_sync = target_sync
        self.n_batch = 0
        
    def reset(self):
        self.learner = find_dp(traverse(self),LearnerBase)
        self.learner.target_model = deepcopy(self.learner.model)
        
    def __iter__(self):
        if self._snapshot_state.NotStarted: self.reset()
        for batch in self.source_datapipe:
            if self.n_batch%self.target_sync==0:
                self.learner.target_model.load_state_dict(self.learner.model.state_dict())
            self.n_batch+=1
            yield batch

    @classmethod
    def insert_dp(cls,old_dp=ModelLearnCalc,target_sync=300) -> Callable[[DataPipe],DataPipe]:
        def _insert_dp(pipe):
            # v = insert_dp(
            #     traverse(pipe),
            #     find_dp(traverse(pipe),old_dp),
            #     cls(source_datapipe=PassThroughIterPipe(None),target_sync=target_sync)
            # )
            v = replace_dp(
                traverse(pipe,only_datapipe=True),
                find_dp(traverse(pipe,only_datapipe=True),old_dp),
                cls(find_dp(traverse(pipe,only_datapipe=True),old_dp),target_sync=target_sync)
            )
            return list(v.values())[0][0]
        return _insert_dp

# %% ../../../nbs/07_Agents/01_Discrete/12h_agents.dqn.target.ipynb 7
class TargetModelQCalc(dp.iter.IterDataPipe):
    def __init__(self,source_datapipe=None):
        self.source_datapipe = source_datapipe
        if source_datapipe is not None: self.learner = find_dp(traverse(self),LearnerBase)
        
    def reset(self):
        self.learner = find_dp(traverse(self),LearnerBase)
        
    def __iter__(self):
        for batch in self.source_datapipe:
            self.learner.done_mask = batch.terminated.reshape(-1,)
            with torch.no_grad():
                self.learner.next_q = self.learner.target_model(batch.next_state)
            self.learner.next_q = self.learner.next_q.max(dim=1).values.reshape(-1,1)
            self.learner.next_q[self.learner.done_mask] = 0 
            yield batch
            
    @classmethod
    def replace_dp(cls,old_dp=QCalc) -> Callable[[DataPipe],DataPipe]:
        def _replace_dp(pipe):
            old_dp_instance = find_dp(traverse(pipe),old_dp)
            v = replace_dp(
                traverse(pipe),
                old_dp_instance,
                cls(old_dp_instance.source_datapipe)
            )
            return list(v.values())[0][0]
        return _replace_dp
