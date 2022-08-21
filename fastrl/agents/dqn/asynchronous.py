# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/12i_agents.dqn.asynchronous.ipynb.

# %% auto 0
__all__ = ['ModelSubscriber', 'ModelPublisher', 'DQNLearner', 'DQNAgent']

# %% ../nbs/12i_agents.dqn.asynchronous.ipynb 3
# Python native modules
import os
from collections import deque
# Third party libs
from fastcore.all import *
import torchdata.datapipes as dp
from torch.utils.data.dataloader_experimental import DataLoader2
from torch.utils.data.datapipes._typing import _DataPipeMeta, _IterDataPipeMeta
import torch.multiprocessing as mp
import torch
from torch.nn import *
import torch.nn.functional as F
from torch.optim import *

from fastai.torch_basics import *
from fastai.torch_core import *
# Local modules

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

# %% ../nbs/12i_agents.dqn.asynchronous.ipynb 8
class ModelSubscriber(dp.iter.IterDataPipe):
    "If an agent is passed to another process and 'spawn' start method is used, then this module is needed."
    def __init__(self,
                 source_datapipe,
                 device:str='cpu'
                ): 
        super().__init__()
        self.source_datapipe = source_datapipe
        self.model = find_pipe_instance(self.source_datapipe,AgentBase).model
        self.main_queue = self.initialize_queue()
        self.device = device
        
    def initialize_queue(self):
        "If the start method is `spawn` then the queue will need to be managed using a Manager."
        if mp.get_start_method()=='spawn':
            ctx = mp.get_context('spawn')
            manager = ctx.Manager()
            queue = manager.Queue()
            return queue
        else:
            return mp.Queue()
    
    def __iter__(self):
        for x in self.source_datapipe:
            if not self.main_queue.empty():
                state = self.main_queue.get(timeout=1)
                self.model.load_state_dict(state)
                self.model.to(device=torch.device(self.device))
            yield x

# %% ../nbs/12i_agents.dqn.asynchronous.ipynb 9
class ModelPublisher(dp.iter.IterDataPipe):
    def __init__(self,source_datapipe,
                 agents=None,
                 publish_freq:int=1
                ):
        super().__init__()
        self.source_datapipe = source_datapipe
        if not isinstance(agents,(list,tuple)): raise ValueError(f'Agents must be a list or tuple, not {type(agents)}')
        self.queues = [find_pipe_instance(agent,ModelSubscriber).main_queue for agent in agents]
        self.model = find_pipe_instance(self,LearnerBase).model
        self.publish_freq = publish_freq
                
    def __iter__(self):
        for i,batch in enumerate(self.source_datapipe):
            if i%self.publish_freq==0:
                for q in self.queues: 
                    with torch.no_grad():
                        q.put(deepcopy(self.model).cpu().state_dict())
            yield batch

# %% ../nbs/12i_agents.dqn.asynchronous.ipynb 10
def DQNLearner(
    model,
    dls,
    agent,
    logger_bases=None,
    loss_func=MSELoss(),
    opt=AdamW,
    lr=0.005,
    bs=128,
    max_sz=10000,
    nsteps=1,
    device=None
) -> LearnerHead:
    learner = LearnerBase(model,dls,loss_func=MSELoss(),opt=opt(model.parameters(),lr=lr))
    learner = ModelPublisher(learner,agent)
    learner = BatchCollector(learner,logger_bases=logger_bases,batch_on_pipe=LearnerBase)
    learner = EpocherCollector(learner,logger_bases=logger_bases)
    for logger_base in L(logger_bases): learner = logger_base.connect_source_datapipe(learner)
    if logger_bases: 
        learner = RollingTerminatedRewardCollector(learner,logger_bases)
        learner = EpisodeCollector(learner,logger_bases)
    learner = ExperienceReplay(learner,bs=bs,max_sz=max_sz,clone_detach=dls[0].num_workers>0)
    learner = StepBatcher(learner,device=device)
    learner = QCalc(learner,nsteps=nsteps)
    learner = ModelLearnCalc(learner)
    if logger_bases: 
        learner = LossCollector(learner,logger_bases)
    learner = LearnerHead(learner)
    return learner

# %% ../nbs/12i_agents.dqn.asynchronous.ipynb 11
def DQNAgent(
    model,
    logger_bases=None,
    min_epsilon=0.02,
    max_epsilon=1,
    max_steps=1000,
    device='cpu'
)->AgentHead:
    agent = AgentBase(model)
    agent = StepFieldSelector(agent,field='state')
    agent = ModelSubscriber(agent,device=device)
    agent = SimpleModelRunner(agent,device=device)
    agent = ArgMaxer(agent)
    selector = EpsilonSelector(agent,min_epsilon=min_epsilon,max_epsilon=max_epsilon,max_steps=max_steps,device=device)
    if logger_bases is not None: agent = EpsilonCollector(selector,logger_bases)
    agent = ArgMaxer(agent,only_idx=True)
    agent = NumpyConverter(agent)
    agent = PyPrimativeConverter(agent)
    agent = AgentHead(agent)
    return agent