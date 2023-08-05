# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/06_Learning/10a_learner.core.ipynb.

# %% auto 0
__all__ = ['LearnerBase', 'LearnerHead', 'StepBatcher']

# %% ../../nbs/06_Learning/10a_learner.core.ipynb 2
# Python native modules
import os
from contextlib import contextmanager
from typing import List,Union
# Third party libs
from fastcore.all import add_docs
import torchdata.datapipes as dp
import torch
from torch import nn
from ..torch_core import evaluating
from torchdata.dataloader2 import DataLoader2
from torchdata.dataloader2.graph import traverse_dps,DataPipeGraph,DataPipe
# Local modules
# from fastrl.core import *
# from fastrl.torch_core import *
from ..pipes.core import find_dp
from ..loggers.core import Record,EpocherCollector
# from fastrl.data.dataloader2 import *

# %% ../../nbs/06_Learning/10a_learner.core.ipynb 4
class LearnerBase(dp.iter.IterDataPipe):
    def __init__(self,
            model:nn.Module, # The base NN that we getting raw action values out of.
            dls:List[DataLoader2], # The dataloaders to read data from for training
            device=None,
            loss_func=None, # The loss function to use
            opt=None, # The optimizer to use
            # LearnerBase will yield each dl individually by default. If `zipwise=True`
            # next() will be called on `dls` and will `yield next(dl1),next(dl2),next(dl1)...`
            zipwise:bool=False,
            # For reinforcement learning, the iterables/workers will live forever and so we dont want
            # to shut them down. We still want a concept of "batch" and "epoch" so this param
            # can handle that.
            batches:int=None,
            # If dl is more than 1, we can switch the dl to use when fitting, or a
            # slice of dls
            fit_idx:Union[int,slice]=0
    ):
        self.loss_func = loss_func
        self.opt = opt
        # TODO: DDPG is demonstrating this drawback. We really should support the 
        # use of multiple models. We might possibly want to just embed the opt and loss 
        # in the models also.
        self.model = model
        self.iterable = dls
        self.zipwise = zipwise
        self.learner_base = self
        self.infinite_dls = False
        self.fit_idx = slice(fit_idx,fit_idx+1) if type(fit_idx)==int else fit_idx
        self._dls = None
        if batches is not None: 
            self.batches = batches
            self.infinite_dls = True
        else:                   
            self.batches = find_dp(traverse_dps(dls[0].datapipe),dp.iter.Header).limit

    def __getstate__(self):
        state = {k:v for k,v in self.__dict__.items() if k not in ['_dls','opt','iterable']}
        # TODO: Needs a better way to serialize / deserialize states.
        # state['iterable'] = [d.state_dict() for d in state['iterable']]
        if dp.iter.IterDataPipe.getstate_hook is not None:
            return dp.iter.IterDataPipe.getstate_hook(state)

        return state

    def __setstate__(self, state):
        # state['iterable'] = [d.from_state_dict() for d in state['iterable']]
        for k,v in state.items():
            setattr(self,k,v)

    def reset(self):
        if not self.infinite_dls:
            self._dls = [iter(dl) for dl in self.iterable]
        elif self._dls is None:
            self._dls = [iter(dl) for dl in self.iterable]
            
    def increment_batch(self,value):
        return not isinstance(value,
            (Record,)
        )
        # return not isinstance(value,
        #     (Record,GetInputItemResponse)
        # )
            
    def __iter__(self):
        self.reset()
        exhausted = []
        zip_list = []
        dl_batch_tracker = [0 for _ in self._dls[self.fit_idx]]
        while len(exhausted)!=len(self._dls[self.fit_idx]):
            # zip_list = []
            for i,dl in enumerate(self._dls[self.fit_idx]):
                if self.zipwise:

                            if i in exhausted: 
                                zip_list.append(None)
                            else:              
                                try: 
                                    zip_list.append(next(dl))
                                    if self.increment_batch(zip_list[-1]): dl_batch_tracker[i]+=1
                                    if self.infinite_dls and dl_batch_tracker[i]>=self.batches:
                                        raise StopIteration
                                except StopIteration:
                                    exhausted.append(i)
                                    zip_list.append(None)
                else:
            # while len(exhausted)!=len(self._dls[self.fit_idx]):
            #     for i,dl in enumerate(self._dls[self.fit_idx]): 
                    while i not in exhausted:
                        try:
                            v = next(dl)
                            if self.increment_batch(v): dl_batch_tracker[i]+=1
                            yield v
                            if self.infinite_dls and dl_batch_tracker[i]>=self.batches:
                                raise StopIteration
                        except StopIteration:
                            exhausted.append(i)

add_docs(
LearnerBase,
"Combines models,dataloaders, and optimizers together for running a training pipeline.",
reset="""If `infinite_dls` is false, then all dls will be reset, otherwise they will be
kept alive.""",
increment_batch="Decides when a single batch is actually 'complete'."
)

# %% ../../nbs/06_Learning/10a_learner.core.ipynb 5
class LearnerHead(dp.iter.IterDataPipe):
    def __init__(self,source_datapipe):
        self.source_datapipe = source_datapipe
        self.learner_base = find_dp(traverse_dps(self.source_datapipe),LearnerBase)

    def __iter__(self): yield from self.source_datapipe
    
    def fit(self,epochs):
        epocher = find_dp(traverse_dps(self),EpocherCollector)
        epocher.epochs = epochs
        
        for iteration in self: 
            pass

    def validate(self,epochs=1,dl_idx=1) -> DataPipe:
        with evaluating(self.learner_base.model):
            for epoch in range(epochs):
                for el in self.learner_base.iterable[dl_idx]:pass 

            pipe = self.learner_base.iterable[dl_idx].datapipe
            return pipe.show() if hasattr(pipe,'show') else pipe
        
add_docs(
LearnerHead,
"""
""",
fit="Runs the `LearnerHead` pipeline for `epochs`",
validate="""If there is more than 1 dl, then run 1 epoch of that dl based on 
`dl_idx` and returns the original datapipe for displaying."""
)  

# %% ../../nbs/06_Learning/10a_learner.core.ipynb 15
class StepBatcher(dp.iter.IterDataPipe):
    def __init__(self,
            source_datapipe,
            device=None
        ):
        self.source_datapipe = source_datapipe
        self.device = device
        
    def vstack_by_fld(self,batch,fld):
        try:
            if self.device is None: return torch.vstack(tuple(getattr(step,fld) for step in batch))
            return torch.vstack(tuple(getattr(step,fld) for step in batch)).to(torch.device(self.device))
        except RuntimeError as e:
            print(f'Failed to stack {fld} given batch: {batch}')
            raise
        
    def __iter__(self):
        for batch in self.source_datapipe:
            cls = batch[0].__class__
            yield cls(**{fld:self.vstack_by_fld(batch,fld) for fld in cls._fields})

add_docs(
StepBatcher,
"Converts multiple `StepType` into a single `StepType` with the fields concated.",
vstack_by_fld="vstacks a `fld` in `batch`"
)
