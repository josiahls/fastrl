# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/06_Learning/10a_learner.core.ipynb.

# %% auto 0
__all__ = ['LearnerBase', 'evaluating', 'LearnerHead']

# %% ../../nbs/06_Learning/10a_learner.core.ipynb 3
# Python native modules
import os
from contextlib import contextmanager
from typing import *
# Third party libs
from fastcore.all import *
import torchdata.datapipes as dp
import torch
from ..torch_core import *
from torchdata.dataloader2 import DataLoader2
from torchdata.dataloader2.graph import find_dps,traverse,DataPipeGraph,Type,DataPipe
# Local modules
from ..core import *
from ..pipes.core import *
from ..loggers.core import *
from ..data.dataloader2 import *

# %% ../../nbs/06_Learning/10a_learner.core.ipynb 5
class LearnerBase(dp.iter.IterDataPipe):
    def __init__(self,
            model:Module, # The base NN that we getting raw action values out of.
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
            self.batches = find_dp(traverse(dls[0].datapipe,only_datapipe=True),dp.iter.Header).limit

    def __getstate__(self):
        state = super().__getstate__()
        # TODO: Needs a better way to serialize / deserialize states.
        # state['iterable'] = [d.state_dict() for d in state['iterable']]
        return {k:v for k,v in state.items() if k not in ['_dls','opt','iterable']}

    def __setstate__(self, state):
        # state['iterable'] = [d.from_state_dict() for d in state['iterable']]
        super().__setstate__(state)

    def reset(self):
        if not self.infinite_dls:
            self._dls = [iter(dl) for dl in self.iterable]
        elif self._dls is None:
            self._dls = [iter(dl) for dl in self.iterable]
            
    def increment_batch(self,value):
        return not isinstance(value,
            (Record,GetInputItemResponse)
        )
            
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
                                    if self.infinite_dls and dl_batch_tracker[i]>self.batches:
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
                            if self.infinite_dls and dl_batch_tracker[i]>self.batches:
                                raise StopIteration
                        except StopIteration:
                            exhausted.append(i)

# %% ../../nbs/06_Learning/10a_learner.core.ipynb 6
@contextmanager
def evaluating(model):
    "Temporarily switch to evaluation mode."
    istrain = model.training
    try:
        model.eval()
        yield model
    finally:
        if istrain:
            model.train()

# %% ../../nbs/06_Learning/10a_learner.core.ipynb 7
class LearnerHead(dp.iter.IterDataPipe):
    def __init__(self,source_datapipe):
        self.source_datapipe = source_datapipe
        self.learner_base = find_dp(traverse(self.source_datapipe),LearnerBase)

    def __iter__(self): yield from self.source_datapipe
    
    def fit(self,epochs):
        epocher = find_dp(traverse(self),EpocherCollector)
        epocher.epochs = epochs
        
        for iteration in self: 
            pass

    def validate(self,epochs=1,dl_idx=1) -> DataPipe:
        with evaluating(self.learner_base.model):
            for epoch in range(epochs):
                for el in self.learner_base.iterable[dl_idx]:pass 

            pipe = self.learner_base.iterable[dl_idx].datapipe
            return pipe.show() if hasattr(pipe,'show') else pip
        
add_docs(
LearnerHead,
"""
""",
fit="Runs the `LearnerHead` pipeline for `epochs`",
validate="""If there is more than 1 dl, then run 1 epoch of that dl based on 
`dl_idx` and returns the original datapipe for displaying."""
)  
