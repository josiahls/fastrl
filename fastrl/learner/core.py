# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/10a_learner.core.ipynb.

# %% auto 0
__all__ = ['LearnerBase', 'LearnerHead']

# %% ../nbs/10a_learner.core.ipynb 3
# Python native modules
import os
# Third party libs
from fastcore.all import *
import torchdata.datapipes as dp
import torch
from fastai.torch_basics import *
from fastai.torch_core import *
from torchdata.dataloader2 import DataLoader2
from torchdata.dataloader2.graph import find_dps,traverse
# Local modules
from ..core import *
from ..pipes.core import *
from ..loggers.core import *
from ..dataloader2_ext import *

# %% ../nbs/10a_learner.core.ipynb 5
class LearnerBase(dp.iter.IterDataPipe):
    def __init__(self,
            model:Module, # The base NN that we getting raw action values out of.
            dls:List[DataLoader2], # The dataloaders to read data from for training
            loss_func=None, # The loss function to use
            opt=None, # The optimizer to use
            # LearnerBase will yield each dl individually by default. If `zipwise=True`
            # next() will be called on `dls` and will `yield next(dl1),next(dl2),next(dl1)...`
            zipwise:bool=False,
            # For reinforcement learning, the iterables/workers will live forever and so we dont want
            # to shut them down. We still want a concept of "batch" and "epoch" so this param
            # can handle that.
            batches:int=None
    ):
        self.loss_func = loss_func
        self.opt = opt
        self.model = model
        self.iterable = dls
        self.zipwise = zipwise
        self.learner_base = self
        self.infinite_dls = False
        self._dls = None
        if batches is not None: 
            self.batches = batches
            self.infinite_dls = True
        else:                   
            self.batches = find_dp(traverse(dls[0].datapipe,only_datapipe=True),dp.iter.Header).limit

    def reset(self):
        if not self.infinite_dls:
            self._dls = [iter(dl) for dl in self.iterable]
        elif self._dls is None:
            self._dls = [iter(dl) for dl in self.iterable]
            
    def increment_batch(self,value):
        # I dont make this inline, because there is a likihood we will have additional conditions
        # and I want to actually be able to read and understand each one...
        if type(value)==Record: return False
        if type(value)==GetInputItemResponse: return False
        return True
            
    def __iter__(self):
        self.reset()
        exhausted = []
        dl_batch_tracker = [0 for _ in self._dls]
        if self.zipwise:
            while len(exhausted)!=len(self._dls):
                zip_list = []
                for i,dl in self._dls:
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
            while len(exhausted)!=len(self._dls):
                for i,dl in enumerate(self._dls): 
                    while i not in exhausted:
                        try:
                            v = next(dl)
                            if self.increment_batch(v): dl_batch_tracker[i]+=1
                            yield v
                            if self.infinite_dls and dl_batch_tracker[i]>self.batches:
                                raise StopIteration
                        except StopIteration:
                            exhausted.append(i)

# %% ../nbs/10a_learner.core.ipynb 6
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
        
add_docs(
    LearnerHead,
    """
    """,
    fit="Runs the `LearnerHead` pipeline for `epochs`"
)  
