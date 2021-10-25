# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/06a_memory.experience_replay.ipynb (unless otherwise specified).

__all__ = ['ExperienceReplayException', 'ExperienceReplay', 'ExperienceReplayCallback', 'snapshot_memory',
           'ExperienceReplayTensorboard']

# Cell
# Python native modules
import os
from typing import *
from warnings import warn
# Third party libs
from fastcore.all import *
from fastai.learner import *
from fastai.torch_basics import *
from fastai.torch_core import *
from fastai.callback.all import *
from torch.utils.tensorboard import SummaryWriter
# Local modules
from ..core import *
from ..callback.core import *
from ..data.block import *

# Cell
class ExperienceReplayException(Exception): pass

class ExperienceReplay(object):
    def __init__(self,
                 bs:int=16,         # Number of entries to query from memory
                 max_sz:int=200,    # Maximum number of entries to hold. Will start overwriting after.
                 warmup_sz:int=100,  # Minimum number of entries needed to continue with a batch
                 # Used for testing. Once the memory has reached max size, it will not
                 # Add any more data. This is useful for checking whether a model is training correctly.
                 freeze_at_max:bool=False,
                 memory:Optional[BD]=None # Optionally, you can initialize a new `ExperienceReplay` with an existing dictionary
                 ):
        "Stores `BD`s in a rotating list `self.memory`"
        store_attr()
        test_lt(warmup_sz-1,max_sz)
        self.memory=memory
        self.pointer=0

    def __add__(self,other:BD):
        "In-place add `other` to memory, overwriting if len(self.memory)>self.max_sz"
        if isinstance(other,tuple) and len(other)==1: other=other[0]
        elif isinstance(other,tuple):                 raise ExperienceReplayException('records need to be `BD`s or 1 element tuples')
        if isinstance(other,dict):                    other=BD(other)
        elif isinstance(other,list):                  other=sum(other)

        if 'td_error' not in other: other['td_error']=TensorBatch(torch.zeros((other.bs(),1)))

        if self.memory is None:
            if other.bs()>self.max_sz:
                self.memory=other[:self.max_sz]
                self.pointer=0           # Keep the pointer 0 since we have basically replaced the memory
                self+other[self.max_sz:] # Recursively add the rest of the batch
            else:
                self.memory=other
                self.pointer=self.memory.bs() # remember that pointer is not an index but number of elements
        else:
            if self.freeze_at_max and self.memory.bs()>=self.max_sz: return self
            n_over=(other.bs()+self.pointer)-self.max_sz
            if n_over>0: # e.g.: max_sz 200, pointer 195, other is 5.
                self.memory=self.memory[:self.pointer]+other[:-n_over]
                self.pointer=0
                self+other[other.bs()-n_over:]
            else:
                # If the number of elements is not over
                next_pointer=self.pointer+other.bs()
                self.memory=self.memory[:self.pointer]+other+self.memory[next_pointer:]
                self.pointer=next_pointer
        return self

    def __getitem__(self,i):
        return ExperienceReplay(bs=self.bs,max_sz=self.max_sz,
                                warmup_sz=self.warmup_sz,memory=self.memory[i])

    def __radd__(self,other:BD): raise ExperienceReplayException('You can only do experience_reply+[some other element]')

    def __len__(self): return self.memory.bs() if self.memory is not None else 0

    def sample(self)->BD:
        "Returns a sample of size `self.bs`"
        with torch.no_grad():
            idxs=np.random.randint(0,self.memory.bs(),self.bs).tolist()
            samples=self.memory[idxs].mapv(to_device)

        if self.memory.bs()<self.warmup_sz: raise CancelBatchException
        return samples,idxs

    def update_td(self,td_errors:Tensor,idxs:Tensor):
        if not isinstance(idxs,list):
            test_len(idxs.shape,1)
        test_len(td_errors.shape,2)
        self.memory['td_error'][idxs]=to_detach(td_errors)

# Cell
class ExperienceReplayCallback(Callback):
    @delegates(ExperienceReplay)
    def __init__(self,
                 verbose=False, # Will show warnings for recommended behavior.
                 **kwargs):
        "Stores `BD`s in a rotating list `self.memory`"
        store_attr()
        self._kwargs=kwargs

    def before_fit(self):
        if not hasattr(self.learn,'experience_replay') or \
           not isinstance(self.learn.experience_replay,ExperienceReplay):
            self.learn.experience_replay=ExperienceReplay(**self._kwargs)

    def after_pred(self):
        "Adds `learn.xb` to memory, then sets `learn.xb=experience_replay.sample()`"
        xb=BD(self.learn.xb[0]).mapv(to_detach)
        self.learn.experience_replay+xb

        self.learn.xb,self.learn.sample_indexes=self.experience_replay.sample()

    def after_batch(self):
        if hasattr(self.learn,'td_error'):
            self.experience_replay.update_td(
                self.td_error,
                self.sample_indexes
            )
        elif self.verbose:
            warn("""The learner does not have a `td_error` field. Produced logs
                    will not be useful unless `td_error` exists.""")

# Cell
def snapshot_memory(writer:SummaryWriter,epoch:int,experience_replay,prefix='experience_replay'):
    for i,v in enumerate(experience_replay.memory['td_error'].numpy().reshape(-1)):
        writer.add_scalar(f'{prefix}/{epoch}/td_error',v,i)

    if 'image' not in experience_replay.memory:
        warn('image is missing from the experience replay. Image section of the replay will not be logged.')
        return

    for i,frame in enumerate(experience_replay.memory['image'].permute(0,3, 1, 2)):
        writer.add_video(f'{prefix}/{epoch}/video',frame.unsqueeze(0).unsqueeze(0),global_step=i)

# Cell
class ExperienceReplayTensorboard(Callback):
    def __init__(self,writer=None,comment='',every_epoch=1):
        store_attr()
        self.writer=ifnone(writer,SummaryWriter(comment=comment))

    def before_fit(self):
        if not hasattr(self.learn,'experience_replay'):
            warn('Learner does not have `experience_replay`, nothing will be logged.')

    def after_epoch(self):
        if self.epoch%self.every_epoch==0:
            snapshot_memory(self.writer,epoch=self.epoch,
                            experience_replay=self.learn.experience_replay)