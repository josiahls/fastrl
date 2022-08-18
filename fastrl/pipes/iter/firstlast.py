# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01f_pipes.iter.firstlast.ipynb.

# %% auto 0
__all__ = ['FirstLastMerger', 'n_first_last_steps_expected']

# %% ../nbs/01f_pipes.iter.firstlast.ipynb 3
# Python native modules
import os
from warnings import warn
# Third party libs
from fastcore.all import *
import torchdata.datapipes as dp
import typing
from fastai.torch_basics import *
from fastai.torch_core import *
# Local modules
from ...core import *
from ..core import *
from ...fastai.data.block import *
from ..core import *

# %% ../nbs/01f_pipes.iter.firstlast.ipynb 5
class FirstLastMerger(dp.iter.IterDataPipe):
    def __init__(self, 
                 source_datapipe, 
                 gamma:float=0.99
        ):
        self.source_datapipe = source_datapipe
        self.gamma = gamma
        
    def __iter__(self) -> StepType:
        self.env_buffer = {}
        for steps in self.source_datapipe:
            if not isinstance(steps,(list,tuple)):
                raise ValueError(f'Expected {self.source_datapipe} to return a list/tuple of steps, however got {type(steps)}')
                
            if len(steps)==1:
                yield steps[0]
                continue
                
            fstep,lstep = steps[0],steps[-1]
            
            reward = fstep.reward
            for step in steps[1:]:
                reward*=self.gamma
                reward+=step.reward
                
            yield SimpleStep(
                state=tensor(fstep.state),
                next_state=tensor(lstep.next_state),
                action=fstep.action,
                terminated=lstep.terminated,
                truncated=lstep.truncated,
                reward=reward,
                total_reward=lstep.total_reward,
                env_id=lstep.env_id,
                proc_id=lstep.proc_id,
                step_n=lstep.step_n,
                episode_n=fstep.episode_n,
                image=fstep.image
            )
                
add_docs(
    FirstLastMerger,
    """Takes multiple steps and converts them into a single step consisting of properties
    from the first and last steps. Reward is recalculated to factor in the multiple steps.""",
)

# %% ../nbs/01f_pipes.iter.firstlast.ipynb 13
def n_first_last_steps_expected(
    default_steps:int, # The number of steps the episode would run without n_steps
):
    return default_steps 
    
n_first_last_steps_expected.__doc__=r"""
This function doesnt do much for now. `FirstLastMerger` pretty much undoes the number of steps `nsteps` does.
"""    
