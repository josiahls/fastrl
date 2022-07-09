# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/05a_envs.pipes_core.nskip.ipynb (unless otherwise specified).

__all__ = ['NSkipper', 'pipe', 'pipe', 'n_skips_expected']

# Cell
# Python native modules
import os
# Third party libs
from fastcore.all import *
import torchdata.datapipes as dp
import typing
from fastai.torch_basics import *
from fastai.torch_core import *
# Local modules
from ...core import *
from ...fastai.data.pipes.core import *
from ...fastai.data.load import *
from ...fastai.data.block import *

from .nstep import *

# Cell
_msg = """
NSkipper should not go after NStepper. Please make the order:

```python
...
pipe = NSkipper(pipe,n=3)
pipe = NStepper(pipe,n=3)
...
```

"""

class NSkipper(dp.iter.IterDataPipe):
    def __init__(self, source_datapipe, n=1) -> None:
        if isinstance(source_datapipe,NStepper): raise Exception(_msg)
        self.source_datapipe = source_datapipe
        self.n = n
        self.env_buffer = {}

    def __iter__(self) -> typing.NamedTuple:
        self.env_buffer = {}
        for step in self.source_datapipe:
            if not issubclass(step.__class__,StepType):
                raise Exception(f'Expected typing.NamedTuple object got {type(step)}\n{step}')

            env_id,done,step_n = int(step.env_id),bool(step.done),int(step.step_n)

            if env_id in self.env_buffer: self.env_buffer[env_id] += 1
            else:                         self.env_buffer[env_id] = 1

            if self.env_buffer[env_id]%self.n==0: yield step
            elif done:                            yield step
            elif step_n==1:                       yield step

            if done: self.env_buffer[env_id] = 1

add_docs(
    NSkipper,
    """Accepts a `source_datapipe` or iterable whose `next()` produces a `typing.NamedTuple` that
       skips N steps for individual environments *while always producing 1st steps and done steps.*
    """,
)

# Cell
def n_skips_expected(
    default_steps:int, # The number of steps the episode would run without n_skips
    n:int # The n-skip value that we are planning to use
):
    if n==1: return default_steps # All the steps will eb retained including the 1st step. No offset needed
    # If n goes into default_steps evenly, then the final "done" will be technically an "extra" step
    elif default_steps%n==0: return (default_steps // n) + 1 # first step will be kept
    else:
        # If the steps dont divide evenly then it will attempt to skip done, but ofcourse, we dont
        # let that happen
        return (default_steps // n) + 2 # first step and done will be kept

n_skips_expected.__doc__=r"""
Produces the expected number of steps, assuming a fully deterministic episode based on `default_steps` and `n`

Given `n=2`, given 1 envs, knowing that `CartPole-v1` when `seed=0` will always run 18 steps, the total
steps will be:

$$
18 // n + 1 (1st+last)
$$
"""