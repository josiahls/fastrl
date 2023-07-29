# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/05_Logging/09d_loggers.jupyter_visualizers.ipynb.

# %% auto 0
__all__ = ['SimpleJupyterVideoPlayer', 'ImageCollector']

# %% ../../nbs/05_Logging/09d_loggers.jupyter_visualizers.ipynb 1
# Python native modules
import os
from torch.multiprocessing import Queue
from typing import Tuple,NamedTuple
# Third party libs
from fastcore.all import add_docs
import matplotlib.pyplot as plt
import torchdata.datapipes as dp
from IPython.core.display import clear_output
import torch
import numpy as np
# Local modules
from ..core import Record
from .core import LoggerBase,LogCollector,LoggerBasePassThrough
# from fastrl.torch_core import *

# %% ../../nbs/05_Logging/09d_loggers.jupyter_visualizers.ipynb 4
class SimpleJupyterVideoPlayer(LoggerBase):
    def __init__(self, 
                 source_datapipe=None, 
                 between_frame_wait_seconds:float=0.1
        ):
        super().__init__(source_datapipe)
        self.source_datapipe = source_datapipe
        self.between_frame_wait_seconds = 0.1
        
    def __iter__(self) -> Tuple[NamedTuple]:
        img = None
        for record in self.source_datapipe:
            for o in self.dequeue():
                if o.value is None: continue
                if img is None: img = plt.imshow(o.value)
                img.set_data(o.value) 
                plt.axis('off')
                display(plt.gcf())
                clear_output(wait=True)
            yield record
add_docs(
    SimpleJupyterVideoPlayer,
    """Displays video from a `source_datapipe` that produces `typing.NamedTuples` that contain an `image` field.
       This only can handle 1 env input.""",
    dequeue="Grabs records from the `main_queue` and attempts to display them"
)

# %% ../../nbs/05_Logging/09d_loggers.jupyter_visualizers.ipynb 5
class ImageCollector(LogCollector):
    header:str='image'

    def convert_np(self,o):
        if isinstance(o,torch.Tensor): return o.detach().numpy()
        elif isinstance(o,np.ndarray): return o
        else:                          raise ValueError(f'Expects Tensor or np.ndarray not {type(o)}')
    
    def __iter__(self):
        # for q in self.main_buffers: q.append(Record('image',None))
        for steps in self.source_datapipe:
            if isinstance(steps,dp.DataChunk):
                for step in steps:
                    for q in self.main_buffers: 
                        q.append(Record('image',self.convert_np(step.image)))
            else:
                for q in self.main_buffers: q.append(Record('image',self.convert_np(steps.image)))
            yield steps
