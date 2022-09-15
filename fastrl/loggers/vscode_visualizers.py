# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/05_Logging/09f_loggers.vscode_visualizers.ipynb.

# %% auto 0
__all__ = ['SimpleVSCodeVideoPlayer', 'VSCodeTransformBlock']

# %% ../../nbs/05_Logging/09f_loggers.vscode_visualizers.ipynb 1
# Python native modules
import os
import io
from typing import *
# Third party libs
import imageio
from fastcore.all import *
import matplotlib.pyplot as plt
import torchdata.datapipes as dp
from IPython.core.display import Video,Image
from torchdata.dataloader2.dataloader2 import DataLoader2
# Local modules
from ..core import *
from .core import *
from ..data.dataloader2 import *
from ..data.block import *
from ..pipes.core import *
from .jupyter_visualizers import ImageCollector

# %% ../../nbs/05_Logging/09f_loggers.vscode_visualizers.ipynb 5
class SimpleVSCodeVideoPlayer(LoggerBase):
    def __init__(self, 
                 source_datapipe=None, 
                 skip_frames:int=1,
                 fps:int=30,
                 downsize_res=(2,2)
        ):
        super().__init__(source_datapipe)
        self.source_datapipe = source_datapipe
        self.fps = fps
        self.skip_frames = skip_frames
        self.downsize_res = downsize_res
        self._bytes_object = None
        self.frames = [] 

    def reset(self):
        super().reset()
        self._bytes_object = io.BytesIO()

    def show(self,start:int=0,end:Optional[int]=None,step:int=1):
        print(f'Creating gif from {len(self.frames)} frames')
        imageio.mimwrite(
            self._bytes_object,
            self.frames[start:end:step],
            format='GIF',
            fps=self.fps
        )
        return Image(self._bytes_object.getvalue())
        
    def __iter__(self) -> typing.Tuple[typing.NamedTuple]:
        n_frame = 0
        for record in self.source_datapipe:
            for o in self.dequeue():
                if o.value is None: continue
                n_frame += 1
                if n_frame%self.skip_frames!=0: continue
                self.frames.append(
                    o.value[::self.downsize_res[0],::self.downsize_res[1]]
                )
            yield record
add_docs(
SimpleVSCodeVideoPlayer,
"""Displays video from a `source_datapipe` that produces `typing.NamedTuples` that contain an `image` field.
This only can handle 1 env input.""",
dequeue="Grabs records from the `main_queue` and attempts to display them",
show="In order to show the video, this must be called in a notebook cell.",
reset="Will reset the bytes object that is used to store file data."
)

# %% ../../nbs/05_Logging/09f_loggers.vscode_visualizers.ipynb 6
def VSCodeTransformBlock(
    # Additional pipelines to insert, replace, remove
    dp_augmentation_fns:Tuple[DataPipeAugmentationFn]=None
) -> TransformBlock:
    "Basic OpenAi gym `DataPipeGraph` with first-last, nstep, and nskip capability"
    def _VSCodeTransformBlock(
        # `source` likely will be an iterable that gets pushed into the pipeline when an 
        # experiment is actually being run.
        source:Any,
        # Any parameters needed for the dataloader
        num_workers:int=0,
        # This param must exist: as_dataloader for the datablock to create dataloaders
        as_dataloader:bool=False
    ) -> DataPipeOrDataLoader:
        "This is the function that is actually run by `DataBlock`"
        video_logger = SimpleVSCodeVideoPlayer()
        pipe = LoggerBasePassThrough(source,[video_logger])
        pipe = ImageCollector(pipe)
        pipe = video_logger.connect_source_datapipe(pipe)
        pipe = apply_dp_augmentation_fns(pipe,ifnone(dp_augmentation_fns,()))
        
        if as_dataloader:
            pipe = DataLoader2(
                datapipe=pipe,
                reading_service=PrototypeMultiProcessingReadingService(
                    num_workers = num_workers,
                    protocol_client_type = InputItemIterDataPipeQueueProtocolClient,
                    protocol_server_type = InputItemIterDataPipeQueueProtocolServer,
                    pipe_type = item_input_pipe_type,
                    eventloop = SpawnProcessForDataPipeline
                ) if num_workers>0 else None
            )
        return pipe 
    return _VSCodeTransformBlock
