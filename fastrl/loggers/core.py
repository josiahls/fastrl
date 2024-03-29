# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/05_Logging/09a_loggers.core.ipynb.

# %% auto 0
__all__ = ['not_record', 'is_record', 'RecordCatchBufferOverflow', 'RecordCatcher', 'RecordDumper', 'LoggerBase', 'LogCollector',
           'EpochCollector', 'BatchCollector', 'ProgressBarLogger', 'RewardCollector', 'EpisodeCollector',
           'RollingTerminatedRewardCollector']

# %% ../../nbs/05_Logging/09a_loggers.core.ipynb 2
# Python native modules
from typing import Optional,List,Any,Iterable,Union
from collections import deque
from multiprocessing import Queue
from queue import Empty
import logging
# Third party libs
from fastcore.all import add_docs,merge,ifnone
import torchdata.datapipes as dp
from fastprogress.fastprogress import master_bar,progress_bar
from torchdata.dataloader2.graph import find_dps,traverse_dps,list_dps
from torchdata.datapipes import functional_datapipe
from tqdm.auto import tqdm
import pandas as pd
from IPython.display import display,HTML
import numpy as np
# Local modules
from ..pipes.core import find_dp
from ..core import Record,StepTypes

# %% ../../nbs/05_Logging/09a_loggers.core.ipynb 4
_logger = logging.getLogger(__name__)

def not_record(data:Any):
    "Intended for use with dp.iter.Filter"
    return type(data)!=Record
def is_record(data:Any):
    "Intended for use with dp.iter.Filter"
    return type(data)==Record

# %% ../../nbs/05_Logging/09a_loggers.core.ipynb 7
_RECORD_CATCH_LIST = []

class RecordCatchBufferOverflow(Exception):
    def __init__(self,msg, *args, **kwargs):
        msg=f"""_RECORD_CATCH_QUEUE got larger than {msg}.
        
        Make sure that `RecordDumper` or `dump_records` is being called
        at some point in the pipeline, in the current process. Reference
        documentation for examples.
        """
        super().__init__(msg, *args, **kwargs)


def _clear_record_catch_list():
    while _RECORD_CATCH_LIST:
        yield _RECORD_CATCH_LIST.pop(0)

@functional_datapipe("catch_records")
class RecordCatcher(dp.iter.IterDataPipe):
    def __init__(
            self,
            source_datapipe,
            # Max size of _RECORD_CATCH_LIST before raising in exception.
            # Important to avoid memory leaks, and indicates that `dump_records`
            # is not being called or used.
            buffer_size=10000,
            # If True, instead of appending to _RECORD_CATCH_LIST, 
            #  drop the record so it does not continue thorugh the 
            # pipeline.
            drop:bool=False
        ):
        self.source_datapipe = source_datapipe
        self.buffer_size = buffer_size
        self.drop = drop
        if _RECORD_CATCH_LIST and not self.drop:
            _logger.debug(
                "Clearing _RECORD_CATCH_LIST since it is not empty: %s elements",
                len(_RECORD_CATCH_LIST)
            )
            _RECORD_CATCH_LIST.clear()

    def __iter__(self):
        for o in self.source_datapipe:
            if is_record(o):
                if not self.drop:                   
                    _RECORD_CATCH_LIST.append(o)
                    if len(_RECORD_CATCH_LIST) > self.buffer_size:
                        raise RecordCatchBufferOverflow(self.buffer_size)
            else:
                yield o

@functional_datapipe("dump_records")
class RecordDumper(dp.iter.IterDataPipe):
    def __init__(
            self,
            source_datapipe=None
        ):
        self.source_datapipe = source_datapipe

    def __iter__(self):
        if self.source_datapipe is None:
            yield from _clear_record_catch_list()
        else:
            for o in self.source_datapipe:
                yield from _clear_record_catch_list()
                yield o

# %% ../../nbs/05_Logging/09a_loggers.core.ipynb 10
class LoggerBase(object):
    debug:bool
    buffer:list
    source_datapipe:dp.iter.IterDataPipe
    
    def dequeue(self): 
        while self.buffer: yield self.buffer.pop(0)
    
    # def reset(self):
        # Note: trying to decide if this is really needed.
        # if self.debug:
        #     print(self,' resetting buffer.')
        # if self._snapshot_state!=_SnapshotState.Restored:
        #     self.buffer = []
        
add_docs(
LoggerBase,
"""The `LoggerBase` class is an iterface for datapipes that also collect `Record` objects
for logging purposes.
""",
dequeue="Empties the `self.buffer` yielding each of its contents."
)        

# %% ../../nbs/05_Logging/09a_loggers.core.ipynb 12
class LogCollector(object):
    debug:bool=False
    title:Optional[str] = None
    main_buffers:Optional[List] = None        

    def enqueue_title(self):
        "Sends a empty `Record` to tell all the `LoggerBase`s of the `LogCollector's` existance."
        record = Record(self.title,None)
        if self.main_buffers is not None:
            for q in self.main_buffers: 
                q.append(record)
        return record
    
    def enqueue_value(
        self,
        value:Any
    ):
        "Sends a `Record` with `value` to all `LoggerBase`s"
        record = Record(self.title,value)
        if self.main_buffers is not None:
            for q in self.main_buffers:
                q.append(Record(self.title,value))
        return record

    def reset(self):
        if self.main_buffers is None:
            if self.debug: print(f'Resetting {self}')
            logger_bases = list_dps(traverse_dps(self))
            logger_bases = [o for o in logger_bases if isinstance(o,LoggerBase)]
            self.main_buffers = [o.buffer for o in logger_bases]
            self.enqueue_title()

add_docs(
LogCollector,
"""`LogCollector` specifically manages finding and attaching itself to
`LoggerBase`s found earlier in the pipeline.""",
reset="Grabs buffers from all logger bases in the pipeline."
)  

# %% ../../nbs/05_Logging/09a_loggers.core.ipynb 14
class EpochCollector(dp.iter.IterDataPipe):
    debug:bool=False
    title:str='epoch'

    def __init__(self,
            source_datapipe,
            # Epochs is the number of times we iterate, and exhaust `source_datapipe`.
            # This is expected behavior of more traditional dataset iteration where
            # an epoch is a single full run through of a dataset.
            epochs:int=0
        ):
        self.source_datapipe = source_datapipe
        self.main_buffers = None
        self.iteration_started = False
        self.epochs = epochs
        self.epoch = 0

    def __iter__(self): 
        if self.main_buffers is None:
            yield Record(self.title,None)
        for i in range(self.epochs):
            # self.reset() 
            self.epoch = i
            yield from self.source_datapipe
            yield Record(self.title,self.epoch)
            
add_docs(
EpochCollector,
"""Tracks the number of epochs that the pipeline is currently on.""",
reset="Grabs buffers from all logger bases in the pipeline."
)

# %% ../../nbs/05_Logging/09a_loggers.core.ipynb 16
class BatchCollector(dp.iter.IterDataPipe, LogCollector):
    title:str='batch'

    def __init__(self,
            source_datapipe,
            batches:Optional[int]=None,
            # If `batches` is None, `BatchCollector` with try to find: `batch_on_pipe` instance
            # and try to grab a `batches` field from there.
            batch_on_pipe:dp.iter.IterDataPipe=None 
        ):
        self.source_datapipe = source_datapipe
        self.main_buffers = None
        self.iteration_started = False
        self.batches = (
            batches if batches is not None else self.batch_on_pipe_get_batches(batch_on_pipe)
        )
        self.batch = 0
        
    def batch_on_pipe_get_batches(self,batch_on_pipe):
        pipe = find_dp(traverse_dps(self.source_datapipe),batch_on_pipe)
        if hasattr(pipe,'batches'):
            return pipe.batches
        elif hasattr(pipe,'limit'):
            return pipe.limit
        else:
            raise RuntimeError(f'Pipe {pipe} isnt recognized as a batch tracker.')

    def __iter__(self): 
        self.batch = 0
        if self.main_buffers is None:
            yield Record(self.title,None)
        for data in self.source_datapipe: 
            yield data
            if not_record(data):
                record = Record(self.title,self.batch)
                self.batch += 1
                yield record
            if self.batch>=self.batches: 
                break

add_docs(
BatchCollector,
"""Tracks the number of batches that the pipeline is currently on.""",
batch_on_pipe_get_batches="Gets the number of batches from `batch_on_pipe`",
reset="Grabs buffers from all logger bases in the pipeline."
)

# %% ../../nbs/05_Logging/09a_loggers.core.ipynb 18
#TODO(josiahls): Put this in a different module, maybe in jupyter visualizers
# so logger.core doesn't force the user to need tqdm or ipython installed.
class ProgressBarLogger(dp.iter.IterDataPipe):
    def __init__(
            self,
            # An iterable that yields `Any` data, which will be passed through,
            # of `Record`  objects
            source_datapipe:Iterable[Union[Any,Record]], 
            # For automatic pipe attaching, we can designate which pipe this should be
            # referneced for information on which epoch we are on
            epoch_on_pipe:dp.iter.IterDataPipe=EpochCollector,
            # For automatic pipe attaching, we can designate which pipe this should be
            # referneced for information on which batch we are on
            batch_on_pipe:dp.iter.IterDataPipe=BatchCollector,
            # Whether to close the progress bars at end of iter
            close_bars:bool = False
        ):
        self.source_datapipe = source_datapipe
        self.epoch_on_pipe = epoch_on_pipe
        self.batch_on_pipe = batch_on_pipe
        self.close_bars = close_bars
        
        self.metrics_df = pd.DataFrame()
        self.current_row = pd.Series()
        self._table_ref = None

    def update_dataframe(self):
        new_df = pd.DataFrame([self.current_row])
        self.metrics_df = pd.concat([self.metrics_df, new_df], axis=0, ignore_index=True).fillna(0)
        # Display without index and keep the progress bars persistent
        html_str = self.metrics_df.to_html(index=False)

        # Check if the table is being displayed for the first time
        if self._table_ref is None:
            self._table_ref = display(HTML(html_str),display_id=True)
        else:
            self._table_ref.update(HTML(html_str))

    def __iter__(self):
        epocher = find_dp(traverse_dps(self),self.epoch_on_pipe)
        batcher = find_dp(traverse_dps(self),self.batch_on_pipe)
        master_pbar = tqdm(total=epocher.epochs, desc="Epochs", position=0, leave=False)
        batch_pbar = tqdm(total=batcher.batches, desc="Batches", position=1, leave=False)
        
        for data in self.source_datapipe:
            if is_record(data):
                self.current_row[data.name] = data.value
                if data.name == "batch" and data.value is not None:
                    batch_pbar.update(1)
                if data.name == "epoch" and data.value is not None:
                    self.update_dataframe()
                    self.current_row = pd.Series()  # Reset for next epoch
                    master_pbar.update(1)
                    batch_pbar.reset()
            else:
                yield data
        if self.close_bars:
            batch_pbar.close()
            master_pbar.close()


# %% ../../nbs/05_Logging/09a_loggers.core.ipynb 22
class RewardCollector(dp.iter.IterDataPipe, LogCollector):
    title:str='reward'

    def __init__(self,source_datapipe):
        self.source_datapipe = source_datapipe

    def make_record(self,step:StepTypes.types):
        reward = step.reward.detach().numpy()
        if len(reward.shape)!=0:
            reward = reward[0]
        return Record(self.title,reward)

    def __iter__(self):
        yield Record(self.title,None)
        for i,steps in enumerate(self.source_datapipe):
            if not_record(steps):
                if isinstance(steps,dp.DataChunk):
                    for step in steps:
                        yield self.make_record(step)
                else:
                    yield self.make_record(steps)
            yield steps

# %% ../../nbs/05_Logging/09a_loggers.core.ipynb 25
class EpisodeCollector(dp.iter.IterDataPipe):
    title:str='episode'

    def __init__(self,source_datapipe):
        self.source_datapipe = source_datapipe
    
    def make_episode(self,step): 
        try:
            v = step.episode_n.cpu().detach().numpy()
            if len(v.shape)==1: 
                v = v[0]
            return Record(self.title,v)
        except IndexError:
            print(f'Got IndexError getting episode_n which is unexpected: \n{step}')
            raise
    
    def __iter__(self):
        yield Record(self.title,None)
        for i,steps in enumerate(self.source_datapipe):
            if not_record(steps):
                if isinstance(steps,dp.DataChunk):
                    for step in steps:
                        yield self.make_episode(step)
                else:
                    yield self.make_episode(steps)
            yield steps

add_docs(
EpisodeCollector,
"""Collects the `episode_n` field from steps.""",
make_episode="Moves the `episode_n` tensor to numpy.",
)

# %% ../../nbs/05_Logging/09a_loggers.core.ipynb 26
class RollingTerminatedRewardCollector(dp.iter.IterDataPipe):
    title:str='rolling_reward'

    def __init__(self,
         source_datapipe, # The parent datapipe, likely the one to collect metrics from
         rolling_length:int=100
        ):
        self.source_datapipe = source_datapipe
        self.main_buffers = None
        self.rolling_rewards = deque([],maxlen=rolling_length)
        
    def step2terminated(self,step): return bool(step.terminated)

    def make_reward(self,step): 
        try:
            v = step.total_reward.cpu().detach().numpy()
            if len(v.shape)==0: return float(v)
            return v[0]
        except IndexError:
            print(f'Got IndexError getting reward which is unexpected: \n{step}')
            raise

    def __iter__(self):
        yield Record(self.title,None)
        for i,steps in enumerate(self.source_datapipe):
            if not_record(steps):
                if isinstance(steps,dp.DataChunk):
                    for step in steps:
                        if self.step2terminated(step):
                            self.rolling_rewards.append(self.make_reward(step))
                            yield Record(self.title,np.average(self.rolling_rewards))
                elif self.step2terminated(steps):
                    self.rolling_rewards.append(self.make_reward(steps))
                    yield Record(self.title,np.average(self.rolling_rewards))
            yield steps

add_docs(
RollingTerminatedRewardCollector,
"""Collects the `total_reward` field from steps if `terminated` is true and 
logs a rolling average of size `rolling_length`.""",
make_reward="Moves the `total_reward` tensor to numpy.",
step2terminated="Casts the `terminated` field in steps to a bool"
)
