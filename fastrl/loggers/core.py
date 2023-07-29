# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/05_Logging/09a_loggers.core.ipynb.

# %% auto 0
__all__ = ['LoggerBase', 'LoggerBasePassThrough', 'LogCollector', 'ProgressBarLogger', 'RewardCollector', 'EpocherCollector',
           'BatchCollector', 'EpisodeCollector', 'RollingTerminatedRewardCollector', 'CacheLoggerBase']

# %% ../../nbs/05_Logging/09a_loggers.core.ipynb 2
# Python native modules
# import os,typing
from typing import Optional,List
from collections import deque
# Third party libs
from fastcore.all import add_docs
# from torch.multiprocessing import Pool,Process,set_start_method,Manager,get_start_method,Queue
import torchdata.datapipes as dp
# from fastprogress.fastprogress import *
from torchdata.dataloader2.graph import find_dps,traverse_dps
# from torch.utils.data.datapipes._hook_iterator import _SnapshotState
import numpy as np
# Local modules
# from fastrl.core import *
from ..pipes.core import find_dp,find_dps
from ..core import StepType,Record
# from fastrl.torch_core import *

# %% ../../nbs/05_Logging/09a_loggers.core.ipynb 4
class LoggerBase(dp.iter.IterDataPipe):
    debug:bool=False
    
    def __init__(self,source_datapipe=None,do_filter=True):
        self.source_datapipe = source_datapipe
        self.buffer = []
        self.do_filter = do_filter
        
    def connect_source_datapipe(self,pipe):
        self.source_datapipe = pipe
        return self
    
    def filter_record(self,record):
        return type(record)==Record and self.do_filter
    
    def dequeue(self): 
        while self.buffer: yield self.buffer.pop(0)
    
    def reset(self):
        # We can chain multiple `LoggerBase`s together, but if we do this, we dont want the 
        # first one in the chain filtering out the Records before the others!
        if issubclass(type(self.source_datapipe),LoggerBase):
            self.source_datapipe.do_filter = False
        # Note: trying to decide if this is really needed.
        # if self.debug:
        #     print(self,' resetting buffer.')
        # if self._snapshot_state!=_SnapshotState.Restored:
        #     self.buffer = []
    
    def __iter__(self):
        raise NotImplementedError
        
add_docs(
    LoggerBase,
    """The `LoggerBase` class outlines simply the `buffer`. 
    It works in combo with `LogCollector` datapipe which will add to the `buffer`.
    
    `LoggerBase` also filters out the log records to as to not disrupt the training pipeline""",
    filter_record="Returns True of `record` is actually a record and that `self` actually is set to filter.",
    connect_source_datapipe="""`LoggerBase` does not need to be part of a `DataPipeGraph` 
    when its initialized, so this method allows for inserting into a `DataPipeGraph` later on.""",
    reset="""Checks if `self.source_datapipe` is also a logger base, and if so will tell `self.source_datapipe`
    not to filter out the log records.""",
    dequeue="Empties the `self.buffer` yielding each of its contents."
)        

# %% ../../nbs/05_Logging/09a_loggers.core.ipynb 5
class LoggerBasePassThrough(dp.iter.IterDataPipe):
    def __init__(self,source_datapipe,logger_bases=None):
        self.source_datapipe = source_datapipe
        self.logger_bases = logger_bases

    def __iter__(self):
        yield from self.source_datapipe

add_docs(
LoggerBasePassThrough,
"""Allows for collectors to find `LoggerBase`s early in the pipeline without
worrying about accidently iterating the logger bases at the incorrect time/frequency.

This is mainly used for collectors to call `find_dps` easily on the pipeline.
"""
)    

# %% ../../nbs/05_Logging/09a_loggers.core.ipynb 8
class LogCollector(dp.iter.IterDataPipe):
    debug:bool=False
    header:Optional[str]=None

    def __init__(self,
         source_datapipe, # The parent datapipe, likely the one to collect metrics from
        ):
        self.source_datapipe = source_datapipe
        self.main_buffers = None
        
    def __iter__(self): raise NotImplementedError

    def push_header(
            self,
            key:str
        ):
        # self.reset()
        for q in self.main_buffers: q.append(Record(key,None))

    def reset(self):
        if self.main_buffers is None:
            if self.debug: print(f'Resetting {self}')
            logger_bases = find_dps(traverse_dps(self),LoggerBase,include_subclasses=True)
            self.main_buffers = [o.buffer for o in logger_bases]
            self.push_header(self.header)

add_docs(
LogCollector,
"""`LogCollector` specifically manages finding and attaching itself to
`LoggerBase`s found earlier in the pipeline.""",
reset="Grabs buffers from all logger bases in the pipeline.",
push_header="""Should be called after the pipeline is initialized. Sends header
`Record`s to the `LoggerBase`s so they know what logs are going to be sent to them."""
)  


# %% ../../nbs/05_Logging/09a_loggers.core.ipynb 10
class ProgressBarLogger(LoggerBase):
    debug:bool=False

    def __init__(self,
                 # This does not need to be immediately set since we need the `LogCollectors` to 
                 # first be able to reference its queues.
                 source_datapipe=None, 
                 # For automatic pipe attaching, we can designate which pipe this should be
                 # referneced for information on which epoch we are on
                 epoch_on_pipe:dp.iter.IterDataPipe=None,
                 # For automatic pipe attaching, we can designate which pipe this should be
                 # referneced for information on which batch we are on
                 batch_on_pipe:dp.iter.IterDataPipe=None
                ):
        super().__init__(source_datapipe=source_datapipe)
        self.epoch_on_pipe = epoch_on_pipe
        self.batch_on_pipe = batch_on_pipe
        
        self.collector_keys = None
        self.attached_collectors = None
    
    def __iter__(self):
        epocher = find_dp(traverse_dps(self),self.epoch_on_pipe)
        batcher = find_dp(traverse_dps(self),self.batch_on_pipe)
        mbar = master_bar(range(epocher.epochs)) 
        pbar = progress_bar(range(batcher.batches),parent=mbar,leave=False)

        mbar.update(0)
        i = 0
        for record in self.source_datapipe:
            if self.filter_record(record):
                self.buffer.append(record)
                # We only want to start setting up logging when the data loader starts producing 
                # real data.
                continue
   
            if i==0:
                self.attached_collectors = {o.name:o.value for o in self.dequeue()}
                if self.debug: print('Got initial values: ',self.attached_collectors)
                mbar.write(self.attached_collectors, table=True)
                self.collector_keys = list(self.attached_collectors)
                pbar.update(0)
                    
            attached_collectors = {o.name:o.value for o in self.dequeue()}
            if self.debug: print('Got running values: ',self.attached_collectors)

            if attached_collectors:
                self.attached_collectors = merge(self.attached_collectors,attached_collectors)
            
            if 'batch' in attached_collectors: 
                pbar.update(attached_collectors['batch'])
                
            if 'epoch' in attached_collectors:
                mbar.update(attached_collectors['epoch'])
                collector_values = {k:self.attached_collectors.get(k,None) for k in self.collector_keys}
                mbar.write([f'{l:.6f}' if isinstance(l, float) else str(l) for l in collector_values.values()], table=True)
                
            i+=1  
            yield record

        attached_collectors = {o.name:o.value for o in self.dequeue()}
        if attached_collectors: self.attached_collectors = merge(self.attached_collectors,attached_collectors)

        collector_values = {k:self.attached_collectors.get(k,None) for k in self.collector_keys}
        mbar.write([f'{l:.6f}' if isinstance(l, float) else str(l) for l in collector_values.values()], table=True)

        pbar.on_iter_end()
        mbar.on_iter_end()
            

# %% ../../nbs/05_Logging/09a_loggers.core.ipynb 11
class RewardCollector(LogCollector):
    header:str='reward'

    def __iter__(self):
        for i,steps in enumerate(self.source_datapipe):
            # if i==0: self.push_header('reward')
            if isinstance(steps,dp.DataChunk):
                for step in steps:
                    for q in self.main_buffers: q.append(Record('reward',step.reward.detach().numpy()))
            else:
                for q in self.main_buffers: q.append(Record('reward',steps.reward.detach().numpy()))
            yield steps

# %% ../../nbs/05_Logging/09a_loggers.core.ipynb 12
class EpocherCollector(LogCollector):
    debug:bool=False
    header:str='epoch'

    def __init__(self,
            source_datapipe,
            epochs:int=0,
            logger_bases:List[LoggerBase]=None # `LoggerBase`s that we want to send metrics to
        ):
        self.source_datapipe = source_datapipe
        self.main_buffers = None
        self.iteration_started = False
        self.epochs = epochs
        self.epoch = 0

    def __iter__(self): 
        # if self.main_buffers is not None and not self.iteration_started:
        #     self.push_header('epoch')
        #     self.iteration_started = True
            
        for i in range(self.epochs): 
            self.epoch = i
            if self.main_buffers is not None:
                for q in self.main_buffers: q.append(Record('epoch',self.epoch))
            # print('yielding on epoch',self.epoch)
            yield from self.source_datapipe
            
add_docs(
EpocherCollector,
"""Tracks the number of epochs that the pipeline is currently on.""",
reset="Grabs buffers from all logger bases in the pipeline."
)

# %% ../../nbs/05_Logging/09a_loggers.core.ipynb 13
class BatchCollector(LogCollector):
    header:str='batch'

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
        # self.header = 'batch'
        
    def batch_on_pipe_get_batches(self,batch_on_pipe):
        pipe = find_dp(traverse_dps(self.source_datapipe),batch_on_pipe)
        if hasattr(pipe,'batches'):
            return pipe.batches
        elif hasattr(pipe,'limit'):
            return pipe.limit
        else:
            raise RuntimeError(f'Pipe {pipe} isnt recognized as a batch tracker.')

    def __iter__(self): 
        # if self.main_buffers is not None and not self.iteration_started:
        #     self.push_header('batch')
        #     if self.debug: print('pushing batch',self.main_buffers)
        #     self.iteration_started = True
            
        self.batch = 0
        for batch,record in enumerate(self.source_datapipe): 
            yield record
            if type(record)!=Record:
                self.batch += 1
                if self.main_buffers is not None:
                    # print('posting batch values: ',Record('batch',self.batch))
                    for q in self.main_buffers: q.append(Record('batch',self.batch))
            if self.batch>self.batches: 
                break

add_docs(
BatchCollector,
"""Tracks the number of batches that the pipeline is currently on.""",
batch_on_pipe_get_batches="Gets the number of batches from `batch_on_pipe`",
reset="Grabs buffers from all logger bases in the pipeline."
)

# %% ../../nbs/05_Logging/09a_loggers.core.ipynb 14
class EpisodeCollector(LogCollector):
    header:str='episode'
    
    def episode_detach(self,step): 
        try:
            v = step.episode_n.cpu().detach().numpy()
            if len(v.shape)==0: return int(v)
            return v[0]
        except IndexError:
            print(f'Got IndexError getting episode_n which is unexpected: \n{step}')
            raise
    
    def __iter__(self):
        for i,steps in enumerate(self.source_datapipe):
            # if i==0: self.push_header('episode')
            if isinstance(steps,dp.DataChunk):
                for step in steps:
                    for q in self.main_buffers: q.append(Record('episode',self.episode_detach(step)))
            else:
                for q in self.main_buffers: q.append(Record('episode',self.episode_detach(steps)))
            yield steps

add_docs(
EpisodeCollector,
"""Collects the `episode_n` field from steps.""",
episode_detach="Moves the `episode_n` tensor to numpy.",
)

# %% ../../nbs/05_Logging/09a_loggers.core.ipynb 15
class RollingTerminatedRewardCollector(LogCollector):
    debug:bool=False
    header:str='rolling_reward'

    def __init__(self,
         source_datapipe, # The parent datapipe, likely the one to collect metrics from
         rolling_length:int=100
        ):
        self.source_datapipe = source_datapipe
        self.main_buffers = None
        self.rolling_rewards = deque([],maxlen=rolling_length)
        
    def step2terminated(self,step): return bool(step.terminated)

    def reward_detach(self,step): 
        try:
            v = step.total_reward.cpu().detach().numpy()
            if len(v.shape)==0: return float(v)
            return v[0]
        except IndexError:
            print(f'Got IndexError getting reward which is unexpected: \n{step}')
            raise

    def __iter__(self):
        for i,steps in enumerate(self.source_datapipe):
            if self.debug: print(f'RollingTerminatedRewardCollector: ',steps)
            if isinstance(steps,dp.DataChunk):
                for step in steps:
                    if self.step2terminated(step):
                        self.rolling_rewards.append(self.reward_detach(step))
                        for q in self.main_buffers: q.append(Record('rolling_reward',np.average(self.rolling_rewards)))
            elif self.step2terminated(steps):
                self.rolling_rewards.append(self.reward_detach(steps))
                for q in self.main_buffers: q.append(Record('rolling_reward',np.average(self.rolling_rewards)))
            yield steps

add_docs(
RollingTerminatedRewardCollector,
"""Collects the `total_reward` field from steps if `terminated` is true and 
logs a rolling average of size `rolling_length`.""",
reward_detach="Moves the `total_reward` tensor to numpy.",
step2terminated="Casts the `terminated` field in steps to a bool"
)

# %% ../../nbs/05_Logging/09a_loggers.core.ipynb 22
class CacheLoggerBase(LoggerBase):
    "Short lived logger base meant to dump logs"
    def reset(self):
        # This logger will be exhausted frequently if used in an agent.
        # We need to get the buffer alive so we dont lose reference
        pass
    
    def __iter__(self):
        print('Iterating through buffer of len: ',len(self.buffer))
        yield from self.buffer
        self.buffer.clear()
