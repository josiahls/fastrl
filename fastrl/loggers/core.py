# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/09a_loggers.core.ipynb.

# %% auto 0
__all__ = ['LoggerBase', 'LogCollector', 'ProgressBarLogger', 'RewardCollector', 'EpocherCollector', 'BatchCollector', 'TestSync',
           'ActionPublish']

# %% ../nbs/09a_loggers.core.ipynb 3
# Python native modules
import os,typing
# Third party libs
from fastcore.all import *
from torch.multiprocessing import Pool,Process,set_start_method,Manager,get_start_method,Queue
import torchdata.datapipes as dp
from fastprogress.fastprogress import *
from torchdata.dataloader2.graph import find_dps,traverse
# Local modules
from ..core import *
from ..pipes.core import *

# %% ../nbs/09a_loggers.core.ipynb 6
class LoggerBase(dp.iter.IterDataPipe):
    
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
        self.buffer = []
    
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

# %% ../nbs/09a_loggers.core.ipynb 9
class LogCollector(dp.iter.IterDataPipe):
    def __init__(self,
         source_datapipe, # The parent datapipe, likely the one to collect metrics from
         logger_bases:List[LoggerBase] # `LoggerBase`s that we want to send metrics to
        ):
        self.source_datapipe = source_datapipe
        self.main_buffers = [o.buffer for o in logger_bases]
        
    def __iter__(self): raise NotImplementedError

# %% ../nbs/09a_loggers.core.ipynb 11
class ProgressBarLogger(LoggerBase):
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
        epocher = find_dp(traverse(self),self.epoch_on_pipe)
        batcher = find_dp(traverse(self),self.batch_on_pipe)
        mbar = master_bar(range(epocher.epochs)) 
        pbar = progress_bar(range(batcher.batches),parent=mbar,leave=False)

        mbar.update(0)
        for i,record in enumerate(self.source_datapipe):
            if self.filter_record(record):
                self.buffer.append(record)
                # We only want to start setting up logging when the data loader starts producing 
                # real data.
                continue
                
            if i==0:
                self.attached_collectors = {o.name:o.value for o in self.dequeue()}
                mbar.write(self.attached_collectors, table=True)
                self.collector_keys = list(self.attached_collectors)
                    
            attached_collectors = {o.name:o.value for o in self.dequeue()}

            if attached_collectors:
                self.attached_collectors = merge(self.attached_collectors,attached_collectors)
            
            if 'batch' in attached_collectors:
                pbar.update(attached_collectors['batch'])
                
            if 'epoch' in attached_collectors:
                mbar.update(attached_collectors['epoch'])
                collector_values = {k:self.attached_collectors.get(k,None) for k in self.collector_keys}
                mbar.write([f'{l:.6f}' if isinstance(l, float) else str(l) for l in collector_values.values()], table=True)
                
            if self.filter_record(record): continue
            yield record

        attached_collectors = {o.name:o.value for o in self.dequeue()}
        if attached_collectors: self.attached_collectors = merge(self.attached_collectors,attached_collectors)

        collector_values = {k:self.attached_collectors.get(k,None) for k in self.collector_keys}
        mbar.write([f'{l:.6f}' if isinstance(l, float) else str(l) for l in collector_values.values()], table=True)

        pbar.on_iter_end()
        mbar.on_iter_end()
            

# %% ../nbs/09a_loggers.core.ipynb 12
class RewardCollector(LogCollector):
    def __iter__(self):
        for q in self.main_buffers: q.append(Record('reward',None))
        for steps in self.source_datapipe:
            if isinstance(steps,dp.DataChunk):
                for step in steps:
                    for q in self.main_buffers: q.append(Record('reward',step.reward.detach().numpy()))
            else:
                for q in self.main_buffers: q.append(Record('reward',steps.reward.detach().numpy()))
            yield steps

# %% ../nbs/09a_loggers.core.ipynb 13
class EpocherCollector(dp.iter.IterDataPipe):
    def __init__(self,
            source_datapipe,
            epochs:int=0,
            logger_bases:List[LoggerBase]=None # `LoggerBase`s that we want to send metrics to
        ):
        self.source_datapipe = source_datapipe
        self.main_buffers = [o.buffer for o in logger_bases] if logger_bases is not None else None
        self.iteration_started = False
        self.epochs = epochs
        self.epoch = 0

    def __iter__(self): 
        if self.main_buffers is not None and not self.iteration_started:
            for q in self.main_buffers: q.append(Record('epoch',None))
            self.iteration_started = True
            
        for i in range(self.epochs): 
            self.epoch = i
            if self.main_buffers is not None:
                for q in self.main_buffers: q.append(Record('epoch',self.epoch))
            yield from self.source_datapipe
            
add_docs(
    EpocherCollector,
    """Tracks the number of epochs that the pipeline is currently on."""
)

# %% ../nbs/09a_loggers.core.ipynb 14
class BatchCollector(dp.iter.IterDataPipe):
    def __init__(self,
            source_datapipe,
            logger_bases:List[LoggerBase], # `LoggerBase`s that we want to send metrics to
            batches:Optional[int]=None,
            # If `batches` is None, `BatchCollector` with try to find: `batch_on_pipe` instance
            # and try to grab a `batches` field from there.
            batch_on_pipe:dp.iter.IterDataPipe=None 
        ):
        self.source_datapipe = source_datapipe
        self.main_buffers = [o.buffer for o in logger_bases] if logger_bases is not None else None
        self.iteration_started = False
        self.batches = (
            batches if batches is not None else self.batch_on_pipe_get_batches(batch_on_pipe)
        )
        self.batch = 0
        
    def batch_on_pipe_get_batches(self,batch_on_pipe):
        pipe = find_dp(traverse(self.source_datapipe),batch_on_pipe)
        if hasattr(pipe,'batches'):
            return pipe.batches
        elif hasattr(pipe,'limit'):
            return pipe.limit
        else:
            raise RuntimeError(f'Pipe {pipe} isnt recognized as a batch tracker.')

    def __iter__(self): 
        if self.main_buffers is not None and not self.iteration_started:
            for q in self.main_buffers: q.append(Record('batch',None))
            self.iteration_started = True
            
        self.batch = 0
        for batch,record in enumerate(self.source_datapipe): 
            yield record
            self.batch = batch
            if self.main_buffers is not None:
                for q in self.main_buffers: q.append(Record('batch',batch))
                
add_docs(
    BatchCollector,
    """Tracks the number of batches that the pipeline is currently on.""",
    batch_on_pipe_get_batches="Gets the number of batches from `batch_on_pipe`"
)

# %% ../nbs/09a_loggers.core.ipynb 15
class TestSync(dp.iter.IterDataPipe):
    def __init__(self,
            source_datapipe
        ):
        self.source_datapipe = source_datapipe
        self.actions_augments = []
        
    def __iter__(self): 
        for step in self.source_datapipe:
            # print('Got step: ',step)
            if isinstance(step,GetInputItemRequest):
                # print('augmenting!!!!!')
                self.actions_augments.append(step.value)
                continue
            elif self.actions_augments:
                step = step.__class__(**{fld:getattr(step,fld)+self.actions_augments.pop(0) 
                                         if fld=='action' else 
                                         getattr(step,fld) for fld in step._fields})
            yield step
add_docs(
    TestSync,
    """Tests getting values from data loader requests."""
)

# %% ../nbs/09a_loggers.core.ipynb 20
from ..core import StepType

class ActionPublish(dp.iter.IterDataPipe):
    def __init__(self,
            source_datapipe, # Pretend this is in the middle of a learner training segment
            dls
        ):
        self.source_datapipe = source_datapipe
        self.dls = dls
        self.protocol_clients = []
        self._expect_response = []
        self.initialized = False
        
    def __iter__(self): 
        for step in self.source_datapipe:
            if not self.initialized:
                for dl in self.dls:
                    # dataloader.IterableWrapperIterDataPipe._IterateQueueDataPipes,[QueueWrappers]
                    for q_wrapper in dl.datapipe.iterable.datapipes:
                        self.protocol_clients.append(q_wrapper.protocol)
                        self._expect_response.append(False)
                self.initialized = True
            
            if isinstance(step,StepType):
                for i,client in enumerate(self.protocol_clients):
                    if self._expect_response[i]: 
                        client.get_response_input_item()
                    else:
                        client.request_input_item(
                            'action_augmentation',value=100
                        )

            yield step
        self.protocol_clients = []
        self._expect_response = []
add_docs(
    ActionPublish,
    """Publishes an action augmentation to the dataloader."""
)
