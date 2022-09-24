# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/02_DataLoading/02f_data.dataloader2.ipynb.

# %% auto 0
__all__ = ['DataPipeToQueuesLoop', 'SpawnProcessForDataPipeline', 'GetInputItemResponse', 'GetInputItemRequest',
           'InputItemIterDataPipeQueueProtocolClient', 'InputItemIterDataPipeQueueProtocolServer', 'AgentLoggerMerger',
           'PrototypeMultiProcessingReadingService', 'InputInjester', 'DataPipeBehindQueues', 'item_input_pipe_type']

# %% ../../nbs/02_DataLoading/02f_data.dataloader2.ipynb 4
# Python native modules
import os,typing
# Third party libs
from fastcore.all import *
from torch.multiprocessing import Pool,Process,set_start_method,Manager,get_start_method,Queue
import torchdata.datapipes as dp
from fastprogress.fastprogress import *
from torchdata.dataloader2.graph import find_dps,traverse

from torchdata.dataloader2.dataloader2 import *
from torchdata.dataloader2.reading_service import *
from torchdata.dataloader2.reading_service import _IterateQueueDataPipes
from torchdata.dataloader2.communication.protocol import *
from torchdata.dataloader2.communication.messages import *
from torchdata.dataloader2.communication.iter import EnsureNonBlockingDataPipe,NotAvailable,InvalidStateResetRequired
from torch.utils.data import IterDataPipe, MapDataPipe
# Local modules
from ..core import *
from ..agents.core import AgentBase
from ..pipes.core import *
from ..loggers.core import LoggerBase

# %% ../../nbs/02_DataLoading/02f_data.dataloader2.ipynb 5
def DataPipeToQueuesLoop(source_datapipe, req_queue, res_queue, call_locally_fn=None, protocol_type=None, pipe_type=None):
    if call_locally_fn is not None:
        result = call_locally_fn(source_datapipe)
        if result is not None: 
            source_datapipe = result
        
    if isinstance(source_datapipe, IterDataPipe):
        if pipe_type is None:
            pipe_type = communication.iter
        if protocol_type is None:
            protocol_type = communication.protocol.IterDataPipeQueueProtocolServer
    elif isinstance(source_datapipe, MapDataPipe):
        if pipe_type is None:
            pipe_type = communication.map  # type: ignore[misc]
        if protocol_type is None:
            protocol_type = communication.protocol.MapDataPipeQueueProtocolServer  # type: ignore[assignment]
    else:
        raise Exception("Only supports IterDataPipe or MapDataPipe, got", source_datapipe)

    torch.set_num_threads(1)
    for _ in pipe_type.DataPipeBehindQueues(
        source_datapipe, protocol_type(req_queue, res_queue), blocking_request_get=True
    ):
        pass


def SpawnProcessForDataPipeline(multiprocessing_ctx, datapipe, call_locally_fn=None, protocol_type=None, pipe_type=None):
    req_queue = multiprocessing_ctx.Queue()
    res_queue = multiprocessing_ctx.Queue()
    process = multiprocessing_ctx.Process(
        target=DataPipeToQueuesLoop, args=(datapipe, req_queue, res_queue, call_locally_fn, protocol_type, pipe_type)
    )
    return process, req_queue, res_queue


# %% ../../nbs/02_DataLoading/02f_data.dataloader2.ipynb 6
# The opposite of the GetItemRequest/GetItemResponse api. We want to input items into the dataloader's processes
class GetInputItemResponse(Response):
    __slots__ = "value"

    def __init__(self, value):
        self.value = value
    
class GetInputItemRequest(Response):
    __slots__ = ("key", "value")

    def __init__(self, key, value):
        self.key = key
        self.value = value

# %% ../../nbs/02_DataLoading/02f_data.dataloader2.ipynb 7
class InputItemIterDataPipeQueueProtocolClient(communication.protocol.IterDataPipeQueueProtocolClient):
    def request_input_item(self,key, value):
        if not self.can_take_request():
            raise Exception("Can not reset while we are still waiting response for previous request")
        request = GetInputItemRequest(key, value)
        self.request_queue.put(request)
        self.request_sent(request)

    def get_response_input_item(self, block=False):
        try:
            response = self.response_queue.get(block=block)
        except Exception:  # TODO(627): Catch only timeout exceptions
            raise EmptyQueue("queue is empty")
        self.request_served(response)

        if not isinstance(response, GetInputItemResponse):
            raise Exception("Invalid response received")
            
class InputItemIterDataPipeQueueProtocolServer(IterDataPipeQueueProtocolServer):
    def response_input_item(self, key):
        if not self.have_pending_request():
            raise Exception("Attempting to reply with pending request")
        if not isinstance(self._req_received, GetInputItemRequest):
            raise Exception("Replaying with reset status to other type of message")
        self.response_queue.put(GetInputItemResponse(
            # We need this to make it all the way to the main process datapipe
            # Without this, we get a string... which isnt great to build triggers / filters on.
            GetInputItemResponse(key) 
        ))
        self._req_received = None


# %% ../../nbs/02_DataLoading/02f_data.dataloader2.ipynb 9
class AgentLoggerMerger(dp.iter.IterDataPipe):
    def __init__(self,
            source_datapipe
        ):
        self.source_datapipe = source_datapipe
        try:
            self.logger_bases = [o for o in find_dp(traverse(self),AgentBase).logger_bases]
        except LookupError:
            self.logger_bases = []
        try:
            self.logger_bases.extend([o for o in find_dp(traverse(self),LoggerBase)])
        except LookupError:pass
        
    def __iter__(self): 
        for value in self.source_datapipe:
            for logger_base in self.logger_bases:
                # print('iterating through logger bases',self.logger_bases)
                for record in logger_base.dequeue():
                    # print('Yielding record ',record)
                    yield record
            yield value
add_docs(
    AgentLoggerMerger,
    """Inserts values from `input_jests` into the current pipeline."""
)

# %% ../../nbs/02_DataLoading/02f_data.dataloader2.ipynb 10
class PrototypeMultiProcessingReadingService(ReadingServiceInterface):
    num_workers: int
    processes: List
    datapipes: List

    def __init__(
        self,
        num_workers: int = 0,
        multiprocessing_context=None,
        protocol_client_type = None,
        protocol_server_type = None,
        pipe_type = None,
        eventloop = None
    ) -> None:
        self.num_workers = num_workers
        # TODO(613): Should be one of 'fork', 'spawn'
        self.multiprocessing_context = multiprocessing_context
        self.processes = []
        self.datapipes = []
        if protocol_client_type is None:
            self.protocol_client_type = communication.protocol.IterDataPipeQueueProtocolClient
        else:
            self.protocol_client_type = protocol_client_type
        self.protocol_server_type = protocol_server_type
        # pipe_type (should) use self.protocol_server_type in a forever loop
        self.pipe_type = pipe_type 

        if eventloop is None:
            self.eventloop = communication.eventloop.SpawnProcessForDataPipeline
        else:
            self.eventloop = eventloop

    @staticmethod
    def init_datapipe_process(num_workers, worker_id, datapipe):
        # TODO(614): Add distributed support
        # TODO(615): Add shuffle determinism support
        torch.utils.data.graph_settings.apply_sharding(datapipe, num_workers, worker_id)
        return AgentLoggerMerger(datapipe)

    def initialize(self, datapipe: DataPipe) -> DataPipe:
        if self.num_workers == 0:
            # TODO(616): Warn and recommend usage of InProcessReadingService
            return datapipe
        for worker_id in range(self.num_workers):
            # TODO(617): Separate into function, because we also need to apply distributed seed
            #            and call it inside process
            call_inside_process = functools.partial(self.init_datapipe_process, self.num_workers, worker_id)
            ctx = mp.get_context(self.multiprocessing_context)
            (process, req_queue, res_queue) = self.eventloop(
                ctx, datapipe, call_inside_process, self.protocol_server_type,self.pipe_type
            )
            process.start()
            self.processes.append((process, req_queue, res_queue))  # These queues are independent
            local_datapipe = communication.iter.QueueWrapper(
                self.protocol_client_type(req_queue, res_queue)
            )
            self.datapipes.append(local_datapipe)

        return IterableWrapper(_IterateQueueDataPipes(self.datapipes), deepcopy=False)  # type: ignore[return-value]


# %% ../../nbs/02_DataLoading/02f_data.dataloader2.ipynb 11
class InputInjester(dp.iter.IterDataPipe):
    def __init__(self,
            source_datapipe
        ):
        self.source_datapipe = source_datapipe
        self.input_injests = []
        
    def __iter__(self): 
        for value in self.source_datapipe:
            if self.input_injests:
                for input_value in self.input_injests:
                    yield input_value
            yield value
add_docs(
    InputInjester,
    """Inserts values from `input_jests` into the current pipeline."""
)

# %% ../../nbs/02_DataLoading/02f_data.dataloader2.ipynb 12
def DataPipeBehindQueues(source_datapipe, protocol, full_stop=False, blocking_request_get=False):
    """
    Indefinitely iterates over req_queue and passing values from source_datapipe to res_queue
    If raise_stop is true, raises exception when StopIteration received from the source_datapipe
    """
    if not isinstance(protocol, communication.protocol.IterDataPipeQueueProtocolServer):
        raise Exception("Expecting IterDataPipeQueueProtocolServer, got", protocol)
    source_datapipe = EnsureNonBlockingDataPipe(source_datapipe)
    input_injester_pipes = find_dps(traverse(source_datapipe),InputInjester)
    forever = True
    while forever:
        try:
            # Non-blocking call is Extremely slow here for python.mp, need to figure out a good workaround
            request = protocol.get_new_request(block=blocking_request_get)
        except communication.protocol.EmptyQueue:
            yield True
            continue
        # Requires there to be InputInjester pipelines in `source_datapipe`
        if isinstance(request, GetInputItemRequest):
            for input_dp in input_injester_pipes: input_dp.input_injests.append(request)
            protocol.response_input_item(request.key)
            
        elif isinstance(request, communication.messages.ResetIteratorRequest):
            source_datapipe.reset_iterator()
            protocol.response_reset_iterator()

        elif isinstance(request, communication.messages.TerminateRequest):
            forever = False
            protocol.response_terminate()

        elif isinstance(request, communication.messages.GetNextRequest):
            while forever:
                try:
                    value = source_datapipe.nonblocking_next()
                except NotAvailable:
                    yield True
                    continue
                except StopIteration:
                    protocol.response_stop_iteration()
                    if full_stop:
                        forever = False
                    else:
                        yield True
                    break
                except InvalidStateResetRequired:
                    protocol.response_invalid_state()
                    if full_stop:
                        forever = False
                    else:
                        yield True
                    break
                protocol.response_next(value)
                
                yield True  # Returns control
                break
        else:
            raise Exception("Unrecognized type of request received", request)

class item_input_pipe_type():
    DataPipeBehindQueues = DataPipeBehindQueues