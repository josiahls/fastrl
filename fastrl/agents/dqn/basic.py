# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/10b_agents.dqn.basic.ipynb (unless otherwise specified).

__all__ = ['DQN', 'BasicModelForwarder', 'StepFieldSelector', 'ArgMaxer', 'EpsilonSelector', 'GymTransformBlock',
           'DataBlock', 'LearnerBase', 'is_epocher', 'Epocher', 'is_learner_base', 'find_learner_base', 'LearnerHead',
           'QCalc', 'ModelLearnCalc', 'StepBatcher']

# Cell
# Python native modules
import os
# Third party libs
from fastcore.all import *
import torchdata.datapipes as dp
from torch.utils.data.dataloader_experimental import DataLoader2
from torch.utils.data.datapipes._typing import _DataPipeMeta, _IterDataPipeMeta
# Local modules
import torch
from torch.nn import *
import torch.nn.functional as F
from torch.optim import *
from fastai.torch_basics import *
from fastai.torch_core import *

from ...core import *
from ..core import *
from ...fastai.data.load import *
from ...fastai.data.block import *
from ...memory.experience_replay import *

# Cell
class DQN(Module):
    def __init__(self,state_sz:int,action_sz:int,hidden=512):
        self.layers=Sequential(
            Linear(state_sz,hidden),
            ReLU(),
            Linear(hidden,action_sz),
        )
    def forward(self,x): return self.layers(x)


# Cell
class BasicModelForwarder(dp.iter.IterDataPipe):
    "Takes input from `source_datapipe` and pushes through the agent bases model assuming there is only one model field."
    def __init__(self,source_datapipe):
        self.source_datapipe = source_datapipe
        self.agent_base = find_agent_base(self.source_datapipe)

    def __iter__(self):
        yield from map(self.agent_base.model,self.source_datapipe)

# Cell
class StepFieldSelector(dp.iter.IterDataPipe):
    "Grabs `field` from `source_datapipe` to push to the rest of the pipeline."
    def __init__(self,
         source_datapipe, # datapipe whose next(source_datapipe) -> `StepType`
         field='state' # A field in `StepType` to grab
        ):
        # TODO: support multi-fields
        self.source_datapipe = source_datapipe
        self.field = field

    def __iter__(self):
        for step in self.source_datapipe:
            if not issubclass(step.__class__,StepType):
                raise Exception(f'Expected typing.NamedTuple object got {type(step)}\n{step}')
            yield getattr(step,self.field)

# Cell
class ArgMaxer(dp.iter.IterDataPipe):
    debug=False

    "Given input `Tensor` from `source_datapipe` returns a tensor of same shape with argmax set to 1."
    def __init__(self,source_datapipe,axis=1):
        self.source_datapipe = source_datapipe
        self.axis = axis

    def debug_display(self,step,idx):
        print(f'Step: {step}\n{idx}')

    def __iter__(self) -> torch.LongTensor:
        for step in self.source_datapipe:
            if not issubclass(step.__class__,Tensor):
                raise Exception(f'Expected Tensor to take the argmax, got {type(step)}\n{step}')
            # Might want to support simple tuples also depending on if we are processing multiple fields.
            idx = torch.argmax(step,axis=self.axis).reshape(-1,1)
            step[:] = 0
            if self.debug: self.debug_display(step,idx)
            step.scatter_(1,idx,1)
            yield step.long()


# Cell
class EpsilonSelector(dp.iter.IterDataPipe):
    debug=False
    "Given input `Tensor` from `source_datapipe`."
    def __init__(self,
            source_datapipe, # a datapipe whose next(source_datapipe) -> `Tensor`
            min_epsilon:float=0.2, # The minimum epsilon to drop to
            # The max/starting epsilon if `epsilon` is None and used for calculating epislon decrease speed.
            max_epsilon:float=1,
            # Determines how fast the episilon should drop to `min_epsilon`. This should be the number
            # of steps that the agent was run through.
            max_steps:int=100,
            # The starting epsilon
            epsilon:float=None,
            # Based on the `base_agent.model.training`, by default no decrement or step tracking will
            # occur during validation steps.
            decrement_on_val:bool=False,
            # Based on the `base_agent.model.training`, by default random actions will not be attempted
            select_on_val:bool=False,
            # Also return the mask that, where True, the action should be randomly selected.
            ret_mask:bool=False
        ):
        self.source_datapipe = source_datapipe
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.decrement_on_val = decrement_on_val
        self.select_on_val = select_on_val
        self.ret_mask = ret_mask
        self.agent_base = find_agent_base(self.source_datapipe)
        self.step = 0

    def __iter__(self):
        for action in self.source_datapipe:
            # TODO: Support tuples of actions also
            if not issubclass(action.__class__,Tensor):
                raise Exception(f'Expected Tensor, got {type(action)}\n{action}')
            if action.dtype!=torch.int64:
                raise ValueError(f'Expected Tensor of dtype int64, got: {action.dtype} from {self.source_datapipe}')

            if self.agent_base.model.training or self.decrement_on_val:
                self.step+=1

            self.epsilon=max(self.min_epsilon,self.max_epsilon-self.step/self.max_steps)
            # Add a batch dim if missing
            if len(action.shape)==1: action.unsqueeze_(0)
            mask = None
            if self.agent_base.model.training or self.select_on_val:
                # Given N(action.shape[0]) actions, select the ones we want to randomly assign...
                mask = torch.rand(action.shape[0],)<self.epsilon
                # Get random actions as their indexes
                rand_action_idxs = torch.LongTensor(int(mask.sum().long()),).random_(action.shape[1])
                # If the input action is [[0,1],[1,0]] and...
                # If mask is [True,False] and...
                # if rand_action_idxs is [0]
                # the action[mask] will have [[1,0]] assigned to it resulting in...
                # an action with [[1,0],[1,0]]
                # print(action.shape[1])
                if self.debug: print(f'Mask: {mask}\nRandom Actions: {rand_action_idxs}\nPre-random Actions: {action}')
                action[mask] = F.one_hot(rand_action_idxs,action.shape[1])

            yield ((action,mask) if self.ret_mask else action)

# Cell
def GymTransformBlock(
    agent,
    seed=None,
    nsteps=1,
    nskips=1,
    dl_type = DataLoader2,
    pipe_fn_kwargs=None,
    type_tfms=None,
    **kwargs
):
    pipe_fn_kwargs = ifnone(pipe_fn_kwargs,{})
    type_tfms = ifnone(type_tfms,GymTypeTransform)

    def pipe_fn(source:List[str],bs,n,seed,nsteps,nskips,
                type_tfms=None,item_tfms=None,batch_tfms=None,cbs=None):
        pipe = dp.map.Mapper(source)
        pipe = TypeTransformLoop(pipe,type_tfms)
        pipe = dp.iter.MapToIterConverter(pipe)
        pipe = dp.iter.InMemoryCacheHolder(pipe)
        pipe = pipe.cycle(count=n)
        pipe = GymStepper(pipe,seed=seed)
        if nskips!=1: pipe = NSkipper(pipe,n=nskips)
        if nsteps!=1:
            pipe = NStepper(pipe,n=nsteps)
            pipe = Flattener(pipe)
        pipe = ItemTransformLoop(pipe,item_tfms)
        pipe  = pipe.batch(batch_size=bs)
        pipe = BatchTransformLoop(pipe,batch_tfms)
        pipe = add_cbs_to_pipes(pipe,cbs)
        return pipe

    return TransformBlock(
        pipe_fn = pipe_fn,
        dl_type = dl_type,
        pipe_fn_kwargs = merge(pipe_fn_kwargs,kwargs,dict(nsteps=nsteps,nskips=nskips,seed=seed,type_tfms=type_tfms))
    )

# Cell
class DataBlock(object):
    def __init__(
        self,
        # Each transform block will have its own dataloader.
        blocks:List[TransformBlock]=None,
    ):
        store_attr(but='blocks')
        self.blocks = L(blocks)

    def datapipes(
        self,
        source:Any,
        bs=1,
        n=1,
        return_blocks:bool=False
    ) -> Generator[Union[Tuple[_DataPipeMeta,TransformBlock],_DataPipeMeta],None,None]:
        for b in self.blocks:
            pipe = b.pipe_fn(source,bs=bs,n=n,**b.pipe_fn_kwargs)
            yield (pipe,b) if return_blocks else pipe

    def dataloaders(
        self,
        source:Any,
        bs=1,
        n=1,
        n_workers=0,
        **kwargs
    ) -> Generator[DataLoader2,None,None]:
        for pipe,block in self.datapipes(source,bs=bs,n=n,return_blocks=True,**kwargs):
            yield block.dl_type(pipe,**merge(kwargs,block.dls_kwargs))

# Cell
class LearnerBase(dp.iter.IterDataPipe):
    def __init__(self,
            model:Module, # The base NN that we getting raw action values out of.
            dls:List[DataLoader2], # The dataloaders to read data from for training
            loss_func=None, # The loss function to use
            opt=None, # The optimizer to use
            # LearnerBase will yield each dl individually by default. If `zipwise=True`
            # next() will be called on `dls` and will `yield next(dl1),next(dl2),next(dl1)...`
            zipwise:bool=False
    ):
        self.loss_func = loss_func
        self.opt = opt
        self.model = model
        self.iterable = dls
        self.zipwise = zipwise
        self.learner_base = self

    def __iter__(self):
        dls = [iter(dl) for dl in self.iterable]
        exhausted = []
        if self.zipwise:
            yield from [next(dl) for i,dl in enumerate(dls) if i not in exhausted]
        else:
            while not exhausted:
                for i,dl in enumerate(dls):
                    if i not in exhausted:
                        try:
                            yield next(dl)
                        except StopIteration:
                            exhausted.append(i)

# Cell
def is_epocher(pipe): return isinstance(pipe,Epocher)

class Epocher(dp.iter.IterDataPipe):
    def __init__(self,source_datapipe):
        self.source_datapipe = source_datapipe
        self.epoch = 0

    def __iter__(self):
        for i in range(self.epoch):
            yield from self.source_datapipe


# Cell
def is_learner_base(pipe): return isinstance(pipe,LearnerBase)

def find_learner_base(pipe):
    "Basically just find_pipes+is_learner_base with exception handling"
    learner_base = find_pipes(pipe,is_learner_base)
    if not learner_base:
        raise Exception('`LearnerBase` must be at the start of the pipeline, but it seems to be missing.')
    return learner_base[0]

class LearnerHead(dp.iter.IterDataPipe):
    def __init__(self,source_datapipe):
        self.source_datapipe = source_datapipe
        self.learner_base = find_learner_base(self.source_datapipe)

    def __iter__(self): yield from self.source_datapipe

    def fit(self,epoch):
        epocher = find_pipes(self,is_epocher)[0]
        epocher.epoch = epoch

        for iteration in self:
            pass

# Cell
class QCalc(dp.iter.IterDataPipe):
    def __init__(self,source_datapipe,discount=0.99,nsteps=1):
        self.source_datapipe = source_datapipe
        self.discount = discount
        self.nsteps = nsteps
        self.learner = find_learner_base(self.source_datapipe)

    def __iter__(self):
        for batch in self.source_datapipe:
            self.learner.done_mask = batch.terminated.reshape(-1,)
            self.learner.next_q = self.learner.model(batch.next_state)
            self.learner.next_q = self.learner.next_q.max(dim=1).values.reshape(-1,1)
            self.learner.next_q[self.learner.done_mask] = 0 #xb[done_mask]['reward']
            self.learner.targets = batch.reward+self.learner.next_q*(self.discount**self.nsteps)
            self.learner.pred = self.learner.model(batch.state)


            t_q=self.learner.pred.clone()
            t_q.scatter_(1,batch.action.long(),self.learner.targets)

            self.learner.loss_grad = self.learner.loss_func(self.learner.pred, self.learner.targets)
            yield batch

class ModelLearnCalc(dp.iter.IterDataPipe):
    def __init__(self,source_datapipe):
        self.source_datapipe = source_datapipe
        self.learner = find_learner_base(self.source_datapipe)

    def __iter__(self):
        for batch in self.source_datapipe:
            self.learner.loss_grad.backward()
            self.learner.opt.step()
            self.learner.opt.zero_grad()
            self.learner.loss = self.learner.loss_grad.clone()
            yield self.learner.loss

# Cell
class StepBatcher(dp.iter.IterDataPipe):
    def __init__(self,source_datapipe):
        "Converts multiple `StepType` into a single `StepType` with the fields concated."
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for batch in self.source_datapipe:
            cls = batch[0].__class__
            yield cls(
                **{
                    fld:torch.vstack(tuple(getattr(step,fld) for step in batch)) for fld in cls._fields
                }
            )