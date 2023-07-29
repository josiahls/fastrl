# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/07_Agents/12a_agents.core.ipynb.

# %% auto 0
__all__ = ['AgentBase', 'AgentHead', 'SimpleModelRunner', 'StepFieldSelector', 'StepModelFeeder', 'NumpyConverter']

# %% ../../nbs/07_Agents/12a_agents.core.ipynb 2
# Python native modules
import os
from typing import List
# Third party libs
from fastcore.all import add_docs,ifnone
import torchdata.datapipes as dp
import torch
from torch import nn
from torchdata.dataloader2.graph import traverse_dps
# Local modules
from ..core import StepType,SimpleStep
from ..torch_core import evaluating,Module
from ..pipes.core import find_dps,find_dp

# %% ../../nbs/07_Agents/12a_agents.core.ipynb 4
class AgentBase(dp.iter.IterDataPipe):
    def __init__(self,
            model:nn.Module, # The base NN that we getting raw action values out of.
            action_iterator:list=None, # A reference to an iterator that contains actions to process.
            logger_bases=None
    ):
        self.model = model
        self.iterable = ifnone(action_iterator,[])
        self.agent_base = self
        self.logger_bases = logger_bases
        
    def to(self,*args,**kwargs):
        self.model.to(**kwargs)

    def __iter__(self):
        while self.iterable:
            yield self.iterable.pop(0)
            
add_docs(
AgentBase,
"""Acts as the footer of the Agent pipeline. 
Maintains important state such as the `model` being used for get actions from.
Also optionally allows passing a reference list of `action_iterator` which is a
persistent list of actions for the entire agent pipeline to process through.

> Important: Must be at the start of the pipeline, and be used with AgentHead at the end.

> Important: `action_iterator` is stored in the `iterable` field. However the recommended
way of passing actions to the pipeline is to call an `AgentHead` instance.
""",
to=torch.Tensor.to.__doc__
) 

# %% ../../nbs/07_Agents/12a_agents.core.ipynb 5
class AgentHead(dp.iter.IterDataPipe):
    def __init__(self,source_datapipe):
        self.source_datapipe = source_datapipe
        self.agent_base = find_dp(traverse_dps(self.source_datapipe),AgentBase)

    def __call__(self,steps:list):
        if issubclass(steps.__class__,StepType):
            raise Exception(f'Expected List[{StepType}] object got {type(steps)}\n{steps}')
        self.agent_base.iterable.extend(steps)
        return self

    def __iter__(self): yield from self.source_datapipe
    
    def augment_actions(self,actions): return actions

    def create_step(self,**kwargs): return SimpleStep(**kwargs)
    
add_docs(
    AgentHead,
    """Acts as the head of the Agent pipeline. 
    Used for conveniently adding actions to the pipeline to process.
    
    > Important: Must be paired with `AgentBase`
    """,
    augment_actions="""Called right before being fed into the env. 
    
    > Important: The results of this function will not be kept / used in the step or forwarded to 
    any training code.

    There are cases where either the entire action shouldn't be fed into the env,
    or the version of the action that we want to train on would be compat with the env.
    
    This is also useful if we want to train on the original raw values of the action prior to argmax being run on it for example.
    """,
    create_step="Creates the step used by the env for running, and used by the model for training."
)  

# %% ../../nbs/07_Agents/12a_agents.core.ipynb 6
class SimpleModelRunner(dp.iter.IterDataPipe):
    "Takes input from `source_datapipe` and pushes through the agent bases model assuming there is only one model field."
    def __init__(self,
                 source_datapipe
                ): 
        self.source_datapipe = source_datapipe
        self.agent_base = find_dp(traverse_dps(self.source_datapipe),AgentBase)
        self.device = None

    def to(self,*args,**kwargs):
        if 'device' in kwargs: self.device = kwargs.get('device',None)
    
    def __iter__(self):
        for x in self.source_datapipe:
            if self.device is not None: x = x.to(self.device)
            if len(x.shape)==1: x = x.unsqueeze(0)
            with evaluating(self.agent_base.model):
                res = self.agent_base.model(x)
            yield res

# %% ../../nbs/07_Agents/12a_agents.core.ipynb 12
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

# %% ../../nbs/07_Agents/12a_agents.core.ipynb 22
class StepModelFeeder(dp.iter.IterDataPipe):
    def __init__(self,
                 source_datapipe, # next() must produce a `StepType`,
                 keys:List[str] # A list of field names to grab and push into `self.agent_base.model`
                ): 
        self.source_datapipe = source_datapipe
        self.keys = keys
        self.agent_base = find_dp(traverse_dps(self.source_datapipe),AgentBase)

    def __iter__(self):
        for step in self.source_datapipe: 
            
            if not issubclass(step.__class__,StepType):
                raise Exception(f'Expected {StepType} object got {type(step)}\n{step}')
            
            tensors = tuple(getattr(step,k) for k in self.keys)
            
            try: yield self.agent_base.model(tensors)
            except Exception:
                print('Failed on ',step)
                raise
        
add_docs(
    StepModelFeeder,
    """Converts `StepTypes` into unified tensors using `keys` and feeds them into `self.agent_base.model`
    """
)  
    

# %% ../../nbs/07_Agents/12a_agents.core.ipynb 23
class NumpyConverter(dp.iter.IterDataPipe):
    debug=False

    def __init__(self,source_datapipe): 
        self.source_datapipe = source_datapipe
        
    def debug_display(self,step):
        print(f'Step: {step}')
    
    def __iter__(self) -> torch.LongTensor:
        for step in self.source_datapipe:
            if not issubclass(step.__class__,torch.Tensor):
                raise Exception(f'Expected Tensor to  convert to numpy, got {type(step)}\n{step}')
            if self.debug: self.debug_display(step)
            yield step.detach().cpu().numpy()

add_docs(
NumpyConverter,
"""Given input `Tensor` from `source_datapipe` returns a numpy array of same shape with argmax set to 1.""",
debug_display="Display the step being processed"
)
