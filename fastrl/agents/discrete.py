# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/07_Agents/01_Discrete/12b_agents.discrete.ipynb.

# %% auto 0
__all__ = ['ArgMaxer', 'EpsilonSelector', 'EpsilonCollector', 'PyPrimativeConverter']

# %% ../../nbs/07_Agents/01_Discrete/12b_agents.discrete.ipynb 2
# Python native modules
import os
# Third party libs
# from fastcore.all import *
import torchdata.datapipes as dp
import torch
# from torch.nn import *
import torch.nn.functional as F
from torchdata.dataloader2.graph import traverse_dps
import numpy as np
# Local modules
# from fastrl.core import *
# from fastrl.pipes.core import *
# from fastrl.agents.core import *
# from fastrl.loggers.core import *
# from fastrl.torch_core import *

# %% ../../nbs/07_Agents/01_Discrete/12b_agents.discrete.ipynb 4
class ArgMaxer(dp.iter.IterDataPipe):
    debug=False
    
    "Given input `Tensor` from `source_datapipe` returns a tensor of same shape with argmax set to 1."
    def __init__(self,source_datapipe,axis=1,only_idx=False): 
        self.source_datapipe = source_datapipe
        self.axis = axis
        self.only_idx = only_idx
        
    def debug_display(self,step,idx):
        print(f'Step: {step}\n{idx}')
    
    def __iter__(self) -> torch.LongTensor:
        for step in self.source_datapipe:
            if not issubclass(step.__class__,torch.Tensor):
                raise Exception(f'Expected Tensor to take the argmax, got {type(step)}\n{step}')
            # Might want to support simple tuples also depending on if we are processing multiple fields.
            idx = torch.argmax(step,axis=self.axis).reshape(-1,1)
            if self.only_idx: 
                yield idx.long()
                continue
            step[:] = 0
            if self.debug: self.debug_display(step,idx)
            step.scatter_(1,idx,1)
            yield step.long()
            

# %% ../../nbs/07_Agents/01_Discrete/12b_agents.discrete.ipynb 8
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
            ret_mask:bool=False,
            # The device to create the masks one
            device='cpu'
        ): 
        self.source_datapipe = source_datapipe
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.decrement_on_val = decrement_on_val
        self.select_on_val = select_on_val
        self.ret_mask = ret_mask
        self.agent_base = find_dp(traverse(self.source_datapipe,only_datapipe=True),AgentBase)
        self.step = 0
        self.device = torch.device(device)
    
    def __iter__(self):
        for action in self.source_datapipe:
            # TODO: Support tuples of actions also
            if not issubclass(action.__class__,torch.Tensor):
                raise Exception(f'Expected Tensor, got {type(action)}\n{action}')
            if action.dtype!=torch.int64:
                raise ValueError(f'Expected Tensor of dtype int64, got: {action.dtype} from {self.source_datapipe}')
                
            if self.agent_base.model.training or self.decrement_on_val:
                self.step+=1
                
            self.epsilon = max(self.min_epsilon,self.max_epsilon-self.step/self.max_steps)
            # Add a batch dim if missing
            if len(action.shape)==1: action.unsqueeze_(0)
            mask = None
            if self.agent_base.model.training or self.select_on_val:
                # Given N(action.shape[0]) actions, select the ones we want to randomly assign... 
                mask = torch.rand(action.shape[0],).to(self.device)<self.epsilon
                # Get random actions as their indexes
                rand_action_idxs = torch.LongTensor(int(mask.sum().long()),).to(self.device).random_(action.shape[1])
                # If the input action is [[0,1],[1,0]] and...
                # If mask is [True,False] and...
                # if rand_action_idxs is [0]
                # the action[mask] will have [[1,0]] assigned to it resulting in... 
                # an action with [[1,0],[1,0]]
                # print(action.shape[1])
                if self.debug: print(f'Mask: {mask}\nRandom Actions: {rand_action_idxs}\nPre-random Actions: {action}')
                action[mask] = F.one_hot(rand_action_idxs,action.shape[1])
            
            yield ((action,mask) if self.ret_mask else action)

# %% ../../nbs/07_Agents/01_Discrete/12b_agents.discrete.ipynb 22
class EpsilonCollector(LogCollector):
    header:str='epsilon'
    # def __init__(self,
    #      source_datapipe, # The parent datapipe, likely the one to collect metrics from
    #      logger_bases:List[LoggerBase] # `LoggerBase`s that we want to send metrics to
    #     ):
    #     self.source_datapipe = source_datapipe
    #     self.main_buffers = [o.buffer for o in logger_bases]
        
    def __iter__(self):
        # for q in self.main_buffers: q.append(Record('epsilon',None))
        for action in self.source_datapipe:
            for q in self.main_buffers: 
                q.append(Record('epsilon',self.source_datapipe.epsilon))
            yield action

# %% ../../nbs/07_Agents/01_Discrete/12b_agents.discrete.ipynb 23
class PyPrimativeConverter(dp.iter.IterDataPipe):
    debug=False
    
    "Given input `Tensor` from `source_datapipe` returns a numpy array of same shape with argmax set to 1."
    def __init__(self,source_datapipe,remove_batch_dim=True): 
        self.source_datapipe = source_datapipe
        self.remove_batch_dim = remove_batch_dim
        
    def debug_display(self,step): print(f'Step: {step}')
    
    def __iter__(self) -> Union[float,bool,int]:
        for step in self.source_datapipe:
            if not issubclass(step.__class__,(np.ndarray)):
                raise Exception(f'Expected list or np.ndarray to  convert to python primitive, got {type(step)}\n{step}')
            if self.debug: self.debug_display(step)
            
            if len(step)>1 or len(step)==0:
                raise Exception(f'`step` from {self.source_datapipe} needs to be len 1, not {len(step)}')
            else:
                step = step[0]
                
            if np.issubdtype(step.dtype,np.integer):
                yield int(step)
            elif np.issubdtype(step.dtype,np.floating):
                yield float(step)
            elif np.issubdtype(step.dtype,np.bool8):
                yield bool(step)
            else:
                raise Exception(f'`step` from {self.source_datapipe} must be one of the 3 python types: bool,int,float, not {step.dtype}')
