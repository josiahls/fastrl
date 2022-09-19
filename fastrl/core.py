# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_core.ipynb.

# %% auto 0
__all__ = ['StepType', 'add_namedtuple_doc', 'SimpleStep', 'Record', 'test_in', 'test_len', 'test_lt']

# %% ../nbs/00_core.ipynb 2
# Python native modules
import os,warnings,typing
# Third party libs
from fastcore.all import *
from .torch_core import *
from fastai.basics import *
import pandas as pd
import torch
import numpy as np
# Local modules

# %% ../nbs/00_core.ipynb 6
def _fmt_fld(t:typing.Tuple[str,type],namedtuple):
    default_v = ''
    if t[0] in namedtuple._field_defaults:
        default_v = f' = `{namedtuple._field_defaults[t[0]]}`'
    return ' - **%s**:`%s` '%t+default_v+getattr(namedtuple,t[0]).__doc__

def add_namedtuple_doc(
    t:typing.NamedTuple, # Primary tuple to get docs from
    doc:str, # Primary doc for the overall tuple, where the docs for individual fields will be concated.
    **fields_docs:dict # Field names with associated docs to be attached in the format: field_a='some documentation'
):
    "Add docs to `t` from `doc` along with individual doc fields `fields_docs`"
    if not hasattr(t,'__base_doc__'): t.__base_doc__ = doc
    for k,v in fields_docs.items(): getattr(t,k).__doc__ = v
    # TODO: can we add optional default fields also?
    flds = L(t.__annotations__.items()).map(_fmt_fld,namedtuple=t)
    
    s = 'Parameters:\n'+'\n'.join(flds)
    t.__doc__ = doc + '\n\n' + s    

# %% ../nbs/00_core.ipynb 7
class SimpleStep(typing.NamedTuple):
    state:       torch.FloatTensor=torch.FloatTensor([0])
    action:      torch.FloatTensor=torch.FloatTensor([0])
    next_state:  torch.FloatTensor=torch.FloatTensor([0])
    terminated:  torch.BoolTensor=torch.BoolTensor([1])
    truncated:   torch.BoolTensor=torch.BoolTensor([1])
    reward:      torch.FloatTensor=torch.LongTensor([0])
    total_reward:torch.FloatTensor=torch.FloatTensor([0])
    env_id:      torch.LongTensor=torch.LongTensor([0])
    proc_id:     torch.LongTensor=torch.LongTensor([0])
    step_n:      torch.LongTensor=torch.LongTensor([0])
    episode_n:   torch.LongTensor=torch.LongTensor([0])
    image:       torch.FloatTensor=torch.FloatTensor([0])
    
    def clone(self):
        return self.__class__(
            **{fld:getattr(self,fld).clone() for fld in self.__class__._fields}
        )
    
    def detach(self):
        return self.__class__(
            **{fld:getattr(self,fld).detach() for fld in self.__class__._fields}
        )
    
    def device(self,device='cpu'):
        return self.__class__(
            **{fld:getattr(self,fld).to(device=device) for fld in self.__class__._fields}
        )
    
    @classmethod
    def random(cls,seed=None,**flds):
        _flds,_annos = cls._fields,cls.__annotations__

        def _random_annos(anno):
            t = anno(1)
            if anno==torch.BoolTensor: t.random_(2) 
            else:                      t.random_(100)
            return t

        return cls(
            *(flds.get(
                f,_random_annos(_annos[f])
            ) for f in _flds)
        )

add_namedtuple_doc(
    SimpleStep,
    'Represents a single step in an environment.',
    state = 'Both the initial state of the environment and the previous state.',
    next_state = 'Both the next state, and the last state in the environment',
    terminated = """Represents an ending condition for an environment such as reaching a goal or 'living long enough' as 
                    described by the MDP.
                    Good reference is: https://github.com/openai/gym/blob/39b8661cb09f19cb8c8d2f59b57417517de89cb0/gym/core.py#L151-L155""",
    truncated = """Represents an ending condition for an environment that can be seen as an out of bounds condition either
                   literally going out of bounds, breaking rules, or exceeding the timelimit allowed by the MDP.
                   Good reference is: https://github.com/openai/gym/blob/39b8661cb09f19cb8c8d2f59b57417517de89cb0/gym/core.py#L151-L155'""",
    reward = 'The single reward for this step.',
    total_reward = 'The total accumulated reward for this episode up to this step.',
    action = 'The action that was taken to transition from `state` to `next_state`',
    env_id = 'The environment this step came from (useful for debugging)',
    proc_id = 'The process this step came from (useful for debugging)',
    step_n = 'The step number in a given episode.',
    episode_n = 'The episode this environment is currently running through.',
    image = """Intended for display and logging only. If the intention is to use images for training an
               agent, then use a env wrapper instead."""
)

# %% ../nbs/00_core.ipynb 14
StepType = (SimpleStep,)

# %% ../nbs/00_core.ipynb 15
class Record(typing.NamedTuple):
    name:str
    value:typing.Any

# %% ../nbs/00_core.ipynb 17
def test_in(a,b):
    "`test` that `a in b`"
    test(a,b,in_, ' in ')

# %% ../nbs/00_core.ipynb 19
def _len_check(a,b): 
    return len(a)==(len(b) if not isinstance(b,int) else b)

def test_len(a,b,meta_info=''):
    "`test` that `len(a) == int(b) or len(a) == len(b)`"
    test(a,b,_len_check, f' len == len {meta_info}')

# %% ../nbs/00_core.ipynb 21
def _less_than(a,b): return a < b
def test_lt(a,b):
    "`test` that `a < b`"
    test(a,b,_less_than, ' a < b')
