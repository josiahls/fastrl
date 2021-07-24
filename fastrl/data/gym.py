# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/05c_data.gym.ipynb (unless otherwise specified).

__all__ = ['source_events', 'return_data', 'Source', 'ReturnHistory', 'TstCallback', 'np_type_cast', 'GymLoop',
           'FirstLastTfm', 'FirstLast', 'Reward', 'NEpisodes']

# Cell
# Python native modules
import os
from collections import deque
from copy import deepcopy
from time import sleep
from typing import *
# Third party libs
from fastcore.all import *
from fastai.torch_basics import *
from fastai.data.all import *
from fastai.basics import *
from fastai.callback.all import *
from torch.utils.data import Dataset
from torch import nn
import torch
import gym
import pybulletgym
import numpy as np
# Local modules
from ..core import *
from ..callback.core import *
from .block_simple import *
from ..agent import *

# Cell
_loop=L(['event.after_create','Start Setup','event.initialize','End Setup',
             'event.before_episodes',
             'Start Episodes',
                 'event.reset',
                 'event.do_action',
                 'event.do_step',
                 'event.do_render',
                 'event.before_history',
                 'event.history',
                 'event.after_history',
             'End Episodes',
             'event.after_episodes'
             ])

mk_class('source_events', **parse_events(_loop).map_dict(),
         doc="All possible events as attributes to get tab-completion and typo-proofing")

#nbdev_comment _all_=['source_events']

# Cell
def return_data(o:dict):
    source,data=o['source'],o['history']

    return {'source':source,'history':data}

class Source(Loop):
    _loop=_loop
    _events=source_events
    _default='source'
    end_event=parse_events(_loop)[-1]

    @delegates(Loop)
    def __init__(self,cbs=None,test_mode=False,**kwargs):
        self.idx=0
        self.data_fields='state,next_state,done,all_done,env_id,worker_id'\
                           ',action,episode_id,accum_rewards,reward,step'.split(',')
        self.ignore_fields=[]
        self.test_field=torch.full((1,5),self.idx)
        self.return_fn=return_data
        self.loop_history_yield=False
        store_attr(but='cbs',state=None,next_state=None,done=None,all_done=None,
                   env_id=0,action=None,episode_id=0,accum_rewards=0,reward=0,step=0,
                   skip_history_return=False)
        super().__init__(cbs=cbs,**kwargs)

    def after_create(self):
        self('initialize')
        return self

    def _history(self):
        self.loop_history_yield=False
        self('history')
        if self.test_mode:
            self.this=torch.full((1,5),self.idx)
            if 'test_field' not in self.data_fields: self.data_fields.append('test_field')
        return self.return_fn(dict(source=self,history=self.data()))['history']

    def data(self)->BD:
        return BD({s:(ifnone(getattr(self,s,None),TensorBatch([[0]])) if self.test_mode else getattr(self,s))
                for s in self.data_fields if not in_(s,self.ignore_fields)})

    def __iter__(self):
        self('before_episodes')
        while True:
            self.idx+=1
            self('reset')
            self('do_action')
            self('do_step')
            if self.test_mode: self.test_field=torch.full((1,5),self.idx)
            self('do_render')
            self('before_history')
            if not self.skip_history_return: yield self._history()
            while self.loop_history_yield:   yield self._history()
            self('after_history')

# Cell
class ReturnHistory(Transform):
    def encodes(self,o:dict):
        source,data=o['source'],o['history']
        history=sum(source.histories)
        if len(source.histories)==1 and sum(history['done'])>0:
            source.all_done=TensorBatch([[True]])
            history['all_done']=TensorBatch([[True]])
        source.histories.popleft()
        return {'source':source,'history':history}

# Cell
class TstCallback(AgentCallback):
    def __init__(self,action_space=None,constant=None): store_attr()
    def before_noise(self):
        bs=self.experience['state'].shape[0]
        self.agent.action=Tensor([[self.constant] if self.constant is not None else
                                   self.action_space.sample()
                                   for _ in range(bs)])
        self.agent.experience=D(merge(self.experience,{'random_action':np.random.randint(0,3,(bs,1))}))

def np_type_cast(a,dtype): return a.astype(dtype)

class GymLoop(LoopCallback):
    _methods=source_events
    _default='source'
    def __init__(self,env_name:str,agent=None,mode=None,
                 steps_count:int=1,steps_delta:int=1,seed:int=None,
                 worker_id:int=0,dtype=None):
        store_attr()

    def initialize(self):
        self.source.histories=deque(maxlen=self.steps_count)
        self.source.return_fn=Pipeline([ReturnHistory])
        if self.mode!='rgb_array': self.source.ignore_fields.append('image')
        self.source.worker_id=get_worker_info()
        self.source.worker_id=TensorBatch([[getattr(self.worker_id,'id',0)]])

        self.source.env=gym.make(self.env_name)
        if isinstance(self.env.action_space,gym.spaces.Discrete):
            self.dtype=int
        elif isinstance(self.env.action_space,gym.spaces.Box):
            if issubclass(self.env.action_space.dtype.type,(float,np.floating)):
                self.dtype=partial(np_type_cast,dtype=float)
            if issubclass(self.env.action_space.dtype.type,(int,np.integer)):
                self.dtype=partial(np_type_cast,dtype=int)
        else:
            self.dtype=float

        self.agent=ifnone(self.agent,Agent(cbs=TstCallback(action_space=self.env.action_space)))
        self.source.episode_id=TensorBatch([[0]])
        self.source.done=TensorBatch([[True]])
        self.source.all_done=TensorBatch([[True]])

    def reset(self):
        if len(self.histories)==0 and sum(self.all_done)>=1:
            self.source.env_id=TensorBatch([[0]])
            self.source.reward=TensorBatch([[0.0]])
            self.source.episode_id+=1
            self.source.accum_rewards=TensorBatch([[0.0]])
            self.source.step=TensorBatch([[0]])
            self.source.env_id=TensorBatch([[0]])
            self.source.done=TensorBatch([[False]])
            self.source.all_done=TensorBatch([[False]])
            self.source.env.seed(self.seed)
            self.source.state=TensorBatch(self.env.reset()).unsqueeze(0)

    def before_episodes(self): self('initialize')

    def do_step(self):
        action,experience=self.agent.do_action(**self.data())
        for k,v in experience.items():
            if v is not None: setattr(self.source,k,TensorBatch(v))
        self.source.action=TensorBatch(action)
        step_action=self.action.numpy()[0]
        next_state,reward,done,_=self.env.step(self.dtype(step_action))
        self.source.next_state=TensorBatch(next_state).unsqueeze(0)
        self.source.reward=TensorBatch([[reward]])
        self.source.accum_rewards+=reward
        self.source.done=TensorBatch([[done]])
        self.source.step+=1

    def before_history(self):
        self.source.skip_history_return=False
        self.histories.append(deepcopy(self.data()))

        if self.done.sum().item()<1:
            self.source.skip_history_return=len(self.histories)<self.steps_count or \
                                            int(self.step.item())%self.steps_delta!=0

    def history(self):
        if sum(self.done)>0 and len(self.histories)>1 and self.steps_count>1:
            self.source.loop_history_yield=True


add_docs(GymLoop,"Handles iterating through single openai gym (and varients).",
         before_history="""Primarily appends `self.data()` to `self.histories` however it also...

         If the number of elements in `self.histories` is empty, or about to be empty, and
         `self.done==True` it sets the `self.all_done` field.

         If `self.done==True` also determine if we should skip returning a history.
         Mainly if we need to accumulate the histories first.
         """,
         initialize="Sets up most of the needed fields as well as the environment itself.",
         reset="Resets the env and fields if the histories have been emptied.",
         before_episodes="Call the initialization method again.",
         do_step="Get actions from the agent and take a step through the env.",
         history="If the environment is done, we want to loop through the histories and empty them.")

# Cell
class FirstLastTfm(Transform):
    def __init__(self,gamma):
        store_attr()
        super().__init__()


    def encodes(self,o):
#         print('encoding')
        source,history=o['source'],o['history']
#         print(source,history,o)
        element=history[0]
        if history.bs()!=1:
            remainder=history[1:]
            reward=element['reward']
            for e in reversed(remainder['reward']):
                reward*=self.gamma
                reward+=e
            element['reward']=reward
            element['next_state']=history[-1]['next_state']
            element['done']=history[-1]['done']

#         print(source,element)
        return {'source':source,'history':element}

class FirstLast(LoopCallback):
    _methods=source_events
    _default='source'

    def __init__(self,gamma=0.99): store_attr()
    def before_episodes(self): self('initialize')
    def initialize(self):
        if isinstance(self.source.return_fn,Pipeline):
            self.source.return_fn.add(FirstLastTfm(self.gamma))
        else:
            self.source.return_fn=Pipeline([self.source.return_fn,FirstLastTfm(self.gamma)])

# Cell
class Reward(Metric):
    order=30
    reward=None
    rolling_reward_n=100
    keep_rewards=False

    def reset(self):
        if self.reward is None: self.reward=deque(maxlen=self.rolling_reward_n)

    def accumulate(self,learn):
        xb=learn.xb[0]
        if xb['all_done'].sum()>0:
            final_rewards=to_detach(xb['accum_rewards'][xb['all_done']])
            for i in final_rewards.reshape(-1,).numpy().tolist(): self.reward.append(i)

    @property
    def value(self): return np.average(self.reward) if len(self.reward)>0 else 0

# Cell
class NEpisodes(Metric):
    order=30
    n_episodes=None

    def reset(self):
        if self.n_episodes is None: self.n_episodes=0

    def accumulate(self,learn):
        xb=learn.xb[0]
        if xb['all_done'].sum()>0:
            new_episodes=torch.unique(to_detach(xb['episode_id'][xb['all_done']]))
            self.n_episodes+=len(new_episodes)


    @property
    def value(self): return self.n_episodes