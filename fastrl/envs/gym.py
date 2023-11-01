# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/03_Environment/05b_envs.gym.ipynb.

# %% auto 0
__all__ = ['GymStepper', 'GymDataPipe']

# %% ../../nbs/03_Environment/05b_envs.gym.ipynb 2
# Python native modules
import os
import warnings
from functools import partial
from typing import Callable, Any, Union, Iterable, Optional
# Third party libs
import gymnasium as gym
import torch
# from fastrl.torch_core import *
from fastcore.all import add_docs
import torchdata.datapipes as dp
from torchdata.dataloader2.graph import find_dps,DataPipeGraph,DataPipe,traverse_dps
from torchdata.dataloader2 import MultiProcessingReadingService,DataLoader2
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.map import MapDataPipe
# Local modules
from ..core import StepTypes,SimpleStep
from ..pipes.core import find_dps
from ..pipes.iter.nskip import NSkipper
from ..pipes.iter.nstep import NStepper,NStepFlattener
from ..pipes.iter.firstlast import FirstLastMerger
import fastrl.pipes.iter.cacheholder

# %% ../../nbs/03_Environment/05b_envs.gym.ipynb 5
class GymStepper(dp.iter.IterDataPipe):
    def __init__(self,
        source_datapipe:Union[Iterable,dp.iter.IterDataPipe], # Calling `next()` should produce a `gym.Env`
        agent=None, # Optional `Agent` that accepts a `SimpleStep` to produce a list of actions.
        seed:int=None, # Optional seed to set the env to and also random action sames if `agent==None`
        synchronized_reset:bool=False, # Some `gym.Envs` require reset to be terminated on *all* envs before proceeding to step.
        include_images:bool=False, # Render images from the environment
        terminate_on_truncation:bool=True
    ):
        self.source_datapipe = source_datapipe
        self.agent = agent
        self._agent_iter = None
        self.seed = seed
        self.include_images = include_images
        self.synchronized_reset = synchronized_reset
        self.terminate_on_truncation = terminate_on_truncation
        self._env_ids = {}
        
    def env_reset(self,
      env:gym.Env, # The env to rest along with its numeric object id
      env_id:int # Resets env in `self._env_ids[env_id]`
    ) -> StepTypes.types:
        # self.agent.reset()
        state, info = env.reset(seed=self.seed)
        env.action_space.seed(seed=self.seed)
        episode_n = self._env_ids[env_id].episode_n+1 if env_id in self._env_ids else torch.tensor(1)

        step = (self.no_agent_create_step if self.agent is None else self.agent.create_step)(
            state=torch.tensor(state),
            next_state=torch.tensor(state),
            terminated=torch.tensor(False),
            truncated=torch.tensor(False),
            reward=torch.tensor(0),
            total_reward=torch.tensor(0.),
            env_id=torch.tensor(env_id),
            proc_id=torch.tensor(os.getpid()),
            step_n=torch.tensor(0),
            episode_n=episode_n,
            # image=env.render(mode='rgb_array') if self.include_images else torch.FloatTensor([0])
            image=torch.tensor(env.render()) if self.include_images else torch.FloatTensor([0]),
            raw_action=torch.FloatTensor([0])
        )
        self._env_ids[env_id] = step
        return step
    
    def no_agent_create_step(self,**kwargs): return SimpleStep(**kwargs)

    def __iter__(self) -> SimpleStep:
        for env in self.source_datapipe:
            assert issubclass(env.__class__,gym.Env),f'Expected subclass of gym.Env, but got {env.__class__}'    
            env_id = id(env)
            
            if env_id not in self._env_ids or self._env_ids[env_id].terminated:
                if self.synchronized_reset:
                    if env_id in self._env_ids \
                    and not self._env_ids[env_id].terminated \
                    and self._resetting_all:
                        # If this env has already been reset, and we are currently in the 
                        # self._resetting_all phase, then skip this so we can reset all remaining envs
                        continue
                    elif env_id not in self._env_ids \
                    or all([self._env_ids[s].terminated for s in self._env_ids])\
                    or self._resetting_all:
                        # If the id is not in the _env_ids, we can assume this is a fresh start.
                        # OR 
                        # If all the envs are terminated, then we can start doing a reset operation.
                        # OR
                        # If we are currently resetting all the envs anyways
                        # This means we want to reset ALL the envs before doing any steps.
                        self.env_reset(env,env_id)
                        # Move to the next env, eventually we will reset all the envs in sync.
                        # then we will be able to start calling `step` for each of them.
                        # _resetting_all is True when there are envs still "terminated".
                        self._resetting_all = any([self._env_ids[s].terminated for s in self._env_ids])
                        continue 
                    elif self._env_ids[env_id].terminated:
                        continue
                    else:
                        raise ValueError('This else should never happen.')
                else:
                    step = self.env_reset(env,env_id)
            else:
                step = self._env_ids[env_id]

            action = None
            raw_action = None
            self._agent_iter = iter((self.agent([step]) if self.agent is not None else [env.action_space.sample()]))
            while True:
                try:
                    action = next(self._agent_iter)
                    if isinstance(action,tuple):
                        action, raw_action = action
                    next_state,reward,terminated,truncated,_ = env.step(
                        self.agent.augment_actions(action) if self.agent is not None else action
                    )

                    if self.terminate_on_truncation and truncated: terminated = True

                    step = (self.no_agent_create_step if self.agent is None else self.agent.create_step)(
                        state=step.next_state.clone().detach(),
                        next_state=torch.tensor(next_state),
                        action=torch.tensor(action).float(),
                        terminated=torch.tensor(terminated),
                        truncated=torch.tensor(truncated),
                        reward=torch.tensor(reward),
                        total_reward=step.total_reward+reward,
                        env_id=torch.tensor(env_id),
                        proc_id=torch.tensor(os.getpid()),
                        step_n=step.step_n+1,
                        episode_n=step.episode_n,
                        # image=env.render(mode='rgb_array') if self.include_images else torch.FloatTensor([0])
                        image=torch.tensor(env.render()) if self.include_images else torch.FloatTensor([0]),
                        raw_action=raw_action if raw_action is not None else torch.FloatTensor([0])
                    )
                    self._env_ids[env_id] = step
                    yield step
                    if terminated: break
                except StopIteration:
                    self._agent_iter = None
                    break
                finally:
                    if self._agent_iter is not None:
                        while True:
                            try: next(self._agent_iter)
                            except StopIteration:break
            if action is None: 
                raise Exception('The agent produced no actions. This should never occur.')
                
add_docs(
GymStepper,
"""Accepts a `source_datapipe` or iterable whose `next()` produces a single `gym.Env`.
    Tracks multiple envs using `id(env)`.""",
env_reset="Resets a env given the env_id.",
no_agent_create_step="If there is no agent for creating the step output, then `GymStepper` will create its own",
reset="Resets the env's back to original str types to avoid pickling issues."
)

# %% ../../nbs/03_Environment/05b_envs.gym.ipynb 56
def GymDataPipe(
    source,
    agent:DataPipe=None, # An AgentHead
    seed:Optional[int]=None, # The seed for the gym to use
    # Used by `NStepper`, outputs tuples / chunks of assiciated steps
    nsteps:int=1, 
    # Used by `NSkipper` to skip a certain number of steps (agent still gets called for each)
    nskips:int=1,
    # Whether when nsteps>1 to merge it into a single `StepType`
    firstlast:bool=False,
    # The batch size, which is different from `nsteps` in that firstlast will be 
    # run prior to batching, and a batch of steps might come from multiple envs,
    # where nstep is associated with a single env
    bs:int=1,
    # The prefered default is for the pipeline to be infinate, and the learner
    # decides how much to iter. If this is not None, then the pipeline will run for 
    # that number of `n`
    n:Optional[int]=None,
    # Whether to reset all the envs at the same time as opposed to reseting them 
    # the moment an episode ends. 
    synchronized_reset:bool=False,
    # Should be used only for validation / logging, will grab a render of the gym
    # and assign to the `StepType` image field. This data should not be used for training.
    # If it images are needed for training, then you should wrap the env instead. 
    include_images:bool=False,
    # If an environment truncates, terminate it.
    terminate_on_truncation:bool=True
) -> Callable:
    "Basic `gymnasium` `DataPipeGraph` with first-last, nstep, and nskip capability"
    pipe = dp.iter.IterableWrapper(source)
    if include_images:
        pipe = pipe.map(partial(gym.make,render_mode='rgb_array'))
    else:
        pipe = pipe.map(gym.make)
    # pipe = dp.iter.InMemoryCacheHolder(pipe)
    pipe = pipe.pickleable_in_memory_cache()
    pipe = pipe.cycle() # Cycle through the envs inf
    pipe = GymStepper(pipe,agent=agent,seed=seed,
                        include_images=include_images,
                        terminate_on_truncation=terminate_on_truncation,
                        synchronized_reset=synchronized_reset)
    if nskips!=1: pipe = NSkipper(pipe,n=nskips)
    if nsteps!=1:
        pipe = NStepper(pipe,n=nsteps)
        if firstlast:
            pipe = FirstLastMerger(pipe)
        else:
            pipe = NStepFlattener(pipe) # We dont want to flatten if using FirstLastMerger
    if n is not None: pipe = pipe.header(limit=n)
    pipe  = pipe.batch(batch_size=bs)
    return pipe
