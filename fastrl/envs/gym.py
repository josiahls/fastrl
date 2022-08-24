# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/05b_envs.gym.ipynb.

# %% auto 0
__all__ = ['GymTypeTransform', 'GymStepper', 'isrecord', 'isnone', 'GymTransformBlock']

# %% ../nbs/05b_envs.gym.ipynb 3
# Python native modules
import os
import warnings
# Third party libs
from fastcore.all import *
import torchdata.datapipes as dp
import gym
from fastai.torch_basics import *
from fastai.torch_core import *
from torch.utils.data.dataloader_experimental import DataLoader2
from torchdata.dataloader2.graph import find_dps,traverse
# Local modules
from ..core import *
from ..pipes.core import *
from ..pipes.iter.nskip import *
from ..pipes.iter.nstep import *
from ..pipes.iter.firstlast import *
from ..fastai.data.block import *

# %% ../nbs/05b_envs.gym.ipynb 6
class GymTypeTransform(Transform):
    "Creates an gym.env"
    def encodes(self,o): return gym.make(o,new_step_api=True)

# %% ../nbs/05b_envs.gym.ipynb 7
class GymStepper(dp.iter.IterDataPipe):
    def __init__(self,
        source_datapipe:Union[Iterable,dp.iter.IterDataPipe], # Calling `next()` should produce a `gym.Env`
        agent=None, # Optional `Agent` that accepts a `SimpleStep` to produce a list of actions.
        seed:int=None, # Optional seed to set the env to and also random action sames if `agent==None`
        synchronized_reset:bool=False, # Some `gym.Envs` require reset to be terminated on *all* envs before proceeding to step.
        include_images:bool=False,
    ):
        self.source_datapipe = source_datapipe
        self.agent = agent
        self.seed = seed
        self.include_images = include_images
        self.synchronized_reset = synchronized_reset
        self._env_ids = {}
        
    def env_reset(self,
      env:gym.Env, # The env to rest along with its numeric object id
      env_id:int # Resets env in `self._env_ids[env_id]`
    ) -> SimpleStep:
        state = env.reset(seed=self.seed)
        env.action_space.seed(seed=self.seed)
        episode_n = self._env_ids[env_id].episode_n+1 if env_id in self._env_ids else tensor(1)

        step = (self.no_agent_create_step if self.agent is None else self.agent.create_step)(
            state=tensor(state),
            next_state=tensor(state),
            terminated=tensor(False),
            truncated=tensor(False),
            reward=tensor(0),
            total_reward=tensor(0.),
            env_id=tensor(env_id),
            proc_id=tensor(os.getpid()),
            step_n=tensor(0),
            episode_n=episode_n,
            image=env.render(mode='rgb_array') if self.include_images else torch.FloatTensor([0])
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
            for action in (self.agent([step]) if self.agent is not None else [env.action_space.sample()]):
                next_state,reward,terminated,truncated,_ = env.step(
                    self.agent.augment_actions(action) if self.agent is not None else action
                )

                step = (self.no_agent_create_step if self.agent is None else self.agent.create_step)(
                    state=tensor(step.next_state),
                    next_state=tensor(next_state),
                    action=tensor(action).float(),
                    terminated=tensor(terminated),
                    truncated=tensor(truncated),
                    reward=tensor(reward),
                    total_reward=step.total_reward+reward,
                    env_id=tensor(env_id),
                    proc_id=tensor(os.getpid()),
                    step_n=step.step_n+1,
                    episode_n=step.episode_n,
                    image=env.render(mode='rgb_array') if self.include_images else torch.FloatTensor([0])
                )
                self._env_ids[env_id] = step
                yield step
                if terminated: break
            if action is None: 
                raise Exception('The agent produced no actions. This should never occur.')
                
add_docs(
    GymStepper,
    """Accepts a `source_datapipe` or iterable whose `next()` produces a single `gym.Env`.
       Tracks multiple envs using `id(env)`.""",
    env_reset="Resets a env given the env_id.",
    no_agent_create_step="If there is no agent for creating the step output, then `GymStepper` will create its own"
)

# %% ../nbs/05b_envs.gym.ipynb 48
def isrecord(o): return type(o)==Record
def isnone(o): return o is None
def GymTransformBlock(
    agent,
    seed=None,
    nsteps=1,
    nskips=1,
    firstlast=False,
    dl_type=DataLoader2,
    pipe_fn_kwargs=None,
    type_tfms=None,
    **kwargs
):
    pipe_fn_kwargs = ifnone(pipe_fn_kwargs,{})
    type_tfms = ifnone(type_tfms,GymTypeTransform)
    
    def pipe_fn(source:List[str],bs,n,seed,nsteps,nskips,synchronized_reset=False, 
                firstlast=False,
                include_images=False,type_tfms=None,item_tfms=None,batch_tfms=None,cbs=None):
        pipe = dp.map.Mapper(source)
        pipe = TypeTransformLoop(pipe,type_tfms)
        pipe = dp.iter.MapToIterConverter(pipe)
        pipe = dp.iter.InMemoryCacheHolder(pipe)
        pipe = pipe.cycle() # Cycle through the envs inf
        pipe = GymStepper(pipe,agent=agent,seed=seed,
                          include_images=include_images,synchronized_reset=synchronized_reset)
        if nskips!=1: pipe = NSkipper(pipe,n=nskips)
        if nsteps!=1:
            pipe = NStepper(pipe,n=nsteps)
            if firstlast:
                pipe = FirstLastMerger(pipe)
            else:
                pipe = dp.iter.Flattener(pipe) # We dont want to flatten if using FirstLastMerger
        if n is not None: pipe = pipe.header(limit=n)
        pipe = ItemTransformLoop(pipe,item_tfms)
        pipe  = pipe.batch(batch_size=bs)
        pipe = BatchTransformLoop(pipe,batch_tfms)
        return pipe

    return TransformBlock(
        pipe_fn = pipe_fn,
        dl_type = dl_type,
        pipe_fn_kwargs = merge(pipe_fn_kwargs,kwargs,dict(nsteps=nsteps,nskips=nskips,seed=seed,firstlast=firstlast,
                                                          type_tfms=type_tfms))
    )
