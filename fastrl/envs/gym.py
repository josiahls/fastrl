# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/03_Environment/05b_envs.gym.ipynb.

# %% auto 0
__all__ = ['GymTypeTransform', 'GymStepper', 'GymTransformBlock']

# %% ../../nbs/03_Environment/05b_envs.gym.ipynb 3
# Python native modules
import os
import warnings
# Third party libs
from fastcore.all import *
import torchdata.datapipes as dp
import gym
from fastai.torch_basics import *
from fastai.torch_core import *
from torchdata.dataloader2.graph import find_dps,traverse
from ..data.dataloader2 import *
from torchdata.dataloader2 import DataLoader2,DataLoader2Iterator
from torchdata.dataloader2.graph import find_dps,traverse,DataPipe,IterDataPipe,MapDataPipe
# Local modules
from ..core import *
from ..pipes.core import *
from ..pipes.iter.nskip import *
from ..pipes.iter.nstep import *
from ..pipes.iter.firstlast import *
from ..pipes.iter.transforms import *
from ..pipes.map.transforms import *
from ..data.block import *

# %% ../../nbs/03_Environment/05b_envs.gym.ipynb 6
class GymTypeTransform(Transform):
    "Creates an gym.env"
    def encodes(self,o): return gym.make(o,new_step_api=True)

# %% ../../nbs/03_Environment/05b_envs.gym.ipynb 7
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

# %% ../../nbs/03_Environment/05b_envs.gym.ipynb 48
def GymTransformBlock(
    agent:DataPipe, # An AgentHead
    seed:Optional[int]=None, # The seed for the gym to use
    # Used by `NStepper`, outputs tuples / chunks of assiciated steps
    nsteps:int=1, 
    # Used by `NSkipper` to skip a certain number of steps (agent still gets called for each)
    nskips:int=1,
    # Whether when nsteps>1 to merge it into a single `StepType`
    firstlast:bool=False,
    # Functions to run once, at the beginning of the pipeline
    type_tfms:Optional[List[Callable]]=None,
    # Functions to run over individual steps before batching
    item_tfms:Optional[List[Callable]]=None,
    # Functions to run over batches (as specified by `bs`)
    batch_tfms:Optional[List[Callable]]=None,
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
    # Additional pipelines to insert, replace, remove
    dp_augmentation_fns:Tuple[DataPipeAugmentationFn]=None
) -> TransformBlock:
    "Basic OpenAi gym `DataPipeGraph` with first-last, nstep, and nskip capability"
    def _GymTransformBlock(
        # `source` likely will be an iterable that gets pushed into the pipeline when an 
        # experiment is actually being run.
        source:Any,
        # Any parameters needed for the dataloader
        num_workers:int=0,
        # This param must exist: as_dataloader for the datablock to create dataloaders
        as_dataloader:bool=False
    ) -> DataPipeOrDataLoader:
        _type_tfms = ifnone(type_tfms,GymTypeTransform)
        "This is the function that is actually run by `DataBlock`"
        pipe = dp.map.Mapper(source)
        pipe = TypeTransformer(pipe,_type_tfms)
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
                pipe = NStepFlattener(pipe) # We dont want to flatten if using FirstLastMerger
        if n is not None: pipe = pipe.header(limit=n)
        pipe = ItemTransformer(pipe,item_tfms)
        pipe  = pipe.batch(batch_size=bs)
        pipe = BatchTransformer(pipe,batch_tfms)
        
        pipe = apply_dp_augmentation_fns(pipe,ifnone(dp_augmentation_fns,()))
        
        if as_dataloader:
            pipe = DataLoader2(
                datapipe=pipe,
                reading_service=PrototypeMultiProcessingReadingService(
                    num_workers = num_workers,
                    protocol_client_type = InputItemIterDataPipeQueueProtocolClient,
                    protocol_server_type = InputItemIterDataPipeQueueProtocolServer,
                    pipe_type = item_input_pipe_type,
                    eventloop = SpawnProcessForDataPipeline
                ) if num_workers>0 else None
            )
        return pipe 
    return _GymTransformBlock
