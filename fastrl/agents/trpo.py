# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/07_Agents/02_Continuous/12t_agents.trpo.ipynb.

# %% auto 0
__all__ = ['AdvantageStep', 'pipe2device', 'discounted_cumsum_', 'get_flat_params_from', 'set_flat_params_to', 'AdvantageBuffer',
           'OptionalClampLinear', 'Actor', 'NormalExploration', 'AdvantageGymTransformBlock',
           'ProbabilisticStdCollector', 'ProbabilisticMeanCollector', 'TRPOAgent', 'conjugate_gradients',
           'backtrack_line_search', 'actor_prob_loss', 'pre_hessian_kl', 'auto_flat', 'forward_pass',
           'CriticLossProcessor', 'ActorOptAndLossProcessor', 'TRPOLearner']

# %% ../../nbs/07_Agents/02_Continuous/12t_agents.trpo.ipynb 3
# Python native modules
from typing import *
from typing_extensions import Literal
import typing 
from warnings import warn
# Third party libs
import numpy as np
import torch
from torch import nn
from torch.distributions import *
import torchdata.datapipes as dp 
from torchdata.dataloader2.graph import DataPipe,traverse,replace_dp
from fastcore.all import test_eq,test_ne,add_docs,store_attr,ifnone,L
from torchdata.dataloader2.graph import find_dps,traverse
from ..data.dataloader2 import *
from torchdata.dataloader2 import DataLoader2,DataLoader2Iterator
from torchdata.dataloader2.graph import find_dps,traverse,DataPipe,IterDataPipe,MapDataPipe
# Local modules
from ..core import *
from ..pipes.core import *
from ..torch_core import *
from ..layers import *
from ..data.block import *
from ..envs.gym import *
from .ddpg import LossCollector,BasicOptStepper,StepBatcher
from ..loggers.core import LogCollector
from .discrete import EpsilonCollector
from copy import deepcopy
from torch.optim import AdamW,Adam
from ..learner.core import LearnerBase,LearnerHead
from ..loggers.core import LoggerBasePassThrough,BatchCollector,EpocherCollector,RollingTerminatedRewardCollector,EpisodeCollector

from .ddpg import BasicOptStepper
from ..loggers.vscode_visualizers import VSCodeTransformBlock
from ..loggers.jupyter_visualizers import ProgressBarLogger
from ..layers import Critic
from .discrete import EpsilonCollector
from .core import AgentHead,StepFieldSelector,AgentBase 
from .ddpg import ActionClip,ActionUnbatcher,NumpyConverter,OrnsteinUhlenbeck,SimpleModelRunner
from ..loggers.core import LoggerBase,CacheLoggerBase
from ..dataloader2_ext import InputInjester
from ..core import *
from ..pipes.core import *
from ..pipes.iter.nskip import *
from ..pipes.iter.nstep import *
from ..pipes.iter.firstlast import *
from ..pipes.iter.transforms import *
from ..pipes.map.transforms import *
from ..data.block import *

# %% ../../nbs/07_Agents/02_Continuous/12t_agents.trpo.ipynb 7
class AdvantageStep(typing.NamedTuple):
    state:           torch.FloatTensor=torch.FloatTensor([0])
    action:          torch.FloatTensor=torch.FloatTensor([0])
    next_state:      torch.FloatTensor=torch.FloatTensor([0])
    terminated:      torch.BoolTensor=torch.BoolTensor([1])
    truncated:       torch.BoolTensor=torch.BoolTensor([1])
    reward:          torch.FloatTensor=torch.LongTensor([0])
    total_reward:    torch.FloatTensor=torch.FloatTensor([0])
    advantage:       torch.FloatTensor=torch.FloatTensor([0])
    next_advantage:  torch.FloatTensor=torch.FloatTensor([0])
    env_id:          torch.LongTensor=torch.LongTensor([0])
    proc_id:         torch.LongTensor=torch.LongTensor([0])
    step_n:          torch.LongTensor=torch.LongTensor([0])
    episode_n:       torch.LongTensor=torch.LongTensor([0])
    image:           torch.FloatTensor=torch.FloatTensor([0])
    
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

    def to(self,*args,**kwargs):
        return self.__class__(
            **{fld:getattr(self,fld).to(*args,**kwargs) for fld in self.__class__._fields}
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
AdvantageStep,
"""Represents a single step in an environment similar to `SimpleStep` however has
an addition field called `advantage`.""",
advantage="""Generally characterized as $A(s,a) = Q(s,a) - V(s)$""",
**{f:getattr(SimpleStep,f).__doc__ for f in SimpleStep._fields}
)

# %% ../../nbs/07_Agents/02_Continuous/12t_agents.trpo.ipynb 10
def pipe2device(pipe,device,debug=False):
    "Attempt to move an entire `pipe` and its pipeline to `device`"
    pipes = find_dps(traverse(pipe),dp.iter.IterDataPipe,include_subclasses=True)
    for pipe in pipes:
        if hasattr(pipe,'to'): 
            if debug: print(f'Moving {pipe} to {device}')
            pipe.to(device=device)

# %% ../../nbs/07_Agents/02_Continuous/12t_agents.trpo.ipynb 11
@torch.jit.script
def discounted_cumsum_(t:torch.Tensor,gamma:float,reverse:bool=False):
    """Performs a cumulative sum on `t` where `gamma` is applied for each index
    >1."""
    if reverse:
        # We do +2 because +1 is needed to avoid out of index t[idx], and +2 is needed
        # to avoid out of index for t[idx+1].
        for idx in range(t.size(0)-2,-1,-1):
            t[idx] = t[idx] + t[idx+1] * gamma
    else:
        for idx in range(1,t.size(0)):
            t[idx] = t[idx] + t[idx-1] * gamma

# %% ../../nbs/07_Agents/02_Continuous/12t_agents.trpo.ipynb 12
def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size

# %% ../../nbs/07_Agents/02_Continuous/12t_agents.trpo.ipynb 14
class AdvantageBuffer(dp.iter.IterDataPipe):
    debug=False
    def __init__(self,
            # A datapipe that produces `StepType`s.
            source_datapipe:DataPipe,
            # A model that takes in a `state` and outputs a single value 
            # representing $V$, where as $Q$ is $V + reward$
            critic:nn.Module,
            # Will accumulate up to `bs` or when the episode has terminated.
            bs=1000,
            # The discount factor, otherwise known as $\gamma$, is defined in 
            # (Shulman et al., 2016) as '... $\gamma$ introduces bias into
            # the policy gradient estimate...'.
            discount:float=0.99,
            # $\lambda$ is unqiue to GAE and manages importance to values when 
            # they are in accurate is defined in (Shulman et al., 2016) as '... $\lambda$ < 1
            # introduces bias only when the value function is inaccurate....'.
            gamma:float=0.99
        ):
        self.source_datapipe = source_datapipe
        self.bs = bs
        self.critic = critic
        self.device = None
        self.discount = discount
        self.gamma = gamma
        self.env_advantage_buffer:Dict[Literal['env'],list] = {}

    def to(self,*args,**kwargs):
        self.device = kwargs.get('device',None)

    def __repr__(self):
        return str({k:v if k!='env_advantage_buffer' else f'{len(self)} elements' 
                    for k,v in self.__dict__.items()})

    def __len__(self): return self._sz_tracker

    def update_advantage_buffer(self,step:StepType) -> int:
        if self.debug: 
            print('Adding to advantage buffer: ',step)
        env_id = int(step.env_id.detach().cpu())
        if env_id not in self.env_advantage_buffer: 
            self.env_advantage_buffer[env_id] = []
        self.env_advantage_buffer[env_id].append(step)
        return env_id
        
    def zip_steps(
            self,
            steps:List[StepType]
        ) -> Tuple[torch.FloatTensor,torch.FloatTensor,torch.BoolTensor]:
            step_subset = [(o.reward,o.state,o.truncated or o.terminated) for o in steps]
            zipped_fields = zip(*step_subset)
            return L(zipped_fields).map(torch.vstack)

    def delta_calc(self,reward,v,v_next,done):
        return reward + (self.gamma * v * done) - v_next

    def __iter__(self) -> AdvantageStep:
        self.env_advantage_buffer:Dict[Literal['env'],list] = {}
        for step in self.source_datapipe:
            env_id = self.update_advantage_buffer(step)
            done = step.truncated or step.terminated
            if done or len(self.env_advantage_buffer[env_id])>self.bs:
                steps = self.env_advantage_buffer[env_id]
                rewards,states,dones = self.zip_steps(steps)
                # We vstack the final next_state so we have a complete picture
                # of the state transitions and matching reward/done shapes.
                values = self.critic(torch.vstack((states,steps[-1].next_state)))
                delta = self.delta_calc(rewards,values[:-1],values[1:],dones)
                discounted_cumsum_(delta,self.discount*self.gamma,reverse=True)

                for _step,gae_advantage,v in zip(*(steps,delta,values)):
                    yield AdvantageStep(
                        advantage=gae_advantage,
                        next_advantage=gae_advantage+v,
                        **{f:getattr(_step,f) for f in _step._fields}
                    )
                self.env_advantage_buffer[env_id].clear()

    @classmethod
    def insert_dp(cls,critic,old_dp=GymStepper) -> Callable[[DataPipe],DataPipe]:
        def _insert_dp(pipe):
            v = replace_dp(
                traverse(pipe,only_datapipe=True),
                find_dp(traverse(pipe,only_datapipe=True),old_dp),
                cls(find_dp(traverse(pipe,only_datapipe=True),old_dp),critic=critic)
            )
            return list(v.values())[0][0]
        return _insert_dp

add_docs(
AdvantageBuffer,
"""Collects an entire episode, calculates the advantage for each step, then
yields that episode's `AdvantageStep`s.

This is described in the original paper `(Shulman et al., 2016) High-Dimensional 
Continuous Control Usin Generalized Advantage Estimation`.

This algorithm is based on the concept of advantage:

$A_{\pi}(s,a) = Q_{\pi}(s,a) - V_{\pi}(s)$

Where (Shulman et al., 2016) pg 5 calculates it as:

$\hat{A}_{t}^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}^V$

Where (Shulman et al., 2016) pg 4 defines $\delta$ as:

$\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_{t})$
""",
to=torch.Tensor.to.__doc__,
update_advantage_buffer="Adds `step` to `env_advantage_buffer` based on the environment id.",
zip_steps="""Given `steps`, strip out the `Tuple[reward,state,truncated or terminated]` fields,
and `torch.vstack` them.""",
delta_calc="""Calculates $\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_{t})$ which 
is the advantage difference between state transitions."""
)

# %% ../../nbs/07_Agents/02_Continuous/12t_agents.trpo.ipynb 18
class OptionalClampLinear(Module):
    def __init__(self,num_inputs,state_dims,fix_variance:bool=False,
                 clip_min=0.3,clip_max=10.0):
        "Linear layer or constant block used for std."
        store_attr()
        if not self.fix_variance: 
            self.fc=nn.Linear(self.num_inputs,self.state_dims)
    
    def forward(self,x):
        if self.fix_variance: 
            return torch.full((x.shape[0],self.state_dims),1.0)
        else:                 
            return torch.clamp(nn.Softplus()(self.fc(x)),self.clip_min,self.clip_max)

# TODO(josiahls): This is probably a highly generic SimpleGMM tbh. Once we know this
# works, we should just rename this to SimpleGMM
class Actor(Module):
    def __init__(            
            self,
            state_sz:int,   # The input dim of the state / flattened conv output
            action_sz:int,  # The output dim of the actions
            hidden:int=400, # Number of neurons connected between the 2 input/output layers
            fix_variance:bool=False,
            clip_min=0.3,
            clip_max=10.0
        ):
        "Single-component GMM parameterized by a fully connected layer with optional std layer."
        store_attr()
        self.mu = nn.Sequential(
            nn.Linear(state_sz, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_sz),
            nn.Tanh(),
        )
        # self.std = OptionalClampLinear(state_sz,action_sz,fix_variance,
        #                                clip_min=clip_min,clip_max=clip_max)
        # self.std = nn.Linear(state_sz,action_sz)
        # self.std.weight.data.fill_(0.5)
        # self.std.bias.data.fill_(0.5)
        self.std = nn.Parameter(torch.zeros(action_sz)+.5)
        
    def forward(self,x): return Independent(Normal(self.mu(x),self.std),1)


add_docs(
Actor,
"""Produces continuous outputs from mean of a Gaussian distribution.""",
forward="Mean outputs from a parameterized Gaussian distribution."
)

# %% ../../nbs/07_Agents/02_Continuous/12t_agents.trpo.ipynb 22
class NormalExploration(dp.iter.IterDataPipe):
    def __init__(
                self,
                source_datapipe:DataPipe,
                # Based on the `base_agent.model.training`, by default no decrement or step tracking will
                # occur during validation steps.
                decrement_on_val:bool=False,
                # Based on the `base_agent.model.training`, by default random actions will not be attempted
                explore_on_val:bool=False,
                # Also return the original action prior to exploratory noise
                ret_original:bool=False,
        ):
                self.source_datapipe = source_datapipe
                self.decrement_on_val = decrement_on_val
                self.explore_on_val = explore_on_val
                self.ret_original = ret_original
                self.agent_base = None
                self.agent_base = find_dp(traverse(self.source_datapipe),AgentBase)
                self.model = self.agent_base.model
                self.last_std = None 
                self.last_mean = None

    def __iter__(self):
        for action in self.source_datapipe:
                if not issubclass(action.__class__,Independent):
                        raise Exception(f'Expected Independent, got {type(action)}\n{action}')

                # Add a batch dim if missing
                if len(action.batch_shape)==0: action = action.expand((1,))

                self.last_mean = action.mean
                self.last_std = action.stddev
                if self.explore_on_val or self.agent_base.model.training:
                        if self.ret_original: yield (action.sample(),action.mean)
                        else:                 yield action.sample()
                else:
                        yield action.mean

# %% ../../nbs/07_Agents/02_Continuous/12t_agents.trpo.ipynb 24
#|export


# %% ../../nbs/07_Agents/02_Continuous/12t_agents.trpo.ipynb 25
class AdvantageGymTransformBlock():

    def __init__(self,
        agent:DataPipe, # An AgentHead
        critic:Critic,
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
        # The max steps for the advatage buffer to run an environment
        max_steps:int=200,
        discount=0.99,
        gamma:float=0.99,
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
        terminate_on_truncation:bool=True,
        # Additional pipelines to insert, replace, remove
        dp_augmentation_fns:Tuple[DataPipeAugmentationFn]=None
    ) -> None:
        "Basic OpenAi gym `DataPipeGraph` with first-last, nstep, and nskip capability"
        self.agent = agent
        store_attr()

    def __call__(
        self,
        # `source` likely will be an iterable that gets pushed into the pipeline when an 
        # experiment is actually being run.
        source:Any,
        # Any parameters needed for the dataloader
        num_workers:int=0,
        # This param must exist: as_dataloader for the datablock to create dataloaders
        as_dataloader:bool=False
    ) -> DataPipeOrDataLoader:
        _type_tfms = ifnone(self.type_tfms,GymTypeTransform)
        "This is the function that is actually run by `DataBlock`"
        pipe = dp.map.Mapper(source)
        pipe = TypeTransformer(pipe,_type_tfms)
        pipe = dp.iter.MapToIterConverter(pipe)
        pipe = dp.iter.InMemoryCacheHolder(pipe)
        pipe = pipe.cycle() # Cycle through the envs inf
        pipe = GymStepper(pipe,agent=self.agent,seed=self.seed,
                          include_images=self.include_images,
                          terminate_on_truncation=self.terminate_on_truncation,
                          synchronized_reset=self.synchronized_reset)
        if self.nskips!=1: pipe = NSkipper(pipe,n=self.nskips)
        if self.nsteps!=1:
            pipe = NStepper(pipe,n=self.nsteps)
            if self.firstlast:
                pipe = FirstLastMerger(pipe)
            else:
                pipe = NStepFlattener(pipe) # We dont want to flatten if using FirstLastMerger
        pipe = AdvantageBuffer(pipe,critic=self.critic,bs=self.max_steps,
                               discount=self.discount,gamma=self.gamma)
        if self.n is not None: pipe = pipe.header(limit=self.n)
        pipe = ItemTransformer(pipe,self.item_tfms)
        pipe = pipe.batch(batch_size=self.bs)
        pipe = BatchTransformer(pipe,self.batch_tfms)
        
        pipe = apply_dp_augmentation_fns(pipe,ifnone(self.dp_augmentation_fns,()))
        
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

# %% ../../nbs/07_Agents/02_Continuous/12t_agents.trpo.ipynb 29
class ProbabilisticStdCollector(LogCollector):
    header:str='std'
    def __init__(self,
         source_datapipe, # The parent datapipe, likely the one to collect metrics from
        ):
        self.source_datapipe = source_datapipe
        self.record_pipe = find_dp(traverse(self.source_datapipe),NormalExploration)
        self.main_buffers = None

    def __iter__(self):
        # for q in self.main_buffers: q.append(Record('epsilon',None))
        for action in self.source_datapipe:
            for q in self.main_buffers: 
                q.append(Record('std',self.record_pipe.last_std.item()))
            yield action

class ProbabilisticMeanCollector(LogCollector):
    header:str='mean'
    def __init__(self,
         source_datapipe, # The parent datapipe, likely the one to collect metrics from
        ):
        self.source_datapipe = source_datapipe
        self.record_pipe = find_dp(traverse(self.source_datapipe),NormalExploration)
        self.main_buffers = None

    def __iter__(self):
        # for q in self.main_buffers: q.append(Record('epsilon',None))
        for action in self.source_datapipe:
            for q in self.main_buffers: 
                q.append(Record('mean',self.record_pipe.last_mean.item()))
            yield action

# %% ../../nbs/07_Agents/02_Continuous/12t_agents.trpo.ipynb 30
def TRPOAgent(
    model:Actor, # The actor to use for mapping states to actions
    # LoggerBases push logs to. If None, logs will be collected and output
    # by the dataloader.
    logger_bases:Optional[LoggerBase]=None, 
    clip_min=-1,
    clip_max=1,
    # Any augmentations to the DDPG agent.
    dp_augmentation_fns:Optional[List[DataPipeAugmentationFn]]=None
)->AgentHead:
    "Produces continuous action outputs."
    agent_base = AgentBase(model,logger_bases=ifnone(logger_bases,[CacheLoggerBase()]))
    agent = StepFieldSelector(agent_base,field='state')
    agent = InputInjester(agent)
    agent = SimpleModelRunner(agent)
    agent = NormalExploration(agent)
    # agent = ProbabilisticStdCollector(agent)
    # agent = ProbabilisticMeanCollector(agent)
    agent = ActionClip(agent,clip_min=clip_min,clip_max=clip_max)
    agent = ActionUnbatcher(agent)
    agent = NumpyConverter(agent)
    agent = AgentHead(agent)
    
    agent = apply_dp_augmentation_fns(agent,dp_augmentation_fns)

    return agent

# %% ../../nbs/07_Agents/02_Continuous/12t_agents.trpo.ipynb 38
def conjugate_gradients(
    # A function that takes the direction `d` and applies it to `A`.
    # The simplest example of this found would be:
    # `lambda d:A@d`
    Ad_f:Callable[[torch.Tensor],torch.Tensor],  
    # The bias or in TRPO's case the loss.
    b:torch.Tensor, 
    # Number of steps to go for assuming we are not less than `residual_tol`.
    nsteps:int, 
    # If the residual is less than this, then we have arrived at the local minimum.
    # Note that (Shewchuk, 1994) they mention that this should be E^2 * rdotr_0
    residual_tol=1e-10, 
    device="cpu"
):
    # The final direction to go in.
    x = torch.zeros(b.size()).to(device)
    # Would typically be b - Ax, however in TRPO's case this has already been 
    # done in the loss function.
    r = b.clone()
    # The first direction is the first residual.
    d = b.clone()
    rdotr = r.T @ r # \sigma_{new} pg50
    for i in range(nsteps):
        _Ad = Ad_f(d) # _Ad is also considered `q`
        # Determines the size / rate / step size of the direction
        alpha = rdotr / (d.T @ _Ad)

        x += alpha * d
        # [Shewchuk, 1994](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf) pg 49:
        #
        # The fast recursive formula for the residual is usually used, but once every 50 iterations, the exact residual
        # is recalculated to remove accumulated floating point error. Of course, the number 50 is arbitrary; for large
        # n \sqrt{n}, ©
        # might be appropriate.
        #
        # @josiah: This is kind of weird since we are using `Ad_f`. Maybe we can
        # have an optional param for A direction to do the residual reset?
        #
        # if nsteps > 50 and i % int(torch.sqrt(i)) == 0:
        #     r = b - Ax
        # else:
        r -= alpha * _Ad
        new_rdotr = r.T @ r
        beta = new_rdotr / rdotr
        d = r + beta * d
        rdotr = new_rdotr
        # Same as \sigma_{new} < E^2\sigma
        if rdotr < residual_tol:
            break
    return x

add_docs(
conjugate_gradients,
"""Conjugating Gradients builds on the idea of Conjugate Directions.

As noted in:
[Shewchuk, 1994](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)

We want "everytime we take a step, we got it right the first time" pg 21. 

In otherwords, we have a model, and we have the gradients and the loss. Using the 
loss, what is the the smartest way to change/optimize the gradients?

`Conjugation` is the act of makeing the `parameter space / gradient space` easier to 
optimize over. In technical terms, we find `nsteps` directions to change the gradients
toward that are orthogonal to each other and to the `parameter space / gradient space`.

In otherwords, what is the direction that is most optimal, and what is the 
direction that if used to find `x` will reduce `Ax - b` to 0. 
"""
)

# %% ../../nbs/07_Agents/02_Continuous/12t_agents.trpo.ipynb 40
def backtrack_line_search(
    # A Tensor of gradients or weights to optimize
    x:torch.Tensor,
    # The residual that when applied to `x`, hopefully optimizes it closer to the 
    # solution/ i.e. is orthogonal.
    r:torch.Tensor,
    # An error function that outputs the new error given the `x_new, where
    # `x_new` is passed as a param, and the error is returned as a float.
    # This error is compared, and expected greater than 0.
    error_f:Callable[[torch.Tensor],float],
    # The region of improvement we expect the see.
    expected_improvement_rate:torch.Tensor,
    # The minimal amount of improvement we expect to see.
    accaptance_tolerance:float=0.1,
    # The number of increments to attempt to improve `x`. 
    # Each "backtrack", the step size on the weights will be larger.
    n_max_backtracks:int=10
):
    e = error_f(x)
    # print("fval before", e.item())
    for (n_back,alpha) in enumerate(.5**torch.arange(0,n_max_backtracks)):
        x_new = x + alpha * r 
        e_new = error_f(x_new)
        improvement = e - e_new
        expected_improvement = expected_improvement_rate * alpha 
        ratio = improvement / expected_improvement
        if ratio.item() > accaptance_tolerance and improvement.item() > 0:
            # print("fval after", e_new.item(),' on ',n_back)
            return True, x_new
    return False, x

add_docs(
backtrack_line_search,
"""Backtrack line search attempts an update to a set of weights/gradients `x` `n_max_backtracks` times.

Each backtrack updates the weights/gradients a little more aggressively, and checks if `error_f`
decreases / improves. 
"""
)

# %% ../../nbs/07_Agents/02_Continuous/12t_agents.trpo.ipynb 42
def actor_prob_loss(weights,s,a,r,actor,old_log_prob):
    if weights is not None:
        set_flat_params_to(actor,weights)
    dist = actor(s)
    log_prob = dist.log_prob(a)
    # loss = -r * torch.exp(log_prob-old_log_prob) 
    loss = -r.squeeze(1) * torch.exp(log_prob-old_log_prob) 
    return loss.mean()

# %% ../../nbs/07_Agents/02_Continuous/12t_agents.trpo.ipynb 44
def pre_hessian_kl(
    model:Actor, # An Actor or any model that outputs a probability distribution
    x:torch.Tensor # Input into the model
):
    r"""
    Provides a KL conculation for the 2nd dirivative hessian to be calculated later.

    It is important to note that this function will return a tensor of 0, however
    the goal is to do autograd as opposed to doing anything with the value directly.

    The "confusing" part of the code can be found in [4]:

        "For two univariate normal distributions p and q the above simplifies to:"

    $D_{\text{KL}}\left({\mathcal {p}}\parallel {\mathcal {q}}\right)=\log {\frac {\sigma _{2}}{\sigma _{1}}}+{\frac {\sigma _{1}^{2}+(\mu _{1}-\mu _{2})^{2}}{2\sigma _{2}^{2}}}-{\frac {1}{2}}$

    Notes:
    - [1] https://github.com/ikostrikov/pytorch-trpo/issues/2
    - [2] [(Schulman et al., 2015) [TRPO] Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
    
    Appendix C:
    
        One could alternatively use a generic method for calculating Hessian-vector products using 
        reverse mode automatic differentiation ((Wright & Nocedal, 1999), chapter 8), computing the 
        Hessian of DKL with respect to θ. This method would be slightly less efficient as it does 
        not exploit the fact that the second derivatives of μ(x) (i.e., the second term in Equation (57))
        can be ignored, but may be substantially easier to implement.

    - [3] http://rail.eecs.berkeley.edu/deeprlcoursesp17/docs/lec5.pdf
    - [4] https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#cite_note-27
    """
    dist = model(x)
    mu_v = dist.mean
    logstd_v = torch.log(dist.stddev)
    mu0_v = mu_v.detach()
    logstd0_v = logstd_v.detach()
    std_v = torch.exp(logstd_v)
    std0_v = std_v.detach()
    kl = logstd_v - logstd0_v + (std0_v ** 2 + (mu0_v - mu_v) ** 2) / (2.0 * std_v ** 2) - 0.5
    return kl.sum(1, keepdim=True)

# %% ../../nbs/07_Agents/02_Continuous/12t_agents.trpo.ipynb 46
def auto_flat(outputs,inputs,contiguous=False,create_graph=False)->torch.Tensor:
    "Calculates the gradients and flattens them into a single tensor"
    grads = torch.autograd.grad(outputs,inputs,create_graph=create_graph)
    # TODO: Does it always need to be contiguous?
    if contiguous:
        return torch.cat([grad.contiguous().view(-1) for grad in grads])
    else:
        return torch.cat([grad.view(-1) for grad in grads])

def forward_pass(
        weights:torch.Tensor,
        s:torch.Tensor,
        actor:Actor,
        damping:float=0.1
    ):
    kl = pre_hessian_kl(actor,s)
    kl = kl.mean()

    # Calculate the 1st derivative hessian
    flat_grad_kl = auto_flat(kl,actor.parameters(),create_graph=True)

    kl_v = (flat_grad_kl * weights.detach()).sum()
    # Calculate the 2nd derivative hessian
    flat_grad_grad_kl = auto_flat(kl_v,actor.parameters(),contiguous=True).data

    return flat_grad_grad_kl + weights * damping

# %% ../../nbs/07_Agents/02_Continuous/12t_agents.trpo.ipynb 50
class CriticLossProcessor(dp.iter.IterDataPipe):
    debug:bool=False

    def __init__(self,
            source_datapipe:DataPipe, # The parent datapipe that should yield step types
            critic:Critic, # The critic to optimize
            # The loss function to use
            loss:nn.Module=nn.MSELoss,
            # The discount factor of `q`. Typically does not need to be changed,
            # and determines the importants of earlier state qs verses later state qs
            discount:float=0.99,
            # If the environment has `nsteps>1`, it is recommended to change this
            # param to reflect that so the reward estimates are more accurate.
            nsteps:int=1
        ):
        self.source_datapipe = source_datapipe
        self.critic = critic
        self.loss = loss()
        self.discount = discount
        self.nsteps = nsteps
        self.device = None

    def to(self,*args,**kwargs):
        self.critic.to(**kwargs)
        self.device = kwargs.get('device',None)

    def __iter__(self) -> Union[Dict[Literal['loss'],torch.Tensor],SimpleStep]:
        for batch in self.source_datapipe:
            # Slow needs better strategy
            with torch.no_grad():
                batch = batch.clone()

                batch.to(self.device)

                # traj_adv_v = (batch.advantage - torch.mean(batch.advantage)) / torch.std(batch.advantage)
            m = batch.terminated.reshape(-1,)==False
            self.critic.zero_grad()
            pred = self.critic(batch.state[m])
            yield {'loss':self.loss(pred,batch.next_advantage[m])}
            yield batch

# %% ../../nbs/07_Agents/02_Continuous/12t_agents.trpo.ipynb 51
class ActorOptAndLossProcessor(dp.iter.IterDataPipe):
    debug:bool=False

    def __init__(self,
            source_datapipe:DataPipe, # The parent datapipe that should yield step types
            actor:Actor, # The actor to optimize
            max_kl:float=0.01
        ):
        self.source_datapipe = source_datapipe
        self.actor = actor
        self.device = None
        self.max_kl = max_kl
        self.counter = 0

    def to(self,*args,**kwargs):
        self.actor.to(**kwargs)
        self.device = kwargs.get('device',None)

    def __iter__(self) -> Union[Dict[Literal['loss'],torch.Tensor],SimpleStep]:
        for batch in self.source_datapipe:
            # Slow needs better strategy
            with torch.no_grad():
                batch = batch.clone()
                batch.to(self.device)
                traj_adv_v = (batch.advantage - torch.mean(batch.advantage)) / torch.std(batch.advantage)
            
            m = batch.terminated.reshape(-1,)==False
            dist = self.actor(batch.state[m])
            old_log_prob = dist.log_prob(batch.action[m]).detach()

            loss_fn = partial(
                actor_prob_loss,
                s=batch.state[m],
                a=batch.action[m],
                r=traj_adv_v[m],
                actor=self.actor,
                old_log_prob=old_log_prob
            )
            self.counter += 1
            # Calculate gradient backprop on initial loss function.
            # Since the `actor` has not been updated yet, then loss is 
            # basically just going to be the `-traj_adv_v.mean()`.
            loss = loss_fn(None)
            loss_grad = auto_flat(loss,self.actor.parameters()).data
            assert loss_grad.sum()!=0
 
            forward_pass_fn = partial(
                forward_pass,
                s=batch.state[m],
                actor=self.actor
            )
            # -loss_grad will be the `b` variable. Out goal is to find the gradient
            # update direction that gets the output of `forward_pass_fn` to
            # have an orthogonal step size to hit that loss_grad.
            # The step direction (d) is going to be constrained by the f``KLdiv.
            d = conjugate_gradients(forward_pass_fn,-loss_grad,10)

            shs = 0.5 * (d * forward_pass_fn(d)).sum(0,keepdim=True)
            lm = torch.sqrt(shs/self.max_kl)
            full_step = d/lm[0]
            neggdotstepdir = (-loss_grad * d).sum(0, keepdim=True)

            prev_params = get_flat_params_from(self.actor)
            success,params = backtrack_line_search(prev_params,full_step,loss_fn,neggdotstepdir/lm[0])
            if success:
                set_flat_params_to(self.actor,params)

            yield {'loss':loss}
            yield batch

# %% ../../nbs/07_Agents/02_Continuous/12t_agents.trpo.ipynb 52
def TRPOLearner(
    # The actor model to use
    actor:Actor,
    # The critic model to use
    critic:Critic,
    # A list of dls, where index=0 is the training dl.
    dls:List[DataPipeOrDataLoader],
    # Optional logger bases to log training/validation data to.
    logger_bases:Optional[List[LoggerBase]]=None,
    # The learning rate for the actor. Expected to learn slower than the critic
    actor_lr:float=1e-3,
    # The optimizer for the actor
    actor_opt:torch.optim.Optimizer=Adam,
    # The learning rate for the critic. Expected to learn faster than the actor
    critic_lr:float=1e-2,
    # The optimizer for the critic
    # Note that weight decay doesnt seem to be great for 
    # Pendulum, so we use regular Adam, which has the decay rate
    # set to 0. (Lillicrap et al., 2016) would instead use AdamW
    critic_opt:torch.optim.Optimizer=Adam,
    # Reference: GymStepper docs
    nsteps:int=1,
    # The device for the entire pipeline to use. Will move the agent, dls, 
    # and learner to that device.
    device:torch.device=None,
    # Number of batches per epoch
    batches:int=None,
    # Any augmentations to the learner
    dp_augmentation_fns:Optional[List[DataPipeAugmentationFn]]=None,
    # Debug mode will output device moves
    debug:bool=False
) -> LearnerHead:
    warn("TRPO only kind of converges. There is a likely a bug, however I am unable to identify until after PPO implimentation")

    learner = LearnerBase(actor,dls,batches=batches)
    learner = LoggerBasePassThrough(learner,logger_bases)
    learner = BatchCollector(learner,batch_on_pipe=LearnerBase)
    learner = EpocherCollector(learner)
    for logger_base in L(logger_bases): learner = logger_base.connect_source_datapipe(learner)
    if logger_bases: 
        learner = RollingTerminatedRewardCollector(learner)
        learner = EpisodeCollector(learner)
    learner = StepBatcher(learner)
    learner = CriticLossProcessor(learner,critic=critic)
    learner = LossCollector(learner,header='critic-loss')
    learner = BasicOptStepper(learner,critic,critic_lr,opt=critic_opt,filter=True,do_zero_grad=False)
    learner = ActorOptAndLossProcessor(learner,actor)
    learner = LossCollector(learner,header='actor-loss',filter=True)
    learner = LearnerHead(learner)
    
    learner = apply_dp_augmentation_fns(learner,dp_augmentation_fns)
    pipe2device(learner,device,debug=debug)
    for dl in dls: pipe2device(dl.datapipe,device,debug=debug)
    
    return learner

TRPOLearner.__doc__=""""""
