# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02b_fastai.data.pipes.core.ipynb (unless otherwise specified).

__all__ = ['Callback', 'filter_call_on_cbs', 'filter_exclude_under_cbs', 'find_pipes', 'after_pipes',
           'add_hooks_before', 'PassThroughIterPipe', 'add_cbs_to_pipes']

# Cell
# Python native modules
import os
import logging
import inspect
# Third party libs
from fastcore.all import *
import torchdata.datapipes as dp
from torch.utils.data.graph import traverse
from torchdata.datapipes import functional_datapipe
# Local modules


_logger = logging.getLogger()

# Cell
_allowed_hook_params = ['before','after','not_under']

class Callback():
    @property
    def name(self):
        "Name of the `Callback`, camel-cased and with '*Callback*' removed"
        return class2attr(self, 'Callback')

    def hooks(self):
        if inspect.isclass(self): raise ValueError(f'{self} needs to be instantiated!')
        hooks = []
        def in_allowed_hooks(param): return param in _allowed_hook_params
        for k in self.__class__.__dict__:
            if k.startswith('_'): continue
            params = L(inspect.signature(getattr(self,k)).parameters).map(in_allowed_hooks)
            if params and all(params): hooks.append(getattr(self,k))
        return hooks

# Cell
def filter_call_on_cbs(obj, cbs): return tuple(cb for cb in cbs if obj.__class__ in cb.call_on)

# Cell
def filter_exclude_under_cbs(
    pipe:Union[dp.map.MapDataPipe,dp.iter.IterDataPipe],
    cbs:List[Callback]
):
    cbs = tuple(cb for cb in cbs if pipe.__class__  not in cb.exclude_under)
    for v in traverse(pipe,only_datapipe=True).values(): # We dont want to traverse non-dp objects.
        for k,_ in v.items():
            cbs = filter_exclude_under_cbs(k,cbs)
    return cbs

# Cell
def find_pipes(
    pipe:Union[dp.map.MapDataPipe,dp.iter.IterDataPipe],
    fn,
    pipe_list=None
):
    pipe_list = ifnone(pipe_list,[])
    if issubclass(pipe.__class__,(dp.map.MapDataPipe,dp.iter.IterDataPipe)) and fn(pipe): pipe_list.append(pipe)
    for v in traverse(pipe,only_datapipe=True).values(): # We dont want to traverse non-dp objects.
        for k,_ in v.items(): cbs = find_pipes(k,fn,pipe_list)
    return pipe_list

# Cell
for _pipe in [dp.map.MapDataPipe,dp.iter.IterDataPipe]:
    _pipe.callbacks = L()

    @patch
    def __repr__(self:_pipe):
        if self.repr_hook is not None:
            return self.repr_hook(self)
        # Instead of showing <torch. ... .MapperMapDataPipe object at 0x.....>, return the class name
        str_rep = str(self.__class__.__qualname__)
        if self.callbacks: return str_rep + str(self.callbacks)
        return str_rep

    @patch
    def __str__(self:_pipe):
        if self.str_hook is not None:
            return self.str_hook(self)
        # Instead of showing <torch. ... .MapperMapDataPipe object at 0x.....>, return the class name
        str_rep = str(self.__class__.__qualname__)
        if self.callbacks: return str_rep + str(self.callbacks)
        return str_rep

    @patch
    def add_cbs_before(self:_pipe,cbs):
        pipe = self
        if cbs is None or len(cbs)==0: return pipe

        for cb in cbs:
            for hook in cb.hooks():
                pipe = add_hooks_before(pipe,hook,base_pipe=self)
        if pipe.__class__==PassThroughIterPipe: return pipe.source_datapipe
        return pipe

    @patch
    def add_cbs_after(self:_pipe,cbs):
        pipe = self
        if cbs is None or len(cbs)==0: return pipe
        after_pipe,fld = after_pipes(pipe)

        for cb in cbs:
            for hook in cb.hooks():
                # In this instance, we want to add the hook if the event is `after_pipe`
                # So if after_pipe->pipe,
                # we add `hook` before `pipe` which ends up also being after `after_pipe`
                # So the result is: `after_pipe->hook_results->pipe`
                pipe = add_hooks_before(pipe,hook,base_pipe=after_pipe,event_key='after')
        if pipe.__class__==PassThroughIterPipe: return pipe.source_datapipe
        return pipe

# Cell
def after_pipes(dp):
    if hasattr(dp,'iterable'):          return dp.iterable,'iterable'
    elif hasattr(dp,'datapipe'):        return dp.datapipe,'datapipe'
    elif hasattr(dp,'source_datapipe'): return dp.source_datapipe,'source_datapipe'
    elif hasattr(dp,'main_datapipe'):   return dp.main_datapipe,'main_datapipe'
    elif hasattr(dp,'datapipes'):       return dp.datapipes,'datapipes'
    else:                               return None,None


# Cell
_supported_pipe_attrs = ['iterable','datapipe','source_datapipe','main_datapipe','datapipes']

def add_hooks_before(dp,cb_hook,base_pipe=None,event_key='before'):
    "Given `dp`, attach a `cb_hook` before or after it. It will not be attached if there is a `not_under` farthur up the pipeline."
    events = {k:v.default for k,v in inspect.signature(cb_hook).parameters.items()}

    if events['not_under'] is not None:
        for not_under_pipe in L(events['not_under']):
            if not find_pipes(dp,lambda o:o is not_under_pipe):
                return dp
    if events[event_key] is not None:
        for pipe in L(events[event_key]):
            if pipe==base_pipe.__class__:
                for cb_dp in cb_hook():
                    if hasattr(dp,'iterable'):
                        cb_dp = cb_dp(dp.iterable)
                        dp.iterable = cb_dp
                    elif hasattr(dp,'datapipe'):
                        cb_dp = cb_dp(dp.datapipe)
                        dp.datapipe = cb_dp
                    elif hasattr(dp,'source_datapipe'):
                        cb_dp = cb_dp(dp.source_datapipe)
                        dp.source_datapipe = cb_dp
                    elif hasattr(dp,'main_datapipe'):
                        cb_dp = cb_dp(dp.main_datapipe)
                        dp.main_datapipe = cb_dp
                    elif hasattr(dp,'datapipes'):
                        dp.datapipes = tuple(cb_dp(_dp) for _dp in dp.datapipes)
                    else:
                        raise ValueError(f'Given {cb_hook}, tried adding {cb_dp} to {after_pipe}:{dp}:base:{base_pipe} \
                            but doesnt have any of the expected attrs: {_supported_pipe_attrs}')
    return dp


# Cell
class PassThroughIterPipe(dp.iter.IterDataPipe):
    def __init__(self,source_datapipe): self.source_datapipe = source_datapipe
    def __iter__(self): return (o for o in self.source_datapipe)

# Cell
def add_cbs_to_pipes(pipe,cbs):
    for _pipe in reversed(find_pipes(PassThroughIterPipe(pipe),lambda o:True)): pipe = _pipe.add_cbs_after(cbs)
    for _pipe in reversed(find_pipes(PassThroughIterPipe(pipe),lambda o:True)): pipe = _pipe.add_cbs_before(cbs)
    return pipe