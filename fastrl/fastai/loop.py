# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02a_fastai.loop.ipynb (unless otherwise specified).

__all__ = ['IN_IPYTHON', 'map_obj_attr2func_attr', 'isevent', 'EventException', 'grab_parent_event', 'KwargSetAttr',
           'Events', 'Event', 'EVENT_ORDER_MAPPING', 'PREFIXES', 'event', 'Loops', 'Loop', 'events', 'run_section',
           'eq_loops', 'connect_loops2loop', 'dict2loops', 'dict2events', 'CallbackException', 'Callback',
           'map_obj_attr2func_attr', 'ipy_handle_exception']

# Cell
# Python native modules
import os,sys,json
from copy import deepcopy,copy
from typing import *
import types
import logging
import inspect
from itertools import chain,product
from functools import partial
# Third party libs
from fastcore.all import *
import numpy as np
# Local modules
from ..core import test_in

IN_IPYTHON=False

_logger=logging.getLogger(__name__)

# Cell
def map_obj_attr2func_attr(obj,fn):
    got_attrs={}
    for k,v in inspect.signature(fn).parameters.items():
        if k=='self':continue
        elif v.default==inspect._empty:
            got_attrs[k]=getattr(obj,k)
        else:
            got_attrs[k]=getattr(obj,k,v.default)
    return got_attrs

# Cell
EVENT_ORDER_MAPPING={}
PREFIXES=['before_','on_','after_','failed_','finally_']

def isevent(o): return issubclass(o.__class__,Event)
class EventException(Exception):pass
def _default_raise(_placeholder): raise
def grab_parent_event(o): return o.parent_event

class KwargSetAttr(object):
    def __setattr__(self,name,value):
        "Allow setting attrs via kwarg."
        super().__setattr__(name,value)

class Events(KwargSetAttr,L):
    def __init__(self,items=None,postfix=None,prefix=None,item_iter_hint='prefix',
                 order=0,parent_event=None,*args,**kwargs):
        store_attr(but='items')
        super().__init__(items=items,*args,**kwargs)

    def flat(self):
        return Events(chain.from_iterable(self),
                      postfix=self.postfix,prefix=self.prefix,
                      item_iter_hint=self.item_iter_hint,order=self.order,
                      parent_event=self.parent_event)

    def __lt__(self,o:'Event'): return self.order<o.order
    def todict(self):
        return {getattr(o,self.item_iter_hint):o for o in self}

    def __repr__(self):
        if len(self)==0: return super().__repr__()
        return '['+'\n'.join([str(o) for o in self])+']'
    def run(self):
        for o in self: o.run()                                                  # fastrl.skip_traceback

class Event(KwargSetAttr):
    def __init__(self,
                 function:Callable,
                 loop=None,
                 override_name=None,
                 override_qualname=None,
                 override_module=None,
                 order=None
                ):
        store_attr()
        if self.function==noop and self.prefix=='failed_':
            self.function=_default_raise
        # We set the order over the entire Loop definition
        if self.order is None:
            if self.outer_name not in EVENT_ORDER_MAPPING: self.order=1
            else: self.order=EVENT_ORDER_MAPPING[self.outer_name]
            EVENT_ORDER_MAPPING[self.outer_name]=self.order+1

        # self.original_name=self.function.__module__+'.'+self.function.__qualname__

        if self.name.startswith('_') or not any(self.name.startswith(pre) for pre in PREFIXES):
            raise EventException(f'{self.name} needs to start with any {PREFIXES}')

        self.cbs=L()

    def climb(self):
        "Returns a generator that moves up to the parent/root event"
        if self.loop is not None:
            yield from self.loop.climb()

    @property
    def level(self): return len(list(self.climb()))

    @classmethod
    def from_override_name(cls,name,**kwargs):
        return cls(noop,override_name=name,**kwargs)

    def init_cbs(self):
        "Look at the cbs in the `parent_loop` and add them to `self`"
        cbs=L(self.climb())[-1].cbs
        # parent_events=[self.name]+[o.parent_event.name for o in self.climb() if o.parent_event is not None]
        parent_events=[self.qualname]+L(self.climb())\
                                   .map(grab_parent_event)\
                                   .filter(ifnone,b=False)\
                                   .map(Self.qualname())
        # Check if the callback has an event relevent to self
        for cb in L(cbs):
            if hasattr(cb,self.name):
                if not cb.call_on or any(o.qualname in parent_events for o in cb.call_on):
                    self.cbs.append(cb)

    @property
    def root_loop(self): return list(self.climb())[-1]
    def __call__(self,*args,**kwargs):
        ret=self.function(self.loop,*args,**kwargs)                             # fastrl.skip_traceback
        for cb in self.cbs:
            fn=getattr(cb,self.name)
            params=map_obj_attr2func_attr(self.root_loop,fn)

            cb_ret=fn(**params)

            if isinstance(cb_ret,dict):
                loop=self.root_loop
                for k,v in cb_ret.items(): setattr(loop,k,v)

        return ret

    def __lt__(self,o:'Event'): return self.order<o.order
    @property
    def name(self): return ifnone(self.override_name,self.function.__name__)
    @property
    def module(self): return ifnone(self.override_module,self.function.__module__)
    @property
    def qualname(self): return ifnone(self.override_qualname,self.function.__qualname__)
    @property
    def prefix(self): return self.name.split('_')[0]+'_'
    @property
    def postfix(self): return '_'.join(self.name.split('_')[1:])
    @property
    def outer_name(self): return self.module+'.'+self.qualname.split('.')[0]
    @property
    def original_name(self):
        return self.function.__module__+'.'+self.function.__qualname__

    def __repr__(self): return self.module+'.'+self.name
    def with_inner(self):
        return (self,Events(postfix=self.postfix,
                            prefix=self.prefix+'inner',
                            order=self.order))

event=Event

# Cell
class Loops(L):
    def run(self):
        for o in self: o.run()                                                  # fastrl.skip_traceback

class Loop(object):
    def __init__(self,cbs:L=None,verbose:bool=False):
        store_attr()
        # When a loop is initialized, we need to make sure that the events
        # are re-initialized also
        events(self,reset=True)

        self.parent_loop=None
        self.parent_event=None

        _events=Events(inspect.getmembers(self)).map(Self[-1]).filter(isevent).sorted()
        # print(Events(inspect.getmembers(self)).map(Self[-1]))
        # 1. Make Events have the same module as the function being run
        # 2. Convert the Events to Events+Inner Events
        # 3. Convert [(Event,[]*inner events*)...] to [Event,[]*inner events*...]
        # 4. Sure they are sorted correctly
        self.default_events=Events(PREFIXES)\
            .map(Event.from_override_name,override_module=_events[0].module)\
            .map(Event.with_inner)\
            .flat()\
            .sorted()
        self.events=_events.sorted().map(Event.with_inner).flat().sorted()
        self.events.map(Event.__setattr__,name='loop',value=self)
        self.sections=groupby(self.events,Self.postfix())
        for k,v in self.sections.items():
            self.sections[k]=merge(self.default_events.map(copy).todict(),
                                   Events(v).todict())


    def copy(self):

        return self.__class__()


    def climb(self):
        "Returns a generator that moves up to the parent/root event"
        yield self
        if self.parent_loop is not None:
            yield from self.parent_loop.climb()

    def run(self):
        try:                                                                    # fastrl.skip_traceback
            for v in self.sections.values(): run_section(v)                     # fastrl.skip_traceback
        except Exception as e:
            e._show_loop_errors=self.verbose
            raise


# Cell
class _Events():
    def __call__(self,loop,reset=False):
        # Handle types/instances...
        if isinstance(loop,type): attrs=loop.__dict__.items()
        else:                     attrs=inspect.getmembers(loop)

        for k,v in attrs:
            if not callable(v): continue
            if any(k.startswith(s) for s in PREFIXES):
                if not isevent(v): setattr(loop,k,Event(v))
                if isevent(v) and reset: setattr(loop,k,Event(v.function))
        return loop

events=_Events()

# Cell
def run_section(section:Dict):
    try:
        section['before_']()
        section['before_inner'].run()                                           # fastrl.skip_traceback
        section['on_']()
        section['on_inner'].run()
        section['after_']()
        section['after_inner'].run()                                            # fastrl.skip_traceback
    except Exception as ex:
        try:
            section['failed_']()                                                # fastrl.skip_traceback
            raise
        finally:
            section['failed_inner'].run()                                       # fastrl.skip_traceback
    finally:
        section['finally_']()
        section['finally_inner'].run()                                          # fastrl.skip_traceback

# Cell
def eq_loops(a:Loop,b:Loop): return a.__class__==b.__class__

def connect_loops2loop(loops:Loops,to_loop):
    # Given `to_loop`, generate some fresh `loops`...
    loops=Loops(loops)
    loops=loops.map(Self.copy())
    to_events=to_loop.events.filter(isevent).map(Self.original_name())
    for from_loop in loops.filter(eq_loops,b=to_loop,negate=True):
        for call_on in from_loop.call_on:
            if call_on.original_name in to_events:
                _from_loop=from_loop.copy()

                _from_loop.parent_event=to_loop.sections[call_on.postfix][call_on.prefix]
                _from_loop.parent_loop=to_loop

                _from_loop.events.filter(isevent).map(Self.init_cbs())

                to_loop.sections[call_on.postfix][call_on.prefix+'inner'].extend([_from_loop])
                connect_loops2loop(loops,_from_loop)
    return to_loop

# Cell
def dict2loops(d):
    if isinstance(d,dict):
        for o in d.values():
            yield from dict2loops(o)
    elif isinstance(d,(Loops,Events)):
        for o in d:
            yield from dict2loops(o)
    elif issubclass(d.__class__,Loop):
        yield d
        yield from dict2loops(d.sections)

def dict2events(d):
    if isinstance(d,dict):
        for o in d.values():
            yield from dict2events(o)
    elif isinstance(d,(Loops,Events)):
        for o in d:
            yield from dict2events(o)
    elif issubclass(d.__class__,Loop):
        yield from dict2events(d.sections)
    elif issubclass(d.__class__,Event):
        yield d

# Cell
class CallbackException(Exception):pass

class Callback(object):
    call_on,loop=None,None

    @property
    def root(self): return self.loop.root_loop

# Cell
def map_obj_attr2func_attr(obj,fn):
    got_attrs={}
    for k,v in inspect.signature(fn).parameters.items():
        if k=='self':continue
        elif v.default==inspect._empty:
            got_attrs[k]=getattr(obj,k)
        else:
            got_attrs[k]=getattr(obj,k,v.default)
    return got_attrs

# Cell
def _skip_traceback(s):
    return in_('# fastrl.skip_traceback',s)

def ipy_handle_exception(self, etype, value, tb, tb_offset):
    ## Do something fancy
    stb = self.InteractiveTB.structured_traceback(etype,value,tb,tb_offset=tb_offset)
    if not getattr(value,'_show_loop_errors',True):
        tmp,idxs=[],L(stb).argwhere(_skip_traceback)
        prev_skipped_idx=idxs[0] if idxs else 0
        for i,s in enumerate(stb):
            if i in idxs and i-1!=prev_skipped_idx:
                msg='Skipped Loop Code due to # fastrl.skip_traceback found in source code,'
                msg+=' please use Loop(...verbose=True) to view loop tracebacks\n'
                tmp.append(msg)
            if i not in idxs:
                tmp.append(s)
            else:
                prev_skipped_idx=i
        stb=tmp
    ## Do something fancy
    self._showtraceback(type, value, stb)

if IN_IPYTHON:
    get_ipython().set_custom_exc((Exception,),ipy_handle_exception)
