# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02c_fastai.data.block.ipynb (unless otherwise specified).

__all__ = ['TransformBlock', 'CategoryBlock', 'DataBlock', 'Cacher', 'T_co', 'GrandparentSplitter']

# Cell
# Python native modules
import os
from inspect import isfunction,ismethod
from typing import *
# Third party libs
from fastcore.all import *
from fastai.torch_basics import *
# from torch.utils.data.dataloader import DataLoader as OrgDataLoader
import torchdata.datapipes as dp
from torch.utils.data.dataloader_experimental import DataLoader2
from fastai.data.transforms import *
# Local modules
from ..loop import *
from .load import *

# Cell
class TransformBlock():
    "A basic wrapper that links defaults transforms for the data block API"
    def __init__(self,
                 type_tfms:Transform=None, # Executed when the DataPipe is
                 # initialized / wif is run. Intended as a 1 time transform.
                 item_tfms:Transform=None, # Executed on individual elements.
                 batch_tfms:Transform=None, # Executed over a batch.
                 dl_type:MinimumDataLoader=None, # Its recommended not to set this,
                 # all custom behaviors should be done via callbacks.
                 dls_kwargs:dict=None
                ):
        self.type_tfms  =            L(type_tfms)
        self.item_tfms  = ToTensor + L(item_tfms)
        self.batch_tfms =            L(batch_tfms)
        self.dl_type,self.dls_kwargs = dl_type,ifnone(dls_kwargs,{})

# Cell
def CategoryBlock(vocab=None, sort=True, add_na=False):
    "`TransformBlock` for single-label categorical targets"
    return TransformBlock(type_tfms=Categorize(vocab=vocab, sort=sort, add_na=add_na))

# Cell
def _merge_grouper(o):
    if isinstance(o, LambdaType): return id(o)
    elif isinstance(o, type): return o
    elif (isfunction(o) or ismethod(o)): return o.__qualname__
    return o.__class__

def _merge_tfms(*tfms):
    "Group the `tfms` in a single list, removing duplicates (from the same class) and instantiating"
    g = groupby(concat(*tfms), _merge_grouper)
    return L(v[-1] for k,v in g.items()).map(instantiate)

def _zip(x): return L(x).zip()

# Cell
class DataBlock():
    "Generic container to quickly build `Datasets` and `DataLoaders`"
    _msg = """If you wanted to compose several transforms in your getter don't
    forget to wrap them in a `Pipeline`."""
    def __init__(self,
                 blocks=TransformBlock,
                 dl_type=MinimumDataLoader,
                 get_items=None,
                 type_tfms=None,
                 item_tfms=None,
                 batch_tfms=None,
                 bs=1,
                 splitter:Optional[Union[dp.iter.IterDataPipe,Callable]]=None,
                 # If a callable, it is
                 # assumed to split the datapipe into 2. If you want more than 2,
                 # create a custom dp.iter.IterDataPipe with `__len__` for the number of splits.
                 shuffle:bool=False,
                 mapped:bool=False
                ):
        blocks = L(self.blocks if blocks is None else blocks)
        blocks = L(b() if callable(b) else b for b in blocks)
        self.default_type_tfms  = _merge_tfms(*blocks.attrgot('type_tfms',  L()))
        self.default_item_tfms  = _merge_tfms(*blocks.attrgot('item_tfms',  L()))
        self.default_batch_tfms = _merge_tfms(*blocks.attrgot('batch_tfms', L()))
        for b in blocks:
            if getattr(b, 'dl_type', None) is not None: self.dl_type = b.dl_type
        if dl_type is not None: self.dl_type = dl_type
        self.get_items = get_items
        self.splitter = splitter
        self.shuffle = shuffle
        self.mapped = mapped
        self.bs = bs
        self.dls_kwargs = merge(*blocks.attrgot('dls_kwargs', {}))
        self.new(item_tfms, batch_tfms, type_tfms)

    def _combine_type_tfms(self): return L([self.getters, self.type_tfms]).map_zip(
        lambda g,tt: (g.fs if isinstance(g, Pipeline) else L(g)) + tt)

    def new(self, item_tfms=None, batch_tfms=None, type_tfms=None):
        self.type_tfms  = _merge_tfms(self.default_type_tfms,  type_tfms)
        self.item_tfms  = _merge_tfms(self.default_item_tfms,  item_tfms)
        self.batch_tfms = _merge_tfms(self.default_batch_tfms, batch_tfms)
        return self

    def datapipes(self,
                  source:Union[L,Callable] # Absolute initial items for create the `dp.iter.IterDataPipe`s from.
                  # These should be picklable/probably uninitialized.
                 )->List[dp.iter.IterDataPipe]:
        items = source() if source is callable else source
        items = ifnone(Pipeline(self.get_items),noop)(items)

#         if mapped:
#             dps = dp.map.SequenceWrapper(items)
#             if callable(self.splitter):     dps = dps.map(self.splitter)
#             elif self.splitter is not None: dps = self.splitter(dps)

#             # Regardless of the splitter or not, we will assume it to be a list to
#             # standardize the following code.
#             dps = L(dps)
#             if self.shuffle:
#                 for i in range(len(dps)): dps[i] = dps[i].shuffle()
#             return dps


#         else:
        dps = dp.iter.IterableWrapper(items)
        if callable(self.splitter): dps = dps.demux(2,self.splitter)
        elif self.splitter is not None: dps = self.splitter(dps)

        # Regardless of the splitter or not, we will assume it to be a list to
        # standardize the following code.
        dps = L(dps)
        if self.shuffle:
            for i in range(len(dps)): dps[i] = dps[i].shuffle()

        dps = dps.map(Self.map(Pipeline(self.type_tfms)))
        dps = dps.map(Cacher)
        dps = dps.map(Self.map(Pipeline(self.item_tfms)))

        for i in range(len(dps)): dps[i] = dps[i].batch(self.bs)

        def force_fail(o):
            raise Exception

        dps = dps[0].map(lambda o:force_fail(o))#     Self.map(Pipeline(self.batch_tfms)))

        return dps


    def dataloaders(self, source, verbose=False, **kwargs)->List[DataLoader2]:
        dsets = self.datasets(source, verbose=verbose)
        kwargs = {**self.dls_kwargs, **kwargs, 'verbose': verbose}
        return dsets.dataloaders(path=path, after_item=self.item_tfms, after_batch=self.batch_tfms, **kwargs)

# Cell
T_co = TypeVar("T_co", covariant=True)

class Cacher(dp.iter.IterDataPipe[T_co]):
    def __init__(self, source_datapipe, **kwargs) -> None:
        self.source_datapipe = source_datapipe
        self.kwargs = kwargs

    def __iter__(self) -> Iterator[T_co]:
        cached_entries=[]
        use_cache=False
        while True:
            if use_cache:
                yield from cached_entries
            else:
                try:
                    for v in self.source_datapipe:
                        cached_entries.append(v)
                        yield v
                except StopIteration:
                    use_cache=True
                    cached_entries=cycle(cached_entries)

# Cell
def GrandparentSplitter(train_name='train', valid_name='valid'):
    "Split `items` to indexes 0 (train) and 1 (valid)."
    def _inner(o,negate=False):
        return o.parent.parent.name==(train_name if negate else valid_name)
    return _inner