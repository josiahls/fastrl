

# Cell
from fastai.torch_basics import *
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter,_SingleProcessDataLoaderIter,_DatasetKind
_loaders = (_MultiProcessingDataLoaderIter,_SingleProcessDataLoaderIter)

# Cell
def _wif(worker_id):
    set_num_threads(1)
    info = get_worker_info()
    ds = info.dataset.d
    ds.num_workers,ds.offs = info.num_workers,info.id
    set_seed(info.seed)
    ds.wif()

class _FakeLoader:
    def _fn_noops(self, x=None, *args, **kwargs): return x

    _IterableDataset_len_called,_auto_collation,collate_fn,drop_last = None,False,_fn_noops,False
    _index_sampler,generator,prefetch_factor  = Inf.count,None,2
    dataset_kind = _dataset_kind = _DatasetKind.Iterable

    def __init__(self, d, pin_memory, num_workers, timeout, persistent_workers):
        self.dataset,self.default,self.worker_init_fn = self,d,_wif
        store_attr('d,pin_memory,num_workers,timeout,persistent_workers')

    def __iter__(self): return iter(self.d.create_batches(self.d.sample()))

    @property
    def multiprocessing_context(self): return (None,multiprocessing)[self.num_workers>0]

    @contextmanager
    def no_multiproc(self):
        old_num_workers = self.num_workers
        try:
            self.num_workers = 0
            yield self.d
        finally: self.num_workers = old_num_workers

_collate_types = (ndarray, Tensor, typing.Mapping, str)

# Cell
from fastai.torch_basics import *
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter,_SingleProcessDataLoaderIter,_DatasetKind
_loaders = (_MultiProcessingDataLoaderIter,_SingleProcessDataLoaderIter)

# Cell
def _wif(worker_id):
    set_num_threads(1)
    info = get_worker_info()
    ds = info.dataset.d
    ds.num_workers,ds.offs = info.num_workers,info.id
    set_seed(info.seed)
    ds.wif()

class _FakeLoader:
    def _fn_noops(self, x=None, *args, **kwargs): return x

    _IterableDataset_len_called,_auto_collation,collate_fn,drop_last = None,False,_fn_noops,False
    _index_sampler,generator,prefetch_factor  = Inf.count,None,2
    dataset_kind = _dataset_kind = _DatasetKind.Iterable

    def __init__(self, d, pin_memory, num_workers, timeout, persistent_workers):
        self.dataset,self.default,self.worker_init_fn = self,d,_wif
        store_attr('d,pin_memory,num_workers,timeout,persistent_workers')

    def __iter__(self): return iter(self.d.create_batches(self.d.sample()))

    @property
    def multiprocessing_context(self): return (None,multiprocessing)[self.num_workers>0]

    @contextmanager
    def no_multiproc(self):
        old_num_workers = self.num_workers
        try:
            self.num_workers = 0
            yield self.d
        finally: self.num_workers = old_num_workers

_collate_types = (ndarray, Tensor, typing.Mapping, str)

# Cell
from fastai.torch_basics import *
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter,_SingleProcessDataLoaderIter,_DatasetKind
_loaders = (_MultiProcessingDataLoaderIter,_SingleProcessDataLoaderIter)

# Cell
def _wif(worker_id):
    set_num_threads(1)
    info = get_worker_info()
    ds = info.dataset.d
    ds.num_workers,ds.offs = info.num_workers,info.id
    set_seed(info.seed)
    ds.wif()

class _FakeLoader:
    def _fn_noops(self, x=None, *args, **kwargs): return x

    _IterableDataset_len_called,_auto_collation,collate_fn,drop_last = None,False,_fn_noops,False
    _index_sampler,generator,prefetch_factor  = Inf.count,None,2
    dataset_kind = _dataset_kind = _DatasetKind.Iterable

    def __init__(self, d, pin_memory, num_workers, timeout, persistent_workers):
        self.dataset,self.default,self.worker_init_fn = self,d,_wif
        store_attr('d,pin_memory,num_workers,timeout,persistent_workers')

    def __iter__(self): return iter(self.d.create_batches(self.d.sample()))

    @property
    def multiprocessing_context(self): return (None,multiprocessing)[self.num_workers>0]

    @contextmanager
    def no_multiproc(self):
        old_num_workers = self.num_workers
        try:
            self.num_workers = 0
            yield self.d
        finally: self.num_workers = old_num_workers

_collate_types = (ndarray, Tensor, typing.Mapping, str)

# Cell
from fastai.torch_basics import *
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter,_SingleProcessDataLoaderIter,_DatasetKind
_loaders = (_MultiProcessingDataLoaderIter,_SingleProcessDataLoaderIter)

# Cell
def _wif(worker_id):
    set_num_threads(1)
    info = get_worker_info()
    ds = info.dataset.d
    ds.num_workers,ds.offs = info.num_workers,info.id
    set_seed(info.seed)
    ds.wif()

class _FakeLoader:
    def _fn_noops(self, x=None, *args, **kwargs): return x

    _IterableDataset_len_called,_auto_collation,collate_fn,drop_last = None,False,_fn_noops,False
    _index_sampler,generator,prefetch_factor  = Inf.count,None,2
    dataset_kind = _dataset_kind = _DatasetKind.Iterable

    def __init__(self, d, pin_memory, num_workers, timeout, persistent_workers):
        self.dataset,self.default,self.worker_init_fn = self,d,_wif
        store_attr('d,pin_memory,num_workers,timeout,persistent_workers')

    def __iter__(self): return iter(self.d.create_batches(self.d.sample()))

    @property
    def multiprocessing_context(self): return (None,multiprocessing)[self.num_workers>0]

    @contextmanager
    def no_multiproc(self):
        old_num_workers = self.num_workers
        try:
            self.num_workers = 0
            yield self.d
        finally: self.num_workers = old_num_workers

_collate_types = (ndarray, Tensor, typing.Mapping, str)

# Cell
# Python native modules
import os
# Third party libs
from fastai.imports import *
from fastai.torch_imports import *
from packaging.version import parse
# Local modules