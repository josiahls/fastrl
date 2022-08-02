# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02f_fastai.learner.ipynb (unless otherwise specified).

__all__ = ['Learner']

# Cell
# Python native modules
import os
# Third party libs
from fastcore.all import *
# Local modules

# Cell
class Learner(object): #(dp.iter.IterDataPipe):
    def __init__(self,model,dls,opt,loss_func,cbs,train_loop=None):
        store_attr('model,dls,opt,loss_func')
        self.cbs = L()
        self.add_cbs(cbs)
        self.train_loop = ifnone(train_loop,default_train_loop)

    def fit(self,epochs):
        self.it = iter(self.dls[0])
        self.train_pipe = only_train_loop(L(self.it),epochs,self,self.cbs) # Do not pass tuple, otherwise traverse will try to read the dl datapipes
        for res in self.train_pipe:pass

    def add_cbs(self, cbs):
        L(cbs).map(self.add_cb)
        return self

    def remove_cbs(self, cbs):
        L(cbs).map(self.remove_cb)
        return self

    def add_cb(self, cb):
        if isinstance(cb, type): cb = cb()
        cb.learn = self
        # cb.init_pipes()
        setattr(self, cb.name, cb)
        self.cbs.append(cb)
        return self