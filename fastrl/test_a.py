# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/test_A.ipynb.

# %% auto 0
__all__ = ['Dog']

# %% ../nbs/test_A.ipynb 1
import torchdata.datapipes as dp
from torchdata.datapipes import functional_datapipe
from torchdata.dataloader2.graph import find_dps,DataPipeGraph,Type,DataPipe,MapDataPipe,IterDataPipe

# %% ../nbs/test_A.ipynb 2
@functional_datapipe("test_dog")
class Dog(MapDataPipe):pass
