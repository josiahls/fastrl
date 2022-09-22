import torchdata.datapipes as dp
from torch.utils.data import IterableDataset

class AddABunch1(dp.iter.IterDataPipe):
    def __init__(self,q):
        super().__init__()
        self.q = [q]

    def __iter__(self):
        for o in range(10): 
            self.q[0].put(o)
            yield o
            
class AddABunch2(dp.iter.IterDataPipe):
    def __init__(self,source_datapipe,q):
        super().__init__()
        self.q = q
        print(id(self.q))
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for o in self.source_datapipe: 
            print(id(self.q))
            self.q.put(o)
            yield o
            
class AddABunch3(IterableDataset):
    def __init__(self,q):
        self.q = q

    def __iter__(self):
        for o in range(10): 
            print(id(self.q))
            self.q.put(o)
            yield o

if __name__=='__main__':
    from torch.multiprocessing import Pool,Process,set_start_method,Manager,get_start_method
    import torch
    
    try: set_start_method('spawn')
    except RuntimeError: pass
    # from torch.utils.data.dataloader_experimental import DataLoader2
    from torchdata.dataloader2 import DataLoader2
    from torchdata.dataloader2.reading_service import MultiProcessingReadingService

    m = Manager()
    q = m.Queue()
    
    pipe = AddABunch2(list(range(10)),q)
    print(type(pipe))
    dl = DataLoader2(pipe,
        reading_service=MultiProcessingReadingService(num_workers=1)
    ) # Will fail if num_workers>0
    
    # dl = DataLoader2(AddABunch1(q),num_workers=1) # Will fail if num_workers>0
    # dl = DataLoader2(AddABunch2(q),num_workers=1) # Will fail if num_workers>0
    # dl = DataLoader2(AddABunch3(q),num_workers=1) # Will succeed if num_workers>0
    list(dl)
    
    while not q.empty():
        print(q.get())
