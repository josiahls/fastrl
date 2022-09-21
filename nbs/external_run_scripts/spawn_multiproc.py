import torch
import torchdata.datapipes as dp
from torch.utils.data.dataloader_experimental import DataLoader2
       
class PointlessLoop(dp.iter.IterDataPipe):
    def __init__(self,datapipe=None):
        self.datapipe = datapipe
    
    def __iter__(self):
        while True:
            yield torch.LongTensor(4).detach().clone()
            

if __name__=='__main__':
    from torch.multiprocessing import Pool, Process, set_start_method
    try:
         set_start_method('spawn')
    except RuntimeError:
        pass


    pipe = PointlessLoop()
    pipe = pipe.header(limit=10)
    dls = [DataLoader2(pipe,num_workers=1)]
    # Setup the Learner
    print('type: ',type(dls[0]))
    for o in dls[0]:
        print(o)
