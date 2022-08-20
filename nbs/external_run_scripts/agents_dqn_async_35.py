# %%python

if __name__=='__main__':
    from torch.multiprocessing import Pool, Process, set_start_method
    
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    
    from fastcore.all import *
    import torch
    from torch.nn import *
    import torch.nn.functional as F
    from fastrl.loggers.core import *
    from fastrl.loggers.jupyter_visualizers import *
    from fastrl.learner.core import *
    from fastrl.fastai.data.block import *
    from fastrl.envs.gym import *
    from fastrl.agents.core import *
    from fastrl.agents.discrete import *
    from fastrl.agents.dqn.basic import *
    from fastrl.agents.dqn.asynchronous import *
    from torch.utils.data.dataloader_experimental import DataLoader2
    
    logger_base = ProgressBarLogger(epoch_on_pipe=EpocherCollector,
                     batch_on_pipe=BatchCollector)

    # Setup up the core NN
    torch.manual_seed(0)
    model = DQN(4,2).cuda()
    # model.share_memory() # This will not work in spawn
    # Setup the Agent
    agent = DQNAgent(model,[logger_base],max_steps=8000,device='cuda')
    # Setup the DataBlock
    block = DataBlock(
        blocks = GymTransformBlock(agent=agent,
                                   nsteps=1,nskips=1,firstlast=False,
                                   dl_type=partial(DataLoader2,persistent_workers=True
                                                  )
                                  )
    )
    # pipes = L(block.datapipes(['CartPole-v1']*1,n=10))
    dls = L(block.dataloaders(['CartPole-v1']*1,n=1000,bs=1,num_workers=1))
    print('persistent workers: ',dls[0].persistent_workers)
    # # Setup the Learner
    learner = DQNLearner(model,dls,[agent],logger_bases=[logger_base],bs=128,max_sz=100_000,device='cuda')
    learner.fit(20)
