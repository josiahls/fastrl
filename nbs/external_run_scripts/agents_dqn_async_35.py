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
    from fastrl.data.block import *
    from fastrl.envs.gym import *
    from fastrl.agents.core import *
    from fastrl.agents.discrete import *
    from fastrl.agents.dqn.basic import *
    from fastrl.agents.dqn.asynchronous import *
    
    from torchdata.dataloader2 import DataLoader2
    from torchdata.dataloader2.graph import traverse
    from fastrl.data.dataloader2 import *
    
    logger_base = ProgressBarLogger(epoch_on_pipe=EpocherCollector,
                    batch_on_pipe=BatchCollector)

    # Setup up the core NN
    torch.manual_seed(0)
    model = DQN(4,2).cuda()
    # model.share_memory() # This will not work in spawn
    # Setup the Agent
    agent = DQNAgent(model,max_steps=4000,device='cuda',
                    dp_augmentation_fns=[ModelSubscriber.insert_dp()])
    # Setup the DataBlock
    block = DataBlock(
        GymTransformBlock(agent=agent,nsteps=1,nskips=1,firstlast=False),
        GymTransformBlock(agent=agent,nsteps=1,nskips=1,firstlast=False,include_images=True)
    )
    dls = L(block.dataloaders(['CartPole-v1']*1,num_workers=1))
    # # Setup the Learner
    learner = DQNLearner(model,dls,batches=1000,logger_bases=[logger_base],bs=128,max_sz=100_000,device='cuda',
                        dp_augmentation_fns=[ModelPublisher.insert_dp(publish_freq=10)])
    # print(traverse(learner))
    learner.fit(20)
