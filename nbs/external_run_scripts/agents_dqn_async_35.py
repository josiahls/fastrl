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
    from fastrl.data.dataloader2 import *
    
    logger_base = ProgressBarLogger(epoch_on_pipe=EpocherCollector,
                     batch_on_pipe=BatchCollector)
    
    # RollingTerminatedRewardCollector.debug=True

    # Setup up the core NN
    torch.manual_seed(0)
    model = DQN(4,2).cuda()
    # model.share_memory() # This will not work in spawn
    # Setup the Agent
    agent = DQNAgent(model,max_steps=8000,device='cuda')
    # Setup the DataBlock
    block = DataBlock(
        blocks = GymTransformBlock(agent=agent,
                                   nsteps=1,nskips=1,firstlast=False
                                  )
    )
    pipe = L(block.datapipes(['CartPole-v1']*1))
    
    dl = DataLoader2(
        pipe[0],
        reading_service=PrototypeMultiProcessingReadingService(
            num_workers = 5,
            # persistent_workers=True,
            protocol_client_type = InputItemIterDataPipeQueueProtocolClient,
            protocol_server_type = InputItemIterDataPipeQueueProtocolServer,
            pipe_type = item_input_pipe_type,
            eventloop = SpawnProcessForDataPipeline
        )
    )

    dls = [dl]
    
    # from torchdata.dataloader2.graph import find_dps,traverse
    # print(traverse(dls[0].datapipe))
    
    # dls = L(block.dataloaders(['CartPole-v1']*1,n=1000,bs=1,num_workers=1))
    # print('persistent workers: ',dls[0].persistent_workers)
    # # Setup the Learner
    learner = DQNLearner(model,dls,[agent],batches=1000,logger_bases=[logger_base],
                         publish_freq=100,
                         bs=128,max_sz=100_000,device='cuda')
    learner.fit(20)
