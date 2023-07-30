# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/05_Logging/09e_loggers.tensorboard.ipynb.

# %% auto 0
__all__ = ['run_tensorboard']

# %% ../../nbs/05_Logging/09e_loggers.tensorboard.ipynb 1
# Python native modules
import os
from torch.multiprocessing import Queue
# Third party libs
import torchdata.datapipes as dp
# Local modules

# %% ../../nbs/05_Logging/09e_loggers.tensorboard.ipynb 4
def run_tensorboard(
        port:int=6006, # The port to run tensorboard on/connect on
        start_tag:str=None, # Starting regex e.g.: experience_replay/1
        samples_per_plugin:str=None, # Sampling freq such as  images=0 (keep all)
        extra_args:str=None, # Any additional arguments in the `--arg value` format
        rm_glob:bool=None # Remove old logs via a parttern e.g.: '*' will remove all files: runs/* 
    ):
    if rm_glob is not None:
        for p in Path('runs').glob(rm_glob): p.delete()
    import socket
    from tensorboard import notebook
    a_socket=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cmd=None
    if not a_socket.connect_ex(('127.0.0.1',6006)):
        notebook.display(port=port,height=1000)
    else:
        cmd=f'--logdir runs --port {port} --host=0.0.0.0'
        if samples_per_plugin is not None: cmd+=f' --samples_per_plugin {samples_per_plugin}'
        if start_tag is not None:          cmd+=f' --tag {start_tag}'
        if extra_args is not None:         cmd+=f' {extra_args}'
        notebook.start(cmd)
    return cmd
