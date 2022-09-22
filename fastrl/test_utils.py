# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/20_test_utils.ipynb.

# %% auto 0
__all__ = ['get_env', 'try_import', 'nvidia_mem', 'nvidia_smi', 'show_install']

# %% ../nbs/20_test_utils.ipynb 3
# Python native modules
import os
import importlib
# Third party libs
from fastcore.all import *
# from fastai.test_utils import *
import torch
# Local modules

# %% ../nbs/20_test_utils.ipynb 6
def get_env(name):
    "Return env var value if it's defined and not an empty string, or return Unknown"
    res = os.environ.get(name,'')
    return res if len(res) else "Unknown"

# %% ../nbs/20_test_utils.ipynb 7
def try_import(module):
    "Try to import `module`. Returns module's object on success, None on failure"
    try: return importlib.import_module(module)
    except: return None

# %% ../nbs/20_test_utils.ipynb 8
def nvidia_mem():
    try: mem = run("nvidia-smi --query-gpu=memory.total --format=csv,nounits,noheader")
    except: return None
    return mem.strip().split('\n')

# %% ../nbs/20_test_utils.ipynb 9
def nvidia_smi(cmd = "nvidia-smi"):
    try: res = run(cmd)
    except OSError as e: return None
    return res

# %% ../nbs/20_test_utils.ipynb 10
def show_install(show_nvidia_smi:bool=False):
    "Print user's setup information"

    import fastai, platform, fastprogress, fastcore, fastrl

    rep = []
    opt_mods = []

    rep.append(["=== Software ===", None])
    rep.append(["python", platform.python_version()])
    rep.append(["fastrl", fastrl.__version__])
    rep.append(["fastai", fastai.__version__])
    rep.append(["fastcore", fastcore.__version__])
    rep.append(["fastprogress", fastprogress.__version__])
    rep.append(["torch",  torch.__version__])

    # nvidia-smi
    smi = nvidia_smi()
    if smi:
        match = re.findall(r'Driver Version: +(\d+\.\d+)', smi)
        if match: rep.append(["nvidia driver", match[0]])

    available = "available" if torch.cuda.is_available() else "**Not available** "
    rep.append(["torch cuda", f"{torch.version.cuda} / is {available}"])

    # no point reporting on cudnn if cuda is not available, as it
    # seems to be enabled at times even on cpu-only setups
    if torch.cuda.is_available():
        enabled = "enabled" if torch.backends.cudnn.enabled else "**Not enabled** "
        rep.append(["torch cudnn", f"{torch.backends.cudnn.version()} / is {enabled}"])

    rep.append(["\n=== Hardware ===", None])

    gpu_total_mem = []
    nvidia_gpu_cnt = 0
    if smi:
        mem = nvidia_mem()
        nvidia_gpu_cnt = len(ifnone(mem, []))

    if nvidia_gpu_cnt: rep.append(["nvidia gpus", nvidia_gpu_cnt])

    torch_gpu_cnt = torch.cuda.device_count()
    if torch_gpu_cnt:
        rep.append(["torch devices", torch_gpu_cnt])
        # information for each gpu
        for i in range(torch_gpu_cnt):
            rep.append([f"  - gpu{i}", (f"{gpu_total_mem[i]}MB | " if gpu_total_mem else "") + torch.cuda.get_device_name(i)])
    else:
        if nvidia_gpu_cnt:
            rep.append([f"Have {nvidia_gpu_cnt} GPU(s), but torch can't use them (check nvidia driver)", None])
        else:
            rep.append([f"No GPUs available", None])


    rep.append(["\n=== Environment ===", None])

    rep.append(["platform", platform.platform()])

    if platform.system() == 'Linux':
        distro = try_import('distro')
        if distro:
            # full distro info
            rep.append(["distro", ' '.join(distro.linux_distribution())])
        else:
            opt_mods.append('distro');
            # partial distro info
            rep.append(["distro", platform.uname().version])

    rep.append(["conda env", get_env('CONDA_DEFAULT_ENV')])
    rep.append(["python", sys.executable])
    rep.append(["sys.path", "\n".join(sys.path)])

    print("\n\n```text")

    keylen = max([len(e[0]) for e in rep if e[1] is not None])
    for e in rep:
        print(f"{e[0]:{keylen}}", (f": {e[1]}" if e[1] is not None else ""))

    if smi:
        if show_nvidia_smi: print(f"\n{smi}")
    else:
        if torch_gpu_cnt: print("no nvidia-smi is found")
        else: print("no supported gpus found on this system")

    print("```\n")

    print("Please make sure to include opening/closing ``` when you paste into forums/github to make the reports appear formatted as code sections.\n")

    if opt_mods:
        print("Optional package(s) to enhance the diagnostics can be installed with:")
        print(f"pip install {' '.join(opt_mods)}")
        print("Once installed, re-run this utility to get the additional information")
