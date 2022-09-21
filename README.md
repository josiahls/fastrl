fastrl
================

<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

[![CI
Status](https://github.com/josiahls/fastrl/workflows/Fastrl%20Testing/badge.svg)](https://github.com/josiahls/fastrl/actions?query=workflow%3A%22Fastrl+Testing%22)
[![pypi fastrl
version](https://img.shields.io/pypi/v/fastrl.svg)](https://pypi.python.org/pypi/fastrl)
[![Conda fastrl
version](https://img.shields.io/conda/v/josiahls/fastrl.svg)](https://anaconda.org/josiahls/fastrl)
[![Docker Image
Latest](https://img.shields.io/docker/v/josiahls/fastrl?label=Docker&sort=date.png)](https://hub.docker.com/repository/docker/josiahls/fastrl)
[![Docker Image-Dev
Latest](https://img.shields.io/docker/v/josiahls/fastrl-dev?label=Docker%20Dev&sort=date.png)](https://hub.docker.com/repository/docker/josiahls/fastrl-dev)

[![Anaconda-Server
Badge](https://anaconda.org/josiahls/fastrl/badges/platforms.svg)](https://anaconda.org/josiahls/fastrl)
[![fastrl python
compatibility](https://img.shields.io/pypi/pyversions/fastrl.svg)](https://pypi.python.org/pypi/fastrl)
[![fastrl
license](https://img.shields.io/pypi/l/fastrl.svg)](https://pypi.python.org/pypi/fastrl)

`nbdev_torchdata_incompat` produces the error:

``` bash
fastrl_nbdev_docs --n_workers 0
NB: From v1.2 `_quarto.yml` is no longer auto-updated. Please remove the `custom_quarto_yml` line from `settings.ini`
Traceback (most recent call last):
  File "/opt/conda/lib/python3.7/site-packages/IPython/core/interactiveshell.py", line 3441, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
  File "<ipython-input-1-2ad89adecba4>", line 1, in <module>
  File "/home/fastrl_user/fastrl/fastrl/pipes/map/demux.py", line 25, in <module>
    class DemultiplexerMapDataPipe(MapDataPipe[T_co]):
  File "/opt/conda/lib/python3.7/site-packages/torch/utils/data/datapipes/_decorator.py", line 36, in __call__
    MapDataPipe.register_datapipe_as_function(self.name, cls)
  File "/opt/conda/lib/python3.7/site-packages/torch/utils/data/datapipes/datapipe.py", line 263, in register_datapipe_as_function
    raise Exception("Unable to add DataPipe function name {} as it is already taken".format(function_name))
Exception: Unable to add DataPipe function name demux as it is already taken

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/fastrl_user/.local/bin/fastrl_nbdev_docs", line 33, in <module>
    sys.exit(load_entry_point('fastrl', 'console_scripts', 'fastrl_nbdev_docs')())
  File "/home/fastrl_user/.local/lib/python3.7/site-packages/fastcore/script.py", line 119, in _f
    return tfunc(**merge(args, args_from_prog(func, xtra)))
  File "/home/fastrl_user/fastrl/fastrl/cli.py", line 107, in fastrl_nbdev_docs
    cache,cfg,path = _pre_docs(path, n_workers=n_workers, verbose=verbose, **kwargs)
  File "/home/fastrl_user/fastrl/fastrl/cli.py", line 96, in _pre_docs
    cache = proc_nbs.__wrapped__(path, n_workers=n_workers, verbose=verbose)
  File "/home/fastrl_user/fastrl/fastrl/cli.py", line 79, in proc_nbs
    parallel(nbdev.serve_drv.main, files, n_workers=n_workers, pause=0.01, **kw)
  File "/home/fastrl_user/.local/lib/python3.7/site-packages/fastcore/parallel.py", line 117, in parallel
    return L(r)
  File "/home/fastrl_user/.local/lib/python3.7/site-packages/fastcore/foundation.py", line 98, in __call__
    return super().__call__(x, *args, **kwargs)
  File "/home/fastrl_user/.local/lib/python3.7/site-packages/fastcore/foundation.py", line 106, in __init__
    items = listify(items, *rest, use_list=use_list, match=match)
  File "/home/fastrl_user/.local/lib/python3.7/site-packages/fastcore/basics.py", line 66, in listify
    elif is_iter(o): res = list(o)
  File "/home/fastrl_user/.local/lib/python3.7/site-packages/nbdev/serve_drv.py", line 22, in main
    if src.suffix=='.ipynb': exec_nb(src, dst, x)
  File "/home/fastrl_user/.local/lib/python3.7/site-packages/nbdev/serve_drv.py", line 16, in exec_nb
    cb()(nb)
  File "/home/fastrl_user/.local/lib/python3.7/site-packages/nbdev/processors.py", line 221, in __call__
    def __call__(self, nb): return self.nb_proc(nb).process()
  File "/home/fastrl_user/.local/lib/python3.7/site-packages/nbdev/process.py", line 126, in process
    for proc in self.procs: self._proc(proc)
  File "/home/fastrl_user/.local/lib/python3.7/site-packages/nbdev/process.py", line 119, in _proc
    for cell in self.nb.cells: self._process_cell(proc, cell)
  File "/home/fastrl_user/.local/lib/python3.7/site-packages/nbdev/process.py", line 110, in _process_cell
    if callable(proc) and not _is_direc(proc): cell = opt_set(cell, proc(cell))
  File "/home/fastrl_user/.local/lib/python3.7/site-packages/nbdev/processors.py", line 201, in __call__
    raise Exception(f"Error{' in notebook: '+title if title else ''} in cell {cell.idx_} :\n{cell.source}") from self.k.exc[1]
Exception: Error in notebook: Multiplexer in cell 11 :
from fastrl.pipes.map.demux import DemultiplexerMapDataPipe
```
