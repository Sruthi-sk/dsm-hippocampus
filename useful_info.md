

GPU
***********************************
```sh
nvidia-smi
ps aux | grep <process number>
kill -9 

import psutil
memory_info = psutil.virtual_memory()
print(f"Available memory: {memory_info.available / 1024 / 1024 / 1024} GB")
import sys
print(f"Model size: {sys.getsizeof(trained_modelstate)} bytes")


import torch
torch.cuda.set_per_process_memory_fraction(0.6, 0)
#torch.cuda.list_gpu_processes(0)
# torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024
# total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
# total_memory
total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
total_memory*0.6
torch.cuda.list_gpu_processes(0)
torch.cuda.empty_cache()

```
-----

import importlib 
importlib.reload(modelviz_utils)

Change env and parameters inside configs.py whenever doing training or inference - we're importing it in many files inside

Codes
- Comparison of movement policies
- predictions over entire plot - in 2D env
- some atoms seem to converge predictions to single line or centre in the maze


1. comparison between previous SR and DSM - 1 gamma model vs ensemble of gamma (train with config distributional False) vs DSM
  - changing a basis feature  - turning off a PC (change activity to 0) - for all sources and generate rate map
  - 2% of world is changed - subtle changes to input can cause some PCs to change bt not others - previous SR models may not show that effect - 
  - models break
  atoms - predict - variations in atom responses may mimic the brain
- To test how much of the responses f an atom is a reslt of a change we made - keep same latents

2. Generalisation changes 
    train with a quarter of the maze not explored - and test with sources in unexplored env

3. DONE ?? : how to evaluate - 
to get the directional responses of atoms - 1 source - pmake dsm produce 1 sample - get head direction this way - rayleigh vector thing
 Cell directionality
Head direction cells  signal the direction in which an animal's head is pointing- part of the brain's internal compass system.
The Rayleigh vector length is a measure used in circular statistics- of the concentration of a set of directions. 

HIGH Rayleigh vector length indicates that the directions are concentrated around a certain value, while a 
LOW Rayleigh vector length indicates that the directions are spread out.

- Directionality of cells - collect all the recorder activities of a neuron and all the head directions then bin them. Strictly you also want to normalise each bin count by the number of times you sampled a particular head direction.

4. DONE: Checkpoints - euclidean similarity with original PCs - normalize cuz outputs different range - - learning profiles - checkpoints


5. Code: models trained on different sampling regimes - comparison
random vs high Th 
 - measure symmetry of rate maps - skew = width/height  ------ do animals with high Th have skewed PC rate maps ?
 - isometric vs concentration of field - entropy of states - check Slack - 
- average of all PCs as a population measure 
- similarity to original PCs - spatial correlation - colum vectorn - pearson corr
diff 
CHECK WHAT WAS SENT ON SLACK - SPARSITY FOR PLACE FIELD DEFINITION

6. 
- rate map outputs not constrained 0-1
in thesis: 
Atoms as respresenting different parts of world in real brain?
in dsm - unified training but change in env - compare mixture of how PCs change

7.
- relu instead of leaky_relu - models.py 


------
Errors (In development of another dataset with different number of place cells)
------
ScopeParamShapeError: Initializer expected to generate shape (, 32) but got shape (, 32) 
- In train.py: change num_state_dims or   -- replace math.prod(observation_spec.shape) everywhere with desired shape in train.py



wsl
===

- https://documentation.ubuntu.com/wsl/en/latest/guides/install-ubuntu-wsl2/
- https://documentation.ubuntu.com/wsl/en/latest/tutorials/gpu-cuda/ - THIS
- https://docs.nvidia.com/cuda/wsl-user-guide/index.html
- https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local

- https://ubuntu.com/blog/upgrade-data-science-workflows-ubuntu-wsl
- https://discourse.ubuntu.com/t/install-ubuntu-on-wsl2-and-get-started-with-graphical-applications/26296

- https://code.visualstudio.com/docs/remote/wsl 