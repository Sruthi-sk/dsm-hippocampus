
- cd /home/sruthi/Documents/thesis/distributional-sr/
- source ~/Documents/thesis/.venv/bin/activate

------------------------------------------------------------------------
************************************************************************
major changes when experimenting with different inputs / env to the DSM model <br>
- dsm/scripts/make_dataset.py
- dataset path - in dsm/datasets.py
- configs.py env="Ratinabox-v0-pc-teleport", or env="Ratinabox-v0-pc-random",
- types.py
- envs.py - if adding walls or changing goals of env - mostly unnecessary if dataset created with goal unless we want to compute reward distr metrics?
- train.py - NUM_STATE_DIM_CELLS (if changing no of PCs)
- add codefile to /distributional-sr/dsm/plotting- (functions for source state and plotting) for the module (eg - ratinabox.py) 
- add env in _PLOT_BY_ENVIRONMENT in /distributional-sr/dsm/plotting/__init__.py
- MLP output  -- replace math.prod(observation_spec.shape) everywhere with desired shape in train.py
- mlp_model_atom.num_outputs in model_viz_xx.ipynb

- monte carlo returns and rewards.py - not used
************************************************************************
-------------------------------------------------------------------------

MAIN CODE for DSM here (dataset should be creaed first, and wandb configured)
- python -m dsm --workdir logdir --fdl_config=base

Environments <br>
> distributional-sr-nav/blob/main/dsm/envs.py
- https://www.gymlibrary.dev/environments/classic_control/pendulum/
- https://github.com/RatInABox-Lab/RatInABox/blob/main/ratinabox/contribs/TaskEnv_example_files/TaskEnvironment_basics.md

- DATASET FOLDER DEFINED IN datasets.py FR EACH ENV

************************************************************************

### Pendulum
- Dataset: python -m dsm.scripts.make_dataset --env Pendulum-v1 --dataset_path datasets/pendulum/sac/dataset.pkl --policy_path datasets/pendulum/sac/policy
- changes in config.py and datasets.py
- Build model: python -m dsm --workdir logdir_pendulum_200k --fdl_config=base
- Inference : python load_model_compute_distr.py --fdl_config=base   #--workdir logdir 

Tested in another env

python -m dsm.scripts.make_dataset --env MountainCarContinuous-v0 --dataset_path datasets/mountaincar/sac/dataset.pkl --policy_path datasets/mountaincar/sac/policy \\
python -m dsm --workdir logdir-car --fdl_config=base

************************************************************************

### RatInABox

python -m dsm.scripts.make_dataset_rat --env Ratinabox-v0 --dataset_path datasets/ratinabox/sac/dataset.pkl --policy_path datasets/ratinabox/sac/policy --force \\
python -m dsm --workdir logdir-rat_test --fdl_config=base \\
wandb - ruby blaze  embedding 4.5 , mmd 0.14, obs 0.46   (logdir-rat   -10k)




Errors
------
ScopeParamShapeError: Initializer expected to generate shape (7, 32) but got shape (55, 32) 
- In train.py: num_state_dims or   -- replace math.prod(observation_spec.shape) everywhere with desired shape in train.py

TO-DO
-----
- random walk trajectories - save as dataset - ju

Saved Rat dataset has 1 goal - in the centre
so the dataset and generated dataset also regenerates those trajectories - all towards the centre
make multiple goals in env?
or is that something to investigate? - how it reacts to novel rewards - (done for pendulum in original paper)



python -m dsm.scripts.make_dataset_rat_PC_walls --dataset_path datasets/ratinaboxPCrandom/walls99/dataset.pkl 
python -m dsm --workdir logdir-rat_50pc_walls95 --fdl_config=base
-----------------------------------------------------------------------------

python -m dsm.scripts.make_dataset_rat --env Ratinabox-v0 --dataset_path datasets/ratinaboxtest/sac/dataset.pkl --policy_path datasets/ratinaboxtest/sac/policy --force

--------------------------------------------------------------------
python -m dsm.scripts.make_dataset_rat_PC_goal --dataset_path datasets/ratinaboxPCgoal/sac/dataset.pkl --policy_path datasets/ratinaboxPCgoal/sac/policy --force 

python -m dsm --workdir logdir-rat_50pc_goal --fdl_config=base
--------------------------------------------------------------------
python -m dsm.scripts.make_dataset_rat_PC_random --env Ratinabox-v0 --dataset_path datasets/ratinaboxPCrandom/sac/dataset.pkl --policy_path datasets/ratinaboxPCrandom/sac/policy --force 

python -m dsm --workdir logdir-rat_50pc_random --fdl_config=base
--------------------------------------------------------------------
python -m dsm.scripts.make_dataset_rat_PC_RANDOM --dataset_path datasets/ratinaboxPCrandom/simple/dataset.pkl 

python -m dsm --workdir logdir-rat_50pc_random_simple --fdl_config=base
--------------------------------------------------------------------
python -m dsm.scripts.make_dataset_rat_PC_TELEPORT --dataset_path datasets/ratinaboxPCteleport/simple/dataset.pkl 

python -m dsm --workdir logdir-rat_50pc_teleport --fdl_config=base
--------------------------------------------------------------------

datasets/ratinaboxPCrandom/simple/

***********************************
GPU
nvidia-smi
ps aux | grep <process number>
kill -9 
***********************************

make_dataset_rat_PC_random - ADD IMAGE PLOT TRAJECTORY



python -m dsm.scripts.make_dataset_rat_PC_goal --env Ratinabox-v0 --dataset_path datasets/ratinaboxPCgoal_test/sac/dataset.pkl --policy_path datasets/ratinaboxPCgoal_test/sac/policy 

!pip install tf-agents==0.18.0
Attempting uninstall: tensorflow-probability
    Found existing installation: tensorflow-probability 0.22.0
    Uninstalling tensorflow-probability-0.22.0:
      Successfully uninstalled tensorflow-probability-0.22.0
Successfully installed tensorflow-probability-0.22.1

TODO
-    In make_dataset, how to save the policy # policy_func = datasets.load_policy(policy_path) # ratinabox?

python -m dsm --workdir logdir-test-delete --fdl_config=base



ENVIRONMENT = "Ratinabox-v0-pc-goal" 
model_path = "logdir-rat_50pc_goal" 

ENVIRONMENT = "Ratinabox-v0-pc-random"
model_path = "logdir-rat_50pc_random" 


import psutil
memory_info = psutil.virtual_memory()
print(f"Available memory: {memory_info.available / 1024 / 1024 / 1024} GB")
import sys
print(f"Model size: {sys.getsizeof(trained_modelstate)} bytes")




Head direction cells  signal the direction in which an animal's head is pointing- part of the brain's internal compass system.
The Rayleigh vector length is a measure used in circular statistics- of the concentration of a set of directions. 

HIGH Rayleigh vector length indicates that the directions are concentrated around a certain value, while a 
LOW Rayleigh vector length indicates that the directions are spread out.



---------------------------------------------------------------------------------------------------

12th Aug

CODES
----
- configs.py: distributional = off   env="Ratinabox-v0-pc-random"
python -m dsm --workdir logdir-rat_50pc_random_dFalse --fdl_config=base
 env="Ratinabox-v0-pc-teleport", or env="Ratinabox-v0-pc-random",


1.  comparison between previous SR and DSM - 1 gamma model vs ensemble of gamma (train with config distributional False) vs DSM
  - changing a basis feature  - turning off a PC (change activity to 0) - for all sources and generate rate map
  - 2% of world is changed - subtle changes to input can cause some PCs to change bt not others - previous SR models may not show that effect - 
  - models break
  atoms - predict - variations in atom responses may mimic the brain

  changing world not same as PC being 0 !!!!!!!!!!!!!!!!!!!!!!!!!! ??????????????
  CHANGE PLACE CELL CENTRES

To test how much of the responses f an atom is a reslt of a change we made - keep same latents

2. Generalisation changes 
    train with a quarter of the maze not explored - and test with sources in unexplored env

3. DONE ?? : how to evluate - 
to get the directional responses of atoms - 1 source - pmake dsm produce 1 sample - get head direction this way - rayleigh vector thing

4. DONE: Checkpoints - euclidean similarity with original PCs - normalize cuz outputs different range ?

5. Code: models trained on different sampling regimes - comparison
random vs high Th 
 - measure symmetry of rate maps - skew = width/height  ------ do animals with high Th have skewed PC rate maps ?
 - isometric vs concentration of field - entropy of states - check Slack - 
- average of all PCs as a population measure 
- similarity to original PCs - spatial correlation - colum vectorn - pearson corr
diff 
CHECK WHAT WAS SENT ON SLACK - SPARSITY FOR PLACE FIELD DEFINITION

6. Problem
- rate map outputs not constrained 0-1
iNPUT AS BVCs?
----------------------------------------------------------------------------
6. in thesis: 
Atoms as respresenting different parts of world in real brain?
in dsm - unified training but change in env - compare mixture of how PCs change
Results 
- movement policies 4 - 
- learning profiles - checkpoints
- goal directed behavior may not work in this setting
--------------------------


in models.py
 x = jnn.sigmoid(x)  # apply sigmoid activation to final output ?


import importlib 
importlib.reload(modelviz_utils)


TRY RELU instead of leaky_relu - models.py - disentaglement paper - https://openreview.net/pdf?id=9Z_GfhZnGH
- maybe insted of ocntour plots - something else? then will i see grid cells in hex pattern - check other codes - mabe Tom's

To show
- relu outputs
- predictions over entire plot - in 2D env - some atoms seem to converge predictions to single line or centre in the maze
- collect all the recorder activities of a neuron and all the head directions then bin them. Strictly you also want to normalise each bin count by the number of times you sampled a particular head direction.


--------------------
to check correct decoder - build with walls and see? -


Change env and parameters inside configs.py whenever doing training or inference - we're importing it in many files inside