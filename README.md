# Distributional Successor Measure - Interpreting as a Hippocampal model

This repository contains the reference implementation of the Distributional Successor Measure presented in:

**[A Distributional Analogue to the Successor Representation](https://arxiv.org/abs/2402.08530)**

by [Harley Wiltzer](https://harwiltz.github.io/)* & [Jesse Farebrother](https://brosa.ca)*, [Arthur Gretton](https://www.gatsby.ucl.ac.uk/~gretton/), [Yunhao Tang](https://robintyh1.github.io/), [André Baretto](https://sites.google.com/view/andrebarreto/about), [Will Dabney](https://willdabney.com/), [Marc G. Bellemare](http://www.marcgbellemare.info/), and [Mark Rowland](https://sites.google.com/view/markrowland).

https://github.com/JesseFarebro/distributional-sr/assets/1377567/eea0a53a-65d7-4201-a234-6609d1166d11

The Distributional Successor Measure (DSM) a new approach for distributional reinforcement learning which proposes a clean separation of transition structure and reward in the learning process. Analogous to how the successor representation (SR) describes the expected consequences of behaving according to a given policy, our distributional successor measure describes the distributional consequences of this behaviour. This repository contains the code for learning a $\delta$-model, our proposed representation that learns the distributional SM.

## Setup

This project makes heavy use of [Jax](https://github.com/google/jax), [Flax](https://github.com/google/flax), [Optax](https://github.com/google-deepmind/optax), and [Fiddle](https://github.com/google/fiddle). We use [pdm](https://pdm-project.org/latest/) to manage our dependencies. With the lockfile `pdm.lock` you should be able to faithfully instantiate the same environment we used to train our $\delta$-models. To do so, you can run the following commands,
```sh
pdm venv create
pdm install
```


## Generating the Fixed Datasets

The following command will train a policy on the desired environment before generating a dataset of transitions from the learned policy. For example,
```sh
python -m sr.scripts.make_dataset --env Pendulum-v1 --dataset_path datasets/pendulum/sac/dataset.pkl --policy_path datasets/pendulum/sac/policy
```
NOTE: The policy will be cached and if you don't specify the `--force` flag it will skip the policy optimization step.

## Training a $\delta$-Model

To train the $\delta$-model from the paper you can simply run:

```sh
python -m dsm --workdir logdir
```

where `logdir` will store checkpoints of the saved model. Plots of the learned return distributions and various metrics will be logged periodically throughout training. These plots and metrics can be found in the experiment tracker (defaults to Aim).

### Experiment Tracking

You can switch how the experiment is logged either using Weights & Biases or Aim with the flag `--metric_writer {wandb, aim}`. Specific options for each of these methods can be configured via `--wandb.{save_code,tags,name,group,mode}` and `--aim.{repo=,experiment,log_system_params}` respectively.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.


## Installation
- If windows - install ubuntu on wsl , cuda drivers
- sudo add-apt-repository ppa:deadsnakes/ppa
- sudo apt update
- sudo apt install python3.10
- python3.10 -m venv myenv  (or) python3.10 -m pip install pdm, pdm venv create
- source .venv/bin/activate
(if needed 
python -m ensurepip --upgrade
python -m pip install --upgrade pip
)
- pip install pdm
- pdm install ( pdm lock --update-reuse (if needed))
If needed - version mismatch
(
- pip install tensorflow==2.14.1
- pip install tensorflow-probability==0.22
- pip install absl-py
- pdm install jaxlib==0.4.23 jax==0.4.23
)
- pip install ratinabox
- place TaskEnvironmentGym inside .venv/lib/python3.10/site-packages/ratinabox/contribs/
- pip install joblib
- pip install pettingzoo 



## Thesis

- source .venv/bin/activate
- cd dsm-hippocampus


### Pendulum
- Dataset:  
 python -m dsm.scripts.make_dataset --env Pendulum-v1 --dataset_path datasets/pendulum/sac/dataset.pkl --policy_params_path datasets/pendulum/sac/policy_params.msgpack --force
- changes in config.py and datasets.py
- Build model: python -m dsm --workdir logdir_pendulum --fdl_config=base
- Inference : python load_model_compute_distr.py --fdl_config=base   #--workdir logdir 

Tested in MountainCarContinuous:
python -m dsm.scripts.make_dataset --env MountainCarContinuous-v0 --dataset_path datasets/mountaincar/sac/dataset.pkl --policy_path datasets/mountaincar/sac/policy \\
python -m dsm --workdir logdir-car --fdl_config=base

************************************************************************

### RatInABox

python -m dsm.scripts.make_dataset_rat_PC_RANDOM --dataset_path datasets/ratinaboxPC/randomwalk/dataset.pkl 
python -m dsm --workdir logdir-rat_50pc_random_walk --fdl_config=base
-------------------------------------------------------------------------------------------------------------
python -m dsm.scripts.make_dataset_rat_PC_highTH --dataset_path datasets/ratinaboxPC/highTH/dataset.pkl
python -m dsm --workdir logdir-rat_50pc_highTH --fdl_config=base
-------------------------------------------------------------------------------------------------------------
python -m dsm.scripts.make_dataset_rat_PC_goal --dataset_path datasets/ratinaboxPC/goal/sac/dataset.pkl --policy_path datasets/ratinaboxPC/goal/sac/policy --force 
python -m dsm --workdir logdir-rat_50pc_goal --fdl_config=base
-------------------------------------------------------------------------------------------------------------
python -m dsm.scripts.make_dataset_rat_PC_TELEPORT --dataset_path datasets/ratinaboxPC/teleport/dataset.pkl 
python -m dsm --workdir logdir-rat_50pc_teleport --fdl_config=base
-------------------------------------------------------------------------------------------------------------
python -m dsm.scripts.make_dataset_rat_PC_walls --dataset_path datasets/ratinaboxPC/walls/dataset.pkl 
python -m dsm --workdir logdir-rat_50pc_walls --fdl_config=base
-------------------------------------------------------------------------------------------------------------


For comparison with SR experiment
- In configs.py, changed num_outer=1, distributional=False, inner_separate_discriminator=True 
- python -m dsm --workdir logdir-rat_50pc_random_SR_2 --fdl_config=base   

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


Errors (In development of another dataset with different number of place cells)
------
ScopeParamShapeError: Initializer expected to generate shape (, 32) but got shape (, 32) 
- In train.py: change num_state_dims or   -- replace math.prod(observation_spec.shape) everywhere with desired shape in train.py



Saved Rat dataset has 1 goal - in the centre
so the dataset and generated dataset also regenerates those trajectories - all towards the centre
make multiple goals in env?
or is that something to investigate? - how it reacts to novel rewards - (done for pendulum in original paper)





