# Interpreting the Distributional Successor Measure as a model for the Hippocampus

## Abstract


Understanding how the brain represents and processes spatial information is fundamental to unraveling the mechanisms of navigation and memory. Hippocampal place cells play a central role in encoding spatial environments, forming cognitive maps that support flexible navigation. Traditional computational models, such as the Successor Representation (SR), have provided insights into how the hippocampus might encode expected future states based on current positions. However, these models are limited in capturing the dynamic and distributional nature of place cell activity, particularly their ability to adapt to environmental changes and encode distributions over possible future states.

This thesis explores the Distributional Successor Measure (DSM), an extension of the SR framework that models the full distribution of possible future states, aligning more closely with biological observations of hippocampal function. We employ neuroscientific analysis techniques to examine the internal representations of the DSM in controlled environments, such as the inverted pendulum task, to gain insights into its mechanisms. Additionally, we simulate rodent foraging behavior to assess whether the DSM can effectively model place cell dynamics and remapping behaviors observed in hippocampal neurons.

Our experiments compare the DSM with the traditional SR model, evaluating their ability to generalize across environments and capture the flexible, predictive nature of place cells. Results demonstrate that the DSM provides richer and more adaptable predictive representations, exhibiting greater resilience to environmental changes. The DSM's distributed predictive framework captures the variability and complexity of future states, mirroring the heterogeneous and dynamic nature of hippocampal representations.

These findings suggest that the DSM offers a more accurate and robust computational model for how the hippocampus encodes predictive maps in a distributional manner. By leveraging the DSM, we provide new insights into the neural mechanisms underlying spatial cognition and contribute to bridging the gap between computational models and biological neural systems.



This repository contains the implementation of the model and experiments presented in the thesis.



## Setup

This project makes heavy use of [Jax](https://github.com/google/jax), [Flax](https://github.com/google/flax), [Optax](https://github.com/google-deepmind/optax), and [Fiddle](https://github.com/google/fiddle). We use [pdm](https://pdm-project.org/latest/) to manage our dependencies. With the lockfile `pdm.lock` you should be able to faithfully instantiate the same environment we used to train our $\delta$-models. To do so, you can run the following commands,


## Setup

This project uses [JAX](https://github.com/google/jax), [Flax](https://github.com/google/flax), [Optax](https://github.com/google-deepmind/optax), and [Fiddle](https://github.com/google/fiddle). Dependency management is handled with [PDM](https://pdm-project.org/latest/). You can recreate the same environment we used to train our $\delta$-models by following these steps:


### Prerequisites
- For Windows users: Install Ubuntu on WSL and ensure CUDA drivers are installed.
- Python: 3.10

```sh
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10
python3.10 -m venv myenv  
source .venv/bin/activate
# pdm venv create
pip install pdm
pdm install
# pdm lock --update-reuse 
# pip install tensorflow==2.14.1
# pip install tensorflow-probability==0.22
# pip install absl-py
# pdm install jaxlib==0.4.23 jax==0.4.23
pip install ratinabox
# - place TaskEnvironmentGym inside .venv/lib/python3.10/site-packages/ratinabox/contribs/
pip install joblib
pip install pettingzoo 
pip install umap-learn

```
## Generating the Fixed Datasets

The following command will train a policy on the desired environment before generating a dataset of transitions from the learned policy. For example,
```sh
python -m dsm.scripts.make_dataset --env Pendulum-v1 --dataset_path <path/file> --policy_path <path>
```
NOTE: The policy will be cached and if you don't specify the `--force` flag it will skip the policy optimization step.


```sh
# Pendulum
python -m dsm.scripts.make_dataset_pendulum --env Pendulum-v1 --dataset_path datasets/pendulum/sac/dataset.pkl --policy_params_path datasets/pendulum/sac/policy_params.msgpack --force

# Rat-In-A-Box environments
# Ensure TaskEnvironmentGym.py is placed in .venv/packages/ratinabox/contribs

# RiaB - Random policy
python -m dsm.scripts.make_dataset_rat_PC_RANDOM --dataset_path datasets/ratinaboxPC/randomwalk/dataset.pkl 

# High thigmotaxis movement policy
python -m dsm.scripts.make_dataset_rat_PC_highTH --dataset_path datasets/ratinaboxPC/highTH/dataset.pkl

# Goal in centre policy
python -m dsm.scripts.make_dataset_rat_PC_goal --dataset_path datasets/ratinaboxPC/goal/sac/dataset.pkl --policy_path datasets/ratinaboxPC/goal/sac/policy --force 

# teleport policy
python -m dsm.scripts.make_dataset_rat_PC_TELEPORT --dataset_path datasets/ratinaboxPC/teleport/dataset.pkl 

# random wal - with wall constraining movement
python -m dsm.scripts.make_dataset_rat_PC_walls --dataset_path datasets/ratinaboxPC/walls/dataset.pkl 
```


## Training a $\delta$-Model

To train the $\delta$-model: 
- configure wandb
- change configs.py
- ensure dataset path in datasets.py

```sh
python -m dsm --workdir logdir --fdl_config=base
```
where `logdir` will store checkpoints of the saved model. Plots of the learned return distributions and various metrics will be logged periodically throughout training. These plots and metrics can be found in the experiment tracker 

```sh
# config.py env = "Pendulum-v1"
python -m dsm --workdir logdir_pendulum --fdl_config=base

# config.py env = "Ratinabox-v0-pc-random"
python -m dsm --workdir logdir-rat_50pc_random_walk --fdl_config=base

# config.py env = "Ratinabox-v0-pc-goal"
python -m dsm --workdir logdir-rat_50pc_goal --fdl_config=base


# config.py env = "Ratinabox-v0-pc-highTH"
python -m dsm --workdir logdir-rat_50pc_highTH --fdl_config=base

# config.py env = "Ratinabox-v0-pc-teleport"
python -m dsm --workdir logdir-rat_50pc_teleport --fdl_config=base

# config.py env = "Ratinabox-v0-pc-walls"
python -m dsm --workdir logdir-rat_50pc_walls --fdl_config=base


# For  SR experiment
# Change in configs.py: 
# num_outer=1, distributional=False, inner_separate_discriminator=True, 
# env = "Ratinabox-v0-pc-random" 
python -m dsm --workdir logdir-rat_50pc_random_SR --fdl_config=base   

```


### Experiment Tracking

You can switch how the experiment is logged either using Weights & Biases or Aim with the flag `--metric_writer {wandb, aim}`. Specific options for each of these methods can be configured via `--wandb.{save_code,tags,name,group,mode}` and `--aim.{repo=,experiment,log_system_params}` respectively.


### Main figures in thesis

Change environment in configs.py when using the notebooks, else ERRORS: Initializer expected to generate shape () but () was generated

All experiments in ```notebooks/```
- `model_viz_pendulum.ipynb` contains the code for all pendulum experiments
- `model_viz_RiaB_xxx.ipynb` where xxx represents each policy-env-model, contains the code for all models trained on different policies in RiaB environments
- `model_viz_RiaB_DSM_all.ipynb` contains the code for comparing the PCs when environment is distorted (by changing DSM input activities incrementally)
- `model_viz_RiaB_SR_randomwalk.ipynb` contains the code to compare the DSM representations with SR representations of preddicted PCs
These noteboos use supporting codes in `model_viz_loaders.py` and `model_viz_functions_pendulum.py`/ `model_viz_functions_riab.py`

Some figures shown in `Results.md`

If new environment: 
- make changes to dsm/types.py, dsm/envs.py
- add codefile to /distributional-sr/dsm/plotting- (functions for source state and plotting) for the module (eg - ratinabox.py) 
- add env in _PLOT_BY_ENVIRONMENT in /distributional-sr/dsm/plotting/__init__.py

If changing number of place cells 
- change NUM_STATE_DIM_CELLS in train.py 
- mlp_model_atom.num_outputs in model_viz_xx.ipynb

--------------------------------------------



## Future Work

- **Hyperparameter Tuning** Investigate the optimal hyperparameters of the DSM concerning the RatInABox environment
- **Improve Representation Disentanglement:** Implement constraints like positive activations or energy regularization to enhance efficiency and interpretability.
- **Incorporate Features for Biological Plausibility :** Adding recurrent connections could improve its .
- **Explore Behavioral Feature Representation:** Investigate whether DSM's latent variables capture distinct behavioral features like boundary adherence or goal-directed movement.
- **Test in Animal Models:** Validate DSM predictions in new experiments with rodent experiments in virtual reality, manipulating environmental cues.

## Other


Model: [A Distributional Analogue to the Successor Representation](https://arxiv.org/abs/2402.08530)

Gamma models 
- https://gammamodels.github.io/ 
-  https://github.com/jannerm/gamma-models/blob/main/scripts/gamma-pendulum.ipynb

RatInABox environment: <br>
 Tom M George, Mehul Rastogi, William de Cothi, Claudia Clopath, Kimberly Stachenfeld, Caswell Barry. "RatInABox, a toolkit for modelling locomotion and neuronal activity in continuous environments" (2024), eLife, https://doi.org/10.7554/eLife.85274 .


