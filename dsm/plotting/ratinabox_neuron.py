from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from dsm import datasets
from dsm.configs import Config
from dsm.plotting import utils as plotting_utils
from dsm.state import FittedValueTrainState

import decoder_PC2d

# ENVIRONMENT = "Ratinabox-v0-neuron"
print('DEBUG: ratinabox_neuron.py')
from dsm.configs import base
config = base()
ENVIRONMENT = config.env

if ENVIRONMENT.startswith("Ratinabox-v0-pc"):
    from dsm.datasets import _DATASET_REGISTRY
    import os
    import joblib
    from dsm import envs
    from ratinabox.Neurons import PlaceCells

    folder_dataset_path = os.path.dirname(_DATASET_REGISTRY[ENVIRONMENT])
    print(f'ENVIRONMENT: {ENVIRONMENT}, folder_dataset_path: {folder_dataset_path}')
    pc_path = folder_dataset_path+'/placecells_params.pkl'
    env = envs.make(ENVIRONMENT)  #config.env
    Ag = env.agent_lookup('agent_0')[0] # Agent(env) will create a new agent - agent0 already added in envs.py
    PC_params = joblib.load(pc_path)
    # NUM_STATE_DIM_CELLS = PC_params['n']
    # print('NUM_STATE_DIM_CELLS:',NUM_STATE_DIM_CELLS)
    PCs = PlaceCells(Ag, params=PC_params) 

    decoder_PC2d.train_decoder(PCs,Ag=Ag)

def source_states() -> tuple[list[Any], list[npt.NDArray]]:
    print('CALLED source_states()')
    
    env_coords_small = env.discretise_environment(dx=0.3) # dx=Ag.environment.scale/10   # generate 9 images?
    env_coords_small = env_coords_small.reshape(-1, env_coords_small.shape[-1])
    # pc_full_env = PCs.get_state(evaluate_at="all").T # N of 10000 values corresponding to ag.Environment.flattened_discrete_coords # len 10000
    source_states_env = PCs.get_state(evaluate_at=None, pos=env_coords_small).T
    source_states_env = [np.array(x) for x in source_states_env]
    env_coords_small = [np.array(x) for x in env_coords_small]

    return env_coords_small, source_states_env  # OR SHOULD BOTH BE source_states_env? #TODO: check
    # return states, observations

def plot_samples(
    source: npt.NDArray,
    state: FittedValueTrainState,
    rng: jax.random.KeyArray,
    *,
    config: Config,
    env_source = None,
) -> npt.NDArray:
    dataset = datasets.make_dataset(ENVIRONMENT)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6 * 2, 6))

    samples = plotting_utils.sample_from_sr(
        state,
        rng,
        jnp.array(source),
        num_samples=config.plot_num_samples,
        num_outer=config.num_outer,
        num_latent_dims=config.latent_dims,
    )

    # # Left scatter plot
    # dataset_positions = decoder_PC2d.simple_decode_position(dataset.observation,env_coords,pc_full_env,plot=False)
    dataset_positions = decoder_PC2d.decode_position(PCs,dataset.observation,plot=False)
    # # converts Cartesian coordinates to polar coordinates (thetas) and extracts velocities
    xs = dataset_positions[:, 0]
    ys = dataset_positions[:, 1]
    # colors = range(len(xs))
    axs[0].scatter(xs, ys, alpha=0.2, color='blue')
    # axs[0].scatter(xs, ys, alpha=0.1, c=colors, cmap='viridis')
    # axs[0].scatter(xs[0], ys[0], color='k',s=20)

    # # Plot atom scatter & kde
    cmap = plt.get_cmap("Dark2")  # pyright: ignore  #cmap = plt.get_cmap('tab10')
    for i in range(samples.shape[0]):
        atoms_samples = samples[i]
        atoms_samples_xy = decoder_PC2d.decode_position(PCs,atoms_samples,plot=False)
        xs = atoms_samples_xy[:, 0]
        ys = atoms_samples_xy[:, -1]
        axs[1].scatter(xs, ys,color=cmap(i),s=40, alpha=0.25)

    # Plot source state
    source_xy = decoder_PC2d.decode_position(PCs,source.reshape(1,-1),plot=False)[0]
    for ax in axs:
        ax.scatter(source_xy[0], source_xy[-1], marker="x", s=32, alpha=0.8, color="coral")
        if env_source is not None:
            # print('DEBUG env_source:',env_source, 'source_xy: ',source_xy)
            ax.scatter(env_source[0], env_source[-1], marker="x", s=64, alpha=0.8, color="red")

    # # set bounds
    for ax in axs:
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        # ax.set_xlim([-0.5, 1.5])
        # ax.set_ylim([-0.5, 1.5])
        # ax.set_aspect("auto")
        # ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi], ["0", "π/2", "π", "3π/2", "2π"])

    image = plotting_utils.fig_to_ndarray(fig)
    plt.close(fig)

    return image