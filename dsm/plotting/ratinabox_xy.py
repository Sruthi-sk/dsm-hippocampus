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

ENVIRONMENT = "Ratinabox-v0-xy"

def source_states() -> tuple[list[Any], list[npt.NDArray]]:
    xposes = [0.1, 0.3, 0.5, 0.7, 0.9]
    yposes = [0.05, 0.2, 0.4, 0.5, 0.6, 0.8, 0.95]

    states = []
    observations = []
    for theta in xposes:
        for thetadot in yposes:
            states.append(np.array([theta, thetadot]))
            observations.append(np.array([theta, thetadot]))

    return states, observations


def plot_samples(
    source: npt.NDArray,
    state: FittedValueTrainState,
    rng: jax.random.KeyArray,
    *,
    config: Config,
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

    # Left scatter plot
    xs = dataset.observation[:, 0]
    ys = dataset.observation[:, -1]
    axs[0].scatter(xs, ys, alpha=0.1, s=1.0, color="grey")

    # Plot atom scatter & kde
    cmap = plt.get_cmap("Dark2")  # pyright: ignore
    for i in range(samples.shape[0]):
        xs = samples[i, :, 0]
        ys = samples[i, :, -1]
        axs[1].scatter(xs, ys, color=cmap(i), s=2.0, alpha=0.25)

    # Plot source state
    for ax in axs:
        ax.scatter(source[0], source[-1], marker="x", s=64, alpha=0.8, color="red")

    # set bounds
    for ax in axs:
        # ax.set_ylim(-8.5, 8.5)
        ax.set_aspect("auto")
        # ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi], ["0", "π/2", "π", "3π/2", "2π"])

    image = plotting_utils.fig_to_ndarray(fig)
    plt.close(fig)

    return image
