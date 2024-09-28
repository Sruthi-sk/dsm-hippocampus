"""
python -m dsm.scripts.make_dataset_rat_PC_TELEPORT --dataset_path datasets/ratinaboxPC/teleport/dataset.pkl 
- add to datasets.py
_DATASET_REGISTRY: 
    "Ratinabox-v0-pc-teleport": pathlib.Path("datasets/ratinaboxPCteleport/simple/dataset.pkl"),
- add to - dsm/plotting/_init_.py - _PLOT_BY_ENVIRONMENT 
- change env in configs.py
python -m dsm --workdir logdir-rat_50pc_teleport --fdl_config=base
"""

from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import PlaceCells

import numpy as np
import pandas as pd

import os
import tyro
import functools
import logging
import pathlib
import pickle
from typing import Annotated


NUM_PLACE_CELLS = 50
NUM_SECS = 20*60
# folder_dataset_path = 'datasets/ratinaboxPC/randomwalk/'
# dataset_path = pathlib.Path(folder_dataset_path+"dataset.pkl")



def main(
    dataset_path: pathlib.Path,
    *,
    # env_id: Annotated[Environment, tyro.conf.arg(name="env")],
    seed: int | None = None,
    # train_steps: int = TRAIN_STEPS,
    # train_steps: int = 10,
    # policy_path: pathlib.Path,
    # num_eval_steps: int = NUM_SECS,
    # num_eval_steps: int = 10_0,
    # sticky_action_prob: float = 0.0,
    # force: bool = False,
):
    
    env_id = "Ratinabox-v0-pc-teleport"

    folder_dataset_path = os.path.dirname(dataset_path)
    # base_path = 'datasets/ratinaboxPCgoal/'  # dataset_path
    if not os.path.exists(folder_dataset_path):
        os.makedirs(folder_dataset_path)
    logger = logging.getLogger('my_logger')
    # Set the level of the logger. This can be DEBUG, INFO, WARNING, ERROR, CRITICAL.
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(folder_dataset_path+'/my_log.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.debug(env_id)


    Env = Environment()
    # Env.add_wall(np.array([[0.4, 0], [0.4, 0.4]]))
    # Ag = Agent(Env,params={'dt':1,'speed_mean':0.2})
    Ag = Agent(Env)
    # Ag.pos = np.array([0.5, 0.5])
    # Env.add_agents(Ag)  ### ???? #TODO: check if this is needed

    N=12000
    positions = np.random.random((N,2)) # generating random positions

    placecells = PlaceCells(Ag, params={'n':NUM_PLACE_CELLS,}) 
    logger.debug(f'placecells: {NUM_PLACE_CELLS}')

    Ag.import_trajectory(times = np.linspace(0, NUM_SECS,len(positions)), positions=positions) #,interpolate=True
    for i in range(int(NUM_SECS / Ag.dt)):
            Ag.update()
            placecells.update()

    logger.debug(f'dt: {Ag.dt}')
    # # Ag.animate_trajectory(t_end=120, speed_up=2)

    length = placecells.get_history_arrays()["t"].shape[0]
    step_type = np.ones((length, 1))
    # Set the first element to 0
    step_type[0, 0] = 0

    observation = placecells.get_history_arrays()["firingrate"]

    import dm_env
    transitions = dm_env.TimeStep(
            reward=None, discount=None, observation=observation, step_type=step_type)
    # import jax
    # transitions = jax.tree_util.tree_map(lambda *arrs: np.vstack(arrs), *transitions)
    with dataset_path.open("wb") as fp:
        pickle.dump(transitions, fp)


    import ratinabox
    import joblib

    ratinabox.figure_directory = folder_dataset_path+'/figures'
    joblib.dump(placecells.params, folder_dataset_path+'/placecells_params.pkl')
    placecells.plot_rate_map(autosave=True,method="history")
    Ag.plot_trajectory(color="changing", autosave=True)
    
    joblib.dump(Ag.get_history_arrays()['pos'],folder_dataset_path+'/agent_positions.pkl')


if __name__ == "__main__":
    tyro.cli(main)
