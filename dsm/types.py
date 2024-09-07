import typing
from typing import Callable, Hashable, Iterable, Literal, Mapping, NamedTuple, Protocol, TypeVar

import jax

Environment = Literal[
    "Pendulum-v1",
    "WindyGridWorld-v0",
    "WindyGridWorld-top-v0",
    "WindyGridWorld-bottom-v0",
    "Ratinabox-v0-xy",
    "Ratinabox-v0-pc-random",
    "Ratinabox-v0-pc-lowTH",
    "Ratinabox-v0-pc-highTH",
    "Ratinabox-v0-pc-teleport",
    "Ratinabox-v0-pc-goal",
    "MountainCarContinuous-v0",
]
Policy = Callable[[jax.random.KeyArray, jax.Array], tuple[jax.random.KeyArray, jax.Array]]


@typing.runtime_checkable
class DistanceFunction(Protocol):
    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        ...


@typing.runtime_checkable
class Kernel(Protocol):
    def __call__(self, zz: jax.Array) -> jax.Array:
        ...


@typing.runtime_checkable
class KernelFactory(Protocol):
    def __call__(self, *, bandwidths: tuple[float, ...]) -> Kernel:
        ...


PyTreeLeaf = TypeVar("PyTreeLeaf")
PyTree = PyTreeLeaf | Iterable["PyTree[PyTreeLeaf]"] | Mapping[Hashable, "PyTree[PyTreeLeaf]"]


class TransitionDataset(NamedTuple):
    step_type: jax.Array
    reward: jax.Array
    discount: jax.Array
    observation: jax.Array
