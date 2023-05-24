from functools import partial
from typing import Callable, Optional, Tuple

import flax
import jax
import jax.numpy as jnp

from dataclasses import dataclass
from qdax.types import Fitness, Genotype, Mask, RNGKey
from qdax.core.cmaes import CMAESState, CMAES

# @dataclass
# class CMAESConfig:
    