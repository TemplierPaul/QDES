from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Tuple

import flax.linen as nn
import jax
import optax
from jax import numpy as jnp

from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdes.core.neuroevolution.buffers.buffer import QDTransition, ReplayBuffer
from qdax.core.neuroevolution.losses.td3_loss import make_td3_loss_fn
from qdax.core.neuroevolution.networks.networks import QModule
from qdax.environments.base_wrappers import QDEnv
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, Params, RNGKey

from qdax.core.emitters.qpg_emitter import QualityPGConfig, QualityPGEmitterState, QualityPGEmitter

@dataclass
class RLConfig(QualityPGConfig):
    # TD3 params
    replay_buffer_size: int = 1000000
    critic_hidden_layer_size: Tuple[int, ...] = (256, 256)
    critic_learning_rate: float = 3e-4
    actor_learning_rate: float = 3e-4
    policy_learning_rate: float = 1e-3
    noise_clip: float = 0.5
    policy_noise: float = 0.2
    discount: float = 0.99
    reward_scaling: float = 1.0
    batch_size: int = 256
    soft_tau_update: float = 0.005
    policy_delay: int = 2

class RLEmitterState(QualityPGEmitterState):
    """Contains training state for the learner."""

    critic_params: Params
    critic_optimizer_state: optax.OptState
    actor_params: Params
    actor_opt_state: optax.OptState
    target_critic_params: Params
    target_actor_params: Params
    replay_buffer: ReplayBuffer
    random_key: RNGKey
    steps: jnp.ndarray