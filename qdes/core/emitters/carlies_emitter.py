from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Tuple
from typing import Callable, Tuple


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
from qdes.core.emitters.vanilla_es_emitter import VanillaESConfig, VanillaESEmitterState, VanillaESEmitter, NoveltyArchive
from qdax.core.emitters.qpg_emitter import QualityPGConfig, QualityPGEmitterState, QualityPGEmitter
from qdes.core.rl_es_parts.es_utils import ESRepertoire, ESMetrics
from qdax.core.cmaes import CMAESState
from qdes.core.emitters.esrl_emitter import ESRLConfig, ESRLEmitterState, ESRLEmitter



class CARLIES(ESRLEmitter):
    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def choose_es_update(self, 
        emitter_state: ESRLEmitterState,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> bool:
        """Choose between ES and RL update with probability 0.5.

        Params:
            emitter_state
            fitnesses: the fitnesses of the offspring
            descriptors: the descriptors of the offspring
            extra_scores: the extra scores of the offspring

        Returns:
            True if ES update False else
        """
        cond = emitter_state.metrics.es_updates <= emitter_state.metrics.rl_updates

        return cond