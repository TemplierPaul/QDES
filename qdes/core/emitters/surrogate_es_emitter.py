from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Tuple
from typing import Callable, Tuple

import flax.linen as nn
import jax
import optax
from jax import numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_map

from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, Params, RNGKey
from qdes.core.emitters.vanilla_es_emitter import VanillaESConfig, VanillaESEmitterState, VanillaESEmitter, NoveltyArchive
from qdax.core.emitters.qpg_emitter import QualityPGConfig, QualityPGEmitterState, QualityPGEmitter
from qdes.core.rl_es_parts.es_utils import ESRepertoire, ESMetrics
from qdes.core.emitters.esrl_emitter import ESRLConfig, ESRLEmitterState, ESRLEmitter
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdes.core.emitters.test_gradients import TestGradientsEmitter, flatten_genotype

@dataclass
class SurrogateESConfig:
    """Configuration for ESRL Emitter"""

    es_config: VanillaESConfig
    rl_config: QualityPGConfig
    surrogate_omega: float = 0.6


class SurrogateESEmitter(TestGradientsEmitter):
    @property
    def config_string(self):
        s = self.es_emitter.config_string + " | " + self.rl_emitter.config_string
        if self._config.surrogate_omega > 0:
            s += f" | \u03C9 {self._config.surrogate_omega} ({self._config.rl_config.surrogate_batch})" # \u03C9 is omega
        return s

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def state_update(
        self,
        emitter_state: ESRLEmitterState,
        repertoire: ESRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> ESRLEmitterState:
        """Generate the gradient offspring for the next emitter call. Also
        update the novelty archive and generation count from current call.

        Chooses between ES and RL update.

        Params:
            emitter_state
            repertoire: unused
            genotypes: the genotypes of the offspring
            fitnesses: the fitnesses of the offspring
            descriptors: the descriptors of the offspring
            extra_scores: the extra scores of the offspring

        Returns:
            the updated emitter state
        """

        key, emitter_state = emitter_state.get_key()
        subkey, key = jax.random.split(key)
        
        # Choose between true eval and surrogate fitness
        # True: surrogate fitness
        # False: true fitness
        cond = jax.random.choice(
            subkey, 
            jnp.array([True, False]), 
            p=jnp.array([self._config.surrogate_omega, 1-self._config.surrogate_omega]))

        # cond = cond and emitter_state.rl_state.replay_buffer.size > self._config.rl_config.surrogate_batch

        # Do RL if the ES has done more steps than RL
        # cond = emitter_state.metrics.es_updates <= emitter_state.metrics.rl_updates

        # cond = self.choose_es_update(emitter_state, fitnesses, descriptors, extra_scores)
        old_center = emitter_state.es_state.offspring

        emitter_state, pop_extra_scores = jax.lax.cond(
            cond,
            self.surrogate_state_update,
            self.es_state_update,
            # Arguments to the two functions:
            emitter_state, 
            repertoire, 
            genotypes, 
            fitnesses, 
            descriptors, 
            extra_scores
        )

        offspring = emitter_state.es_state.offspring

        new_evaluations = self._config.es_config.sample_number * (1 - cond) # 0 if cond is True, sample_number if cond is False

        metrics = self.get_metrics(
            emitter_state,
            offspring,
            pop_extra_scores,
            fitnesses,
            random_key=key,
            new_evaluations=new_evaluations,
        )

        # Actor evaluation
        key, emitter_state = emitter_state.get_key()
        subkey, key = jax.random.split(key)

        actor = emitter_state.rl_state.actor_params
        actor_genome = flatten_genotype(actor)
        new_center_genome = flatten_genotype(offspring)
        actor_dist = jnp.linalg.norm(actor_genome - new_center_genome)

        angles = self.compute_angles(
            g1=offspring,
            g2=actor,
            center=old_center,
        )

        metrics = metrics.replace(
            actor_es_dist = actor_dist,
            es_step_norm = angles["v1_norm"],
            rl_step_norm = angles["v2_norm"],
            es_rl_cosine = angles["cosine_similarity"],
            es_rl_sign = angles["same_sign"],
        )

        actor_fitness, _ = self.multi_eval(actor, subkey)

        metrics = metrics.replace(
            actor_fitness=actor_fitness,
            # center_fitness=center_fitness,
        )

        emitter_state = emitter_state.set_metrics(metrics)
        emitter_state = emitter_state.set_key(key)

        return emitter_state