from __future__ import annotations
from qdes.core.emitters.vanilla_es_emitter import VanillaESConfig, VanillaESEmitterState, VanillaESEmitter

from dataclasses import dataclass
from functools import partial
from typing import Callable, Tuple

import flax
import jax
import jax.numpy as jnp
import optax

from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey

@dataclass
class OpenESConfig(VanillaESConfig):
    """Configuration for the ES or NSES emitter.

    Args:
        nses_emitter: if True, use NSES, if False, use ES
        sample_number: num of samples for gradient estimate
        sample_sigma: std to sample the samples for gradient estimate
        sample_mirror: if True, use mirroring sampling
        sample_rank_norm: if True, use normalisation
        adam_optimizer: if True, use ADAM, if False, use SGD
        learning_rate
        l2_coefficient: coefficient for regularisation
            novelty_nearest_neighbors
    """

    nses_emitter: bool = False
    sample_number: int = 1000
    sample_sigma: float = 0.02
    sample_mirror: bool = True
    sample_rank_norm: bool = True
    adam_optimizer: bool = True
    learning_rate: float = 0.01
    l2_coefficient: float = 0.02
    novelty_nearest_neighbors: int = 10
    actor_injection: bool = False


class OpenESEmitter(VanillaESEmitter):
    """
    Emitter allowing to reproduce an ES or NSES emitter with
    a passive archive. This emitter never sample from the reperoire.

    Uses OpenAI ES as optimizer.

    One can choose between ES and NSES by setting nses_emitter boolean.
    """

    @partial(
        jax.jit,
        static_argnames=("self", "scores_fn", "fitness_function"),
    )
    def _es_emitter(
        self,
        parent: Genotype,
        optimizer_state: optax.OptState,
        random_key: RNGKey,
        scores_fn: Callable[[Fitness, Descriptor], jnp.ndarray],
        fitness_function: Callable[[Genotype], RNGKey],
        actor: Genotype=None,
    ) -> Tuple[Genotype, optax.OptState, RNGKey]:
        """Main es component, given a parent and a way to infer the score from
        the fitnesses and descriptors fo its es-samples, return its
        approximated-gradient-generated offspring.

        Args:
            parent: the considered parent.
            scores_fn: a function to infer the score of its es-samples from
                their fitness and descriptors.
            random_key

        Returns:
            The approximated-gradients-generated offspring and a new random_key.
        """

        # Sampling noise
        total_sample_number = self._config.sample_number

        sample_noise, random_key = jax.lax.cond(
            self._config.sample_mirror,
            self._sample_mirror,
            self._sample,
            parent,
            random_key,
        )


        # Actor injection if needed in config and if actor is not None
        sample_noise = self._actor_injection(
            sample_noise,
            actor,
            parent,
        )

        # Applying noise
        # Repeat center
        samples = jax.tree_map(
            lambda x: jnp.repeat(x, total_sample_number, axis=0),
            parent,
        )
        # Add noise
        samples = jax.tree_map(
            lambda mean, noise: mean + self._config.sample_sigma * noise,
            samples,
            sample_noise,
        )

        # Evaluating samples
        fitnesses, descriptors, extra_scores, random_key = fitness_function(
            samples, random_key
        )
        extra_scores["population_fitness"] = fitnesses

        # Computing rank with normalisation
        scores = scores_fn(fitnesses, descriptors)


        ranking_indices = jnp.argsort(scores, axis=0) # Lowest fitness has rank 0
        ranks = jnp.argsort(ranking_indices, axis=0) 
        weights = (ranks / (total_sample_number - 1)) - 0.5

        weights = jax.tree_map(
            lambda x: jnp.reshape(
                jnp.repeat(weights.ravel(), x[0].ravel().shape[0], axis=0), x.shape
            ),
            sample_noise,
        )

        # Computing the gradients
        # Noise is multiplied by rank
        gradient = jax.tree_map(
            lambda noise, rank: jnp.multiply(noise, rank),
            sample_noise,
            weights,
        )
        # Gradients are summed over the sample dimension, and divided by sigma and the number of samples
        # Gradient is negated to match the direction of the optimizer
        gradient = jax.tree_map(
            lambda g, p: jnp.reshape(
                -jnp.sum(g, axis=0) / (total_sample_number * self._config.sample_sigma),
                p.shape,
            ),
            gradient,
            parent,
        )

        # Adding regularisation
        gradient = jax.tree_map(
            lambda g, p: g + self._config.l2_coefficient * p,
            gradient,
            parent,
        )

        # Applying gradients
        (offspring_update, optimizer_state) = self._optimizer.update(
            gradient, optimizer_state
        )
        offspring = optax.apply_updates(parent, offspring_update)

        return offspring, optimizer_state, random_key, extra_scores