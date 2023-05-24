from __future__ import annotations
from qdes.core.emitters.vanilla_es_emitter import VanillaESConfig, VanillaESEmitterState, VanillaESEmitter, NoveltyArchive
from qdes.core.rl_es_parts.es_utils import ESRepertoire, ESMetrics

from dataclasses import dataclass
from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import optax
from qdax.core.emitters.emitter import EmitterState

from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey

@dataclass
class RandomConfig(VanillaESConfig):
    """Configuration for the random search emitter.

    Args:
        nses_emitter: if True, use NSES, if False, use ES
        sample_number: num of samples for gradient estimate
        sample_sigma: std to sample the samples for gradient estimate
    """

    nses_emitter: bool = False
    sample_number: int = 1000
    sample_sigma: float = 0.02


class RandomEmitter(VanillaESEmitter):
    '''Random search emitter.'''

    def __init__(
        self,
        config: VanillaESConfig,
        scoring_fn: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        total_generations: int = 1,
        num_descriptors: int = 2,
    ) -> None:
        """Initialise the ES or NSES emitter.
        WARNING: total_generations and num_descriptors are required for NSES.

        Args:
            config: algorithm config
            scoring_fn: used to evaluate the samples for the gradient estimate.
            total_generations: total number of generations for which the
                emitter will run, allow to initialise the novelty archive.
            num_descriptors: dimension of the descriptors, used to initialise
                the empty novelty archive.
        """
        raise NotImplementedError("Random search emitter not implemented yet.")
        self._config = config
        self._scoring_fn = scoring_fn
        self._total_generations = total_generations
        self._num_descriptors = num_descriptors

        # Actor injection
        self._actor_injection = lambda x, a, p: x

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> Tuple[VanillaESEmitterState, RNGKey]:
        """Initializes the emitter state.

        Args:
            init_genotypes: The initial population.
            random_key: A random key.

        Returns:
            The initial state of the VanillaESEmitter, a new random key.
        """
        # Initialisation requires one initial genotype
        if jax.tree_util.tree_leaves(init_genotypes)[0].shape[0] > 1:
            init_genotypes = jax.tree_util.tree_map(
                lambda x: x[0],
                init_genotypes,
            )

        # Create empty Novelty archive
        novelty_archive = NoveltyArchive.init(
            self._total_generations, self._num_descriptors
        )

        return (
            VanillaESEmitterState(
                offspring=init_genotypes,
                generation_count=0,
                novelty_archive=novelty_archive,
                optimizer_state=None,
                random_key=random_key,
                initial_center=init_genotypes,
            ),
            random_key,
        )

    @partial(
        jax.jit,
        static_argnames=("self", "scores_fn"),
    )
    def _es_emitter(
        self,
        parent: Genotype,
        optimizer_state: optax.OptState,
        random_key: RNGKey,
        scores_fn: Callable[[Fitness, Descriptor], jnp.ndarray],
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

        random_key, subkey = jax.random.split(random_key)

        # Sampling mirror noise
        sample_number = self._config.sample_number 
        # Sampling noise
        sample_number = sample_number 
        sample_noise = jax.tree_map(
            lambda x: jax.random.normal(
                key=subkey,
                shape=jnp.repeat(x, sample_number, axis=0).shape,
            ),
            parent,
        )

        # Applying noise
        samples = jax.tree_map(
            lambda x: jnp.repeat(x, sample_number, axis=0),
            parent,
        )
        samples = jax.tree_map(
            lambda mean, noise: mean + self._config.sample_sigma * noise,
            samples,
            sample_noise,
        )

        # Evaluating samples
        fitnesses, descriptors, extra_scores, random_key = self._scoring_fn(
            samples, random_key
        )

        extra_scores["population_fitness"] = fitnesses

        # Get the one with highest fitness
        best_index = jnp.argmax(fitnesses)
        
        # Get the best sample
        offspring = jax.tree_map(
            lambda x: jnp.expand_dims(x[best_index], axis=0),
            samples,
        )
        
        return offspring, optimizer_state, random_key, extra_scores