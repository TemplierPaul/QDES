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
from jax.tree_util import tree_flatten, tree_unflatten, tree_map

@dataclass
class CanonicalESConfig(VanillaESConfig):
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
    canonical_mu: int = int(sample_number / 2)
    sample_sigma: float = 0.02
    # learning_rate: float = 0.01
    novelty_nearest_neighbors: int = 10
    actor_injection: bool = False
    injection_clipping: bool = False

class CanonicalESEmitterState(VanillaESEmitterState):
    """Emitter State for the ES or NSES emitter.

    Args:
        optimizer_state: current optimizer state
        offspring: offspring generated through gradient estimate
        generation_count: generation counter used to update the novelty archive
        novelty_archive: used to compute novelty for explore
        random_key: key to handle stochastic operations
    """

    offspring: Genotype
    generation_count: int
    novelty_archive: NoveltyArchive
    random_key: RNGKey
    optimizer_state: optax.OptState = None # Not used by canonical ES
    initial_center: Genotype = None
    metrics: ESMetrics = ESMetrics()


class CanonicalESEmitter(VanillaESEmitter):
    """
    Emitter allowing to reproduce an ES or NSES emitter with
    a passive archive. This emitter never sample from the reperoire.

    Uses OpenAI ES as optimizer.

    One can choose between ES and NSES by setting nses_emitter boolean.
    """
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
        self._config = config
        self._scoring_fn = scoring_fn
        self._total_generations = total_generations
        self._num_descriptors = num_descriptors

        # Actor injection
        if self._config.actor_injection:
            print(f"Doing actor injection x {self._config.nb_injections}")
            self._actor_injection = self._inject_actor
        else:
            print("Not doing actor injection")
            def no_injection(sample_noise, actor, parent):
                # Applying noise
                networks = jax.tree_map(
                    lambda x: jnp.repeat(x, self._config.sample_number, axis=0),
                    parent,
                )
                networks = jax.tree_map(
                    lambda mean, noise: mean + self._config.sample_sigma * noise,
                    networks,
                    sample_noise,
                )

                norm = -jnp.inf

                return sample_noise, networks, norm
            
            self._actor_injection = no_injection

        # Add a wrapper to the scoring function to handle the surrogate data
        extended_scoring = lambda networks, random_key, extra: self._scoring_fn(
            networks, random_key)

        self._es_emitter = partial(
            self._base_es_emitter, 
            fitness_function=extended_scoring,
            surrogate_data=None,
        )
        self._es_emitter = partial(
            jax.jit,
            static_argnames=("scores_fn"),
        )(self._es_emitter)

        self.tree_def = None
        self.layer_sizes = None
        self.split_indices = None

        self.c_y = jnp.inf

    @property
    def config_string(self):
        """Returns a string describing the config."""
        s = f"Canonical {self._config.sample_number} "
        s += f"- \u03C3 {self._config.sample_sigma} "
        # learning rate
        # s += f"- lr {self._config.learning_rate} "
        if self._config.actor_injection:
            s += f"| AI {self._config.nb_injections}"
            if self._config.injection_clipping:
                s += " (clip)"
        return s


    # @partial(
    #     jax.jit,
    #     static_argnames=("self",),
    # )
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

        metrics = ESMetrics(
            es_updates=0,
            rl_updates=0,
            evaluations=0,
            actor_fitness=-jnp.inf,
            center_fitness=-jnp.inf,
        )

        flat_variables, tree_def = tree_flatten(init_genotypes)
        self.layer_shapes = [x.shape[1:] for x in flat_variables]
        print("layer_shapes", self.layer_shapes)

        sizes = [x.size for x in flat_variables]
        sizes = jnp.array(sizes)

        print("sizes", sizes)

        vect = jnp.concatenate([jnp.ravel(x) for x in flat_variables])
        n = len(vect)
        # Scaling for injection
        # sqrt(n) + 2n/(n + 2)
        if self._config.injection_clipping:
            self.c_y = jnp.sqrt(n) + 2 * n / (n + 2)

        self.tree_def = tree_def
        self.layer_sizes = sizes.tolist()
        print("layer_sizes", self.layer_sizes)
        self.split_indices = jnp.cumsum(jnp.array(self.layer_sizes))[:-1].tolist()
        print("split_indices", self.split_indices)


        return (
            CanonicalESEmitterState(
                offspring=init_genotypes,
                generation_count=0,
                novelty_archive=novelty_archive,
                random_key=random_key,
                initial_center=init_genotypes,
                metrics=metrics,
            ),
            random_key,
        )
    
    @partial(
        jax.jit,
        static_argnames=("self"),
    )
    def _inject_actor(
        self, 
        sample_noise: Genotype,
        actor: Genotype,
        parent: Genotype,
    ) -> Genotype:
        """
        Replace the last genotype of the sample_noise by the actor minus the parent.
        """
        # parent = jax.tree_util.tree_map(
        #     lambda x: x[0],
        #     parent,
        # )
        # print("actor shape", jax.tree_map(lambda x: x.shape, actor))
        # print("parent shape", jax.tree_map(lambda x: x.shape, parent))
        # print("sample_noise shape", jax.tree_map(lambda x: x.shape, sample_noise))

        sigma = self._config.sample_sigma
        x_actor = self.flatten(actor)
        # print("x_actor shape", x_actor.shape)
        x_parent = self.flatten(parent)
        # print("x_parent shape", x_parent.shape)
        y_actor = (x_actor - x_parent) / sigma

        norm = jnp.linalg.norm(y_actor)
        # alpha clip self.c_y / norm to 1
        norm = jnp.minimum(1, self.c_y / norm)
        normed_y_actor = norm * y_actor
        normed_y_net = self.unflatten(normed_y_actor)
        # Add 1 dimension
        normed_y_net = jax.tree_map(
            lambda x: x[None, ...],
            normed_y_net,
        )

        # Applying noise
        networks = jax.tree_map(
            lambda x: jnp.repeat(x, self._config.sample_number, axis=0),
            parent,
        )
        # print("networks shape", jax.tree_map(lambda x: x.shape, networks))
        networks = jax.tree_map(
            lambda mean, noise: mean + self._config.sample_sigma * noise,
            networks,
            sample_noise,
        )
        # print("networks shape", jax.tree_map(lambda x: x.shape, networks))

        # Repeat actor
        actor = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x[None, ...], self._config.nb_injections, axis=0),
            actor,
        )
        # print("repeated actor shape", jax.tree_map(lambda x: x.shape, actor))

        # Replace the n last one, with n = self._config.nb_injections
        networks = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x[:-self._config.nb_injections], y], axis=0),
            networks,
            actor,
        )

        # replace actor in sample_noise by scaled_actor
        sample_noise = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x[:-self._config.nb_injections], y], axis=0),
            sample_noise,
            normed_y_net,
        )

        return sample_noise, networks, norm

    @partial(
        jax.jit,
        static_argnames=("self", "scores_fn", "fitness_function"),
    )
    def _base_es_emitter(
        self,
        parent: Genotype,
        optimizer_state: optax.OptState,
        random_key: RNGKey,
        scores_fn: Callable[[Fitness, Descriptor], jnp.ndarray],
        fitness_function: Callable[[Genotype], RNGKey],
        surrogate_data = None,
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
        sample_noise = jax.tree_map(
            lambda x: jax.random.normal(
                key=subkey,
                shape=jnp.repeat(x, sample_number, axis=0).shape,
            ),
            parent,
        )

        # Actor injection if needed in config and if actor is not None
        gradient_noise, networks, norm_factor = self._actor_injection(
            sample_noise,
            actor,
            parent,
        )

        # Evaluating samples
        fitnesses, descriptors, extra_scores, random_key = fitness_function(
            networks, random_key, surrogate_data
        )

        extra_scores["injection_norm"] = norm_factor

        extra_scores["population_fitness"] = fitnesses

        # Computing rank, with or without normalisation
        scores = scores_fn(fitnesses, descriptors)

        ranking_indices = jnp.argsort(scores, axis=0) 
        ranks = jnp.argsort(ranking_indices, axis=0) 
        ranks = self._config.sample_number - ranks # Inverting the ranks
        
        mu = self._config.canonical_mu # Number of parents

        weights = jnp.where(ranks <= mu, jnp.log(mu+0.5) - jnp.log(ranks), 0) 
        weights /= jnp.sum(weights) # Normalizing the weights

        # Reshaping rank to match shape of genotype_noise
        weights = jax.tree_map(
            lambda x: jnp.reshape(
                jnp.repeat(weights.ravel(), x[0].ravel().shape[0], axis=0), x.shape
            ),
            gradient_noise,
        )

        # Computing the update
        # Noise is multiplied by rank
        gradient = jax.tree_map(
            lambda noise, rank: jnp.multiply(noise, rank),
            gradient_noise,
            weights,
        )
        # Noise is summed over the sample dimension and multiplied by sigma
        gradient = jax.tree_map(
            lambda x: jnp.reshape(x, (sample_number, -1)),
            gradient,
        )
        gradient = jax.tree_map(
            lambda g, p: jnp.reshape(
                jnp.sum(g, axis=0) * self._config.sample_sigma,
                p.shape,
            ),
            gradient,
            parent,
        )

        offspring = optax.apply_updates(parent, gradient)

        return offspring, optimizer_state, random_key, extra_scores

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def state_update(
        self,
        emitter_state: VanillaESEmitterState,
        repertoire: ESRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> VanillaESEmitterState:
        """Generate the gradient offspring for the next emitter call. Also
        update the novelty archive and generation count from current call.

        Args:
            emitter_state: current emitter state.
            repertoire: unused.
            genotypes: the genotypes of the batch of emitted offspring.
            fitnesses: the fitnesses of the batch of emitted offspring.
            descriptors: the descriptors of the emitted offspring.
            extra_scores: a dictionary with other values outputted by the
                scoring function.

        Returns:
            The modified emitter state.
        """

        assert jax.tree_util.tree_leaves(genotypes)[0].shape[0] == 1, (
            "ERROR: Vanilla-ES generates 1 offspring per generation, "
            + "batch_size should be 1, the inputed batch has size:"
            + str(jax.tree_util.tree_leaves(genotypes)[0].shape[0])
        )

        # Updating novelty archive
        novelty_archive = emitter_state.novelty_archive.update(descriptors)

        # Define scores for es process
        def scores(fitnesses: Fitness, descriptors: Descriptor) -> jnp.ndarray:
            if self._config.nses_emitter:
                return novelty_archive.novelty(
                    descriptors, self._config.novelty_nearest_neighbors
                )
            else:
                return fitnesses

        # Run es process
        offspring, optimizer_state, random_key, extra_scores = self._es_emitter(
            parent=genotypes,
            random_key=emitter_state.random_key,
            scores_fn=scores,
            optimizer_state=emitter_state.optimizer_state,
            # fitness_function=self._scoring_fn,
        )

        metrics = self.get_metrics(
            emitter_state,
            offspring,
            extra_scores,
            fitnesses,
            new_evaluations=self._config.sample_number,
            random_key=random_key,
        )

        return emitter_state.replace(  # type: ignore
            optimizer_state=optimizer_state,
            offspring=offspring,
            novelty_archive=novelty_archive,
            generation_count=emitter_state.generation_count + 1,
            random_key=random_key,
            metrics=metrics,
        )
    

