from __future__ import annotations
from typing import Callable, List, Optional, Tuple, Union
import jax
import jax.numpy as jnp
from functools import partial
from brax.io import html

from qdax.types import Centroid, Descriptor, ExtraScores, Fitness, Genotype, RNGKey, Metrics

# from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
# from qdax.core.emitters.vanilla_es_emitter import VanillaESConfig, VanillaESEmitter
from qdax.core.map_elites import MAPElites
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire, get_cells_indices
# from qdax.utils.metrics import CSVLogger, default_qd_metrics
from qdax.core.emitters.emitter import Emitter, EmitterState

from dataclasses import dataclass, asdict
import flax
from flax.struct import dataclass as fdataclass

@fdataclass
class ESMetrics:
    es_updates: int=0
    surrogate_updates: int=0
    rl_updates: int=0
    evaluations: int=0
    actor_fitness: Fitness = -jnp.inf
    center_fitness: Fitness = -jnp.inf
    # fitness: Fitness = -jnp.inf
    # Population metrics
    pop_mean: Fitness = -jnp.inf
    pop_median: Fitness = -jnp.inf
    pop_std: Fitness = -jnp.inf
    pop_min: Fitness = -jnp.inf
    pop_max: Fitness = -jnp.inf
    # Center multi evals metrics
    center_mean: Fitness = -jnp.inf
    center_median: Fitness = -jnp.inf
    center_std: Fitness = -jnp.inf
    center_min: Fitness = -jnp.inf
    center_max: Fitness = -jnp.inf
    # CMAES metrics
    sigma: float = -jnp.inf
    eigen_change: float = -jnp.inf
    injection_norm: float = -jnp.inf

    ## ES + RL
    # Step size metrics
    rl_step_norm: float = -jnp.inf
    es_step_norm: float = -jnp.inf
    surrogate_step_norm: float = -jnp.inf
    # RL - ES metrics
    es_rl_cosine: float = -jnp.inf
    es_rl_sign: float = -jnp.inf
    actor_es_dist: float = -jnp.inf
    # surrogate - true fitness metrics
    surr_fit_cosine: float = -jnp.inf
    surr_fit_sign: float = -jnp.inf
    # Surrogate - RL metrics
    surr_rl_cosine: float = -jnp.inf
    surr_rl_sign: float = -jnp.inf
    # Tracking how far we are from the initial center
    es_dist: float = -jnp.inf
    rl_dist: float = -jnp.inf
    start_cos_sim: float = -jnp.inf

    # Spearman correlation between surrogate and true fitness
    spearmans_correlation: float = -jnp.inf
    spearmans_pvalue: float = -jnp.inf
    
    ## Canonical in CMAES
    canonical_step_norm: float = -jnp.inf
    # CMAES - Canonical metrics
    cma_canonical_cosine: float = -jnp.inf
    cma_canonical_sign: float = -jnp.inf
    # RL - Canonical metrics
    canonical_rl_cosine: float = -jnp.inf
    canonical_rl_sign: float = -jnp.inf
    
    # Spearman-based surrogate ES
    spearman_omega: float = -jnp.inf

class ESRepertoire(MapElitesRepertoire):
    """ A MapElitesRepertoire for ES that keeps the fitness of the last added ES center for logging
    """

    @classmethod
    def init_default(
        cls,
        genotype: Genotype,
        centroids: Centroid,
    ) -> ESRepertoire:
        """Initialize a Map-Elites repertoire with an initial population of
        genotypes. Requires the definition of centroids that can be computed
        with any method such as CVT or Euclidean mapping.

        Note: this function has been kept outside of the object MapElites, so
        it can be called easily called from other modules.

        Args:
            genotype: the typical genotype that will be stored.
            centroids: the centroids of the repertoire

        Returns:
            A repertoire filled with default values.
        """

        # get number of centroids
        num_centroids = centroids.shape[0]

        # default fitness is -inf
        default_fitnesses = -jnp.inf * jnp.ones(shape=num_centroids)

        # default genotypes is all 0
        default_genotypes = jax.tree_util.tree_map(
            lambda x: jnp.zeros(shape=(num_centroids,) + x.shape, dtype=x.dtype),
            genotype,
        )

        # default descriptor is all zeros
        default_descriptors = jnp.zeros_like(centroids)

        return cls(
            genotypes=default_genotypes,
            fitnesses=default_fitnesses,
            descriptors=default_descriptors,
            centroids=centroids,
        )

    @jax.jit
    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
        batch_of_extra_scores: Optional[ExtraScores] = None,
    ) -> ESRepertoire:
        """
        Add a batch of elements to the repertoire.

        Args:
            batch_of_genotypes: a batch of genotypes to be added to the repertoire.
                Similarly to the self.genotypes argument, this is a PyTree in which
                the leaves have a shape (batch_size, num_features)
            batch_of_descriptors: an array that contains the descriptors of the
                aforementioned genotypes. Its shape is (batch_size, num_descriptors)
            batch_of_fitnesses: an array that contains the fitnesses of the
                aforementioned genotypes. Its shape is (batch_size,)
            batch_of_extra_scores: unused tree that contains the extra_scores of
                aforementioned genotypes.

        Returns:
            The updated MAP-Elites repertoire.
        """

        batch_of_indices = get_cells_indices(batch_of_descriptors, self.centroids)
        batch_of_indices = jnp.expand_dims(batch_of_indices, axis=-1)
        batch_of_fitnesses = jnp.expand_dims(batch_of_fitnesses, axis=-1)

        num_centroids = self.centroids.shape[0]

        # get fitness segment max
        best_fitnesses = jax.ops.segment_max(
            batch_of_fitnesses,
            batch_of_indices.astype(jnp.int32).squeeze(axis=-1),
            num_segments=num_centroids,
        )

        cond_values = jnp.take_along_axis(best_fitnesses, batch_of_indices, 0)

        # put dominated fitness to -jnp.inf
        batch_of_fitnesses = jnp.where(
            batch_of_fitnesses == cond_values, x=batch_of_fitnesses, y=-jnp.inf
        )

        # get addition condition
        repertoire_fitnesses = jnp.expand_dims(self.fitnesses, axis=-1)
        current_fitnesses = jnp.take_along_axis(
            repertoire_fitnesses, batch_of_indices, 0
        )
        addition_condition = batch_of_fitnesses > current_fitnesses

        # assign fake position when relevant : num_centroids is out of bound
        batch_of_indices = jnp.where(
            addition_condition, x=batch_of_indices, y=num_centroids
        )

        # create new repertoire
        new_repertoire_genotypes = jax.tree_util.tree_map(
            lambda repertoire_genotypes, new_genotypes: repertoire_genotypes.at[
                batch_of_indices.squeeze(axis=-1)
            ].set(new_genotypes),
            self.genotypes,
            batch_of_genotypes,
        )

        # compute new fitness and descriptors
        new_fitnesses = self.fitnesses.at[batch_of_indices.squeeze(axis=-1)].set(
            batch_of_fitnesses.squeeze(axis=-1)
        )
        new_descriptors = self.descriptors.at[batch_of_indices.squeeze(axis=-1)].set(
            batch_of_descriptors
        )

        return ESRepertoire(
            genotypes=new_repertoire_genotypes,
            fitnesses=new_fitnesses,
            descriptors=new_descriptors,
            centroids=self.centroids,
        )

    def record_video(self, env, policy_network):
        """Record a video of the best individual in the repertoire.
        """
        best_idx = jnp.argmax(self.fitnesses)

        elite = jax.tree_util.tree_map(
            lambda x: x[best_idx],
            self.genotypes
        )

        jit_env_reset = jax.jit(env.reset)
        jit_env_step = jax.jit(env.step)
        jit_inference_fn = jax.jit(policy_network.apply)

        rollout = []
        rng = jax.random.PRNGKey(seed=1)
        state = jit_env_reset(rng=rng)
        while not state.done:
            rollout.append(state)
            action = jit_inference_fn(elite, state.obs)
            state = jit_env_step(state, action)

        return html.render(env.sys, [s.qp for s in rollout[:500]])

    
class ES(MAPElites):
    """ Map-Elite structure to run a standalone ES
    """

    # @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        init_genotypes: Genotype,
        centroids: Centroid,
        random_key: RNGKey,
    ) -> Tuple[ESRepertoire, Optional[EmitterState], RNGKey]:
        """
        Initialize a Map-Elites repertoire with an initial population of genotypes.
        Requires the definition of centroids that can be computed with any method
        such as CVT or Euclidean mapping.

        Args:
            init_genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            centroids: tesselation centroids of shape (batch_size, num_descriptors)
            random_key: a random key used for stochastic operations.

        Returns:
            An initialized MAP-Elite repertoire with the initial state of the emitter,
            and a random key.
        """
        # score initial genotypes
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            init_genotypes, random_key
        )

        # init the repertoire
        repertoire = ESRepertoire.init(
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
            extra_scores=extra_scores,
        )

        # get initial state of the emitter
        emitter_state, random_key = self._emitter.init(
            init_genotypes=init_genotypes, random_key=random_key
        )

        # print("Before ES init state_update", emitter_state.rl_state.replay_buffer.current_position)

        # update emitter state
        # emitter_state = self._emitter.state_update(
        #     emitter_state=emitter_state,
        #     repertoire=repertoire,
        #     genotypes=init_genotypes,
        #     fitnesses=fitnesses,
        #     descriptors=descriptors,
        #     extra_scores=extra_scores,
        # )

        # print("After ES init state_update", emitter_state.rl_state.replay_buffer.current_position)

        return repertoire, emitter_state, random_key
    
    @partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState], Metrics, RNGKey]:
        """
        Performs one iteration of the MAP-Elites algorithm.
        1. A batch of genotypes is sampled in the repertoire and the genotypes
            are copied.
        2. The copies are mutated and crossed-over
        3. The obtained offsprings are scored and then added to the repertoire.


        Args:
            repertoire: the MAP-Elites repertoire
            emitter_state: state of the emitter
            random_key: a jax PRNG random key

        Returns:
            the updated MAP-Elites repertoire
            the updated (if needed) emitter state
            metrics about the updated repertoire
            a new jax PRNG key
        """
        # generate offsprings with the emitter
        genotypes, random_key = self._emitter.emit(
            repertoire, emitter_state, random_key
        )
        # scores the offsprings
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            genotypes, random_key
        )

        # add genotypes in the repertoire
        repertoire = repertoire.add(genotypes, descriptors, fitnesses, extra_scores)

        # update the metrics
        metrics = self._metrics_function(repertoire, emitter_state)

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        return repertoire, emitter_state, metrics, random_key

def default_es_metrics(repertoire: ESRepertoire, emitter_state: EmitterState, qd_offset: float) -> Metrics:
    """Compute the usual QD metrics that one can retrieve
    from a MAP Elites repertoire.

    Args:
        repertoire: a MAP-Elites repertoire
        qd_offset: an offset used to ensure that the QD score
            will be positive and increasing with the number
            of individuals.

    Returns:
        a dictionary containing the QD score (sum of fitnesses
            modified to be all positive), the max fitness of the
            repertoire, the coverage (number of niche filled in
            the repertoire).
    """

    # QD metrics from archive
    archive_metrics = {}
    repertoire_empty = repertoire.fitnesses == -jnp.inf
    archive_metrics["qd_score"] = jnp.sum(repertoire.fitnesses, where=~repertoire_empty)
    archive_metrics["qd_score"] += qd_offset * jnp.sum(1.0 - repertoire_empty)
    archive_metrics["coverage"] = 100 * jnp.mean(1.0 - repertoire_empty)
    archive_metrics["max_fitness"] = jnp.max(repertoire.fitnesses)

    # ES metrics
    metrics = emitter_state.metrics
    # Turn into a dict
    metrics = metrics.__dict__.copy()

    # Merge
    metrics.update(archive_metrics)
    return metrics