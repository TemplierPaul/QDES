from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Tuple

import flax
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
import optax

from qdes.core.emitters.vanilla_es_emitter import VanillaESConfig, VanillaESEmitterState, VanillaESEmitter, NoveltyArchive
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey
from qdes.core.rl_es_parts.es_utils import ESRepertoire, ESMetrics
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.cmaes import CMAES, CMAESState

# @partial(
#         jax.jit,
#         static_argnames=("c","x"),
#     )
# def alpha_clip(c, x):
#     return jnp.min([1, c/x])


@dataclass
class MonoCMAESConfig(VanillaESConfig):
    """Configuration for the CMAES with mono solution emitter."""
    nses_emitter: bool = False
    sample_number: int = 1000
    sample_sigma: float = 1e-3
    actor_injection: bool = False


class MonoCMAESState(VanillaESEmitterState):
    """Emitter State for the ES or NSES emitter.

    Args:
        optimizer_state: current optimizer state
        offspring: offspring generated through gradient estimate
        generation_count: generation counter used to update the novelty archive
        novelty_archive: used to compute novelty for explore
        random_key: key to handle stochastic operations
    """

    optimizer_state: CMAESState
    canonical_update: Genotype = None


class MonoCMAESEmitter(VanillaESEmitter):
    """
    Emitter allowing to reproduce an ES or NSES emitter with
    a passive archive. This emitter never sample from the reperoire.

    Uses CMAES as optimizer.
    """
    
    def __init__(
        self,
        config: MonoCMAESConfig,
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
        
        # define a CMAES instance
        self._cmaes = None
        self.c_y = jnp.inf
        self.tree_def = None
        self.layer_sizes = None
        self.split_indices = None

        # Actor injection not available yet
        # if self._config.actor_injection:
        #     raise NotImplementedError("Actor injection not available for CMAES yet.")

        if self._config.actor_injection:
            print(f"Doing actor injection x {self._config.nb_injections}")
            self._actor_injection = self._inject_actor
        else:
            print("Not doing actor injection")
            def no_injection(
                    genomes: Genotype,
                    actor: Genotype,
                    invsqrt_cov: jnp.ndarray,
                    center: Genotype,
                    sigma: float,
                ) -> Tuple[Genotype, Genotype, float]:
                
                networks = jax.vmap(self.unflatten)(genomes)
                norm = -jnp.inf

                return genomes, networks, norm
            
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

    @property
    def config_string(self):
        """Returns a string describing the config."""
        s = f"CMAES {self._config.sample_number} "
        s += f"- \u03C3 {self._config.sample_sigma} "
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
    ) -> Tuple[MonoCMAESState, RNGKey]:
        """Initializes the emitter state.

        Args:
            init_genotypes: The initial population.
            random_key: A random key.

        Returns:
            The initial state of the VanillaESEmitter, a new random key.
        """
        # Initialisation requires one initial genotype
        # print("CMAES init_genotypes", jax.tree_map(lambda x: x.shape, init_genotypes))

        if jax.tree_util.tree_leaves(init_genotypes)[0].shape[0] > 1:
            init_genotypes = jax.tree_util.tree_map(
                lambda x: x[0],
                init_genotypes,
            )

        flat_variables, tree_def = tree_flatten(init_genotypes)
        self.layer_shapes = [x.shape[1:] for x in flat_variables]
        # print("layer_shapes", self.layer_shapes)

        vect = jnp.concatenate([jnp.ravel(x) for x in flat_variables])
        sizes = [x.size for x in flat_variables]
        sizes = jnp.array(sizes)

        # print("sizes", sizes)

        self.tree_def = tree_def
        self.layer_sizes = sizes.tolist()
        # print("layer_sizes", self.layer_sizes)
        self.split_indices = jnp.cumsum(jnp.array(self.layer_sizes))[:-1].tolist()
        # print("split_indices", self.split_indices)

        genotype_dim = len(vect)
        # print("genotype_dim", genotype_dim)

        self._cmaes = CMAES(
            population_size=self._config.sample_number,
            search_dim=genotype_dim,
            # no need for fitness function in that specific case
            fitness_function=None,  # type: ignore
            num_best= self._config.sample_number // 2,
            init_sigma= self._config.sample_sigma,
            mean_init=None,  # will be init at zeros in cmaes
            bias_weights=True,
            delay_eigen_decomposition=True,
        )

        # Scaling for injection
        n = genotype_dim
        # sqrt(n) + 2n/(n + 2)
        if self._config.injection_clipping:
            self.c_y = jnp.sqrt(n) + 2 * n / (n + 2)

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

        return (
            MonoCMAESState(
                optimizer_state=self._cmaes.init(),
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
        genomes: Genotype,
        actor: Genotype,
        invsqrt_cov: jnp.ndarray,
        center: Genotype,
        sigma: float,
    ) -> Tuple[Genotype, Genotype, float]:
        actor = jax.tree_util.tree_map(
            lambda x: x[0],
            actor,
        )
        # print("Actor", jax.tree_map(lambda x: x.shape, actor))

        # Get actor genome
        x_actor = self.flatten(actor)
        y_actor = (x_actor - center) / sigma

        ## Scale the actor genome
        # Injecting External Solutions Into CMA-ES, https://arxiv.org/pdf/1110.4181.pdf
        Cy = invsqrt_cov @ y_actor
        # Get normalizing factor
        norm = jnp.linalg.norm(Cy)

        # alpha clip self.c_y / norm to 1
        norm = jnp.minimum(1, self.c_y / norm)
        normed_y_actor = norm * y_actor
        normed_x_actor = center + sigma * normed_y_actor

        # Population networks
        networks = jax.vmap(self.unflatten)(genomes)
        # print("Networks", jax.tree_map(lambda x: x.shape, networks))

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

        # print("After injection", jax.tree_map(lambda x: x.shape, networks))

        # Replace the n last one, with n = self._config.nb_injections
        genomes = genomes.at[-self._config.nb_injections:].set(normed_x_actor)
        # print("After injection", genomes.shape)

        return genomes, networks, norm

    
    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit(
        self,
        repertoire: ESRepertoire,
        emitter_state: VanillaESEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """Return the offspring generated through gradient update.

        Params:
            repertoire: unused
            emitter_state
            random_key: a jax PRNG random key

        Returns:
            a new gradient offspring
            a new jax PRNG key
        """

        offspring_genome = emitter_state.optimizer_state.mean
        offspring = self.unflatten(offspring_genome)

        offspring = jax.tree_map(
            lambda x: jnp.expand_dims(x, axis=0), 
            offspring
        )

        # print("Init offspring", jax.tree_map(lambda x: x.shape, offspring))

        return offspring, random_key
    

    # @partial(
    #     jax.jit,
    #     static_argnames=("self", "scores_fn"),
    # )
    def _base_es_emitter(
        self,
        parent: Genotype,
        optimizer_state: CMAESState,
        random_key: RNGKey,
        scores_fn: Callable[[Fitness, Descriptor], jnp.ndarray],
        fitness_function: Callable[[Genotype], RNGKey],
        surrogate_data = None,
        actor: Genotype=None,
    ) -> Tuple[Genotype, CMAESState, RNGKey]:
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
        old_eigen = optimizer_state.eigenvalues

        random_key, subkey = jax.random.split(random_key)
        # print("Parent", jax.tree_map(lambda x: x.shape, parent))

        # Parent genome
        parent_genome = self.flatten(parent)
        # print("parent_genome", parent_genome.shape)

        genomes = jax.random.multivariate_normal(
                key=subkey,
                shape=(self._config.sample_number,),
                mean=parent_genome,
                # Idendity matrix
                # cov=jnp.eye(parent_genome.shape[0])
                cov=(optimizer_state.sigma**2) * optimizer_state.cov_matrix
        )

        # print("genomes", genomes.shape)
        
        # Turn each sample into a network
        # networks = jax.vmap(self.unflatten)(genomes)
        invsqrt_cov = optimizer_state.invsqrt_cov
        genomes, networks, norm_factor = self._actor_injection(
            genomes,
            actor,
            invsqrt_cov,
            center=parent_genome,
            sigma=optimizer_state.sigma,
            )
        
        # print("Population", jax.tree_map(lambda x: x.shape, networks))
        # print("networks", networks.shape)
        
        # Evaluating genomes
        fitnesses, descriptors, extra_scores, random_key = fitness_function(
            networks, random_key, surrogate_data
        )

        # print("fitnesses", fitnesses)
        # print("descriptors", descriptors is not None)

        extra_scores["injection_norm"] = norm_factor

        extra_scores["population_fitness"] = fitnesses

        # Computing rank with normalisation
        scores = scores_fn(fitnesses, descriptors)

        # print("scores", scores.shape)

        # Compute the canonical update
        # canonical_update = self._canonical_update(
        #     parent = parent,
        #     genotypes = networks,
        #     fitnesses = scores,
        #     )
        # extra_scores["canonical_update"] = (parent, canonical_update)


        # Sort genomes by scores (descending order)
        idx_sorted = jnp.argsort(-scores)

        sorted_candidates = genomes[idx_sorted[: self._cmaes._num_best]]

        new_cmaes_state = self._cmaes.update_state(
            optimizer_state, 
            sorted_candidates)

        new_eigen = new_cmaes_state.eigenvalues
        # Norm of the eigenvalues change
        extra_scores["eigen_change"] = jnp.linalg.norm(new_eigen - old_eigen)

        # print("CMA state updated")

        offspring_genome = new_cmaes_state.mean
        offspring = self.unflatten(offspring_genome)
        # print("offspring", offspring)

        # print("Offspring", jax.tree_map(lambda x: x.shape, offspring))


        offspring = jax.tree_map(
            lambda x: jnp.expand_dims(x, axis=0), 
            offspring
        )

        # print("Expanded", jax.tree_map(lambda x: x.shape, offspring))
        return offspring, new_cmaes_state, random_key, extra_scores

    # @partial(
    #     jax.jit,
    #     static_argnames=("self"),
    # )
    # def _canonical_update(self, 
    #     parent: Genotype,
    #     genotypes: Genotype,
    #     fitnesses: Fitness,
    #     ) -> Genotype:
    #     """
    #     Simulate the canonical ES update from a population of genotypes and
    #     their fitnesses.
    #     """

    #     # print("Canonical update simulation")

    #     ranking_indices = jnp.argsort(fitnesses, axis=0) 
    #     ranks = jnp.argsort(ranking_indices, axis=0) 
    #     ranks = self._config.sample_number - ranks # Inverting the ranks
        
    #     mu = self._cmaes._num_best # Number of parents

    #     weights = jnp.where(ranks <= mu, jnp.log(mu+0.5) - jnp.log(ranks), 0) 
    #     weights /= jnp.sum(weights) # Normalizing the weights

    #     # Get noise from population and parent
    #     gradient_noise = jax.tree_map(
    #         lambda x, p: x - p,
    #         genotypes,
    #         parent,
    #     )

    #     # Reshaping rank to match shape of genotype_noise
    #     weights = jax.tree_map(
    #         lambda x: jnp.reshape(
    #             jnp.repeat(weights.ravel(), x[0].ravel().shape[0], axis=0), x.shape
    #         ),
    #         gradient_noise,
    #     )

    #     # Computing the update
    #     # Noise is multiplied by rank
    #     gradient = jax.tree_map(
    #         lambda noise, rank: jnp.multiply(noise, rank),
    #         gradient_noise,
    #         weights,
    #     )
    #     # Noise is summed over the sample dimension and multiplied by sigma
    #     gradient = jax.tree_map(
    #         lambda x: jnp.reshape(x, (self._config.sample_number, -1)),
    #         gradient,
    #     )
    #     gradient = jax.tree_map(
    #         lambda g, p: jnp.reshape(
    #             jnp.sum(g, axis=0) * self._config.sample_sigma,
    #             p.shape,
    #         ),
    #         gradient,
    #         parent,
    #     )

    #     offspring = optax.apply_updates(parent, gradient)

    #     return offspring

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def get_metrics(
            self, 
            emitter_state: VanillaESEmitterState,
            offspring: Genotype,
            extra_scores: ExtraScores,
            fitnesses: Fitness,
            random_key: RNGKey,
            new_evaluations: int = 0,
        ) -> ESMetrics:

        # Super
        metrics = super().get_metrics(
            emitter_state=emitter_state,
            offspring=offspring,
            extra_scores=extra_scores,
            fitnesses=fitnesses,
            random_key=random_key,
            new_evaluations=new_evaluations,
        )

        metrics = metrics.replace(
            sigma=emitter_state.optimizer_state.sigma,
        )

        if "eigen_change" in extra_scores:
            # print(type(metrics))
            metrics = metrics.replace(
                eigen_change = extra_scores["eigen_change"],
            )
        
        if "injection_norm" in extra_scores:
            metrics = metrics.replace(
                injection_norm = extra_scores["injection_norm"],
            )

        if "canonical_update" in extra_scores:
            parent, canonical_update = extra_scores["canonical_update"]
            angles = self.compute_angles(
                g1=offspring,
                g2=canonical_update,
                center=parent,
            )

            metrics = metrics.replace(
                cma_canonical_cosine = angles["cosine_similarity"],
                cma_canonical_sign = angles["same_sign"],
                # cma_norm = angles["v1_norm"],
                canonical_step_norm = angles["v2_norm"]
            )
        return metrics
            

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def compute_angles(self, 
        g1: Genotype,
        g2: Genotype,
        center: Genotype,
    ) -> float:  
        """Compute the cosine similarity between two vectors."""
        g1 = self.flatten(g1)
        g2 = self.flatten(g2)
        center = self.flatten(center)
        v1 = g1 - center
        v2 = g2 - center

        v1_norm = jnp.linalg.norm(v1)
        v2_norm = jnp.linalg.norm(v2)
        cos_sim = jnp.dot(v1, v2) / (v1_norm * v2_norm)

        # Compute the % of dimensions where the sign of the step is the same
        v1_sign = jnp.sign(v1)
        v2_sign = jnp.sign(v2)
        same_sign = jnp.sum(v1_sign == v2_sign) / len(v1_sign)

        return {
            "v1_norm": v1_norm,
            "v2_norm": v2_norm,
            "cosine_similarity": cos_sim,
            "same_sign": same_sign,
        }