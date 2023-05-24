from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Tuple
from typing import Callable, Tuple


import flax.linen as nn
import jax
import optax
from jax import numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_map

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
from jax.tree_util import tree_flatten, tree_unflatten, tree_map

def flatten_genotype(genotype: Genotype) -> jnp.ndarray:
        flat_variables, _ = tree_flatten(genotype)
        # print("Flatten", flat_variables)
        vect = jnp.concatenate([jnp.ravel(x) for x in flat_variables])
        return vect

class TestGradientsEmitter(ESRLEmitter):
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
        
        # Store es center
        # old_offspring = emitter_state.es_state.offspring
        # To vector
        old_center = emitter_state.es_state.offspring
        old_emitter_state = emitter_state

        # Do ES update
        emitter_state, pop_extra_scores = self.es_state_update(
            old_emitter_state,
            repertoire,
            old_center,
            fitnesses,
            descriptors,
            extra_scores
        )

        # Do surrogate ES update
        surrogate_emitter_state, _ = self.surrogate_state_update(
            old_emitter_state,
            repertoire,
            old_center,
            fitnesses,
            descriptors,
            extra_scores
        )

        # check "population_fitness" in pop_extra_scores and surrogate_extra_scores
        # assert "population_fitness" in pop_extra_scores
        # assert "population_fitness" in surrogate_extra_scores

        
        # corr is the correlation coefficient
        # pval is the two-sided p-value for a hypothesis test whose null hypothesis is that two sets of data are uncorrelated

        # Compute ES step
        # es_step = emitter_state.es_state.offspring - old_center
        new_center = emitter_state.es_state.offspring
        surrogate_center = surrogate_emitter_state.es_state.offspring

        key, emitter_state = emitter_state.get_key()
        
        # Update random key
        rl_state = emitter_state.rl_state.replace(
            random_key=key,
        )

        # Compute RL update on old center
        rl_center = self.rl_emitter.emit_pg(
            emitter_state = rl_state,
            parents=old_center,
        )

        # Log

        offspring = emitter_state.es_state.offspring

        metrics = self.get_metrics(
            emitter_state,
            offspring,
            pop_extra_scores,
            fitnesses,
            new_evaluations=self._config.es_config.sample_number,
            random_key=key,
        )

        # True fit - Surrogate fit updates
        surrogate_angles = self.compute_angles(
            g1=new_center,
            g2=surrogate_center,
            center=old_center,
        )

        metrics = metrics.replace(
            surr_fit_cosine= surrogate_angles["cosine_similarity"],
            surr_fit_sign = surrogate_angles["same_sign"],
            surrogate_step_norm = surrogate_angles["v2_norm"],
        )
        
        # ES - RL
        actor_genome = emitter_state.rl_state.actor_params
        actor_genome = flatten_genotype(actor_genome)
        new_center_genome = flatten_genotype(new_center)
        actor_dist = jnp.linalg.norm(actor_genome - new_center_genome)

        angles = self.compute_angles(
            g1=new_center,
            g2=rl_center,
            center=old_center,
        )

        metrics = metrics.replace(
            actor_es_dist = actor_dist,
            es_step_norm = angles["v1_norm"],
            rl_step_norm = angles["v2_norm"],
            es_rl_cosine = angles["cosine_similarity"],
            es_rl_sign = angles["same_sign"],
        )

        # RL - Surrogate ES updates
        surrogate_angles = self.compute_angles(
            g1=surrogate_center,
            g2=rl_center,
            center=old_center,
        )

        metrics = metrics.replace(
            surr_rl_cosine= surrogate_angles["cosine_similarity"],
            surr_rl_sign = surrogate_angles["same_sign"],
        )

        # Stats since start
        initial_center = emitter_state.es_state.initial_center
        angles = self.compute_angles(
            g1=new_center,
            g2=rl_center,
            center=initial_center,
        )

        metrics = metrics.replace(
            es_dist = angles["v1_norm"],
            rl_dist = angles["v2_norm"],
            start_cos_sim = angles["cosine_similarity"],
        )

        # Canonical - RL

        if "canonical_update" in pop_extra_scores:
            _, canonical_update = pop_extra_scores["canonical_update"]
            angles = self.compute_angles(
                g1=canonical_update,
                g2=rl_center,
                center=old_center,
            )

            metrics = metrics.replace(
                canonical_step_norm = angles["v1_norm"],
                rl_step_norm = angles["v2_norm"],
                canonical_rl_cosine = angles["cosine_similarity"],
                canonical_rl_sign = angles["same_sign"],
            )

        emitter_state = emitter_state.set_metrics(metrics)

        return emitter_state
    
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
        g1 = flatten_genotype(g1)
        g2 = flatten_genotype(g2)
        center = flatten_genotype(center)
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
    
    # @partial(
    #     jax.jit,
    #     static_argnames=("self",),
    # )
    # def es_state_update(
    #     self,
    #     emitter_state: ESRLEmitterState,
    #     repertoire: ESRepertoire,
    #     genotypes: Genotype,
    #     fitnesses: Fitness,
    #     descriptors: Descriptor,
    #     extra_scores: ExtraScores,
    # ) -> ESRLEmitterState:
    #     """Generate the gradient offspring for the next emitter call. Also
    #     update the novelty archive and generation count from current call.

    #     Params:
    #         emitter_state
    #         repertoire: unused
    #         genotypes: the genotypes of the offspring
    #         fitnesses: the fitnesses of the offspring
    #         descriptors: the descriptors of the offspring
    #         extra_scores: the extra scores of the offspring

    #     Returns:
    #         the updated emitter state
    #     """
    #     random_key, emitter_state = emitter_state.get_key()

    #     assert jax.tree_util.tree_leaves(genotypes)[0].shape[0] == 1, (
    #         "ERROR: ES generates 1 offspring per generation, "
    #         + "batch_size should be 1, the inputed batch has size:"
    #         + str(jax.tree_util.tree_leaves(genotypes)[0].shape[0])
    #     )

    #     # Updating novelty archive
    #     novelty_archive = emitter_state.es_state.novelty_archive.update(descriptors)

    #     # Define scores for es process
    #     def scores(fitnesses: Fitness, descriptors: Descriptor) -> jnp.ndarray:
    #         if self.es_emitter._config.nses_emitter:
    #             return novelty_archive.novelty(
    #                 descriptors, self.es_emitter._config.novelty_nearest_neighbors
    #             )
    #         else:
    #             return fitnesses

    #     # Run es process
    #     offspring, optimizer_state, new_random_key, extra_scores = self.true_es_emitter(
    #         parent=genotypes,
    #         optimizer_state=emitter_state.es_state.optimizer_state,
    #         random_key=random_key,
    #         scores_fn=scores,
    #         actor=emitter_state.rl_state.actor_params,
    #     )

    #     # Surrogate update simulation

    #     surrogate_offspring, _, _, _ = self.surrogate_es_emitter(
    #         parent=genotypes,
    #         optimizer_state=emitter_state.es_state.optimizer_state,
    #         random_key=random_key,
    #         scores_fn=scores,
    #         actor=emitter_state.rl_state.actor_params,
    #         surrogate_data= emitter_state
    #     )

    #     # Angle
    #     surrogate_angles = self.compute_angles(
    #         g1=offspring,
    #         g2=surrogate_offspring,
    #         center=genotypes,
    #     )

    #     random_key = new_random_key

    #     # Update ES emitter state
    #     es_state = emitter_state.es_state.replace(
    #         offspring=offspring,
    #         optimizer_state=optimizer_state,
    #         random_key=random_key,
    #         novelty_archive=novelty_archive,
    #         generation_count=emitter_state.es_state.generation_count + 1,
    #     )

    #     # Update QPG emitter to train RL agent
    #     # Update random key
    #     rl_state = emitter_state.rl_state.replace(
    #         random_key=random_key,
    #         es_center=genotypes,
    #     )

    #     rl_state = self.rl_emitter.state_update(
    #         emitter_state = rl_state,
    #         repertoire = repertoire,
    #         genotypes = genotypes,
    #         fitnesses = fitnesses,
    #         descriptors = descriptors,
    #         extra_scores = extra_scores,
    #     )

    #     # metrics = self.es_emitter.get_metrics(
    #     #     es_state,
    #     #     offspring,
    #     #     extra_scores,
    #     #     fitnesses,
    #     #     # evaluations=emitter_state.metrics.evaluations,
    #     #     random_key=random_key,
    #     # )

    #     metrics = emitter_state.metrics.replace(
    #         es_updates=emitter_state.metrics.es_updates + 1,
    #         rl_updates=emitter_state.metrics.rl_updates,
    #         surr_fit_cosine= surrogate_angles["cosine_similarity"],
    #         surr_fit_sign = surrogate_angles["same_sign"],
    #         surrogate_step_norm = surrogate_angles["v2_norm"],
    #     )
    #     # Share random key between ES and RL emitters

    #     state = ESRLEmitterState(es_state, rl_state)
    #     state = state.set_metrics(metrics)
    #     state = state.set_key(random_key)
    #     return state, extra_scores