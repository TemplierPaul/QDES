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
from qdes.core.emitters.esrl_emitter import ESRLConfig, ESRLEmitterState, ESRLEmitter, spearman
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdes.core.emitters.test_gradients import TestGradientsEmitter, flatten_genotype

class SpearmanSurrogateState(ESRLEmitterState):
    surrogate_omega: float

class SpearmanSurrogateEmitter(TestGradientsEmitter):
    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> Tuple[SpearmanSurrogateState, RNGKey]:
        """
        Initializes the emitter state.

        Args:
            init_genotypes: the initial genotypes to use.
            random_key: the random key to use for initialization.

        Returns:
            the initialized emitter state.
        """
        # print("RLES init_genotypes", jax.tree_map(lambda x: x.shape, init_genotypes))

        es_state, random_key = self.es_emitter.init(init_genotypes, random_key)
        rl_state, random_key = self.rl_emitter.init(init_genotypes, random_key)
        # Make sure the random key is the same for both emitters

        # print("Init QPG Emitter", rl_state.replay_buffer.current_position)

        es_state = es_state.replace(random_key=random_key)
        rl_state = rl_state.replace(random_key=random_key)

        metrics = ESMetrics(
            es_updates=0,
            rl_updates=0,
            surrogate_updates=0,
            evaluations=0,
            actor_fitness=-jnp.inf,
            center_fitness=-jnp.inf,
        )

        state = SpearmanSurrogateState(
            es_state, 
            rl_state,
            surrogate_omega = 0.0
            )
        state = state.set_metrics(metrics)
        state = state.set_key(random_key)

        self._surrogate_eval = jax.vmap(
            self.rl_emitter.surrogate_eval, 
            in_axes=(0, None, None),
        )

        self.true_es_emitter = self.es_emitter._es_emitter

        self.surrogate_es_emitter = partial(
            self.es_emitter._base_es_emitter, 
            fitness_function=self.surrogate_evaluate,
        )

        # jit it
        self.surrogate_es_emitter = partial(
            jax.jit,
            static_argnames=("scores_fn"),
        )(self.surrogate_es_emitter)

        # print("Init ESRL Emitter", state.rl_state.replay_buffer.current_position)

        return state, random_key
    
    @property
    def config_string(self):
        s = self.es_emitter.config_string + " | " + self.rl_emitter.config_string
        s += f" | SM ({self._config.rl_config.surrogate_batch})" # \u03C9 is omega
        return s
    
    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def spearman_update(
        self, 
        emitter_state: SpearmanSurrogateState,
        spearman_score: jnp.ndarray,
    ) -> SpearmanSurrogateState:
        # Update the surrogate probability
        current_omega = emitter_state.surrogate_omega
        new_omega = spearman_score
        new_omega = jnp.clip(new_omega, 0, 0.9)
        # new_omega = (current_omega + new_omega) / 2
        # update the state
        emitter_state = emitter_state.replace(surrogate_omega=new_omega)
        return emitter_state

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def state_update(
        self,
        emitter_state: SpearmanSurrogateState,
        repertoire: ESRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> SpearmanSurrogateState:
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
        omega = emitter_state.surrogate_omega
        cond = jax.random.choice(
            subkey, 
            jnp.array([True, False]), 
            p=jnp.array([omega, 1-omega]))

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
    
    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def es_state_update(
        self,
        emitter_state: SpearmanSurrogateState,
        repertoire: ESRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> SpearmanSurrogateState:
        """Generate the gradient offspring for the next emitter call. Also
        update the novelty archive and generation count from current call.

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
        random_key, emitter_state = emitter_state.get_key()

        assert jax.tree_util.tree_leaves(genotypes)[0].shape[0] == 1, (
            "ERROR: ES generates 1 offspring per generation, "
            + "batch_size should be 1, the inputed batch has size:"
            + str(jax.tree_util.tree_leaves(genotypes)[0].shape[0])
        )

        # Updating novelty archive
        novelty_archive = emitter_state.es_state.novelty_archive.update(descriptors)

        # Define scores for es process
        def scores(fitnesses: Fitness, descriptors: Descriptor) -> jnp.ndarray:
            return fitnesses

        base_optim_state = emitter_state.es_state.optimizer_state
        # Run es process
        offspring, optimizer_state, new_random_key, extra_scores = self.true_es_emitter(
            parent=genotypes,
            optimizer_state=base_optim_state,
            random_key=random_key,
            scores_fn=scores,
            actor=emitter_state.rl_state.actor_params,
        )

        # Compute surrogate update
        surrogate_offspring, _, _, surrogate_extra_scores = self.surrogate_es_emitter(
            parent=genotypes,
            optimizer_state=base_optim_state,
            random_key=random_key,
            scores_fn=scores,
            actor=emitter_state.rl_state.actor_params,
            surrogate_data= emitter_state
        )

        random_key = new_random_key

        true_fit = extra_scores["population_fitness"]
        surr_fit = surrogate_extra_scores["population_fitness"]

        corr, pval = spearman(true_fit, surr_fit)

        # Update ES emitter state
        es_state = emitter_state.es_state.replace(
            offspring=offspring,
            optimizer_state=optimizer_state,
            random_key=random_key,
            novelty_archive=novelty_archive,
            generation_count=emitter_state.es_state.generation_count + 1,
        )

        # Update QPG emitter to train RL agent
        # Update random key
        rl_state = emitter_state.rl_state.replace(
            random_key=random_key,
            es_center=genotypes,
        )

        rl_state = self.rl_emitter.state_update(
            emitter_state = rl_state,
            repertoire = repertoire,
            genotypes = genotypes,
            fitnesses = fitnesses,
            descriptors = descriptors,
            extra_scores = extra_scores,
        )

        # metrics = self.es_emitter.get_metrics(
        #     es_state,
        #     offspring,
        #     extra_scores,
        #     fitnesses,
        #     # evaluations=emitter_state.metrics.evaluations,
        #     random_key=random_key,
        # )

        metrics = emitter_state.metrics.replace(
            es_updates=emitter_state.metrics.es_updates + 1,
            rl_updates=emitter_state.metrics.rl_updates,
        )

        metrics = metrics.replace(
            spearmans_correlation = corr,
            spearmans_pvalue = pval,
        )
        # Share random key between ES and RL emitters

        state = SpearmanSurrogateState(
            es_state, 
            rl_state,
            surrogate_omega = emitter_state.surrogate_omega,
            )
        state = state.set_key(random_key)
        # print("ES offspring", jax.tree_map(lambda x: x.shape, offspring))

        state = self.spearman_update(
            emitter_state=state,
            spearman_score=metrics.spearmans_correlation,
        )

        metrics = metrics.replace(
            spearman_omega = state.surrogate_omega,
        )
        state = state.set_metrics(metrics)

        return state, extra_scores
    
    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def surrogate_state_update(
        self,
        emitter_state: ESRLEmitterState,
        repertoire: ESRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> ESRLEmitterState:
        omega = emitter_state.surrogate_omega
        emitter_state, extra_scores = TestGradientsEmitter.surrogate_state_update(
            self,
            emitter_state,
            repertoire,
            genotypes,
            fitnesses,
            descriptors,
            extra_scores,
        )
        state = SpearmanSurrogateState(
            es_state = emitter_state.es_state,
            rl_state = emitter_state.rl_state,
            surrogate_omega = omega,
        )
        return state, extra_scores
        

