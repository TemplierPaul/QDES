# Import after parsing arguments
import functools
import time
from typing import Dict

import jax

print("Device count:", jax.device_count(), jax.devices())
import jax.numpy as jnp

from qdax import environments
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdes.core.emitters.vanilla_es_emitter import VanillaESConfig, VanillaESEmitter
# from qdax.core.map_elites import MAPElites
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.tasks.brax_envs import (
    make_policy_network_play_step_fn_brax,
    reset_based_scoring_function_brax_envs,
)
from qdax.utils.metrics import CSVLogger, default_qd_metrics
from qdax.utils.plotting import plot_map_elites_results

from qdes.core.rl_es_parts.open_es import OpenESEmitter, OpenESConfig
from qdes.core.rl_es_parts.canonical_es import CanonicalESConfig, CanonicalESEmitter
from qdes.core.rl_es_parts.random_search import RandomConfig, RandomEmitter
from qdes.core.rl_es_parts.mono_cmaes import MonoCMAESEmitter, MonoCMAESConfig
from qdes.core.rl_es_parts.es_utils import ES, default_es_metrics, ESMetrics

from qdax.core.emitters.qpg_emitter import QualityPGConfig, QualityPGEmitterState, QualityPGEmitter
from qdes.core.emitters.custom_qpg_emitter import CustomQualityPGConfig, CustomQualityPGEmitter

from qdes.core.emitters.esrl_emitter import ESRLConfig, ESRLEmitter
from qdes.core.emitters.test_gradients import TestGradientsEmitter
from qdes.core.emitters.carlies_emitter import CARLIES
from qdes.core.emitters.surrogate_es_emitter import SurrogateESConfig, SurrogateESEmitter

import wandb
from dataclasses import dataclass

@dataclass
class ESMaker:
    es = None
    env = None
    emitter = None
    emitter_state = None
    repertoire = None
    random_key = None
    wandb_run = None 
    policy_network = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def setup_es(args):
    print("Imported modules")

    entity = None
    project = args.wandb
    wandb_run = None
    if project != "":
        if "/" in project:
            entity, project = project.split("/")
        wandb_run = wandb.init(
            project=project,
            entity=entity,
            config = {**vars(args)})

        print("Initialized wandb")

    ###############
    # Environment #

    # Init environment
    env = environments.create(
        args.env_name, 
        episode_length=args.episode_length,
        fixed_init_state= args.deterministic
        )

    # Init a random key
    random_key = jax.random.PRNGKey(args.seed)

    # Init policy network
    policy_layer_sizes = args.policy_hidden_layer_sizes + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    # Init population of controllers
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=1)
    fake_batch = jnp.zeros(shape=(1, env.observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

    # print("Init variables", jax.tree_map(lambda x: x.shape, init_variables))

    # Play reset fn
    # WARNING: use "env.reset" for stochastic environment,
    # use "lambda random_key: init_state" for deterministic environment
    play_reset_fn = env.reset

    # Prepare the scoring function
    bd_extraction_fn = environments.behavior_descriptor_extractor[args.env_name]
    scoring_fn = functools.partial(
        reset_based_scoring_function_brax_envs,
        episode_length=args.episode_length,
        play_reset_fn=play_reset_fn,
        play_step_fn=make_policy_network_play_step_fn_brax(env, policy_network),
        behavior_descriptor_extractor=bd_extraction_fn,
    )

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = environments.reward_offset[args.env_name]

    # Define a metrics function
    metrics_function = functools.partial(
        default_es_metrics,
        qd_offset=reward_offset * args.episode_length,
    )

    #############
    # Algorithm #

    # ES emitter
    if args.es in ["open", "openai"]:
        es_config = OpenESConfig(
            nses_emitter=args.nses_emitter,
            sample_number=args.pop,
            sample_sigma=args.es_sigma,
            sample_mirror=args.sample_mirror,
            sample_rank_norm=args.sample_rank_norm,
            adam_optimizer=args.adam_optimizer,
            learning_rate=args.learning_rate,
            l2_coefficient=args.l2_coefficient,
            novelty_nearest_neighbors=args.novelty_nearest_neighbors,
            actor_injection = args.actor_injection,
            nb_injections = args.nb_injections,
            episode_length = args.episode_length,
        )

        es_emitter = OpenESEmitter(
            config=es_config,
            scoring_fn=scoring_fn,
            total_generations=args.num_gens,
            num_descriptors=env.behavior_descriptor_length,
        )
    elif args.es in ["canonical"]:
        es_config = CanonicalESConfig(
            nses_emitter=args.nses_emitter,
            sample_number=args.pop,
            canonical_mu=int(args.pop / 2),
            sample_sigma=args.es_sigma,
            learning_rate=args.learning_rate,
            novelty_nearest_neighbors=args.novelty_nearest_neighbors,
            actor_injection = args.actor_injection,
            nb_injections = args.nb_injections,
            episode_length = args.episode_length,
        )

        es_emitter = CanonicalESEmitter(
            config=es_config,
            scoring_fn=scoring_fn,
            total_generations=args.num_gens,
            num_descriptors=env.behavior_descriptor_length,
        )

    elif args.es in ["cmaes"]:
        es_config = MonoCMAESConfig(
            nses_emitter=args.nses_emitter,
            sample_number=args.pop,
            sample_sigma=args.es_sigma,
            actor_injection = args.actor_injection,
            nb_injections = args.nb_injections,
            episode_length = args.episode_length,
        )

        es_emitter = MonoCMAESEmitter(
            config=es_config,
            scoring_fn=scoring_fn,
            total_generations=args.num_gens,
            num_descriptors=env.behavior_descriptor_length,
        )

    elif args.es in ["random"]:
        es_config = RandomConfig(
            nses_emitter=args.nses_emitter,
            sample_number=args.pop,
            actor_injection = args.actor_injection,
            nb_injections = args.nb_injections,
            episode_length = args.episode_length,
        )
        
        es_emitter = RandomEmitter(
            config=es_config,
            scoring_fn=scoring_fn,
            total_generations=args.num_gens,
            num_descriptors=env.behavior_descriptor_length,
        )

    else:
        raise ValueError(f"Unknown ES type: {args.es}")

    if args.rl:
        
        rl_config = CustomQualityPGConfig(
            env_batch_size = 100,
            num_critic_training_steps = args.critic_training,
            num_pg_training_steps = args.pg_training,

            # TD3 params
            replay_buffer_size = 1000000,
            critic_hidden_layer_size = args.critic_hidden_layer_sizes,
            critic_learning_rate = 3e-4,
            actor_learning_rate = 3e-4,
            policy_learning_rate = 1e-3,
            noise_clip = 0.5,
            policy_noise = 0.2,
            discount = 0.99,
            reward_scaling = 1.0,
            batch_size = 256,
            soft_tau_update = 0.005,
            policy_delay = 2,

            elastic_pull = args.elastic_pull,
            surrogate_batch = args.surrogate_batch,
        )
            
        rl_emitter = CustomQualityPGEmitter(
            config=rl_config,
            policy_network=policy_network,
            env=env,
        )

        # ESRL emitter
        esrl_emitter_type = ESRLEmitter
        if args.carlies:
            esrl_emitter_type = CARLIES
        elif args.testrl:
            esrl_emitter_type = TestGradientsEmitter

        if args.surrogate:
            esrl_config = SurrogateESConfig(
                es_config=es_config,
                rl_config=rl_config,
                surrogate_omega=args.surrogate_omega,
            )
            esrl_emitter_type = SurrogateESEmitter

        else:
            esrl_config = ESRLConfig(
                es_config=es_config,
                rl_config=rl_config,
            )

        emitter = esrl_emitter_type(
            config=esrl_config,
            es_emitter=es_emitter,
            rl_emitter=rl_emitter,
        )

    else:
        emitter = es_emitter

    # Instantiate ES
    es = ES(
        scoring_function=scoring_fn,
        emitter=emitter,
        metrics_function=metrics_function,
    )

    # Compute the centroids
    centroids, random_key = compute_cvt_centroids(
        num_descriptors=env.behavior_descriptor_length,
        num_init_cvt_samples=args.num_init_cvt_samples,
        num_centroids=args.num_centroids,
        minval=args.min_bd,
        maxval=args.max_bd,
        random_key=random_key,
    )

    # Compute initial repertoire and emitter state
    repertoire, emitter_state, random_key = es.init(
        init_variables, centroids, random_key
    )
    # print("After ES init:", emitter_state.rl_state.replay_buffer.current_position)

    # print("Initialized ES")
    print(es_emitter)

    return ESMaker(
        es=es,
        env=env,
        emitter=emitter,
        emitter_state=emitter_state,
        repertoire=repertoire,
        random_key=random_key,
        wandb_run = wandb_run,
        policy_network = policy_network,
        scoring_fn=scoring_fn,
    )