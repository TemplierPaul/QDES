try:
    from qdax.core.emitters.esrl_emitter import ESRLConfig, ESRLEmitter
    # Raise exception if qdax is not clean
    raise Exception("QDax is not clean")
except ModuleNotFoundError:
    print("Clean QDax")

import argparse
import os
try:
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']
except KeyError:
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1'
print("XLA memory", os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'])

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--env', 
    type=str, 
    default="walker2d_uni", 
    help='Environment name', 
    # choices=['ant_uni', 'hopper_uni', 'walker2d_uni', 'halfcheetah_uni', 'humanoid_uni', 'ant_omni', 'humanoid_omni', 'anttrap'],
    dest='env_name'
    )
parser.add_argument('--episode_length', type=int, default=1000, help='Number of steps per episode')
# parser.add_argument('--gen', type=int, default=10000, help='Generations', dest='num_iterations')
parser.add_argument('--evals', type=int, default=1000000, help='Evaluations')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--policy_hidden_layer_sizes', type=int, default=128, help='Policy network hidden layer sizes')
parser.add_argument('--critic_hidden_layer_sizes', type=int, default=128, help='critic network hidden layer sizes')
parser.add_argument('--deterministic', default=False, action="store_true", help='Fixed init state')

# Map-Elites
parser.add_argument('--num_init_cvt_samples', type=int, default=50000, help='Number of samples to use for CVT initialization')
parser.add_argument('--num_centroids', type=int, default=1024, help='Number of centroids')
parser.add_argument('--min_bd', type=float, default=0.0, help='Minimum value for the behavior descriptor')
parser.add_argument('--max_bd', type=float, default=1.0, help='Maximum value for the behavior descriptor')

# ES
# ES type
parser.add_argument('--es', type=str, default='es', help='ES type', choices=['open', 'canonical', 'cmaes', 'random'])
parser.add_argument('--pop', type=int, default=512, help='Population size')
parser.add_argument('--es_sigma', type=float, default=0.01, help='Standard deviation of the Gaussian distribution')
parser.add_argument('--sample_mirror', type=bool, default=True, help='Mirror sampling in ES')
parser.add_argument('--sample_rank_norm', type=bool, default=True, help='Rank normalization in ES')
parser.add_argument('--adam_optimizer', type=bool, default=True, help='Use Adam optimizer instead of SGD')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for ES optimizer')
parser.add_argument('--l2_coefficient', type=float, default=0.02, help='L2 coefficient for Adam optimizer')

# NSES
parser.add_argument('--nses_emitter', type=bool, default=False, help='Use NSES instead of ES')
parser.add_argument('--novelty_nearest_neighbors', type=int, default=10, help='Number of nearest neighbors to use for novelty computation')

# RL
parser.add_argument('--rl', default=False, action="store_true", help='Add RL')
parser.add_argument('--testrl', default=False, action="store_true", help='Add RL/ES testing')
parser.add_argument('--carlies', default=False, action="store_true", help='Add CARLIES')
parser.add_argument('--elastic_pull', type=float, default=0, help='Penalization for pulling the actor too far from the ES center')
parser.add_argument('--discount', type=float, default=0.99, help='Discount factor')
parser.add_argument('--actor_injection', action="store_true", default=False, help='Use actor injection')
parser.add_argument('--injection_clip', action="store_true", default=False, help='Clip actor vector norm for injection')
parser.add_argument('--nb_injections', type=int, default=1, help='Number of actors to inject if actor_injection is True')
parser.add_argument('--critic_training', type=int, default=1000, help='Number of critic training steps')
parser.add_argument('--pg_training', type=int, default=1000, help='Number of PG training steps')
parser.add_argument('--actor_lr', type=float, default=3e-4, help='Learning rate for actor Adam optimizer')
parser.add_argument('--critic_lr', type=float, default=3e-4, help='Learning rate for critic Adam optimizer')
parser.add_argument('--es_target', action="store_true", default=False, help='Use ES center as critic target')


# RL + ES
parser.add_argument('--surrogate', default=False, action="store_true", help='Use surrogate')
parser.add_argument('--surrogate_batch', type=int, default=1024, help='Number of samples to use for surrogate evaluation')
parser.add_argument('--surrogate_omega', type=float, default=0.6, help='Probability of using surrogate')
parser.add_argument('--spearman', default=False, action="store_true", help='Use surrogate with spearman-ajusted probability')
# parser.add_argument('--spearman_decay', type=float, default=1.0, help='Spearman decay')

# File output
parser.add_argument('--output', type=str, default='', help='Output file')
parser.add_argument('--plot', default=False, action="store_true", help='Make plots')

# Wandb
parser.add_argument('--wandb', type=str, default='', help='Wandb project name')
parser.add_argument('--tag', type=str, default='', help='Project tag')
parser.add_argument('--jobid', type=str, default='', help='Job ID')

# Log period
parser.add_argument('--log_period', type=int, default=1, help='Log period')

# Debug flag 
parser.add_argument('--debug', default=False, action="store_true", help='Debug flag')
parser.add_argument('--logall', default=False, action="store_true", help='Lot at each generation')

# parse arguments
args = parser.parse_args()

if args.carlies or args.testrl or args.surrogate or args.spearman or args.actor_injection:
    args.rl = True
    # args.actor_injection = False

if args.es_target and not args.rl:
    raise ValueError("ES target requires RL")

if args.injection_clip and not args.actor_injection:
    raise ValueError("Injection clip requires actor injection")

if args.debug:
    # Cheap ES to debug
    debug_values = {
        # 'env_name': 'walker2d_uni',
        'episode_length': 100,
        "pop": 10,
        'evals': 100,
        'policy_hidden_layer_sizes': 16,
        'critic_hidden_layer_sizes': 16,
        "output": "debug",
        'surrogate_batch': 10,
    }
    for k, v in debug_values.items():
        setattr(args, k, v)

log_period = args.log_period
args.num_gens = args.evals // args.pop
# num_loops = int(args.num_gens / log_period)

args.policy_hidden_layer_sizes = (args.policy_hidden_layer_sizes, args.policy_hidden_layer_sizes)
args.critic_hidden_layer_sizes = (args.critic_hidden_layer_sizes, args.critic_hidden_layer_sizes)

algos = {
    'open': 'OpenAI',
    'openai': 'OpenAI',
    'canonical': 'Canonical',
    'cmaes': 'CMAES',
    'random': 'Random',
}
args.algo = algos[args.es]

suffix = ''
if args.rl:
    suffix = '-RL'
if args.carlies:
    suffix = '-CARLIES'
if args.testrl:
    suffix = '-TestRL'
if args.surrogate:
    suffix = '-Surrogate'
if args.spearman:
    suffix = '-Spearman'
args.algo += f"{suffix}"


if args.actor_injection:
    args.algo += "-AI"

# args.config = f"ES {args.pop} - \u03C3 {args.es_sigma} - \u03B1 {args.learning_rate}"
# if args.elastic_pull > 0:
#     args.config += f" - \u03B5 {args.elastic_pull}" # \u03B5 is epsilon
# if args.surrogate:
#     args.config += f" - \u03C9 {args.surrogate_omega} ({args.surrogate_batch})" # \u03C9 is omega

print("Parsed arguments:", args)


# Import after parsing arguments
import functools
import time
from typing import Dict

import jax

print("Device count:", jax.device_count(), jax.devices())
import jax.numpy as jnp

from qdax import environments
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
# from qdax.core.emitters.vanilla_es_emitter import VanillaESConfig, VanillaESEmitter
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
from qdes.core.emitters.custom_qpg_emitter import CustomQualityPGConfig, CustomQualityPGEmitter, ESTargetQualityPGEmitter

from qdes.core.emitters.esrl_emitter import ESRLConfig, ESRLEmitter
from qdes.core.emitters.test_gradients import TestGradientsEmitter
from qdes.core.emitters.carlies_emitter import CARLIES
from qdes.core.emitters.surrogate_es_emitter import SurrogateESConfig, SurrogateESEmitter
from qdes.core.emitters.spearman_surrogate import SpearmanSurrogateEmitter


###############
# Environment #

# Init environment
env = environments.create(
    args.env_name, 
    episode_length=args.episode_length,
    fixed_init_state= args.deterministic,
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
        # learning_rate=args.learning_rate,
        novelty_nearest_neighbors=args.novelty_nearest_neighbors,
        actor_injection = args.actor_injection,
        nb_injections = args.nb_injections,
        episode_length = args.episode_length,
        injection_clipping = args.injection_clip,
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
        critic_learning_rate = args.critic_lr,
        actor_learning_rate = args.actor_lr,
        policy_learning_rate = args.actor_lr,
        noise_clip = 0.5,
        policy_noise = 0.2,
        discount = args.discount,
        reward_scaling = 1.0,
        batch_size = 256,
        soft_tau_update = 0.005,
        policy_delay = 2,

        elastic_pull = args.elastic_pull,
        surrogate_batch = args.surrogate_batch,
    )

    if args.es_target:
        rl_emitter = ESTargetQualityPGEmitter(
        config=rl_config,
        policy_network=policy_network,
        env=env,
    )
    else:
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
    elif args.spearman:
        esrl_config = ESRLConfig(
            es_config=es_config,
            rl_config=rl_config,
        )
        esrl_emitter_type = SpearmanSurrogateEmitter
        
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

print("Initialized ES")
print(es_emitter)
print(emitter.config_string)

args.config = emitter.config_string

import wandb
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

#######
# Run #
if args.output != "":
    import os

    directory = args.output

    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully!")
    else:
        print(f"Directory '{directory}' already exists!")


    plot_file = args.output + "/plot.png"
    log_file = args.output + "/log.csv"

    import json
    with open(args.output + "/config.json", "w") as file:
        json.dump(args.__dict__, file, indent=4)

# get all the fields in ESMetrics
header = ESMetrics.__dataclass_fields__.keys()

csv_logger = CSVLogger(
    log_file,
    header=["loop", "generation", 
            "qd_score",  "max_fitness", "coverage", 
            "time", "frames"] + list(header),
)
all_metrics: Dict[str, float] = {}

# main loop
es_scan_update = es.scan_update

# main iterations
from tqdm import tqdm
bar = tqdm(range(args.evals))
evaluations = 0
gen = 0
try:
    while evaluations < args.evals:
        start_time = time.time()
        (repertoire, emitter_state, random_key,), metrics = jax.lax.scan(
            es_scan_update,
            (repertoire, emitter_state, random_key),
            (),
            length=log_period,
        )
        timelapse = time.time() - start_time

        # log metrics
        gen += 1
        logged_metrics = {
            "time": timelapse, 
            # "loop": 1 + i, 
            "generation": gen,
            "frames": gen * args.episode_length * args.pop,
            }
        for key, value in metrics.items():
            # take last value
            logged_metrics[key] = value[-1]

            # take all values
            if key in all_metrics.keys():
                all_metrics[key] = jnp.concatenate([all_metrics[key], value])
            else:
                all_metrics[key] = value

        csv_logger.log(logged_metrics)  
        if wandb_run:
            wandb_run.log(logged_metrics)

        if args.logall and args.output != "":
            output = args.output + "/gen_" + str(gen)
            print("Saving to", output)
            emitter_state.save(output)

        # Update bar
        evaluations = logged_metrics["evaluations"]
        evaluations = int(evaluations)
        # Set bar progress
        bar.update(evaluations - bar.n)
        bar.set_description(f"Gen: {gen}, qd_score: {logged_metrics['qd_score']:.2f}, max_fitness: {logged_metrics['max_fitness']:.2f}, coverage: {logged_metrics['coverage']:.2f}, time: {timelapse:.2f}")
except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    # Save
    if args.output != "":
        output = args.output + "/gen_" + str(gen)
        print("Saving to", output)
        emitter_state.save(output)

# print(logged_metrics)
for k, v in logged_metrics.items():
    print(f"{k}: {v}")

#################
# Visualisation #
if args.plot:
    # create the x-axis array
    env_steps = jnp.arange(logged_metrics["evaluations"]) * args.episode_length

    # Check the number of dimensions of the descriptors
    if len(repertoire.descriptors.shape) == 2:
        # create the plots and the grid
        fig, axes = plot_map_elites_results(
            env_steps=env_steps,
            metrics=all_metrics,
            repertoire=repertoire,
            min_bd=args.min_bd,
            max_bd=args.max_bd,
        )

        import matplotlib.pyplot as plt
        plt.savefig(plot_file)

        # Log the repertoire plot
        if wandb_run:
            from qdax.utils.plotting import plot_2d_map_elites_repertoire

            fig, ax = plot_2d_map_elites_repertoire(
                centroids=repertoire.centroids,
                repertoire_fitnesses=repertoire.fitnesses,
                minval=args.min_bd,
                maxval=args.max_bd,
                repertoire_descriptors=repertoire.descriptors,
            )
            wandb_run.log({"archive": wandb.Image(fig)})

    html_content = repertoire.record_video(env, policy_network)
    video_file = plot_file.replace(".png", ".html")
    with open(video_file, "w") as file:
        file.write(html_content)

    # Log the plot
    if wandb_run:
        wandb.log({"best_agent": wandb.Html(html_content)})
        wandb.finish()
