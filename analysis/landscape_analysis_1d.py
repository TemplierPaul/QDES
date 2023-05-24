import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import jax.numpy as jnp
import json

from qdax.core.rl_es_parts.es_setup import setup_es

import argparse


# Repeat same network n_evals time
def multi_evals(genome, n_evals=10):
    # print(genome.shape)
    net = emitter.es_emitter.unflatten(genome)
    nets = jax.tree_map(
        lambda x: jnp.repeat(x[None, ...], n_evals, axis=0),
        net
    )
    key = jax.random.PRNGKey(0)
    fitnesses, descriptors, extra_scores, random_key = scoring_fn(nets, key)
    # print(extra_scores.keys())
    return jnp.mean(fitnesses), extra_scores["transitions"]

def interpolate(save_path, gen, n=1001, dx=0.2):
    offspring_genes = jnp.load(save_path + f"/gen_{gen}_offspring.npy")
    actor_genes = jnp.load(save_path + f"/gen_{gen}_actor.npy")

    # Interpolate between the two
    x = jnp.linspace(0 - dx, 1 + dx, n)[:, None]
    genomes = x * actor_genes + (1 - x) * offspring_genes
    
    return genomes

def batch_eval(genomes, n_evals=20, buffer=None):
    n = len(genomes)
    batch_eval = min(n_evals, 20)

    num_batches = n_evals // batch_eval
    print(num_batches, "batches of", batch_eval, "evaluations each. Total:", num_batches * batch_eval)

    batch_fit = []
    from tqdm import tqdm
    for i in tqdm(range(num_batches)):
        fit, transitions = jax.vmap(multi_evals, in_axes=(0, None))(genomes, batch_eval)
        # print(fit.shape)
        batch_fit.append(fit)
        # print(jax.tree_map(lambda x: x.shape, transitions))
        # Only get first component of 2nd axis
        small_trans = jax.tree_map(
            lambda x: x[:, 0, ...],
            transitions
        )
        if buffer is not None:
            buffer = buffer.insert(small_trans)
    fitnesses = jnp.concatenate(batch_fit).reshape((num_batches, n)).mean(axis=0)
    return fitnesses, buffer

def surrogate_eval(genomes, emitter_state):
    random_key = jax.random.PRNGKey(0)

    networks = jax.vmap(emitter.es_emitter.unflatten)(genomes)

    fitnesses, descriptors, extra_scores, random_key = emitter.surrogate_evaluate(
        networks, 
        random_key=random_key,
        emitter_state=emitter_state,
    )

    return fitnesses

def distance(save_path, gen):
    offspring_genes = jnp.load(save_path + f"/gen_{gen}_offspring.npy")
    actor_genes = jnp.load(save_path + f"/gen_{gen}_actor.npy")

    return jnp.linalg.norm(offspring_genes - actor_genes)


if __name__ == "__main__":
    # parse first cli argument
    parser = argparse.ArgumentParser()
    parser.add_argument("save_path", type=str, help="Path to the folder containing the config.json")
    # List of generations to plot
    parser.add_argument("--gens", type=int, nargs="+", help="Generations to plot", default=None)
    args = parser.parse_args()
    save_path = args.save_path
    print(args)
    print(args.gens)
    
    # get all numbers in gen_nums that are 1 or a multiple of 100
    gens = args.gens
    if gens is None:
        # get number of generations
        import glob
        import re
        gen_files = glob.glob(save_path + "/gen_*.npy")
        # print(gen_files)
        gen_nums = [int(re.findall(r'\d+', f.split("/")[-1])[0]) for f in gen_files]
        # get unique
        gen_nums = list(set(gen_nums))
        gen_nums.sort()
        print(f"Found {len(gen_nums)} generations, max gen: {max(gen_nums)}")

        gens = [g for g in gen_nums if g == 1 or g % 100 == 0]

    print(f"Plotting {len(gens)} generations: {gens}")

    # Load config
    print(save_path + "/config.json")
    with open(save_path + "/config.json", "r") as f:
        args = json.load(f)
        # Lists to tuples
        for k, v in args.items():
            if isinstance(v, list):
                args[k] = tuple(v)
        args = argparse.Namespace(**args)
        args.wandb = ""
        print(args)

    config_name = args.config
    if "TD3" not in config_name:
        # Exit
        print("Not a TD3 config, exiting")
        exit()

    default = {
        "surrogate_batch": 1024,
        "surrogate": False
    }
    # Replace missing values with default
    for k, v in default.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    EM = setup_es(args)

    es = EM.es
    env = EM.env
    policy_network = EM.policy_network
    emitter = EM.emitter
    emitter_state = EM.emitter_state
    repertoire = EM.repertoire
    random_key = EM.random_key
    wandb_run  = EM. wandb_run
    scoring_fn = EM.scoring_fn

    # Interpolation
    n_evals = 10
    n = 201
    dx = 0.2
    emitter._config.rl_config.surrogate_batch = 15000

    # emitter_state = emitter_state
    base_replay_buffer = emitter_state.rl_state.replay_buffer
    print(base_replay_buffer.current_size)

    base_x = jnp.linspace(0 - dx, 1 + dx, n)[:, None]

    # if fitnesses not defined, define it
    try:
        fitnesses
    except NameError:
        fitnesses = {}
    try:
        surrogate_fit
    except NameError:
        surrogate_fit = {}

    for gen in gens:
        print("Gen", gen)
        genomes = interpolate(save_path, gen, n=n, dx=dx)
        fit, replay_buffer = batch_eval(genomes, n_evals=n_evals, buffer=base_replay_buffer)
        fitnesses[gen] = fit

        print(replay_buffer.current_size)

        critic_genes = jnp.load(save_path + f"/gen_{gen}_critic.npy")
        critic = emitter.rl_emitter.unflatten_critic(critic_genes)
        full_rl_state = emitter_state.rl_state.replace(
            replay_buffer=replay_buffer,
            critic_params=critic,
            )

        full_emitter_state = emitter_state.replace(rl_state=full_rl_state)

        surr_fit = surrogate_eval(genomes, full_emitter_state)
        surrogate_fit[gen] = surr_fit

    to_plot = gens
    
    # Fitness landscape
    fig, ax = plt.subplots(figsize=(20, 10))
    # colors = plt.cm.viridis(jnp.linspace(0, 1, len(to_plot))) 
    colors = plt.cm.tab10(jnp.linspace(0, 1, len(to_plot)))
    # colors = plt.cm.Set1(jnp.linspace(0, 1, len(to_plot)))
    # colors = plt.cm.Set2(jnp.linspace(0, 1, len(to_plot)))

    for i, gen in enumerate(to_plot):
        fit = fitnesses[gen]
        dist = distance(save_path, gen)
        plt.plot(base_x, fit, label = f"Gen {gen} | d={dist:.2f}", color=colors[i])

    # Vertical bar at 0 labeled ES center
    plt.axvline(0, color="black", linestyle="--")
    # label on x axis under the bar
    # plt.text(-0.01, 0.5, "ES center", rotation=90, va="center", ha="center")

    # Vertical bar at 1 labeled Actor
    plt.axvline(1, color="black", linestyle="--")
    # label on x axis under the bar
    # plt.text(1.01, 0.5, "Actor", rotation=90, va="center", ha="center")

    plt.xlabel(" <- ES center | Actor ->")
    plt.ylabel("Fitness")
    plt.title(f"{args.algo}\n{args.config}\nInterpolating: {n} points, {n_evals} evals per point")
    plt.legend()
    # save
    fig.savefig(save_path + "/interpolation.png")

    # Surrogate fitness landscape
    fig, ax = plt.subplots(figsize=(20, 10))

    for i, gen in enumerate(to_plot):
        fit = surrogate_fit[gen]
        # normalize with mean std
        fit = (fit - fit.mean()) / fit.std()
        dist = distance(save_path, gen)
        plt.plot(base_x, fit, label = f"Gen {gen} | d={dist:.2f}", color=colors[i])

    # Vertical bar at 0 labeled ES center
    plt.axvline(0, color="black", linestyle="--")
    # label on x axis under the bar
    # plt.text(-0.01, 0.5, "ES center", rotation=90, va="center", ha="center")

    # Vertical bar at 1 labeled Actor
    plt.axvline(1, color="black", linestyle="--")
    # label on x axis under the bar
    # plt.text(1.01, 0.5, "Actor", rotation=90, va="center", ha="center")

    plt.xlabel(" <- ES center | Actor ->")
    plt.ylabel("Surrogate fitness")
    plt.title(f"{args.algo}\n{args.config}\nInterpolating: {n} points, {emitter._config.rl_config.surrogate_batch} transitions in batch")
    plt.legend()
    plt.savefig(save_path + "/surrogate_interpolation.png")

    # Normalized comparison
    # fig, ax = plt.subplots(figsize=(20, 10))
    # MAke one subplot per generation
    base_x = jnp.linspace(0 - dx, 1 + dx, n)[:, None]

    fig, axs = plt.subplots(len(to_plot), 1, figsize=(10, 3*len(to_plot)))
    for i, gen in enumerate(to_plot):
        # surrogate
        surr_fit = surrogate_fit[gen]
        surr_fit = (surr_fit - surr_fit.mean()) / surr_fit.std()
        
        # true fit
        fit = fitnesses[gen]
        fit = (fit - fit.mean()) / fit.std()

        dist = distance(save_path, gen)
        # Scale x to be between 0 and dist
        # x = base_x * dist
        # color = colors[i]
        axs[i].plot(base_x, fit, label = f"True fitness", )
        axs[i].plot(base_x, surr_fit, label = f"Surrogate fitness")
        axs[i].legend()

        # vertical bar at 0
        axs[i].axvline(0, color="black", linestyle="--")
        # vertical bar at 1
        axs[i].axvline(1, color="black", linestyle="--")

        # xlabel only on bottom
        if i == len(to_plot) - 1:
            axs[i].set_xlabel(" <- ES center | Actor ->")
        # ylabel only on left
        axs[i].set_ylabel(f"Gen {gen} | d={dist:.2f}")
        # title: gen
        # axs[i].set_title(f"Gen {gen} | d={dist:.2f}")

    # for i, gen in enumerate(to_plot):
    #     # surrogate
    #     surr_fit = surrogate_fit[gen]
    #     surr_fit = (surr_fit - surr_fit.mean()) / surr_fit.std()
        
    #     # true fit
    #     fit = fitnesses[gen]
    #     fit = (fit - fit.mean()) / fit.std()

    #     dist = distance(save_path, gen)
    #     # Scale x to be between 0 and dist
    #     # x = base_x * dist
    #     color = colors[i]
    #     plt.plot(base_x, fit, label = f"Gen {gen} | True | d={dist:.2f}", color=color)
    #     # dotted line with same color
    #     plt.plot(base_x, surr_fit, label = f"Gen {gen} | Surr. | d={dist:.2f}", linestyle="--", color=color)

    # Vertical bar at 0 labeled ES center
    # plt.axvline(0, color="black", linestyle="--")

    # # Vertical bar at 1 labeled Actor
    # plt.axvline(1, color="black", linestyle="--")

    # plt.xlabel(" <- ES center | Actor ->")
    # plt.ylabel("Surrogate fitness")
    fig.suptitle(f"{args.algo}\n{args.config}\nNormalized landscapes comparison")
    plt.legend()
    plt.savefig(save_path + "/normalized_comparison.png")
