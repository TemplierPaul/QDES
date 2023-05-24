import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import jax.numpy as jnp
import json

from qdax.core.rl_es_parts.es_setup import setup_es

import argparse

FIT_NORM = {
    "halfcheetah_uni": (-2000, 5000),
    "walker2d_uni": (0, 4000),
}

def net_shape(net):
    return jax.tree_map(lambda x: x.shape, net)

def surrogate_eval(genomes, emitter, emitter_state):
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

# @jax.jit
def interpolate2d(save_path, gen, n=100):
    offspring_genes = jnp.load(save_path + f"/gen_{gen}_offspring.npy")
    actor_genes = jnp.load(save_path + f"/gen_{gen}_actor.npy")

    # First axis: ES to actor
    v1 = actor_genes - offspring_genes
    # Second axis: random but orthogonal to first
    key = jax.random.PRNGKey(2)
    v2 = jax.random.normal(key, shape=v1.shape)
    v2 = v2 - jnp.dot(v2, v1) * v1 / jnp.dot(v1, v1)
    v2 = v2 / jnp.linalg.norm(v2) * jnp.linalg.norm(v1)
    
    # Interpolate as grid
    x, y = jnp.meshgrid(
        jnp.linspace(-1, 2, n), 
        jnp.linspace(-1, 1, n)
    )
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    genomes = x * v1 + y * v2 + offspring_genes

    return genomes, x, y

def plot2d(X, Y, Z, save=None, title="2d interpolation", env_name=None):
    # Create a 3D surface plot
    plt.figure()
    contour = plt.contour(X, Y, Z, 20, cmap="viridis")
    plt.legend()
    # plt.colorbar()
    if env_name is not None and env_name in FIT_NORM:
        contour.set_clim(*FIT_NORM[env_name])
    else:
        f_min, f_max = Z.min(), Z.max()
        contour.set_clim(f_min, f_max)
    plt.colorbar(contour)
    plt.scatter(0, 0, c="red", marker="x", label="ES")
    plt.scatter(1, 0, c="red", marker="o", label="Actor")
    plt.xlabel("v1: ES to actor")
    plt.ylabel("v2")
    plt.title(title)
    # same scale both axis
    plt.gca().set_aspect('equal', adjustable='box')
    # save
    if save is not None:
        plt.savefig(save)
        plt.close()
    # plt.show()

import plotly.graph_objs as go
import numpy as np
import plotly.offline as pyo

def plot3d(X, Y, Z, save=None):
    # Create a 3D surface plot
    fig = go.Figure(data=[go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale='Viridis'
    )])

    min_z = Z.min()
    max_z = Z.max()

    x_size = X.max() - X.min()
    y_size = Y.max() - Y.min()
    # get as float from jax array
    x_size = float(x_size)
    y_size = float(y_size)

    # Add vertical lines
    fig.add_trace(go.Scatter3d(
        x=[0, 0],
        y=[0, 0],
        z=[min_z, max_z],
        marker=dict(line=dict(width=10)),
        mode='lines',
        name='ES'
    ))

    fig.add_trace(go.Scatter3d(
        x=[1, 1],
        y=[0, 0],
        z=[min_z, max_z],
        marker=dict(line=dict(width=10)),
        mode='lines',
        name='actor'
    ))
    
    # Set the axis labels and title
    fig.update_layout(scene=dict(
        xaxis_title='ES to actor',
        yaxis_title='Y',
        zaxis_title='Fitness',
        aspectratio=dict(x=x_size, y=y_size, z=0.7),
        camera_eye=dict(x=1.2, y=1.2, z=0.6)
    ))

    # Show the plot in a web browser
    if save is not None:
        pyo.plot(fig, filename=save, auto_open=False)
    # Do not open in browser

def make_plot(EM, gen, env_name):
    es = EM.es
    env = EM.env
    policy_network = EM.policy_network
    emitter = EM.emitter
    emitter_state = EM.emitter_state
    repertoire = EM.repertoire
    random_key = EM.random_key
    wandb_run  = EM. wandb_run
    scoring_fn = EM.scoring_fn

    print(f"Plotting generation {gen}")
    offspring_genes = jnp.load(save_path + f"/gen_{gen}_offspring.npy")
    offspring = emitter.es_emitter.unflatten(offspring_genes)

    nets = jax.tree_map(
        lambda x: jnp.repeat(x[None, ...], 100, axis=0),
        offspring
    )
    key = jax.random.PRNGKey(0)
    fitnesses, descriptors, extra_scores, random_key = scoring_fn(nets, key)

    n_points = 100
    genomes, x, y = interpolate2d(save_path, gen, n=n_points)

    nets = jax.vmap(emitter.es_emitter.unflatten)(genomes)

    base_replay_buffer = emitter_state.rl_state.replay_buffer

    key = jax.random.PRNGKey(0)
    fitnesses, descriptors, extra_scores, random_key = scoring_fn(nets, key)

    transitions = extra_scores["transitions"]
    small_trans = jax.tree_map(
        lambda x: x[:, :10, ...],
        transitions
    )
    buffer = base_replay_buffer.insert(small_trans)
    print(buffer.current_size)

    critic_genes = jnp.load(save_path + f"/gen_{gen}_critic.npy")
    critic = emitter.rl_emitter.unflatten_critic(critic_genes)
    full_rl_state = emitter_state.rl_state.replace(
        replay_buffer=buffer,
        critic_params=critic,
        )

    full_emitter_state = emitter_state.replace(rl_state=full_rl_state)

    # Split into batches
    batch_size = 500
    n_batches = len(genomes) // batch_size + int(len(genomes) % batch_size > 0)
    # genomes_batches = genomes.reshape((n_batches, batch_size, -1))

    # Evaluate with surrogate by batches
    from tqdm import tqdm
    surr_fit = []
    for i in tqdm(range(n_batches)):
        batch = genomes[i * batch_size: (i + 1) * batch_size]
        surr_fit.append(surrogate_eval(batch, emitter, full_emitter_state))

    surr_fit = jnp.concatenate(surr_fit)

    # surr_fit = surrogate_eval(genomes, full_emitter_state)

    surr_fit = (surr_fit - surr_fit.mean()) / surr_fit.std()
    surr_fit.shape
    # dist = distance(save_path, gen)

    x_grid = x.reshape((n_points, n_points))
    y_grid = y.reshape((n_points, n_points))
    z_grid = fitnesses.reshape((n_points, n_points))
    surr_z_grid = surr_fit.reshape((n_points, n_points))

    # 2D plots
    # true fitness
    title = f"True fitness, gen {gen}"
    plot2d(x_grid, y_grid, z_grid, save=save_path + f"/2dlandscape_{gen}.png", title=title, env_name=env_name)
    print(f"Saved as {save_path + f'/2dlandscape_{gen}.png'}")

    # surrogate fitness
    title = f"Surrogate fitness, gen {gen}"
    plot2d(x_grid, y_grid, surr_z_grid, save=save_path + f"/2dsurrogate_{gen}.png", title=title)
    print(f"Saved as {save_path + f'/2dsurrogate_{gen}.png'}")

    # 3D plots
    # true fitness
    plot3d(x_grid, y_grid, z_grid, save=save_path + f"/3dlandscape_{gen}.html")
    print(f"Saved as {save_path + f'/3dlandscape_{gen}.html'}")

    # surrogate fitness
    plot3d(x_grid, y_grid, surr_z_grid, save=save_path + f"/3dsurrogate_{gen}.html")
    print(f"Saved as {save_path + f'/3dsurrogate_{gen}.html'}")

def write_report(args, save_path):
    import glob
    # Create report_id.md file
    job_id = args.jobid
    # make file
    with open(save_path + f"/report_{job_id}.md", "w") as f:
        # Title: Report + job_id
        f.write(f"# Report {job_id}\n")
        # Date
        import datetime
        f.write(f"Date: {datetime.datetime.now()}\n  ")
        # Config
        f.write(f"## {args.env_name}\n")
        f.write(f"## {args.config}\n")
        if args.deterministic:
            f.write(f"## Deterministic\n")
        f.write(f"![]({job_id}/plot.png)\n")
        # Plots
        evo_paths = glob.glob(save_path + f"/2dpath_*.png")
        print("Plots for evolution paths:", evo_paths)
        for p in evo_paths:
            # parse
            elts = p.replace(save_path + "/2dpath_", "").replace(".png", "").split("_")
            f.write(f"## Path: {' + '.join(elts)}\n")
            f.write(f"![]({job_id}/2dpath_{'_'.join(elts)}.png)\n")
        f.write("\n## Fitness landscape\n")
        # Table top
        f.write("| Generation |          True fitness          |       Surrogate fitness         |\n")
        f.write("| :--------: | :----------------------------: | :----------------------------: |\n")
        # Table body
        # get .png files 
        gens = glob.glob(save_path + f"/2dlandscape_*.png")
        print(gens)
        gens = [int(re.findall(r"\d+", gen)[2]) for gen in gens]
        # Sort gens
        gens.sort()
        print(gens)
        for gen in gens:
            f.write(f"| {gen} | ![]({job_id}/2dlandscape_{gen}.png) | ![]({job_id}/2dsurrogate_{gen}.png) |\n")
        # Table bottom
        f.write("\n")
        # 1D
        f.write("\n## 1D interpolation\n")
        f.write("\n### True fitness\n")
        f.write(f"![]({job_id}/interpolation.png)\n")
        f.write("\n### Normalized comparison\n")
        f.write(f"![]({job_id}/normalized_comparison.png)\n")



if __name__ == "__main__":
    # parse first cli argument
    parser = argparse.ArgumentParser()
    parser.add_argument("save_path", type=str, help="Path to the folder containing the config.json")
    # List of generations to plot
    parser.add_argument("--gens", type=int, nargs="+", help="Generations to plot", default=None)
    # --report flag
    parser.add_argument("--report", action="store_true", help="Create report.md file")
    plot_args = parser.parse_args()
    save_path = plot_args.save_path
    print(plot_args)
    print(plot_args.gens)

    if plot_args.gens is None:
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

        plot_args.gens = [g for g in gen_nums if g == 1 or g % 100 == 0]
        # Do every 10 gen until 100
        plot_args.gens += [g for g in gen_nums if g < 100 and g % 10 == 0]
        # Sort
        plot_args.gens.sort()

    print(f"Plotting {len(plot_args.gens)} generations: {plot_args.gens}")

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
        write_report(args, save_path)
        exit()

    default = {
        "surrogate_batch": 1024,
        "surrogate": False,
        # "deterministic": True,
        # "es": "cmaes"
    }
    # Replace missing values with default
    for k, v in default.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    def scores(fitnesses, descriptors) -> jnp.ndarray:
        return fitnesses    

    if not plot_args.report:
        EM = setup_es(args)
        
        for gen in plot_args.gens:
            make_plot(EM, gen, env_name=args.env_name)

    write_report(args, save_path)