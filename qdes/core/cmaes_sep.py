"""
Definition of CMAES class, containing main functions necessary to build
a CMA optimization script. Link to the paper: https://arxiv.org/abs/1604.00772
"""
from functools import partial
from typing import Callable, Optional, Tuple

import flax
import jax
import jax.numpy as jnp

from qdax.types import Fitness, Genotype, Mask, RNGKey
from qdax.core.cmaes import CMAESState, CMAES

class SepCMAESState(flax.struct.PyTreeNode):
    """Describe a state of the Separable Covariance Matrix Adaptation Evolution Strategy
    (CMA-ES) algorithm.

    Args:
        mean: mean of the gaussian distribution used to generate solutions
        cov_vector: covariance vector of the gaussian distribution used to
            generate solutions - (multiplied by sigma for sampling).
        num_updates: number of updates made by the CMAES optimizer since the
            beginning of the process.
        sigma: the step size of the optimization steps. Multiplies the cov matrix
            to get the real cov matrix used for the sampling process.
        p_c: evolution path
        p_s: evolution path
        eigen_updates: track the latest update to know when to do the next one.
        eigenvalues: latest eigenvalues
        D:
    """

    mean: jnp.ndarray
    cov_vector: jnp.ndarray
    num_updates: int
    sigma: float
    p_c: jnp.ndarray
    p_s: jnp.ndarray
    eigen_updates: int
    eigenvalues: jnp.ndarray
    D: jnp.ndarray

class SepCMAES(CMAES):
    """
    Class to run the Separable CMA-ES algorithm.
    """

    def __init__(
        self,
        population_size: int,
        search_dim: int,
        fitness_function: Callable[[Genotype], Fitness],
        num_best: Optional[int] = None,
        init_sigma: float = 1e-3,
        mean_init: Optional[jnp.ndarray] = None,
        bias_weights: bool = True,
        delay_eigen_decomposition: bool = False,
    ):
        """Instantiate a CMA-ES optimizer.

        Args:
            population_size: size of the running population.
            search_dim: number of dimensions in the search space.
            fitness_function: fitness function that is being optimized.
            num_best: number of best individuals in the population being considered
                for the update of the distributions. Defaults to None.
            init_sigma: Initial value of the step size. Defaults to 1e-3.
            mean_init: Initial value of the distribution mean. Defaults to None.
            bias_weights: Should the weights be biased towards best individuals.
                Defaults to True.
            delay_eigen_decomposition: should the update of the inverse of the
                cov matrix be delayed. As this operation is a time bottleneck, having
                it delayed improves the time perfs by a significant margin.
                Defaults to False.
        """
        self._population_size = population_size
        self._search_dim = search_dim
        self._fitness_function = fitness_function
        self._init_sigma = init_sigma

        # Default values if values are not provided
        if num_best is None:
            self._num_best = population_size // 2
        else:
            self._num_best = num_best

        if mean_init is None:
            self._mean_init = jnp.zeros(shape=(search_dim,))
        else:
            self._mean_init = mean_init

        # weights parameters
        if bias_weights:
            # heuristic from Nicolas Hansen original implementation
            self._weights = jnp.log(
                (self._num_best + 0.5) / jnp.arange(start=1, stop=(self._num_best + 1))
            )
        else:
            self._weights = jnp.ones(self._num_best)

        # scale weights
        self._weights = self._weights / (self._weights.sum())
        self._parents_eff = 1 / (self._weights**2).sum()

        # adaptation  parameters
        # Eq 55
        self._c_s = (self._parents_eff + 2) / (self._search_dim + self._parents_eff + 3)
        
        # Eq 56
        self._c_c = 4 / (self._search_dim + 4)

        # learning rate for rank-1 update of C
        self._c_1 = 2 / (self._parents_eff + (self._search_dim + jnp.sqrt(2)) ** 2)

        # learning rate for rank-(num best) updates
        c_cov_full = 2 / self._parents_eff / ((self._search_dim + jnp.sqrt(2)) ** 2)
        c_cov_full += (1 - 1 / self._parents_eff) * min(
            1, 
            (2 * self._parents_eff - 1) / ((self._search_dim + 2) ** 2 + self._parents_eff)
        )
        self._c_cov = (self._search_dim + 2) / 3 * c_cov_full

        # damping for sigma
        self._d_s = (
            1
            + 2 * max(0, jnp.sqrt((self._parents_eff - 1) / (self._search_dim + 1)) - 1)
            + self._c_s
        )
        self._chi = jnp.sqrt(self._search_dim) * (
            1 - 1 / (4 * self._search_dim) + 1 / (21 * self._search_dim**2)
        )

        # threshold for new eigen decomposition - from pyribs
        self._eigen_comput_period = 1
        if delay_eigen_decomposition:
            self._eigen_comput_period = (
                0.5
                * self._population_size
                / (self._search_dim * (self._c_1 + self._c_cov))
            )

    def init(self) -> SepCMAESState:
        """
        Init the CMA-ES algorithm.

        Returns:
            an initial state for the algorithm
        """

        # initial cov vector
        # cov_vector = jnp.eye(self._search_dim)
        cov_vector = jnp.ones(shape=(self._search_dim,))

        # initial inv sqrt of the cov matrix - cov is already diag
        # D = jnp.diag(1 / jnp.sqrt(jnp.diag(cov_vector)))
        D = 1 / jnp.sqrt(cov_vector)

        return SepCMAESState(
            mean=self._mean_init,
            cov_vector=cov_vector,
            sigma=self._init_sigma,
            num_updates=0,
            p_c=jnp.zeros(shape=(self._search_dim,)),
            p_s=jnp.zeros(shape=(self._search_dim,)),
            eigen_updates=0,
            eigenvalues=jnp.ones(shape=(self._search_dim,)),
            D=D,
        )

    @partial(jax.jit, static_argnames=("self",))
    def sample(
        self, cmaes_state: SepCMAESState, random_key: RNGKey
    ) -> Tuple[Genotype, RNGKey]:
        """
        Sample a population.

        Args:
            cmaes_state: current state of the algorithm
            random_key: jax random key

        Returns:
            A tuple that contains a batch of population size genotypes and
            a new random key.
        """
        random_key, subkey = jax.random.split(random_key)

        # N(0, I)
        sample_noise = jax.random.normal(
                key=subkey,
                shape=(
                    self._population_size,
                    self._search_dim,
                )
            )
        print("N(O, I)", sample_noise.shape)

        # Multiply each vector by the covariance diagonal
        # N(O, C)
        sample_noise = jax.vmap(
            lambda n: n * cmaes_state.cov_vector,
            in_axes=0,
        )(sample_noise)

        # sample_noise = jax.vmap( 
        #     lambda n: n * cmaes_state.cov_vector,
        #     # in_axes=0,
        # )(sample_noise) 

        print("N(0, C)", sample_noise.shape)
        
        # Applying noise
        samples = jax.vmap(
            lambda s: s * cmaes_state.sigma + cmaes_state.mean,
            in_axes=0,
        )(sample_noise)

        print("N(m, C)", samples.shape)

        return samples, random_key
    
    @partial(jax.jit, static_argnames=("self",))
    def _update_state(
        self,
        cmaes_state: SepCMAESState,
        sorted_candidates: Genotype,
        weights: jnp.ndarray,
    ) -> SepCMAESState:
        """Updates the state when candidates have already been
        sorted and selected.

        Args:
            cmaes_state: current state of the algorithm
            sorted_candidates: a batch of sorted and selected genotypes
            weights: weights used to recombine the candidates

        Returns:
            An updated algorithm state
        """

        # retrieve elements from the current state
        p_c = cmaes_state.p_c
        p_s = cmaes_state.p_s
        sigma = cmaes_state.sigma
        num_updates = cmaes_state.num_updates
        cov = cmaes_state.cov_vector
        mean = cmaes_state.mean

        eigen_updates = cmaes_state.eigen_updates
        eigenvalues = cmaes_state.eigenvalues
        D = cmaes_state.D

        # update mean by recombination
        old_mean = mean
        mean = weights @ sorted_candidates

        def update_eigen(
            cov: jnp.ndarray, 
            num_updates: int
        ) -> Tuple[int, jnp.ndarray, jnp.ndarray]:

            D = jnp.sqrt(cov)

            # update the eigen value decomposition tracker
            eigen_updates = num_updates

            return eigen_updates, D

        # decomposition of cov
        eigen_updates, D = update_eigen(cov, num_updates)

        z = (1 / sigma) * (mean - old_mean) # Get update vector N(0, C)
        z_w = z / D # Get update vector N(0, I)

        # update evolution paths - cumulation
        p_s = (1 - self._c_s) * p_s + jnp.sqrt(
            self._c_s * (2 - self._c_s) * self._parents_eff
        ) * z_w

        p_s_norm = jnp.linalg.norm(p_s)
        h_sigma = p_s_norm / jnp.sqrt(
            1 - (1 - self._c_s) ** (2 * num_updates)
        ) <= self._chi * (1.4 + 2 / (self._search_dim + 1))

        # Eq 45
        p_c = (1 - self._c_c) * p_c + h_sigma * jnp.sqrt(
            self._c_c * (2 - self._c_c) * self._parents_eff
        ) * z

        delta_h_sigma = (1 - h_sigma) * self._c_c * (2 - self._c_c) # p 28

        # update covariance matrix
        # Eq 46
        # pp_c = jnp.expand_dims(p_c, axis=1)
        
        rank_one = p_c **2 

        y_k = (sorted_candidates - old_mean) / sigma
        # print("y_k", y_k.shape)
        # print("weights", weights.shape)
        # rank_mu = y_k ** 2 * weights 

        rank_mu = jax.vmap(
            lambda y, w: y ** 2 * w,
            # in_axes=(0, 0),
        )(y_k, weights)

        # print("rank_mu", rank_mu.shape)
        rank_mu = jnp.sum(rank_mu, axis=0)

        cov = (
                (
                    1
                    + self._c_1 * delta_h_sigma
                    - self._c_1
                    - self._c_cov 
                )
                * cov
                + self._c_1 * rank_one
                + self._c_cov * rank_mu
        )

        # update step size
        sigma = sigma * jnp.exp(
            (self._c_s / self._d_s) * (p_s_norm / self._chi - 1)
        )

        cmaes_state = SepCMAESState(
            mean=mean,
            cov_vector=cov,
            sigma=sigma,
            num_updates=num_updates + 1,
            p_c=p_c,
            p_s=p_s,
            eigen_updates=eigen_updates,
            eigenvalues=eigenvalues,
            D=D,
        )

        return cmaes_state
