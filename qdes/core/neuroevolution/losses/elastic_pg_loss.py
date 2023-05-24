""" Implements a function to create critic and actor losses for the TD3 algorithm with L2 regularization on the distance between the actor and the center of an ES"""

from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from qdes.core.neuroevolution.buffers.buffer import Transition
from qdax.types import Action, Observation, Params, RNGKey


def elastic_td3_loss_fn(
    policy_fn: Callable[[Params, Observation], jnp.ndarray],
    critic_fn: Callable[[Params, Observation, Action], jnp.ndarray],
    reward_scaling: float,
    discount: float,
    noise_clip: float,
    policy_noise: float,
    elastic_pull: float,
) -> Tuple[
    Callable[[Params, Params, Transition], jnp.ndarray],
    Callable[[Params, Params, Params, Transition, RNGKey], jnp.ndarray],
]:
    """Creates the loss functions for TD3.

    Args:
        policy_fn: forward pass through the neural network defining the policy.
        critic_fn: forward pass through the neural network defining the critic.
        reward_scaling: value to multiply the reward given by the environment.
        discount: discount factor.
        noise_clip: value that clips the noise to avoid extreme values.
        policy_noise: noise applied to smooth the bootstrapping.

    Returns:
        Return the loss functions used to train the policy and the critic in TD3.
    """

    @jax.jit
    def _policy_loss_fn(
        policy_params: Params,
        critic_params: Params,
        transitions: Transition,
        es_center: Params,
    ) -> jnp.ndarray:
        """Policy loss function for TD3 agent"""
        # print("Loss - Actor", jax.tree_map(lambda x: x.shape, policy_params))
        # print("Loss - ES", jax.tree_map(lambda x: x.shape, es_center))

        action = policy_fn(policy_params, transitions.obs)
        q_value = critic_fn(
            critic_params, obs=transitions.obs, actions=action  # type: ignore
        )
        q1_action = jnp.take(q_value, jnp.asarray([0]), axis=-1)
        policy_loss = -jnp.mean(q1_action)

        # L2 regularization on the distance between the actor and the center of an ES
        elastic_loss = jax.tree_util.tree_map(
            lambda x, y: ((x - y) ** 2) * elastic_pull, 
            policy_params,
            es_center
        )
        # print("Loss - Elastic loss", jax.tree_map(lambda x: x.shape, elastic_loss))

        # Sum it
        elastic_loss_sum = jax.tree_util.tree_reduce(
            lambda x, y: x.sum() + y.sum(), 
            elastic_loss
        )
        policy_loss += elastic_loss_sum
        return policy_loss

    @jax.jit
    def _critic_loss_fn(
        critic_params: Params,
        target_policy_params: Params,
        target_critic_params: Params,
        transitions: Transition,
        random_key: RNGKey,
    ) -> jnp.ndarray:
        """Critics loss function for TD3 agent"""
        noise = (
            jax.random.normal(random_key, shape=transitions.actions.shape)
            * policy_noise
        ).clip(-noise_clip, noise_clip)

        next_action = (
            policy_fn(target_policy_params, transitions.next_obs) + noise
        ).clip(-1.0, 1.0)
        next_q = critic_fn(  # type: ignore
            target_critic_params, obs=transitions.next_obs, actions=next_action
        )
        next_v = jnp.min(next_q, axis=-1)
        target_q = jax.lax.stop_gradient(
            transitions.rewards * reward_scaling
            + (1.0 - transitions.dones) * discount * next_v
        )
        q_old_action = critic_fn(  # type: ignore
            critic_params,
            obs=transitions.obs,
            actions=transitions.actions,
        )
        q_error = q_old_action - jnp.expand_dims(target_q, -1)

        # Better bootstrapping for truncated episodes.
        q_error = q_error * jnp.expand_dims(1.0 - transitions.truncations, -1)

        # compute the loss
        q_losses = jnp.mean(jnp.square(q_error), axis=-2)
        q_loss = jnp.sum(q_losses, axis=-1)

        return q_loss

    return _policy_loss_fn, _critic_loss_fn