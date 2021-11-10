import haiku as hk
import jax
import jax.numpy as jnp
import rlax
from rl_algos.REINFORCE.trajectory import Trajectory


def _jitted_select_action(network):
    def select_action(params, rng_key, obs):
        apply_rng_key, action_sample_rng_key = jax.random.split(rng_key, num=2)
        action_probs = network.apply(params, apply_rng_key, obs)

        # Sample randomly using the calculated action probabilities
        a = rlax.categorical_sample(action_sample_rng_key, action_probs)
        return a

    return jax.jit(select_action)


def _jitted_update_step(network, discount, alpha):
    def update_step(rng_key, params, t, rewards, state, a):
        # Calculate discounted return
        discounts = discount ** jnp.arange(len(rewards))
        G = jnp.sum(discounts * jnp.array(rewards))

        # Define the gradient of the log-probability of choosing action a
        @jax.grad  # Behold (one of) the power(s) of JAX!
        def log_prob_of_a_gradient(params):
            action_probs = network.apply(params, rng_key, state)
            prob_of_a = action_probs[a]
            return jnp.log(prob_of_a)

        # Update parameters
        def update_params(old_params, gradient):
            update = alpha * (discount ** t) * G * gradient
            return old_params + update

        # Params (and thus also gradient) is a FlatMapping (pytree), and we must therefore map the update over each leaf.
        new_params = jax.tree_util.tree_map(
            update_params, params, log_prob_of_a_gradient(params)
        )
        return new_params

    return jax.jit(update_step)


class Agent:
    def __init__(
        self,
        network: hk.Transformed,
        initial_params: hk.Params,
        rng_key,
        discount: float = 0.99,
        alpha: float = 0.001,
    ):
        self.params = initial_params
        self.rng_key = rng_key

        # Use jitted (just-in-time compilated) functions because jit makes them run faster.
        self._select_action = _jitted_select_action(network)
        self._update_step = _jitted_update_step(network, discount, alpha)

    def select_action(self, obs):
        self.rng_key, sub_rng_key = jax.random.split(self.rng_key)
        a = self._select_action(self.params, sub_rng_key, obs)
        return int(a)

    def update_params(self, trajectory: Trajectory):
        # For every step t in trajectory, perform update.
        # See REINFORCE algorithm.
        for t in range(len(trajectory)):
            self.rng_key, sub_rng_key = jax.random.split(self.rng_key, num=2)
            s, a, _, _ = trajectory[t]
            self.params = self._update_step(
                sub_rng_key, self.params, t, trajectory[t:].r, s, a
            )
