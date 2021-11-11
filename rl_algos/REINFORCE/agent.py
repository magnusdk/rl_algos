import jax
import jax.numpy as jnp
import rlax
from rl_algos.REINFORCE.trajectory import Trajectory


def _jitted_select_action(network):
    def select_action(params, rng_key, obs):
        action_probs = network(params, obs)

        # Sample randomly using the calculated action probabilities
        return rlax.categorical_sample(rng_key, action_probs)

    return jax.jit(select_action)


def _jitted_update_step(network, discount, alpha):
    def update_step(params, t, rewards_from_t, state, a):
        # Calculate discounted return
        discounts = discount ** jnp.arange(len(rewards_from_t))
        G = jnp.sum(discounts * jnp.array(rewards_from_t))

        # Define the gradient of the log-probability of choosing action a
        @jax.grad  # Behold (one of) the power(s) of JAX!
        def log_prob_of_a_gradient(params):
            action_probs = network(params, state)
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
        network,
        initial_params,
        rng_key,
        discount: float = 0.99,
        alpha: float = 0.01,
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
            s, a, _, _ = trajectory[t]
            rewards_from_t = trajectory[t:].r
            self.params = self._update_step(self.params, t, rewards_from_t, s, a)
