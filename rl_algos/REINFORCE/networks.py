import jax
import jax.numpy as jnp


def get_network():
    def logistic(x):
        return 1 / (1 + jnp.exp(-x))

    def network(params, state):
        p = logistic(jnp.dot(params, state))
        return jnp.array([p, 1 - p])

    return jax.jit(network)
