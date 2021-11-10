import haiku as hk
import jax


def get_network(n_actions) -> hk.Transformed:
    def network(x):
        f = hk.Sequential(
            [
                hk.Linear(32),
                jax.nn.relu,
                hk.Linear(n_actions),
                jax.nn.softmax,
            ]
        )
        return f(x)

    return hk.transform(network)
