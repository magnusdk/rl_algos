import gym
import jax
from rl_algos.REINFORCE import gym_util, networks
from rl_algos.REINFORCE.agent import Agent


def train_and_print_rewards():
    # Create environment
    env = gym.make("CartPole-v1")

    # Let's seed away randomness and create some PRNG keys
    rng_seed = 1
    env.seed(rng_seed)
    (
        network_init_rng_key,
        agent_rng_key,
    ) = jax.random.split(jax.random.PRNGKey(rng_seed), num=2)

    # Create network and initial parameters
    n_actions = env.action_space.n
    policy_network = networks.get_network(n_actions)
    mock_observation = env.observation_space.sample()
    initial_params = policy_network.init(network_init_rng_key, mock_observation)

    # Create agent
    agent = Agent(policy_network, initial_params, agent_rng_key)

    # Train agent and print sum of rewards for each episode along the way
    for ep in range(2000):
        trajectory = gym_util.gen_trajectory(env, agent)
        agent.update_params(trajectory)
        sum_rewards = sum([r for _, _, r, _ in trajectory])
        print(f"ep={ep}, sum(r)={sum_rewards}")


if __name__ == "__main__":
    train_and_print_rewards()
