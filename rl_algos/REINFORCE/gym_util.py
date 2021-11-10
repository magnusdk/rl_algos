import gym
from rl_algos.REINFORCE.agent import Agent
from rl_algos.REINFORCE.trajectory import Trajectory


def gen_trajectory(env: gym.Env, agent: Agent) -> Trajectory:
    s = env.reset()
    done = False
    trajectory = Trajectory(traj=[])
    while not done:
        a = agent.select_action(s)
        next_s, r, done, _ = env.step(a)
        trajectory.append(s, a, r, next_s)
        s = next_s

    return trajectory
