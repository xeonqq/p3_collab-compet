import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from unityagents import UnityEnvironment

from multi_agents import MultiAgents


def get_next_states(env_info):
    next_state = env_info.vector_observations  # get the next state
    return next_state


def get_rewards(env_info):
    rewards = env_info.rewards  # get the reward
    rewards = np.asarray(rewards)
    rewards = rewards[..., np.newaxis]
    return rewards


def get_dones(env_info):
    dones = env_info.local_done  # see if episode has finished
    dones = np.asarray(dones)
    dones = dones[..., np.newaxis]
    return dones


def get_env_step_results(env_info):
    return get_next_states(env_info), get_rewards(env_info), get_dones(env_info)


class Environment(object):
    def __init__(self, env):
        self._env = env
        self._brain_name = env.brain_names[0]
        self._brain = env.brains[self._brain_name]
        self._env_info = env.reset(train_mode=True)[self._brain_name]

        # number of agents in the environment
        print('Number of agents:', len(self._env_info.agents))

        # number of actions
        self._action_size = self._brain.vector_action_space_size
        print('Number of actions:', self._action_size)

        # examine the state space
        states = self._env_info.vector_observations
        self._state_size = states.shape[1]
        print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], self._state_size))
        print('The state for the first agent looks like:', states[0])
        self._multi_agents = MultiAgents(len(self._env_info.agents), self._state_size, self._action_size)
        self._share_rewards = False

    def run_model(self, actor_models, num_episode=3, steps_per_episode=1000):
        self._multi_agents.load_actor_models(actor_models)
        scores = []
        for i in range(num_episode):
            env_info = self._env.reset(train_mode=False)[self._brain_name]
            states = get_next_states(env_info)
            score = 0
            for j in range(steps_per_episode):
                actions = self._multi_agents.act(states, i, 0)
                env_info = self._env.step(actions)[self._brain_name]  # send the action to the environment
                next_states, rewards, dones = get_env_step_results(env_info)
                score += np.mean(rewards)  # update the score
                states = next_states  # roll over the state to next time step
                if dones.any():
                    break
            scores.append(score)  # save most recent score
            print("Episode: {}, score: {}".format(i, score))
        return scores


    def close(self):
        self._env.close()

    def train(self, min_score, n_episodes=10000, max_t=1000):
        """Deep Q-Learning.
            Params
            ======
                n_episodes (int): maximum number of training episodes
                max_t (int): maximum number of timesteps per episode
                eps_start (float): starting value of epsilon, for epsilon-greedy action selection
                eps_end (float): minimum value of epsilon
                eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
            """
        scores = []  # list containing scores from each episode
        moving_average_scores = []
        scores_window = deque(maxlen=100)  # last 100 scores
        noise = 0.5

        print("use_noise:", noise)

        for i_episode in range(1, n_episodes + 1):
            env_info = self._env.reset(train_mode=True)[self._brain_name]
            states = get_next_states(env_info)
            score = np.zeros((2,1))
            for t in range(max_t):
                actions = self._multi_agents.act(states, i_episode, noise)
                env_info = self._env.step(actions)[self._brain_name]  # send the action to the environment
                next_states, rewards, dones = get_env_step_results(env_info)
                feed_in_rewards = rewards
                if self._share_rewards:
                    feed_in_rewards = np.mean(rewards)*np.ones((2,1))
                self._multi_agents.learn_step(states, np.asarray(actions), feed_in_rewards, next_states, dones)
                score += rewards  # update the score
                states = next_states  # roll over the state to next time step
                if dones.any():
                    break
            self._multi_agents.step(i_episode)
            max_score=np.max(score)

            scores_window.append(max_score)  # save most recent score
            scores.append(max_score)  # save most recent score
            mean_score = np.mean(scores_window)
            moving_average_scores.append(mean_score)
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_score))
            if i_episode > 100 and mean_score >= min_score:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                                             mean_score))
                self._multi_agents.save_models()
                break

        return scores, moving_average_scores


def plot_scores(scores, moving_average_scores):
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores, label='scores')
    plt.plot(np.arange(len(moving_average_scores)), moving_average_scores, label='moving average')
    plt.legend()
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig("maddpg_score.png")
    plt.show()


def train(min_score, unity_env_file='Tennis_Linux/Tennis.x86_64'):
    env = Environment(UnityEnvironment(file_name=unity_env_file))
    scores, moving_average_scores = env.train(min_score)
    plot_scores(scores, moving_average_scores)
    env.close()


if __name__ == "__main__":
    train(min_score=0.5)
