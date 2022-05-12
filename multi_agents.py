import torch

from ddpg_agent import DDPG_Agent
from device import device
from replay_buffer import ReplayBuffer

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64  # minibatch size


class MultiAgents(object):
    def __init__(self, n_agents, state_dim, action_dim, seed=42):
        self._agents = [DDPG_Agent(state_dim, action_dim, agent_id, n_agents) for agent_id in range(n_agents)]
        self._replay_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        self._n_agents = n_agents
        self._action_dim = action_dim

    def _get_exp_values_by_agent(self, values, agent_id):
        return values[agent_id::self._n_agents, :]

    def save_models(self):
        for i, agent in enumerate(self._agents):
            torch.save(agent._actor_target.state_dict(), 'actor_{}.pth'.format(i))

    def act(self, obs_all_agents, i_episode, noise=0.05):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, i_episode, noise) for agent, obs in zip(self._agents, obs_all_agents)]
        return actions

    def reset(self):
        for agent in self._agents:
            agent.reset()

    def _actions_local(self, states, current_agent):
        actions = torch.empty(
            (BATCH_SIZE, self._n_agents, self._action_dim),
            device=device)
        for agent in self._agents:
            action = agent.local_act(states[:, agent.agent_id])
            if agent.agent_id != current_agent.agent_id:
                action = action.detach()
            actions[:, agent.agent_id] = action
        return actions

    def _actions_target(self, states):
        actions = torch.empty(
            (BATCH_SIZE, self._n_agents, self._action_dim),
            device=device)
        for agent in self._agents:
            with torch.no_grad():
                actions[:, agent.agent_id] = agent.target_act(states[:, agent.agent_id])
        return actions

    def _learn(self):

        for agent in self._agents:
            experiences = self._replay_buffer.sample()

            states = torch.from_numpy(experiences['states']).float().to(device)
            obs = states.view(BATCH_SIZE, -1)
            actions = torch.from_numpy(experiences['actions']).float().to(device)
            obs = obs.view(BATCH_SIZE, -1)

            next_states = torch.from_numpy(experiences['next_states']).float().to(device)
            next_obs = next_states.view(BATCH_SIZE, -1)

            rewards = torch.from_numpy(experiences['rewards']).float().to(device)
            dones = torch.from_numpy(experiences['dones']).float().to(device)

            next_actions = self._actions_target(states)
            local_actions = self._actions_local(states, agent)

            agent.critic_learn(obs, actions.view(BATCH_SIZE, -1), next_actions.view(BATCH_SIZE, -1), next_obs,
                               rewards[:, agent.agent_id], dones)
            agent.actor_learn(obs, local_actions.view(BATCH_SIZE, -1))

        for agent in self._agents:
            agent.soft_update_target_weights()

    def step(self, states, actions, rewards, next_states, dones):
        experience = (states, actions, rewards, next_states, dones)
        self._replay_buffer.add(experience)
        # Learn, if enough samples are available in memory
        if len(self._replay_buffer) > BATCH_SIZE:
            # if self._t % self._update_interval == 0:
            #     for _ in range(self._n_updates):
            self._learn()
