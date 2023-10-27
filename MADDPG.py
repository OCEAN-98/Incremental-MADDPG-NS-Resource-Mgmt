import pandas as pd
import numpy as np
import torch
from multi_agent import *
from Environment import *
# here we make a bold attempt to ignore actions of all actors and take global state as a compensation

class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=1.0, sigma=0.08):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn()
        self.state = x + dx
        return self.state

class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, chkpt_dir, load_dir, alpha, beta):
        self.agents = []
        self.n_agents = n_agents
        self.noise = OUNoise(actor_dims)

        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims, n_actions, agent_idx,
                                     chkpt_dir=chkpt_dir, load_dir=load_dir, n_agents=n_agents, alpha=alpha, beta=beta))

    def save_checkpoint(self):
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, actor_state,  noise_scale):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(actor_state[agent_idx])

            if noise_scale is not None:
                noise_0 = self.noise.sample()
                noise_1 = self.noise.sample()
                noise_2 = self.noise.sample()
                noise_3 = self.noise.sample()
                noise_4 = self.noise.sample()
                noise = [noise_0 * noise_scale, noise_1 * noise_scale, noise_2 * noise_scale, noise_3 * noise_scale, noise_4 * noise_scale]
                # print(noise)
                for act in range(len(action)):
                    if 0 < action[act] + noise[act] < 1:
                        action[act] += noise[act]
                    else:
                        continue
            actions.append(action)
            # print(actions)
        return actions

    def learn(self, memory, memory_length):
        actor_states, states, actions, rewards, actor_next_states, next_states = memory.sample_buffer(memory_length)
        device = self.agents[0].actor.device
        
        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.float).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(device)

        all_agents_next_actions = []
        all_agents_next_mu_actions = []
        current_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            actor_next_state = torch.tensor(actor_next_states[agent_idx],
                                  dtype=torch.float).to(device)

            next_pi = agent.target_actor.forward(actor_next_state)

            all_agents_next_actions.append(next_pi)
            actor_state = torch.tensor(actor_states[agent_idx],
                                 dtype=torch.float).to(device)
            pi = agent.actor.forward(actor_state)
            all_agents_next_mu_actions.append(pi)
            current_agents_actions.append(actions[agent_idx])

        for agent_idx, agent in enumerate(self.agents):
            agent.actor.optimizer.zero_grad()

        for agent_idx, agent in enumerate(self.agents):      
            next_critic_value = agent.target_critic.forward(next_states).flatten()
            critic_value = agent.critic.forward(states).flatten()
            target = rewards[:, agent_idx] + agent.gamma * next_critic_value
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            actor_loss = agent.critic.forward(states).flatten()
            actor_loss = -torch.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)

        for agent_idx, agent in enumerate(self.agents):
            agent.actor.optimizer.step()
            agent.update_network_parameters()
