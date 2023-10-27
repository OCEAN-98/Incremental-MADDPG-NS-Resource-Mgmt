import os
import torch.optim as optim
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiAgentReplayBuffer:
    def __init__(self, max_size, critic_dims, actor_dims, n_actions, n_agents, batch_size):
        self.mem_size = max_size
        self.mem_cntr = 0 # counter
        self.actor_dims = actor_dims # dimentions
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.state_memory = np.zeros((self.mem_size, critic_dims))  # if mem_size = 10000，there will be 10000 []
        self.next_state_memory = np.zeros((self.mem_size, critic_dims))
        self.reward_memory = np.zeros((self.mem_size, n_agents))  # 10000 reward
        # self.action_memory = []  # batch_size * agent number * action number
        self.init_actor_memory()

    def init_actor_memory(self):
        self.action_memory = []
        self.actor_state_memory = []  # record state of all agent
        self.actor_next_state_memory = []

        for i in range(self.n_agents):
            self.action_memory.append(np.zeros((self.mem_size, self.n_actions)))
            self.actor_state_memory.append(np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_next_state_memory.append(np.zeros((self.mem_size, self.actor_dims[i])))

    def store_transition(self, actor_state, state, action, reward, actor_next_state, next_state):
        index = self.mem_cntr % self.mem_size
        for agent_idx in range(self.n_agents):  # get actor's state from obs
            self.actor_state_memory[agent_idx][index] = actor_state[agent_idx]
            self.actor_next_state_memory[agent_idx][index] = actor_next_state[agent_idx]
            self.action_memory[agent_idx][index] = action[agent_idx]

        self.state_memory[index] = state
        self.next_state_memory[index] = next_state

        self.reward_memory[index] = reward
        self.mem_cntr += 1

    def sample_buffer(self, memory_length):
        batch = np.random.choice(memory_length, self.batch_size, replace=False)

        states = [self.state_memory[i] for i in batch]
        next_states = [self.next_state_memory[i] for i in batch]
        rewards = [self.reward_memory[i] for i in batch]

        actions = []
        actor_states = []
        actor_next_states = []

        for agent_index in range(self.n_agents):
            actions.append([self.action_memory[agent_index][i] for i in batch] )
            actor_states.append([self.actor_state_memory[agent_index][i] for i in batch])
            actor_next_states.append([self.actor_next_state_memory[agent_index][i] for i in batch])


        return actor_states, states, actions, rewards, actor_next_states, next_states


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_size, fc1, fc2, chkpt_dir, load_dir, name):
        super(CriticNetwork, self).__init__()
        self.chkpt_file = os.path.join(chkpt_dir, name)
        self.load_dir = os.path.join(load_dir, name)

        self.fc1 = nn.Linear(input_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = torch.tanh(self.fc3(x))
        return q

    def save(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load(self):
        self.load_state_dict(torch.load(self.load_dir))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_size, f1_dims, f2_dims, action_number, chkpt_dir, load_dir, name):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)
        self.load_dir = os.path.join(load_dir, name)
        self.fc1 = nn.Linear(input_size, f1_dims)
        self.fc2 = nn.Linear(f1_dims, f2_dims)
        self.pi = nn.Linear(f2_dims, action_number)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        pi = torch.sigmoid(self.pi(x))
        return pi

    def save(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load(self):
        self.load_state_dict(torch.load(self.load_dir))

class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, agent_idx, chkpt_dir, load_dir, n_agents,
                    alpha, beta, conv1=64, conv2=64, fc1=32, fc2=32, gamma=0.99, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, chkpt_dir=chkpt_dir, load_dir=load_dir, name=self.agent_name+'_actor')
        self.critic = CriticNetwork(beta, critic_dims, conv1, conv2, chkpt_dir=chkpt_dir, load_dir=load_dir, name=self.agent_name+'_critic')
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, chkpt_dir=chkpt_dir, load_dir=load_dir, name=self.agent_name+'_target_actor')
        self.target_critic = CriticNetwork(beta, critic_dims, conv1, conv2, chkpt_dir=chkpt_dir, load_dir=load_dir, name=self.agent_name+'_target_critic')

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.actor.device)
        actions = self.actor.forward(state)

        return actions.detach().cpu().numpy()[0]

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        self.actor.save()
        self.target_actor.save()
        self.critic.save()
        self.target_critic.save()

    def load_models(self):
        self.actor.load()
        self.critic.load()
