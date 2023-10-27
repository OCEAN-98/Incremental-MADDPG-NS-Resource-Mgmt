import pandas as pd
from multi_agent import *
#from Comparison.Positive_reward_Env import *
#from Comparison.Pos_MADDPG import *
from Environment import *
from MADDPG import *
import os
import json

network = NetworkSetup(incremental_number=3)
network.reset()
alpha = 0.005
beta = 0.005
current_dir = os.path.abspath(os.path.dirname(__file__))

chkpt_dir = os.path.join(current_dir, 'models')
load_dir = os.path.join(current_dir, 'models')
n_agents = 3
actor_dims = [10, 10, 10]  # each 8 represent the observation of a slice (4 resource + 4 requests)
critic_dims = 7 + (5 + 1) * 6 # 5 + 1 # len of x, len of edge_attr
n_actions = 5
Episodes = 1
Observe = 10000
Explore = 1000
Train = 2000
memory_length = 2000
reward_log = []  # used for drawing the training figure, collect variation
noise_scale = 1 # IMPORTANT: don't set it smaller than 1. 
Time_results = []

maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, chkpt_dir, load_dir, alpha, beta)
memory = MultiAgentReplayBuffer(memory_length, critic_dims, actor_dims, n_actions, n_agents, batch_size=200)
for epi in range(Episodes):
    time_record = []
    power_record = []
    state, actor_state = network.reset()
    while network.timer < Observe:
        action = maddpg_agents.choose_action(actor_state, noise_scale) # IMPORTANT: 1. use neural network is better than random in complex scenario; 2.
                                                                        #           2. in the exploring period, it's good to have noise as well; 
        
        actor_state, actor_next_state, state, next_state, reward = network.step(action, time_record, power_record, Time_results)
        memory.store_transition( actor_state, state, action, reward, actor_next_state, next_state)  # during this period, we will only storage without pick out
        if network.timer % 1 == 0: 
            log = 'time_step {}/ process {}/  reward {}/'.format(network.timer, 'observe', reward)
            print(log)
            f = open(
                current_dir + '/results/multiagent_data_reward', 'a')
            f.write(log + '\n')
            f.close()

        # if network.timer % 10 == 0:
        #     maddpg_agents.save_checkpoint() # here we will store the models, which will be used in incremental learning

        if network.timer % 1 == 0:
            reward_log.append(reward)

        if network.timer % 1 == 0:
            data_reward = {'reward': reward_log}
            data_reward = pd.DataFrame(data_reward)
            data_reward.to_csv(
                current_dir + '/results/data_reward')
    
    index = 1       
    maddpg_agents.load_checkpoint()
    while network.timer >= Observe and network.timer <  (Observe + Explore + Train):

        noise_scale = 0.001 # max(0.001, (Explore - index) / Explore)
        index += 1

        
        action = maddpg_agents.choose_action(actor_state, noise_scale)
    
        actor_state, actor_next_state, state, next_state, reward = network.step(action, time_record, power_record, Time_results)
        memory.store_transition(actor_state, state, action, reward, actor_next_state, next_state)  


        maddpg_agents.learn(memory, memory_length)

        if network.timer <= Observe + Explore:
            process = 'Explore'
        else:
            process = 'Train'

        if network.timer % 1 == 0:  
            sss = 'time_step {}/ process {}/ reward {}/'.format(network.timer, process, reward)
            print(sss)
            print(noise_scale)
            f = open(current_dir + '/results/multiagent_data_reward', 'a')
            f.write(sss + '\n')
            f.close()

        if network.timer % 1 == 0 and network.timer <= Observe + Explore + Train - 200:
            maddpg_agents.save_checkpoint()

        if network.timer % 1 == 0:
            reward_log.append(reward)

        if network.timer % 1 == 0:
            data_reward = {'reward': reward_log}
            data_reward = pd.DataFrame(data_reward)
            data_reward.to_csv(current_dir + '/results/data_reward')