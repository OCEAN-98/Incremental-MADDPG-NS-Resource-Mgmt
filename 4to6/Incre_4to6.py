import torch
from multi_agent import *

current_dir = os.path.abspath(os.path.dirname(__file__))

chkpt_dir = os.path.join(current_dir, 'models')
load_dir = '.../4_slices/models'

n_actions = 5
alpha=0.01
beta=0.01
conv1=64
conv2=64
fc1 = 32
fc2 = 32

gamma=0.9
tau=0.01
actor_input = 10
input_size = 7 + (5 + 1) * 6
agent_0_actor = ActorNetwork(alpha, actor_input, fc1, fc2, n_actions, chkpt_dir=chkpt_dir, load_dir=load_dir, name= 'agent_0_actor') # only use load
agent_0_actor.load()
agent_0_actor.save()
agent_1_actor = ActorNetwork(alpha, actor_input, fc1, fc2, n_actions, chkpt_dir=chkpt_dir, load_dir=load_dir, name= 'agent_1_actor')
agent_1_actor.load()
agent_1_actor.save()
agent_2_actor = ActorNetwork(alpha, actor_input, fc1, fc2, n_actions, chkpt_dir=chkpt_dir, load_dir=load_dir, name= 'agent_2_actor')
agent_2_actor.load()
agent_2_actor.save()
agent_3_actor = ActorNetwork(alpha, actor_input, fc1, fc2, n_actions, chkpt_dir=chkpt_dir, load_dir=load_dir, name= 'agent_3_actor')
agent_3_actor.load()
agent_3_actor.save()

agent_0_critic = CriticNetwork(beta, input_size, conv1, conv2, chkpt_dir=chkpt_dir, load_dir=load_dir, name= 'agent_0_critic')
agent_0_critic.load()
agent_0_critic.save()
agent_1_critic = CriticNetwork(beta, input_size, conv1, conv2, chkpt_dir=chkpt_dir, load_dir=load_dir, name= 'agent_1_critic')
agent_1_critic.load()
agent_1_critic.save()
agent_2_critic = CriticNetwork(beta, input_size, conv1, conv2, chkpt_dir=chkpt_dir, load_dir=load_dir, name= 'agent_2_critic')
agent_2_critic.load()
agent_2_critic.save()
agent_3_critic = CriticNetwork(beta, input_size, conv1, conv2, chkpt_dir=chkpt_dir, load_dir=load_dir, name= 'agent_3_critic')
agent_3_critic.load()
agent_3_critic.save()

models = [agent_0_actor, agent_1_actor, agent_2_actor, agent_3_actor]

# initialize an empty dictionary to hold the average weights
avg_weights = {}

# loop over all keys in the state_dict
for key in models[0].state_dict().keys():
    # gather the same weights from all models
    weights = [model.state_dict()[key] for model in models]
    
    # compute the average weight
    avg_weight = torch.mean(torch.stack(weights), dim=0)
    
    # insert the average weight into the dictionary
    avg_weights[key] = avg_weight

# load the average weights into a new model
agent_4_actor = ActorNetwork(alpha, actor_input, fc1, fc2, n_actions, chkpt_dir=chkpt_dir, load_dir=load_dir, name= 'agent_4_actor')
agent_4_actor.load_state_dict(avg_weights)
agent_4_actor.save()
agent_5_actor = ActorNetwork(alpha, actor_input, fc1, fc2, n_actions, chkpt_dir=chkpt_dir, load_dir=load_dir, name= 'agent_5_actor')
agent_5_actor.load_state_dict(avg_weights)
agent_5_actor.save()

# repeat the process for the critic network
models = [agent_0_critic, agent_1_critic, agent_2_critic, agent_3_critic]

avg_weights = {}

for key in models[0].state_dict().keys():
    weights = [model.state_dict()[key] for model in models]
    avg_weight = torch.mean(torch.stack(weights), dim=0)
    avg_weights[key] = avg_weight

agent_4_critic = CriticNetwork(beta, input_size, conv1, conv2, chkpt_dir=chkpt_dir, load_dir=load_dir, name= 'agent_4_critic')
agent_4_critic.load_state_dict(avg_weights)
agent_4_critic.save()
agent_5_critic = CriticNetwork(beta, input_size, conv1, conv2, chkpt_dir=chkpt_dir, load_dir=load_dir, name= 'agent_5_critic')
agent_5_critic.load_state_dict(avg_weights)
agent_5_critic.save()



