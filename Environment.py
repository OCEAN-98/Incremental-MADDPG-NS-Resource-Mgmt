import random
import math
from typing import Optional, Union
import copy
import numpy as np
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


def store_data(data_list, new_data, number):
    data_list.append(new_data)

    if len(data_list) > number:
        del data_list[0]

def bottom_avg(data_list):
    new_data_list = copy.deepcopy(data_list)
    new_data_list.sort()
    n = len(new_data_list)
    percent = int(n * 0.01)
    avg = sum(new_data_list[:percent]) / percent
    minimun = new_data_list[0]
    return avg

class Vertex:
    def __init__(self, key, computing_resource):
        self.id = key
        self.computing_resource = computing_resource
        self.connected_node = {}  # {key: bandwidth}

    def __str__(self):
        return 'Node' + str(self.id) + ' has ' + str(
            self.computing_resource) + ' computing resource, and connected with ' + \
               'Node ' + str([x.id for x in self.connected_node]) + ' with distance of ' + str(
            x.value for x in self.connected_node)

    def get_id(self):
        return self.id

    def get_resource(self):
        return self.computing_resource

    def add_neighbor(self, nbr, bw):
        if nbr not in self.connected_node:
            self.connected_node[nbr] = bw

    def get_bandwidth(self, nbr):
        return self.connected_node[nbr]

    def change_resource(self, variation):
        self.computing_resource += variation

    def change_bandwidth(self, nbr, variation):
        self.connected_node[nbr] += variation

    def get_neighbors(self):
        return self.connected_node.keys()


class Network:
    def __init__(self):
        self.node_list = {}
        self.total_node_number = 0

    def get_node_list(self):
        return self.node_list.keys()

    def add_node(self, id, computing_resource):
        self.node_list[id] = Vertex(id, computing_resource)
        self.total_node_number += 1

    def add_edge(self, node_1, node_2, bw):
        if node_1 in self.node_list and node_2 in self.node_list:
            self.node_list[node_1].add_neighbor(node_2, bw)
            self.node_list[node_2].add_neighbor(node_1, bw)

image = 'Figures/Slices.png'

class NetworkSetup():  # Obs, action architecture
    def __init__(self, incremental_number):
        self.total_node_number = 6  # initialized number
        self.network = None
        self.requests = {}  # { Node_0: [2 vnf data_size in Mbits, 2 computing_req in Gcycles], Node_1: [], ... ]
        self.incremental_number = incremental_number
        self.timer = 0
        self.recorder = {} # {timer: occupying reousrces}

    def build_network(self):
        self.network = Network()
        self.network.add_node("Node0", 100) # GHz
        self.network.add_node("Node1", 100)
        self.network.add_node("Node2", 100)
        self.network.add_node("Node3", 100)
        self.network.add_node("Node4", 100)
        self.network.add_node("Node5", 100)

        self.network.add_edge('Node0', 'Node1', 100) # Gbps
        self.network.add_edge('Node0', 'Node3', 100)
        self.network.add_edge('Node1', 'Node2', 100)
        self.network.add_edge('Node1', 'Node4', 100)
        self.network.add_edge('Node2', 'Node5', 100)
        self.network.add_edge('Node3', 'Node4', 100)
        self.network.add_edge('Node4', 'Node5', 100)

    def generate_requests(self): 
        self.requests = {}
        if self.incremental_number == 3:
            self.requests['Node4'] = [random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), 'Node4', 'Node1', 'Node2'] # Gbits, Gcycles
            self.requests['Node3'] = [random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), 'Node3', 'Node0', 'Node1']
            self.requests['Node0'] = [random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), 'Node0', 'Node3', 'Node4']
        elif self.incremental_number == 4:
            self.requests['Node4'] = [random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), 'Node4', 'Node1', 'Node2']
            self.requests['Node3'] = [random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), 'Node3', 'Node0', 'Node1']
            self.requests['Node0'] = [random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), 'Node0', 'Node3', 'Node4']
            self.requests['Node1'] = [random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), 'Node1', 'Node2', 'Node5']
        elif self.incremental_number == 5:
            self.requests['Node4'] = [random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), 'Node4', 'Node1', 'Node2']
            self.requests['Node3'] = [random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), 'Node3', 'Node0', 'Node1']
            self.requests['Node0'] = [random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), 'Node0', 'Node3', 'Node4']
            self.requests['Node1'] = [random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), 'Node1', 'Node2', 'Node5']
            self.requests['Node2'] = [random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), 'Node2', 'Node5', 'Node4']
        elif self.incremental_number == 6:
            self.requests['Node4'] = [random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), 'Node4', 'Node1', 'Node2']
            self.requests['Node3'] = [random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), 'Node3', 'Node0', 'Node1']
            self.requests['Node0'] = [random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), 'Node0', 'Node3', 'Node4']
            self.requests['Node1'] = [random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), 'Node1', 'Node2', 'Node5']
            self.requests['Node2'] = [random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), 'Node2', 'Node5', 'Node4']
            self.requests['Node5'] = [random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), random.randrange(10, 20), 'Node5', 'Node4', 'Node1']


    def generate_state(self):  # Convert Network into state; used in step and reset | 7 + (4 + 1) * 6
        data_sample = []
        data_sample.append(self.network.node_list['Node0'].get_bandwidth('Node1') / 100)
        data_sample.append(self.network.node_list['Node0'].get_bandwidth('Node3') / 100)
        data_sample.append(self.network.node_list['Node1'].get_bandwidth('Node2') / 100)
        data_sample.append(self.network.node_list['Node1'].get_bandwidth('Node4') / 100)
        data_sample.append(self.network.node_list['Node2'].get_bandwidth('Node5') / 100)
        data_sample.append(self.network.node_list['Node3'].get_bandwidth('Node4') / 100)
        data_sample.append(self.network.node_list['Node4'].get_bandwidth('Node5') / 100) 
        for i in range(self.total_node_number):
            individual_x = []
            if 'Node' + str(i) in self.requests:
                individual_x = self.requests['Node' + str(i)][:-3]
                normalized_individual_x = [i/100 for i in individual_x]
                normalized_individual_x.append(self.network.node_list['Node' + str(i)].computing_resource / 100)
                data_sample.extend(normalized_individual_x)
            else:
                normalized_individual_x = [0, 0, 0, 0, 0]
                normalized_individual_x.append(self.network.node_list['Node' + str(i)].computing_resource / 100)
                data_sample.extend(normalized_individual_x)

        return data_sample

    def read_actor_state(self): 
        actor_states = []
        actor_state_1 = []
        actor_state_2 = []
        actor_state_3 = []
        actor_state_4 = []
        actor_state_5 = []
        actor_state_6 = []

        for information in self.requests['Node4'][:5]:
            actor_state_1.append(information)
        actor_state_1 += [self.network.node_list['Node4'].computing_resource, self.network.node_list['Node1'].computing_resource, self.network.node_list['Node2'].computing_resource, self.network.node_list['Node4'].get_bandwidth('Node1'), self.network.node_list['Node1'].get_bandwidth('Node2')]
        for information in self.requests['Node3'][:5]:
            actor_state_2.append(information)
        actor_state_2 += [self.network.node_list['Node3'].computing_resource, self.network.node_list['Node0'].computing_resource, self.network.node_list['Node1'].computing_resource, self.network.node_list['Node3'].get_bandwidth('Node0'), self.network.node_list['Node0'].get_bandwidth('Node1')]
        for information in self.requests['Node0'][:5]:
            actor_state_3.append(information)
        actor_state_3 += [self.network.node_list['Node0'].computing_resource, self.network.node_list['Node3'].computing_resource, self.network.node_list['Node4'].computing_resource, self.network.node_list['Node0'].get_bandwidth('Node3'), self.network.node_list['Node3'].get_bandwidth('Node4')]
        
        if self.incremental_number == 3:
            actor_states = [actor_state_1, actor_state_2, actor_state_3]
        elif self.incremental_number == 4:
            for information in self.requests['Node1'][:5]:
                actor_state_4.append(information)
            actor_state_4 += [self.network.node_list['Node1'].computing_resource, self.network.node_list['Node2'].computing_resource, self.network.node_list['Node5'].computing_resource, self.network.node_list['Node1'].get_bandwidth('Node2'), self.network.node_list['Node2'].get_bandwidth('Node5')]
        
            actor_states = [actor_state_1, actor_state_2, actor_state_3, actor_state_4]
        elif self.incremental_number == 5:
            for information in self.requests['Node1'][:5]:
                actor_state_4.append(information)
            actor_state_4 += [self.network.node_list['Node1'].computing_resource, self.network.node_list['Node2'].computing_resource, self.network.node_list['Node5'].computing_resource, self.network.node_list['Node1'].get_bandwidth('Node2'), self.network.node_list['Node2'].get_bandwidth('Node5')]
            for information in self.requests['Node2'][:5]:
                actor_state_5.append(information)
            actor_state_5 += [self.network.node_list['Node2'].computing_resource, self.network.node_list['Node5'].computing_resource, self.network.node_list['Node4'].computing_resource, self.network.node_list['Node2'].get_bandwidth('Node5'), self.network.node_list['Node5'].get_bandwidth('Node4')]
        
            actor_states = [actor_state_1, actor_state_2, actor_state_3, actor_state_4, actor_state_5]
        elif self.incremental_number == 6:
            for information in self.requests['Node1'][:5]:
                actor_state_4.append(information)
            actor_state_4 += [self.network.node_list['Node1'].computing_resource, self.network.node_list['Node2'].computing_resource, self.network.node_list['Node5'].computing_resource, self.network.node_list['Node1'].get_bandwidth('Node2'), self.network.node_list['Node2'].get_bandwidth('Node5')]
            for information in self.requests['Node2'][:5]:
                actor_state_5.append(information)
            actor_state_5 += [self.network.node_list['Node2'].computing_resource, self.network.node_list['Node5'].computing_resource, self.network.node_list['Node4'].computing_resource, self.network.node_list['Node2'].get_bandwidth('Node5'), self.network.node_list['Node5'].get_bandwidth('Node4')]
            for information in self.requests['Node5'][:5]:
                actor_state_6.append(information)
            actor_state_6 += [self.network.node_list['Node5'].computing_resource, self.network.node_list['Node4'].computing_resource, self.network.node_list['Node1'].computing_resource, self.network.node_list['Node5'].get_bandwidth('Node4'), self.network.node_list['Node4'].get_bandwidth('Node1')]
        
            actor_states = [actor_state_1, actor_state_2, actor_state_3, actor_state_4, actor_state_5, actor_state_6]
        return actor_states # shape of actor_action: [[actor_1], [actor_2], ...]

    
    # over occupy has two drawbacks: first is left less resources for following requests, second is resuling high energy consumption
    def step(self, actions, time_record, power_record, Time_results): # action will be [[Bandwidth1_1, B1_2, ComputingLoad1_1, C1_2, C1_3], [B2_1, B2_2, C2_1, C2_2, C2_3], [B3_1, B3_2, C3_1, C3_2, C3_3]]
        state = self.generate_state()
        actor_state = self.read_actor_state()
        reward = 0
        bad_reward = None
        ##################      firstly we need to release the resources logged in self.recoder     ###################
        # where self.recoder = {time_1: {Node1: occupying_resource, Node2: 20, ... Node1Node2: 10, Node2Node3: 10...}, time_2:{}, ... }
        recorder_copy = copy.deepcopy(self.recorder)
        for slot in recorder_copy:
            if int(slot) < self.timer:
                for information in recorder_copy[slot]:
                    if len(information) == 5:  # 5 is the lens of a Nodex, not 5 means NodexNodex, which is path
                        self.network.node_list[information].change_resource(recorder_copy[slot][information])
                    else:
                        self.network.node_list[information[:5]].change_bandwidth(information[5:], recorder_copy[slot][information])
                del self.recorder[slot]

        ###################       secondely we need to check if action is legal, if not, update action      ####################
        updated_actions = copy.deepcopy(actions)
        all_values = [value for value in self.requests.values()]

        for act_idx in range(len(actions)):
            if self.network.node_list[all_values[act_idx][-3]].get_bandwidth(all_values[act_idx][-2]) - (actions[act_idx][0] * 40 + 2)  >= 0.05: 
                updated_actions[act_idx][0] = actions[act_idx][0] * 40 + 2
            else:
                if self.network.node_list[all_values[act_idx][-3]].get_bandwidth(all_values[act_idx][-2]) >= 5:
                    updated_actions[act_idx][0] = self.network.node_list[all_values[act_idx][-3]].get_bandwidth(all_values[act_idx][-2]) * 0.9
                else:
                    updated_actions[act_idx][0] = 0
            
            if self.network.node_list[all_values[act_idx][-2]].get_bandwidth(all_values[act_idx][-1]) - (actions[act_idx][1] * 40 + 2) >= 0.05:
                updated_actions[act_idx][1] = actions[act_idx][1] * 40 + 2
            else:
                if self.network.node_list[all_values[act_idx][-2]].get_bandwidth(all_values[act_idx][-1]) >= 5:
                    updated_actions[act_idx][1] = self.network.node_list[all_values[act_idx][-2]].get_bandwidth(all_values[act_idx][-1]) * 0.9
                else:
                    updated_actions[act_idx][1] = 0

            if self.network.node_list[all_values[act_idx][-3]].get_resource() - (actions[act_idx][2] * 40 + 2) >= 0.05:
                updated_actions[act_idx][2] = actions[act_idx][2]  * 40 + 2
            else:
                if self.network.node_list[all_values[act_idx][-3]].get_resource() >= 5:
                    updated_actions[act_idx][2] = self.network.node_list[all_values[act_idx][-3]].get_resource() * 0.9
                else:
                    updated_actions[act_idx][2] = 0

            if self.network.node_list[all_values[act_idx][-2]].get_resource() - (actions[act_idx][3] * 40 + 2) >= 0.05:
                updated_actions[act_idx][3] = actions[act_idx][3] * 40 + 2
            else:
                if self.network.node_list[all_values[act_idx][-2]].get_resource() >= 5:
                    updated_actions[act_idx][3] = self.network.node_list[all_values[act_idx][-2]].get_resource() * 0.9 # this 0.9 and 0.05 is to avoid division by zero
                else:
                    updated_actions[act_idx][3] = 0
            
            if self.network.node_list[all_values[act_idx][-1]].get_resource() - (actions[act_idx][4] * 40 + 2) >= 0.05:
                updated_actions[act_idx][4] = actions[act_idx][4] * 40 + 2
            else:
                if self.network.node_list[all_values[act_idx][-1]].get_resource() >= 5:
                    updated_actions[act_idx][4] = self.network.node_list[all_values[act_idx][-1]].get_resource() * 0.9 # this 0.9 and 0.05 is to avoid division by zero
                else:
                    updated_actions[act_idx][4] = 0

            # updata the network. It the request fail, this action will not be executed, as in real network, resource will be wasted.
            if updated_actions[act_idx][0] == 0 or updated_actions[act_idx][1] == 0 or updated_actions[act_idx][2] == 0 or updated_actions[act_idx][3] == 0 or updated_actions[act_idx][4] == 0:
                continue
            else:
                self.network.node_list[all_values[act_idx][-3]].change_bandwidth(all_values[act_idx][-2], -updated_actions[act_idx][0]) # change on the network
                self.network.node_list[all_values[act_idx][-2]].change_bandwidth(all_values[act_idx][-1], -updated_actions[act_idx][1]) # change on the network
                self.network.node_list[all_values[act_idx][-3]].change_resource(-updated_actions[act_idx][2])
                self.network.node_list[all_values[act_idx][-2]].change_resource(-updated_actions[act_idx][3]) # change on the network
                self.network.node_list[all_values[act_idx][-1]].change_resource(-updated_actions[act_idx][4]) # change on the network

        ############      With updated_actions, thirdly we need to get reward and next_state, and update self.recoder    ##############
        # self.recorder = {} # {time_1: {Node1: occupying_resource, Node2: 20, ... Node1Node2: 10, Node2Node3: 10...}, time_2:{}, ... }
        for act_idx in range(len(updated_actions)):
            if updated_actions[act_idx][0] == 0 or updated_actions[act_idx][1] == 0 or updated_actions[act_idx][2] == 0 or updated_actions[act_idx][3] == 0 or updated_actions[act_idx][4] == 0: # as long as one is 0, the request will be failed.
                if self.incremental_number == 4:
                    Time_results.append('bed')
                    reward -= 1/4
                elif self.incremental_number == 5:
                    Time_results.append('bed')
                    reward -= 1/5
                elif self.incremental_number == 3:
                    Time_results.append('bed')
                    reward -= 1/3
                elif self.incremental_number == 6:
                    Time_results.append('bed')
                    reward -= 1/6
                continue
            else:
                time_list = []
                total_power = 0
                # calculate the reward related parameters and log time into recoder
                data_rate_1 = updated_actions[act_idx][0] * math.log2(1 + 10) # we assume SNR = 10dB
                data_rate_2 = updated_actions[act_idx][1] * math.log2(1 + 10)
                tran_time_1 = all_values[act_idx][0] / data_rate_1 # bandwidth1 
                tran_time_2 = all_values[act_idx][1] / data_rate_2
                tran_time = tran_time_1 + tran_time_2

                if str(self.timer + math.ceil(tran_time_1)) in self.recorder:
                    if all_values[act_idx][-3] + all_values[act_idx][-2] in self.recorder[str(self.timer + math.ceil(tran_time_1))]:
                        self.recorder[str(self.timer + math.ceil(tran_time_1))][all_values[act_idx][-3] + all_values[act_idx][-2]] += updated_actions[act_idx][0]
                    else:
                        self.recorder[str(self.timer + math.ceil(tran_time_1))][all_values[act_idx][-3] + all_values[act_idx][-2]] = updated_actions[act_idx][0]
                else:
                    self.recorder[str(self.timer + math.ceil(tran_time_1))] = {}
                    self.recorder[str(self.timer + math.ceil(tran_time_1))][all_values[act_idx][-3] + all_values[act_idx][-2]] = updated_actions[act_idx][0]

                if str(self.timer + math.ceil(tran_time_2)) in self.recorder:
                    if all_values[act_idx][-2] + all_values[act_idx][-1] in self.recorder[str(self.timer + math.ceil(tran_time_2))]:
                        self.recorder[str(self.timer + math.ceil(tran_time_2))][all_values[act_idx][-2] + all_values[act_idx][-1]] += updated_actions[act_idx][1]
                    else:
                        self.recorder[str(self.timer + math.ceil(tran_time_2))][all_values[act_idx][-2] + all_values[act_idx][-1]] = updated_actions[act_idx][1]
                else:
                    self.recorder[str(self.timer + math.ceil(tran_time_2))] = {}
                    self.recorder[str(self.timer + math.ceil(tran_time_2))][all_values[act_idx][-2] + all_values[act_idx][-1]] = updated_actions[act_idx][1]

                computing_time_1 = all_values[act_idx][2] / updated_actions[act_idx][2]
                if str(self.timer + math.ceil(computing_time_1)) in self.recorder:
                    if all_values[act_idx][-3] in self.recorder[str(self.timer + math.ceil(computing_time_1))]:
                        self.recorder[str(self.timer + math.ceil(computing_time_1))][all_values[act_idx][-3]] += updated_actions[act_idx][2]
                    else:
                        self.recorder[str(self.timer + math.ceil(computing_time_1))][all_values[act_idx][-3]] = updated_actions[act_idx][2]
                else:
                    self.recorder[str(self.timer + math.ceil(computing_time_1))] = {}
                    self.recorder[str(self.timer + math.ceil(computing_time_1))][all_values[act_idx][-3]] = updated_actions[act_idx][2]

                computing_time_2 = all_values[act_idx][3] / updated_actions[act_idx][3]
                if str(self.timer + math.ceil(computing_time_2)) in self.recorder:
                    if all_values[act_idx][-2] in self.recorder[str(self.timer + math.ceil(computing_time_2))]:
                        self.recorder[str(self.timer + math.ceil(computing_time_2))][all_values[act_idx][-2]] += updated_actions[act_idx][3]
                    else:
                        self.recorder[str(self.timer + math.ceil(computing_time_2))][all_values[act_idx][-2]] = updated_actions[act_idx][3]
                else:
                    self.recorder[str(self.timer + math.ceil(computing_time_2))] = {}
                    self.recorder[str(self.timer + math.ceil(computing_time_2))][all_values[act_idx][-2]] = updated_actions[act_idx][3]

                computing_time_3 = all_values[act_idx][4] / updated_actions[act_idx][4]
                if str(self.timer + math.ceil(computing_time_3)) in self.recorder:
                    if all_values[act_idx][-1] in self.recorder[str(self.timer + math.ceil(computing_time_3))]:
                        self.recorder[str(self.timer + math.ceil(computing_time_3))][all_values[act_idx][-1]] += updated_actions[act_idx][4]
                    else:
                        self.recorder[str(self.timer + math.ceil(computing_time_3))][all_values[act_idx][-1]] = updated_actions[act_idx][4]
                else:
                    self.recorder[str(self.timer + math.ceil(computing_time_3))] = {}
                    self.recorder[str(self.timer + math.ceil(computing_time_3))][all_values[act_idx][-1]] = updated_actions[act_idx][4]

                computing_time = max(computing_time_1, computing_time_2, computing_time_3)
                time_list.append(tran_time + computing_time)

            # here as all requests have been consider in the last step, so it considers the load at the end not the individuals
            # As soon some requests might release resource and then be occuyied again, we will not consider varing power for less complexity
                load_1 = 1 - self.network.node_list[all_values[act_idx][-3]].get_resource()/100
                power_1 = (130*load_1**2 + 90) * computing_time_1

                load_2 = 1 - self.network.node_list[all_values[act_idx][-2]].get_resource()/100

                
                power_2 = (130*load_2**2 + 90) * computing_time_2
                
                load_3 = 1 - self.network.node_list[all_values[act_idx][-1]].get_resource()/100
                power_3 = (130*load_3**2 + 90) * computing_time_3
                total_power += (power_1 + power_2 + power_3)
                total_time = max(time_list)
                slicing_window = 500
                Time_results.append(total_time)
                store_data(time_record, total_time, slicing_window)
                store_data(power_record, total_power, slicing_window)
                if len(time_record) < slicing_window:
                    if self.incremental_number == 4:
                        reward += (  1200 / total_power + 10 / total_time) / 8
                    elif self.incremental_number == 5:
                        reward += (  1200 / total_power + 10 / total_time) / 10
                    elif self.incremental_number == 3:
                        reward += (  1200 / total_power + 10 / total_time) / 6
                    elif self.incremental_number == 6:
                        reward += (  1200 / total_power + 10 / total_time) / 12
                else:
                    if self.incremental_number == 4:
                        reward += (bottom_avg(power_record) / total_power + bottom_avg(time_record) / total_time) / 8
                    elif self.incremental_number == 5:
                        reward += (bottom_avg(power_record) / total_power + bottom_avg(time_record) / total_time) / 10
                    elif self.incremental_number == 3:
                        reward += (bottom_avg(power_record) / total_power + bottom_avg(time_record) / total_time) / 6
                    elif self.incremental_number == 6:
                        reward += (bottom_avg(power_record) / total_power + bottom_avg(time_record) / total_time) / 12


        next_state = self.generate_state()
        actor_next_state = self.read_actor_state()
        self.timer += 1
        return actor_state, actor_next_state, state, next_state, reward

    # DONE
    def reset(self):
        # initialize self.network
        self.build_network()
        # initialize request requirement
        self.generate_requests()
        # initialize time slot
        self.timer = 0
        # integrate obseration state
        state = self.generate_state()
        actor_state = self.read_actor_state()
        # initialize self.occupying_node_resource, self.occupying_link_resource
        self.recorder = {}

        return state, actor_state
    
