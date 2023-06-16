import os
    # 只使用第0块GPU。
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import datetime
import random
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# import gen
import csv
from sc_dataset import SC_DataGenerator
from config_for_real_QWS import get_config
import pandas as pd
import numpy as np
from tqdm import tqdm
from PriorityMemory1 import Memory1
from PriorityMemoryfd import Memoryfd
from PriorityMemory2 import Memory2
from collections import defaultdict as ddict
from functools import reduce
config, _ = get_config()
f_path = 'data/nodeSet.txt'
data_set = SC_DataGenerator()
# mase -- f_path 就是 nodeSet 文件的路径，这条 调用init函数会 使 *num_service*数组中存放每个 state节点的候选服务数量
data_set.init(f_path)
# ff = (os.path.basename(__file__)).split(".")[0]
# mase -- 下面干了三件事：1、初始化记忆区，容量为300 。 2、初始化num_service数组，每个元素为 每个state节点的候选服务数量
# mase -- 3、初始化 max_action_num 为最大候选服务数量（即action最多到哪列）
# mase -- 记忆区容量memory_capacity，默认值为300。 则 MEMORY 维度为 300 * 4 的 二维矩阵
# mase -- 每个 state节点的候选服务数量
num_service = data_set.num_service
# mase -- 最大action 的列数
max_action_num = data_set.count
act_count = data_set.act_count
# print(act_count)
current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
fd = open(f_path, 'r')
nodeSets = fd.readlines()[1]  # str
# print(nodeSets)
nodeSets = nodeSets.split(' ')
if nodeSets[-1] == '':
    nodeSets = nodeSets[:-1]


device = torch.device('cuda:0')


#  超参数
BATCH_SIZE = 32
LR = config.lr_start
GAMMA = 0.90
MEMORY_CAPACITY = config.memory_capacity
Q_NETWORK_ITERATION = config.target_replace_iter
EPSILON = config.min_epsilon
# MEMORY = np.zeros([MEMORY_CAPACITY, 4])  # [s,a,r,s+1] mase -- [state,action,reward,state_]


# mase -- 计算出三个QoS的最大和最小值，用于后面计算 reward
QoS_Data = pd.read_csv('./data/QWS_Dataset_With_Head.csv')  # 读取QoS文件\n,
maxResponseTime = QoS_Data.iloc[:, [1]].values.max()
maxAvailbility = QoS_Data.iloc[:, [2]].values.max()
maxThroughput = QoS_Data.iloc[:, [3]].values.max()
minResponseTime = QoS_Data.iloc[:, [1]].values.min()
minAvailbility = QoS_Data.iloc[:, [2]].values.min()
minThroughput = QoS_Data.iloc[:, [3]].values.min()
normResponseTime = (-QoS_Data['Response Time'] + maxResponseTime) / (maxResponseTime - minResponseTime)
normAvailbility = (QoS_Data['Availability'] - minAvailbility) / (maxAvailbility - minAvailbility)
normThroughput = (QoS_Data['Throughput'] - minThroughput) / (maxThroughput - minThroughput)

class Net(nn.Module):
    """docstring for Net"""

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 30)
        self.fc2 = nn.Linear(30, 30)
        self.out = nn.Linear(30, max_action_num)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        q_prediciton = self.out(x)
        return q_prediciton

class DuelingNet(nn.Module):
    """docstring for Net"""

    def __init__(self):
        super(DuelingNet, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        # self.fc2 = nn.Linear(64, 30)
        # self.out = nn.Linear(30, 300)

        self.value_stream = nn.Sequential(
            nn.Linear(64, 30),
            nn.ReLU(),
            nn.Linear(30, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 30),
            nn.ReLU(),
            nn.Linear(30, max_action_num)
        )
        # self.device = device
    def forward(self, x):
        # x = x.to(device)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.fc2(x)
        # x = F.relu(x)
        values = self.value_stream(x)
        advantages = self.advantage_stream(x)
        q_prediciton = values + (advantages - advantages.mean())
        # q_prediciton = self.out(x)
        return q_prediciton
class NoisyLinear1(nn.Linear):
    # 初始化函数
    def __init__(self, in_features, out_features):
        super(NoisyLinear1, self).__init__(in_features, out_features)
        self.sigma_zero=0.5
        self.sigma_init = self.sigma_zero / math.sqrt(in_features)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), -1 / math.sqrt(in_features)))
        self.sigma_bias = nn.Parameter(torch.full((out_features,), self.sigma_init / math.sqrt(in_features)))
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
        # self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        # self.register_buffer("epsilon_bias", torch.zeros(out_features))
    # 前向传播函数
    def forward(self, input):
        bias = self.bias
        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))

        with torch.no_grad():
            self.epsilon_input.normal_()
            self.epsilon_output.normal_()
            eps_in = func(self.epsilon_input)
            eps_out = func(self.epsilon_output)
            noise_v = torch.mul(eps_in, eps_out).detach()
            bias = bias + self.sigma_bias * eps_out.t()
        # self.epsilon_weight.normal_()
        # self.epsilon_bias.normal_()
        # weight = self.weight + self.sigma_weight * self.epsilon_weight.data
        # bias = self.bias + self.sigma_bias * self.epsilon_bias.data
        return F.linear(input, self.weight + self.sigma_weight * noise_v, bias)


class NoisyDuelingNet1(nn.Module):
    def __init__(self):
        super(NoisyDuelingNet1, self).__init__()
        self.fc1 = nn.Linear(1, 64)

        self.value_stream = nn.Sequential(
            nn.Linear(64, 30),
            nn.ReLU(),
            nn.Linear(30, 1)
        )

        self.advantage_stream = nn.Sequential(
            NoisyLinear1(64, 30),
            nn.ReLU(),
            NoisyLinear1(30, max_action_num)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        values = self.value_stream(x)
        advantages = self.advantage_stream(x)
        q_prediction = values + (advantages - advantages.mean())
        return q_prediction
class NoisyDuelingNet101(nn.Module):
    def __init__(self):
        super(NoisyDuelingNet101, self).__init__()
        self.fc1 = nn.Linear(1, 64)

        self.value_stream = nn.Sequential(
            nn.Linear(64, 30),
            nn.ReLU(),
            nn.Linear(30, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 30),
            nn.ReLU(),
            NoisyLinear1(30, max_action_num)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        values = self.value_stream(x)
        advantages = self.advantage_stream(x)
        q_prediction = values + (advantages - advantages.mean())
        return q_prediction
class NoisyLinear2(nn.Linear):
    # 初始化函数
    def __init__(self, in_features, out_features):
        super(NoisyLinear2, self).__init__(in_features, out_features)
        self.sigma_init=0.01
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), self.sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        self.sigma_bias = nn.Parameter(torch.full((out_features,), self.sigma_init))
        self.register_buffer("epsilon_bias", torch.zeros(out_features))
    # 前向传播函数
    def forward(self, input):
        self.epsilon_weight.normal_()
        self.epsilon_bias.normal_()
        weight = self.weight + self.sigma_weight * self.epsilon_weight.data
        bias = self.bias + self.sigma_bias * self.epsilon_bias.data
        return F.linear(input, weight, bias)


class NoisyDuelingNet2(nn.Module):
    def __init__(self):
        super(NoisyDuelingNet2, self).__init__()
        self.fc1 = nn.Linear(1, 64)

        self.value_stream = nn.Sequential(
            nn.Linear(64, 30),
            nn.ReLU(),
            nn.Linear(30, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 30),
            nn.ReLU(),
            NoisyLinear2(30, max_action_num)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        values = self.value_stream(x)
        advantages = self.advantage_stream(x)
        q_prediction = values + (advantages - advantages.mean())
        return q_prediction
class Pd3qnd(nn.Module):
    def __init__(self):
        super(Pd3qnd, self).__init__()
        self.eval_net, self.target_net = NoisyDuelingNet1(), NoisyDuelingNet1()
        self.target_net.load_state_dict(self.eval_net.state_dict())  # mase -- 同步初始权重
        self.learn_step_counter = 0  # mase -- 学习步数，用于更新权重
        self.memory_counter = 0  # mase -- 记忆区目前装了多少条了
        self.memory = np.zeros((MEMORY_CAPACITY, 4))  # 0
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.demo_optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=2*LR, weight_decay=1e-5)
        self.loss_func = nn.MSELoss()
        self.ReplayMemory = Memory1(MEMORY_CAPACITY)
        self.demoReplayMemory = Memoryfd(500)
        self.n_step = 3
        self.lamda1 = 1.0
        self.lamda2 = 1.0
        self.demoReplay = ddict(list)
        self.gamma = 0.9

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor([state]), 0)
        # state = state.to(device)
        # if np.random.uniform() < EPSILON:
            # mase -- 先计算出当前state的所有action的Q值，再让 action 为 最大的Q值 对应action的 index
        all_action_q_value = self.eval_net(state)#bug
        a_choosen = int(act_max(all_action_q_value, state).item())
        # else:
        #     a_choosen = np.random.randint(act_count[int(state.item())],act_count[int(state.item())+1])
        #     # y -- 从当前节点的服务列表中随机选一个
        return a_choosen

    def store_transition(self, state, action, reward, next_state):
        transition = [state, action, reward, next_state]
        self.ReplayMemory.store(transition)
        self.memory_counter += 1

    def learn(self):
        global EPSILON
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            EPSILON = EPSILON + config.epsilon_increment if EPSILON + config.epsilon_increment < config.max_epsilon else config.max_epsilon

        self.learn_step_counter += 1
        # sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        # batch_memory = self.memory[sample_index, :]
        tree_idx, batch_memory, weights = self.ReplayMemory.sample(BATCH_SIZE)
        weights = torch.sqrt(torch.FloatTensor(weights))
        batch_state = torch.FloatTensor(batch_memory[:, 0:1])
        # batch_state = batch_state.to(device)
        batch_action = torch.LongTensor(batch_memory[:, 1:2].astype(int))
        # batch_action = batch_action.to(device)
        batch_reward = torch.FloatTensor(batch_memory[:, 2:3])
        # batch_reward = batch_reward.to(device)
        batch_next_state = torch.FloatTensor(batch_memory[:, 3:4])
        # batch_next_state = batch_next_state.to(device)

        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.get_q_next_values(batch_next_state, self.eval_net, self.target_net)
        q_target = batch_reward + GAMMA * q_next
        errors = torch.abs(q_eval - q_target)
        errors = errors.detach().numpy().reshape(-1)
        self.ReplayMemory.batch_update(tree_idx, errors)
        loss = self.loss_func(weights * q_eval, weights * q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def store_demotransition(self,result,node_data):
        count = 0
        for combination, _ in result:
            episodeReplay = self.demoReplay[count]
            count += 1
            index = len(episodeReplay)
            for state in range(len(combination)):
                action = combination[state] + act_count[state]
                atom_node_id = node_data[nodeSets[state]][combination[state]]
                reward = normResponseTime[atom_node_id - 1] + normAvailbility[atom_node_id - 1] + normThroughput[
                    atom_node_id - 1]
                next_state = state + 1 if state < 89 else -1
                transition = [state, action, reward, next_state,count, index]
                episodeReplay.append(transition)
                self.demoReplayMemory.store(transition)
                # self.ReplayMemory.store(transition[0:4])
    def get_q_eval(self, batch_state, batch_action):
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        return q_eval
    def get_q_next_values(self,batch_next_state, eval_net, target_net):
        mask = (batch_next_state != -1).any(dim=1)
        non_terminal_next_states = batch_next_state[mask]
        if non_terminal_next_states.size(0) > 0:
            q_evals = eval_net(non_terminal_next_states)
            max_actions_from_q_evals = act_max(q_evals, non_terminal_next_states)
            q_next = target_net(non_terminal_next_states).detach().gather(1, max_actions_from_q_evals)

        q_next_all = torch.zeros(batch_next_state.size(0))
        q_next_all[mask] = q_next.squeeze()
        q_next_all = q_next_all.unsqueeze(dim=1)
        return q_next_all
    def l_ns(self,q_eval, samples, n_step):
        count = 0
        loss = torch.tensor(0.0)
        for i, (s, a, r, s_, episode, idx) in enumerate(samples):
            nidx = idx + self.n_step
            lepoch = len(self.demoReplay[episode])
            if nidx > lepoch:
                continue
            count += 1
            ns, na, nr, ns_,ua,ub = zip(*self.demoReplay[int(episode)][int(idx):int(nidx)])
            ns, na, ns_,  = ns[-1], na[-1], ns_[-1]
            discountedR = reduce(lambda x, y: (x[0] + self.gamma ** x[1] * y, x[1] + 1), nr, (0, 0))[0]
            q_next = self.get_q_next_values(torch.tensor([ns_], dtype=torch.float32).unsqueeze(dim = 1), self.eval_net, self.target_net)
            q_target = discountedR + GAMMA ** n_step * q_next.squeeze()
            q_predict = q_eval[i].squeeze()
            loss += (q_target - q_predict) ** 2
            # print(loss)
        return loss / count
    def l_1s(self,q_eval,q_target,weights):
        loss = self.loss_func(weights * q_eval, weights * q_target)
        return loss
    def l_jeq(self, batch_state, batch_action, q_eval, act_count):
        jeq = torch.tensor(0.0)
        count = 0
        for idx, state in enumerate(batch_state):
            ae = batch_action[idx] - act_count[int(state.item())]
            max_value = torch.tensor(float('-inf'), dtype=torch.float32)
            tq_values = q_eval[idx][act_count[int(state.item())]:act_count[int(state.item() + 1)]]
            for a in range(act_count[int(state.item() + 1)] - act_count[int(state.item())]):
                loss_l_value = 0.0 if ae == a else 0.8
                max_value = torch.max(tq_values[a] + loss_l_value, max_value)
            jeq += max_value - tq_values[ae].squeeze()
            if max_value > tq_values[ae].squeeze():
                count += 1
        return jeq / count if count != 0 else jeq
    def demo_learn(self):
        global EPSILON
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            EPSILON = EPSILON + config.epsilon_increment if EPSILON + config.epsilon_increment < config.max_epsilon else config.max_epsilon
        self.learn_step_counter += 1

        tree_idx, batch_memory, weights = self.demoReplayMemory.sample(BATCH_SIZE)
        weights = torch.sqrt(torch.FloatTensor(weights))
        batch_state = torch.FloatTensor(batch_memory[:, 0:1])
        batch_action = torch.LongTensor(batch_memory[:, 1:2].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, 2:3])
        batch_next_state = torch.FloatTensor(batch_memory[:, 3:4])
        q_eval_a = self.eval_net(batch_state)
        q_eval = self.get_q_eval(batch_state, batch_action)

        # Calculate q_target for 1-step return
        q_next = self.get_q_next_values(batch_next_state, self.eval_net, self.target_net)
        q_target_1s = batch_reward + GAMMA * q_next
        errors = torch.abs(q_eval - q_target_1s)
        errors = errors.detach().numpy().reshape(-1)
        self.demoReplayMemory.batch_update(tree_idx, errors)

        # Calculate q_target for n-step return
        l_1s = self.l_1s(q_eval, q_target_1s, weights)
        # l_ns = self.l_ns(q_eval, batch_memory, self.n_step)
        l_jeq = self.l_jeq(batch_state, batch_action, q_eval_a, act_count)
        # loss = l_1s + self.lamda1 * l_ns + self.lamda2 * l_jeq
        loss = l_1s + self.lamda2 * l_jeq
        # loss = self.l_1s(q_eval, q_target_1s, weights) + self.lamda1 * self.l_ns(q_eval, batch_memory, self.n_step) + \
        #        self.lamda2 * self.l_jeq(batch_state, batch_action, q_eval_a, act_count)

        self.demo_optimizer.zero_grad()
        loss.backward()
        self.demo_optimizer.step()

class Pd3qn11(nn.Module):
    def __init__(self):
        super(Pd3qn11, self).__init__()
        self.eval_net, self.target_net = NoisyDuelingNet1(), NoisyDuelingNet1()
        self.target_net.load_state_dict(self.eval_net.state_dict())  # mase -- 同步初始权重
        self.learn_step_counter = 0  # mase -- 学习步数，用于更新权重
        self.memory_counter = 0  # mase -- 记忆区目前装了多少条了
        self.memory = np.zeros((MEMORY_CAPACITY, 4))  # 0
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.ReplayMemory = Memory1(MEMORY_CAPACITY)
        # self.device = device  # mase -- GPU

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor([state]), 0)
        # state = state.to(device)
        # if np.random.uniform() < EPSILON:
            # mase -- 先计算出当前state的所有action的Q值，再让 action 为 最大的Q值 对应action的 index
        all_action_q_value = self.eval_net(state)#bug
        a_choosen = int(act_max(all_action_q_value, state).item())
        # else:
        #     a_choosen = np.random.randint(act_count[int(state.item())],act_count[int(state.item())+1])
        #     # y -- 从当前节点的服务列表中随机选一个
        return a_choosen

    def store_transition(self, state, action, reward, next_state):
        transition = [state, action, reward, next_state]
        self.ReplayMemory.store(transition)
        self.memory_counter += 1

    def get_q_next_values(self,batch_next_state, eval_net, target_net):
        mask = (batch_next_state != -1).any(dim=1)
        non_terminal_next_states = batch_next_state[mask]
        if non_terminal_next_states.size(0) > 0:
            q_evals = eval_net(non_terminal_next_states)
            max_actions_from_q_evals = act_max(q_evals, non_terminal_next_states)
            q_next = target_net(non_terminal_next_states).detach().gather(1, max_actions_from_q_evals)

        q_next_all = torch.zeros(batch_next_state.size(0))
        q_next_all[mask] = q_next.squeeze()
        q_next_all = q_next_all.unsqueeze(dim=1)
        return q_next_all
    def learn(self):
        global EPSILON
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            EPSILON = EPSILON + config.epsilon_increment if EPSILON + config.epsilon_increment < config.max_epsilon else config.max_epsilon

        self.learn_step_counter += 1
        # sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        # batch_memory = self.memory[sample_index, :]
        tree_idx, batch_memory, weights = self.ReplayMemory.sample(BATCH_SIZE)
        weights = torch.sqrt(torch.FloatTensor(weights))
        batch_state = torch.FloatTensor(batch_memory[:, 0:1])
        # batch_state = batch_state.to(device)
        batch_action = torch.LongTensor(batch_memory[:, 1:2].astype(int))
        # batch_action = batch_action.to(device)
        batch_reward = torch.FloatTensor(batch_memory[:, 2:3])
        # batch_reward = batch_reward.to(device)
        batch_next_state = torch.FloatTensor(batch_memory[:, 3:4])

        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.get_q_next_values(batch_next_state, self.eval_net, self.target_net)
        q_target = batch_reward + GAMMA * q_next
        errors = torch.abs(q_eval - q_target)
        errors = errors.detach().numpy().reshape(-1)
        self.ReplayMemory.batch_update(tree_idx, errors)
        loss = self.loss_func(weights * q_eval, weights * q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
# class Pd3qn101(nn.Module):
#     def __init__(self):
#         super(Pd3qn101, self).__init__()
#         self.eval_net, self.target_net = NoisyDuelingNet101(), NoisyDuelingNet101()
#         self.target_net.load_state_dict(self.eval_net.state_dict())  # mase -- 同步初始权重
#         self.learn_step_counter = 0  # mase -- 学习步数，用于更新权重
#         self.memory_counter = 0  # mase -- 记忆区目前装了多少条了
#         self.memory = np.zeros((MEMORY_CAPACITY, 4))  # 0
#         self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
#         self.loss_func = nn.MSELoss()
#         self.ReplayMemory = Memory1(MEMORY_CAPACITY)
#         # self.device = device  # mase -- GPU
#
#     def choose_action(self, state):
#         state = torch.unsqueeze(torch.FloatTensor([state]), 0)
#         # state = state.to(device)
#         # if np.random.uniform() < EPSILON:
#             # mase -- 先计算出当前state的所有action的Q值，再让 action 为 最大的Q值 对应action的 index
#         all_action_q_value = self.eval_net(state)#bug
#         a_choosen = int(act_max(all_action_q_value, state).item())
#         # else:
#         #     a_choosen = np.random.randint(act_count[int(state.item())],act_count[int(state.item())+1])
#         #     # y -- 从当前节点的服务列表中随机选一个
#         return a_choosen
#
#     def store_transition(self, state, action, reward, next_state):
#         transition = [state, action, reward, next_state]
#         self.ReplayMemory.store(transition)
#         self.memory_counter += 1
#
#     # def store_transition(self, state, action, reward, next_state):
#     #     transition = [state, action, reward, next_state]
#     #     index = self.memory_counter % MEMORY_CAPACITY
#     #     self.memory[index, :] = transition
#     #     self.memory_counter += 1
#
#     def learn(self):
#         global EPSILON
#         if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
#             self.target_net.load_state_dict(self.eval_net.state_dict())
#             EPSILON = EPSILON + config.epsilon_increment if EPSILON + config.epsilon_increment < config.max_epsilon else config.max_epsilon
#
#         self.learn_step_counter += 1
#         # sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
#         # batch_memory = self.memory[sample_index, :]
#         tree_idx, batch_memory, weights = self.ReplayMemory.sample(BATCH_SIZE)
#         weights = torch.sqrt(torch.FloatTensor(weights))
#         batch_state = torch.FloatTensor(batch_memory[:, 0:1])
#         # batch_state = batch_state.to(device)
#         batch_action = torch.LongTensor(batch_memory[:, 1:2].astype(int))
#         # batch_action = batch_action.to(device)
#         batch_reward = torch.FloatTensor(batch_memory[:, 2:3])
#         # batch_reward = batch_reward.to(device)
#         batch_next_state = torch.FloatTensor(batch_memory[:, 3:4])
#         # batch_next_state = batch_next_state.to(device)
#
#         q_eval = self.eval_net(batch_state).gather(1, batch_action)
#         q_evals = self.eval_net(batch_next_state)
#         # q_evals = Masking(q_evals, batch_next_state)#001
#         # max_actions_from_q_evals = torch.argmin(q_evals, dim=1)#001
#         # max_actions_from_q_evals = max_actions_from_q_evals.to(device)
#         max_actions_from_q_evals = act_max(q_evals, batch_next_state)
#         # max_actions_from_q_evals = torch.unsqueeze(max_actions_from_q_evals, 1)
#         # max_actions_from_q_evals = max_actions_from_q_evals.to(device)
#
#         q_next = self.target_net(batch_next_state).detach().gather(1, max_actions_from_q_evals)
#         q_target = batch_reward + GAMMA * q_next
#         errors = torch.abs(q_eval - q_target)
#         errors = errors.detach().numpy().reshape(-1)
#         self.ReplayMemory.batch_update(tree_idx, errors)
#         loss = self.loss_func(weights * q_eval, weights * q_target)
#
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()


def act_max(q_values,states):
    maxact = torch.zeros_like(states,dtype=torch.int64)
    for idx, state in enumerate(states):
        tq_values = q_values[idx][act_count[int(state.item())]:act_count[int(state.item() + 1)]]
        maxact[idx] = torch.argmax(tq_values)+act_count[int(state.item())]
    return maxact
class Pd3qn21(nn.Module):
    """docstring for Pd3qn"""
    def __init__(self):
        super(Pd3qn21, self).__init__()
        self.eval_net, self.target_net = NoisyDuelingNet1(), NoisyDuelingNet1()
        self.target_net.load_state_dict(self.eval_net.state_dict())  # mase -- 同步初始权重
        self.learn_step_counter = 0  # mase -- 学习步数，用于更新权重
        self.memory_counter = 0  # mase -- 记忆区目前装了多少条了
        self.memory = Memory2(MEMORY_CAPACITY)  # 0
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        # self.ReplayMemory = ReplayMemory_Per(MEMORY_CAPACITY)
        # self.device = device  # mase -- GPU

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor([state]), 0)
        # state = state.to(device)
        if np.random.uniform() < EPSILON:
            # mase -- 先计算出当前state的所有action的Q值，再让 action 为 最大的Q值 对应action的 index
            all_action_q_value = self.eval_net(state)  # bug
            # all_action_q_value = all_action_q_value.cuda()
            # all_action_q_value = all_action_q_value[0][0:num_service[int(state.item())]]
            # all_action_q_value = Masking(all_action_q_value, state)#001
            # mase -- 获取Q值最大的action 的 index
            a_choosen = int(act_max(all_action_q_value, state).item())
        else:
            # mase -- 随机选一个action就可以
            # a_choosen = np.random.randint(num_service[int(state.item()-1)], num_service[int(state.item())])
            a_choosen = np.random.randint(act_count[int(state.item())], act_count[int(state.item()) + 1])
            # y -- 从当前节点的服务列表中随机选一个
        return a_choosen

    def store_transition(self, state, action, reward, next_state):
        transition = [state, action, reward, next_state]
        self.memory.store(transition)
        self.memory_counter += 1

    # def store_transiti
#     ^_^ition = [state, action, reward, next_state]
    #     index = self.memory_counter % MEMORY_CAPACITY
    #     self.memory[index, :] = transition
    #     self.memory_counter += 1

    def learn(self):
        global EPSILON
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            EPSILON = EPSILON + config.epsilon_increment if EPSILON + config.epsilon_increment < config.max_epsilon else config.max_epsilon

        self.learn_step_counter += 1
        # sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        # batch_memory = self.memory[sample_index, :]
        idxs, batch_memory = self.memory.sample(BATCH_SIZE)
        # batch = Transition(*zip(*transitions))

        batch_state = torch.FloatTensor(batch_memory[:, 0:1])
        # batch_state = batch_state.to(device)
        batch_action = torch.LongTensor(batch_memory[:, 1:2].astype(int))
        # batch_action = batch_action.to(device)
        batch_reward = torch.FloatTensor(batch_memory[:, 2:3])
        # batch_reward = batch_reward.to(device)
        batch_next_state = torch.FloatTensor(batch_memory[:, 3:4])
        # batch_next_state = batch_next_state.to(device)

        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_evals = self.eval_net(batch_next_state)
        # q_evals = Masking(q_evals, batch_next_state)#001
        # max_actions_from_q_evals = torch.argmin(q_evals, dim=1)#001
        # max_actions_from_q_evals = max_actions_from_q_evals.to(device)
        max_actions_from_q_evals = act_max(q_evals, batch_next_state)
        # max_actions_from_q_evals = torch.unsqueeze(max_actions_from_q_evals, 1)
        # max_actions_from_q_evals = max_actions_from_q_evals.to(device)

        q_next = self.target_net(batch_next_state).detach().gather(1, max_actions_from_q_evals)
        q_target = batch_reward + GAMMA * q_next
        errors = torch.abs(q_eval - q_target)
        errors = errors.detach().numpy().reshape(-1)
        self.memory.batch_update(idxs, errors)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
class Pd3qn2(nn.Module):
    """docstring for Pd3qn"""
    def __init__(self):
        super(Pd3qn2, self).__init__()
        self.eval_net, self.target_net = DuelingNet(), DuelingNet()
        self.target_net.load_state_dict(self.eval_net.state_dict())  # mase -- 同步初始权重
        self.learn_step_counter = 0  # mase -- 学习步数，用于更新权重
        self.memory_counter = 0  # mase -- 记忆区目前装了多少条了
        self.memory = Memory2(MEMORY_CAPACITY)  # 0
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        # self.ReplayMemory = ReplayMemory_Per(MEMORY_CAPACITY)
        # self.device = device  # mase -- GPU

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor([state]), 0)
        # state = state.to(device)
        if np.random.uniform() < EPSILON:
            # mase -- 先计算出当前state的所有action的Q值，再让 action 为 最大的Q值 对应action的 index
            all_action_q_value = self.eval_net(state)  # bug
            # all_action_q_value = all_action_q_value.cuda()
            # all_action_q_value = all_action_q_value[0][0:num_service[int(state.item())]]
            # all_action_q_value = Masking(all_action_q_value, state)#001
            # mase -- 获取Q值最大的action 的 index
            a_choosen = int(act_max(all_action_q_value, state).item())
        else:
            # mase -- 随机选一个action就可以
            # a_choosen = np.random.randint(num_service[int(state.item()-1)], num_service[int(state.item())])
            a_choosen = np.random.randint(act_count[int(state.item())], act_count[int(state.item()) + 1])
            # y -- 从当前节点的服务列表中随机选一个
        return a_choosen

    def store_transition(self, state, action, reward, next_state):
        transition = [state, action, reward, next_state]
        self.memory.store(transition)
        self.memory_counter += 1


    def learn(self):
        global EPSILON
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            EPSILON = EPSILON + config.epsilon_increment if EPSILON + config.epsilon_increment < config.max_epsilon else config.max_epsilon

        self.learn_step_counter += 1
        idxs, batch_memory = self.memory.sample(BATCH_SIZE)

        batch_state = torch.FloatTensor(batch_memory[:, 0:1])        # batch_state = batch_state.to(device)
        batch_action = torch.LongTensor(batch_memory[:, 1:2].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, 2:3])
        batch_next_state = torch.FloatTensor(batch_memory[:, 3:4])

        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_evals = self.eval_net(batch_next_state)
        max_actions_from_q_evals = act_max(q_evals, batch_next_state)

        q_next = self.target_net(batch_next_state).detach().gather(1, max_actions_from_q_evals)
        q_target = batch_reward + GAMMA * q_next
        errors = torch.abs(q_eval - q_target)
        errors = errors.detach().numpy().reshape(-1)
        self.memory.batch_update(idxs, errors)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
class DuelingDDQN(nn.Module):
    def __init__(self):
        super(DuelingDDQN, self).__init__()
        self.eval_net, self.target_net = DuelingNet(), DuelingNet()
        self.target_net.load_state_dict(self.eval_net.state_dict())  # mase -- 同步初始权重
        self.learn_step_counter = 0  # mase -- 学习步数，用于更新权重
        self.memory_counter = 0  # mase -- 记忆区目前装了多少条了
        self.memory = np.zeros((MEMORY_CAPACITY, 4))#0
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        # self.device = device  # mase -- GPU

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor([state]), 0)
        # state = state.to(device)
        if np.random.uniform() < EPSILON:
            # mase -- 先计算出当前state的所有action的Q值，再让 action 为 最大的Q值 对应action的 index
            all_action_q_value = self.eval_net(state)#bug
            # mase -- 获取Q值最大的action 的 index
            a_choosen = int(act_max(all_action_q_value, state).item())
        else:
            # mase -- 随机选一个action就可以
            a_choosen = np.random.randint(act_count[int(state.item())],act_count[int(state.item())+1])
            # y -- 从当前节点的服务列表中随机选一个
        return a_choosen

    def store_transition(self, state, action, reward, next_state):
        transition = [state, action, reward, next_state]
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        global EPSILON
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            EPSILON = EPSILON + config.epsilon_increment if EPSILON + config.epsilon_increment < config.max_epsilon else config.max_epsilon

        self.learn_step_counter += 1
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, 0:1])
        batch_action = torch.LongTensor(batch_memory[:, 1:2].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, 2:3])
        batch_next_state = torch.FloatTensor(batch_memory[:, 3:4])

        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_evals = self.eval_net(batch_next_state)
        max_actions_from_q_evals = act_max(q_evals, batch_next_state)

        q_next = self.target_net(batch_next_state).detach().gather(1, max_actions_from_q_evals)
        q_target = batch_reward + GAMMA * q_next
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
def q_max(q_values, states):
    max_q = torch.zeros_like(states, dtype=torch.float32)
    for idx, state in enumerate(states):
        if state.item() == -1:
            state = torch.tensor(config.node_num - 1, dtype=torch.float32)
        tq_values = q_values[idx][act_count[int(state.item())]:act_count[int(state.item() + 1)]]
        max_q[idx] = torch.max(tq_values)
    return max_q
class DDQN():
    """docstring for DQN"""
    def __init__(self):
        super(DDQN, self).__init__()
        self.eval_net, self.target_net = Net(), Net()
        self.target_net.load_state_dict(self.eval_net.state_dict())  # mase -- 同步初始权重
        self.learn_step_counter = 0   # mase -- 学习步数，用于更新权重
        self.memory_counter = 0       # mase -- 记忆区目前装了多少条了
        self.memory = np.zeros((MEMORY_CAPACITY, 4))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        # self.device = device  # mase -- GPU

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor([state]), 0)
        # state = state.to(device)
        if np.random.uniform() < EPSILON:
            # mase -- 先计算出当前state的所有action的Q值，再让 action 为 最大的Q值 对应action的 index
            all_action_q_value = self.eval_net(state)
            a_choosen = int(act_max(all_action_q_value, state).item())

        else:
            # mase -- 随机选一个action就可以
            a_choosen = np.random.randint(act_count[int(state.item())],act_count[int(state.item())+1])
        return a_choosen

    def store_transition(self, state, action, reward, next_state):
        transition = [state, action, reward, next_state]
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        global EPSILON
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            EPSILON = EPSILON + config.epsilon_increment if EPSILON + config.epsilon_increment < config.max_epsilon else config.max_epsilon

        self.learn_step_counter += 1
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, 0:1])
        batch_action = torch.LongTensor(batch_memory[:, 1:2].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, 2:3])
        batch_next_state = torch.FloatTensor(batch_memory[:, 3:4])

        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_evals = self.eval_net(batch_next_state)
        max_actions_from_q_evals = act_max(q_evals, batch_next_state)

        q_next = self.target_net(batch_next_state).detach().gather(1, max_actions_from_q_evals)
        q_target = batch_reward + GAMMA * q_next

        loss = self.loss_func(q_eval, q_target)
        self.demo.zero_grad()
        loss.backward()
        self.optimizer.step()
class DQN():
    """docstring for DQN"""
    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Net(), Net()
        self.target_net.load_state_dict(self.eval_net.state_dict())  # mase -- 同步初始权重
        self.learn_step_counter = 0   # mase -- 学习步数，用于更新权重
        self.memory_counter = 0       # mase -- 记忆区目前装了多少条了
        self.memory = np.zeros((MEMORY_CAPACITY, 4))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        # self.device = device  # mase -- GPU

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor([state]), 0)
        # state = state.to(device)
        if np.random.uniform() < EPSILON:
            # mase -- 先计算出当前state的所有action的Q值，再让 action 为 最大的Q值 对应action的 index
            all_action_q_value = self.eval_net(state)
            a_choosen = int(act_max(all_action_q_value, state).item())

        else:
            a_choosen = np.random.randint(act_count[int(state.item())],act_count[int(state.item())+1])
        return a_choosen

    def store_transition(self, state, action, reward, next_state):
        transition = [state, action, reward, next_state]
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        global EPSILON
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            EPSILON = EPSILON + config.epsilon_increment if EPSILON + config.epsilon_increment < config.max_epsilon else config.max_epsilon

        self.learn_step_counter += 1
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, 0:1])
        batch_action = torch.LongTensor(batch_memory[:, 1:2].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, 2:3])
        batch_next_state = torch.FloatTensor(batch_memory[:, 3:4])

        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_next = q_max(q_next, batch_next_state)
        q_target = batch_reward + GAMMA * q_next.view(BATCH_SIZE, 1)

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
def load_node_data(node_sets):
    node_data = {}
    for node in node_sets:
        with open("服务名聚类最终结果/" + str(node) + '.txt', 'r') as f:
            lines = f.readlines()
            atom_nodes = [int(line.strip('\r\n').split(':')[0]) for line in lines]
            node_data[node] = atom_nodes
    return node_data

def reward_func(state, position, node_data):
    atom_nodes = node_data[nodeSets[state]]
    choose_atom_node_id = atom_nodes[position[state]]
    qos_row = QoS_Data.iloc[choose_atom_node_id - 1]
    ResponseTime = qos_row[1]
    Availbility = qos_row[2]
    Throughput = qos_row[3]

    normResponseTime = (-ResponseTime + maxResponseTime) / (maxResponseTime - minResponseTime)
    normAvailbility = (Availbility - minAvailbility) / (maxAvailbility - minAvailbility)
    normThroughput = (Throughput - minThroughput) / (maxThroughput - minThroughput)

    immediate_reward = normThroughput + normAvailbility + normResponseTime
    return immediate_reward
def reward_func_pre(position, node_data, normResponseTime, normAvailbility, normThroughput):
    sum_reward = 0
    for state in range(len(position)):
        atom_nodes = node_data[nodeSets[state]]
        choose_atom_node_id = atom_nodes[position[state]]

        # 使用归一化值计算奖励
        immediate_reward = normThroughput[choose_atom_node_id - 1] + normAvailbility[choose_atom_node_id - 1] + normResponseTime[choose_atom_node_id - 1]
        sum_reward += immediate_reward
    return sum_reward
def Get_numservice(f):
    num_service = []
    f.readline()
    line = f.readline()
    candidates_c = line.split(' ')
    candidates = []
    candidates = [candidates_c[index] for index in range(len(candidates_c))]
    # print('Candidates: ',candidates)
    for candidate in candidates:
        num = 0
        # rows = 0  # 使得服务限制在2个
        f1 = open('服务名聚类最终结果/' + candidate + '.txt')
        line1 = f1.readline()
        while line1:
            num = num + 1
            line1 = f1.readline()
        num_service.append(num)
    return num_service

#if __name__ =='__main__':
def generate_data(node_data):
    path = "data/nodeSet.txt"
    f = open(path)
    num_service = Get_numservice(f)


    def calculate_reward(position):
        return reward_func_pre(position, node_data, normResponseTime, normAvailbility, normThroughput)
    random_combinations = [
        [random.randint(0, num_service[j] - 1) for j in range(90)] for _ in range(10000)
    ]

    # 计算每个组合的奖励值
    rewards = list(map(calculate_reward, random_combinations))

    # 将组合及其奖励值合并为一个列表，并根据奖励值排序
    result = sorted(zip(random_combinations, rewards), key=lambda x: x[1], reverse=True)[:5]


    csvfile = open('pretrain/result.csv', "w", newline="")
    writer = csv.writer(csvfile)
    writer.writerows(result)
    csvfile.close()
    return result

def run_experiment(algorithm, algorithm_name, cut_point_add_to_plot=99):
    global EPSILON
    EPSILON = config.min_epsilon
    episodes = config.nb_epoch
    max_qos = -10000
    plot_array = []
    y_matplot = []
    y_matplot_smooth = []
    y_matplot_std = []
    total_cumulate = 0
    print(f'{algorithm_name}-----------实验开始-----------')
    # sum_rewards = 0
    # sampling_interval = 20
    window_size = 5  # 用于计算滑动平均的窗口大小
    epi_n = episodes / 10  # 每n个episode输出一次平均累积回报
    cumulative_reward_of_n_episodes = 0
    for i in tqdm(range(episodes)):
        state = 0
        position = []
        cumulative_reward_of_a_composition = 0

        for t in range(config.node_num):
            action = algorithm.choose_action(state)
            position.append(action - act_count[state])

            immediate_reward = reward_func(state, position, node_data)
            reward = immediate_reward
            cumulative_reward_of_a_composition += reward

            if state == (config.node_num - 1):
                state_next = -1
            else:
                state_next = state + 1

            algorithm.store_transition(state, action, reward, state_next)
            state += 1

            if algorithm.memory_counter >= BATCH_SIZE:
                algorithm.learn()
        if state_next == -1:
            print(position)
            cumulative_reward_of_n_episodes += cumulative_reward_of_a_composition
            if (i + 1) % epi_n == 0:
                mean = cumulative_reward_of_n_episodes / epi_n
                print(f'Mean cumulative reward of last {epi_n} episodes: {mean}')
                # std = np.std(y_matplot[int(-epi_n):])  # 计算最近epi_n个episodes的标准差
                # y_matplot_std.append(std)
                # y_matplot.append(mean)
                cumulative_reward_of_n_episodes = 0  # 重置
            print(
                f'{algorithm_name}_cumulative_reward = {cumulative_reward_of_a_composition}, max = {max_qos},episodes={i}')
            y_matplot.append(cumulative_reward_of_a_composition)
            if cumulative_reward_of_a_composition > max_qos:
                max_qos = cumulative_reward_of_a_composition
        if i != 0 and i % cut_point_add_to_plot == 0:
            y_matplot_smooth = moving_average(y_matplot, window_size)
            plt.plot(np.arange(len(y_matplot_smooth)), y_matplot_smooth)
            plt.xlabel('Episode')
            plt.ylabel('Cumulative Reward')
            plt.title(f'{algorithm_name} Training')
            if not os.path.exists(f'result/{current_time}/{algorithm_name}'):
                os.makedirs(f'result/{current_time}/{algorithm_name}')
            plt.savefig(f'result/{current_time}/{algorithm_name}/{algorithm_name}_training_episode_{i}.png')
    plot_array.append(
        (y_matplot_smooth[:], np.arange(0, len(y_matplot_smooth[:])), np.mean(y_matplot), np.std(y_matplot)))
    print(f'{algorithm_name}-----------实验结束-----------, max_qos = {max_qos}')
    return plot_array
def run_demo_experiment(algorithm, algorithm_name, cut_point_add_to_plot=99):
    global EPSILON
    EPSILON = config.min_epsilon
    episodes = config.nb_epoch
    max_qos = -10000
    plot_array = []
    y_matplot = []
    y_matplot_std = []
    y_matplot_smooth = []
    total_cumulate = 0
    print(f'{algorithm_name}-----------预训练-----------')

    # 加载演示数据
    demo_data = generate_data(node_data)

    # 将演示数据存储到 demoReplayMemory
    algorithm.store_demotransition(demo_data,node_data)

    # 预训练阶段
    for _ in tqdm(range(config.pre_epoch)):
        algorithm.demo_learn()
    print(f'{algorithm_name}-----------实验开始-----------')
    # sum_rewards = 0
    # sampling_interval = 20
    window_size = 5  # 用于计算滑动平均的窗口大小
    epi_n = episodes / 10  # 每n个episode输出一次平均累积回报
    cumulative_reward_of_n_episodes = 0
    for i in tqdm(range(episodes)):
        state = 0
        position = []
        cumulative_reward_of_a_composition = 0

        for t in range(config.node_num):
            action = algorithm.choose_action(state)
            position.append(action - act_count[state])

            immediate_reward = reward_func(state, position, node_data)
            reward = immediate_reward
            cumulative_reward_of_a_composition += reward

            if state == (config.node_num - 1):
                state_next = -1
            else:
                state_next = state + 1

            algorithm.store_transition(state, action, reward, state_next)
            state += 1

            if algorithm.memory_counter >= BATCH_SIZE:
                algorithm.learn()
        if state_next == -1:
            print(position)
            cumulative_reward_of_n_episodes += cumulative_reward_of_a_composition
            if (i + 1) % epi_n == 0:
                mean = cumulative_reward_of_n_episodes / epi_n
                print(f'Mean cumulative reward of last {epi_n} episodes: {mean}')
                # std = np.std(y_matplot[int(-epi_n):])  # 计算最近epi_n个episodes的标准差
                # y_matplot_std.append(std)
                # y_matplot.append(mean)
                cumulative_reward_of_n_episodes = 0  # 重置
            print(
                f'{algorithm_name}_cumulative_reward = {cumulative_reward_of_a_composition}, max = {max_qos},episodes={i}')
            y_matplot.append(cumulative_reward_of_a_composition)
            if cumulative_reward_of_a_composition > max_qos:
                max_qos = cumulative_reward_of_a_composition
        if i != 0 and i % cut_point_add_to_plot == 0:
            y_matplot_smooth = moving_average(y_matplot, window_size)
            plt.plot(np.arange(len(y_matplot_smooth)), y_matplot_smooth)
            plt.xlabel('Episode')
            plt.ylabel('Cumulative Reward')
            plt.title(f'{algorithm_name} Training')
            if not os.path.exists(f'result/{current_time}/{algorithm_name}'):
                os.makedirs(f'result/{current_time}/{algorithm_name}')
            plt.savefig(f'result/{current_time}/{algorithm_name}/{algorithm_name}_training_episode_{i}.png')
    plot_array.append((y_matplot_smooth[:], np.arange(0, len(y_matplot_smooth[:])), np.mean(y_matplot), np.std(y_matplot)))
    print(f'{algorithm_name}-----------实验结束-----------, max_qos = {max_qos}')
    return plot_array
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
def plot_comparison(algorithms_data,seed):
    for plottimes in range(len(next(iter(algorithms_data.values())))):
        plt.figure(dpi=200)
        plt.title("figure")
        plt.xlabel("Episodes")
        plt.ylabel("Mean Cumulative reward")

        for name, plot_array in algorithms_data.items():
            x_episode = plot_array[plottimes][1]
            y_matplot = plot_array[plottimes][0]
            std = plot_array[plottimes][3]
            # data = pd.DataFrame({'Episode': x_episode, 'Mean Cumulative Reward': y_matplot, 'Standard Deviation': std})
            data = pd.DataFrame({'Episode': x_episode, 'Mean Cumulative Reward': y_matplot,
                                 'Standard Deviation': std})
            csv_dir = f'result/csv/{current_time}/{name}'
            if not os.path.exists(csv_dir):
                os.makedirs(csv_dir)
            data.to_csv(f'{csv_dir}/data_{plottimes}.csv', index=False)
            # 绘制平滑曲线
            plt.plot(x_episode, y_matplot, label=name)
            # 绘制阴影区域
            # plt.fill_between(x_episode, y_matplot - std, y_matplot + std, alpha=0.2)
            # std_lower = [y - s for y, s in zip(y_matplot, std_expanded[:len(y_matplot)])]
            # std_upper = [y + s for y, s in zip(y_matplot, std_expanded[:len(y_matplot)])]
            plt.fill_between(x_episode, y_matplot - std, y_matplot + std, alpha=0.2)
        plt.legend(loc='best')
        plt.savefig(f'result/{current_time}/comparison_plot_{seed}.png')
        plt.show()
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
def main():
    for i in range(5):
        sd = random.randint(1,50)
        set_seed(sd)
        global node_data
        global current_time
        node_data = load_node_data(nodeSets)
        algorithms_data = {}
        algorithms = {
            # 'PD3QN2': Pd3qn2(),
#             'DuelingDDQN': DuelingDDQN(),
            'PD3QND': Pd3qnd(),
            'PD3QN11': Pd3qn11(),
#             'PD3QN101': Pd3qn101(),
#             'DQN': DQN(),
#             'DDQN': DDQN()
        }

        for algo_name, algo_instance in algorithms.items():
            if algo_name == 'PD3QND':
                algorithms_data[algo_name] = run_demo_experiment(algo_instance, algo_name)
            else:
                algorithms_data[algo_name] = run_experiment(algo_instance, algo_name)

        plot_comparison(algorithms_data,sd)
if __name__ == '__main__':
    main()