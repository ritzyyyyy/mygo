import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import random
from Memory import Memory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQfD(nn.Module):
    def __init__(self, env, config, demo_transitions=None):
        super(DQfD, self).__init__()

        self.config = config
        self.replay_memory = Memory(capacity=self.config.replay_buffer_size, permanent_data=len(demo_transitions))
        self.demo_memory = Memory(capacity=self.config.demo_buffer_size, permanent_data=self.config.demo_buffer_size)
        self.add_demo_to_memory(demo_transitions=demo_transitions)  # add demo data to both demo_memory & replay_memory
        self.time_step = 0
        self.epsilon = self.config.INITIAL_EPSILON
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.select_net = self.build_layers().to(device)
        self.eval_net = self.build_layers().to(device)

        self.loss_fn = nn.MSELoss(reduction='none')
        self.optimizer = optim.Adam(self.select_net.parameters(), lr=self.config.LEARNING_RATE)
        self.update_target_net()

    def build_layers(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_dim)
        )

    def add_demo_to_memory(self, demo_transitions):
        # add demo data to both demo_memory & replay_memory
        for t in demo_transitions:
            self.demo_memory.store(np.array(t, dtype=object))
            self.replay_memory.store(np.array(t, dtype=object))
            assert len(t) == 10

    def pre_train(self):
        print('Pre-training ...')
        for i in range(self.config.PRETRAIN_STEPS):
            self.train_Q_network(pre_train=True)
            if i % 200 == 0 and i > 0:
                print('{} th step of pre-train finish ...'.format(i))
        self.time_step = 0
        print('All pre-train finish.')

    def perceive(self, transition):
        self.replay_memory.store(np.array(transition))
        # epsilon->FINAL_EPSILON(min_epsilon)
        if self.replay_memory.full():
            self.epsilon = max(self.config.FINAL_EPSILON, self.epsilon * self.config.EPSILIN_DECAY)

    def train_Q_network(self, pre_train=False, update=True):
        if not pre_train and not self.replay_memory.full():  # sampling should be executed AFTER replay_memory filled
            return
        self.time_step += 1

        assert self.replay_memory.full() or pre_train

        actual_memory = self.demo_memory if pre_train else self.replay_memory
        tree_idxes, minibatch, ISWeights = actual_memory.sample(self.config.BATCH_SIZE)

        np.random.shuffle(minibatch)
        state_batch = torch.tensor([data[0] for data in minibatch]).float().to(device)
        action_batch = torch.tensor([data[1] for data in minibatch]).long().to(device)
        reward_batch = torch.tensor([data[2] for data in minibatch]).float().to(device)
        next_state_batch = torch.tensor([data[3] for data in minibatch]).float().to(device)
        terminal_batch = torch.tensor([data[4] for data in minibatch]).float().to(device)
        is_demo = torch.tensor([data[9] for data in minibatch]).float().to(device)
    # step 1: calculate the target q values
        with torch.no_grad():
            q_next = self.eval_net(next_state_batch).detach()
            q_target = reward_batch + (1 - terminal_batch) * self.config.GAMMA * torch.max(q_next, dim=1)[0]

        # step 2: calculate the predicted q values
        q_pred = self.select_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()

        # step 3: calculate the TD errors and update priorities
        td_errors = q_target - q_pred
        if not pre_train:
            for idx, error in zip(tree_idxes, td_errors):
                actual_memory.update(idx, abs(error.detach().cpu().numpy()))

        # step 4: calculate the loss and optimize
        demo_losses = (self.config.MARGIN * torch.abs(td_errors)) * is_demo
        non_demo_losses = td_errors ** 2 * (1 - is_demo)
        losses = ISWeights * (demo_losses + non_demo_losses)
        loss = torch.mean(losses)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # step 5: update the target network
        if update and self.time_step % self.config.TARGET_UPDATE_FREQ == 0:
            self.update_target_net()

    def update_target_net(self):
        self.eval_net.load_state_dict(self.select_net.state_dict())

    def egreedy_action(self, state):
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return self.action(state)

    def action(self, state):
        state = torch.tensor(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.select_net(state).cpu().numpy()
        return np.argmax(q_values)

    def save_model(self, save_path):
        torch.save(self.select_net.state_dict(), save_path)

    def load_model(self, load_path):
        self.select_net.load_state_dict(torch.load(load_path))
        self.update_target_net()
