import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import collections  
import random

class Net(nn.Module):
    def __init__(self, nstates, nactions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(nstates, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, nactions),
        )
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)

class Double_DQN:
    def __init__(self, n_states, n_actions,
                 learning_rate, gamma,
                  device):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.device = device
        
        self.q_net = Net(self.n_states, self.n_actions)
        self.target_q_net = Net(self.n_states, self.n_actions)
        self.target_q_net.load_state_dict(self.target_q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
 
    def take_action(self, state, epsilon):
        if random.random() < epsilon:
            action = np.random.randint(self.n_actions)
        else:
            q_values = self.q_net(state)
            action = q_values.argmax().item()
        return action
 
    def max_q_value(self, state):
        state = torch.tensor(state, dtype=torch.float).view(1,-1)
        max_q = self.q_net(state).max().item()
        return max_q
 
    def update1(self, transitions_dict, batch_size):
        states = torch.tensor(transitions_dict['states'], dtype=torch.float)
        states = states.reshape([batch_size,968])
        actions = torch.tensor(transitions_dict['actions'], dtype=torch.int64).view(-1,1)
        rewards = torch.tensor(transitions_dict['rewards'], dtype=torch.float).view(-1,1)
        next_states = torch.tensor(transitions_dict['next_states'], dtype=torch.float)
        next_states = next_states.reshape([batch_size,968])
        dones = torch.tensor(transitions_dict['dones'], dtype=torch.float).view(-1,1)
 
        q_values = self.q_net(states).gather(1, actions)
        max_action = self.q_net(next_states).max(1)[1].view(-1,1)
        max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        q_targets = rewards + self.gamma * max_next_q_values * (1-dones)
 
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

    def update2(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())
 