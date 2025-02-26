import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

#  Réseau de l'Actor
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.max_action * torch.relu(torch.tanh(self.fc3(x)))  # Tanh pour garder les actions entre -1 et 1

#  Réseau du Critic
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)  # Q-value

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)  # Concaténer état et action
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

#  Buffer de Rejeu (Replay Buffer)

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states)

    def size(self):
        return len(self.buffer)

#  Agent DDPG

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005, lr=1e-3, buffer_size=100000):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.buffer = ReplayBuffer(buffer_size)
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action

    def select_action(self, state, noise=0.1):
        """ Sélectionne une action en fonction de l'état actuel """
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        action += noise * np.random.randn(*action.shape)  # Ajout d'un bruit pour l'exploration
        return np.clip(action, -self.max_action, self.max_action)

    def train(self, batch_size=64):
        """ Entraîne le modèle avec des échantillons du Replay Buffer """
        if self.buffer.size() < batch_size:
            return  # Attendre d'avoir assez de données

        states, actions, rewards, next_states = self.buffer.sample(batch_size)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)

        # Cible pour le Critic (Q-learning)
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_Q = self.critic_target(next_states, next_actions)
            target_Q = rewards + self.gamma * target_Q

        # Mise à jour du Critic
        current_Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Mise à jour de l'Actor (maximiser la Q-value)
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Mise à jour des réseaux cibles (soft update)
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

    def soft_update(self, net, target_net):
        """ Fait une mise à jour lente des réseaux cibles """
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
