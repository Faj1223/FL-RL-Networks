import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

#  Réseau de l'Actor
class Actor(nn.Module):
    def __init__(self, demand_dim, response_dim, max_response):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(demand_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, response_dim)
        self.max_response = max_response

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.max_response * torch.relu(self.fc3(x))

#  Réseau du Critic
class Critic(nn.Module):
    def __init__(self, demand_dim, response_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(demand_dim + response_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)  # Q-value

    def forward(self, demand, response):
        x = torch.cat([demand, response], dim=1)  # Concaténer demand et reponse
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

#  Buffer de Rejeu (Replay Buffer)

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, demand, response, reward, next_demand):
        self.buffer.append((demand, response, reward, next_demand))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        demand, reponse, rewards, next_demand = zip(*batch)
        return np.array(demand), np.array(reponse), np.array(rewards), np.array(next_demand)

    def size(self):
        return len(self.buffer)

#  Agent DDPG

class DDPGAgent:
    def __init__(self, demand_dim, response_dim, max_response = 10, gamma=0.99, tau=0.005, lr=1e-3, buffer_size=100000):
        self.actor = Actor(demand_dim, response_dim, max_response)
        self.actor_target = Actor(demand_dim, response_dim, max_response)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(demand_dim, response_dim)
        self.critic_target = Critic(demand_dim, response_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.buffer = ReplayBuffer(buffer_size)
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_response

    def select_response(self, demand, noise=0.1):
        """ donne une reponse en fonction de la demande """
        demand = torch.FloatTensor(demand).unsqueeze(0) # Transformer la demande en tenseur PyTorch
        response = self.actor(demand).detach().numpy()[0] # Prédire la reponse avec le réseau Actor
        response += noise * np.random.randn(*response.shape) # Ajout d'un bruit pour l'exploration
        return np.clip(response, 0, self.max_action) # Limiter la reponse à [0, max_action]
    

    def train(self, batch_size=64):
        """ Entraîne le modèle avec des échantillons du Replay Buffer """
        if self.buffer.size() < batch_size:
            return  # Attendre d'avoir assez de données

        demand, response, rewards, next_demand = self.buffer.sample(batch_size)

        demand = torch.FloatTensor(demand)
        response = torch.FloatTensor(response)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_demand = torch.FloatTensor(next_demand)

        # Cible pour le Critic (Q-learning)
        with torch.no_grad():
            next_actions = self.actor_target(next_demand)
            target_Q = self.critic_target(next_demand, next_actions)
            target_Q = rewards + self.gamma * target_Q # formule de Bellman

        # Mise à jour du Critic
        current_Q = self.critic(demand, response)
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Mise à jour de l'Actor (maximiser la Q-value)
        actor_loss = -self.critic(demand, self.actor(demand)).mean()
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
