import numpy as np
import random

class WirelessEnv:
    def __init__(self, num_antennas=5, num_users=10):
        self.num_antennas = num_antennas
        self.num_users = num_users
        self.antennas = self._generate_antennas()
        self.users = self._generate_users()
        self.reset()

    def _generate_users(self):
        """ Génère des utilisateurs avec des positions aléatoires """
        return [{
            "id": i,
            "pos": np.random.uniform(0, 100, 2),  # Position aléatoire
            "demand": 1+i  # Demande aléatoire en puissance/bande passante
        } for i in range(self.num_users)]

    def _generate_antennas(self):
        """ Génère des antennes avec des positions FIXES """
        return [{
            "id": i,
            "pos": np.array([i * 20, 50]),  # Antennes espacées de 20m
        } for i in range(self.num_antennas)]

    def _get_state(self):
        """ Retourne l'état actuel du réseau """
        return {
            "users": self.users,
            "antennas": self.antennas
        }

    def step(self, demand):
        """
        demand = demandes des utilisateurs sous forme de {user_id: demande}
        """
        if not isinstance(demand, dict):
            raise ValueError("Les actions doivent être un dictionnaire {user_id: demande}")

        rewards = self._compute_rewards(demand)
        self.state = self._get_state()
        return self.state, rewards   


    def reset(self):
        """ l'environnement ne change pas """
        self.users = self._generate_users()
        self.antennas = self._generate_antennas()
        self.state = self._get_state()
        return self.state
