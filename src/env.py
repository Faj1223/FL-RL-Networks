import numpy as np
import random

class WirelessEnv:
    def __init__(self, num_antennas=5, num_users=10):
        self.num_antennas = num_antennas
        self.num_users = num_users
        self.antennas = self._generate_antennas()
        self.reset()

    def _generate_users(self):
        """ Génère des utilisateurs avec des positions aléatoires """
        return [{
            "id": i,
            "pos": np.random.uniform(0, 100, 2),  # Position aléatoire
            "demand": random.uniform(1, 10)  # Demande aléatoire en puissance/bande passante
        } for i in range(self.num_users)]

    def _generate_antennas(self):
        """ Génère des antennes avec des positions FIXES """
        return [{
            "id": i,
            "pos": np.array([i * 20, 50]),  # Antennes espacées de 20m
            "power": random.uniform(10, 20)  # Puissance maximale disponible
        } for i in range(self.num_antennas)]

    def _get_state(self):
        """ Retourne l'état actuel du réseau """
        return {
            "users": self.users,
            "antennas": self.antennas
        }

    def step(self, actions):
        """
        Actions = demandes des utilisateurs sous forme de {user_id: demande}
        """
        if not isinstance(actions, dict):
            raise ValueError("Les actions doivent être un dictionnaire {user_id: demande}")

        rewards = self._compute_rewards(actions)
        self.state = self._get_state()
        return self.state, rewards

    def _compute_rewards(self, actions):
        """ Calcule les récompenses des antennes en prenant en compte l'interférence """
    
        rewards = {a["id"]: 0 for a in self.antennas}
        antenna_users = {a["id"]: [] for a in self.antennas}

        # Assignation des utilisateurs aux antennes les plus proches
        for user in self.users:
            user_id = user["id"]
            user_demand = actions.get(user_id, 0)  # Demande de l'utilisateur
            distances = {a["id"]: np.linalg.norm(a["pos"] - user["pos"]) for a in self.antennas}
            assigned_antenna_id = min(distances, key=distances.get)  # Antenne la plus proche
            antenna_users[assigned_antenna_id].append((user_id, user_demand, distances[assigned_antenna_id]))

        # Calcul du signal envoyé par chaque antenne
        signals_sent = {antenna_id: sum(demand for _, demand, _ in users) for antenna_id, users in antenna_users.items()}

        # Calcul des récompenses en prenant en compte l'interférence
        for antenna_id, users in antenna_users.items():
            total_reward = 0  # Récompense totale pour cette antenne
        
            for user_id, demand, distance in users:
                # Signal utile reçu par l'utilisateur
                received_signal = signals_sent[antenna_id] / (1 + distance)
            
            # Calcul de l'interférence provenant des autres utilisateurs
                interference = 0
                for other_antenna_id, other_users in antenna_users.items():
                    if other_antenna_id != antenna_id:  # On prend en compte seulement les autres antennes
                        for other_user_id, other_demand, other_distance in other_users:
                            d_user_otheruser = np.linalg.norm(self.users[user_id]["pos"] - self.users[other_user_id]["pos"])  # Distance entre users
                            interference += (1/(1 + d_user_otheruser)) * (signals_sent[other_antenna_id] / (1 + other_distance))
            
                # Calcul final de la récompense
                reward = received_signal - interference
                total_reward += max(0, reward)  # On s'assure que la récompense ne soit pas négative
    
            rewards[antenna_id] = total_reward  # On attribue la récompense finale à l'antenne

        return rewards


    def reset(self):
        """ l'environnement ne change pas """
        #self.users = self._generate_users()
        #self.antennas = self._generate_antennas()
        self.state = self._get_state()
        return self.state
