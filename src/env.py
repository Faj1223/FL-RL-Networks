import numpy as np
import random

class WirelessEnv:
    def __init__(self, num_antennas=5, num_users=10):
        self.num_antennas = num_antennas
        self.num_users = num_users
        self.antennas = self._generate_antennas()
        self.users = self._generate_users()
        self._reset()


    def _generate_antennas(self):
        """ Génère des antennes avec des positions FIXES """
        return [{
            "id": i,
            "pos": np.array([i * 20, 50]),  # Antennes espacées de 20m
        } for i in range(self.num_antennas)]
    



    def _generate_users(self):
        """ Génère des utilisateurs avec des positions aléatoires """
        return [{
            "id": i,
            "pos": np.random.uniform(0, 100, 2),  # Position aléatoire
            "demand": 2*(i^2)+(3*i)+1  # Demande en puissance/bande passante
        } for i in range(self.num_users)]
    



    def _user_antennas_distance(self) -> dict:
        """Cette fonction retourne un disctionnaire qui contient les affectations de 
        chaque utilisateur à chaque antenne la plus proche"""
        
        
        affectation = {
            u["id"]: {
                a["id"]: np.linalg.norm(u["pos"] - a["pos"])
                for a in self.antennas
            }
            for u in self.users
        }
        # pour chaque utilisateur , la distance qui le sépare de toutes les antennes

        return affectation




    def _get_state(self):
        """ Retourne l'état actuel du réseau """
        return {
            "users": self.users,
            "antennas": self.antennas
        }   

    def _reset(self):
        """ l'environnement ne change pas """
        self.users = self._generate_users()
        self.antennas = self._generate_antennas()
        self.state = self._get_state()
        return self.state
