import agent
from env import WirelessEnv


def compute_rewards( demand: dict, env: WirelessEnv)-> dict:
        """
        Calcule les récompenses des antennes en fonction de la demande des utilisateurs
        et de l'interférence entre antennes.
        Les demandes sont un dictionnaire qui contient pour chaque users une demande et la distance à son antenne soit de la forme {user_id : demand, distance}
        
        """
        # Initialisation des récompenses de chaque antenne
        rewards = {a[id] : 0 for a in env.antennas} # Ici on stock les recompenses par antenne
        
        # on recupère les réponses de chaque antennes en sortie du reseau de neuronne agent





        # on calcule le reward de chaque antenne dans un dictionnaire rewards {antenne_id : reward_antenne}

        
        
        return rewards