import agent
from env import WirelessEnv


def compute_rewards( response_per_antenna: dict, env: WirelessEnv, alpha = 0.3)-> dict:
        """
        Calcule les récompenses des antennes en fonction de la reponse des antennes
        et de l'interférence.
        """
        distances = env._user_antennas_distance()
        rewards = {a["id"]: 0 for a in env.antennas}
        for user in env.users:
                user_id = user["id"]
                user_distances = distances[user_id]

                # l'antenne la plus proche
                connected_antenna_id = min(user_distances, key= user_distances.get)

                # signal util
                signal_util = response_per_antenna[connected_antenna_id]/(1+user_distances[connected_antenna_id])

                # interférences
                interference = sum(
                    response_per_antenna[other_antenna_id] / (1 + user_distances[other_antenna_id])**2
                    for other_antenna_id in response_per_antenna
                    if other_antenna_id != connected_antenna_id
                )
                # Reward pour l'utilisateur
                reward_user = signal_util - alpha * interference
        
                # On l'ajoute à l'antenne connectée
                rewards[connected_antenna_id] += reward_user

        return rewards
