# FL-RL-Networks
**Federated Reinforcement Learning for Wireless Networks**  

## 📌 **Description**  
Ce projet explore l'utilisation de l'apprentissage fédéré (**Federated Learning, FL**) combiné à l'apprentissage par renforcement (**Reinforcement Learning, RL**) pour optimiser la gestion dynamique des ressources dans les réseaux sans fil. Chaque antenne s'entraîne localement et partage uniquement les mises à jour de son modèle (sans partage de données locales), réduisant ainsi la charge réseau et garantissant la confidentialité des données.  

## **Objectifs**  
- Implémenter un cadre d'**apprentissage fédéré** pour la gestion des réseaux sans fil.  
- Utiliser des **agents RL** pour ajuster dynamiquement les paramètres du réseau (fréquence, puissance, allocation de spectre).  
- Comparer les performances de **l’apprentissage fédéré** à un apprentissage centralisé classique.  

## 🔧 **Architecture du projet**  
📡 **Modèle Local** (Antenne)  
- Un agent RL entraîne un modèle pour optimiser ses décisions en fonction des conditions et contraintes du réseau qui évoluent dans le temps.
- Les antennes mettent à jour leurs modèles en local, sans partager les données brutes.  

🌍 **Serveur Global**  
- Réception des modèles locaux.  
- Agrégation des poids des modèles avec **Federated Averaging (FedAvg) etc...**.  
- Distribution du modèle mis à jour aux antennes.  

## 🛠 **Technologies utilisées**  
- **Langage** : Python 🐍  
- **Frameworks** : PyTorch/TensorFlow, FedML/Flower (pour FL), Stable-Baselines3 (pour RL)  
- **Librairies** : NumPy, Matplotlib, Gym, Scikit-learn  

## 🚀 **Installation**  
```bash
git clone https://github.com/ton-profil/FedRL-Wireless.git
cd FedRL-Wireless
pip install -r requirements.txt
```

## **Structure du projet**  
```
FedRL-Wireless/
│── data/                # Jeux de données simulées
│── models/              # Modèles entraînés (RL & FL)
│── src/                 # Code source principal
│   ├── env.py           # Environnement de simulation
│   ├── agent.py         # Implémentation de l’agent RL
│   ├── federated.py     # Logique d’apprentissage fédéré
│   ├── train.py         # Script d'entraînement principal
│── results/             # Résultats et visualisations
│── README.md            # Documentation du projet
│── requirements.txt     # Dépendances du projet
```

## **Résultats attendus**  
✅ Optimisation dynamique de l’allocation des ressources réseau.  
✅ Réduction des interférences et amélioration de la qualité du signal.  
✅ Comparaison des performances **FL vs RL Centralisé**.  

## **À venir**  
- Intégration d'autres stratégies d'agrégation FL (FedProx, Scaffold).  
- Expérimentations avec différents algorithmes de RL (DQN, PPO).  

## 👨‍💻 **Contributeurs**  
🔹 [Ton Nom](https://github.com/Faj1223)  
