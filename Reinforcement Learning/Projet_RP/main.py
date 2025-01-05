import random

### Environnements
from environnements.hangar_world import HangarWorldMDP
from environnements.entrepot_world import EntrepotWorldMDP
from environnements.garage_world import GarageWorldMDP

### Agents

## Model-free
# Agent pour Monte Carlo
from agents.agent_monte_carlo import MonteCarloAgent

# Agent pour SARSA
from agents.agent_sarsa import SarsaAgent

# Agent pour Q-Learning
from agents.agent_qlearning import QLearningAgent

## Model-based
#! TODO
from agents.agent_monte_carlo_model_based import MonteCarloAgentModelBased

if __name__ == "__main__":
    # Création de la grille
    hangar = HangarWorldMDP(4, 4, 1)
    print("Grille initiale:")
    hangar.print_board()

    # Monte Carlo
    print("=" * 20, "Monte Carlo (model free)", "=" * 20, "\n")

    mc_agent = MonteCarloAgent(hangar, epsilon=0.3, gamma=0.9)

    mc_agent.train(episodes=1000)

    print("Politique après apprentissage Monte Carlo:")
    mc_agent.print_policy()

    print("Valeur des états après apprentissage Monte Carlo:")
    mc_agent.print_value_function()

    # Monte Carlo (model based)
    print("=" * 20, "Monte Carlo (model based)", "=" * 20, "\n")

    mc_agent = MonteCarloAgentModelBased(hangar, epsilon=0.3, gamma=0.9)

    mc_agent.train(episodes=1000)

    print("Politique après apprentissage Monte Carlo:")
    mc_agent.print_policy()

    print("Valeur des états après apprentissage Monte Carlo:")
    mc_agent.print_value_function()
