### Environnements
from environnements.hangar_world import HangarWorldMDP
from environnements.entrepot_world import EntrepotWorldMDP
from environnements.garage_world import GarageWorldMDP

### Agents
# Agent pour Monte Carlo (model free)
from agents.agent_monte_carlo import MonteCarloAgent

# Agent pour Value Iteration (model based)
from agents.agent_value_iteration import ValueIterationAgent

if __name__ == "__main__":
    # Création de Hangar
    hangar = HangarWorldMDP(4, 4, 1)
    print("=" * 20, "Environnement 1: HANGAR", "=" * 20)
    hangar.print_board()

    # Monte Carlo (model free)
    print("=" * 10, "Monte Carlo (model free)", "=" * 10, "\n")

    mc_agent = MonteCarloAgent(hangar, epsilon=0.3, gamma=0.9)

    mc_agent.train(episodes=1000)

    print("Politique après apprentissage Monte Carlo:")
    mc_agent.print_policy()

    print("Valeur des états après apprentissage Monte Carlo:")
    mc_agent.print_value_function()

    # Value Iteration (model based)
    print("=" * 10, "Value Iteration (model based)", "=" * 10, "\n")

    vi_agent = ValueIterationAgent(hangar, gamma=0.9, theta=0.0001)

    vi_agent.train(max_iterations=1000)

    print("Politique après apprentissage Value Iteration:")
    vi_agent.print_policy()

    print("Valeur des états après apprentissage Value Iteration:")
    vi_agent.print_value_function()
