### Environnements
from environnements.hangar_world import HangarWorldMDP
from environnements.entrepot_world import EntrepotWorldMDP
from environnements.garage_world import GarageWorldMDP

### Agents
# Monte Carlo (model free)
from agents.agent_monte_carlo import MonteCarloAgent

# Value Iteration (model based)
from agents.agent_value_iteration import ValueIterationAgent


def run_task(
    env,
    agent_model_free,
    name_model_free,
    agent_model_based,
    name_model_based,
    episodes=1000,
    max_iterations=5000,
):
    # Environment
    env.print_board()

    # Model Free
    print("." * 10, f"{name_model_free} (model free)", "." * 10, "\n")

    agent_model_free.train(episodes=episodes)

    print(f"Policy after learning {name_model_free}:")
    agent_model_free.print_policy()

    # Model Based
    print("." * 10, f"{name_model_based} (model based)", "." * 10, "\n")

    agent_model_based.train(max_iterations=max_iterations)

    print(f"Policy after learning {name_model_based}:")
    agent_model_based.print_policy()


if __name__ == "__main__":
    # Environnement 1: Hangar
    print("=" * 20, "Environnement 1: HANGAR", "=" * 20)
    env = HangarWorldMDP(4, 4, 1, 2)
    agent_model_free = MonteCarloAgent(env, epsilon=0.3, gamma=0.99)
    agent_model_based = ValueIterationAgent(env, gamma=0.9, theta=0.0001)
    run_task(
        env,
        agent_model_free,
        "Monte Carlo",
        agent_model_based,
        "Value Iteration",
        episodes=1000,
        max_iterations=5000,
    )

    # Environnement 2: Entrepot
    print("=" * 20, "Environnement 2: ENTREPOT", "=" * 20)
    env = EntrepotWorldMDP(4, 4, 1, 3)
    agent_model_free = MonteCarloAgent(env, epsilon=0.3, gamma=0.99)
    agent_model_based = ValueIterationAgent(env, gamma=0.9, theta=0.0001)
    run_task(
        env,
        agent_model_free,
        "Monte Carlo",
        agent_model_based,
        "Value Iteration",
        episodes=1000,
        max_iterations=5000,
    )

    # Environnement 3: Garage
    print("=" * 20, "Environnement 3: GARAGE", "=" * 20)
    env = GarageWorldMDP(4, 4, 1, 2)
    agent_model_free = MonteCarloAgent(env, epsilon=0.3, gamma=0.99)
    agent_model_based = ValueIterationAgent(env, gamma=0.9, theta=0.0001)
    run_task(
        env,
        agent_model_free,
        "Monte Carlo",
        agent_model_based,
        "Value Iteration",
        episodes=1000,
        max_iterations=5000,
    )
