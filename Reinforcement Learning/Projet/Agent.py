from Environments import Hangar, Entrepot, Garage
from collections import defaultdict
import random

class QLearningAgent:
    def __init__(self, mdp, alpha=0.1, epsilon=0.1, gamma=0.99):
        self.mdp = mdp
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = defaultdict(lambda: 0)  # Q-values initialized to 0
        self.policy = {state: random.choice(self.mdp.actions) for state in self.mdp.states}

    def select_action(self, state):
        # ε-greedy policy
        if random.random() < self.epsilon:
            return random.choice(self.mdp.actions)
        else:
            q_values = {action: self.Q[(state, action)] for action in self.mdp.actions}
            return max(q_values, key=q_values.get)

    def train(self, episodes=1000):
        for episode in range(episodes):
            state = self.mdp.start_state

            while state != self.mdp.exit_state:
                action = self.select_action(state)

                # Use environment's step method to get next state and reward
                next_state, reward = self.mdp.step(state, action)

                # Update Q-value
                best_next_action = max(self.mdp.actions, key=lambda a: self.Q[(next_state, a)])
                td_target = reward + self.gamma * self.Q[(next_state, best_next_action)]
                td_error = td_target - self.Q[(state, action)]
                self.Q[(state, action)] += self.alpha * td_error

                state = next_state

            # Optionally log progress
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{episodes} completed.")

    def update_policy(self):
        self.policy = {
            state: max(
                {action: self.Q[(state, action)] for action in self.mdp.actions},
                key=lambda a: self.Q[(state, a)]
            ) for state in self.mdp.states
        }

    def print_policy(self):
        self.update_policy()
        self.mdp.print_policy(self.policy)

    def print_value_function(self):
        V = {state: max(self.Q[(state, action)] for action in self.mdp.actions) for state in self.mdp.states}
        self.mdp.print_value_function(V)


if __name__ == "__main__":
    # Initialiser les environnements
    hangar = Hangar()
    entrepot = Entrepot()
    garage = Garage()

    # Afficher les environnements initiaux
    print("Initial Hangar Environment:")
    hangar.print_board()

    print("\nInitial Entrepot Environment:")
    entrepot.print_board()

    print("\nInitial Garage Environment:")
    garage.print_board()

    # Créer et entraîner les agents pour chaque tâche
    print("\nTraining for Hangar...")
    agent_h = QLearningAgent(hangar)
    agent_h.train(episodes=1000)
    agent_h.print_policy()

    print("\nTraining for Entrepot...")
    agent_e = QLearningAgent(entrepot)
    agent_e.train(episodes=1000)
    agent_e.print_policy()

    print("\nTraining for Garage...")
    agent_g = QLearningAgent(garage)
    agent_g.train(episodes=1000)
    agent_g.print_policy()
