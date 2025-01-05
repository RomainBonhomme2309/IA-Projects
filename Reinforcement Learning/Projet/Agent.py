from Environments import Hangar, Entrepot, Garage
from collections import defaultdict
import random

class MultiTaskQLearningAgent:
    def __init__(self, alpha=0.1, epsilon=0.1, gamma=0.99):
        self.alpha = alpha  # Learning rate
        self.epsilon = epsilon  # Exploration rate for ε-greedy policy
        self.gamma = gamma  # Discount factor for future rewards
        self.Q = defaultdict(lambda: 0)  # Q-values initialized to 0

    def select_action(self, state, actions):
        """
        Select an action based on an ε-greedy policy.
        - Explore with probability epsilon
        - Exploit the learned policy (choose action with highest Q-value) otherwise
        """
        if random.random() < self.epsilon:
            return random.choice(actions)  # Explore: random action
        else:
            q_values = {action: self.Q[(state, action)] for action in actions}
            return max(q_values, key=q_values.get)  # Exploit: action with max Q-value

    def train(self, environments, episodes_per_env=500):
        """
        Train the agent across multiple environments by alternating between them.
        Each environment is trained for a specified number of episodes.
        """
        for episode in range(episodes_per_env * len(environments)):
            # Alternate between environments (round-robin style)
            env = environments[episode % len(environments)]
            state = (env.task_type, env.start_state)  # Include task type in the state

            while state[1] != env.exit_state:  # Continue until reaching the exit
                action = self.select_action(state, env.actions)
                next_state_coords, reward = env.step(state[1], action)
                next_state = (env.task_type, next_state_coords)  # Update next state with task type

                # Update Q-value using the Bellman equation
                best_next_action = max(env.actions, key=lambda a: self.Q[(next_state, a)])
                td_target = reward + self.gamma * self.Q[(next_state, best_next_action)]
                td_error = td_target - self.Q[(state, action)]
                self.Q[(state, action)] += self.alpha * td_error

                state = next_state  # Move to the next state

            # Log progress every 100 episodes
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{episodes_per_env * len(environments)} completed.")

    def update_policy(self, environments):
        """
        Extract optimal policies for each environment after training.
        A policy maps states to the best possible action for that state.
        """
        self.policies = {}
        for env in environments:
            task_type = env.task_type
            self.policies[task_type] = {
                state: max(
                    {action: self.Q[((task_type, state), action)] for action in env.actions},
                    key=lambda a: self.Q[((task_type, state), a)]
                ) for state in env.states
            }

    def print_policy(self, env):
        """
        Print the learned policy for a specific environment.
        """
        task_policy = self.policies.get(env.task_type, {})
        env.print_policy(task_policy)

    def print_value_function(self, env):
        """
        Print the learned value function for a specific environment.
        """
        task_type = env.task_type
        V = {state: max(self.Q[((task_type, state), action)] for action in env.actions) for state in env.states}
        env.print_value_function(V)


if __name__ == "__main__":
    # Initialize the environments
    hangar = Hangar()
    entrepot = Entrepot()
    garage = Garage()

    # Display the initial environments
    print("Initial Hangar Environment:")
    hangar.print_board()

    print("\nInitial Entrepot Environment:")
    entrepot.print_board()

    print("\nInitial Garage Environment:")
    garage.print_board()

    # Create and train the multi-task agent across all environments
    print("\nTraining Multi-Task Agent...")
    agent = MultiTaskQLearningAgent()
    agent.train([hangar, entrepot, garage], episodes_per_env=500)

    # Update and display the policies for each environment
    print("\nPolicies for Hangar:")
    agent.update_policy([hangar, entrepot, garage])
    agent.print_policy(hangar)

    print("\nPolicies for Entrepot:")
    agent.print_policy(entrepot)

    print("\nPolicies for Garage:")
    agent.print_policy(garage)

