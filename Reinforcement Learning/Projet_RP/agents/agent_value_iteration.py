class ValueIterationAgent:
    def __init__(self, mdp, gamma=0.99, theta=0.0001):
        self.mdp = mdp
        self.gamma = gamma
        self.theta = theta
        self.V = {state: 0 for state in self.mdp.states}
        self.policy = {state: self.mdp.actions[0] for state in self.mdp.states}

    def get_max_action_value(self, state):
        if state == self.mdp.terminal_state or state in self.mdp.bad_states:
            return 0, self.mdp.actions[0]

        action_values = {}
        for action in self.mdp.actions:
            value = 0
            for next_state in self.mdp.states:
                prob = self.mdp.transition_probabilities.get(
                    (state, action, next_state), 0
                )
                reward = self.mdp.get_reward((state, action, next_state))
                value += prob * (reward + self.gamma * self.V[next_state])
            action_values[action] = value

        best_action = max(action_values, key=action_values.get)
        return action_values[best_action], best_action

    def train(self, max_iterations=1000):
        for i in range(max_iterations):
            delta = 0
            self.mdp.reset()
            for state in self.mdp.states:
                old_value = self.V[state]
                max_value, best_action = self.get_max_action_value(state)
                self.V[state] = max_value
                self.policy[state] = best_action
                delta = max(delta, abs(old_value - self.V[state]))

            print(f"\rIteration {i+1}/{max_iterations}, Delta: {delta:.6f}", end="")
            if delta < self.theta:
                print(f"\nConverged after {i+1} iterations")
                break
        print("\n")

    def print_policy(self):
        self.mdp.print_policy(self.policy)

    def print_value_function(self):
        V = {state: self.V[state] for state in self.mdp.states}
        self.mdp.print_value_function(V)
