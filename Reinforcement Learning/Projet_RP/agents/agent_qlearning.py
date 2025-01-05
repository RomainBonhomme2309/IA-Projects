from collections import defaultdict
import random

class QLearningAgent:
    def __init__(self, mdp, alpha=0.1, epsilon=0.1, gamma=0.99):
        self.mdp = mdp
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = defaultdict(lambda: 0)
        self.policy = {state: random.choice(self.mdp.actions) for state in self.mdp.states}

    def select_action(self, state):
        # ε-greedy
        if random.random() < self.epsilon:
            return random.choice(self.mdp.actions)
        else:
            q_values = {action: self.Q[(state, action)] for action in self.mdp.actions}
            return max(q_values, key=q_values.get)

    def take_action(self, state, action):
        new_state = (state[0] + action[0], state[1] + action[1])
        if new_state not in self.mdp.states or new_state in self.mdp.bad_states:
            new_state = state
        return new_state

    def train(self, episodes=1000):
        for _ in range(episodes):
            state = self.mdp.initial_state

            while state != self.mdp.terminal_state:
                action = self.select_action(state)

                new_state = self.take_action(state, action)
                reward = self.mdp.rewards.get((state, action, new_state), 0)

                best_next_action = max(self.mdp.actions, key=lambda a: self.Q[(new_state, a)])
                td_target = reward + self.gamma * self.Q[(new_state, best_next_action)]
                td_error = td_target - self.Q[(state, action)]

                self.Q[(state, action)] += self.alpha * td_error

                state = new_state

    def update_policy(self):
        self.policy = {state: max(
            {action: self.Q[(state, action)] for action in self.mdp.actions},
            key=lambda a: self.Q[(state, a)]
        ) for state in self.mdp.states}

    def print_policy(self):
        self.update_policy()
        self.mdp.print_policy(self.policy)

    def print_value_function(self):
        V = {state: max(self.Q[(state, action)] for action in self.mdp.actions) for state in self.mdp.states}
        self.mdp.print_value_function(V)
