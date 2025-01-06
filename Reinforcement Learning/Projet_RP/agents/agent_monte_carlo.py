from collections import defaultdict
import random


class MonteCarloAgent:
    def __init__(self, mdp, epsilon=0.1, gamma=0.99):
        self.mdp = mdp
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = defaultdict(lambda: 0)
        self.returns = defaultdict(list)
        self.policy = {
            state: random.choice(self.mdp.actions) for state in self.mdp.states
        }

    def generate_episode(self, max_steps=1000):
        state = self.mdp.initial_state
        episode = []

        count = 0
        while state != self.mdp.terminal_state and count < max_steps:
            action = self.select_action(state)
            new_state = self.mdp.take_action(state, action)
            reward = self.mdp.rewards.get((state, action, new_state), 0)
            episode.append((state, action, reward))
            state = new_state
            count += 1

        return episode

    def select_action(self, state):
        # ε-greedy
        if random.random() < self.epsilon:
            return random.choice(self.mdp.actions)
        else:
            q_values = {action: self.Q[(state, action)] for action in self.mdp.actions}
            return max(q_values, key=q_values.get)

    def update_policy(self):
        for state in self.mdp.states:
            q_values = {action: self.Q[(state, action)] for action in self.mdp.actions}
            self.policy[state] = max(q_values, key=q_values.get)

    def train(self, episodes=1000):
        for _ in range(episodes):
            print("\rEpisode {}/{}".format(_ + 1, episodes), end="")
            episode = self.generate_episode()
            G = 0
            visited = set()

            for state, action, reward in reversed(episode):
                G = reward + self.gamma * G
                if (state, action) not in visited:
                    self.returns[(state, action)].append(G)
                    self.Q[(state, action)] = sum(self.returns[(state, action)]) / len(
                        self.returns[(state, action)]
                    )
                    visited.add((state, action))

            self.update_policy()
        print("\n")

    def print_policy(self):
        self.mdp.print_policy(self.policy)

    def print_value_function(self):
        V = {
            state: max(self.Q[(state, action)] for action in self.mdp.actions)
            for state in self.mdp.states
        }
        self.mdp.print_value_function(V)
