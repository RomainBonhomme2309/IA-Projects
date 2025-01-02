import random
from Environments import Hangar, Entrepot, Garage

class Agent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {(state, action): 0.0 for state in env.states for action in env.actions}
        self.path = []  # To track the path taken by the agent

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.env.actions)
        else:
            q_values = {action: self.q_table[(state, action)] for action in self.env.actions}
            max_q = max(q_values.values())
            return random.choice([action for action, q in q_values.items() if q == max_q])

    def update_q_value(self, state, action, reward, next_state):
        max_next_q = max(self.q_table[(next_state, a)] for a in self.env.actions)
        self.q_table[(state, action)] += self.alpha * (
            reward + self.gamma * max_next_q - self.q_table[(state, action)]
        )

    def step(self, state, action):
        next_state = (state[0] + action[0], state[1] + action[1])
        if next_state in self.env.obstacle_cells or next_state not in self.env.states:
            return state  # Stay in the same state if invalid move
        return next_state

    def get_reward(self, state, action, next_state):
        if self.env.task_type == 'H':
            if next_state in self.env.object_cells:
                self.env.object_cells.remove(next_state)
                return 10
            elif next_state == self.env.exit_state:
                return 20
        elif self.env.task_type == 'E':
            if next_state in self.env.object_cells:
                self.env.object_cells.remove(next_state)
                return 10
            elif not self.env.object_cells:
                return 20
        elif self.env.task_type == 'G':
            if next_state in self.env.object_cells:
                return 20
        return -1

    def train(self, episodes=1000):
        for _ in range(episodes):
            state = self.env.start_state
            while True:
                action = self.choose_action(state)
                next_state = self.step(state, action)
                reward = self.get_reward(state, action, next_state)
                self.update_q_value(state, action, reward, next_state)
                state = next_state
                if state == self.env.exit_state or state not in self.env.states:
                    break

    def execute_task(self):
        state = self.env.start_state
        self.path = [state]
        while True:
            action = self.choose_action(state)
            next_state = self.step(state, action)

            # Update environment for specific tasks
            if self.env.task_type == 'H' and next_state in self.env.object_cells:
                self.env.object_cells.remove(next_state)
            elif self.env.task_type == 'E' and next_state in self.env.object_cells:
                self.env.object_cells.remove(next_state)

            self.path.append(next_state)
            if next_state == self.env.exit_state or next_state not in self.env.states:
                break
            state = next_state

        return self.path

    def print_with_path(self, path):
        """
        Print the grid with the path marked by '*'.
        """
        grid = [["." for _ in range(self.env.width)] for _ in range(self.env.height)]
        for (i, j) in self.env.obstacle_cells:
            grid[i][j] = "X"
        for (i, j) in self.env.object_cells:
            if self.env.task_type == 'H':
                grid[i][j] = "Bu" if (i, j) == self.env.bucket_cell else "Br"
            elif self.env.task_type == 'E':
                grid[i][j] = "D"
            elif self.env.task_type == 'G':
                grid[i][j] = "R"
        for (i, j) in path:
            grid[i][j] = "*" if grid[i][j] == "." else grid[i][j]

        # Print the grid
        horizontal_border = "+" + "---+" * self.env.width
        print(horizontal_border)
        for row in grid:
            print("|" + "|".join(cell.center(3) for cell in row) + "|")
            print(horizontal_border)


# Example usage
if __name__ == "__main__":
    print("Hangar Environment:")
    hangar = Hangar()
    hangar.print_board()
    hangar_agent = Agent(hangar)
    hangar_agent.train(episodes=500)
    path_hangar = hangar_agent.execute_task()
    print("\nPath taken in Hangar:", path_hangar)
    hangar_agent.print_with_path(path_hangar)

    print("\nEntrepot Environment:")
    entrepot = Entrepot()
    entrepot.print_board()
    entrepot_agent = Agent(entrepot)
    entrepot_agent.train(episodes=500)
    path_entrepot = entrepot_agent.execute_task()
    print("\nPath taken in Entrepot:", path_entrepot)
    entrepot_agent.print_with_path(path_entrepot)

    print("\nGarage Environment:")
    garage = Garage()
    garage.print_board()
    garage_agent = Agent(garage)
    garage_agent.train(episodes=500)
    path_garage = garage_agent.execute_task()
    print("\nPath taken in Garage:", path_garage)
    garage_agent.print_with_path(path_garage)
