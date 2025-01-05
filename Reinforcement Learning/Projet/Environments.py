import random

class GridWorld:
    def __init__(self, height: int, width: int, task_type: str):
        self.height = height
        self.width = width
        self.task_type = task_type

        self.states = set((i, j) for i in range(height) for j in range(width))
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # UP, DOWN, LEFT, RIGHT

        self.obstacle_cells = set(random.sample(
            list(self.states - {(0, 0), (height - 1, width - 1)}),
            min(len(self.states) // 5, len(self.states) - 2)
        ))

    def is_valid(self, state):
        return state in self.states and state not in self.obstacle_cells

    def get_reward(self, current_state, next_state):
        """
        Compute the reward for a given transition.
        """
        if self.task_type == "H":
            # Hangar: Reward for collecting items and reaching the exit
            if next_state == self.exit_state:
                return 100  # Large reward for reaching the target
            elif next_state in self.object_cells:
                return 50  # Reward for collecting objects
        elif self.task_type == "E":
            # Entrepot: Reward for cleaning dirt
            if next_state in self.object_cells:
                return 20  # Reward for cleaning a dirty cell
            elif next_state == self.exit_state:
                return 50  # Bonus for reaching the exit
        elif self.task_type == "G":
            # Garage: Reward for reaching a recharge station
            if next_state in self.object_cells:
                return 30  # Reward for reaching a recharge station
        return -1  # Default step penalty to encourage efficient paths

    def step(self, current_state, action):
        """
        Take an action and return the next state and associated reward.
        """
        next_state = (current_state[0] + action[0], current_state[1] + action[1])
        if not self.is_valid(next_state):
            next_state = current_state  # Stay in place if invalid
        reward = self.get_reward(current_state, next_state)
        return next_state, reward

    def print_board(self):
        """
        Print the grid layout.
        """
        cell_width = 3
        horizontal_border = "+" + ("-" * cell_width + "+") * self.width

        print(horizontal_border)
        for i in range(self.height):
            row = "|"
            for j in range(self.width):
                cell = "."
                if (i, j) in self.obstacle_cells:
                    cell = "X"
                elif self.task_type == "H" and (i, j) in self.object_cells:
                    cell = "Bu" if (i, j) == self.bucket_cell else "Br"
                elif self.task_type == "E" and (i, j) in self.object_cells:
                    cell = "D"
                elif self.task_type == "G" and (i, j) in self.object_cells:
                    cell = "R"
                elif (i, j) == self.exit_state and self.task_type != "G":
                    cell = "T"
                row += cell.center(cell_width) + "|"
            print(row)
            print(horizontal_border)


class Hangar(GridWorld):
    def __init__(self, height=5, width=5):
        super().__init__(height, width, task_type='H')
        self.start_state = (0, 0)
        self.exit_state = (height - 1, width - 1)

        self.bucket_cell, self.broom_cell = random.sample(
            list(self.states - {self.start_state, self.exit_state}),
            2
        )
        self.object_cells = {self.bucket_cell, self.broom_cell}


class Entrepot(GridWorld):
    def __init__(self, height=7, width=7, num_dirt=10):
        super().__init__(height, width, task_type='E')
        self.start_state = (0, 0)
        self.exit_state = (0, width - 1)

        self.obstacle_cells = set(self.obstacle_cells)
        self.object_cells = set(random.sample(
            list(self.states - self.obstacle_cells - {self.start_state, self.exit_state}),
            num_dirt
        ))


class Garage(GridWorld):
    def __init__(self, height=5, width=5, num_recharge_stations=2):
        super().__init__(height, width, task_type='G')
        self.start_state = (height - 1, 0)
        self.exit_state = None

        self.obstacle_cells = set(self.obstacle_cells)
        self.object_cells = set(random.sample(
            list(self.states - self.obstacle_cells - {self.start_state}),
            num_recharge_stations
        ))


if __name__ == "__main__":
    print("Hangar Environment:")
    hangar = Hangar()
    hangar.print_board()

    print("\nEntrepot Environment:")
    entrepot = Entrepot()
    entrepot.print_board()

    print("\nGarage Environment:")
    garage = Garage()
    garage.print_board()
