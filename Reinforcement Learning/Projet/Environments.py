import random

class GridWorld:
    def __init__(self, height: int, width: int, task_type: str):
        """
        Initialize the GridWorld.
        :param height: Number of rows in the grid.
        :param width: Number of columns in the grid.
        :param task_type: Type of the task ('H' for Hangar, 'E' for Entrepot, 'G' for Garage).
        """
        self.height = height
        self.width = width
        self.task_type = task_type

        self.states = set((i, j) for i in range(height) for j in range(width))
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # UP, DOWN, LEFT, RIGHT

        self.obstacle_cells = set(random.sample(
            list(self.states - {(0, 0), (height - 1, width - 1)}),
            min(len(self.states) // 5, len(self.states) - 2)
        ))

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
                    cell = "Bu" if (i, j) == self.bucket_cell else "Br"  # Bucket or Broom
                elif self.task_type == "E" and (i, j) in self.object_cells:
                    cell = "D"  # Dirt
                elif self.task_type == "G" and (i, j) in self.object_cells:
                    cell = "R"  # Recharge station
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

        # Define bucket and broom locations
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

        # Randomly place dirt cells
        self.object_cells = set(random.sample(
            list(self.states - self.obstacle_cells - {self.start_state, self.exit_state}),
            num_dirt
        ))


class Garage(GridWorld):
    def __init__(self, height=5, width=5, num_recharge_stations=2):
        super().__init__(height, width, task_type='G')
        self.start_state = (height - 1, 0)
        self.exit_state = None  # No exit for garage

        self.obstacle_cells = set(self.obstacle_cells)

        # Randomly place recharge stations
        self.object_cells = set(random.sample(
            list(self.states - self.obstacle_cells - {self.start_state}),
            num_recharge_stations
        ))


# Example usage
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
