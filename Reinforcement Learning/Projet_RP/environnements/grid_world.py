class GridWorldMDP:
    def print_board(self):
        cell_width = 3
        horizontal_border = "+" + ("-" * cell_width + "+") * self.width

        print(horizontal_border)
        for i in range(self.height):
            row = "|"
            for j in range(self.width):
                if (i, j) == self.terminal_state:
                    cell = "T".center(cell_width)
                elif (i, j) in self.bad_states:
                    cell = "X".center(cell_width)
                else:
                    cell = ".".center(cell_width)
                row += cell + "|"
            print(row)
            print(horizontal_border)
        print()

    def print_policy(self, policy: dict):
        cell_width = 3

        horizontal_border = "+" + ("-" * cell_width + "+") * self.width

        print(horizontal_border)
        for i in range(self.height):
            row = "|"
            for j in range(self.width):
                if (i, j) == self.terminal_state:
                    cell = "T".center(cell_width)
                elif (i, j) in self.bad_states:
                    cell = "X".center(cell_width)
                else:
                    action = policy[(i, j)]
                    # Use arrows to represent actions
                    if action == (1, 0):
                        cell = "↓".center(cell_width)
                    elif action == (-1, 0):
                        cell = "↑".center(cell_width)
                    elif action == (0, 1):
                        cell = "→".center(cell_width)
                    elif action == (0, -1):
                        cell = "←".center(cell_width)
                    else:
                        cell = " ".center(cell_width)  # Fallback for undefined actions
                row += cell + "|"
            print(row)
            print(horizontal_border)
        print()

    def print_value_function(self, V):
        max_length = max(
            len(f"{V.get((i, j), 0):.2f}")
            for i in range(self.height)
            for j in range(self.width)
        )

        cell_width = max_length + 2
        horizontal_border = "+" + ("-" * cell_width + "+") * self.width

        print(horizontal_border)
        for i in range(self.height):
            row = "|"
            for j in range(self.width):
                if (i, j) == self.terminal_state:
                    cell = "T".center(cell_width)
                elif (i, j) in self.bad_states:
                    cell = "X".center(cell_width)
                else:
                    cell = f"{V.get((i, j), 0):.2f}".center(cell_width)
                row += cell + "|"
            print(row)
            print(horizontal_border)
        print()
