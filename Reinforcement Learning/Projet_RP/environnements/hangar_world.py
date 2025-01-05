import random
from .grid_world import GridWorldMDP


class HangarWorldMDP(GridWorldMDP):
    def __init__(self, height: int, width: int, number_of_holes: int):
        self.height = height
        self.width = width
        self.number_of_holes = number_of_holes

        self.states = set((i, j) for i in range(height) for j in range(width))

        UP = (-1, 0)
        DOWN = (1, 0)
        LEFT = (0, -1)
        RIGHT = (0, 1)

        self.actions = [UP, DOWN, LEFT, RIGHT]

        self.bad_states = random.sample(
            list(self.states - {(0, 0), (height - 1, width - 1)}), self.number_of_holes
        )

        self.initial_state = (0, 0)
        self.terminal_state = (height - 1, width - 1)

        forbidden_states = self.bad_states + [self.initial_state, self.terminal_state]

        self.material_state1, self.material_state2 = random.sample(
            list(self.states - set(forbidden_states)), 2
        )

        self.transition_probabilities = {
            (state, action, new_state): 0
            for state in self.states
            for action in self.actions
            for new_state in self.states
        }
        self.rewards = {
            (state, action, new_state): 0
            for state in self.states
            for action in self.actions
            for new_state in self.states
        }

        self.visited_material_states = set()
        self.visited_states = set()  # Track visited states for penalizing revisits

        for state in self.states:
            if state in self.bad_states or state == self.terminal_state:
                continue
            for action in self.actions:
                new_state = (state[0] + action[0], state[1] + action[1])
                if new_state not in self.states or new_state in self.bad_states:
                    new_state = state
                self.transition_probabilities[(state, action, new_state)] = 1
                self.rewards[(state, action, new_state)] = self.compute_reward(
                    state, new_state
                )

    def compute_reward(self, state, new_state):
        if new_state in self.visited_states:
            return -0.2  # Penalty for revisiting the same state

        # Mark the state as visited
        self.visited_states.add(new_state)

        if (
            new_state == self.material_state1
            and self.material_state1 not in self.visited_material_states
        ):
            self.visited_material_states.add(self.material_state1)
            return 0.5  # Partial reward for reaching material_state1
        elif (
            new_state == self.material_state2
            and self.material_state2 not in self.visited_material_states
        ):
            self.visited_material_states.add(self.material_state2)
            return 0.5  # Partial reward for reaching material_state2
        elif new_state == self.terminal_state:
            if (
                self.material_state1 in self.visited_material_states
                and self.material_state2 in self.visited_material_states
            ):
                return 1.0  # Full reward for reaching the terminal after visiting both materials
            else:
                return -1.0  # Penalty for reaching terminal without visiting materials
        return 0.0  # No reward for other transitions

    def reset(self):
        """Reset the visited states and material states for a new episode."""
        self.visited_states = set()
        self.visited_material_states = set()

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
                elif (i, j) == self.material_state1:
                    cell = "M1".center(cell_width)
                elif (i, j) == self.material_state2:
                    cell = "M2".center(cell_width)
                else:
                    cell = ".".center(cell_width)
                row += cell + "|"
            print(row)
            print(horizontal_border)
        print()
