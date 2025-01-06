import random
from .grid_world import GridWorldMDP


class HangarWorldMDP(GridWorldMDP):
    def __init__(
        self, height: int, width: int, number_of_holes: int, number_of_materials: int
    ):
        self.height = height
        self.width = width
        self.number_of_holes = number_of_holes
        self.number_of_materials = number_of_materials

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

        self.material_states = random.sample(
            list(self.states - set(forbidden_states)), number_of_materials
        )

        self.visited_states = set()
        self.visited_material_states = set()

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
        if new_state == self.terminal_state:
            return 2
        elif new_state in self.material_states:
            return 1
        else:
            return 0

    def get_reward(self, key):
        new_state = key[2]
        if (
            new_state in self.material_states
            and new_state not in self.visited_material_states
        ):
            self.visited_material_states.add(new_state)
            self.visited_states.add(new_state)
            return 1.0
        elif (
            new_state == self.terminal_state
            and len(self.visited_material_states) == self.number_of_materials
        ):
            return 2.0
        elif new_state in self.visited_states:
            return -0.1
        else:
            self.visited_states.add(new_state)
            return 0.0

    def reset(self):
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
                elif (i, j) in self.material_states:
                    cell = "M".center(cell_width)
                else:
                    cell = ".".center(cell_width)
                row += cell + "|"
            print(row)
            print(horizontal_border)
        print()
