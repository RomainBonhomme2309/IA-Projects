import random
from .grid_world import GridWorldMDP


class GarageWorldMDP(GridWorldMDP):
    def __init__(self, height: int, width: int, number_of_holes: int, number_of_recharge_station: int):
        self.height = height
        self.width = width
        self.number_of_holes = number_of_holes
        self.number_of_recharge_station = number_of_recharge_station

        self.states = set((i, j) for i in range(height) for j in range(width))

        UP = (-1, 0)
        DOWN = (1, 0)
        LEFT = (0, -1)
        RIGHT = (0, 1)

        self.actions = [UP, DOWN, LEFT, RIGHT]

        self.bad_states = random.sample(
            list(self.states - {(height - 1, 0)}), self.number_of_holes
        )

        self.initial_state = (0, 0)

        forbidden_states = self.bad_states + [self.initial_state]

        self.recharge_states = random.sample(
            list(self.states - set(forbidden_states)), number_of_recharge_station
        )

        self.transition_probabilities = {
            (state, action, new_state): 0
            for state in self.states
            for action in self.actions
            for new_state in self.states
        }  # a dictionary of type "{(s,a,s') : probability of the transition(s,a,s')}"
        self.rewards = {
            (state, action, new_state): 0
            for state in self.states
            for action in self.actions
            for new_state in self.states
        }  # a dictionary of type "{(s,a,s') : reward of the transition(s,a,s')}"

        for state in self.states:
            if state in self.bad_states:
                continue
            for action in self.actions:
                new_state = (state[0] + action[0], state[1] + action[1])
                if new_state not in self.states or new_state in self.bad_states:
                    new_state = state
                self.transition_probabilities[(state, action, new_state)] = 1
                if new_state in self.recharge_states:
                    self.rewards[(state, action, new_state)] = 1
                else:
                    self.rewards[(state, action, new_state)] = 0

    def print_board(self):
        cell_width = 3
        horizontal_border = "+" + ("-" * cell_width + "+") * self.width

        print(horizontal_border)
        for i in range(self.height):
            row = "|"
            for j in range(self.width):
                if (i, j) in self.bad_states:
                    cell = "X".center(cell_width)
                elif (i, j) in self.recharge_states:
                    cell = "R".center(cell_width)
                else:
                    cell = ".".center(cell_width)
                row += cell + "|"
            print(row)
            print(horizontal_border)
        print()
