import random


class GarageWorldMDP:
    def __init__(
        self,
        height: int,
        width: int,
        number_of_holes: int,
        number_of_recharging_stations: int,
    ):
        self.height = height
        self.width = width
        self.number_of_holes = number_of_holes
        self.number_of_recharging_stations = number_of_recharging_stations

        # Create all possible states (position + recharging_stations)
        self.states = set((i, j) for i in range(height) for j in range(width))

        # Define the actions
        UP = (-1, 0)
        DOWN = (1, 0)
        LEFT = (0, -1)
        RIGHT = (0, 1)

        self.actions = [UP, DOWN, LEFT, RIGHT]

        # Define the bad states (holes)
        self.bad_states = random.sample(
            [
                (i, j)
                for i in range(height)
                for j in range(width)
                if (i, j) != (height - 1, 0)
            ],
            self.number_of_holes,
        )

        # Define the initial state
        self.initial_state = (height - 1, 0)

        # Define the recharging_station states (terminal states)
        forbidden_states = self.bad_states + [self.initial_state]

        available_states = [
            x
            for x in self.states
            if x not in forbidden_states
            and x[:2] not in [fs[:2] for fs in forbidden_states]
        ]

        unique_available_states = list({x[:2]: x for x in available_states}.keys())

        self.terminal_states = [
            x[:2]
            for x in random.sample(
                unique_available_states, number_of_recharging_stations
            )
        ]

        # Define the transition probabilities and rewards
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
            if state in self.bad_states or state in self.terminal_states:
                continue
            for action in self.actions:
                new_state = self.take_action(state, action)
                # If the new state is not valid, stay in the same state and malus
                if new_state not in self.states or new_state in self.bad_states:
                    new_state = state
                    self.transition_probabilities[(state, action, new_state)] = 0
                    self.rewards[(state, action, new_state)] = -1.0
                else:
                    self.transition_probabilities[(state, action, new_state)] = 1
                    # If the new state is a terminal state, then bonus
                    if new_state in self.terminal_states:
                        self.rewards[(state, action, new_state)] = 1.0
                    # Default case, no bonus or malus
                    else:
                        self.rewards[(state, action, new_state)] = 0

    def take_action(self, state, action):
        new_state = (state[0] + action[0], state[1] + action[1])
        if new_state not in self.states or new_state in self.bad_states:
            new_state = state

        return new_state

    def print_board(self):
        cell_width = 3
        horizontal_border = "+" + ("-" * cell_width + "+") * self.width

        print(horizontal_border)
        for i in range(self.height):
            row = "|"
            for j in range(self.width):
                if (i, j) == self.initial_state:
                    cell = "S".center(cell_width)
                elif any(state[0] == i and state[1] == j for state in self.bad_states):
                    cell = "X".center(cell_width)
                elif (i, j) in self.terminal_states:
                    for k in range(len(self.terminal_states)):
                        if (i, j) == self.terminal_states[k]:
                            cell = f"T{k}".center(cell_width)
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
                if (i, j) in self.terminal_states:
                    for k in range(len(self.terminal_states)):
                        if (i, j) == self.terminal_states[k]:
                            cell = f"T{k}".center(cell_width)
                elif (i, j) in self.bad_states:
                    cell = "X".center(cell_width)
                else:
                    prefixe = ""
                    if (i, j) == self.initial_state:
                        prefixe = "S"

                    action = policy[(i, j)]
                    # Use arrows to represent actions
                    if action == (1, 0):
                        cell = (prefixe + "↓").center(cell_width)
                    elif action == (-1, 0):
                        cell = (prefixe + "↑").center(cell_width)
                    elif action == (0, 1):
                        cell = (prefixe + "→").center(cell_width)
                    elif action == (0, -1):
                        cell = (prefixe + "←").center(cell_width)
                    else:
                        cell = " ".center(cell_width)  # Fallback for undefined actions
                row += cell + "|"
            print(row)
            print(horizontal_border)
        print()
