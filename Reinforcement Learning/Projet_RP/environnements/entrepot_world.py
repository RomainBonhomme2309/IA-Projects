import random
import itertools


class EntrepotWorldMDP:
    def __init__(
        self,
        height: int,
        width: int,
        number_of_holes: int,
        number_of_uncleaned_areas: int,
    ):
        self.height = height
        self.width = width
        self.number_of_holes = number_of_holes
        self.number_of_uncleaned_areas = number_of_uncleaned_areas

        # Create all possible states (position + uncleaned_areas)
        self.states = set(
            (i, j, tuple(combination))
            for i in range(height)
            for j in range(width)
            for combination in itertools.product(
                [True, False], repeat=number_of_uncleaned_areas
            )
        )

        # Define the actions
        UP = (-1, 0)
        DOWN = (1, 0)
        LEFT = (0, -1)
        RIGHT = (0, 1)

        self.actions = [UP, DOWN, LEFT, RIGHT]

        # Define the bad states (holes)
        bad_states_pos = random.sample(
            [
                (i, j)
                for i in range(height)
                for j in range(width)
                if (i, j) != (0, 0) and (i, j) != (0, width - 1)
            ],
            self.number_of_holes,
        )

        self.bad_states = [
            (i, j, tpl)
            for i, j in bad_states_pos
            for tpl in itertools.product(
                [True, False], repeat=number_of_uncleaned_areas
            )
        ]

        # Define the initial and terminal states
        self.initial_state = (0, 0, tuple([False] * number_of_uncleaned_areas))
        self.terminal_state = (
            0,
            width - 1,
            tuple([True] * number_of_uncleaned_areas),
        )

        # Define the uncleaned_area states
        forbidden_states = self.bad_states + [self.initial_state, self.terminal_state]

        available_states = [
            x
            for x in self.states
            if x not in forbidden_states
            and x[:2] not in [fs[:2] for fs in forbidden_states]
        ]

        unique_available_states = list({x[:2]: x for x in available_states}.keys())

        self.uncleaned_area_states = [
            x[:2]
            for x in random.sample(unique_available_states, number_of_uncleaned_areas)
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
            if state in self.bad_states or state == self.terminal_state:
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
                    # If the new state is the terminal state (all uncleaned areas cleaned), then bonus
                    if new_state == self.terminal_state:
                        self.rewards[(state, action, new_state)] = 3.0
                    # If the new state is a new uncleaned area state, then bonus
                    elif self.is_new_uncleaned_area_state(state, new_state):
                        self.rewards[(state, action, new_state)] = 2.0
                    # If the agent goes to the terminal state without all the uncleaned_areas cleaned, then malus
                    elif new_state[:2] == self.terminal_state[:2]:
                        self.rewards[(state, action, new_state)] = -1.0
                    # Default case, no bonus or malus
                    else:
                        self.rewards[(state, action, new_state)] = 0

    def is_new_uncleaned_area_state(self, state, new_state):
        for i in range(len(state[2])):
            if not state[2][i] and new_state[2][i]:
                uncleaned_area_pos = self.uncleaned_area_states[i]
                if new_state[:2] == uncleaned_area_pos:
                    return True
        return False

    def take_action(self, state, action):
        new_state_pos = (state[0] + action[0], state[1] + action[1])
        new_state = (new_state_pos[0], new_state_pos[1], state[2])
        for i in range(len(self.uncleaned_area_states)):
            if new_state_pos == self.uncleaned_area_states[i]:
                new_state_uncleaned_areas = list(state[2])
                new_state_uncleaned_areas[i] = True
                new_state = (
                    new_state_pos[0],
                    new_state_pos[1],
                    tuple(new_state_uncleaned_areas),
                )
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
                if (i, j) == self.initial_state[:2]:
                    cell = "S".center(cell_width)
                elif (i, j) == self.terminal_state[:2]:
                    cell = "T".center(cell_width)
                elif any(state[0] == i and state[1] == j for state in self.bad_states):
                    cell = "X".center(cell_width)
                elif (i, j) in self.uncleaned_area_states:
                    for k in range(len(self.uncleaned_area_states)):
                        if (i, j) == self.uncleaned_area_states[k]:
                            cell = f"U{k}".center(cell_width)
                else:
                    cell = ".".center(cell_width)
                row += cell + "|"
            print(row)
            print(horizontal_border)
        print()

    def print_policy(self, policy: dict):
        iter_list = list(
            itertools.product([True, False], repeat=self.number_of_uncleaned_areas)
        )
        reversed_iter_list = [tuple(reversed(item)) for item in reversed(iter_list)]
        print("Arrow order (U0, U1, ...):", reversed_iter_list)

        cell_width = 3 * self.number_of_uncleaned_areas
        horizontal_border = "+" + ("-" * cell_width + "+") * self.width

        print(horizontal_border)
        for i in range(self.height):
            row = "|"
            for j in range(self.width):
                if any(state[0] == i and state[1] == j for state in self.bad_states):
                    cell = "X".center(cell_width)
                else:
                    actions_for_state = [
                        policy[(i, j, collected)] for collected in reversed_iter_list
                    ]

                    action_symbols = ""
                    if (i, j) == self.initial_state[:2]:
                        action_symbols += "S"
                    if (i, j) == self.terminal_state[:2]:
                        action_symbols += "T"
                    for action in actions_for_state:
                        if action == (1, 0):
                            action_symbols += "↓"
                        elif action == (-1, 0):
                            action_symbols += "↑"
                        elif action == (0, 1):
                            action_symbols += "→"
                        elif action == (0, -1):
                            action_symbols += "←"
                        else:
                            action_symbols += " "
                    cell = action_symbols.center(cell_width)

                row += cell + "|"
            print(row)
            print(horizontal_border)
        print()
