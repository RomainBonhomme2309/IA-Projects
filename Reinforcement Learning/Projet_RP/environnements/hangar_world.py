import random
import itertools


class HangarWorldMDP:
    def __init__(
        self, height: int, width: int, number_of_holes: int, number_of_materials: int
    ):
        self.height = height
        self.width = width
        self.number_of_holes = number_of_holes
        self.number_of_materials = number_of_materials

        self.states = set(
            (i, j, tuple(combination))
            for i in range(height)
            for j in range(width)
            for combination in itertools.product(
                [True, False], repeat=number_of_materials
            )
        )

        UP = (-1, 0)
        DOWN = (1, 0)
        LEFT = (0, -1)
        RIGHT = (0, 1)

        self.actions = [UP, DOWN, LEFT, RIGHT]

        self.bad_states = random.sample(
            [
                state
                for state in self.states
                if not (state[0] == 0 and state[1] == 0)
                and not (state[0] == height - 1 and state[1] == width - 1)
            ],
            self.number_of_holes,
        )

        self.initial_state = (0, 0, tuple([False] * number_of_materials))
        self.terminal_state = (
            height - 1,
            width - 1,
            tuple([True] * number_of_materials),
        )

        forbidden_states = self.bad_states + [self.initial_state, self.terminal_state]

        self.material_states = random.sample(
            list(self.states - set(forbidden_states)), number_of_materials
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
            if state in self.bad_states or state == self.terminal_state:
                continue
            for action in self.actions:
                new_state = self.take_action(state, action)
                # If the new state is not valid, stay in the same state and malus
                if new_state not in self.states or new_state in self.bad_states:
                    new_state = state
                    self.transition_probabilities[(state, action, new_state)] = 0
                    self.rewards[(state, action, new_state)] = -0.2
                else:
                    self.transition_probabilities[(state, action, new_state)] = 1
                    # If the new state is the terminal state, then bonus
                    if new_state == self.terminal_state:
                        self.rewards[(state, action, new_state)] = 2.0
                    # If the new state is a new material state, then bonus
                    elif self.is_new_material_state(state, new_state):
                        self.rewards[(state, action, new_state)] = 1.0
                    # Default case, no bonus or malus
                    else:
                        self.rewards[(state, action, new_state)] = 0

    def is_new_material_state(self, state, new_state):
        # Recup l'index du false -> true
        # recup i j du material en question
        # si le i j de new state correspond au i j du material c'est TRUE
        # else FALSE

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
                if (i, j) == self.terminal_state[:2]:
                    cell = "T".center(cell_width)
                elif any(state[0] == i and state[1] == j for state in self.bad_states):
                    cell = "X".center(cell_width)
                elif any(
                    state[0] == i and state[1] == j for state in self.material_states
                ):
                    cell = "M".center(cell_width)
                else:
                    cell = ".".center(cell_width)
                row += cell + "|"
            print(row)
            print(horizontal_border)
        print()

    def print_policy(self, policy: dict):
        cell_width = 5  # Augmente la largeur des cellules pour mieux voir plusieurs actions par position
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
                    # Afficher la politique pour chaque combinaison de matériaux collectés
                    actions_for_state = [
                        policy[(i, j, collected)]
                        for collected in itertools.product(
                            [True, False], repeat=self.number_of_materials
                        )
                    ]

                    # Créer une représentation de la politique pour tous les états de la position
                    action_symbols = ""
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
                            action_symbols += (
                                " "  # Fallback pour les actions non définies
                            )
                    # Afficher les symboles des actions pour chaque état possible de la position
                    cell = action_symbols.center(cell_width)

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
