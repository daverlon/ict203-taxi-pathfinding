"""
0: Red
1: Green
2: Yellow
3: Blue
"""
state_grid_positions = [
    (0,0),
    (4,0),
    (0,4),
    (3,4)
]

"""
0: Move south (down)
1: Move north (up)
2: Move east (right)
3: Move west (left)
4: Pickup passenger
5: Drop off passenger
"""
action_descriptions = [
    "D",
    "U",
    "R",
    "L",
    "PU",
    "DO"
]

def decode_state(encoded_state: int) -> list:
    """
    Decodes a state int value for Taxi-v3 into a 4-element array
    "((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination"
    https://gymnasium.farama.org/environments/toy_text/taxi/

    Args:
        encoded_state (int): encoded state value.

    Returns:
        list: Decoded state value with structure: [taxi_col, taxi_row, passenger_location, destinatio]
    """
    destination = encoded_state % 4
    encoded_state //= 4
    passenger_location = encoded_state % 5
    encoded_state //= 5
    taxi_col = encoded_state % 5
    encoded_state //= 5
    taxi_row = encoded_state
    return [taxi_col, taxi_row, passenger_location, destination]

def encode_state(decoded_state: list) -> int:
    """
    Encodes a valid 4-element state list into its' original single value format.

    Args:
        decoded_state (list): decoded 4-element state list (see decode_state above)

    Returns:
        int: Encoded state, matching gymnasium.env's encoded state
    """
    # used for debugging/testing and 'completeness'
    return ((decoded_state[1] * 5 + decoded_state[0]) * 5 + decoded_state[2]) * 4 + decoded_state[3]

class TaxiPuzzle:
    """
    Represents a Taxi-v3 environment for search algorithms such as A*.

    This class is able to act as a node in a search tree, where each node is a state of the
    Taxi-v3 environment. It is able to mimmic the behaviour of env.step() independently.

    Attributes:
        state (list): Current state represented by a decoded-4-element list (see decode_state above).
        parent (TaxiPuzzle): TaxiPuzzle parent instance, connected like a linked-list.
        action (str): Action taken to transition to this state (see action_descriptions above).
        path_cost (int): The total path cost to reach this state from the initial state.
        heuristic_function (Callable): Optional heuristic function to affect the path_cost, for h(x) based algorithms like A*.

    Methods:
        __init__(state, parent, action, path_cost, heuristic_function):
            Initialises the TaxiPuzzle instance with provided parameters.
            If a heuristic function is provided, the cost calculated by it is
            added to the path_cost member.
        __repr__():
            String representation for the class, primarily used for debugging.
        __lt__():
            Less-than operator. This class can be less-than compared with other instances
            where the total path_cost is compared. It is used by search algorithms such as
            Uniform-Cost Search, to order a priority queue filled with TaxiPuzzle instances.
        reached_goal():
            Checks if the current state is the goal state.
        generate_action_mask():
            Generates an action mask for the current state. 
            Tested to be 100% consistent with the gym.env's built-in implementation.
        generaet_children():
            Generate child nodes for all actions, with updated path_costs based on the transition.
        find_solution():
            Creates a path of all actions to reach the current state by traversing through parent nodes.
    """

    def __init__(self, state: list, parent, action, path_cost: int, heuristic_function):
        self.state: list = state
        self.parent: TaxiPuzzle = parent
        self.action: str = action

        if parent: self.path_cost = parent.path_cost + path_cost
        else: self.path_cost = path_cost

        self.heuristic_function = heuristic_function
        # if heuristic_function:
        #   f(x) = g(x) + h(x)
        # else:
        #   f(x) = g(x)
        self.evaluation_function = self.path_cost + (self.heuristic_function(self.state) if self.heuristic_function is not None else 0)

    def __repr__(self) -> str:
        return str(self.state)

    # used for comparison within priority queues in some search algorithms
    def __lt__(self, other):
        return self.evaluation_function < other.evaluation_function

    def reached_goal(self) -> bool:
        return self.state[2] == self.state[3] # passenger_location == destination

    def generate_action_mask(self) -> list:
        """
        Generate an action mask for the current state.
        Illegal actions will be set to 0, legal set to 1.
            
        Returns:
            list: Action mask list with legal actions set to 1 at the corresponding index.
        """

        """
        +---------+
        |R: | : :G|
        | : | : : |
        | : : : : |
        | | : | : |
        |Y| : |B: |
        +---------+
        """
        taxi_col, taxi_row, _, _ = self.state

        # only restriction from moving down is if already at the bottom
        D = 1 if taxi_row < 4 else 0
        # only restriction from movnig up is if already at the top
        U = 1 if taxi_row > 0 else 0

        # wall detection (right and left movement)
        R = 1 if taxi_col < 4 and not (
            (taxi_col == 0 and taxi_row > 2) or
            (taxi_col == 1 and taxi_row < 2) or
            (taxi_col == 2 and taxi_row > 2)
        ) else 0
        L = 1 if taxi_col > 0 and not (
            (taxi_col == 1 and taxi_row > 2) or
            (taxi_col == 2 and taxi_row < 2) or
            (taxi_col == 3 and taxi_row > 2)
        ) else 0

        # check if passenger can be picked up (taxi position same as passenger position)
        PU = 1 if self.state[2] != 4 and (
            (taxi_col, taxi_row) == state_grid_positions[self.state[2]]
        ) else 0

        # check if passenger can be dropped off (taxi position is one of the drop-off/pickup positions)
        DO = 1 if self.state[2] == 4 and (
            (taxi_col, taxi_row) in state_grid_positions
        ) else 0

        # return the action mask
        am = [D, U, R, L, PU, DO]
        return am

    def generate_children(self):
        """
        Generate child nodes with updated states and path costs.
        This function creates a child node for every action.
        If an action is illegal, the path_cost transferred to the child node is affected.
        The legality of the action is decided by generate_action_mask()'s outputted action mask.
        The reward/cost rules are defined to match the environment's specifications.
            
        Returns:
            list: All created child nodes.
        """
        children = []
        action_mask = self.generate_action_mask()
        # generate a child node for all legal actions
        # with the (their) corresponding action applied to their state
        for i,action in enumerate(action_mask):

            new_state = self.state.copy()
            desc = action_descriptions[i]

            # reward = -cost
            # cost = -reward
            reward = -1

            # valid action, change state and update rewards
            if action == 1:
                if desc == "D": new_state[1] += 1
                elif desc == "U": new_state[1] -= 1
                elif desc == "R": new_state[0] += 1
                elif desc == "L": new_state[0] -= 1
                elif desc == "PU": new_state[2] = 4
                elif desc == "DO": 
                    new_state[2] = state_grid_positions.index((new_state[0], new_state[1]))
                    reward = 20
            # invalid action, do not change state and update rewards (penalty)
            else:
                if desc == "PU" or desc == "DO":
                    reward = -10

            # add -reward (cost) to the child node
            # the cost is used by priority queues for sorting
            # it is possible to use reward, the sorting would have to be inverted
            # this can be done by changing "-reward" below to "reward" and changing "<" in self.__lt__ to ">"
            children.append(TaxiPuzzle(new_state, self, desc, -reward, self.heuristic_function))
        return children

    # traverse to root parent node and build a solution path
    def find_solution(self):
        """
        Repeatedly traverse through parent nodes and build a solution path.
        Uses the 'action' member for each node and adds them to a list.
            
        Returns:
            list: Actions taken to reach this node, starting at the root node.
            int: Total reward to reach this node (-cost).
        """
        solution = []
        solution.append(self.action)
        path = self
        while path.parent != None:
            path = path.parent
            solution.append(path.action)
        solution = solution[:-1]
        solution.reverse()
        return solution, -self.path_cost
