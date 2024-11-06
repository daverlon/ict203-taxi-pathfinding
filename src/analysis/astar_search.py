from queue import PriorityQueue
from taxi_puzzle import TaxiPuzzle, state_grid_positions

def heuristic(state: list) -> int:
    # heuristic function (used by A*)
    # if the passenger is not in the taxi,
    # manhattan search to the passenger location
    # if the passenger is in the taxi
    # manhattan search to the destination
    # else: no other 'goal states' exist, therefore goal has been reached

    x1, y1 = state[0], state[1]
    if state[2] != 4 and state[2] != state[3]:
        # go to passenger location
        x2, y2 = state_grid_positions[state[2]][0], state_grid_positions[state[2]][1]
        return abs(x1 - x2) + abs(y1 - y2)
    elif state[2] == 4:
        # go to destination
        x2, y2 = state_grid_positions[state[3]][0], state_grid_positions[state[3]][1]
        return abs(x1 - x2) + abs(y1 - y2)
    else:
        return 0 # found goal, cost is 0

def astar_search_analyse(state):
    count = 0
    q = PriorityQueue()
    q.put((TaxiPuzzle(state, None, None, 0, heuristic), count))
    explored = list()

    expansions = 0
    frontier_sizes = []

    while not q.empty():
        frontier_sizes.append(q.qsize())
        node, _ = q.get()
        expansions += 1

        if node.reached_goal():
            return node.find_solution(), expansions, frontier_sizes

        if node.state not in explored:
            explored.append(node.state)

        children = node.generate_children()
        for child in children:
            if child.state not in explored:
                q.put((child, count))
                count += 1