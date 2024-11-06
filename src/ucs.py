from queue import PriorityQueue
from taxi_puzzle import TaxiPuzzle

def uniform_cost_search(state):
    frontier = PriorityQueue()
    frontier.put(TaxiPuzzle(state, None, None, 0, None))
    explored = list()

    while not frontier.empty():
        node = frontier.get()

        if node.reached_goal():
            return node.find_solution()

        if node.state not in explored:
            explored.append(node.state)

        children = node.generate_children()
        for child in children:
            if child.state not in explored:
                frontier.put(child)


