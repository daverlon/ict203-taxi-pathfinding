from queue import PriorityQueue
from taxi_puzzle import TaxiPuzzle

def uniform_cost_search_analyse(state):
    frontier = PriorityQueue()
    frontier.put(TaxiPuzzle(state, None, None, 0, None))
    explored = list()
    expansions = 0
    frontier_sizes = []

    while not frontier.empty():
        frontier_sizes.append(frontier.qsize())
        node = frontier.get()
        expansions += 1

        if node.reached_goal():
            return node.find_solution(), expansions, frontier_sizes

        if node.state not in explored:
            explored.append(node.state)

        children = node.generate_children()
        for child in children:
            if child.state not in explored:
                frontier.put(child)


