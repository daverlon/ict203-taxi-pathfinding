from queue import Queue
from taxi_puzzle import TaxiPuzzle

def breadth_first_search_analyse(state):
    q = Queue()
    q.put(TaxiPuzzle(state, None, None, 0, None))
    explored = list()

    expansions = 0
    frontier_sizes = []

    while True:
        frontier_sizes.append(q.qsize())
        node = q.get()
        expansions += 1

        if node.reached_goal():
            return node.find_solution(), expansions, frontier_sizes

        if node.state not in explored:
            explored.append(node.state)

        children = node.generate_children()
        for child in children:
            if child.state not in explored:
                q.put(child)