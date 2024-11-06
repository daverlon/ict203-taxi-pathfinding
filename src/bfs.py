from queue import Queue
from taxi_puzzle import TaxiPuzzle

def breadth_first_search(state):
    q = Queue()
    q.put(TaxiPuzzle(state, None, None, 0, None))
    explored = list()

    while True:
        node = q.get()

        if node.reached_goal():
            return node.find_solution()

        if node.state not in explored:
            explored.append(node.state)

        children = node.generate_children()
        for child in children:
            if child.state not in explored:
                q.put(child)