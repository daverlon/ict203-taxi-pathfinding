from taxi_puzzle import TaxiPuzzle

def depth_first_search(state):
    stack = list()
    stack.append(TaxiPuzzle(state, None, None, 0, None))
    explored = list()

    while stack:
        node = stack.pop()

        if node.reached_goal():
            return node.find_solution() # first solution found, probably not optimal

        if node.state not in explored:
            explored.append(node.state)

        children = node.generate_children()
        for child in children:
            if child.state not in explored:
                stack.append(child)