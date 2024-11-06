from taxi_puzzle import TaxiPuzzle

def depth_first_search_analyse(state):
    stack = list()
    stack.append(TaxiPuzzle(state, None, None, 0, None))
    explored = list()

    expansions = 0
    frontier_sizes = []

    while stack:
        frontier_sizes.append(len(stack))
        node = stack.pop()
        expansions += 1

        if node.reached_goal():
            return node.find_solution(), expansions, frontier_sizes # first solution found, probably not optimal

        if node.state not in explored:
            explored.append(node.state)

        children = node.generate_children()
        for child in children:
            if child.state not in explored:
                stack.append(child)