import sys

from random import randint

import gymnasium as gym

from taxi_puzzle import decode_state, action_descriptions

from astar_search import astar_search

if __name__ == "__main__":
    render = True if len(sys.argv) > 1 and sys.argv[1] == "render" else False

    env = gym.make("Taxi-v3", render_mode="human" if render else None).env

    """
    step() and reset() return a dict with the following keys:
    p - transition proability for the state.
    action_mask - if actions will cause a transition to a new state.
    """

    # continuously simulate different random initial states
    while True:
        seed = randint(0, 499)

        state_n, _ = env.reset(seed=seed)
        initial_state = decode_state(state_n)

        print("-"*30)
        print(f"Solutions found for initial state {initial_state} ({state_n}), seed={seed}:")

        solutions = {}

        solutions["A*"] = astar_search(initial_state)

        # print solutions
        for k in solutions.keys():
            print("%5s" % k, "\t", solutions[k])

        # simulate solutions in gym env for demonstration
        for k in solutions.keys():
            print(f"Simulating {'%3s' % k} solution...", end='', flush=True)
            solution, solution_reward = solutions[k]
            # print(solution)
            env.reset(seed=seed)
            for i, action in enumerate(solution):
                _, reward, done, _, action_mask = env.step(action_descriptions.index(action))

                if done: 
                    print(f" completed in {'%3s' % len(solution)} actions, reward={reward}")
                    break

                if i == len(solution)-1:
                    print("Error: Did not complete solution in simulation.")
                    exit()
