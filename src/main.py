import sys

from random import randint

import gymnasium as gym

from taxi_puzzle import decode_state, action_descriptions

from astar_search import astar_search
from ucs import uniform_cost_search
from dfs import depth_first_search
from bfs import breadth_first_search

if __name__ == "__main__":
    render = True if len(sys.argv) > 1 and sys.argv[1] == "render" else False

    env = gym.make("Taxi-v3", render_mode="human" if render else None).env

    # continuously simulate different random initial states
    while True:
        seed = randint(0, 499)

        # create the environment's initial state
        state_n, _ = env.reset(seed=seed)
        initial_state = decode_state(state_n)

        print("-"*30)
        print(f"Solutions found for initial state {initial_state} ({state_n}), seed={seed}:")

        # run search algorithms and save their solutions for this state
        solutions = {}

        solutions["A*"] = astar_search(initial_state)
        solutions["UCS"] = uniform_cost_search(initial_state)
        solutions["DFS"] = depth_first_search(initial_state)
        solutions["BFS"] = breadth_first_search(initial_state)

        # print solutions
        for k in solutions.keys():
            print("%5s" % k, "\t", solutions[k])

        # simulate solutions in gym env for demonstration
        for k in solutions.keys():
            print(f"Simulating {'%3s' % k} solution...", end='', flush=True)

            solution_actions, solution_reward = solutions[k]
            env_reward_total = 0

            # reset the environment to its' initial state
            env.reset(seed=seed)

            # simulate the path found by the (current) solution
            for i, action in enumerate(solution_actions):
                # step using this action
                _, reward, done, _, action_mask = env.step(action_descriptions.index(action))

                # log the reward to validate
                env_reward_total += reward

                # if the env decides that the solution is found,
                # break the loop
                if done: 
                    print(f" completed in {'%3s' % len(solution_actions)} actions. reward = {solution_reward}")
                    break

                # if this is the last action, and the goal has not been found
                # there must be some discrepancy with the solution
                if i == len(solution_actions)-1:
                    print("Error: Did not complete solution in simulation.")
                    exit()

            # validate if the reward recorded by the TaxiPuzzle class is 
            # consistent with the env
            if solution_reward != env_reward_total:
                print("Error: Discrepancy between env reward and TaxiPuzzle reward")
                exit()