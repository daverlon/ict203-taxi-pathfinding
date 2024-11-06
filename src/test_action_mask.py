import sys

from random import randint

import gymnasium as gym

from taxi_puzzle import decode_state, action_descriptions, TaxiPuzzle

from astar_search import astar_search
from ucs import uniform_cost_search
from dfs import depth_first_search
from bfs import breadth_first_search 

if __name__ == "__main__":
    render = True if len(sys.argv) > 1 and sys.argv[1] == "render" else False

    env = gym.make("Taxi-v3", render_mode="human" if render else None).env

    seed = 10
    state, info = env.reset(seed=seed)

    random_solution = []

    while True:

        # action = env.action_space.sample(info["action_mask"])
        action = randint(0, 5)

        state, reward, done, truncated, info = env.step(action)
        if done: 
            pass
            # break

        random_solution.append(action)

        puzzle = TaxiPuzzle(decode_state(state), None, None, 0, None)

        am1, am2 = info["action_mask"].tolist(), puzzle.generate_action_mask()
        good = am1==am2
        print(decode_state(state), am1, am2, good, "after", action, "(reward):", reward)

        if not good: 
            print()
            break

    print(f"Found solution from {len(random_solution)} random actions:", action)

    env.close()
