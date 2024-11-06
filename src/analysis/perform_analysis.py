import sys

from statistics import mean

from math import log

from datetime import datetime

from random import randint

import gymnasium as gym

from taxi_puzzle import decode_state

from astar_search import astar_search_analyse
from ucs import uniform_cost_search_analyse
from dfs import depth_first_search_analyse
from bfs import breadth_first_search_analyse

import matplotlib.pyplot as plt

def calculate_mean_frontier_sizes(all_frontier_sizes):
    max_steps = max(len(sizes) for sizes in all_frontier_sizes)

    mean_sizes = [] # mean size at each step

    # for every step
    # calculate the mean frontier size across all tests for that step
    for step in range(max_steps):
        step_sizes = []

        for sizes in all_frontier_sizes:
            if step < len(sizes): 
               step_sizes.append(sizes[step])

        mean_sizes.append(mean(step_sizes))

    return mean_sizes
 
if __name__ == "__main__":
    render = True if len(sys.argv) > 1 and sys.argv[1] == "render" else False

    env = gym.make("Taxi-v3", render_mode="human" if render else None).env

    astar_times = []
    ucs_times = []
    dfs_times = []
    bfs_times = []

    astar_rewards = []
    ucs_rewards = []
    dfs_rewards = []
    bfs_rewards = []

    astar_expansions = []
    ucs_expansions = []
    dfs_expansions = []
    bfs_expansions = []

    astar_all_frontier_sizes = []
    ucs_all_frontier_sizes = []
    dfs_all_frontier_sizes = []
    bfs_all_frontier_sizes = []

    for i in range(499):
        seed = randint(0, 499)

        state_n, _ = env.reset(seed=seed)
        initial_state = decode_state(state_n)

        solutions = {}

        start_time = datetime.now()
        solutions["A*"], astar_expansion, astar_frontier_sizes = astar_search_analyse(initial_state)
        end_time = datetime.now()
        astar_times.append((end_time-start_time).total_seconds() * 1000.0)
        astar_rewards.append(solutions["A*"][1])
        astar_expansions.append(astar_expansion)
        astar_all_frontier_sizes.append(astar_frontier_sizes)

        start_time = datetime.now()
        solutions["UCS"], ucs_expansion, ucs_frontier_sizes = uniform_cost_search_analyse(initial_state)
        end_time = datetime.now()
        ucs_times.append((end_time-start_time).total_seconds() * 1000.0)
        ucs_rewards.append(solutions["UCS"][1])
        ucs_expansions.append(ucs_expansion)
        ucs_all_frontier_sizes.append(ucs_frontier_sizes)

        start_time = datetime.now()
        solutions["DFS"], dfs_expansion, dfs_frontier_sizes = depth_first_search_analyse(initial_state)
        end_time = datetime.now()
        dfs_times.append((end_time-start_time).total_seconds() * 1000.0)
        dfs_rewards.append(solutions["DFS"][1])
        dfs_expansions.append(dfs_expansion)
        dfs_all_frontier_sizes.append(dfs_frontier_sizes)

        start_time = datetime.now()
        solutions["BFS"], bfs_expansion, bfs_frontier_sizes = breadth_first_search_analyse(initial_state)
        end_time = datetime.now()
        bfs_times.append((end_time-start_time).total_seconds() * 1000.0)
        bfs_rewards.append(solutions["BFS"][1])
        bfs_expansions.append(bfs_expansion)
        bfs_all_frontier_sizes.append(bfs_frontier_sizes)

        print(f"[{i}] Found solutions for initial state {initial_state} ({state_n}), seed={seed}")


    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    axs[0, 0].plot(astar_times, marker='o', linestyle='-', color='b')
    axs[0, 0].axhline(y=mean(astar_times), color='black', linestyle='--', label=f'Mean: {mean(astar_times):.2f}')
    axs[0, 0].set_title("A* Search Times")
    axs[0, 0].set_xlabel("Test Run")
    axs[0, 0].set_ylabel("Time (ms)")
    axs[0, 0].legend()

    axs[0, 1].plot(ucs_times, marker='o', linestyle='-', color='r')
    axs[0, 1].axhline(y=mean(ucs_times), color='black', linestyle='--', label=f'Mean: {mean(ucs_times):.2f}')
    axs[0, 1].set_title("UCS Search Times")
    axs[0, 1].set_xlabel("Test Run")
    axs[0, 1].set_ylabel("Time (ms)")
    axs[0, 1].legend()

    axs[1, 0].plot(dfs_times, marker='o', linestyle='-', color='g')
    axs[1, 0].axhline(y=mean(dfs_times), color='black', linestyle='--', label=f'Mean: {mean(dfs_times):.2f}')
    axs[1, 0].set_title("DFS Search Times")
    axs[1, 0].set_xlabel("Test Run")
    axs[1, 0].set_ylabel("Time (ms)")
    axs[1, 0].legend()

    axs[1, 1].plot(bfs_times, marker='o', linestyle='-', color='orange')
    axs[1, 1].axhline(y=mean(bfs_times), color='black', linestyle='--', label=f'Mean: {mean(bfs_times):.2f}')
    axs[1, 1].set_title("BFS Search Times")
    axs[1, 1].set_xlabel("Test Run")
    axs[1, 1].set_ylabel("Time (ms)")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()
    plt.close()

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    axs[0, 0].plot(astar_rewards, marker='o', linestyle='-', color='b')
    axs[0, 0].axhline(y=mean(astar_rewards), color='black', linestyle='--', label=f'Mean: {mean(astar_rewards):.2f}')
    axs[0, 0].set_title("A* Search Rewards")
    axs[0, 0].set_xlabel("Test Run")
    axs[0, 0].set_ylabel("Reward")
    axs[0, 0].legend()

    axs[0, 1].plot(ucs_rewards, marker='o', linestyle='-', color='r')
    axs[0, 1].axhline(y=mean(ucs_rewards), color='black', linestyle='--', label=f'Mean: {mean(ucs_rewards):.2f}')
    axs[0, 1].set_title("UCS Search Rewards")
    axs[0, 1].set_xlabel("Test Run")
    axs[0, 1].set_ylabel("Reward")
    axs[0, 1].legend()

    axs[1, 0].plot(dfs_rewards, marker='o', linestyle='-', color='g')
    axs[1, 0].axhline(y=mean(dfs_rewards), color='black', linestyle='--', label=f'Mean: {mean(dfs_rewards):.2f}')
    axs[1, 0].set_title("DFS Search Rewards")
    axs[1, 0].set_xlabel("Test Run")
    axs[1, 0].set_ylabel("Reward")
    axs[1, 0].legend()

    axs[1, 1].plot(bfs_rewards, marker='o', linestyle='-', color='orange')
    axs[1, 1].axhline(y=mean(bfs_rewards), color='black', linestyle='--', label=f'Mean: {mean(bfs_rewards):.2f}')
    axs[1, 1].set_title("BFS Search Rewards")
    axs[1, 1].set_xlabel("Test Run")
    axs[1, 1].set_ylabel("Reward")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()
    plt.close()

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    axs[0, 0].plot(astar_expansions, marker='o', linestyle='-', color='b')
    axs[0, 0].axhline(y=mean(astar_expansions), color='black', linestyle='--', label=f'Mean: {mean(astar_expansions):.2f}')
    axs[0, 0].set_title("A* Search Expansions")
    axs[0, 0].set_xlabel("Test Run")
    axs[0, 0].set_ylabel("Expansions")
    axs[0, 0].legend()

    axs[0, 1].plot(ucs_expansions, marker='o', linestyle='-', color='r')
    axs[0, 1].axhline(y=mean(ucs_expansions), color='black', linestyle='--', label=f'Mean: {mean(ucs_expansions):.2f}')
    axs[0, 1].set_title("UCS Search Expansions")
    axs[0, 1].set_xlabel("Test Run")
    axs[0, 1].set_ylabel("Expansions")
    axs[0, 1].legend()

    axs[1, 0].plot(dfs_expansions, marker='o', linestyle='-', color='g')
    axs[1, 0].axhline(y=mean(dfs_expansions), color='black', linestyle='--', label=f'Mean: {mean(dfs_expansions):.2f}')
    axs[1, 0].set_title("DFS Search Expansions")
    axs[1, 0].set_xlabel("Test Run")
    axs[1, 0].set_ylabel("Expansions")
    axs[1, 0].legend()

    axs[1, 1].plot(bfs_expansions, marker='o', linestyle='-', color='orange')
    axs[1, 1].axhline(y=mean(bfs_expansions), color='black', linestyle='--', label=f'Mean: {mean(bfs_expansions):.2f}')
    axs[1, 1].set_title("BFS Search Expansions")
    axs[1, 1].set_xlabel("Test Run")
    axs[1, 1].set_ylabel("Expansions")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()
    plt.close()

    astar_mean_sizes = calculate_mean_frontier_sizes(astar_all_frontier_sizes)
    ucs_mean_sizes = calculate_mean_frontier_sizes(ucs_all_frontier_sizes)
    dfs_mean_sizes = calculate_mean_frontier_sizes(dfs_all_frontier_sizes)
    bfs_mean_sizes = calculate_mean_frontier_sizes(bfs_all_frontier_sizes)

    def plot_mean_frontier_sizes(astar_mean_sizes, ucs_mean_sizes, dfs_mean_sizes, bfs_mean_sizes):

        plt.figure(figsize=(12, 8))

        plt.plot(astar_mean_sizes, linestyle='-', color='b', label='A*')
        plt.plot(ucs_mean_sizes, linestyle='-', color='r', label='UCS')
        plt.plot(dfs_mean_sizes, linestyle='-', color='g', label='DFS')
        plt.plot(bfs_mean_sizes, linestyle='-', color='orange', label='BFS')

        plt.title('Mean Frontier Sizes Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Mean Frontier Size')
        plt.xscale('log')
        # plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    print("Astar mean final size:", astar_mean_sizes[-1])
    print("UCS mean final size:", ucs_mean_sizes[-1])
    print("DFS mean final size:", dfs_mean_sizes[-1])
    print("BFS mean final size:", bfs_mean_sizes[-1])


    plot_mean_frontier_sizes(astar_mean_sizes, ucs_mean_sizes, dfs_mean_sizes, bfs_mean_sizes)

    plt.close()
