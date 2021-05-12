import argparse
import os
import sys
import re
import time
import pickle
from collections import OrderedDict

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt; plt.ion()
from icecream import ic

from Planner import MyPlanner
import utils

MAPDICT = {file.split('.')[0]: os.path.join('./maps/', file) for file in os.listdir('./maps/')}
MAP_SE = OrderedDict(
    {
        'single_cube': (np.array([2.3, 2.3, 1.3]), np.array([7.0, 7.0, 5.5])),
        'monza': (np.array([0.5, 1.0, 4.9]), np.array([3.8, 1.0, 0.1])),
        'flappy_bird': (np.array([0.5, 2.5, 5.5]), np.array([19.0, 2.5, 5.5])),
        'window': (np.array([0.2, -4.9, 0.2]), np.array([6.0, 18.0, 3.0])),
        'room': (np.array([1.0, 5.0, 1.5]), np.array([9.0, 7.0, 1.5])),
        'maze': (np.array([0.0, 0.0, 1.0]), np.array([12.0, 12.0, 5.0])),
        'tower': (np.array([2.5, 4.0, 0.5]), np.array([4.0, 2.5, 19.5])),
    }
)


def run_test(map_file, start, goal, render=True, verbose=True):
    """
    This function:
        * load the provided map_file
        * creates a motion planner
        * plans a path from start to goal
        * checks whether the path is collision free and reaches the goal
        * computes the path length as a sum of the Euclidean norm of the path segments
    """
    # Load a map and instantiate a motion planner
    boundary, blocks = utils.load_map(map_file)
    # TODO: replace this with your own planner implementation
    MP = MyPlanner(boundary, blocks)

    # Init collision obj
    MP.init_collision_objects(blocks, start, goal)

    # Display the environment
    if render:
        fig, ax, hb, hs, hg = utils.draw_map(boundary, blocks, start, goal)

    # Call the motion planner
    t0 = utils.tic()
    path = MP.greedy_plan(start, goal)
    test_name = re.split('[/.]', map_file)[3]
    utils.toc(t0, f"Planning {test_name}")

    # Plot the path
    if render:
        try:
            ax.plot(path[:, 0], path[:, 1], path[:, 2], 'r-')
        except KeyboardInterrupt:
            sys.exit(0)

    # TODO: You should verify whether the path actually intersects any of the obstacles in continuous space
    # TODO: You can implement your own algorithm or use an existing library for segment and
    #       axis-aligned bounding box (AABB) intersection
    collision = False
    goal_reached = sum((path[-1] - goal) ** 2) <= 0.1
    success = (not collision) and goal_reached
    pathlength = np.sum(np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1)))
    return success, pathlength


def test_env(env_id: str, res: float = 0.1, render=True, verbose=True):
    assert env_id in MAPDICT.keys(), "ENV Not Found!"
    map_file = MAPDICT[env_id]
    # start pos and goal pos
    start, goal = MAP_SE[env_id]
    boundary, blocks = utils.load_map(map_file)
    ic(boundary)
    grid_world, grid_start, grid_goal = utils.make_grid_env(map_file, start, goal, res=0.1)
    ic(grid_start)
    ic(grid_goal)

    # TODO: replace this with your own planner implementation
    MP = MyPlanner(boundary, blocks)

    # Init collision obj
    MP.init_collision_objects(blocks, start, goal)

    # Display the environment
    if render:
        fig, ax, hb, hs, hg = utils.draw_map(boundary, blocks, start, goal)

    # Call the motion planner
    t0 = utils.tic()
    path = MP.greedy_plan(start, goal)
    test_name = re.split('[/.]', map_file)[3]
    utils.toc(t0, f"Planning {test_name}")

    # print(f'Running {env_id} test...\n')
    # success, pathlength = run_test(map_file, start, goal, verbose)
    # print('Success: %r' % success)
    # print('Path length: %d' % pathlength)
    # print('\n')
    if verbose:
        plt.show(block=True)


def calc_cost(grid_path, cost_grid, res):
    cost = 0.0
    for i in range(grid_path.shape[0]):
        state = tuple(grid_path[i, :])
        cost += cost_grid[state]
    return cost * res


def single_test(env_id: str, rad, eps=2, res=0.1, distType: int = 2, verbose=True, render=False):
    """Perform a single test on single Env"""
    print(f"\n###### {env_id}_{distType} ######")
    single_test_result = {}

    # Load map
    map_file = MAPDICT[env_id]
    start, goal = MAP_SE[env_id]
    boundary, blocks = utils.load_map(map_file)

    # Find Shortest Path
    MP = MyPlanner(boundary, blocks)
    t1 = time.time()
    path, grid_path, cost_grid, max_node = MP.A_star(
        start,
        goal,
        rad=rad,
        eps=eps,
        res=res,
        distType=distType)
    utils.toc(t_start=t1, name=env_id)

    # Evaluation
    total_cost = calc_cost(grid_path, cost_grid, res)
    # TODO: modify collision
    collision = False
    goal_reached = sum((path[-1] - goal) ** 2) <= 0.1
    success = (not collision) and goal_reached
    pathlength = np.sum(np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1)))
    single_test_result['total_cost'] = total_cost
    single_test_result['max_node'] = max_node
    single_test_result['success'] = success
    single_test_result['pathlength'] = pathlength
    if verbose:
        ic(total_cost)
        ic(max_node)
        ic(success)
        ic(pathlength)
    print("###################################")

    if render:
        # Original Plot
        fig, ax, hb, hs, hg = utils.draw_map(boundary, blocks, start, goal)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], 'r-')
        plt.title(f"Original {env_id} Env")
        plt.show(block=True)
        plt.close('all')

        # # Grid Plot
        # grid_env = utils.make_grid_env(boundary, blocks, start, goal, res=res)
        # _, grid_boundary, grid_block, grid_start, grid_goal = grid_env
        # grid_fig, grid_ax, grid_hb, grid_hs, grid_hg = utils.draw_map(grid_boundary, grid_block, grid_start, grid_goal)
        # grid_ax.plot(grid_path[:, 0], grid_path[:, 1], grid_path[:, 2], 'r-')
        # plt.title(f"Grid {env_id} Env")
        # plt.show(block=True)

    return single_test_result


def multi_test_single_env(env_id: str, param:dict, verbose=True, render=False):
    info = {}
    # for distType in range(1, 5):
    for distType in range(2, 3):
        print(f"\n###### {distType} ######")
        single_result = single_test(
            env_id,
            rad=param['rad'],
            eps=param['eps'],
            res=param['res'],
            distType=distType,
            verbose=verbose, render=render)
        for k, v in single_result.items():
            info[f"{env_id}_{distType}_{k}"] = v
    return info


def test_all_show_last():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--render", help="Visualize Env", action="store_true", default=True)
    parser.add_argument("-res", help="Resolution", type=float, default=0.1)
    parser.add_argument("--seed", help="Random generator seed", type=int, default=42)
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument(
        "-verb", "--verbose", action="store_true", default=True,
        help="Verbose mode (False: no output, True: INFO)"
    )
    args = parser.parse_args()
    seed = args.seed
    param = {
        'eps': 2,
        'res': 0.1,
        'rad': 0.1,
    }
    lst = []
    info = {}
    single = {}
    for env_id in tqdm(MAP_SE.keys()):
        # Load map
        map_file = MAPDICT[env_id]
        start, goal = MAP_SE[env_id]
        boundary, blocks = utils.load_map(map_file)

        # Find Shortest Path
        MP = MyPlanner(boundary, blocks)
        t1 = time.time()
        path, grid_path, cost_grid, max_node = MP.A_star(
            start,
            goal,
            rad=param['rad'],
            eps=param['eps'],
            res=param['res'],
            distType=2
        )
        utils.toc(t_start=t1, name=env_id)

        # Evaluation
        total_cost = calc_cost(grid_path, cost_grid, param['res'])
        # TODO: modify collision
        collision = False
        goal_reached = sum((path[-1] - goal) ** 2) <= 0.1
        success = (not collision) and goal_reached
        pathlength = np.sum(np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1)))
        single[total_cost] = total_cost
        single[max_node] = max_node
        single[success] = success
        single[pathlength] = pathlength
        ic(total_cost)
        ic(max_node)
        ic(success)
        ic(pathlength)
        print("###################################")
        info[env_id] = single
        # Original Plot
        fig, ax, hb, hs, hg = utils.draw_map(boundary, blocks, start, goal)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], 'r-')
        plt.title(f"Original {env_id} Env")
        lst.append((fig, ax, hb, hs, hg))
    for p in lst:
        f, a, b, s, g = p
        plt.show(block=True)

    return info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--render", help="Visualize Env", action="store_true", default=True)
    parser.add_argument("-res", help="Resolution", type=float, default=0.1)
    parser.add_argument("--seed", help="Random generator seed", type=int, default=42)
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument(
        "-verb", "--verbose", action="store_true", default=True,
        help="Verbose mode (False: no output, True: INFO)"
    )
    args = parser.parse_args()
    seed = args.seed
    print(np.__version__)
    # np.seterr(all='raise')
    param = {
        'eps': 2,
        'res': 0.1,
        'rad': 0.1,
    }

    # info_lst = []
    # for env_id in tqdm(MAP_SE.keys()):
    #     print(f"\n###### {env_id} ######")
    #     info = multi_test_single_env(env_id=env_id, param=param, verbose=True, render=True)
    #     info_lst.append(info)
    # if args.save:
    #     with open("A_star_res.pkl", 'wb') as f:
    #         pickle.dump(info_lst, f)
    #     # del info
    #     with open("A_star_res.pkl", 'rb') as f:
    #         info_lst = pickle.load(f)
    info = test_all_show_last()

