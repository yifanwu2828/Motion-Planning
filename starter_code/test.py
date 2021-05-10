import argparse
import os
import sys
import re
import time
from collections import OrderedDict

import numpy as np
from pqdict import PQDict
import matplotlib.pyplot as plt;plt.ion()
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
    path = MP.plan(start, goal)
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
    path = MP.plan(start, goal)
    test_name = re.split('[/.]', map_file)[3]
    utils.toc(t0, f"Planning {test_name}")


    # print(f'Running {env_id} test...\n')
    # success, pathlength = run_test(map_file, start, goal, verbose)
    # print('Success: %r' % success)
    # print('Path length: %d' % pathlength)
    # print('\n')
    if verbose:
        plt.show(block=True)


def child_of(node):
    [dX, dY, dZ] = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])
    dR = np.vstack((dX.flatten(), dY.flatten(), dZ.flatten()))
    # Remove (0,0,0)
    dR = np.delete(dR, 13, axis=1)
    children = node.reshape(3, -1) + dR 
    return children.T


def is_inbound(node, world):
    if node.ndim > 1:
        node.reshape(-1)
    cond = (
        0 <= node[0] < world.shape[0],
        0 <= node[1] < world.shape[1],
        0 <= node[2] < world.shape[2]
    )
    return all(cond)


def cost(i, j):
    """Define the cost form node i to node j as Euclidean Distance"""
    assert i.ndim == j.ndim and i.size == j.size, "Ensure array have same shape"
    return np.linalg.norm((i - j), ord=2)


def heuristic_fn(
        node: np.ndarray,
        goal: np.ndarray,
        eps: float,
        dist_type=2,
        fn=None
) -> np.ndarray:
    assert isinstance(eps, float) or isinstance(eps, int)
    assert eps >= 1
    assert isinstance(node, np.ndarray)
    assert isinstance(goal, np.ndarray)
    assert node.ndim == goal.ndim, "Ensure array have same dimension"
    assert node.size == goal.size, "Ensure array have same size"
    typeDict = {
        1: "manhattan",
        2: "euclidean",
        3: "diagonal",
        4: "octile",
    }
    if dist_type == 1 or dist_type == typeDict[1]:
        ''' Manhattan Distance '''
        dist = eps * np.linalg.norm((node - goal), ord=1)
    elif dist_type == 2 or dist_type == typeDict[2]:
        ''' Euclidean Distance '''
        dist = eps * np.linalg.norm((node - goal), ord=2)
    elif dist_type == 3 or dist_type == typeDict[3]:
        ''' Diagonal Distance '''
        dist = eps * np.linalg.norm((node - goal), ord=np.inf)
    elif dist_type == 4 or dist_type == typeDict[4]:
        ''' Octile Distance '''
        dist = eps * np.linalg.norm((node - goal), ord=np.inf) - np.linalg.norm((node - goal), ord=-np.inf)
    else:
        print("Using Custom Heuristic Functions")
        dist = fn(node, goal, eps)
    return dist


def path_recon(Parent,start, goal, mapfile, res):
    boundary, blocks = utils.load_map(mapfile)

    a = goal
    path = goal
    while not ((a[0] == start[0]) and (a[1] == start[1]) and (a[2] == start[2])):
        b = np.array(Parent[tuple(a)])
        #         print(b)
        path = np.vstack((path, b))
        a = b
    path = ((path - 1) * res)
    path[:, 0] = path[:, 0] + boundary[0][0]
    path[:, 1] = path[:, 1] + boundary[0][1]
    path[:, 2] = path[:, 2] + boundary[0][2]
    path = np.flip(path, 0)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--render", help="Visualize Env", action="store_true", default=True)
    parser.add_argument("-res", help="Resolution", type=float, default=0.1)
    parser.add_argument("--seed", help="Random generator seed", type=int, default=42)
    parser.add_argument(
        "-verb", "--verbose", action="store_true", default=True,
        help="Verbose mode (False: no output, True: INFO)"
    )
    args = parser.parse_args()
    seed = args.seed
    print(np.__version__)
    np.seterr(all='raise')
    # map_file = MAPDICT[env_id]
    # # start pos and goal pos
    # start, goal = MAP_SE[env_id]
    # boundary, blocks = utils.load_map(map_file)
    # ic(boundary)
    # grid_world, grid_start, grid_goal = utils.make_grid_env(map_file, start, goal, res=0.1)
    # ic(grid_start)
    # for env in MAP_SE.keys():
    #     test_env(env, res=args.res, render=args.render, verbose=args.verbose)

    # test_env(env_id=env_id, res=args.res, render=args.render, verbose=args.verbose)
    env_id = 'monza'
    map_file = MAPDICT[env_id]
    start, goal = MAP_SE[env_id]
    eps = 2.0
    res = 0.1
    boundary, blocks = utils.load_map(map_file)
    MP = MyPlanner(boundary, blocks)
    t0 = utils.tic()
    path = MP.A_star(map_file, start, goal, eps=eps, res=res)
    utils.toc(t_start=t0, name=env_id)

    fig, ax, hb, hs, hg = utils.draw_map(boundary, blocks, start, goal)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], 'r-')
    plt.show(block=True)