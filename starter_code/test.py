import os
import sys
import re
import time
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

import Planner
import utils


def run_test(map_file, start, goal, verbose=True):
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
    MP = Planner.MyPlanner(boundary, blocks)

    # Display the environment
    if verbose:
        fig, ax, hb, hs, hg = utils.draw_map(boundary, blocks, start, goal)

    # Call the motion planner
    t0 = utils.tic()
    path = MP.plan(start, goal)
    test_name = re.split('[/.]', map_file)[3]
    utils.toc(t0, f"Planning {test_name}")

    # Plot the path
    if verbose:
        ax.plot(path[:, 0], path[:, 1], path[:, 2], 'r-')

    # TODO: You should verify whether the path actually intersects any of the obstacles in continuous space
    # TODO: You can implement your own algorithm or use an existing library for segment and
    #       axis-aligned bounding box (AABB) intersection
    collision = False
    goal_reached = sum((path[-1] - goal) ** 2) <= 0.1
    success = (not collision) and goal_reached
    pathlength = np.sum(np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1)))
    return success, pathlength


if __name__ == '__main__':
    print(np.__version__)
    path = './maps/'
    mapDict = {file.split('.')[0]: os.path.join(path, file) for file in os.listdir(path)}
    ic(mapDict)

    # start pos and goal
    start = np.array([2.3, 2.3, 1.3])
    goal = np.array([7.0, 7.0, 5.5])
    ic(start)

    res = 0.1
    map_file = mapDict['single_cube']
    boundary, blocks = utils.load_map(map_file)
    ic(boundary)
    grid_world, grid_start, grid_goal = utils.make_grid_env(map_file, start, goal, res=0.1)
    ic(grid_start)