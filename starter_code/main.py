import re

import numpy as np
from icecream import ic
import matplotlib.pyplot as plt


import Planner
import utils
plt.ion()


def test_single_cube(verbose=False):
    print('Running single cube test...\n')
    start = np.array([2.3, 2.3, 1.3])
    goal = np.array([7.0, 7.0, 5.5])
    success, pathlength = run_test('./maps/single_cube.txt', start, goal, verbose)
    print('Success: %r' % success)
    print('Path length: %d' % pathlength)
    print('\n')


def test_maze(verbose=False):
    print('Running maze test...\n')
    start = np.array([0.0, 0.0, 1.0])
    goal = np.array([12.0, 12.0, 5.0])
    success, pathlength = run_test('./maps/maze.txt', start, goal, verbose)
    print('Success: %r' % success)
    print('Path length: %d' % pathlength)
    print('\n')


def test_window(verbose=False):
    print('Running window test...\n')
    start = np.array([0.2, -4.9, 0.2])
    goal = np.array([6.0, 18.0, 3.0])
    success, pathlength = run_test('./maps/window.txt', start, goal, verbose)
    print('Success: %r' % success)
    print('Path length: %d' % pathlength)
    print('\n')


def test_tower(verbose=False):
    print('Running tower test...\n')
    start = np.array([2.5, 4.0, 0.5])
    goal = np.array([4.0, 2.5, 19.5])
    success, pathlength = run_test('./maps/tower.txt', start, goal, verbose)
    print('Success: %r' % success)
    print('Path length: %d' % pathlength)
    print('\n')


def test_flappy_bird(verbose=False):
    print('Running flappy bird test...\n')
    start = np.array([0.5, 2.5, 5.5])
    goal = np.array([19.0, 2.5, 5.5])
    success, pathlength = run_test('./maps/flappy_bird.txt', start, goal, verbose)
    print('Success: %r' % success)
    print('Path length: %d' % pathlength)
    print('\n')


def test_room(verbose=False):
    print('Running room test...\n')
    start = np.array([1.0, 5.0, 1.5])
    goal = np.array([9.0, 7.0, 1.5])
    success, pathlength = run_test('./maps/room.txt', start, goal, verbose)
    print('Success: %r' % success)
    print('Path length: %d' % pathlength)
    print('\n')


def test_monza(verbose=False):
    print('Running monza test...\n')
    start = np.array([0.5, 1.0, 4.9])
    goal = np.array([3.8, 1.0, 0.1])
    success, pathlength = run_test('./maps/monza.txt', start, goal, verbose)
    print('Success: %r' % success)
    print('Path length: %d' % pathlength)
    print('\n')


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
    MP = Planner.MyPlanner(boundary, blocks)  # TODO: replace this with your own planner implementation

    # Init collision obj
    MP.init_collision_objects(blocks, start, goal)

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



if __name__ == "__main__":
    test_single_cube(True)
    plt.show(block=True)

    test_maze(True)
    plt.show(block=True)

    test_flappy_bird(True)
    plt.show(block=True)

    test_monza(True)
    plt.show(block=True)

    test_window(True)
    plt.show(block=True)

    test_tower(True)
    plt.show(block=True)

    test_room(True)
    plt.show(block=True)

    # print('Running single cube test...\n')
    # start = np.array([2.3, 2.3, 1.3])
    # goal = np.array([7.0, 7.0, 5.5])
    # success, pathlength = run_test('./maps/single_cube.txt', start, goal, verbose=True)
    # print('Success: %r' % success)
    # print('Path length: %d' % pathlength)
    # print('\n')
