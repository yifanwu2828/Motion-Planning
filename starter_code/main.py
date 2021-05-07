from typing import Optional, Tuple
import time

import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import Planner

plt.ion()


############################################
############################################
def tic(message: Optional[str] = None) -> float:
    """ Timing Function """
    if message:
        print(message)
    else:
        print("############ Time Start ############")
    return time.time()


############################################
############################################
def toc(t_start: float, name: Optional[str] = "Operation", ftime=False) -> None:
    """ Timing Function """
    assert isinstance(t_start, float)
    sec: float = time.time() - t_start
    if ftime:
        duration = time.strftime("%H:%M:%S", time.gmtime(sec))
        print(f'\n############ {name} took: {str(duration)} ############\n')
    else:
        print(f'\n############ {name} took: {sec:.4f} sec. ############\n')


############################################


def load_map(fname: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the boundary and blocks from map file fname.

    boundary = [['x_min', 'y_min', 'z_min', 'x_max', 'y_max', 'z_max','r','g','b']]

    blocks = [['x_min', 'y_min', 'z_min', 'x_max', 'y_max', 'z_max','r','g','b'],
              ...,
              ['x_min', 'y_min', 'z_min', 'x_max', 'y_max', 'z_max','r','g','b']]
    """
    map_data = np.loadtxt(
        fname, dtype={
            'names': ('type', 'xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax', 'r', 'g', 'b'),
            'formats': ('S8', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f')
        }
    )

    blockIdx = map_data['type'] == b'block'

    boundary = map_data[~blockIdx][
                   ['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax', 'r', 'g', 'b']
               ].view('<f4').reshape(-1, 11)[:, 2:]

    blocks = map_data[blockIdx][
                 ['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax', 'r', 'g', 'b']
             ].view('<f4').reshape(-1, 11)[:, 2:]
    return boundary, blocks


def draw_map(boundary, blocks, start, goal):
    """
    Visualization of a planning problem with environment boundary, obstacle blocks, and start and goal points
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    hb = draw_block_list(ax, blocks)
    hs = ax.plot(start[0:1], start[1:2], start[2:], 'ro', markersize=7, markeredgecolor='k')
    hg = ax.plot(goal[0:1], goal[1:2], goal[2:], 'go', markersize=7, markeredgecolor='k')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(boundary[0, 0], boundary[0, 3])
    ax.set_ylim(boundary[0, 1], boundary[0, 4])
    ax.set_zlim(boundary[0, 2], boundary[0, 5])
    return fig, ax, hb, hs, hg


def draw_block_list(ax, blocks):
    """
    Subroutine used by draw_map() to display the environment blocks
    """
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
                 dtype='float'
                 )
    f = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [0, 1, 2, 3], [4, 5, 6, 7]])
    clr = blocks[:, 6:] / 255
    n = blocks.shape[0]
    d = blocks[:, 3:6] - blocks[:, :3]
    vl = np.zeros((8 * n, 3))
    fl = np.zeros((6 * n, 4), dtype='int64')
    fcl = np.zeros((6 * n, 3))
    for k in range(n):
        vl[k * 8:(k + 1) * 8, :] = v * d[k] + blocks[k, :3]
        fl[k * 6:(k + 1) * 6, :] = f + k * 8
        fcl[k * 6:(k + 1) * 6, :] = clr[k, :]

    if type(ax) is Poly3DCollection:
        ax.set_verts(vl[fl])
    else:
        pc = Poly3DCollection(vl[fl], alpha=0.25, linewidths=1, edgecolors='k')
        pc.set_facecolor(fcl)
        h = ax.add_collection3d(pc)
        return h


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
    boundary, blocks = load_map(map_file)
    MP = Planner.MyPlanner(boundary, blocks)  # TODO: replace this with your own planner implementation

    # Display the environment
    if verbose:
        fig, ax, hb, hs, hg = draw_map(boundary, blocks, start, goal)

    # Call the motion planner
    t0 = tic()
    path = MP.plan(start, goal)
    toc(t0, "Planning")

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
