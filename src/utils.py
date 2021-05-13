import os
import time
import random
from typing import Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from icecream import ic


############################################
############################################
def tic(message: Optional[str] = None) -> float:
    """ Timing Function """
    if message:
        print(message)
    else:
        print("############ Time Start ############")
    return time.time()


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
############################################

def set_random_seed(seed: int = 42):
    """
    Seed the different random generators.
    :param seed:
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
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


def draw_map(
        boundary: np.ndarray,
        blocks: np.ndarray,
        start: np.ndarray,
        goal: np.ndarray
) -> Tuple[Any, Any, Any, Any, Any]:
    """
    Visualization of a planning problem with environment:
        - boundary,
        - obstacle blocks,
        - start and goal points
    * Add visualization for centroids of blocks
    """
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    hb = draw_block_list(ax, blocks)
    hs = ax.plot(start[0:1], start[1:2], start[2:], 'ro', markersize=7, markeredgecolor='k')
    hg = ax.plot(goal[0:1], goal[1:2], goal[2:], 'go', markersize=7, markeredgecolor='k')

    # draw centroid of each block
    for blk in blocks:
        centroid = get_centroid(block=blk)
        ax.plot(centroid[0], centroid[1], centroid[2], 'bo', markersize=7, markeredgecolor='k')

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
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
                 dtype='float'
                 )
    f = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6],
                  [3, 0, 4, 7], [0, 1, 2, 3], [4, 5, 6, 7]])

    clr = blocks[:, 6:] / 255  # RGB color
    n = blocks.shape[0]     # num of blocks
    d = blocks[:, 3:6] - blocks[:, :3]  # (x_len, y_len, z_len)

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


############################################
############################################

def get_centroid(block: np.ndarray):
    """ Find centroid of blocks"""
    block = block.reshape(-1)
    block_x = (block[3] + block[0]) / 2.0
    block_y = (block[4] + block[1]) / 2.0
    block_z = (block[5] + block[2]) / 2.0
    return block_x, block_y, block_z


def get_XYZ_length(block: np.ndarray):
    """ Find side length of block """
    block = block.reshape(-1)
    x_len = abs(block[0] - block[3])
    y_len = abs(block[1] - block[4])
    z_len = abs(block[2] - block[5])
    return x_len, y_len, z_len


def showXYZboundary(boundary):
    boundary = boundary.reshape(-1)
    print(f"x_boundary: ({boundary[0]}, {boundary[3]})")
    print(f"y_boundary: ({boundary[1]}, {boundary[4]})")
    print(f"z_boundary: ({boundary[2]}, {boundary[5]})")


############################################
############################################

def make_grid_env(boundary, blocks, start, goal, res=0.1):
    """
    Discretize the world into 3D Grid
        Occupied cells are marked with 1 and free cells are marked as zero
    :param boundary: boundary
    :param blocks: obstacles
    :param start: start position in (x, y, z)
    :param goal: goal position in (x, y, z)
    :param res: resolution
    :return: 3D Grid map, discrete start pos, discrete goal pos
    """
    # Discretize start and goal
    grid_start = np.ceil(((start - boundary[0, :3]) / res) + 1).astype('int')
    grid_goal = np.ceil(((goal - boundary[0, :3]) / res) + 1).astype('int')
    # ic(grid_start)
    # ic(grid_goal)


    # Discrete 3D grid dimensions.
    dim = np.ceil(((boundary[0, 3:6] - boundary[0, 0:3]) / res) + 1).astype('int')
    # Initialize the grid world
    grid_world = np.zeros(tuple(dim))
    # Initialize the boundary walls
    grid_world[0, :, :] = 1
    grid_world[:, 0, :] = 1
    grid_world[:, :, 0] = 1

    grid_boundary = boundary.copy()
    grid_boundary[0, 3] = (grid_boundary[0, 3] + abs(grid_boundary[0, 0])) / res + 1
    grid_boundary[0, 4] = (grid_boundary[0, 4] + abs(grid_boundary[0, 1])) / res + 1
    grid_boundary[0, 5] = (grid_boundary[0, 5] + abs(grid_boundary[0, 2])) / res + 1
    grid_boundary[0, 0:3] = 0
    grid_boundary = grid_boundary.astype('int')
    # ic(grid_boundary)

    # Convert blocks to grid coordinates
    grid_block = blocks.copy()
    grid_block[:, 0:3] -= boundary[0, :3]
    grid_block[:, 3:6] -= boundary[0, :3]
    grid_block[:, :6] = np.ceil((grid_block[:, :6] / res) + 1)
    grid_block = grid_block.astype('int')
    # ic(grid_block)

    # Initialize blocks in grid world
    for i in range(blocks.shape[0]):
        grid_world[
            grid_block[i, 0] - 1: grid_block[i, 3] + 1,  # [x_min x_max]
            grid_block[i, 1] - 1: grid_block[i, 4] + 1,  # [y_min y_max]
            grid_block[i, 2] - 1: grid_block[i, 5] + 1,  # [z_min z_max]
        ] = 1

    return grid_world, grid_boundary, grid_block, grid_start, grid_goal


