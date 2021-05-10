import os
import time
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
    fig = plt.figure(figsize=(40, 30))
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
    grid_start = np.ceil(((start - boundary[0, 0:3]) / res) + 1).astype('int')
    grid_goal = np.ceil(((goal - boundary[0, 0:3]) / res) + 1).astype('int')

    # Discrete grid dimensions.
    num_x = np.ceil(int(((boundary[0, 3] - boundary[0, 0]) / res) + 1)).astype('int')
    num_y = np.ceil(int(((boundary[0, 4] - boundary[0, 1]) / res) + 1)).astype('int')
    num_z = np.ceil(int(((boundary[0, 5] - boundary[0, 2]) / res) + 1)).astype('int')

    # Initialize the grid world
    grid_world = np.zeros((num_x, num_y, num_z))

    # Initialize the boundary walls
    grid_world[0, :, :] = 1
    grid_world[:, 0, :] = 1
    grid_world[:, :, 0] = 1
    # grid_boundary = np.vstack([grid_world[0, :, :], grid_world[:, 0, :], grid_world[:, :, 0]])

    # Convert blocks to grid coordinates
    blocks[:, 0:3] -= boundary[0, 0:3]
    blocks[:, 3:6] -= boundary[0, 0:3]
    grid_block = np.ceil((blocks / res) + 1).astype('int')

    # grid_world = np.zeros((6,6,6))

    # Initialize blocks in grid world
    for i in range(blocks.shape[0]):
        grid_world[
            grid_block[i, 0] - 1: grid_block[i, 3] + 1,  # [x_min x_max]
            grid_block[i, 1] - 1: grid_block[i, 4] + 1,  # [y_min y_max]
            grid_block[i, 2] - 1: grid_block[i, 5] + 1,  # [z_min z_max]
        ] = 1
    return grid_world, grid_block, grid_start, grid_goal











# def isBoxIntersect(self, box_node: np.ndarray, k: int):
#     """
#     Check if two box AABB intersect
#     :param box_node: AABB of Node
#     :param k: index of blocks
#     """
#     box_node = box_node.reshape(-1)
#     return all([
#         box_node[0] <= self.blocks[k, 3],  # a.minX <= b.maxX
#         box_node[3] >= self.blocks[k, 0],  # a.maxX >= b.minX
#         box_node[1] <= self.blocks[k, 4],  # a.minY <= b.maxY
#         box_node[4] >= self.blocks[k, 1],  # a.maxY >= b.minY
#         box_node[2] <= self.blocks[k, 5],  # a.minZ <= b.maxZ
#         box_node[5] >= self.blocks[k, 2],  # a.maxZ >= b.minZ
#     ])
#
# def isSphereIntersect(self, sphere_node: np.ndarray, k: int):
#     """
#     Sphere vs. AABB intersect
#     :param sphere_node: Sphere of Node
#         [x, y, z, rad]
#     :param k: index of blocks
#     """
#     sphere_node = sphere_node.reshape(-1)
#
#     x = np.max([self.blocks[k, 0], np.min([sphere_node[0], self.blocks[k, 3]])])
#     y = np.max([self.blocks[k, 1], np.min([sphere_node[1], self.blocks[k, 4]])])
#     z = np.max([self.blocks[k, 2], np.min([sphere_node[2], self.blocks[k, 5]])])
#
#     distance = np.sqrt(np.sum((np.hstack([x, y, z]) - sphere_node[0:3])**2))
#
#     # distance = np.sqrt((x - sphere_node[0]) ** 2 + (y - sphere_node[1]) ** 2 + (z - sphere_node[2]) ** 2)
#     return distance < sphere_node[4]

# def main():
#     params = OrderedDict()
#     print('Running single cube test...\n')
#     start = np.array([0.5, 1.0, 4.9])
#     goal = np.array([3.8, 1.0, 0.1])
#
#     map_file = './maps/monza.txt'
#
#     # success, pathlength = run_test(map_file, start, goal, verbose=True)
#     # print('Success: %r' % success)
#     # print('Path length: %d' % pathlength)
#     # print('\n')
#     boundary, blocks = load_map(map_file)
#
#     fig, ax, hb, hs, hg, = draw_map(boundary, blocks, start, goal)
#     plt.show(block=True)
#
#     # Create object
#     # assume start and goal are sphere, blocks are box
#     # all objects initially at origin and translate to pos
#     geom_start = fcl.Sphere(0.01)
#     geom_goal = fcl.Sphere(0.01)
#     params['start_obj'] = fcl.CollisionObject(geom_start, fcl.Transform(start))
#     params['goal_obj'] = fcl.CollisionObject(geom_goal, fcl.Transform(goal))
#
#     geoms = []
#     for i, blk in enumerate(blocks):
#         geom_box = fcl.Box(*utils.get_XYZ_length(blk))
#         tf_box = fcl.Transform(np.array(utils.get_centroid(block=blk)))
#         params[f'box_obj_{i}'] = fcl.CollisionObject(geom_box, tf_box)
#         geoms.append(geom_box)
#     names = list(params.keys())
#     objs = list(params.values())
#
#     # Create map from geometry IDs to objects
#     geom_id_to_obj = {id(geom): obj for geom, obj in zip(geoms, objs)}
#
#     # Create map from geometry IDs to string names
#     geom_id_to_name = {id(geom): name for geom, name in zip(geoms, names)}
#
#     # Managed one to many collision checking
#     manager = fcl.DynamicAABBTreeCollisionManager()
#     manager.registerObjects(objs)
#     manager.setup()
#
#     req = fcl.CollisionRequest(num_max_contacts=100, enable_contact=True)
#     cdata = fcl.CollisionData(request=req)
#
#     manager.collide(params['start_obj'], cdata, fcl.defaultCollisionCallback)
#     print(f'Collision between manager 1 and agent: {cdata.result.is_collision}')
#
#     # Extract collision data from contacts and use that to infer set of
#     # objects that are in collision
#     objs_in_collision = set()
#     for contact in cdata.result.contacts:
#         # Extract collision geometries that are in contact
#         coll_geom_0 = contact.o1
#         coll_geom_1 = contact.o2
#         print(f'\tO1: {contact.o1}, O2: {contact.o2}')
#
#         # Get their names
#         coll_names = [geom_id_to_name[id(coll_geom_0)], geom_id_to_name[id(coll_geom_1)]]
#         coll_names = tuple(sorted(coll_names))
#         objs_in_collision.add(coll_names)
#
#     for coll_pair in objs_in_collision:
#         print(f'Object {coll_pair[0]} in collision with object {coll_pair[1]}!')