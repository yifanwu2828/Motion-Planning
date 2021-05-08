import numpy as np


def get_centroid(block: np.ndarray):
    block = block.reshape(-1)
    block_x = (block[3] + block[0]) / 2.0
    block_y = (block[4] + block[1]) / 2.0
    block_z = (block[5] + block[2]) / 2.0
    return block_x, block_y, block_z


def get_XYZ_length(block: np.ndarray):
    block = block.reshape(-1)
    x_len = abs(block[0] - block[3])
    y_len = abs(block[1] - block[4])
    z_len = abs(block[2] - block[5])
    return x_len, y_len, z_len


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