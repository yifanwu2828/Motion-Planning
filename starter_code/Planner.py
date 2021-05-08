import numpy as np
import fcl
import tqdm
from icecream import ic

import utils


class MyPlanner(object):
    # __slots__ = ['boundary', 'blocks']

    def __init__(self, boundary, blocks):
        self.boundary = boundary
        self.blocks = blocks

        self.start = None
        self.goal = None
        self.rad = 0

        self.params = {}
        self.geoms = []
        self.geom_id_to_obj = None
        self.geom_id_to_name = None
        self.manager = None

    def init_collision_objects(self, start, goal=None, rad: int = 0.05):
        """
        Create object assume start & goal are sphere, blocks are box
        """
        self.rad = rad
        # init start collision obj
        geom_start = fcl.Sphere(self.rad)
        self.start = fcl.CollisionObject(geom_start, fcl.Transform(start))

        # init goal collision obj
        if goal is not None:
            geom_goal = fcl.Sphere(self.rad)
            self.goal = fcl.CollisionObject(geom_goal, fcl.Transform(goal))

        # init blocks collision obj
        for i, blk in enumerate(self.blocks):
            # len of x,y,z
            x, y, z = utils.get_XYZ_length(blk)
            # centroid position
            centroid = utils.get_centroid(block=blk)
            # geometry
            geom_box = fcl.Box(x, y, z)
            # transformation
            tf_box = fcl.Transform(np.array(centroid))
            self.params[f'box_obj_{i}'] = fcl.CollisionObject(geom_box, tf_box)
            self.geoms.append(geom_box)
        names = list(self.params.keys())
        objs = list(self.params.values())

        # Create map from geometry IDs to objects
        self. geom_id_to_obj = {id(geom): obj for geom, obj in zip(self.geoms, objs)}

        # Create map from geometry IDs to string names
        self.geom_id_to_name = {id(geom): name for geom, name in zip(self.geoms, names)}

        # Managed one to many collision checking
        self.manager = fcl.DynamicAABBTreeCollisionManager()
        self.manager.registerObjects(objs)
        self.manager.setup()

    def plan(self, start: np.ndarray, goal: np.ndarray,):
        # start pos
        path = [start]
        # neighbours
        num_dirs = 26
        num_blocks = self.blocks.shape[0]

        [dX, dY, dZ] = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])
        dR = np.vstack((dX.flatten(), dY.flatten(), dZ.flatten()))
        dR = np.delete(dR, 13, axis=1)
        dR = dR / np.sqrt(np.sum(dR ** 2, axis=0)) / 2.0

        for _ in range(2_000):
            min_dist2goal = 1_000_000
            node = None
            for i in range(num_dirs):
                # current node -> next node
                next_node = path[-1] + dR[:, i]

                # # Check if this direction is valid (next_node with direction is in bound)
                if self.isPointOutBound(next_node):
                    continue

                valid = True
                for k in range(num_blocks):
                    if self.isPointInsideAABB(k, next_node):
                        valid = False
                        break

                if not valid:
                    continue

                # Update next_node
                dist2goal = sum((next_node - goal) ** 2)
                if dist2goal < min_dist2goal:
                    min_dist2goal = dist2goal
                    node = next_node

            if node is None:
                break
            else:
                path.append(node)

            # Check if done
            goal_reached = sum((path[-1] - goal) ** 2) <= 0.1
            if goal_reached:
                break

        return np.array(path)

    def isPointOutBound(self, next_point):
        # Assuming point without init start as a collision object
        next_point = (next_point + self.rad).reshape(-1)
        return any([
            next_point[0] <= self.boundary[0, 0],  # x_min
            next_point[0] >= self.boundary[0, 3],  # x_max

            next_point[1] <= self.boundary[0, 1],  # y_min
            next_point[1] >= self.boundary[0, 4],  # y_max

            next_point[2] <= self.boundary[0, 2],  # z_min
            next_point[2] >= self.boundary[0, 5],  # z_max
        ])

    def isPointInsideAABB(self, k, next_point):
        next_point = (next_point + self.rad).reshape(-1)
        return all([
            self.blocks[k, 0] <= next_point[0] <= self.blocks[k, 3],  # [x_min, x_max]
            self.blocks[k, 1] <= next_point[1] <= self.blocks[k, 4],  # [y_min, y_max]
            self.blocks[k, 2] <= next_point[2] <= self.blocks[k, 5],  # [z_min, z_max]
        ])


