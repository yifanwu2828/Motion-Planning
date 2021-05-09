from typing import Optional, List

import numpy as np
import fcl
from tqdm import tqdm
from icecream import ic

import utils


class MyPlanner(object):

    def __init__(self, boundary, blocks):

        # Init boundary and blocks
        self.boundary: np.ndarray = boundary
        self.blocks: np.ndarray = blocks

        # Visualize the boundary in original and continuous obs space
        utils.showXYZboundary(self.boundary)

        # Init start and goal pos
        self.start_pos = None
        self.goal_pos = None
        self.res: float = 0.1

        # Init start and goal collision obj
        self.start_obj = None
        self.goal_obj = None
        self.rad: float = None


        # Init AABBTree manager for one to many collision checking
        self.manager = None
        # params stores the name of obj as key and collision obj as value
        self.params = {}
        # Geometry
        self.geoms = []
        # mapping from id to obj and name
        self.geom_id_to_obj: dict = None
        self.geom_id_to_name: dict = None


    def init_collision_objects(
            self,
            blocks: np.ndarray,
            start: np.ndarray,
            goal: np.ndarray,
            rad: float = 0.05
    ) -> None:
        """
        Create object assume start & goal are sphere, blocks are box,
        """
        # Radius of Sphere collision obj used for agent
        self.rad = rad
        geom_sphere = fcl.Sphere(self.rad)
        # init start collision obj
        self.start_obj = fcl.CollisionObject(geom_sphere, fcl.Transform(start))

        # init goal collision obj
        self.goal_obj = fcl.CollisionObject(geom_sphere, fcl.Transform(goal))

        # init blocks collision objs as Box
        for idx, blk in enumerate(blocks):
            # Side length of x,y,z
            x, y, z = utils.get_XYZ_length(blk)
            # Centroid position of the block
            centroid = utils.get_centroid(block=blk)
            # Geometry of collision obj
            geom_box = fcl.Box(x, y, z)
            # Transformation(no Rotation only Translation)
            tf_box = fcl.Transform(np.array(centroid))
            self.params[f'box_obj_{idx}'] = fcl.CollisionObject(geom_box, tf_box)
            self.geoms.append(geom_box)
        names = list(self.params.keys())
        objs = list(self.params.values())

        # Create map from geometry IDs to objects
        self.geom_id_to_obj = {id(geom): obj for geom, obj in zip(self.geoms, objs)}

        # Create map from geometry IDs to string names
        self.geom_id_to_name = {id(geom): name for geom, name in zip(self.geoms, names)}

        # Managed one to many collision checking
        self.manager = fcl.DynamicAABBTreeCollisionManager()
        self.manager.registerObjects(objs)
        self.manager.setup()

    def plan(self, start: np.ndarray, goal: np.ndarray,):
        self.start_pos = start
        self.goal_pos = goal
        # start pos
        path: List[np.ndarray] = [start]
        # neighbours
        num_dirs: int = 26
        num_blocks: int = self.blocks.shape[0]
        ic(num_blocks, num_blocks**2)

        [dX, dY, dZ] = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])
        dR = np.vstack((dX.flatten(), dY.flatten(), dZ.flatten()))
        # Remove (0,0,0)
        dR = np.delete(dR, 13, axis=1)
        dR = dR / np.sqrt(np.sum(dR ** 2, axis=0)) / 2.0

        for _ in tqdm(range(5_000)):
            min_dist2goal = np.inf
            node: Optional[np.ndarray] = None
            for i in range(num_dirs):
                # current node -> next node
                current_node = path[-1]
                action = dR[:, i]
                next_node = path[-1] + dR[:, i]

                # Check if next_node with direction is out of bound
                if self.isPointOutBound(next_node):
                    continue

                static_collide = self.is_static_collide(next_node, verbose=False)
                motion_collide = False
                if not static_collide:
                    # Perform Continuous Collision Checking
                    motion_collide = self.is_motion_collide(current_node, T=action)

                if static_collide or motion_collide:
                    continue
                else:
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

    def isPointOutBound(self, next_point: np.ndarray) -> bool:
        # Assuming point without init start as a collision object
        next_point = next_point.reshape(-1)
        return any([
            next_point[0] <= self.boundary[0, 0],  # x_min
            next_point[0] >= self.boundary[0, 3],  # x_max

            next_point[1] <= self.boundary[0, 1],  # y_min
            next_point[1] >= self.boundary[0, 4],  # y_max

            next_point[2] <= self.boundary[0, 2],  # z_min
            next_point[2] >= self.boundary[0, 5],  # z_max
        ])

    def isPointInsideAABB(self, k: int, next_point: np.ndarray) -> bool:
        next_point = next_point.reshape(-1)
        return all([
            self.blocks[k, 0] <= next_point[0] <= self.blocks[k, 3],  # [x_min, x_max]
            self.blocks[k, 1] <= next_point[1] <= self.blocks[k, 4],  # [y_min, y_max]
            self.blocks[k, 2] <= next_point[2] <= self.blocks[k, 5],  # [z_min, z_max]
        ])

    def is_static_collide(self, node: np.ndarray, verbose=False) -> bool:
        """ Managed one to many collision checking """

        current_obj = fcl.CollisionObject(fcl.Sphere(self.rad), fcl.Transform(node))

        req = fcl.CollisionRequest(num_max_contacts=1_000, enable_contact=True)
        cdata = fcl.CollisionData(request=req, result=fcl.CollisionResult())
        self.manager.collide(current_obj, cdata, fcl.defaultCollisionCallback)
        objs_in_collision = set()

        collide = cdata.result.is_collision

        if verbose and collide:
            for contact in cdata.result.contacts:
                # Extract collision geometries that are in contact
                coll_geom_0 = contact.o1
                coll_geom_1 = contact.o2

                # Get their names
                coll_names = [
                    self.geom_id_to_name.get(id(coll_geom_0), 'start'),
                    self.geom_id_to_name.get(id(coll_geom_1), 'start')
                ]
                coll_names = tuple(sorted(coll_names))
                objs_in_collision.add(coll_names)
            for coll_pair in objs_in_collision:
                print(f"Object '{coll_pair[0]}' in collision with object '{coll_pair[1]}'!")
        return collide

    def is_motion_collide(self, node: np.ndarray, T: np.ndarray, verbose=False) -> bool:
        """
        Perform Continuous Collision Checking
        :param node: current position
        :param T: translation
        :param verbose:
        """
        # Agent
        g1 = fcl.Sphere(self.rad)
        t1 = fcl.Transform(node)
        o1 = fcl.CollisionObject(g1, t1)
        t1_final = fcl.Transform(T)

        motion_collide = False
        for k in range(self.blocks.shape[0]):
            blk = self.blocks[k, :]
            g2 = fcl.Box(*utils.get_XYZ_length(blk))
            t2 = fcl.Transform(np.array(utils.get_centroid(blk)))
            o2 = fcl.CollisionObject(g2, t2)

            request = fcl.ContinuousCollisionRequest()
            result = fcl.ContinuousCollisionResult()
            ret = fcl.continuousCollide(o1, t1_final, o2, t2, request, result)
            if verbose:
                print(f"time_of_contact: {result.time_of_contact}")

            motion_collide = result.is_collide
            if motion_collide:
                break
        return motion_collide


