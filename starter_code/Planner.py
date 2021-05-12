from typing import Tuple, List, Optional, Union, Callable

import numpy as np
from numba import jit
from pqdict import PQDict
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
        # utils.showXYZboundary(self.boundary)

        # Init start and goal pos
        self.start_pos = None
        self.goal_pos = None
        self.res = None

        #####################
        #       FCL         #
        #####################
        # Init start and goal collision obj
        self.start_obj = None
        self.goal_obj = None
        self.rad = None


        # Init AABBTree manager for one to many collision checking
        self.manager = None
        # params stores the name of obj as key and collision obj as value
        self.params = {}
        # Geometry
        self.geoms = []
        # mapping from id to obj and name
        self.geom_id_to_obj = {}
        self.geom_id_to_name = {}

        #################################
        # 3D Grid, boundary, and blocks #
        #################################
        self.grid_world = None
        self.grid_boundary = None
        self.grid_block = None
        self.grid_start = None
        self.grid_goal = None
        #####################
        #       A*          #
        #####################
        self.eps = None
        self.PARENT = {}
        self.cost_grid = None
        self.path = None
        self.grid_path = None
        self.max_num_node = None

    def init_collision_objects(
            self,
            blocks: np.ndarray,
            start: np.ndarray,
            goal: np.ndarray,
            rad: float,
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

    def greedy_plan(self, start: np.ndarray, goal: np.ndarray):
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
        dR = dR / np.sqrt(np.sum(dR ** 2, axis=0)) / 2.0 * 0.5

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

    def A_star(
            self,
            start: np.ndarray,
            goal: np.ndarray,
            rad: float,
            eps: float = 1,
            res: float = 0.1,
            distType: int = 2,
            h_fn=None,
    ):
        """
        Epsilon-Consistent A* Algorithm in 3-D Euclidean space
        :param start: start position
        :param goal:  goal position
        :param rad: radius of angent
        :param eps: epsilon
        :param res: resolution
        :param distType: type of distance as heuristic
        :param h_fn: custom heuristic function overwrite self.custom_h_fn
        """
        assert isinstance(eps, float) or isinstance(eps, int)
        assert eps >= 1, "eps should >= 1"
        self.eps = eps
        self.rad = rad
        # Convert map in continuous space to Discrete 3D Grid
        grid_env = utils.make_grid_env(self.boundary, self.blocks, start, goal, res=res)
        grid_world, grid_boundary, grid_block, grid_start, grid_goal = grid_env

        self.grid_world: np.ndarray = grid_world
        self.grid_boundary: np.ndarray = grid_boundary
        self.grid_block: np.ndarray = grid_block
        self.grid_start: np.ndarray = grid_start
        self.grid_goal: np.ndarray = grid_goal

        print("---------------------------")
        print(f"Grid_x_boundary: (0, {grid_world.shape[0]})")
        print(f"Grid_y_boundary: (0, {grid_world.shape[1]})")
        print(f"Grid_z_boundary: (0, {grid_world.shape[2]})")
        print(f"Grid_start: {grid_start}")
        print(f"Grid_goal: {grid_goal}")

        # Init collision objs for discrete collision checking
        self.init_collision_objects(
            blocks=grid_block,
            start=grid_start,
            goal=grid_goal,
            rad=rad,
        )

        # g_i = infinity for all i ∈ V\{s}
        cost_grid = np.empty_like(grid_world)
        cost_grid.fill(np.inf)

        # coords of start and goal as tuple
        s = tuple(grid_start)
        e = tuple(grid_goal)

        # g_s = 0
        cost_grid[s] = 0.0

        # Initialize OPEN  with the start coords and its cost
        OPEN = PQDict({s: 0.0})  # OPEN <- {s}

        # Initialize the CLOSED
        CLOSED = PQDict()

        # Initialize a Parent Dict will be used reconstruct the path
        self.PARENT = {}

        itr = 0
        max_num_node = 0
        while e not in CLOSED.keys():
            ''' util func to know the progress'''
            if len(OPEN) > max_num_node:
                max_num_node = len(OPEN)
            if itr % 100_000 == 0:
                print(len(CLOSED))

            # Remove i with the  smallest f_i := g_i + h_i from OPEN
            state_i: tuple
            f_i: np.ndarray
            state_i, f_i = OPEN.popitem()

            # Insert 'i' (state, cost) pair into CLOSED
            CLOSED.additem(state_i, f_i)

            # for j ∈ Children(i) and j ∈/ CLOSED
            i = np.array(state_i)
            child_i = self.child_of(i.flatten())
            for j in child_i:
                state_j = tuple(j)
                if state_j in CLOSED:
                    continue
                # if child node_j in bound it is not occupied
                if self.is_inbound(j):
                    # if child node_j is not occupied
                    if grid_world[state_j] == 0:
                        # Discrete collision checking
                        if self.is_static_collide(j, rad=self.rad):
                            continue
                        g_j = cost_grid[state_j]
                        g_i = cost_grid[state_i]
                        c_ij = self.cost(i, j)

                        # If find a shorter path
                        if g_j > (g_i + c_ij):
                            # Update Children's Cost.
                            cost_grid[state_j] = g_i + c_ij
                            # Record i as Parent of Child node j
                            self.PARENT[state_j] = state_i  # Parent(j) <- i

                            if h_fn is not None:
                                h_j = self.heuristic_fn(j, grid_goal, fn=self.custom_h_fn)
                            else:
                                h_j = self.heuristic_fn(j, grid_goal, dist_type=distType)

                            f_j = g_j + eps * h_j
                            if state_j in OPEN.keys():
                                # Update Update priority of j
                                OPEN.updateitem(state_j, f_j)
                            else:
                                # Add {j} to OPEN
                                OPEN.additem(state_j, f_j)
            itr += 1
        print("Planning Finished. Start Reconstructing Path Now")
        path, grid_path = self.path_recon(grid_start, grid_goal, res)
        self.path = path
        self.grid_path = grid_path
        self.cost_grid = cost_grid
        self.max_num_node = max_num_node
        return path, grid_path, cost_grid, max_num_node

    def isPointOutBound(self, next_point: np.ndarray) -> bool:
        """
        Check is the position out of bounds in continuous state space and assume agent is a point
        """
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
        """
        Check is the position inside AABB in continuous state space and assume agent is a point
        """
        next_point = next_point.reshape(-1)
        return all([
            self.blocks[k, 0] <= next_point[0] <= self.blocks[k, 3],  # [x_min, x_max]
            self.blocks[k, 1] <= next_point[1] <= self.blocks[k, 4],  # [y_min, y_max]
            self.blocks[k, 2] <= next_point[2] <= self.blocks[k, 5],  # [z_min, z_max]
        ])

    def is_static_collide(self, node: np.ndarray, rad, verbose=False) -> bool:
        """ Managed one to many discrete collision checking """

        current_obj = fcl.CollisionObject(fcl.Sphere(rad), fcl.Transform(node))

        req = fcl.CollisionRequest(num_max_contacts=1000, enable_contact=True)
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
        Perform one-to-one Continuous Collision Checking
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

    @staticmethod
    def child_of(node: np.ndarray) -> np.ndarray:
        assert isinstance(node, np.ndarray)
        assert node.ndim == 1
        [dX, dY, dZ] = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])
        dR = np.vstack((dX.flatten(), dY.flatten(), dZ.flatten()))
        # Remove (0,0,0)
        dR = np.delete(dR, 13, axis=1)
        children = node.reshape(3, -1) + dR
        return children.T

    def is_inbound(self, node: np.ndarray) -> bool:
        assert isinstance(node, np.ndarray)
        if node.ndim > 1:
            node.reshape(-1)
        cond = (
            0 <= node[0] < self.grid_world.shape[0],
            0 <= node[1] < self.grid_world.shape[1],
            0 <= node[2] < self.grid_world.shape[2],
        )
        return all(cond)

    def is_free(self, node: np.ndarray) -> bool:
        assert isinstance(node, np.ndarray)
        if node.ndim > 1:
            node.reshape(-1)
        return self.grid_world[tuple(node)] == 0

    @staticmethod
    @jit(nopython=True, cache=True, fastmath=True)
    def cost(i: np.ndarray, j: np.ndarray) -> np.ndarray:
        """Define the cost form node i to node j as Euclidean Distance"""
        # assert isinstance(i, np.ndarray)
        # assert isinstance(j, np.ndarray)
        # assert i.ndim == j.ndim and i.size == j.size, "Ensure array have same shape"
        # return np.linalg.norm((i - j), ord=2)
        
        # use following line instead to get speed up from numba
        return np.sqrt(np.sum((i - j)**2))

    def heuristic_fn(
            self,
            node: np.ndarray,
            goal: np.ndarray,
            dist_type: Union[str, int] = 2,
            fn: Optional[Callable] = None,
    ) -> np.ndarray:
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
        if fn is not None:
            dist = fn(node, goal)
        else:
            if dist_type == 1 or dist_type == typeDict[1]:
                ''' Manhattan Distance '''
                dist = np.linalg.norm((node - goal), ord=1)
            elif dist_type == 2 or dist_type == typeDict[2]:
                ''' Euclidean Distance '''
                # dist = np.linalg.norm((node - goal), ord=2)
                dist = self.euclidean_distance(node, goal)
            elif dist_type == 3 or dist_type == typeDict[3]:
                ''' Diagonal Distance '''
                dist = np.linalg.norm((node - goal), ord=np.inf)
            elif dist_type == 4 or dist_type == typeDict[4]:
                ''' Octile Distance '''
                dist = np.linalg.norm((node - goal), ord=np.inf) + np.linalg.norm((node - goal), ord=-np.inf)
            else:
                raise ValueError("Please provide valid dist_type or custom heuristic function")
        return dist

    @staticmethod
    @jit(nopython=True, cache=True, fastmath=True)
    def euclidean_distance(node, goal) -> np.ndarray:
        """ Use as a heuristic function in heuristic_fn"""
        dist = np.sqrt(np.sum((node - goal) ** 2))
        return dist

    @staticmethod
    def custom_h_fn(node, goal) -> np.ndarray:
        """ User Defined heuristic function """
        dist = np.linalg.norm((node - goal), ord=np.inf) + 0.7 * np.linalg.norm((node - goal), ord=-np.inf)
        return dist

    def path_recon(self, grid_start: np.ndarray, grid_goal: np.ndarray, res: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct the path from start to goal in 3D Grid and convert back to original map
        :param grid_start:
        :param grid_goal:
        :param res: resolution
        """
        e = grid_goal.astype(np.int32)
        path = grid_goal.astype(np.int32)
        while not np.array_equal(e, grid_start):
            prev = np.array(self.PARENT[tuple(e)], dtype=np.int32)
            path = np.vstack((path, prev))
            e = prev
        grid_path = np.flip(path, axis=0)
        path = (path - 1) * res + self.boundary[0, :3]
        path = np.flip(path, axis=0)
        return path, grid_path
