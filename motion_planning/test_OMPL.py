import argparse
import os
import sys
import pickle
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt;plt.ion()
from tqdm import tqdm
from icecream import ic
import fcl

from Planner import MyPlanner
from ompl_utils import *
import utils

import sys

# add ompl to sys path
sys.path.append('OMPL/ompl-1.5.2/py-bindings')

try:
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    from os.path import abspath, dirname, join

    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), 'py-bindings'))
    print(sys.path)
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og

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


class CCDMotionType:
    CCDM_TRANS, CCDM_LINEAR, CCDM_SCREW, CCDM_SPLINE = range(4)


class CCDSolverType:
    CCDC_NAIVE, CCDC_CONSERVATIVE_ADVANCEMENT, CCDC_RAY_SHOOTING, CCDC_POLYNOMIAL_SOLVER = range(4)


class GJKSolverType:
    GST_LIBCCD, GST_INDEP = range(2)


class StateValidityChecker(ob.StateValidityChecker):
    def __init__(self, si, boundary, blocks, rad):
        super().__init__(si)
        self.si = si
        self.boundary = boundary
        self.blocks = blocks
        self.rad = rad
        self.params = {}
        self.geoms = []

        # init blocks collision objs as Box
        for idx, blk in enumerate(self.blocks):
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

    def isValid(self, state):
        """State Validation Check"""
        bound = self.boundary.flatten()
        # check if state is inbound
        cond_inbound = (
            bound[0] <= state[0] < bound[3],
            bound[1] <= state[1] < bound[4],
            bound[2] <= state[2] < bound[5],
        )

        valid = True
        if all(cond_inbound):
            node = np.array([state[0], state[1], state[2]])
            current_obj = fcl.CollisionObject(fcl.Sphere(self.rad), fcl.Transform(node))
            req = fcl.CollisionRequest(enable_contact=False)
            cdata = fcl.CollisionData(request=req, result=fcl.CollisionResult())
            self.manager.collide(current_obj, cdata, fcl.defaultCollisionCallback)

            collide = cdata.result.is_collision
            if collide:
                valid = False
        else:
            valid = False

        return valid

    @staticmethod
    def get_centroid(block: np.ndarray):
        """ Find centroid of blocks"""
        block = block.reshape(-1)
        block_x = (block[3] + block[0]) / 2.0
        block_y = (block[4] + block[1]) / 2.0
        block_z = (block[5] + block[2]) / 2.0
        return block_x, block_y, block_z


class MotionValidator(ob.MotionValidator):
    def __init__(self, si, boundary, blocks, rad):
        super().__init__(si)
        self.si = si
        self.boundary = boundary
        self.blocks = blocks
        self.rad = rad
        self.params = {}
        self.geoms = []

    def checkMotion(self, s1, s2, check_all=True, verbose=False):
        state = np.array([s1[0], s1[1], s1[2]])
        next_state = np.array([s2[0], s2[1], s2[2]])

        # g1 = fcl.Sphere(self.rad)
        g1 = fcl.Box(self.rad, self.rad, self.rad)
        t1 = fcl.Transform(state)
        o1 = fcl.CollisionObject(g1, t1)

        T = next_state - state
        t1_final = fcl.Transform(T)

        valid = True
        for k in range(blocks.shape[0]):
            blk = blocks[k, :]
            x, y, z = utils.get_XYZ_length(blk)
            g2 = fcl.Box(x, y, z)
            centroid = utils.get_centroid(blk)
            t2 = fcl.Transform(np.array(centroid))
            o2 = fcl.CollisionObject(g2, t2)

            # IMPORTANT solver type
            request = fcl.ContinuousCollisionRequest(
                num_max_iterations=10,
                ccd_motion_type=CCDMotionType.CCDM_LINEAR,#CCDM_TRANS,
                gjk_solver_type=1,
                ccd_solver_type=CCDSolverType.CCDC_CONSERVATIVE_ADVANCEMENT, #CCDC_NAIVE, #CCDC_RAY_SHOOTING,
            )
            result = fcl.ContinuousCollisionResult()
            request.gjk_solver_type = 1
            ret = fcl.continuousCollide(o1, t1_final, o2, fcl.Transform(), request, result)
            motion_collide = result.is_collide
            if motion_collide:
                valid = False
                if verbose:
                    ic(motion_collide)
                    print(f"Agent collide with block: {k}")
                    print(f"Total blocks: {blocks.shape[0] - 1}")
                    ic(blocks[k, :])
                if not check_all:
                    break
        return valid

    @staticmethod
    def get_XYZ_length(block: np.ndarray):
        """ Find side length of block """
        block = block.reshape(-1)
        x_len = abs(block[0] - block[3])
        y_len = abs(block[1] - block[4])
        z_len = abs(block[2] - block[5])
        return x_len, y_len, z_len

    @staticmethod
    def get_centroid(block: np.ndarray):
        """ Find centroid of blocks"""
        block = block.reshape(-1)
        block_x = (block[3] + block[0]) / 2.0
        block_y = (block[4] + block[1]) / 2.0
        block_z = (block[5] + block[2]) / 2.0
        return block_x, block_y, block_z


def plan(runTime: float, plannerType, objectiveType, fname: str, param: dict):
    """planning"""
    path_obj = None

    # Construct the robot state space. planning in X,Y,Z, a subset of R^3.
    space = ob.RealVectorStateSpace(3)

    # Set the bounds of space
    boundary = param["boundary"].copy().astype(np.float64)
    bounds = ob.RealVectorBounds(3)
    # x_lim
    bounds.low[0] = boundary[0, 0]
    bounds.high[0] = boundary[0, 3]
    # y_lim
    bounds.low[1] = boundary[0, 1]
    bounds.high[1] = boundary[0, 4]
    # z_lim
    bounds.low[2] = boundary[0, 2]
    bounds.high[2] = boundary[0, 5]
    # set bound wrt x_lim, y_lim, z_lim
    space.setBounds(bounds)

    # Construct a space information instance for this state space
    si = ob.SpaceInformation(space)

    # Set the object used to check which states in the space are valid
    state_valid_checker = StateValidityChecker(si, boundary, blocks, param['rad'])
    si.setStateValidityChecker(state_valid_checker)

    # continuous motion validation
    motion_valid_checker = MotionValidator(si, boundary, blocks, param['rad'])
    si.setMotionValidator(motion_valid_checker)

    si.setup()

    # Set our robot's starting state
    start = ob.State(space)
    start[0] = param["start"][0]
    start[1] = param["start"][1]
    start[2] = param["start"][2]

    # Set our robot's goal state to be the goal position
    goal = ob.State(space)
    goal[0] = param["goal"][0]
    goal[1] = param["goal"][1]
    goal[2] = param["goal"][2]

    # Create a problem instance
    pdef = ob.ProblemDefinition(si)

    # Set the start and goal states
    pdef.setStartAndGoalStates(start, goal)

    # Create the optimization objective
    pdef.setOptimizationObjective(allocateObjective(si, objectiveType))

    # Construct the optimal planner
    optimizingPlanner = allocatePlanner(si, plannerType)

    # Set the problem instance for our planner to solve
    optimizingPlanner.setProblemDefinition(pdef)
    optimizingPlanner.setup()

    # attempt to solve the planning problem in the given runtime
    solved = optimizingPlanner.solve(runTime)

    if solved:
        path_obj = pdef.getSolutionPath()
        algo_name = optimizingPlanner.getName()
        path_length = pdef.getSolutionPath().length()
        cost = pdef.getSolutionPath().cost(
            pdef.getOptimizationObjective()
        ).value()

        # ic(algo_name)
        # ic(path_length)
        # ic(cost)

        # Output the length of the path found
        print(
            f'\n{algo_name} found solution of path length {path_length:.4f}'
            # f'with an optimization objective value of {cost:.4f}'
        )

        # If a filename was specified, output the path as a matrix to
        # that file for visualization
        if fname:
            with open(fname, 'w') as outFile:
                outFile.write(pdef.getSolutionPath().printAsMatrix())
    else:
        print("No solution found.")
    return path_obj


if __name__ == "__main__":
    if True:
        # Create an argument parser
        parser = argparse.ArgumentParser(description='Optimal motion planning demo program.')

        # Add a filename argument
        parser.add_argument(
            '-t', '--runtime', type=float, default=10.0,
            help='(Optional) Specify the runtime in seconds. Defaults to 1 and must be greater than 0.'
        )
        parser.add_argument(
            '-p', '--planner', default='RRTstar',
            choices=['BFMTstar', 'BITstar', 'FMTstar', 'InformedRRTstar', 'PRMstar', 'RRTstar', 'SORRTstar'],
            help='(Optional) Specify the optimal planner to use, defaults to RRTstar if not given.'
        )
        parser.add_argument(
            '-o', '--objective', default='PathLength',
            choices=['PathClearance', 'PathLength', 'ThresholdPathLength', 'WeightedLengthAndClearanceCombo'],
            help='(Optional) Specify the optimization objective, defaults to PathLength if not given.'
        )
        parser.add_argument(
            '-f', '--file', default=None,
            help='(Optional) Specify an output path for the found solution path.'
        )
        parser.add_argument(
            '-i', '--info', type=int, default=0, choices=[0, 1, 2],
            help='(Optional) Set the OMPL log level. 0 for WARN, 1 for INFO, 2 for DEBUG. Defaults to WARN.')

        # Parse the arguments
        args = parser.parse_args()

        # Check that time is positive
        if args.runtime <= 0:
            raise argparse.ArgumentTypeError(
                "argument -t/--runtime: invalid choice: %r (choose a positive number greater than 0)" % (args.runtime,)
            )

        # Set the log level
        if args.info == 0:
            ou.setLogLevel(ou.LOG_WARN)
        elif args.info == 1:
            ou.setLogLevel(ou.LOG_INFO)
        elif args.info == 2:
            ou.setLogLevel(ou.LOG_DEBUG)
        else:
            ou.OMPL_ERROR("Invalid log-level integer.")
    ###################################################################################################

    args.runtime = 30
    runtime_env = {
        'single_cube': 5,
        'monza': 10,  # Fail go through
        'flappy_bird': 10,
        'window': 10,
        'room': 10,  # Fail go through
        'maze': 240,  # Fail need more time 300
        'tower': 60  # Fail go through,
    }

    lst = []
    env_id = "monza"
    print(f"\n############ {env_id} ############ ")
    map_file = MAPDICT[env_id]
    start, goal = MAP_SE[env_id]
    boundary, blocks = utils.load_map(map_file)

    MV = MotionValidator(None, boundary, blocks, rad=0.1)
    MV.checkMotion(start, goal, check_all=True, verbose=True)

    param = {
        "start": start,
        "goal": goal,
        "boundary": boundary,
        "blocks": blocks,
        "rad": 0.1
    }

    # # Solve the planning problem
    # # t = runtime_env[env_id]
    t = 10
    path_obj = plan(t, args.planner, args.objective, args.file, param)
    #
    print_path = path_obj.printAsMatrix()
    path = None
    if print_path is not None:
        path = utils.extract_print_path(print_path=path_obj.printAsMatrix())
        # Evaluation: one-to-one Continuous Collision Checking

        collide_lst = []
        for i in range(1, len(path)):
            prev_node = path[i - 1]
            node = path[i]
            cnt_collide = not MV.checkMotion(prev_node, node, check_all=True, verbose=True)
            collide_lst.append(cnt_collide)

        collision = True if True in collide_lst else False
        goal_reached = sum((path[-1] - goal) ** 2) <= 0.1
        success = (not collision) and goal_reached
        pathlength = utils.get_path_length(path)

    #     ic(path)
        ic(collision)
        ic(collide_lst)
        ic(success)
        ic(pathlength)
    print("###################################\n")
    # # Original Plot
    fig, ax, hb, hs, hg = utils.draw_map(boundary, blocks, start, goal)
    if path is not None:
        ax.plot(path[:, 0], path[:, 1], path[:, 2], 'r-')
    plt.title(f"Original {env_id} Env")
    plt.show(block=True)
    # # lst.append((fig, ax, hb, hs, hg))
    # # for p in lst:
    # #     f, a, b, s, g = p
    # #     plt.show(block=True)
