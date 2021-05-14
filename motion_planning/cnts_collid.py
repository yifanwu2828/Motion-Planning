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


class MotionValidator:
    def __init__(self, si, boundary, blocks, rad):
        self.si = si
        self.boundary = boundary
        self.blocks = blocks
        self.rad = rad
        self.params = {}
        self.geoms = []


    def checkMotion(self, s1, s2):
        state = np.array([s1[0], s1[1], s1[2]])
        next_state = np.array([s2[0], s2[1], s2[2]])

        g1 = fcl.Sphere(self.rad)
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
                gjk_solver_type=1,
            )
            result = fcl.ContinuousCollisionResult()
            request.gjk_solver_type = 1
            ret = fcl.continuousCollide(o1, t1_final, o2, fcl.Transform(), request, result)
            motion_collide = result.is_collide
            if motion_collide:
                ic(motion_collide)
                valid = False
                print(f"Agent collide with block: {k}")
                print(f"Total blocks: {blocks.shape[0] - 1}")
                ic(blocks[k, :])
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


if __name__ == '__main__':
    env_id = "flappy_bird"
    for env_id in MAP_SE:
        print(f"\n############ {env_id} ############ ")
        map_file = MAPDICT[env_id]
        start, goal = MAP_SE[env_id]
        ic(start)
        ic(goal)
        boundary, blocks = utils.load_map(map_file)
        fig, ax, hb, hs, hg = utils.draw_map(boundary, blocks, start, goal)

        MV = MotionValidator(None, boundary, blocks, rad=0.1)
        MV.checkMotion(start, goal)
        path =[start, goal]
        path = np.vstack(path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], 'r-')
        plt.show(block=True)

