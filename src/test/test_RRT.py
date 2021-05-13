import sys
import os
import time
import random
from collections import OrderedDict

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt; plt.ion()
from icecream import ic

from src.src.rrt.rrt import RRT
from src.src.rrt.rrt_star import RRTStar
from src.src.search_space.search_space import SearchSpace
from src.src.utilities.plotting import Plot


from src.Planner import MyPlanner
from src import utils

MAPDICT = {file.split('.')[0]: os.path.join('../maps/', file) for file in os.listdir('../maps/')}
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

if __name__ == '__main__':
    utils.set_random_seed(42)
    for env_id in tqdm(MAP_SE.keys()):
        map_file = MAPDICT[env_id]
        boundary, blocks = utils.load_map(map_file)
        start, goal = MAP_SE[env_id]
        print(f"##########{env_id}#############")
        # dimensions of Search Space
        X = boundary[0, :6].flatten()
        X_dimensions = np.array([(X[0], X[3]), (X[1], X[4]), (X[2], X[5])])

        # obstacles
        Obstacles = blocks[:, :6].copy()

        x_init = tuple(start)  # starting location
        x_goal = tuple(goal)  # goal location

        Q = np.array([(8, 4)])  # length of tree edges
        r = 10  # length of smallest edge to check for intersection with obstacles
        max_samples = 1024  # max number of samples to take before timing out
        prc = 0.1  # probability of checking for a connection to goal
        rewire_count = 32

        # create Search Space
        X = SearchSpace(X_dimensions, Obstacles)

        # create rrt_search
        # rrt = RRT(X, Q, x_init, x_goal, max_samples, r, prc)
        # rrt_path = rrt.rrt_search()

        rrt = RRTStar(X, Q, x_init, x_goal, max_samples, r, prc, rewire_count)
        rrt_star_path = rrt.rrt_star()

        final_path = rrt_star_path

        fig, ax, hb, hs, hg = utils.draw_map(boundary, blocks, start, goal)
        if final_path is not None:
            path = np.vstack(final_path)
            ax.plot(path[:, 0], path[:, 1], path[:, 2], 'r-')
        else:
            print("NO PATH FOUND")
        plt.title(f"Original {env_id} Env")
        plt.show(block=True)
