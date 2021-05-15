import argparse
import os
import pickle
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt;plt.ion()
from tqdm import tqdm
from icecream import ic

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

inv_dict = {
    0: 'single_cube',
    1: 'monza',
    2: 'flappy_bird',
    3: 'window',
    4: 'room',
    5: 'maze',
    6: 'tower',
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--render", help="Visualize Env", action="store_true", default=True)

    parser.add_argument(
        "-verb", "--verbose", action="store_true", default=True,
        help="Verbose mode (False: no output, True: INFO)"
    )
    args = parser.parse_args()


    with open(f"./result/RRT_star_res.pkl", 'rb') as f:
           RRT = pickle.load(f)

    for env_id, info in RRT.items():
        map_file = MAPDICT[env_id]
        start, goal = MAP_SE[env_id]
        boundary, blocks = utils.load_map(map_file)
        path = info['path']

        if args.verbose:
            plan_time = info['plan_time']
            collision = info['collision']
            success = info['success']
            pathlength = info['pathlength']
            print(f"\n<--------------{env_id}---------------->")
            ic(plan_time)
            ic(collision)
            ic(success)
            ic(pathlength)

        if args.render:
            # Original Plot
            fig, ax, hb, hs, hg = utils.draw_map(boundary, blocks, start, goal)
            ax.plot(path[:, 0], path[:, 1], path[:, 2], 'r-')
            plt.title(f"Path Planing for Original {env_id} Env"
                      f" using RRT* in {info['plan_time']} sec")
            plt.show(block=True)

