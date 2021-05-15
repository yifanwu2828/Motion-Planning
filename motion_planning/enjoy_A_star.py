import argparse
import os
import pickle
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt;plt.ion()
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

h_dict ={
    1: 'manhattan',
    2: 'euclidean',
    3: 'diagonal',
    4: 'octile'
}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--render", help="Visualize Env", action="store_true", default=True)
    parser.add_argument(
        "--showall",  action="store_true", default=False,
        help="Show ALL Env with 4 heuristic plots(28 plots in total) with render = True",
    )
    parser.add_argument(
        "--eps", help="epsilon of A*", type=float, default=2.0,
        choices=[1.0, 1.5, 2.0]
    )
    parser.add_argument(
        "--h", type=int, default=2,
        choices=[1, 2, 3, 4],
        help="heuristic function types: {1: 'manhattan', 2: 'euclidean', 3: 'diagonal', 4: 'octile'}",
    )
    parser.add_argument(
        "-verb", "--verbose", action="store_true", default=True,
        help="Verbose mode (False: no output, True: INFO)"
    )
    args = parser.parse_args()

    valid_h = [1, 2, 3, 4]
    assert args.eps >= 1, "eps should be a float and >= 1. Choices are [1.0, 1.5, 2.0]"
    assert args.h in valid_h

    if args.eps == 1.0:
        with open(f"./result/A_star_res_eps_1.pkl", 'rb') as f:
            A = pickle.load(f)
    elif args.eps == 1.5:
        with open(f"./result/A_star_res_eps_1.5.pkl", 'rb') as f:
            A = pickle.load(f)
    elif args.eps == 2.0:
        with open(f"./result/A_star_res_eps_2.pkl", 'rb') as f:
            A = pickle.load(f)
    else:
        raise ValueError("Valid eps are [1.0, 1.5, 2.0]")


    for idx, info in enumerate(A):
        env_id = inv_dict[idx]
        map_file = MAPDICT[env_id]
        start, goal = MAP_SE[env_id]
        boundary, blocks = utils.load_map(map_file)

        runtime_lst = []
        max_node_lst = []
        pathlength_lst = []
        for h in range(1, 5):
            path = info[f'{env_id}_{h}_path']
            runtime_lst.append(info[f'{env_id}_{h}_runtime'])
            max_node_lst.append(info[f'{env_id}_{h}_max_node'])
            pathlength_lst.append(info[f'{env_id}_{h}_pathlength'])
            if args.verbose:
                max_node = info[f'{env_id}_{h}_max_node']
                collision = info[f'{env_id}_{h}_collision']
                success = info[f'{env_id}_{h}_success']
                pathlength = info[f'{env_id}_{h}_pathlength']
                print(f"\n<--------------{env_id} Heuristic:{h} {h_dict[h]}---------------->")
                ic(max_node)
                ic(collision)
                ic(success)
                ic(pathlength)
    
            if args.render:
                if h == args.h or args.showall:
                    # Original Plot
                    fig, ax, hb, hs, hg = utils.draw_map(boundary, blocks, start, goal)
                    ax.plot(path[:, 0], path[:, 1], path[:, 2], 'r-')
                    plt.title(f"Path Planing for Original {env_id} Env"
                              f" using A* with eps: {args.eps} and heuristic: {h_dict[h]} distance  ")
                    plt.show(block=True)

        print(f"\nBest Heuristic for {env_id}")
        Heuristic_least_runtime = h_dict[int(np.argmin(np.array(runtime_lst))) + 1]
        Heuristic_least_node = h_dict[int(np.argmin(np.array(max_node_lst))) + 1]
        Heuristic_least_pathlength = h_dict[int(np.argmin(np.array(pathlength_lst)))+ 1]

        ic(runtime_lst)
        ic(max_node_lst)
        ic(pathlength_lst)

        ic(Heuristic_least_runtime)
        ic(Heuristic_least_node)
        ic(Heuristic_least_pathlength)