import os
import time
from typing import Optional, Tuple
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

import fcl

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from main import *
import utils
import Planner


RAD = 0.05


def main():
    params = OrderedDict()
    print('Running single cube test...\n')
    start = np.array([0.5, 1.0, 4.9])
    goal = np.array([3.8, 1.0, 0.1])

    map_file = './maps/monza.txt'

    # success, pathlength = run_test(map_file, start, goal, verbose=True)
    # print('Success: %r' % success)
    # print('Path length: %d' % pathlength)
    # print('\n')
    boundary, blocks = load_map(map_file)

    fig, ax, hb, hs, hg, = draw_map(boundary, blocks, start, goal)
    plt.show(block=True)


    # Create object
    # assume start and goal are sphere, blocks are box
    # all objects initially at origin and translate to pos
    geom_start = fcl.Sphere(RAD)
    geom_goal = fcl.Sphere(RAD)
    params['start_obj'] = fcl.CollisionObject(geom_start, fcl.Transform(start))
    params['goal_obj'] = fcl.CollisionObject(geom_goal, fcl.Transform(goal))

    geoms = []
    for i, blk in enumerate(blocks):
        geom_box = fcl.Box(*utils.get_XYZ_length(blk))
        tf_box = fcl.Transform(np.array(utils.get_centroid(block=blk)))
        params[f'box_obj_{i}'] = fcl.CollisionObject(geom_box, tf_box)
        geoms.append(geom_box)
    names = list(params.keys())
    objs = list(params.values())

    # Create map from geometry IDs to objects
    geom_id_to_obj = {id(geom): obj for geom, obj in zip(geoms, objs)}

    # Create map from geometry IDs to string names
    geom_id_to_name = {id(geom): name for geom, name in zip(geoms, names)}


    # Managed one to many collision checking
    manager = fcl.DynamicAABBTreeCollisionManager()
    manager.registerObjects(objs)
    manager.setup()

    req = fcl.CollisionRequest(num_max_contacts=100, enable_contact=True)
    cdata = fcl.CollisionData(request=req)

    manager.collide(params['start_obj'], cdata, fcl.defaultCollisionCallback)
    print(f'Collision between manager 1 and agent: {cdata.result.is_collision}')

    # Extract collision data from contacts and use that to infer set of
    # objects that are in collision
    objs_in_collision = set()
    for contact in cdata.result.contacts:
        # Extract collision geometries that are in contact
        coll_geom_0 = contact.o1
        coll_geom_1 = contact.o2
        print(f'\tO1: {contact.o1}, O2: {contact.o2}')

        # Get their names
        coll_names = [geom_id_to_name[id(coll_geom_0)], geom_id_to_name[id(coll_geom_1)]]
        coll_names = tuple(sorted(coll_names))
        objs_in_collision.add(coll_names)

    for coll_pair in objs_in_collision:
        print(f'Object {coll_pair[0]} in collision with object {coll_pair[1]}!')


if __name__ == '__main__':
    pass

