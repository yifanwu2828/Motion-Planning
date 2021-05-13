import os
import numpy as np
import fcl
import matplotlib.pyplot as plt; plt.ion()
from icecream import ic

import utils

if __name__ == '__main__':
    map_path = './maps/'
    mapDict = {file.split('.')[0]: os.path.join(map_path, file) for file in os.listdir(map_path)}
    ic(mapDict)
    map_file = mapDict['monza']

    # start pos and goal
    start = np.array([0.5, 1.0, 4.9])
    goal = np.array([3.8, 1.0, 0.1])
    ic(start)

    boundary, blocks = utils.load_map(map_file)
    ic(boundary)
    ic(blocks)
    utils.draw_map(boundary, blocks, start, goal)

    params = {}

    start_obj = fcl.CollisionObject(fcl.Sphere(0.01), fcl.Transform(start))

    geoms = []
    for idx, blk in enumerate(blocks):
        # Side length of x,y,z
        x, y, z = utils.get_XYZ_length(blk)
        # Centroid position of the block
        centroid = utils.get_centroid(block=blk)
        # Geometry of collision obj
        geom_box = fcl.Box(x, y, z)
        # Transformation(no Rotation only Translation)
        tf_box = fcl.Transform(np.array(centroid))
        params[f'box_obj_{idx}'] = fcl.CollisionObject(geom_box, tf_box)
        geoms.append(geom_box)
    names = list(params.keys())
    objs = list(params.values())

    # Create map from geometry IDs to objects
    geom_id_to_obj = {id(geom): obj for geom, obj in zip(geoms, objs)}
    # Create map from geometry IDs to string names
    geom_id_to_name = {id(geom): name for geom, name in zip(geoms, names)}


    manager = fcl.DynamicAABBTreeCollisionManager()
    manager.registerObjects(objs)
    manager.setup()


    req = fcl.CollisionRequest(num_max_contacts=100, enable_contact=True)
    cdata = fcl.CollisionData(request=req, result=fcl.CollisionResult())

    test = fcl.CollisionObject(fcl.Sphere(0.01), fcl.Transform(centroid))
    # start_obj
    manager.collide(test, cdata, fcl.defaultCollisionCallback)
    print(f'Collision between manager and Agent?: {cdata.result.is_collision}')
    # print('Contacts:')
    # for c in cdata.result.contacts:
    #     print('\tO1: {}, O2: {}'.format(c.o1, c.o2))
    objs_in_collision = set()
    for contact in cdata.result.contacts:
        # Extract collision geometries that are in contact
        coll_geom_0 = contact.o1
        coll_geom_1 = contact.o2

        # Get their names
        coll_names = [geom_id_to_name.get(id(coll_geom_0)), geom_id_to_name.get(id(coll_geom_1), 'start')]
        coll_names = tuple(sorted(coll_names))
        objs_in_collision.add(coll_names)

    for coll_pair in objs_in_collision:
        print(f"Object '{coll_pair[0]}' in collision with object '{coll_pair[1]}'!")


    # Continuous Collision Checking
    g1 = fcl.Box(1, 2, 3)
    t1 = fcl.Transform()
    o1 = fcl.CollisionObject(g1, t1)
    t1_final = fcl.Transform(np.array([1.0, 0.0, 0.0]))

    g2 = fcl.Cone(1, 3)
    t2 = fcl.Transform()
    o2 = fcl.CollisionObject(g2, t2)
    t2_final = fcl.Transform(np.array([-1.0, 0.0, 0.0]))

    request = fcl.ContinuousCollisionRequest()
    result = fcl.ContinuousCollisionResult()

    ret = fcl.continuousCollide(o1, t1_final, o2, t2_final, request, result)