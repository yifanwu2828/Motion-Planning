import numpy as np
from icecream import ic


class MyPlanner:
    __slots__ = ['boundary', 'blocks']

    def __init__(self, boundary, blocks):
        self.boundary = boundary
        self.blocks = blocks
        ic(self.boundary.shape)
        ic(self.blocks.shape)


    def plan(self, start: np.ndarray, goal: np.ndarray):
        path = [start]
        num_dirs = 26
        [dX, dY, dZ] = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])
        dR = np.vstack((dX.flatten(), dY.flatten(), dZ.flatten()))
        dR = np.delete(dR, 13, axis=1)
        dR = dR / np.sqrt(np.sum(dR ** 2, axis=0)) / 2.0

        for _ in range(2_000):
            min_dist2goal = 1_000_000
            node = None
            for i in range(num_dirs):
                next_node = path[-1] + dR[:, i]

                # Check if this direction is valid
                dir_cond = (
                    next_node[0] < self.boundary[0, 0],
                    next_node[0] > self.boundary[0, 3],

                    next_node[1] < self.boundary[0, 1],
                    next_node[1] > self.boundary[0, 4],

                    next_node[2] < self.boundary[0, 2],
                    next_node[2] > self.boundary[0, 5],
                )
                if any(dir_cond):
                    continue
                # if (next_node[0] < self.boundary[0, 0] or next_node[0] > self.boundary[0, 3] or
                #         next_node[1] < self.boundary[0, 1] or next_node[1] > self.boundary[0, 4] or
                #         next_node[2] < self.boundary[0, 2] or next_node[2] > self.boundary[0, 5]):
                #     continue

                valid = True
                for k in range(self.blocks.shape[0]):
                    cond = (
                        self.blocks[k, 0] < next_node[0] < self.blocks[k, 3],  # [x_min, x_max]
                        # self.blocks[k, 0] < next_node[0],
                        # self.blocks[k, 3] > next_node[0],

                        self.blocks[k, 1] < next_node[1] < self.blocks[k, 4],  # [y_min, y_max]
                        # self.blocks[k, 1] < next_node[1],
                        # self.blocks[k, 4] > next_node[1],

                        self.blocks[k, 2] < next_node[2] < self.blocks[k, 5],  # [z_min, z_max]
                        # self.blocks[k, 2] < next_node[2],
                        # self.blocks[k, 5] > next_node[2],
                    )
                    if all(cond):
                        valid = False
                        break
                    # if (self.blocks[k, 0] < next_node[0] < self.blocks[k, 3] and
                    #         self.blocks[k, 1] < next_node[1] < self.blocks[k, 4] and
                    #         self.blocks[k, 2] < next_node[2] < self.blocks[k, 5]):
                    #     valid = False
                    #     break

                if not valid:
                    continue

                # Update next_node
                dist2goal = sum((next_node - goal) ** 2)
                if dist2goal < min_dist2goal:
                    min_dist2goal = dist2goal
                    node = next_node

            if node is None:
                break

            path.append(node)

            # Check if done
            if sum((path[-1] - goal) ** 2) <= 0.1:
                break

        return np.array(path)
