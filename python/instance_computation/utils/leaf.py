import numpy as np
import copy
import json

class Leaf:

    leaves_list = []

    def __init__(self, id:int = 0, image_id:int = 0, leaf_floor_height:float = 0., position:np.array = np.empty(0), \
        axis_major:float = 0., axis_minor:float = 0., leaf_height:float = 0., area:float = 0., \
            box_area:int = 0, centroids:list = [], image_name:str = "", score:float = 0.):

        self.id                 = id
        self.image_id           = image_id
        self.leaf_floor_height  = leaf_floor_height
        self.position           = position
        self.axis_major         = axis_major
        self.axis_minor         = axis_minor
        self.leaf_height        = leaf_height
        self.area               = area
        self.box_area           = box_area
        self.centroids          = centroids
        self.image_name         = image_name
        self.score              = score

    def append(self, leaf):
        self.leaves_list.append(leaf)

    def get_leaf(self, idx):
        return self.leaves_list[idx]


    def get_json_list(self):
        images = {"leaves": []}

        for element in self.leaves_list:
            temp = element.copy()
            temp.position = temp.position.tolist()
            images["leaves"].append(temp.__dict__)

        return images


    def copy(self):
        return copy.copy(self)

    def save(self, path="runs/detect/exp14/instances/metrics.json"):
        with open(path, 'w+') as f:
            json.dump(self.get_json_list(), f)
