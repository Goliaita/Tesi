from __future__ import annotations
import numpy as np
import copy
import json
from pycocotools import mask as mk

from skimage.measure import regionprops




class Leaf:

    leaves_list = []

    def __add__(self, other: Leaf) -> Leaf:
        pass

    def __init__(self, id:int = 0, image_id:int = 0, leaf_floor_height:float = 0., position:np.array = np.empty(0), \
        axis_major:float = 0., axis_minor:float = 0., leaf_height:float = 0., area:float = 0., \
            box_area:int = 0, centroids:list = [], image_name:str = "", score:float = 0., category_id=None, f_type:str=None):

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
        self.category_id        = category_id

        if f_type is not None and f_type == "COCO":
            self.f_type = f_type

    def append(self, leaf):
        self.leaves_list.append(leaf)

    def get_leaf(self, idx) -> Leaf:
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


    def from_coco_string(self, file):
        with open(file) as f:
            jsonize = json.load(f)
        images = jsonize["images"]
        annotations = jsonize["annotations"]

        for element in annotations:
            leaf = Leaf(f_type="COCO")
            for image in images:
                if image["id"] == element["image_id"]:
                    image_name = image["file_name"]
                    h = image["height"]
                    w = image["width"]


            leaf.id = element["id"]
            leaf.image_id = element["image_id"]
            leaf.category_id = element["category_id"]
            leaf.position = mk.merge(mk.frPyObjects(element["segmentation"], h, w))
            if bool(element["metadata"]):
                leaf.axis_major = float(element["metadata"]["axis_major"].replace(",","."))
                leaf.axis_minor = float(element["metadata"]["axis_minor"].replace(",","."))
                # print(f"found a leaf of: {leaf.axis_major}, {leaf.axis_minor}")
            leaf.box_area = element["area"]
            decoded_position = mk.decode(leaf.position)
            
            regions = regionprops(decoded_position)
            leaf.centroids = [regions[0].centroid[0], regions[0].centroid[1]]
            leaf.image_name = image_name
            self.append(leaf)
        

    def save(self, path="runs/detect/exp14/instances/metrics.json"):
        with open(path, 'w+') as f:
            json.dump(self.get_json_list(), f)
