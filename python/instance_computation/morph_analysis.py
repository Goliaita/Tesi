from PIL import Image
from matplotlib.pyplot import minorticks_off
from pycocotools import mask as mk
import math
import json
from skimage.measure import *
import numpy as np
from utils.compute_dimension import Reparametrize

import pyvista as pv

import pyransac3d as pyrsc
from utils.leaf import Leaf


test_leaves = Leaf(f_type="COCO")
test_leaves.from_coco_string("../dataset/dataset_ours/test/annotations/dataset_ours.json")
# print(test_leaves.leaves_list)
# for i in range(len(test_leaves.leaves_list)):
#     print(f"fount at element {i}, {test_leaves.leaves_list[i].axis_major}, or {test_leaves.get_leaf(i).axis_major}")

file = open("runs/detect/exp15/instances/inference/coco_instances_results.json")
positions_file = open("runs/detect/exp15/instances/position.json")

_names = json.load(open("runs/detect/exp15/instances/labels.json"))
json_data = json.load(file)
positions = json.load(positions_file)


file.close()
positions_file.close()

print(f"Total masks to analyze {len(json_data)}")

masks = []
i_test = 0

rp = Reparametrize()

leaves_list = Leaf()

id = 0

for idx, mask in enumerate(json_data):
    current_leaf = Leaf()
    if idx % 100 == 0:
        print(f"Analyzed {idx} masks")
    if mask["category_id"] == 2 and mask["score"] > 0.8:
        position = [0, 0]
        current_leaf.score = mask["score"]
        for position_ in positions["info"]:
            if position_["id"] == mask["image_id"]:
                for _ in _names["images"]:
                    if position_["id"] == _["id"]:
                        current_leaf.image_name = _["image_name"]
                        current_leaf.image_id = mask["image_id"]
                        current_leaf.id = id
                        image_depth_name = _["image_name"].replace("color", "depth")
                        id += 1
                position = position_["description"]

        current_leaf.position = mask["segmentation"]
        decoded_position = mk.decode(mask["segmentation"])
        json_data[idx]["position"] = position
        points = zip(*np.where(decoded_position == 1))
        area = []
        regions = regionprops(decoded_position)
        

        y0, x0 = regions[0].centroid
        x0 = int(x0)
        y0 = int(y0)

        current_leaf.centroids = [x0, y0]

        orientation = regions[0].orientation

        x1 = x0 + int(math.cos(orientation) * regions[0].minor_axis_length / 2)
        y1 = y0 - int(math.sin(orientation) * regions[0].minor_axis_length / 2)
        x2 = x0 - int(math.sin(orientation) * regions[0].major_axis_length / 2)
        y2 = y0 - int(math.cos(orientation) * regions[0].major_axis_length / 2)
        x3 = x0 - int(math.cos(orientation) * regions[0].minor_axis_length / 2)
        y3 = y0 + int(math.sin(orientation) * regions[0].minor_axis_length / 2)
        x4 = x0 + int(math.sin(orientation) * regions[0].major_axis_length / 2)
        y4 = y0 + int(math.cos(orientation) * regions[0].major_axis_length / 2)

        current_leaf.box_area = np.sqrt(np.square(x1-x3)+np.square(y1-y3)) * \
            np.sqrt(np.square(x2-x4)+np.square(y2-y4))

        img_pil = Image.open(f"../dataset/dataset_ours/test/depth/{image_depth_name}")

        depth = np.array(list(img_pil.getdata())).reshape(
            (img_pil.width, img_pil.height)).astype(np.uint16)

        pc_img = np.ones((sum(depth.shape), 4))

        for i, j in zip(range(depth.shape[0]), range(depth.shape[1])):
            pc_img[i+j, :3] = rp.compute_dimension(i, j, depth[i, j], 1)

        regressor = pyrsc.Plane()
        regressor.fit(pc_img[:, :3], thresh=0.2, maxIteration=10000)

        for i, j in points:
            x = i + int(position[0])
            y = j + int(position[1])
            
            distance = depth[x,y]
            
            temp_x = x
            temp_y = y
            while distance == 0:
                x -= 1
                if x <= 0:
                    x = temp_x
                    y -= 1
                # try:
                distance = depth[x,y]
                # except:
                    # print(f"found {y}, {x} with position at {position}")
            area.append(rp.compute_dimension(x, y, distance, 1))

        X1, Y1, Z1 = rp.compute_dimension(x1, y1, depth[y1, x1], 1)
        X2, Y2, Z2 = rp.compute_dimension(x2, y2, depth[y2, x2], 1)
        X3, Y3, Z3 = rp.compute_dimension(x3, y3, depth[y3, x3], 1)
        X4, Y4, Z4 = rp.compute_dimension(x4, y4, depth[y4, x4], 1)

        x_axis = np.sqrt(np.square(X1 - X3) + np.square(Y1 - Y3) + np.square(Z1 - Z3))
        y_axis = np.sqrt(np.square(X2 - X4) + np.square(Y2 - Y4) + np.square(Z2 - Z4))
    
        if x_axis > y_axis:
            current_leaf.axis_major = x_axis
            current_leaf.axis_minor = y_axis
        else:
            current_leaf.axis_major = y_axis
            current_leaf.axis_minor = x_axis

        ptdiff = lambda p1, p2: (p1[0]-p2[0], p1[1]-p2[1])
        distance = 100
        indx = 0
        found = False
        for idx, test in enumerate(test_leaves.leaves_list):
            if test.axis_major != 0.0 and current_leaf.image_name == test.image_name:
                centroids_gt = [int(test.centroids[0]), int(test.centroids[1])]
                centroids_mk = [current_leaf.centroids[0] + int(position[0]), current_leaf.centroids[1] + int(position[1])]
                # print(f"{test.image_name} found centered in {centroids_gt} and analyzing at {centroids_mk}")
                dist = math.hypot(*ptdiff(centroids_gt, centroids_mk))
                if dist <= distance:
                    distance = dist
                    indx = idx
                    found = True
        
        if found:
            print(indx, len(test_leaves.leaves_list), test_leaves.leaves_list[indx].axis_major, current_leaf.axis_major)
            major_diff = (test_leaves.get_leaf(indx).axis_major - current_leaf.axis_major) #* test_leaves.get_leaf(indx).axis_major
            minor_diff = (test_leaves.get_leaf(indx).axis_minor - current_leaf.axis_minor) #* test_leaves.get_leaf(indx).axis_minor
            print(f"The difference between real dimension and computed one is: {major_diff}, {minor_diff}")

        decoded_position = np.asarray(area)

        current_leaf.leaf_height = max(decoded_position[:,2]) - min(decoded_position[:,2])

        cloud = pv.PolyData(area)
        surf = cloud.delaunay_2d()

        current_leaf.area = surf.area

        max_value = np.argmin(decoded_position[:, 2])

        current_leaf.leaf_floor_height = np.abs(np.dot(regressor.equation, np.concatenate((decoded_position[max_value, :], [1]))).sum()) \
            / (np.sqrt(np.square(regressor.equation[0]) + np.square(regressor.equation[1]) + np.square(regressor.equation[2])))

        # leaves_list.append(current_leaf)

# leaves_list.save()
