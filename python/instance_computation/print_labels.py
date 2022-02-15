import pycocotools.mask as mk
import numpy as np
import matplotlib.pyplot as plt
import json
from skimage.measure import *
import math
from utils.compute_dimension import Reparametrize
from sklearn.linear_model import RANSACRegressor

import open3d as o3d


rp = Reparametrize()

from PIL import Image

with open("runs/detect/exp16/instances/inference/coco_instances_results.json") as f:
    instances = json.load(f)

with open("runs/detect/exp16/instances/position.json") as f:
    position = json.load(f)

with open("../dataset/dataset_ours/test/annotations/dataset_ours.json") as f:
    gt_data = json.load(f)

img = Image.open("../dataset/dataset_ours/test/color/color_000.png")

fig = plt.figure()
plt.imshow(img, interpolation='none')
plt.xticks([])
plt.yticks([])

# print(img.height)
computed_mask = np.array(())
dimensions = np.array(())
distances = np.array(())
img_pil = Image.open(f"../dataset/dataset_ours/test/depth/depth_000.png")

depth = np.array(list(img_pil.getdata())).reshape(
            (img_pil.width, img_pil.height)).astype(np.uint16)

pc_img = np.ones((sum(depth.shape), 4))

for i, j in zip(range(depth.shape[0]), range(depth.shape[1])):
    pc_img[i+j, :3] = rp.compute_dimension(i, j, depth[i, j], 1)

img = o3d.io.read_image("../dataset/dataset_ours/test/depth/depth_000.png")
pcd = o3d.geometry.PointCloud.create_from_depth_image(
    img,
    o3d.camera.PinholeCameraIntrinsic(
        1280, 720, 637.735, 637.735, 645.414, 349.205))
# # Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


xyz = np.asarray(pcd.points)
ransac = RANSACRegressor(residual_threshold=0.01)

ransac.fit(xyz[:, :2],xyz[:, 2])
a, b = ransac.estimator_.coef_  # coefficients
d = ransac.estimator_.intercept_  # intercept



for idx, element in enumerate(instances):
    image = np.zeros((720,1280))
    if (element["image_id"] == 1 or element["image_id"] == 2 or element["image_id"] == 3 or element["image_id"] == 4) and element["score"] >= 0.8:
        segm = mk.decode(element["segmentation"])
        points = zip(*np.where(segm == 1))

        for i in range(segm.shape[0]):
            for j in range(segm.shape[1]):
                image[position["info"][element["image_id"]]["description"][1] + i, position["info"][element["image_id"]]["description"][0] + j] = segm[i,j]
        regions = regionprops(segm)
        y0, x0 = regions[0].centroid
        y0 = int(y0)
        x0 = int(x0)
        x0 += position["info"][element["image_id"]]["description"][0]
        y0 += position["info"][element["image_id"]]["description"][1]
        computed_mask = np.append(computed_mask, [x0,y0])
        masked = np.ma.masked_where(image == 0, image)

        orientation = regions[0].orientation

        x1 = x0 + int(math.cos(orientation) * regions[0].minor_axis_length / 2)
        y1 = y0 - int(math.sin(orientation) * regions[0].minor_axis_length / 2)
        x2 = x0 - int(math.sin(orientation) * regions[0].major_axis_length / 2)
        y2 = y0 - int(math.cos(orientation) * regions[0].major_axis_length / 2)
        x3 = x0 - int(math.cos(orientation) * regions[0].minor_axis_length / 2)
        y3 = y0 + int(math.sin(orientation) * regions[0].minor_axis_length / 2)
        x4 = x0 + int(math.sin(orientation) * regions[0].major_axis_length / 2)
        y4 = y0 + int(math.cos(orientation) * regions[0].major_axis_length / 2)

        X1, Y1, Z1 = rp.compute_dimension(x1, y1, depth[x1, y1], 1)
        X2, Y2, Z2 = rp.compute_dimension(x2, y2, depth[x2, y2], 1)
        X3, Y3, Z3 = rp.compute_dimension(x3, y3, depth[x3, y3], 1)
        X4, Y4, Z4 = rp.compute_dimension(x4, y4, depth[x4, y4], 1)

        add = 0

        mean = 0

        if 0.5 < Z1 < 0.8:
            mean += Z1
            add += 1

        if 0.5 < Z2 < 0.8:
            mean += Z2
            add += 1

        if 0.5 < Z3 < 0.8:
            mean += Z3
            add += 1
        
        if 0.5 < Z4 < 0.8:
            mean += Z4
            add += 1

        if add != 0:
            mean = mean / add
        else:
            mean = (Z1+Z2+Z3+Z4)/4

        if 0.5 > Z1 > 0.8 and add != 0:
            X1, Y1, Z1 = rp.compute_dimension(x1, y1, mean * 1000, 1)

        if 0.5 > Z2 > 0.8 and add != 0:
            X2, Y2, Z2 = rp.compute_dimension(x2, y2, mean * 1000, 1)

        if 0.5 > Z3 > 0.8 and add != 0:
            X3, Y3, Z3 = rp.compute_dimension(x3, y3, mean * 1000, 1)

        if 0.5 > Z4 > 0.8 and add != 0:
            X4, Y4, Z4 = rp.compute_dimension(x4, y4, mean * 1000, 1)

        distance = list(rp.compute_dimension(x0, y0, mean * 1000, 1))

        distance.append(1 if add != 0 else 0)

        distances = np.append(distances, distance)


        minor = np.sqrt(np.square(X1 - X3) + np.square(Y1 - Y3))
        major = np.sqrt(np.square(X2 - X4) + np.square(Y2 - Y4))
        dimensions = np.append(dimensions, [major, minor])

        if x3-x1 != 0:
            m = (y3-y1)/(x3-x1)
            q = y1-m*x1
            ranges = np.arange(x3,x1,0.1)
            plt.plot(ranges, m*ranges+q, c="red")
        
        if x4-x2 != 0:
            m = (y4-y2)/(x4-x2)
            q = y4-m*x4
            ranges = np.arange(x2,x4,0.1)
            plt.plot(ranges, m*ranges+q, c="blue")


        plt.imshow(masked, cmap="Blues", interpolation="none", alpha=0.3)
        plt.scatter(x0,y0, s=2, c='red', marker='o')

computed_mask = computed_mask.reshape((-1,2))
dimension = dimensions.reshape((-1,2))
distances = distances.reshape((-1,4))


ptdiff = lambda p1, p2: (p1[0]-p2[0], p1[1]-p2[1])

for idx, element in enumerate(gt_data["annotations"]):

    if element["category_id"] == 2:
        distance = 10
        h = gt_data["images"][0]["height"]
        w = gt_data["images"][0]["width"]
        mask = mk.merge(mk.frPyObjects(element["segmentation"], h, w))
        segm = mk.decode(mask)
        regions = regionprops(segm)
        y0, x0 = regions[0].centroid
        found = False
        for idy, point in enumerate(computed_mask):
            computed_distance = math.hypot(*ptdiff([x0,y0], point))
            if distance >= computed_distance:
                distance = computed_distance
                found = True
                indx = idy
        if found:
            x1 = x0 + int(math.cos(orientation) * regions[0].minor_axis_length / 2)
            y1 = y0 - int(math.sin(orientation) * regions[0].minor_axis_length / 2)
            x2 = x0 - int(math.sin(orientation) * regions[0].major_axis_length / 2)
            y2 = y0 - int(math.cos(orientation) * regions[0].major_axis_length / 2)
            x3 = x0 - int(math.cos(orientation) * regions[0].minor_axis_length / 2)
            y3 = y0 + int(math.sin(orientation) * regions[0].minor_axis_length / 2)
            x4 = x0 + int(math.sin(orientation) * regions[0].major_axis_length / 2)
            y4 = y0 + int(math.cos(orientation) * regions[0].major_axis_length / 2)

            if x3-x1 != 0:
                m = (y3-y1)/(x3-x1)
                q = y1-m*x1
                ranges = np.arange(x3,x1,0.1)
                plt.plot(ranges, m*ranges+q, c="green")
            
            if x4-x2 != 0:
                m = (y4-y2)/(x4-x2)
                q = y4-m*x4
                ranges = np.arange(x2,x4,0.1)
                plt.plot(ranges, m*ranges+q, c="cyan")
            
            height_error = None

            if 'height' in element['metadata'].keys() and distances[indx,3]:
                height = float(element['metadata']['height'].replace(",","."))

                computed_height = - ransac.predict(distances[indx,:2].reshape(1, -1))[0] - distances[indx,2]
                
                height_error = np.abs(computed_height - height)

            major = float(element['metadata']['axis_major'].replace(",","."))
            minor = float(element['metadata']['axis_minor'].replace(",","."))
            difference_major = np.abs(major - dimension[indx,0])
            difference_minor = np.abs(minor - dimension[indx,1])
            print(f"The difference in major is: {difference_major}, in minor: {difference_minor}, in height: {height_error}")
            
            masked = np.ma.masked_where(segm == 0, segm)
            plt.imshow(masked, cmap="autumn", interpolation="none", alpha=0.3)
            plt.scatter(x0,y0, s=2, c='blue', marker='o')

fig.savefig("./prova.png")
plt.close("all")
