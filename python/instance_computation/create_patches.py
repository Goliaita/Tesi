import argparse
import numpy as np
import os
from PIL import Image
# from utils.create_labels import create_labels
from utils.net_leaf import NetLeaf

from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Extrapolate and manipulate images to obtain leaf dimension')
    parser.add_argument("--detection_path", default="../yolov3/runs/train/exp15/weights/best.pt", type=str,
                    help="Path to pytorch object detection weights")

    parser.add_argument("--instance_path", default="../yolact/weights/komatsuna_38_12000.pth", type=str,
                    help="Path to pytorch instance segmentation weights")

    parser.add_argument("--compute_label", default=False, action="store_true",
                    help="Chose if you are computing new boxes or not")

    parser.add_argument("--image", default="../dataset/rgbd_original/rgb_002_01.png", type=str,
                    help="Path to image to evaluate leaf dimension")

    parser.add_argument("--images", default="", type=str,
                    help="Path to folder to evaluate leaf dimension")

    
    return parser.parse_args(argv)

net_leaves = NetLeaf()

if __name__=="__main__":
    args = parse_args()

    args.weights        = args.detection_path
    args.return_txt     = True
    args.source         = args.image
    args.view_img       = False
    args.img_size       = 640
    args.conf_thres     = 0.25
    args.iou_thres      = 0.45
    args.max_det        = 1000
    args.device         = "0"
    args.save_txt       = False
    args.save_conf      = False
    args.save_crop      = False
    args.nosave         = False
    args.classes        = 0
    args.agnostic_nms   = False
    args.augment        = False
    args.update         = False
    args.project        = "./runs/detect"
    args.name           = ""
    args.exist_ok       = False
    args.line_thickness = 3
    args.hide_labels    = False
    args.hide_conf      = False

    if args.compute_label:
        if args.images != "":
            image = args.images
        else:
            image = args.image
        
        if args.name != "":
            print()
            os.system(f"python ../yolov3/detect.py --source {image} --weights {args.detection_path} \
                --conf 0.25 --save-txt --project {args.project} --name {args.name}")

        else:    
            os.system(f"python ../yolov3/detect.py --source {image} --weights {args.detection_path} \
                --conf 0.25 --save-txt --project {args.project}")


    if args.name == "":
        path_name = str(sorted(Path(args.project).iterdir(), key=os.path.getmtime)[-1]).split("/")[-1]
    else:
        path_name = args.name

    labels_path = f"{path_name}/labels"

    labels_list = os.listdir(f"{args.project}/{labels_path}")

    boxes_list = []
    images = []

    cropped_path = f"{args.project}/{path_name}/cropped"
    cropped_instance_path = f"{args.project}/{path_name}/instances"
    try:
        os.mkdir(cropped_path)
        os.mkdir(cropped_instance_path)
    except:
        print("Not created, maybe it already exist?")

    
    boxes = []

    i = 0

    for element in labels_list:
        box_list = np.loadtxt(f"{args.project}/{labels_path}/{element}", dtype=np.float32) # category, center_x, center_y, x_displ, y_displ
        image = element.replace("txt","png")
        if args.images != "":
            img = Image.open(f"{args.images}/{image}")
            images.append(img)
        else:
            img = Image.open(f"{args.image}")
            images.append(img)
        x,y = img.size

        if len(box_list.shape) == 1:
            box_list = box_list.reshape((1,-1))

        for idx, _ in enumerate(box_list):

            net_leaf = NetLeaf()
            
            box_list[idx,1] = int(box_list[idx,1] * x  )
            box_list[idx,3] = int(box_list[idx,3] * x/2) - 1
            box_list[idx,2] = int(box_list[idx,2] * y  )
            box_list[idx,4] = int(box_list[idx,4] * y/2) - 1

            subtract = 5
            add = 5

            if box_list[idx,1]-box_list[idx,3] <= 6 or box_list[idx,2]-box_list[idx,4] <= 6:
                subtract = 0

            if box_list[idx,1]+box_list[idx,3] >= x-6 or box_list[idx,2]+box_list[idx,4] >= y-6:
                add = 0

            x_min = int(box_list[idx,1]-box_list[idx,3]-subtract)
            y_min = int(box_list[idx,2]-box_list[idx,4]-subtract)
            x_max = int(box_list[idx,1]+box_list[idx,3]+add)
            y_max = int(box_list[idx,2]+box_list[idx,4]+add)

            cropped = img.crop((x_min, y_min, x_max, y_max))
            cropped.save(f"{cropped_path}/cropped_image{i}.png")

            net_leaf.id = i
            net_leaf.file_name = f"cropped_image{i}.png"
            net_leaf.path = cropped_path
            net_leaf.width = x_max - x_min
            net_leaf.height = y_max - y_min
            net_leaf.image_name = image
            net_leaf.position = (x_min, y_min)

            net_leaves.append(net_leaf)

            i += 1



    net_leaves.save(f"{cropped_instance_path}", categories=["overlapped", "non_overlapped"])

    exit(0)

