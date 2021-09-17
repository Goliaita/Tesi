import json
import argparse
import os
import traceback
import random
import sys

import numpy as np
import matplotlib.pyplot as plt





parser = argparse.ArgumentParser(
    description='Transform labeled images into coco format')
parser.add_argument("--path_label", type=str, required=False, help="Path to labeled images", default="./images/")
parser.add_argument("--path_images", type=str, required=True, help="Path to images")
parser.add_argument("--out_path", type=str, required=True, help="Path where save the file")
parser.add_argument("--saved_data", type=str, required=False, help="Path where are saved the segmentation about plants",
    default=None)
parser.add_argument("--mpi", type=bool, required=False,
    help="Run the program as a parallel version (must run with mpiexec) (only for the reading part)",
    default=False)

args = parser.parse_args()

if args.mpi:
    try:
        from mpi4py import MPI
        mpi = True
    except Exception:
        traceback.print_exc()
        print("Error no mpi package found running sequential mode")
        mpi = False
else:
    mpi = False

def order_segmentation(segmented_array, index):
    array_in = segmented_array.copy()
    array_in = array_in.reshape((-1,2))
    array_buff = []
    array_buff = np.append(array_buff, [segmented_array[0], segmented_array[1]])
    array_in = np.delete(array_in, 0, axis=0)

    for i in range(2, len(segmented_array), 2):
        counter = 100

        for k in range(array_in.shape[0]-1):
            try:
                buff = np.linalg.norm([array_buff[i-2]-array_in[k,0],array_buff[i-1]-array_in[k,1]])
            except Exception:
                traceback.print_exc()
                if mpi:
                    print("error from rank: {} at {} and {} with array_in: {}, array_buff: {}".format(rank, k, i, array_in.shape, array_buff.shape))    
                else:
                    print("error at index: {}, with element {} and {} with array_in: {}, array_buff: {}"
                        .format(index, k, i, array_in.shape, array_buff.shape))
                    print(next)
                sys.stdout.flush()
                exit()

            if counter > buff:
                counter = buff
                next = [array_in[k,0], array_in[k,1]]
                to_remove = k

        array_in = np.delete(array_in, to_remove, axis=0)

        array_buff = np.append(array_buff, next)


    return array_buff


def coco_label_creator(label_path: str, ids=None, rank=0):
    listed_images = os.listdir(label_path)
    
    json_image = []
    image_colors = []
    if ids is not None:
        listed_images = listed_images[ids[0]:ids[1]]

    l = 0
    for img_name in listed_images:
        img = plt.imread(label_path + "/" + img_name)
        colors = []
        for i in range(img.shape[0]-2):
            for j in range(img.shape[1]-2):
                try: 
                    if (   img[i,j,0] != img[i,j+1,0] or img[i,j,0] != img[i,j-1,0] \
                        or img[i,j,0] != img[i+1,j,0] or img[i,j,0] != img[i-1,j,0] \
                        or img[i,j,1] != img[i,j+1,1] or img[i,j,1] != img[i,j-1,1] \
                        or img[i,j,1] != img[i+1,j,1] or img[i,j,1] != img[i-1,j,1] \
                        or img[i,j,2] != img[i,j+1,2] or img[i,j,2] != img[i,j-1,2] \
                        or img[i,j,2] != img[i+1,j,2] or img[i,j,2] != img[i-1,j,2] ) and img[i,j].any():
                        color = None

                        if len(colors) == 0:
                            colors.append(img[i,j])
                            color = 0
                        else:
                            for k in range(len(colors)):
                                if colors[k][0] == img[i,j,0] and colors[k][1] == img[i,j,1] and colors[k][2] == img[i,j,2]:
                                    color = k
                                
                            if color == None:
                                colors.append(img[i,j])
                                color = len(colors) - 1
                                
                        json_image = np.append(json_image, (l, i, j, color), axis=0)

                except:
                    print("Error at image {}".format(img_name)) 
                    exit()

        image_colors = np.append(image_colors, len(colors))
        if (l % 10) == 0:
            print("Rank: {}, Examinated {} files".format(rank, l))
            sys.stdout.flush()
        l += 1

    return json_image, image_colors


def compose_json(labels, colors, img_path):

    composed_json                   = {}
    composed_json["images"]         = []
    composed_json["categories"]     = []
    composed_json["annotations"]    = []

    dataset_id = 1
    category_ids = []
    annotated = False
    annotating = []
    num_annotations = 0
    metadata = {}
    deleted = False
    milliseconds = 0
    events = []
    regenerate_thumbnail = False

    

    composed_json["categories"].append({
        "id": 2,
        "name": "leaf",
        "supercategory": "",
        "color": "#78ef51",
        "metadata": {},
        "keypoint_colors": []
    })

    print("Added the categories")
    sys.stdout.flush()

    listed_imgs = os.listdir(img_path)
    id = 0
    for img in listed_imgs:

        image = plt.imread(img_path + "/" + img)

        compose_json = {
            "id": id,
            "dataset_id": dataset_id,
			"category_ids": category_ids,
			"path": img_path + img,
			"width": image.shape[0],
			"height": image.shape[1],
			"file_name": img,
			"annotated": annotated,
			"annotating": annotating,
			"num_annotations": num_annotations,
			"metadata": metadata,
			"deleted": deleted,
			"milliseconds": milliseconds,
			"events": events,
			"regenerate_thumbnail": regenerate_thumbnail
        }
        composed_json["images"].append(compose_json)

        id += 1
    
    max_id = int(labels[labels.shape[0]-1,0]+1)
    id = 0

    print("Added all images")
    sys.stdout.flush()

    for i in range(max_id):
        for j in range(int(colors[i])):
            segmentation = []
            max_x = 0
            max_y = 0
            min_x = composed_json["images"][i]["width"]
            min_y = composed_json["images"][i]["height"]
            for label in labels:
                if i == label[0] and j == label[3]:
                    segmentation = np.append(segmentation, [[label[2], label[1]]])
                    if min_x > label[1]:
                        min_x = label[1]
                    if max_x < label[1]:
                        max_x = label[1]
                    if min_y > label[2]:
                        min_y = label[2]
                    if max_y < label[2]:
                        max_y = label[2]
            
            random_number = random.randint(0,16777215)
            hex_number = str(hex(random_number))
            hex_number ='#'+ hex_number[2:]
            segmentation = order_segmentation(segmentation, i)

            buff_json = {
                "id": id,
                "image_id": i,
                "category_id": 2,
                "segmentation": segmentation.reshape((1,-1)).tolist(),
                "area": (max_x-min_x) * (max_y - min_y),
                "bbox": [max_x, max_y, min_x, min_y],
                "iscrowd": False,
                "isbbox": False,
                "color": hex_number,
                "metadata": {}
            }
            id += 1
            composed_json["annotations"].append(buff_json)

    return composed_json

if __name__=="__main__":
    if args.saved_data is None:

        if mpi:
            world = MPI.COMM_WORLD
            agents_number = world.Get_size()
            rank = world.Get_rank()
            listed = os.listdir(args.path_label)
            portion = len(listed) / agents_number
            ids = [int(rank * portion), int((rank * portion) + portion)]
            print("Agent ", rank, " got ", portion, " rows of dataset")
            sys.stdout.flush()
        else:
            ids = None
            rank = 0
        
        
        label, colors = coco_label_creator(args.path_label, ids, rank)

        if mpi:
            print("Rank: {} finished to extract shape from images".format(rank))
        else:
            print("Finished to extract shape from images")

        sys.stdout.flush()

        label = label.reshape((-1, 4))

        if mpi:
            world.Barrier()
            results = world.gather(label, root=0) 
            results_colors = world.gather(colors, root=0)
            
            if rank == 0:
                sys.stdout.flush()
                merge_data = []
                merge_colors = []
                for j in results:
                    merge_data = np.append(merge_data, j)
                for j in results_colors:
                    merge_colors = np.append(merge_colors, j)
                merge_data = merge_data.reshape((-1,4))
                merge_colors = merge_colors.reshape((-1,1))


                saving_data = open("./data_saved.txt", "w+")
                saving_colors = open("./colors_saved.txt", "w+")

                saving_data.write(str(merge_data.tolist()))
                saving_colors.write(str(merge_colors.tolist()))
        else:
            saving_data = open("./data_saved.txt", "w+")
            saving_colors = open("./colors_saved.txt", "w+")

            saving_data.write(str(label))
            saving_colors.write(str(colors))

    else: 
        saved_data = open(args.saved_data + "data_saved.txt", "r")
        saved_colors = open(args.saved_data + "colors_saved.txt", "r")

        
        label_buff = saved_data.read()
        colors_buff = saved_colors.read()

        label_buff = label_buff.replace("[", '')
        label_buff = label_buff.replace("]", '')
        label_buff = label_buff.replace(",", '')
        label_buff = label_buff.replace(".0", '')

        colors_buff = colors_buff.replace("[", '')
        colors_buff = colors_buff.replace("]", '')
        colors_buff = colors_buff.replace(",", '')
        colors_buff = colors_buff.replace(".0", '')

        label  = np.asarray(label_buff.split(" "), dtype=int)
        colors = np.asarray(colors_buff.split(" "), dtype=int)


        label = label.reshape(-1, 4)
        print("data loaded")

    if mpi:
        if rank == 0:
            my_json = compose_json(label, colors, args.path_images)

            print("Json composed writing out...")

            sys.stdout.flush()

            coco_file = open(args.out_path + "/komatsuna_coco.json", "w+")

            coco_file.write(json.dumps(my_json))

            print("... Wrote finished the task")
            sys.stdout.flush()
    else:
        my_json = compose_json(label, colors, args.path_images)

        print("Json composed writing out...")

        sys.stdout.flush()

        coco_file = open(args.out_path + "/komatsuna_coco.json", "w+")

        coco_file.write(json.dumps(my_json))

        print("... Wrote finished the task")
        sys.stdout.flush()


