import os
import json
import shutil

with open("../dataset/annotations/MSU-PID/validation300.json") as f:
    data = json.load(f)


if not os.path.isdir("./validation"):
    os.mkdir("./validation")

for element in data["images"]:
    shutil.copyfile(f"/data/basile/Project/AdelaiDet/datasets/coco/train_multi/{element['file_name']}", f"./validation/{element['file_name']}")


