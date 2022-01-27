from utils.net_leaf import NetLeaf
# from utils.create_labels import create_labels
import os
import argparse
from PIL import Image


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Extrapolate COCO format of images in a folder')

    parser.add_argument("--images_path",
                        default="../dataset/bean_test",
                        type=str,
                        help="Path of images")

    parser.add_argument("--store_path",
                        default="test",
                        type=str,
                        help="Path where save the json file")

    parser.add_argument("--categories",
                        nargs="+",
                        type=str,
                        help="Path where save the json file")

    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()

    print(args)

    list_ = os.listdir(args.images_path)

    images_names = {"path": [], "file_name": [], "width": [],
                    "height": [], "position": [], "image_name": []}

    net_leaves = NetLeaf()

    for idx, image in enumerate(list_):

        img = Image.open(f"{args.images_path}/{image}")
        width, height = img.size

        net_leaf = NetLeaf()

        net_leaf.id = idx
        net_leaf.file_name = image
        net_leaf.path = args.images_path
        net_leaf.image_name = image
        net_leaf.width = width
        net_leaf.height = height

        net_leaves.append(net_leaf)

    net_leaves.save(args.store_path, args.categories, False)

    exit(0)
