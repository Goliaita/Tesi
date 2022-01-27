import os
import argparse

parser = argparse.ArgumentParser(
        description='Extrapolate and manipulate images to obtain leaf dimension')

parser.add_argument("--output", default="runs/detect/exp/instances", help="Path where save the image masks")

parser.add_argument("--config_file", default="../AdelaiDet/configs/BlendMask/R_101_dcni3_5x.yaml")
parser.add_argument("--train_json_path", default="runs/detect/exp/instances/labels.json")
parser.add_argument("--train_path", default="runs/detect/exp/cropped")
parser.add_argument("--val_json_path", default="runs/detect/exp/instances/labels.json")
parser.add_argument("--val_path", default="runs/detect/exp/cropped")
parser.add_argument("--weights", default="../AdelaiDet/training_dir/blendmask_R_101_dcn_5x_4/model_final.pth")

args = parser.parse_args()


os.system(f"OMP_NUM_THREADS=1 python ../AdelaiDet/tools/train_net.py --eval-only --config-file {args.config_file} \
            OUTPUT_DIR {args.output} \
            dataset_name komatsuna train_json_path {args.train_json_path} \
            val_json_path {args.train_json_path} train_path {args.train_path} \
            val_path {args.train_path} \
            MODEL.WEIGHTS {args.weights}")
