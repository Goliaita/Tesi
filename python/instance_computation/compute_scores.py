from pycocotools.mask import decode

from pycocotools.coco import COCO
import argparse
import numpy as np
import json
from tqdm import tqdm


def iou_coef(y_true, y_pred, smooth=1):
    intersection = np.sum(np.multiply(y_true, y_pred))
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    iou = np.mean((intersection + smooth) / (union + smooth))
    return iou


def dice_coef(y_true, y_pred, smooth=1):
    intersection = np.sum(y_pred * y_true)
    union = np.sum(y_true) + np.sum(y_pred) 
    dice = np.mean((2.0 * (intersection + smooth)) / (union + smooth))
    return dice


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
            description='Extrapolate and manipulate images to obtain leaf dimension')

    parser.add_argument("--result_path", 
            default="../AdelaiDet/training_dir/blendmask_R_101_dcn_5x_4/inference/coco_instances_results.json",
            type=str,
            help="Path to results derived by image segmentation")

    parser.add_argument("--gt_path", 
            default="../dataset/annotations/validation900.json", 
            type=str,
            help="Path to ground true masks")

    parser.add_argument("--score",
            default=0.5,
            type=float, 
            help="Score precision to compute Scores, default = 0.5")
    
    parser.add_argument("--write",
            default=False,
            action="store_true", 
            help="Write out the results inside results.txt")

    return parser.parse_args(argv)


def compute_scores(args):

    computed_data = json.load(open(args.result_path))
   
    gt_data = json.load(open(args.gt_path))

    coco_gt = COCO(args.gt_path)
        
    gt_images = {"image_id": [], "dimensions": []}

    gt_masks = {"masks": []}

    computed_masks = {"masks": []}


    print("Loaded data")
    print()
    

    for image in gt_data["images"]:
        gt_images["image_id"].append(image["id"])
        gt_images["dimensions"].append([image["width"], image["height"]])

    for element in gt_data["annotations"]:
        x, y = gt_images["dimensions"][gt_images["image_id"].index(element["image_id"])]
        if element["category_id"] == 2:
            gt_masks["masks"].append({
                "image_id": element["image_id"],
                "segmentation": coco_gt.annToMask(element),
                "category_id": element["category_id"],
                "dimensions": [y, x]
            })

    for element in computed_data:
        if element["score"] > args.score and element["category_id"] == 2:
            computed_masks["masks"].append({
                "image_id": element["image_id"],
                "segmentation": decode(element["segmentation"]),
                "category_id": element["category_id"],
                "dimensions": element["segmentation"]["size"]
            })


    print("Data Prepared for computation")
    print(f"Total element to analyze: {len(computed_masks['masks'])}")
    print(f"Total element inside the images {len(gt_masks['masks'])}")
    print()
    print("Starting computing Scores")
    print()


    sym_list        = np.zeros((len(computed_masks["masks"]), 1))
    dice_list       = np.zeros((len(computed_masks["masks"]), 1))

    best_dice       = dict()
    sym_best_dice   = dict()

    c_tqdm = tqdm(computed_masks["masks"], ascii=True, desc="Dice")

    for idx, comp_mask in enumerate(c_tqdm):

        dice    = 0
        id      = 0

        for _, gt_mask in enumerate(gt_masks["masks"]):
            
            if comp_mask["image_id"] == gt_mask["image_id"]:
                proposed_dice = dice_coef(gt_mask["segmentation"], comp_mask["segmentation"], smooth=0)

                if dice < proposed_dice:
                    dice = proposed_dice
                    id = comp_mask["image_id"]

        dice_list[idx] = dice
        
        if str(id) in best_dice:
            best_dice[str(id)].append(dice)
        else:
            best_dice[str(id)] = [dice]


    c_tqdm = tqdm(gt_masks["masks"], ascii=True, desc="Symmetric Dice")

    for _, gt_mask in enumerate(c_tqdm):

        sym = 0
        id  = 0

        for _, comp_mask in enumerate(computed_masks["masks"]):
            if comp_mask["image_id"] == gt_mask["image_id"]:

                symmetric_dice = dice_coef(comp_mask["segmentation"], gt_mask["segmentation"], smooth=0)

                if sym < symmetric_dice:
                    sym = symmetric_dice
                    id = comp_mask["image_id"]
                
        if str(id) in sym_best_dice:
            sym_best_dice[str(id)].append(sym)
        else:
            sym_best_dice[str(id)] = [sym]
        
    sym_best_dice_score = []

    for key in sym_best_dice.keys():
        sym_best_dice_score.append(min(np.mean(sym_best_dice[key]), np.mean(best_dice[key])))

    # return sym_mean, dice_mean
    return np.mean((sym_best_dice_score)), dice_list.mean()


if __name__ == "__main__":

    args = parse_args()

    print(args)

    sym_mean, dice_mean = compute_scores(args)
    # dice_mean = compute_scores(args)
    
    if args.write:
        output = open("results.txt", "w+")
        # output.write(f"Symmetric Best Dice: {sym_mean}\n")
        output.write(f"Best Dice: {dice_mean}\n")

    print()
    print(f"Symmetric Best Dice: {sym_mean}")
    print()
    print(f"Best Dice: {dice_mean}")
    print()

