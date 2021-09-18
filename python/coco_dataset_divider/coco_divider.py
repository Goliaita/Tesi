import datetime
import json
import argparse
import random as rd



parser = argparse.ArgumentParser(
    description='Transform coco dataset JSON into randomic split training/validation')
parser.add_argument('--path_i', type=str, required=True,
                    help='Path to original JSON file')
parser.add_argument('--path_o', type=str, required=True,
                    help='Path to folder where write JSON files')
parser.add_argument('--training', type=int, default=60,
                    help='Training split size')
parser.add_argument('--validation', type=int, default=20,
                    help='Validation split size')
args = parser.parse_args()


def data_divider(json_in: json, t_percentage, v_percentage):

    json_images = {}
    training_json = {}
    valid_json = {}
    test_json = {}


    json_images["images"] = rd.sample(json_in["images"], len(json_in["images"]))

    training_set_pic = int(len(json_images["images"])/100*t_percentage)

    validation_set_pic = int(len(json_images["images"])/100*v_percentage)

    test_set_pic = training_set_pic + validation_set_pic

    training_json["images"]     = json_images["images"][:training_set_pic]
    valid_json["images"]        = json_images["images"][training_set_pic+1:test_set_pic]
    test_json["images"]         = json_images["images"][test_set_pic+1:]

    print("The dataset was split as training set: {}, validation set: {}, test set: {}"
    .format(len(training_json["images"]), len(valid_json["images"]), len(test_json["images"])))

    training_json["categories"] = json_in["categories"]
    valid_json["categories"] = json_in["categories"]
    test_json["categories"] = json_in["categories"]

    training_json["annotations"] = []
    valid_json["annotations"] = []
    test_json["annotations"] = []

    for image in training_json["images"]:
        for annotation in json_in["annotations"]:
            if image["id"] == annotation["image_id"]:
                training_json["annotations"].append(annotation)
        
    for image in valid_json["images"]:
        for annotation in json_in["annotations"]:
            if image["id"] == annotation["image_id"]:
                valid_json["annotations"].append(annotation)

    for image in test_json["images"]:
        for annotation in json_in["annotations"]:
            if image["id"] == annotation["image_id"]:
                test_json["annotations"].append(annotation)

    return training_json, valid_json, test_json


def json_read(path):
    return json.load(open(path, "r"))


def json_write(path, training, validation, test_):

    time = datetime.datetime.now().microsecond

    training_w      = open(path + "/training" + str(time) + ".json", "w+")
    validation_w    = open(path + "/validation" + str(time) + ".json", "w+")
    test_w          = open(path + "/test" + str(time) + ".json", "w+")

    training_w.write(json.dumps(training))
    validation_w.write(json.dumps(validation))
    test_w.write(json.dumps(test_))


if __name__=='__main__':

    if args.training_percentage+args.validation_percentage == 100:
        print("Error no training set split found compute the percentage again")
        exit()

    original_json = json_read(args.path_i)

    training, validation, test_ = data_divider(original_json, args.training_percentage, args.validation_percentage)

    json_write(args.path_o, training, validation, test_)

