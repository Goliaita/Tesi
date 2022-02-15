import datetime
import json
import argparse
import random as rd



parser = argparse.ArgumentParser(
    description='Transform coco dataset JSON into randomic split training/validation/test')
parser.add_argument('--path_i', type=str, required=True,
                    help='Path to original JSON file')
parser.add_argument('--path_o', type=str, required=True,
                    help='Path to folder where write JSON files')
parser.add_argument('--training', type=int, default=60,
                    help='Training split size')
parser.add_argument('--validation', type=int, default=20,
                    help='Validation split size')
parser.add_argument('--no_test', action='store_true', help="Use it if you don't need test set")
args = parser.parse_args()


def data_divider(json_in: json, t_percentage, v_percentage, no_test=False):

    json_images = {}
    training_json = {}
    valid_json = {}
    if not no_test:
        test_json = {}
    else:
        test_json = None


    json_images["images"] = rd.sample(json_in["images"], len(json_in["images"]))

    training_set_pic = int(len(json_images["images"])/100*t_percentage)

    validation_set_pic = int(len(json_images["images"])/100*v_percentage)


    test_set_pic = training_set_pic + validation_set_pic


    training_json["images"]     = json_images["images"][:training_set_pic]
    valid_json["images"]        = json_images["images"][training_set_pic+1:test_set_pic]
    if not no_test:
        test_json["images"]         = json_images["images"][test_set_pic+1:]

        print("The dataset was split as training set: {}, validation set: {}, test set: {}"
            .format(len(training_json["images"]), len(valid_json["images"]), len(test_json["images"])))
    else:
        print("The dataset was split as training set: {}, validation set: {}"
            .format(len(training_json["images"]), len(valid_json["images"])))

    training_json["categories"] = json_in["categories"]
    valid_json["categories"] = json_in["categories"]
    if not no_test:
        test_json["categories"] = json_in["categories"]

    training_json["annotations"] = []
    valid_json["annotations"] = []
    if not no_test:
        test_json["annotations"] = []

    for image in training_json["images"]:
        for annotation in json_in["annotations"]:
            if image["id"] == annotation["image_id"]:
                training_json["annotations"].append(annotation)
        
    for image in valid_json["images"]:
        for annotation in json_in["annotations"]:
            if image["id"] == annotation["image_id"]:
                valid_json["annotations"].append(annotation)

    if not no_test:
        for image in test_json["images"]:
            for annotation in json_in["annotations"]:
                if image["id"] == annotation["image_id"]:
                    test_json["annotations"].append(annotation)

    return training_json, valid_json, test_json


def json_read(path):
    return json.load(open(path, "r"))


def json_write(path, training, validation, test_):

    time = datetime.datetime.now().microsecond

    training_w      = open(path + "/training" + str(len(training["images"])) + ".json", "w+")
    validation_w    = open(path + "/validation" + str(len(validation["images"])) + ".json", "w+")
    if test_ is not None:
        test_w          = open(path + "/test" + str(len(test_["images"])) + ".json", "w+")

    training_w.write(json.dumps(training))
    validation_w.write(json.dumps(validation))
    if test_ is not None:
        test_w.write(json.dumps(test_))


if __name__=='__main__':

    if args.training + args.validation == 100 and not args.no_test:
        print("Error no test set split found compute the percentage again")
        exit()
    elif args.training + args.validation != 100 and args.no_test:
        args.validation = 100 - args.training
        print("Percentage not valid modified validation")


    original_json = json_read(args.path_i)

    training, validation, test_ = data_divider(original_json, args.training, args.validation, args.no_test)

    json_write(args.path_o, training, validation, test_)

