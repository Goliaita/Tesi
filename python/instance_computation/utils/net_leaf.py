from utils.leaf import Leaf
import json


class NetLeaf(Leaf):

    leaves_list = []

    def __init__(self, id: int = 0, dataset_id: int = 1, file_name: str = "", path: str = "", width: int = 0, height: int = 0,
                 image_name: str = "", position: tuple = (), category_ids: list = [], annotated: bool = False, annotating: list = [],
                 num_annotations: int = 0, metadata: dict = {}, deleted: bool = False, milliseconds: int = 0, events: list = [],
                 regenerate_thubnail: bool = False):

        self.id = id
        self.dataset_id = dataset_id
        self.file_name = file_name
        self.path = path
        self.width = width
        self.height = height
        self.image_name = image_name
        self.position = position
        self.category_ids = category_ids
        self.annotated = annotated
        self.annotating = annotating
        self.num_annotations = num_annotations
        self.metadata = metadata
        self.deleted = deleted
        self.milliseconds = milliseconds
        self.events = events
        self.regenerate_thubnail = regenerate_thubnail


    def get_json_list(self):
        images = {"images": []}

        for element in self.leaves_list:
            images["images"].append(element.__dict__)

        return images

    def save(self, path, categories=["overlapped", "non_overlapped"], info=True):

        images = {"images": [], "categories": []}
        images["images"] = self.get_json_list()["images"]

        for idx, element in enumerate(categories):

            images["categories"].append({
                "id":idx+1,
                "name": element,
                "supercategory": "",
                "color": "#78ef51",
                "metadata": {},
                "keypoint_colors": []
                })
                


        with open(f"{path}/labels.json", 'w+') as f:
            json.dump(images, f)

        if info:
            with open(f"{path}/position.json", 'w+') as f:
                json.dump(self.get_info(), f)

    def get_info(self):
        info = {"info": []}

        for element in self.leaves_list:
            info["info"].append({
                "description": element.position,
                "id": element.id,
                "image_name": element.image_name
            })
        return info
