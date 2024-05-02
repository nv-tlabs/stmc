import os
import codecs as cs
import orjson  # loading faster than json
import json

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from .collate import collate_text_motion


def read_split(path, split):
    split_file = os.path.join(path, "splits", split + ".txt")
    id_list = []
    with cs.open(split_file, "r") as f:
        for line in f.readlines():
            id_list.append(line.strip())
    return id_list


def load_annotations(path, name="annotations.json"):
    json_path = os.path.join(path, name)
    with open(json_path, "rb") as ff:
        return orjson.loads(ff.read())


class TextMotionDataset(Dataset):
    def __init__(
        self,
        name: str,
        motion_loader,
        text_encoder,
        split: str = "train",
        min_seconds: float = 2.0,
        max_seconds: float = 10.0,
        preload: bool = True,
        tiny: bool = False,
        # only during training
        drop_motion_perc: float = 0.15,
        drop_cond: float = 0.10,
        drop_trans: float = 0.5,
    ):
        if tiny:
            split = split + "_tiny"

        path = f"datasets/annotations/{name}"
        self.collate_fn = collate_text_motion
        self.split = split
        self.keyids = read_split(path, split)

        self.text_encoder = text_encoder
        self.motion_loader = motion_loader

        self.min_seconds = min_seconds
        self.max_seconds = max_seconds

        # remove too short or too long annotations
        self.annotations = load_annotations(path)

        # filter annotations (min/max)
        # but not for the test set
        # otherwise it is not fair for everyone
        if "test" not in split:
            self.annotations = self.filter_annotations(self.annotations)

        self.is_training = "train" in split
        self.drop_motion_perc = drop_motion_perc
        self.drop_cond = drop_cond
        self.drop_trans = drop_trans

        self.keyids = [keyid for keyid in self.keyids if keyid in self.annotations]
        self.nfeats = self.motion_loader.nfeats

        if preload:
            for _ in tqdm(self, desc="Preloading the dataset"):
                continue

    def __len__(self):
        return len(self.keyids)

    def __getitem__(self, index):
        keyid = self.keyids[index]
        return self.load_keyid(keyid)

    def load_keyid(self, keyid):
        annotations = self.annotations[keyid]

        # Take the first one for testing/validation
        # Otherwise take a random one
        index = 0
        if self.is_training:
            index = np.random.randint(len(annotations["annotations"]))

        annotation = annotations["annotations"][index]
        text = annotation["text"]

        drop_motion_perc = None
        load_transition = False
        if self.is_training:
            drop_motion_perc = self.drop_motion_perc
            drop_cond = self.drop_cond
            drop_trans = self.drop_trans
            if drop_cond is not None:
                if np.random.binomial(1, drop_cond) == 1:
                    # uncondionnal
                    text = ""
                    # load a transition
                    if np.random.binomial(1, drop_trans) == 1:
                        load_transition = True

        motion_x_dict = self.motion_loader(
            path=annotations["path"],
            start=annotation["start"],
            end=annotation["end"],
            drop_motion_perc=drop_motion_perc,
            load_transition=load_transition,
        )

        text_encoded = self.text_encoder(text)
        text_uncond_encoded = self.text_encoder("")

        x = motion_x_dict["x"]
        length = motion_x_dict["length"]

        output = {
            "x": x,
            "text": text,
            "tx": text_encoded,
            "tx_uncond": text_uncond_encoded,
            "keyid": keyid,
            "length": length,
        }
        return output

    def filter_annotations(self, annotations):
        filtered_annotations = {}
        for key, val in annotations.items():
            path = val["path"]

            # remove humanact12
            # buggy left/right + no SMPL
            if "humanact12" in path:
                continue

            annots = val.pop("annotations")
            filtered_annots = []
            for annot in annots:
                duration = annot["end"] - annot["start"]
                if self.max_seconds >= duration >= self.min_seconds:
                    filtered_annots.append(annot)

            if filtered_annots:
                val["annotations"] = filtered_annots
                filtered_annotations[key] = val

        return filtered_annotations


def write_json(data, path):
    with open(path, "w") as ff:
        ff.write(json.dumps(data, indent=4))
