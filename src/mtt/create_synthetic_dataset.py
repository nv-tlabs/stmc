import orjson
import numpy as np

from collections import namedtuple

path = "/home/mathis/AMASS-Annotation-Unifier/outputs/p_kitml/iannotations.json"
path = "/home/mathis/AMASS-Annotation-Unifier/outputs/p_babel/iannotations.json"
path = "/home/mathis/AMASS-Annotation-Unifier/outputs/p_merged/annotations.json"

with open(path, "rb") as ff:
    anns = orjson.loads(ff.read())

anns = [y for x in list(anns.values()) for y in x["annotations"]]

Info = namedtuple("Info", "text duration")

infos = []
for x in anns:
    text = x["text"]
    duration = x["end"] - x["start"]
    info = Info(text, duration)
    infos.append(info)


def get_mean_words(words, infos):
    common_words = [
        "left",
        "right",
        "in",
        "a",
        "to",
        "the",
        "with",
        "on",
        "something",
        "his",
        " ",
        "",
    ]
    words = [word for word in words if word not in common_words]
    saved = []
    for info in infos:
        take_it = True
        for word in words:
            if word not in info.text:
                take_it = False
        if info.duration >= 10:
            continue

        if take_it:
            saved.append(info)

    w_dur = sorted([x.duration for x in saved])

    if not saved:
        return 4.0

    to_crop = int(0.05 * len(w_dur))
    w_dur_crop = w_dur[to_crop:-to_crop]
    if not w_dur_crop:
        w_dur_crop = w_dur

    mean = np.mean(w_dur).round(2)
    # std = np.std(w_dur).round(2)
    return mean


Box = namedtuple("Box", "text mean std bodypart")
path = "single_prompts.txt"

with open(path, "r") as fr:
    lines = fr.readlines()


boxes = []
for line in lines:
    line = line.strip()
    if not line:
        continue

    elements = [x.strip() for x in line.split("#")]
    text = elements[0]
    # duration = float(elements[1])
    bodypart = set(elements[1:])

    words = [x.strip() for x in text.split(" ")]
    mean = get_mean_words(words, infos)
    std = np.round(0.2 * mean, 2)

    box = Box(
        text,
        mean,
        std,
        bodypart,
    )
    boxes.append(box)


# saved back
with open("single_dataset.txt", "w") as f:
    for box in boxes:
        text = box.text
        mean = box.mean
        std = box.std
        bodypart = box.bodypart

        line = [text, str(mean), str(std)] + sorted(list(bodypart))
        line = " # ".join(line) + "\n"
        f.write(line)
