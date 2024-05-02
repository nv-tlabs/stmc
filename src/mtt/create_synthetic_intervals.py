from collections import namedtuple
import sys

sys.path.pop(0)
# remove problems with logging and ipdb


BODY_PARTS_LST = ["left arm", "right arm", "legs", "head", "spine"]
BP_WEIGHTS = {
    "left arm": 1,
    "right arm": 1,
    "legs": 3,
    "head": 1,
    "spine": 1,
}

path = "single_dataset.txt"

with open(path, "r") as fr:
    lines = fr.readlines()

Box = namedtuple("Box", "text mean std bodypart")
Interval = namedtuple("Interval", "text start end bodypart")

boxes = []
weights = []
for line in lines:
    line = line.strip()
    if not line:
        continue

    elements = [x.strip() for x in line.split("#")]
    text = elements[0]
    mean = float(elements[1])
    std = float(elements[2])
    bodypart = set(elements[3:])

    weight = sum(BP_WEIGHTS[x] for x in bodypart)
    box = Box(
        text,
        mean,
        std,
        frozenset(bodypart),
    )
    boxes.append(box)
    weights.append(weight)


n_bps = len(BODY_PARTS_LST)
complementary_bps = {}
bodypart_mapping = {}
for x in range(0, 2**n_bps):
    choice = bin(x)[2:].zfill(n_bps)
    subset = frozenset(BODY_PARTS_LST[idx] for idx, x in enumerate(choice) if x == "1")
    comp_subset = frozenset(
        BODY_PARTS_LST[idx] for idx, x in enumerate(choice) if x == "0"
    )
    bodypart_mapping[subset] = []
    complementary_bps[subset] = comp_subset
    for idx, box in enumerate(boxes):
        # add box if compatible within the subset
        if box.bodypart.union(subset) == subset:
            bodypart_mapping[subset].append(idx)


all_intervals = []

import random

random.seed(0)

indices = list(range(len(boxes)))

while len(all_intervals) < 500:
    # More body parts is prefered:
    # more likely to take it as base
    idx1, idx2 = random.sample(indices, counts=weights, k=2)
    box1, box2 = boxes[idx1], boxes[idx2]

    dur1 = round(random.gauss(box1.mean, box1.std), 2)
    dur2 = round(random.gauss(box2.mean, box2.std), 2)

    _, bonus = max((dur1, "left"), (dur2, "right"))
    # if left is larger
    # both: 1/7
    # left: 4/7
    # right: 2/7
    val = random.sample(["left", "right", bonus, "both"], counts=[2, 2, 2, 1], k=1)[0]

    if val == "left":
        max_dur = dur1
        offset = 0
        bodyparts = box1.bodypart
    elif val == "right":
        max_dur = dur2
        offset = dur1
        bodyparts = box2.bodypart
    else:
        max_dur = dur1 + dur2
        offset = 0
        bodyparts = box1.bodypart.union(box2.bodypart)

    eps = random.gauss(0, 1)

    allowed_bps = complementary_bps[bodyparts]
    authorized_indices = [
        x
        for x in bodypart_mapping[allowed_bps]
        if round(boxes[x].mean + eps * boxes[x].std, 2) <= max_dur  # can fit
        and x not in [idx1, idx2]  # not box1 or box2
    ]

    if not authorized_indices:
        continue

    mid_box = boxes[random.choice(authorized_indices)]
    mid_dur = round(mid_box.mean + eps * mid_box.std, 2)

    start = round(offset + random.uniform(0, max_dur - mid_dur), 2)

    intervals = [
        Interval(box1.text, 0, dur1, box1.bodypart),
        Interval(box2.text, dur1, dur1 + dur2, box2.bodypart),
        Interval(mid_box.text, start, start + mid_dur, mid_box.bodypart),
    ]

    # walking backwards # 0.0 # 6.0 # legs
    # raising the right hand in the air # 2.0 # 5.0 # right arm

    all_intervals.append(intervals)

path = "interval_dataset.txt"
with open(path, "w") as f:
    for intervals in all_intervals:
        for interval in intervals:
            text = interval.text
            # avoid things like 17.990000000000002 instead of 18
            start = int(interval.start * 100) / 100
            end = int(interval.end * 100) / 100
            bodypart = interval.bodypart

            line = [text, str(start), str(end)] + sorted(list(bodypart))
            line = " # ".join(line) + "\n"
            f.write(line)
        f.write("\n")
