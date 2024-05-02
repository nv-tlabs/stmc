# Adpated version of STMC for MDM.
# The tensors are no longer
# [B, T, D] (batch x time x feature dimension)
# but
# ### [B, D, 1, T] (batch x feature dimension x 1 x time)
# with always T = 196 (fixed size)
# "combine_features_intervals", "interpolate_intervals" and "interpolate_intervals"
# are affected
import random

from collections import defaultdict
from dataclasses import dataclass

from stmc_mdm.bptools import get_indexes_body_parts

BODY_PARTS_LST = ["left arm", "right arm", "legs", "head", "spine"]


@dataclass(frozen=True)
class Interval:
    start: int
    end: int


@dataclass(frozen=True)
class TextInterval:
    text: str
    start: int
    end: int
    bodyparts: frozenset


@dataclass(frozen=True)
class IntervalInfo:
    out: Interval  # interval for slicing the whole motion (final output)
    crop: Interval  # interval for slicing the crop motion (output of the network)
    index: int  # index in the batch


def read_bodyparts(bodyparts):
    mapping = {x: x for x in BODY_PARTS_LST}
    # make it a little more permisive
    mapping["torso"] = "spine"
    mapping["neck"] = "head"
    mapping["left hand"] = "left arm"
    mapping["right hand"] = "right arm"
    mapping["left leg"] = "legs"
    mapping["right leg"] = "legs"

    # throw an error if the body part does not exists
    return frozenset(mapping[x] for x in bodyparts)


def read_timelines(path, fps, start_at_zero=True):
    # start_at_zero: each timeline will start at 0.0 sec
    with open(path, "r") as fr:
        lines = fr.readlines()

    timelines = []
    timeline = []
    for line in lines:
        if line == "\n":
            if not timeline:
                continue
            timelines.append(timeline)
            timeline = []
            continue

        # extracting the text interval
        line = line.strip()
        elements = [x.strip() for x in line.split("#")]
        text = elements[0]
        start = int(fps * float(elements[1]))
        end = int(fps * float(elements[2]))
        bodyparts = read_bodyparts(elements[3:])

        # adding it to the timeline
        timeline.append(TextInterval(text, start, end, bodyparts))

    if timeline:
        timelines.append(timeline)

    if start_at_zero:
        # make each timeline start at zero
        def shift_right(timeline):
            min_t = min([x.start for x in timeline])
            new_timeline = [
                TextInterval(x.text, x.start - min_t, x.end - min_t, x.bodyparts)
                for x in timeline
            ]
            return new_timeline

        timelines = [shift_right(timeline) for timeline in timelines]
    return timelines


def cut_unique_intervals(timeline):
    # cut the timeline into pieces where there is no "intersections"
    # Finding intersection points
    inter_points = sorted(
        list(set(x.start for x in timeline).union(set(x.end for x in timeline)))
    )

    unique_intervals = []
    indexes = []
    left = inter_points[0]
    # construct sub intervals
    for right in inter_points[1:]:
        unique_intervals.append(Interval(left, right))
        left = right
        indexes.append([])

    # find which indices (of the timeline) correspond to which interval
    for idx, x in enumerate(timeline):
        a, b = x.start, x.end
        for i, y in enumerate(unique_intervals):
            l, r = y.start, y.end
            if a <= l <= r <= b:
                indexes[i].append(idx)
    return unique_intervals, indexes


def sinc_heuristic(indices, timeline):
    # SINC heuristic
    # return the body part assignation
    # idx -> {body parts}

    # 1) Find the base
    # Test for legs
    legs_candidates = []
    for idx in indices:
        if "legs" in timeline[idx].bodyparts:
            legs_candidates.append(idx)

    if legs_candidates:
        # more than one (normally not compatible but let's try anyway)
        # take the one with the more body parts as base
        if len(legs_candidates) > 1:
            random.shuffle(legs_candidates)
            # shuffle in case of equality
            legs_candidates.sort(key=lambda x: len(timeline[x].bodyparts), reverse=True)
            base = legs_candidates[0]
        else:
            base = legs_candidates[0]
    else:
        # No legs: take the one with the more body parts
        candidates = [x for x in indices]
        # shuffle in case of equality
        random.shuffle(candidates)
        candidates.sort(key=lambda x: len(timeline[x].bodyparts), reverse=True)
        base = candidates[0]

    # take the others
    others = [x for x in indices if x != base]
    # sorted by the number of body parts
    others.sort(key=lambda x: len(timeline[x].bodyparts), reverse=True)

    body_parts_assignation = defaultdict(set)

    # 2) Assign all the body to the base
    body_parts_assignation[base] = set(BODY_PARTS_LST)

    # 3) For all the others, override
    for index in others:
        for body_part in timeline[index].bodyparts:
            # override previous
            for key in body_parts_assignation:
                previous = body_parts_assignation[key]
                if body_part in previous:
                    previous.remove(body_part)
            # add to the new one
            body_parts_assignation[index].add(body_part)

    return body_parts_assignation


def create_body_parts_timeline(timeline):
    # Create one timeline per body part
    # for each bodypart (bp) save a list of (index, interval)
    # where index is the index in the timeline, and interval is the unique_interval

    # Step 1: cut the timeline
    unique_intervals, indexes = cut_unique_intervals(timeline)

    # Step 2: "fill the holes"
    bp_timeline_unique = defaultdict(list)
    for indices, c_int in zip(indexes, unique_intervals):
        body_parts_assignation = sinc_heuristic(indices, timeline)
        for index, bps in body_parts_assignation.items():
            for bp in bps:
                bp_timeline_unique[bp].append((index, c_int))

    # Regroup the body part timelines
    # if the same text is cut -> regroup into a contiguous interval
    bp_timeline = {}
    # notation: bp_xxx is a dict of body parts -> info
    #           xxxx_bp is the info of a particular body part
    for bp in BODY_PARTS_LST:
        timeline_bp = []
        for index, c_int in bp_timeline_unique[bp]:
            # add the first one
            if not timeline_bp:
                timeline_bp.append((index, c_int))
                continue

            last_index, last_c_int = timeline_bp.pop()

            if last_index != index:
                # if the index has changed: no need to merge
                timeline_bp.append((last_index, last_c_int))
                timeline_bp.append((index, c_int))
            else:
                # merge the intervals
                new_c_int = Interval(last_c_int.start, c_int.end)
                timeline_bp.append((index, new_c_int))
        bp_timeline[bp] = timeline_bp

    return bp_timeline


def extend_timelines(timeline, bp_timeline, overlap_left, overlap_right, max_t):
    # take more space for temporal composition
    timeline = [
        TextInterval(
            x.text,
            max(x.start - overlap_left, 0),  # don't go out of bounds
            min(x.end + overlap_right, max_t),  # don't go out of bounds
            x.bodyparts,
        )
        for x in timeline
    ]
    # same for the body parts timeline
    bp_timeline = {
        bp: [
            (
                index,
                Interval(
                    max(c_int.start - overlap_left, 0),  # don't go out of bounds
                    min(c_int.end + overlap_right, max_t),  # don't go out of bounds
                ),
            )
            for index, c_int in bp_timeline[bp]
        ]
        for bp in BODY_PARTS_LST
    }
    return timeline, bp_timeline


def get_transitions_info(
    timeline, bp_timeline, overlap_left, overlap_right, max_t, n_texts
):
    # Save the body part associated to transition points
    trans_left_bp = defaultdict(set)
    trans_right_bp = defaultdict(set)

    for bp in BODY_PARTS_LST:
        for index, c_int in bp_timeline[bp]:
            assert bp not in trans_left_bp[c_int.start]
            trans_left_bp[c_int.start].add(bp)
            assert bp not in trans_right_bp[c_int.end]
            trans_right_bp[c_int.end].add(bp)

    # transition points
    inter_points = sorted(
        list(set(x.start for x in timeline).union(set(x.end for x in timeline)))
    )[1:-1]

    # Save the transitions info
    bp_trans_info = defaultdict(list)
    trans_intervals = []
    for idx, x in enumerate(inter_points):
        start = max(x - overlap_left, 0)
        end = min(x + overlap_right, max_t)

        # find which body parts are associated at this time
        # sanity checks
        assert x in trans_left_bp
        assert x in trans_right_bp
        assert trans_left_bp[x] == trans_right_bp[x]
        trans_bp_x = trans_left_bp[x]

        trans_intervals.append(Interval(start, end))

        for bp in trans_bp_x:
            bp_trans_info[bp].append(
                IntervalInfo(
                    out=Interval(start, end),  # interval for slicing the final motion
                    crop=Interval(0, end - start),  # interval for slicing the crop
                    index=n_texts + idx,  # (after all the texts)
                )
            )
    for bp in BODY_PARTS_LST:
        # create the list
        bp_trans_info[bp]

    bp_trans_info = dict(bp_trans_info)

    return trans_intervals, bp_trans_info


def process_timeline(timeline, interval_overlap, extend=True, uncond=True):
    # extend: extend the timeline for temporal composition
    # uncond: create the uncondionnal timeline (for DiffCollage)

    overlap_left = interval_overlap // 2
    overlap_right = interval_overlap - interval_overlap // 2

    n_texts = len(timeline)
    max_t = max([x.end for x in timeline])

    # Step 1: Create one timeline per body parts
    bp_timeline = create_body_parts_timeline(timeline)

    # Step 2: Save the transition info
    trans_intervals, bp_trans_info = get_transitions_info(
        timeline, bp_timeline, overlap_left, overlap_right, max_t, n_texts
    )

    # Save the bp trans info for SINC lerp
    # final step, without touching the intervals
    bp_trans_info_lerp = bp_trans_info
    if not uncond:
        trans_intervals = []
        bp_trans_info = {bp: [] for bp in BODY_PARTS_LST}

    n_uncond = len(trans_intervals)

    # Step 3: Extend the timelines
    original_timeline = timeline
    original_bp_timeline = bp_timeline
    if extend:
        timeline, bp_timeline = extend_timelines(
            timeline, bp_timeline, overlap_left, overlap_right, max_t
        )

    # Step 4: save all the info for fast slicing
    bp_info = defaultdict(list)
    for bp in BODY_PARTS_LST:
        for index, c_int in bp_timeline[bp]:
            bp_info[bp].append(
                IntervalInfo(
                    out=c_int,  # interval for slicing the output motion
                    crop=Interval(  # relative to the original crop
                        start=c_int.start - timeline[index].start,
                        end=c_int.end - timeline[index].start,
                    ),  # interval for slicing the output
                    index=index,  # index in the batch
                )
            )
    bp_info = dict(bp_info)

    # adding the uncondionnal
    lengths = [x.end - x.start for x in (timeline + trans_intervals)]
    texts = [x.text for x in timeline] + [""] * n_uncond

    info = {
        # timeline
        "timeline": timeline,
        "bp_timeline": bp_timeline,
        "bp_trans_info": bp_trans_info,
        "bp_info": bp_info,
        "all_intervals": timeline + trans_intervals,
        # batch
        "n_uncond": n_uncond,
        "texts": texts,
        "max_t": max_t,
        "lengths": lengths,
        # extra
        "original_timeline": original_timeline,
        "original_bp_timeline": original_bp_timeline,
        "bp_trans_info_lerp": bp_trans_info_lerp,
    }

    return info


def process_timelines(timelines, interval_overlap, extend=True, uncond=True):
    infos = defaultdict(list)

    for timeline in timelines:
        info = process_timeline(
            timeline, interval_overlap, extend=extend, uncond=uncond
        )

        # Merge the infos
        for key, val in info.items():
            infos[key].append(val)

    infos = dict(infos)

    infos["all_texts"] = [text for texts in infos["texts"] for text in texts]
    infos["all_lengths"] = [
        length for lengths in infos["lengths"] for length in lengths
    ]

    infos["n_frames"] = max(infos["max_t"])
    infos["n_seq"] = len(infos["timeline"])
    return infos


def combine_features_intervals(x_comb, infos, output):
    lengths = infos["lengths"]
    bp_infos = infos["bp_info"]
    bp_trans_infos = infos["bp_trans_info"]

    real_nseq = len(output)
    indexes_bp = get_indexes_body_parts(infos["featsname"])

    # I use an offset because we have several timeline per batch
    offset = 0
    for idx in range(real_nseq):
        bp_info = bp_infos[idx]
        # for the diffcollage per body parts
        bp_trans_info = bp_trans_infos[idx]
        for bp in BODY_PARTS_LST:
            for x in bp_info[bp]:
                ii = x.index + offset
                val = x_comb[ii, indexes_bp[bp], :, x.crop.start : x.crop.end]
                output[idx, indexes_bp[bp], :, x.out.start : x.out.end] += val

            # remove score for uncondionnal (DiffCollage)
            for x in bp_trans_info[bp]:
                ii = x.index + offset
                val = x_comb[ii, indexes_bp[bp], :, x.crop.start : x.crop.end]
                output[idx, indexes_bp[bp], :, x.out.start : x.out.end] -= val

        # changing the offset for one seq to the other
        # took "lengths" here because it contains
        # transitions (bigger offsets)
        offset += len(lengths[idx])
    return output


def interpolate_intervals(sample, infos):
    import torch

    device = sample.device

    indexes_bp = get_indexes_body_parts(infos["featsname"])
    bp_trans_info = infos["bp_trans_info_lerp"]

    new_sample = sample.clone()
    N = len(sample)
    assert len(bp_trans_info) == N

    for idx in range(N):
        for bp in BODY_PARTS_LST:
            for x in bp_trans_info[idx][bp]:
                begin, end = x.out.start, x.out.end

                # This is the part we want to smooth out
                val_bp = sample[idx, indexes_bp[bp], :, begin:end]
                trans_duration = val_bp.shape[-1]

                # First and end values
                val_bp_1 = 0 * val_bp + val_bp[:, :, [0]]
                val_bp_2 = 0 * val_bp + val_bp[:, :, [-1]]

                # linearly go from 1 to 2
                w = torch.linspace(1, 0, trans_duration, device=device)[None, None]
                interp_val = val_bp_1 * w + val_bp_2 * (1 - w)

                # override the data
                new_sample[idx, indexes_bp[bp], :, begin:end] = interp_val
    return new_sample
