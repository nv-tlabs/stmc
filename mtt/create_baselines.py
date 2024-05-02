import os
from src.stmc import read_timelines, cut_unique_intervals, TextInterval

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))


def timeline_to_lines(timeline):
    lines = []
    for interval in timeline:
        text = interval.text
        start = float(interval.start)
        end = float(interval.end)
        bodypart = interval.bodyparts
        line = [text, str(start), str(end)] + sorted(list(bodypart))
        line = " # ".join(line) + "\n"
        lines.append(line)
    return lines


def save_one_timeline(path, timeline):
    lines = timeline_to_lines(timeline)
    with open(path, "w") as f:
        for line in lines:
            f.write(line)


def save_one_big_timeline(path, timelines):
    with open(path, "w") as f:
        for timeline in timelines:
            lines = timeline_to_lines(timeline)
            for line in lines:
                f.write(line)
            f.write("\n")


def collapse_to_singletrack(timeline):
    texts = [x.text for x in timeline]
    bps = [x.bodyparts for x in timeline]

    unique_intervals, indexes = cut_unique_intervals(timeline)
    new_timeline = []
    for unique_interval, indices in zip(unique_intervals, indexes):
        new_text = " while ".join([texts[x] for x in indices])
        # union of bodyparts
        new_bps = set()
        for x in indices:
            new_bps = new_bps.union(bps[x])
        # add the new interval
        new_timeline.append(
            TextInterval(new_text, unique_interval.start, unique_interval.end, new_bps)
        )
    return new_timeline


def collapse_to_onetext(timeline):
    texts = [x.text for x in timeline]
    all_bps = [x.bodyparts for x in timeline]

    new_bps = set()
    for bps in all_bps:
        new_bps = new_bps.union(bps)

    duration = max([x.end for x in timeline]) - min([x.start for x in timeline])
    # to make it compatilbe with MDM
    duration = min(duration, 9.8)

    # Assumption of the structure
    # 0 1
    # C
    new_text = texts[0] + " and then " + texts[1] + " while " + texts[2]
    new_timeline = [TextInterval(new_text, 0.0, duration, new_bps)]
    return new_timeline


def main():
    interval_file = os.path.join(THIS_FOLDER, "MTT.txt")
    timelines = read_timelines(interval_file, fps=None)

    folder = os.path.join(THIS_FOLDER, "baselines")
    os.makedirs(folder, exist_ok=True)

    # Collapse into a single track (for DiffCollage baseline)
    singletrack_folder = os.path.join(folder, "MTT_singletrack")
    os.makedirs(singletrack_folder, exist_ok=True)
    singletrack_file = os.path.join(folder, "MTT_singletrack.txt")

    # Collapse into one text
    onetext_folder = os.path.join(folder, "MTT_onetext")
    os.makedirs(onetext_folder, exist_ok=True)
    onetext_file = os.path.join(folder, "MTT_onetext.txt")

    s_timelines = []  # singletrack
    o_timelines = []  # onetext

    for idx, timeline in enumerate(timelines):
        keyid = str(idx).zfill(4)

        # DiffCollage baseline
        s_timeline = collapse_to_singletrack(timeline)
        s_timelines.append(s_timeline)

        # saving into different files
        singletrack_path = os.path.join(singletrack_folder, keyid + ".txt")
        save_one_timeline(singletrack_path, s_timeline)

        # One text baseline
        o_timeline = collapse_to_onetext(timeline)
        o_timelines.append(o_timeline)

        onetext_path = os.path.join(onetext_folder, keyid + ".txt")
        save_one_timeline(onetext_path, o_timeline)

    # saving into one big files
    save_one_big_timeline(singletrack_file, s_timelines)
    save_one_big_timeline(onetext_file, o_timelines)


if __name__ == "__main__":
    main()
