import os

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))


def read_interval(path):
    with open(path, "r") as fr:
        lines = fr.readlines()

    all_interval_lines = []
    interval_lines = []
    for line in lines:
        if line == "\n":
            if not interval_lines:
                continue
            all_interval_lines.append(interval_lines)
            interval_lines = []
            continue
        interval_lines.append(line)

    if interval_lines:
        all_interval_lines.append(interval_lines)

    return all_interval_lines


def save(path, lines):
    with open(path, "w") as f:
        for line in lines:
            f.write(line)


interval_file = os.path.join(THIS_FOLDER, "MTT.txt")
all_interval_lines = read_interval(interval_file)

folder = os.path.join(THIS_FOLDER, "MTT")
os.makedirs(folder, exist_ok=True)

for idx, interval_lines in enumerate(all_interval_lines):
    keyid = str(idx).zfill(4)
    path = os.path.join(folder, keyid + ".txt")
    save(path, interval_lines)
