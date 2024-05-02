from dataclasses import dataclass


@dataclass(frozen=True)
class TextDuration:
    text: str
    duration: int


def read_texts(path, fps):
    with open(path, "r") as fr:
        lines = fr.readlines()

    text_durations = []
    for line in lines:
        if line == "\n":
            continue

        # extracting the text interval
        line = line.strip()
        elements = [x.strip() for x in line.split("#")]
        text = elements[0]
        duration = int(fps * float(elements[1]))

        text_durations.append(TextDuration(text, duration))
    return text_durations
