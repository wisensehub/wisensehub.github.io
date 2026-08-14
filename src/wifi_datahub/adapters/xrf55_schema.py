"""Official XRF55 WiFi layout and activity vocabulary.

The file name contract and label names come from the XRF55 repository's
``Q&A.md``.  Activity IDs are one-based in the released file names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


XRF55_SAMPLE_RATE_HZ = 200.0
XRF55_SUBCARRIERS = 30
XRF55_RX_LINKS = 9
XRF55_TIME_SAMPLES = 1000

XRF55_ACTION_NAMES: dict[int, str] = {
    1: "carrying weight",
    2: "mopping the floor",
    3: "using a phone",
    4: "throwing something",
    5: "picking something",
    6: "putting something on the table",
    7: "cutting something",
    8: "wearing a hat",
    9: "putting on clothing",
    10: "blowing dry hair",
    11: "combing hair",
    12: "brushing teeth",
    13: "drinking",
    14: "eating",
    15: "smoking",
    16: "shaking hands",
    17: "hugging",
    18: "handing something to someone",
    19: "kicking someone",
    20: "hitting someone with something",
    21: "choking someone's neck",
    22: "pushing someone",
    23: "body weight squats",
    24: "Tai Chi",
    25: "boxing",
    26: "weightlifting",
    27: "hula hooping",
    28: "jumping rope",
    29: "jumping jack",
    30: "high leg lifting",
    31: "waving",
    32: "clapping hands",
    33: "falling on the floor",
    34: "jumping",
    35: "running",
    36: "sitting down",
    37: "standing up",
    38: "turning",
    39: "walking",
    40: "stretching",
    41: "patting on the shoulder",
    42: "playing Er-Hu",
    43: "playing Ukulele",
    44: "playing drum",
    45: "foot stamping",
    46: "shaking head",
    47: "nodding",
    48: "drawing a circle",
    49: "drawing a cross",
    50: "pushing",
    51: "pulling",
    52: "swiping left",
    53: "swiping right",
    54: "swiping up",
    55: "swiping down",
}


@dataclass(frozen=True)
class XRF55ClipId:
    subject_id: int
    action_id: int
    repetition_id: int

    @property
    def action_name(self) -> str:
        return XRF55_ACTION_NAMES[self.action_id]


def parse_xrf55_clip_id(path: str | Path) -> XRF55ClipId:
    """Parse the official ``subject_action_repetition.npy`` file name."""

    name = Path(path).name
    match = re.fullmatch(r"(\d+)_(\d+)_(\d+)\.npy", name)
    if match is None:
        raise ValueError(
            "XRF55 WiFi file name must be subject_action_repetition.npy, "
            f"got {name!r}"
        )
    subject_id, action_id, repetition_id = (int(item) for item in match.groups())
    if action_id not in XRF55_ACTION_NAMES:
        raise ValueError(f"XRF55 action ID must be 1..55, got {action_id}")
    return XRF55ClipId(subject_id, action_id, repetition_id)
