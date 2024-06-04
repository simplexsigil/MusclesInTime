import json
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

babel_action_cats_encode = {
    "a pose": 0,
    "action with ball": 1,
    "adjust": 2,
    "admire": 3,
    "agressive actions series": 4,
    "aim": 5,
    "animal behavior": 6,
    "arm movements": 7,
    "backwards": 8,
    "backwards movement": 9,
    "balance": 10,
    "bartender behavior series": 11,
    "bend": 12,
    "blow": 13,
    "bow": 14,
    "bump": 15,
    "cartwheel": 16,
    "catch": 17,
    "celebrate": 18,
    "charge": 19,
    "check": 20,
    "chicken dance": 21,
    "chop": 22,
    "circular movement": 23,
    "clap": 24,
    "clasp hands": 25,
    "clean something": 26,
    "close something": 27,
    "come": 28,
    "communicate (vocalise)": 29,
    "conduct": 30,
    "confusion": 31,
    "cough": 32,
    "count": 33,
    "cower": 34,
    "cradle": 35,
    "crawl": 36,
    "crossing limbs": 37,
    "crouch": 38,
    "cry": 39,
    "curtsy": 40,
    "cut": 41,
    "dance": 42,
    "despair": 43,
    "dip": 44,
    "disagree": 45,
    "dive": 46,
    "doing ocean motion": 47,
    "draw": 48,
    "dribble": 49,
    "drink": 50,
    "drive": 51,
    "drunken behavior": 52,
    "duck": 53,
    "eat": 54,
    "endure": 55,
    "engage": 56,
    "evade": 57,
    "excite": 58,
    "exercise/training": 59,
    "face direction": 60,
    "fall": 61,
    "feet movements": 62,
    "fidget": 63,
    "fight": 64,
    "fill": 65,
    "find": 66,
    "fire gun": 67,
    "fish": 68,
    "fist bump": 69,
    "flail arms": 70,
    "flap": 71,
    "flip": 72,
    "fly": 73,
    "follow": 74,
    "foot movements": 75,
    "forward movement": 76,
    "gain": 77,
    "gesture": 78,
    "get injured": 79,
    "give something": 80,
    "glide": 81,
    "golf": 82,
    "grab body part": 83,
    "grab person": 84,
    "grasp object": 85,
    "greet": 86,
    "grind": 87,
    "groom": 88,
    "hand movements": 89,
    "handstand": 90,
    "hang": 91,
    "head movements": 92,
    "headstand": 93,
    "hiccup": 94,
    "hit": 95,
    "hop": 96,
    "hope": 97,
    "hug": 98,
    "hurry": 99,
    "interact with rope": 100,
    "interact with/use object": 101,
    "inward motion": 102,
    "jog": 103,
    "join": 104,
    "juggle": 105,
    "jump": 106,
    "jump rope": 107,
    "jumping jacks": 108,
    "kick": 109,
    "knee movement": 110,
    "kneel": 111,
    "knock": 112,
    "laugh": 113,
    "lead": 114,
    "lean": 115,
    "leap": 116,
    "learn": 117,
    "leave": 118,
    "leg movements": 119,
    "lick": 120,
    "lie": 121,
    "lift something": 122,
    "limp": 123,
    "list body parts": 124,
    "listen": 125,
    "look": 126,
    "lose": 127,
    "lowering body part": 128,
    "lunge": 129,
    "maintain": 130,
    "make": 131,
    "march": 132,
    "martial art": 133,
    "mime": 134,
    "misc. abstract action": 135,
    "misc. action": 136,
    "misc. activities": 137,
    "mix": 138,
    "moonwalk": 139,
    "move back to original position": 140,
    "move misc. body part": 141,
    "move something": 142,
    "move up/down incline": 143,
    "navigate": 144,
    "noisy labels": 145,
    "open something": 146,
    "operate interface": 147,
    "pat": 148,
    "perform": 149,
    "place something": 150,
    "plant feet": 151,
    "play": 152,
    "play catch": 153,
    "play instrument": 154,
    "play sport": 155,
    "plead": 156,
    "point": 157,
    "pose": 158,
    "poses": 159,
    "pray": 160,
    "prepare": 161,
    "press something": 162,
    "protect": 163,
    "punch": 164,
    "puppeteer": 165,
    "raising body part": 166,
    "rake": 167,
    "read": 168,
    "relax": 169,
    "release": 170,
    "remove": 171,
    "repeat": 172,
    "reveal": 173,
    "ride": 174,
    "rocking movement": 175,
    "rolling movement": 176,
    "rolls on ground": 177,
    "rub": 178,
    "run": 179,
    "salute": 180,
    "scratch": 181,
    "search": 182,
    "shake": 183,
    "shave": 184,
    "shivering": 185,
    "shock": 186,
    "shoulder movements": 187,
    "shout": 188,
    "shrug": 189,
    "shuffle": 190,
    "side to side movement": 191,
    "sideways movement": 192,
    "sign": 193,
    "sit": 194,
    "skate": 195,
    "skip": 196,
    "slash gesture series": 197,
    "sleep": 198,
    "sleepwalk": 199,
    "slide": 200,
    "smell": 201,
    "sneak": 202,
    "sneeze": 203,
    "spin": 204,
    "sports move": 205,
    "spread": 206,
    "squat": 207,
    "stagger": 208,
    "stances": 209,
    "stand": 210,
    "stand up": 211,
    "start": 212,
    "steady": 213,
    "step": 214,
    "stick": 215,
    "stomp": 216,
    "stop": 217,
    "strafe": 218,
    "stretch": 219,
    "stroke": 220,
    "stumble": 221,
    "style hair": 222,
    "sudden movement": 223,
    "support": 224,
    "sway": 225,
    "swim": 226,
    "swing body part": 227,
    "swipe": 228,
    "t pose": 229,
    "take/pick something up": 230,
    "tap": 231,
    "taunt": 232,
    "telephone call": 233,
    "tentative movements": 234,
    "think": 235,
    "throw": 236,
    "tie": 237,
    "tiptoe": 238,
    "to loosen": 239,
    "to lower a body part": 240,
    "touch ground": 241,
    "touch object": 242,
    "touching body part": 243,
    "touching face": 244,
    "transition": 245,
    "trip": 246,
    "try": 247,
    "turn": 248,
    "twist": 249,
    "uncross": 250,
    "unknown": 251,
    "upper body movements": 252,
    "vomit": 253,
    "waddle": 254,
    "waist movements": 255,
    "wait": 256,
    "walk": 257,
    "wash": 258,
    "wave": 259,
    "weave": 260,
    "whistle": 261,
    "wiggle": 262,
    "wobble": 263,
    "worry": 264,
    "wring": 265,
    "wrist movements": 266,
    "write": 267,
    "yawn": 268,
    "yoga": 269,
    "zip/unzip": 270,
    "zombie": 271,
}

babel_action_cats_dec = {v: k for k, v in babel_action_cats_encode.items()}


class Label:
    """
    Represents a single action label within a motion sequence, including both raw and processed labels,
    the segment identifier, optional action categories, and the start and end times of the action.

    Parameters:
    - raw_label (str): The original label as provided in the dataset.
    - proc_label (str): The processed or standardized version of the label.
    - seg_id (str): A unique identifier for the segment within the motion sequence.
    - act_cat (Optional[List[str]]): A list of action categories this label belongs to, if applicable.
    - start_t (Optional[float]): The start time of the action within the motion sequence.
    - end_t (Optional[float]): The end time of the action within the motion sequence.
    """

    def __init__(
        self,
        raw_label: str,
        proc_label: str,
        seg_id: str,
        act_cat: Optional[List[str]] = None,
        start_t: Optional[float] = None,
        end_t: Optional[float] = None,
    ):
        self.raw_label: str = raw_label
        self.proc_label: str = proc_label
        self.seg_id: str = seg_id
        self.act_cat: Optional[List[str]] = act_cat if act_cat is not None else []
        self.start_t: Optional[float] = start_t
        self.end_t: Optional[float] = end_t

    def __repr__(self) -> str:
        return json.dumps(self.__dict__, indent=2)


class SeqAnn:
    """
    Represents sequence-level annotations for a motion sequence in the BABEL dataset,
    including the BABEL label identifier, annotator identifier, a flag indicating multiple actions, and a list of action labels.

    Parameters:
    - babel_lid (str): The unique identifier for the label within the BABEL dataset.
    - anntr_id (str): The unique identifier for the annotator who provided the annotations.
    - mul_act (bool): A flag indicating whether the sequence contains multiple actions.
    - labels (Optional[List[Label]]): A list of `Label` objects representing the annotated actions within the sequence.
    """

    def __init__(
        self,
        babel_lid: str,
        anntr_id: str,
        mul_act: bool,
        labels: Optional[List[Label]] = None,
    ):
        self.babel_lid: str = babel_lid
        self.anntr_id: str = anntr_id
        self.mul_act: bool = mul_act
        self.labels: List[Label] = labels if labels is not None else []

    def to_dict(self) -> Dict:
        return {
            "babel_lid": self.babel_lid,
            "anntr_id": self.anntr_id,
            "mul_act": self.mul_act,
            "labels": [label.__dict__ for label in self.labels],
        }

    def __repr__(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class FrameAnn:
    """
    Represents frame-level annotations similar to `SeqAnn` but intended for annotations that apply to individual frames
    within a motion sequence.

    Parameters:
    - babel_lid (str): The unique identifier for the label within the BABEL dataset.
    - anntr_id (str): The unique identifier for the annotator who provided the annotations.
    - mul_act (bool): A flag indicating whether the sequence contains multiple actions.
    - labels (Optional[List[Label]]): A list of `Label` objects representing the annotated actions within the frames.
    """

    def __init__(
        self,
        babel_lid: str,
        anntr_id: str,
        mul_act: bool,
        labels: Optional[List[Label]] = None,
    ):
        self.babel_lid: str = babel_lid
        self.anntr_id: str = anntr_id
        self.mul_act: bool = mul_act
        self.labels: List[Label] = labels if labels is not None else []

    def to_dict(self) -> Dict:
        return {
            "babel_lid": self.babel_lid,
            "anntr_id": self.anntr_id,
            "mul_act": self.mul_act,
            "labels": [label.__dict__ for label in self.labels],
        }

    def __repr__(self) -> str:
        return json.dumps(
            {
                "babel_lid": self.babel_lid,
                "anntr_id": self.anntr_id,
                "mul_act": self.mul_act,
                "labels": [label.__dict__ for label in self.labels],
            },
            indent=2,
        )


class BabelData:
    """
    Encapsulates a single entry in the BABEL dataset, including metadata such as the BABEL sequence ID,
    the URL for the sequence, the path to extracted features, the duration of the sequence,
    and optional sequence-level and frame-level annotations.

    Look here: https://babel.is.tue.mpg.de/data.html

    Parameters:
    - babel_sid (int): The unique sequence identifier within the BABEL dataset.
    - url (str): The URL where the sequence can be accessed or downloaded.
    - feat_p (str): The file path to the extracted features for the sequence.
    - dur (float): The total duration of the sequence in seconds.
    - seq_ann (Optional[SeqAnn]): Sequence-level annotations for the sequence.
    - frame_ann (Optional[FrameAnn]): Frame-level annotations for the sequence.
    """

    def __init__(
        self,
        babel_sid: int,
        url: str,
        feat_p: str,
        dur: float,
        seq_ann: Optional[SeqAnn] = None,
        frame_ann: Optional[FrameAnn] = None,
        split: Optional[str] = None,
    ):
        self.babel_sid: int = babel_sid
        self.url: str = url
        self.feat_p: str = feat_p
        self.dur: float = dur
        self.seq_ann: Optional[SeqAnn] = seq_ann if seq_ann is not None else {}
        self.frame_ann: Optional[FrameAnn] = frame_ann if frame_ann is not None else {}
        self.split = split

    def sequence_actions(self, encoded=False) -> List[Tuple[float, float, str]]:
        """
        Returns a list of sequence level actions performed in the sequence.

        Parameters:
            encoded (bool): If True, the actions will be returned as encoded values.
                            If False, the actions will be returned as their original labels.

        Returns:
            List[Tuple[float, float, str]]: A list of tuples containing the start time (0),
                                             end time (duration), and action label for each action
                                             performed in the sequence.
        """
        actions: List[str] = []
        if isinstance(self.seq_ann, SeqAnn):
            for label in self.seq_ann.labels:
                act_cat = (
                    [babel_action_cats_encode[a] for a in label.act_cat]
                    if encoded
                    else label.act_cat
                )
                actions.extend(act_cat)
        return (0.0, self.dur, actions)

    def clip_actions_in_range(
        self, start_time: float, end_time: float, encoded=False, coverage_threshold=0.5
    ) -> List[Tuple[float, float, str]]:
        """
        Returns a list of clip/frame level actions within the specified time range.
        If no frame level actions are available, defaults to sequence level actions.

        Args:
            start_time (float): The start time of the range.
            end_time (float): The end time of the range.
            encoded (bool, optional): Whether to return the actions encoded or not. Defaults to False.

        Returns:
            List[Tuple[float, float, str]]: A list of tuples containing the start time, end time, and action category.

        """
        actions: List[str] = []
        if isinstance(self.frame_ann, FrameAnn):
            for label in self.frame_ann.labels:
                if (
                    label.start_t is not None
                    and label.end_t is not None
                    and label.end_t > label.start_t + 1e-3
                ):
                    # overlap between the label and the time range
                    overlap = min(end_time, label.end_t) - max(
                        start_time, label.start_t
                    )
                    if overlap / (label.end_t - label.start_t) >= coverage_threshold:
                        act_cat = (
                            [babel_action_cats_encode[a] for a in label.act_cat]
                            if encoded
                            else label.act_cat
                        )
                        actions.append((label.start_t, label.end_t, act_cat))
        else:
            actions.append(self.sequence_actions())

        actions = sorted(actions, key=lambda x: x[0])

        return actions

    def clip_actions_at(
        self, time_point: float, encoded=False
    ) -> List[Tuple[float, float, str]]:
        """
        Returns the clip/frame level actions of the clips that take place during the specified time point.
        If no clip level actions are available, defaults to sequence level actions.

        Args:
            time_point (float): The time point at which the actions are to be retrieved.
            encoded (bool, optional): Specifies whether the actions should be returned in encoded format.
                                      Defaults to False.

        Returns:
            List[Tuple[float, float, str]]: A list of tuples representing the actions of the clips.
                                             Each tuple contains the start time, end time, and action label.
        """
        actions: List[str] = []
        if isinstance(self.frame_ann, FrameAnn):
            for label in self.frame_ann.labels:
                if label.start_t is not None and label.end_t is not None:
                    if label.start_t <= time_point <= label.end_t:
                        act_cat = [babel_action_cats_encode[a] for a in label.act_cat] if encoded else label.act_cat
                        actions.extend(act_cat)
        else:
            actions.extend(self.sequence_actions())

        return actions

    def clip_actions(self, encoded=False) -> List[Tuple[float, float, str]]:
        """
        Returns a list of all clip/frame level actions.
        If no clip/frame level actions are available, defaults to sequence level actions.

        Args:
            encoded (bool): If True, the actions will be encoded.

        Returns:
            List[Tuple[float, float, str]]: A list of tuples containing the start time, end time, and action label for each clip action.
        """
        return self.clip_actions_in_range(0.0, self.dur + 1, encoded)

    def sequence_proc_labels(self, encoded=False) -> List[Tuple[float, float, str]]:
        """
        Return a list of sequence level processed labels.

        Args:
            encoded (bool): Flag indicating whether the labels should be encoded or not.

        Returns:
            Tuple[float, float, List[str]]: A tuple containing the start time, duration, and a list of processed labels.
        """
        proc_labels: List[str] = []
        if isinstance(self.seq_ann, SeqAnn):
            for label in self.seq_ann.labels:
                if label.proc_label is not None:
                    proc_labels.append(label.proc_label)
        return (0.0, self.dur, proc_labels)

    def clip_proc_labels_in_range(
        self, start_time: float, end_time: float
    ) -> List[Tuple[float, float, str]]:
        """
        Returns a list of clip/frame level processed labels within the specified time range.

        Args:
            start_time (float): The start time of the range.
            end_time (float): The end time of the range.

        Returns:
            List[Tuple[float, float, str]]: A list of tuples containing the start time, end time, and processed label for each label within the specified range.
        """
        proc_labels: List[str] = []
        if isinstance(self.frame_ann, FrameAnn):
            for label in self.frame_ann.labels:
                if label.start_t is not None and label.end_t is not None:
                    if (
                        start_time <= label.start_t <= end_time
                        or start_time <= label.end_t <= end_time
                    ):
                        proc_labels.append(
                            (label.start_t, label.end_t, label.proc_label)
                        )
        else:
            proc_labels.append(self.sequence_proc_labels())

        proc_labels = sorted(proc_labels, key=lambda x: x[0])

        return proc_labels

    def clip_proc_labels_at(self, time_point: float) -> List[Tuple[float, float, str]]:
        """
        Returns the clip/frame level processed labels of the clips that take place during the specified time point.

        Args:
            time_point (float): The time point at which the processed labels are to be retrieved.

        Returns:
            List[Tuple[float, float, str]]: A list of tuples representing the processed labels of the clips.
                                           Each tuple contains the start time, end time, and processed label.
        """
        return self.clip_proc_labels_in_range(time_point, time_point)

    def clip_proc_labels(self) -> List[Tuple[float, float, str]]:
        """
        Returns a list of all clip/frame level processed labels.

        Returns:
            List[Tuple[float, float, str]]: A list of tuples containing the start time, end time, and processed label for each label.
        """
        return self.clip_proc_labels_in_range(0.0, self.dur + 1)

    def __repr__(self) -> str:
        return json.dumps(
            {
                "babel_sid": self.babel_sid,
                "url": self.url,
                "feat_p": self.feat_p,
                "dur": self.dur,
                "seq_ann": self.seq_ann.to_dict()
                if isinstance(self.seq_ann, SeqAnn)
                else None,
                "frame_ann": self.frame_ann.to_dict()
                if isinstance(self.frame_ann, FrameAnn)
                else None,
            },
            indent=2,
        )

    @classmethod
    def from_dict(cls, data: dict, split: Optional[str] = None) -> "BabelData":
        babel_sid = data["babel_sid"]
        url = data["url"]
        feat_p = data["feat_p"]
        dur = data["dur"]

        seq_ann_data = data.get("seq_ann", data.get("seq_anns", {}))

        if isinstance(seq_ann_data, list):
            # if len(seq_ann_data) > 1:
                # print(
                #     f"Sequence annotations are ambiguous. Choosing first annotation. Split: {split}, babel_sid {babel_sid}, len(seq_ann_data): {len(seq_ann_data)}"
                # )
            seq_ann_data = seq_ann_data[0]

        seq_ann = SeqAnn(
            seq_ann_data.get("babel_lid", ""),
            seq_ann_data.get("anntr_id", ""),
            seq_ann_data.get("mul_act", False),
            [Label(**label_data) for label_data in seq_ann_data.get("labels", [])],
        )

        frame_ann_data = data.get("frame_ann", data.get("frame_anns", {}))

        if isinstance(frame_ann_data, list):
            if len(frame_ann_data) > 1:
                print(
                    f"Frame annotations are ambiguous. Choosing first annotation. Split: {split}, babel_sid {babel_sid}, len(frame_ann_data): {len(frame_ann_data)}"
                )
            frame_ann_data = frame_ann_data[0]

        frame_ann = None

        if frame_ann_data is not None:  # Check if frame_ann_data is not None
            frame_ann = FrameAnn(
                frame_ann_data.get("babel_lid", ""),
                frame_ann_data.get("anntr_id", ""),
                frame_ann_data.get("mul_act", False),
                [
                    Label(**label_data)
                    for label_data in frame_ann_data.get("labels", [])
                ],
            )

        return BabelData(babel_sid, url, feat_p, dur, seq_ann, frame_ann, split=split)


class BabelDataset:
    """
    A `Dataset` implementation for the BABEL dataset, providing utilities to access data by BABEL sequence ID or segment ID,
    and methods to load dataset instances from dictionaries or JSON files.

    Parameters:
    - data (List[BabelData]): A list of `BabelData` objects representing the dataset entries.

    Methods:
    - `__len__`: Returns the total number of entries in the dataset.
    - `__getitem__`: Retrieves a dataset entry by its index.
    - `by_babel_sid`: Retrieves a dataset entry by its BABEL sequence ID.
    - `by_seg_id`: Retrieves a dataset entry by its segment ID.
    - `from_dict`: Class method to create a `BabelDataset` instance from a dictionary.
    - `from_json_file`: Class method to load a `BabelDataset` instance from a JSON file.
    """

    def __init__(
        self, data: Union[List[BabelData], OrderedDict], split: Optional[str] = None
    ):
        self.data = (
            data
            if isinstance(data, OrderedDict)
            else OrderedDict((item.babel_sid, item) for item in data)
        )
        self.seg_id_mappings = {}
        self.feat_p_mapping = {}

        for v in self.data.values():
            if v.seq_ann:
                self.feat_p_mapping[v.feat_p] = v
                for l in v.seq_ann.labels:
                    self.seg_id_mappings[l.seg_id] = v
            if v.frame_ann:
                self.feat_p_mapping[v.feat_p] = v
                for l in v.frame_ann.labels:
                    self.seg_id_mappings[l.seg_id] = v

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> BabelData:
        return list(self.data.values())[idx]

    def by_babel_sid(
        self, babel_sid: int, raise_missing=True, default=None
    ) -> BabelData:
        try:
            return self.data[babel_sid]
        except KeyError:
            if raise_missing:
                raise KeyError(f"Babel ID {babel_sid} not found in the dataset.")
            else:
                return default

    def by_seg_id(self, seg_id: str, raise_missing=True, default=None) -> BabelData:
        try:
            return self.seg_id_mappings[seg_id]
        except KeyError:
            if raise_missing:
                raise KeyError(f"Segment ID {seg_id} not found in the dataset.")
            else:
                return default

    def by_feat_p(self, feat_p: str, raise_missing=True, default=None) -> BabelData:
        if feat_p[-4:] != ".npz":
            feat_p = feat_p + ".npz"

        try:
            return self.feat_p_mapping[feat_p]
        except KeyError:
            if raise_missing:
                raise KeyError(f"Feature path {feat_p} not found in the dataset.")
            else:
                return default

    @classmethod
    def act_cat_enc(
        cls, act_cat: Union[str, List[str], np.ndarray]
    ) -> Union[int, List[int]]:
        if isinstance(act_cat, str):
            # Single string, return the corresponding int
            return cls.babel_action_cats_encode[act_cat]
        elif isinstance(act_cat, list) or isinstance(act_cat, np.ndarray):
            # A list or tensor of strings, return a list of ints
            if isinstance(act_cat, np.ndarray):
                # If it's a tensor, assuming it's a tensor of string types, convert to a list first
                act_cat = act_cat.tolist()
            return [cls.babel_action_cats_encode[item] for item in act_cat]
        else:
            raise ValueError("Unsupported type for act_cat")

    @classmethod
    def act_cat_dec(
        cls, act_cat: Union[int, List[int], np.ndarray]
    ) -> Union[str, List[str]]:
        if isinstance(act_cat, int):
            # Single integer, return the corresponding string
            return cls.babel_action_cats_dec[act_cat]
        elif isinstance(act_cat, list) or isinstance(act_cat, np.ndarray):
            # A list or tensor of integers, return a list of strings
            if isinstance(act_cat, np.ndarray):
                # If it's a tensor, convert to a list first
                act_cat = act_cat.tolist()
            return [cls.babel_action_cats_dec[item] for item in act_cat]
        else:
            raise ValueError("Unsupported type for act_cat")

    @classmethod
    def from_dict(cls, babel_dict: dict, split: Optional[str] = None) -> "BabelDataset":
        dataset = [BabelData.from_dict(d, split) for d in babel_dict.values()]

        dataset = BabelDataset(dataset)

        return dataset

    @classmethod
    def from_json_file(cls, file_path, split=None) -> "BabelDataset":
        # Load the JSON data from a file or a string
        with open(file_path, "r") as f:
            data = json.load(f)

        return BabelDataset.from_dict(data, split=split)

    @classmethod
    def from_datasets(cls, datasets) -> "BabelDataset":
        """
        Create a BabelDataset from a list of datasets. If there are duplicate BABEL sequence IDs, the first dataset in the list will take precedence.
        This means if the datasets are ordered by [train,val,test,extra_train,extra_val], data from extra_train and extra_val will only be used
        if there is no data for a given BABEL sequence ID in train, val, or test.

        Args:
            datasets (list): A list of datasets.

        Returns:
            BabelDataset: A BabelDataset object containing the combined data from all datasets.
        """
        data = OrderedDict()

        for d in datasets:
            for _, v in d.data.items():
                if v.babel_sid not in data:
                    data[v.babel_sid] = v

        return BabelDataset(data)

    @classmethod
    def from_directory(cls, directory) -> "BabelDataset":
        """
        Create a BabelDataset from a directory containing the JSON files.
        Assumes that the directory contains train, val, test, extra_train, and extra_val JSON file.

        Args:
            directory (str): The path to the directory containing the JSON files.

        Returns:
            BabelDataset: A BabelDataset object containing the data from the JSON files in the directory.
        """
        import os

        babel_train_path = os.path.join(directory, "train.json")
        babel_val_path = os.path.join(directory, "val.json")
        babel_test_path = os.path.join(directory, "test.json")
        babel_extra_train_path = os.path.join(directory, "extra_train.json")
        babel_extra_val_path = os.path.join(directory, "extra_val.json")

        train_dataset = BabelDataset.from_json_file(babel_train_path, split="train")
        test_dataset = BabelDataset.from_json_file(babel_val_path, split="val")
        val_dataset = BabelDataset.from_json_file(babel_test_path, split="test")

        extra_train_dataset = BabelDataset.from_json_file(
            babel_extra_train_path, split="extra_train"
        )
        extra_val_dataset = BabelDataset.from_json_file(
            babel_extra_val_path, split="extra_val"
        )

        common_dataset = BabelDataset.from_datasets(
            [
                train_dataset,
                test_dataset,
                val_dataset,
                extra_train_dataset,
                extra_val_dataset,
            ]
        )

        return common_dataset


if __name__ == "__main__":
    print("Testing BabelDataset class")
    print(
        "Make sure to have the BABEL dataset downloaded and extracted in the correct path."
    )

    babel_train_path = "./babel_dataset/babel_v1-0_release/train.json"
    babel_val_path = "./babel_dataset/babel_v1-0_release/val.json"
    babel_test_path = "./babel_dataset/babel_v1-0_release/test.json"

    babel_extra_train_path = "./babel_dataset/babel_v1-0_release/extra_train.json"
    babel_extra_val_path = "./babel_dataset/babel_v1-0_release/extra_val.json"

    train_dataset = BabelDataset.from_json_file(babel_train_path, split="train")
    test_dataset = BabelDataset.from_json_file(babel_val_path, split="val")
    val_dataset = BabelDataset.from_json_file(babel_test_path, split="test")

    extra_train_dataset = BabelDataset.from_json_file(
        babel_extra_train_path, split="extra_train"
    )
    extra_val_dataset = BabelDataset.from_json_file(
        babel_extra_val_path, split="extra_val"
    )

    common_dataset = BabelDataset.from_datasets(
        [
            train_dataset,
            test_dataset,
            val_dataset,
            extra_train_dataset,
            extra_val_dataset,
        ]
    )

    for d in common_dataset[:50]:
        d: BabelData
        print(d.sequence_actions())
