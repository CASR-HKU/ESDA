from os import listdir
import numpy as np
from .base import BaseDataset
import os


class DVSLipDataset(BaseDataset):
    def __init__(self, root, mode='training', shuffle=True, **kwargs):
        super().__init__()
        self.mode_name = "train" if mode == 'training' else "test"
        self.root = os.path.join(root, self.mode_name)
        self.object_classes = LIP_classes
        self.load()
        self.common_preprocess(shuffle)

    def load(self):
        for act_dir in os.listdir(self.root):
            label = self.object_classes.index(act_dir)

            for file in os.listdir(os.path.join(self.root, act_dir)):
                if file.endswith("npy"):
                    self.labels.append(label)
                    self.files.append(os.path.join(self.root, act_dir, file))

    def __getitem__(self, idx):
        orig_events = np.load(self.files[idx])
        events = np.zeros((orig_events.shape[0], 4), dtype=np.float32)
        for i, (t,x,y,p) in enumerate(orig_events):
            events[i,2] = t
            events[i,0] = x
            events[i,1] = y
            events[i,3] = p
        label = self.labels[idx]
        return events, label




LIP_classes = [
        "accused",
        "action",
        "allow",
        "allowed",
        "america",
        "american",
        "another",
        "around",
        "attacks",
        "banks",
        "become",
        "being",
        "benefit",
        "benefits",
        "between",
        "billion",
        "called",
        "capital",
        "challenge",
        "change",
        "chief",
        "couple",
        "court",
        "death",
        "described",
        "difference",
        "different",
        "during",
        "economic",
        "education",
        "election",
        "england",
        "evening",
        "everything",
        "exactly",
        "general",
        "germany",
        "giving",
        "ground",
        "happen",
        "happened",
        "having",
        "heavy",
        "house",
        "hundreds",
        "immigration",
        "judge",
        "labour",
        "leaders",
        "legal",
        "little",
        "london",
        "majority",
        "meeting",
        "military",
        "million",
        "minutes",
        "missing",
        "needs",
        "number",
        "numbers",
        "paying",
        "perhaps",
        "point",
        "potential",
        "press",
        "price",
        "question",
        "really",
        "right",
        "russia",
        "russian",
        "saying",
        "security",
        "several",
        "should",
        "significant",
        "spend",
        "spent",
        "started",
        "still",
        "support",
        "syria",
        "syrian",
        "taken",
        "taking",
        "terms",
        "these",
        "thing",
        "think",
        "times",
        "tomorrow",
        "under",
        "warning",
        "water",
        "welcome",
        "words",
        "worst",
        "years",
        "young",
    ]