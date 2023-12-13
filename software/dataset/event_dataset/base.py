import random


class BaseDataset:
    def __init__(self):
        self.files = []
        self.labels = []
        self.object_classes = []

    def shuffle(self):
        zipped_lists = list(zip(self.files, self.labels))
        random.seed(7)
        random.shuffle(zipped_lists)
        self.files, self.labels = zip(*zipped_lists)

    def init_labels(self):
        self.nr_samples = len(self.labels)
        print("Total sample num is {}".format(self.nr_samples))
        self.nr_classes = max(self.labels) + 1
        # self.nr_classes = len(self.object_classes)

    def common_preprocess(self, shuffle):
        self.init_labels()
        if shuffle:
            self.shuffle()
