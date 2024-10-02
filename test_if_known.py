import os
import json
from collections import namedtuple
from paths import *

train_dataset = os.path.join(DATASET_PATH, "train")
val_dataset = os.path.join(DATASET_PATH, "dev")

Sample = namedtuple("Sample", ["relation", "question", "gt_answer", "is_train"])


def test_all_samples():
    train_known = set()
    train_unknown = set()
    val_known = set()
    val_unknown = set()
    for filename in os.listdir(train_dataset) + os.listdir(val_dataset):
        relation = filename.split(".")[0]
        is_train = filename.split(".")[1] == "train"
        samples = json.load(open(os.path.join(train_dataset, filename)))
        for sample in samples:
            sample = Sample(relation, sample["question"], sample["answers"], is_train)
            if is_known(sample.question, sample.gt_answer):
                if is_train:
                    train_known.add(sample)
                else:
                    val_known.add(sample)
            else:
                if is_train:
                    train_unknown.add(sample)
                else:
                    val_unknown.add(sample)
    json.dump(
        list(train_known), open(os.path.join(DATASET_PATH, "train_known.json"), "w")
    )
    json.dump(
        list(train_unknown), open(os.path.join(DATASET_PATH, "train_unknown.json"), "w")
    )
    json.dump(list(val_known), open(os.path.join(DATASET_PATH, "val_known.json"), "w"))
    json.dump(
        list(val_unknown), open(os.path.join(DATASET_PATH, "val_unknown.json"), "w")
    )


def is_known(question, answer):
    # Access llama API to check if question-answer pair is known.
    # Always save inference results to a cache file
    cache = json.load(open(os.path.join(LLAMA_CACHE)))
    raise NotImplementedError
