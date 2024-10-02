import os
import json
from collections import namedtuple
from tqdm import tqdm
from paths import *

train_dataset = os.path.join(DATASET_PATH, "train")
val_dataset = os.path.join(DATASET_PATH, "dev")

Sample = namedtuple("Sample", ["relation", "question", "gt_answers", "is_train"])


def test_all_samples():
    train_known = []
    train_unknown = []
    val_known = []
    val_unknown = []
    count = 0
    for filename in tqdm(os.listdir(train_dataset) + os.listdir(val_dataset)):
        relation = filename.split(".")[0]
        is_train = filename.split(".")[1] == "train"
        qa_pairs = json.load(
            open(os.path.join(train_dataset if is_train else val_dataset, filename))
        )
        for qa_pair in tqdm(qa_pairs, desc=f"Processing {filename}"):
            sample = Sample(relation, qa_pair["question"], qa_pair["answers"], is_train)
            if is_known(sample.question, sample.gt_answers):
                if is_train:
                    train_known.append(sample)
                else:
                    val_known.append(sample)
            else:
                if is_train:
                    train_unknown.append(sample)
                else:
                    val_unknown.append(sample)
            count += 1

    json.dump((train_known), open(os.path.join(DATASET_PATH, "train_known.json"), "w"))
    json.dump(
        (train_unknown), open(os.path.join(DATASET_PATH, "train_unknown.json"), "w")
    )
    json.dump((val_known), open(os.path.join(DATASET_PATH, "val_known.json"), "w"))
    json.dump((val_unknown), open(os.path.join(DATASET_PATH, "val_unknown.json"), "w"))
    print(f"Total samples: {count}")
    print(f"Total in train: {len(train_known) + len(train_unknown)}")
    print(f"# known in train: {len(train_known)}")
    print(f"# unknown in train: {len(train_unknown)}")
    print(f"Total in val: {len(val_known) + len(val_unknown)}")
    print(f"# known in val: {len(val_known)}")
    print(f"# unknown in val: {len(val_unknown)}")


def is_known(question, answer):
    # Access llama API to check if question-answer pair is known.
    # Always save inference results to a cache file
    # cache = json.load(open(os.path.join(LLAMA_CACHE)))
    return True


if __name__ == "__main__":
    test_all_samples()
