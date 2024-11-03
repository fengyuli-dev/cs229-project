import os
import json
from collections import namedtuple
from tqdm import tqdm
import random
from paths import *
from inference import generate_greedy_response, generate_sampled_responses

# filtering techniques
# 1: 12 relations for in distribution train/test dataset, 7 for OOD test set 
# 2: filter out examples with more than 1 correct answers, 4.2% in train, 3.9% in test
# TODO 3: make sure no subjects or objects overlap between train and test sets by filtering out overlapping examples from the train set 2.1%

train_dataset = os.path.join(DATASET_PATH, "train")
val_dataset = os.path.join(DATASET_PATH, "dev")

Sample = namedtuple("Sample", ["relation", "question", "gt_answers", "is_train"])

cache_dict = dict()
CacheResponse = namedtuple("Response", ["gt_answers", "greedy_response", "sampled_response", "known_level"])

# filtering 1: 12 relations for in distribution train/test dataset, 7 for OOD test set 
in_dist_relations = ['P17', 'P19', 'P26', 'P36', 'P40', 'P69', 'P131', 'P136', 'P264', 'P495', 'P740', 'P800']

def generate_exemplars(N_ex = 10, k_shot = 4):
    exemplar_dict = dict()
    for filename in tqdm(os.listdir(train_dataset)):
        relation = filename.split(".")[0]
        if relation not in in_dist_relations:
            continue
        qa_pairs = json.load(
            open(os.path.join(train_dataset, filename))
        )
        # print(f"Length of pairs in {relation}: {len(qa_pairs)}")
        qa_pairs = [pair for pair in qa_pairs if len(pair["answers"]) == 1]
        # print(f"Length of pairs in {relation} after removing multiple answers: {len(qa_pairs)}")
        sampled_pairs = random.choices(qa_pairs, k=k_shot * N_ex)
        k_shot_prompts = [sampled_pairs[i*k_shot: (i+1)*k_shot] for i in range(N_ex)]
        exemplar_dict[relation] = k_shot_prompts
    location = os.path.join(DATASET_PATH, "exemplars.json")
    with open(location, "w") as file:
        json.dump(exemplar_dict, file)
    print(f"Exemplars saved at {location}")



exemplars = json.load(open(os.path.join(DATASET_PATH, "exemplars.json")))

def test_all_samples():
    train_known = []
    train_unknown = []
    val_known = []
    val_unknown = []
    count = 0
    for filename in tqdm(os.listdir(train_dataset) + os.listdir(val_dataset)):
        relation = filename.split(".")[0]
        if relation not in in_dist_relations:
            continue
        is_train = filename.split(".")[1] == "train"
        qa_pairs = json.load(
            open(os.path.join(train_dataset if is_train else val_dataset, filename))
        )
        for qa_pair in tqdm(qa_pairs, desc=f"Processing {filename}"):
            sample = Sample(relation, qa_pair["question"], qa_pair["answers"], is_train)
            # filtering 2: filter out examples with more than 1 correct answers, 4.2% in train, 3.9% in test
            if len(sample.gt_answers) > 1: 
                print(sample)
                continue
            if is_known(sample.question, sample.gt_answers, relation):
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
        with open(os.path.join(DATASET_PATH, "llama_cache", f"{relation}.json"), "w") as file:
            json.dump((cache_dict), file)
        cache_dict.clear()

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


def is_known(question, gt_answer, relation):
    # Access llama API to check if question-answer pair is known.
    # Always save inference results to a cache file
    # cache = json.load(open(os.path.join(LLAMA_CACHE)))

    # TODO: load cache

    # get 4-shot prompt
    prompts = exemplars[relation]
    greedy_answers = list()
    sampled_answers = list()
    for prompt in prompts:
        question_ls = [question]
        greedy_ans = generate_greedy_response(prompt, question_ls) 
        sampled_ans = generate_sampled_responses(prompt, question_ls) 
        greedy_answers.extend(greedy_ans)
        sampled_answers.extend(sampled_ans)

    # use Exact Match to compare answers with ground truth answer and calculate P_correct
    P_correct_greedy = greedy_answers.count(gt_answer) / len(greedy_answers)
    P_correct_sampled = sampled_answers.count(gt_answer) / len(sampled_answers)
    if P_correct_greedy == 1:
        known_level = "highly_known"
    elif P_correct_greedy > 0:
        known_level = "maybe_correct"
    elif P_correct_sampled > 0:
        known_level = "weakly_correct"
    else:
        known_level = "unknown"
    # save greedy_answers and sampled_answers
    cache = CacheResponse(gt_answer, greedy_answers, sampled_answers, known_level)
    cache_dict[question] = cache

    return P_correct_greedy > 0 or P_correct_sampled > 0


if __name__ == "__main__":
    test_all_samples()
    # generate_exemplars()
