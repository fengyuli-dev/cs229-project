import os
import re
import json
from collections import namedtuple
from tqdm import tqdm
import random
from paths import *
from inference import generate_greedy_response, generate_sampled_responses
import argparse

# filtering techniques
# 1: 12 relations for in distribution train/test dataset, 7 for OOD test set 
# 2: filter out examples with more than 1 correct answers, 4.2% in train, 3.9% in test
# TODO 3: make sure no subjects or objects overlap between train and test sets by filtering out overlapping examples from the train set 2.1%

train_dataset = os.path.join(DATASET_PATH, "train")
val_dataset = os.path.join(DATASET_PATH, "dev")

Sample = namedtuple("Sample", ["relation", "question", "gt_answer", "is_train"])

cache_dict = dict()
CacheResponse = namedtuple("Response", ["gt_answer", "greedy_response", "sampled_response", "known_level"])
os.makedirs(os.path.join(DATASET_PATH, "llama_cache"), exist_ok=True)

# filtering 1: 12 relations for in distribution train/test dataset, 7 for OOD test set 
in_dist_relations = ['P17', 'P19', 'P26', 'P36', 'P40', 'P69', 'P131', 'P136', 'P264', 'P495', 'P740', 'P800']

if os.path.exists(os.path.join(DATASET_PATH, "exemplars.json")):
    exemplars = json.load(open(os.path.join(DATASET_PATH, "exemplars.json")))
else:
    exemplars = None


def clean_string(answer):
    answer = answer.lower()
    filter = re.compile(r'[^\w\s]')
    answer = filter.sub('', answer)
    answer = ' '.join(answer.split())
    return answer

def generate_exemplars(N_ex = 10, k_shot = 4):
    exemplar_dict = dict()
    for filename in tqdm(os.listdir(train_dataset)):
        relation = filename.split(".")[0]
        if relation not in in_dist_relations:
            continue
        qa_pairs = json.load(
            open(os.path.join(train_dataset, filename))
        )
        print(f"Length of pairs in {relation}: {len(qa_pairs)}")
        qa_pairs = [dict({'question': pair["question"], 'answer': pair["answers"][0]}) for pair in qa_pairs if len(pair["answers"]) == 1]
        # print(f"Length of pairs in {relation} after removing multiple answers: {len(qa_pairs)}")
        sampled_pairs = random.sample(qa_pairs, k=k_shot * N_ex)
        k_shot_prompts = [sampled_pairs[i*k_shot: (i+1)*k_shot] for i in range(N_ex)]
        exemplar_dict[relation] = k_shot_prompts
    location = os.path.join(DATASET_PATH, "exemplars.json")
    with open(location, "w") as file:
        json.dump(exemplar_dict, file)
    print(f"Exemplars saved at {location}")

def generate_dataset_json(percentage_of_known = 0.5):
    train_known = json.load(open(os.path.join(DATASET_PATH, "train_known.json")))
    train_unknown = json.load(open(os.path.join(DATASET_PATH, "train_unknown.json")))
    # TODO: incorporate percentage here
    
    data = [f"Q: {entry[1]} A: {entry[2]}" for entry in train_known]
    json.dump((data), open(os.path.join(DATASET_PATH, "llama_all.json"), "w"))

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
        exemplar_questions = [qs["question"] for group in exemplars[relation] for qs in group]
        for qa_pair in tqdm(qa_pairs, desc=f"Processing {filename}"):
            # filtering 2: filter out examples with more than 1 correct answers, 4.2% in train, 3.9% in test
            if len(qa_pair["answers"]) > 1 or qa_pair["question"] in exemplar_questions: 
                print(qa_pair)
                continue
            sample = Sample(relation, qa_pair["question"], qa_pair["answers"][0], is_train)
            if is_known(sample.question, sample.gt_answer, relation):
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
    cache_location = os.path.join(DATASET_PATH, 'llama_cache', f'{relation}.json')
    if os.path.exists(cache_location):
        cache = json.load(open(cache_location, 'r'))
        if question in cache:
            known_level = cache[question][3]
            return known_level != "unknown"

    # get 4-shot prompt
    prompts = exemplars[relation]
    greedy_answers = list()
    sampled_answers = list()
    for prompt in prompts:
        question_ls = [question]
        greedy_ans = generate_greedy_response(prompt, question_ls) 
        sampled_ans = generate_sampled_responses(prompt, question_ls) 
        greedy_answers.append(clean_string(greedy_ans))
        sampled_ans = [clean_string(answer) for answer in sampled_ans]
        sampled_answers.extend(sampled_ans)

    # use Exact Match to compare answers with ground truth answer and calculate P_correct
    gt_answer = clean_string(gt_answer)
    P_correct_greedy = greedy_answers.count(gt_answer) / len(greedy_answers)
    P_correct_sampled = sampled_answers.count(gt_answer) / len(sampled_answers)
    if P_correct_greedy == 1:
        known_level = "highly_known"
    elif P_correct_greedy > 0:
        known_level = "maybe_known"
    elif P_correct_sampled > 0:
        known_level = "weakly_known"
    else:
        known_level = "unknown"
    # save greedy_answers and sampled_answers
    cache = CacheResponse(gt_answer, greedy_answers, sampled_answers, known_level)
    cache_dict[question] = cache
    with open(cache_location, "w") as file:
        json.dump(cache_dict, file)

    return known_level != "unknown"

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Positional arguments
    parser.add_argument(
        "--exemplars", 
        action="store_true", 
        help="Turn on if you are generating exemplars. "
    )

    parser.add_argument(
        "--testall", 
        action="store_true", 
        help="Turn on if you are generating labels for all samples. Make sure to generate exemplars before running this."
    )

    parser.add_argument(
        "--gendata", 
        action="store_true", 
        help="Generate dataset based on known & unknown labels. Make sure to run testall before attempting this."
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    if args.exemplars:
        generate_exemplars()
    elif args.testall:
        test_all_samples()
    elif args.gendata:
        generate_dataset_json()
    
