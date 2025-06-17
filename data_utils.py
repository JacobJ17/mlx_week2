from datasets import load_dataset, get_dataset_split_names
import random

def get_marco_ds_splits():
    # get_dataset_split_names("microsoft/ms_marco", "v1.1")
    dataset = load_dataset("microsoft/ms_marco", "v1.1", cache_dir="data/raw/marco")
    return dataset["train"], dataset["validation"], dataset["test"]


def generate_triplets(dataset_split, max_negatives_per_query=1):
    triplets = []

    for sample in dataset_split:
        query = sample["query"]
        passages = sample["passages"]["passage_text"]
        labels = sample["passages"]["is_selected"]

        positives = [p for p, sel in zip(passages, labels) if sel == 1]
        negatives = [p for p, sel in zip(passages, labels) if sel == 0]

        if not positives or not negatives:
            continue  # Skip if no usable data

        for pos in positives:
            for _ in range(max_negatives_per_query):
                neg = random.choice(negatives)
                triplets.append((query, pos, neg))

    return triplets


if __name__ == "__main__":
    train_split, val_split, test_split = get_marco_ds_splits()
    triplets = generate_triplets(train_split, max_negatives_per_query=1)
    print(f"Generated {len(triplets)} triplets from the training split.")
    for index in random.sample(range(len(triplets)), 5):
        print(f"Triplet {index}: {triplets[index]}")
    # You can save or process the triplets further as needed.
