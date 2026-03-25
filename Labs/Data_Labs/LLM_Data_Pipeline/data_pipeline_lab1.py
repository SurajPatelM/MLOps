"""
Lab 1: Language Model Data Preparation Pipeline

This script:
1. Loads the WikiText-2 dataset
2. Cleans the text
3. Splits into train and validation sets
4. Loads the GPT-2 tokenizer
5. Generates visualization plots
6. Tokenizes the dataset
7. Groups tokens into fixed-length blocks
8. Creates PyTorch DataLoaders
9. Verifies one batch from train and validation loaders

Plots generated:
- dataset_split_sizes.png
- train_token_length_distribution.png
- validation_token_length_distribution.png
"""
import os
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


MODEL_NAME = "gpt2"
BLOCK_SIZE = 128
BATCH_SIZE = 8
VAL_SPLIT = 0.1
SEED = 42


def clean_text(example):
    text = example["text"].strip()
    return {"text": text}


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"])


def group_texts(examples):
    concatenated_examples = {}
    for key in examples.keys():
        concatenated_examples[key] = sum(examples[key], [])

    total_length = len(concatenated_examples["input_ids"])

    # Drop remainder so every sequence has exact BLOCK_SIZE length
    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE

    result = {}
    for key, tokens in concatenated_examples.items():
        result[key] = [
            tokens[i:i + BLOCK_SIZE]
            for i in range(0, total_length, BLOCK_SIZE)
        ]

    return result


def plot_dataset_sizes(train_dataset, val_dataset):
    labels = ["Train", "Validation"]
    sizes = [len(train_dataset), len(val_dataset)]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, sizes)
    plt.title("Dataset Split Sizes")
    plt.xlabel("Split")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/dataset_split_sizes.png")
    plt.show()


def plot_token_length_distribution(dataset, tokenizer, split_name="train"):
    lengths = [len(tokenizer(text)["input_ids"]) for text in dataset["text"]]

    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=30, edgecolor="black")
    plt.title(f"Token Length Distribution ({split_name})")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"plots/{split_name}_token_length_distribution.png")
    plt.show()

    print(f"\n{split_name} token statistics:")
    print(f"Min length: {min(lengths)}")
    print(f"Max length: {max(lengths)}")
    print(f"Average length: {sum(lengths) / len(lengths):.2f}")


def main():
    print("Loading WikiText-2 dataset...")
    raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    print(f"Original number of rows: {len(raw_dataset)}")

    # Clean whitespace
    cleaned_dataset = raw_dataset.map(clean_text)

    # Remove empty lines
    cleaned_dataset = cleaned_dataset.filter(lambda x: len(x["text"]) > 0)

    print(f"Number of rows after cleaning: {len(cleaned_dataset)}")

    # Train/validation split
    split_dataset = cleaned_dataset.train_test_split(
        test_size=VAL_SPLIT,
        seed=SEED
    )

    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    print(f"Train rows: {len(train_dataset)}")
    print(f"Validation rows: {len(val_dataset)}")

    # Load tokenizer
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # GPT-2 does not have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Visualization plots
    print("\nGenerating plots...")
    plot_dataset_sizes(train_dataset, val_dataset)
    plot_token_length_distribution(train_dataset, tokenizer, split_name="train")
    plot_token_length_distribution(val_dataset, tokenizer, split_name="validation")

    # Tokenize datasets
    print("\nTokenizing datasets...")
    tokenized_train = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=["text"]
    )

    tokenized_val = val_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=["text"]
    )

    # Group into fixed-size blocks
    print(f"\nGrouping tokens into blocks of size {BLOCK_SIZE}...")
    lm_train_dataset = tokenized_train.map(group_texts, batched=True)
    lm_val_dataset = tokenized_val.map(group_texts, batched=True)

    print(f"Number of train sequences: {len(lm_train_dataset)}")
    print(f"Number of validation sequences: {len(lm_val_dataset)}")

    # Data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # DataLoaders
    train_loader = DataLoader(
        lm_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator
    )

    val_loader = DataLoader(
        lm_val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator
    )

    # Verify one batch from train loader
    print("\nChecking one batch from train DataLoader...")
    train_batch = next(iter(train_loader))
    print("Train batch keys:", train_batch.keys())
    print("Train input_ids shape:", train_batch["input_ids"].shape)
    print("Train labels shape:", train_batch["labels"].shape)
    if "attention_mask" in train_batch:
        print("Train attention_mask shape:", train_batch["attention_mask"].shape)

    # Verify one batch from validation loader
    print("\nChecking one batch from validation DataLoader...")
    val_batch = next(iter(val_loader))
    print("Validation batch keys:", val_batch.keys())
    print("Validation input_ids shape:", val_batch["input_ids"].shape)
    print("Validation labels shape:", val_batch["labels"].shape)
    if "attention_mask" in val_batch:
        print("Validation attention_mask shape:", val_batch["attention_mask"].shape)

    # Decode one sample sequence for sanity check
    print("\nDecoded sample from train batch:")
    sample_ids = train_batch["input_ids"][0]
    print(tokenizer.decode(sample_ids, skip_special_tokens=False)[:500])

    print("\nLab 1 pipeline completed successfully.")
    print("Saved plots:")
    print("- dataset_split_sizes.png")
    print("- train_token_length_distribution.png")
    print("- validation_token_length_distribution.png")


if __name__ == "__main__":
    main()