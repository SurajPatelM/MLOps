# Lab Assignment 5 – Language Model Data Preparation Pipeline

This lab implements a data preparation pipeline for training a causal language model using the WikiText-2 dataset and the GPT-2 tokenizer. The goal is to convert raw text data into fixed-length token sequences suitable for training language models.

**Main lab file:** The runnable pipeline and the code you edit for this lab live in [`Labs/Data_Labs/LLM_Data_Pipeline/data_pipeline_lab1.py`](data_pipeline_lab1.py) (same directory as this README).

## Enhancements Over the Base Lab

The original lab notebook was extended with several improvements to make the pipeline more robust, modular, and informative.

### 1. Dataset Cleaning

Preprocessing removes:

- Leading and trailing whitespace
- Empty text rows

This improves dataset quality before tokenization.

### 2. Train / Validation Dataset Split

Instead of a single dataset split, the data is divided into:

- **Training set:** 90%
- **Validation set:** 10%

Implemented with `train_test_split()` so you can evaluate model inputs on unseen data.

### 3. Dataset Visualization

Two plots were added to understand the dataset better:

| Visualization | Description | Output file |
|---------------|-------------|-------------|
| **Dataset split size** | Counts of samples in training vs. validation | `dataset_split_sizes.png` |
| **Token length distribution** | Tokens per text sample before grouping | `train_token_length_distribution.png`, `validation_token_length_distribution.png` |

These plots help justify why fixed-length token blocks are needed.

### 4. Modular Pipeline Design

The code is refactored into reusable functions:

- `clean_text()`
- `tokenize_function()`
- `group_texts()`
- `plot_dataset_sizes()`
- `plot_token_length_distribution()`

This improves readability and maintainability.

### 5. DataLoader Verification

The pipeline is validated by printing:

- Batch keys
- Tensor shapes
- Decoded text samples

**Expected shapes:**

| Tensor | Shape |
|--------|--------|
| `input_ids` | `[8, 128]` |
| `labels` | `[8, 128]` |
| `attention_mask` | `[8, 128]` |

### 6. Output Logging

Script output can be redirected to a file (e.g. `lab1_output.txt`) to confirm successful execution.

---

## Dataset

| Property | Value |
|----------|--------|
| **Dataset** | WikiText-2 |
| **Source** | [Hugging Face Datasets](https://huggingface.co/docs/datasets) |
| **Load spec** | `load_dataset("wikitext", "wikitext-2-raw-v1", split="train")` |

WikiText-2 is a common benchmark for language modeling.

---

## Pipeline Architecture

```text
Raw Text Dataset
        │
        ▼
Text Cleaning
(remove whitespace & empty lines)
        │
        ▼
Train / Validation Split
        │
        ▼
Tokenizer Initialization (GPT-2)
        │
        ▼
Dataset Analysis
(token length visualization)
        │
        ▼
Tokenization
(text → token IDs)
        │
        ▼
Token Grouping
(fixed block size = 128)
        │
        ▼
Language Model Dataset
        │
        ▼
DataLoader Creation
(batch size = 8)
        │
        ▼
Batch Verification
```

---

## Implementation Steps

### 1. Load Dataset

Use the Hugging Face `datasets` library:

```python
load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
```

### 2. Clean the Text

For each entry:

- Remove leading and trailing whitespace
- Drop empty text rows

### 3. Train / Validation Split

- `test_size = 0.1`
- `random_state` / seed: `42`

### 4. Initialize the Tokenizer

Use GPT-2 from Transformers:

```python
AutoTokenizer.from_pretrained("gpt2")
```

GPT-2 has no default padding token; set:

```python
tokenizer.pad_token = tokenizer.eos_token
```

### 5. Dataset Visualization

Generate:

- Dataset size comparison (train vs. validation)
- Token length distributions

### 6. Tokenization

Map text to token IDs with the tokenizer, typically using `Dataset.map()` with batching.

### 7. Token Grouping

Concatenate tokens and split into blocks of size **`BLOCK_SIZE = 128`**. Remainders that do not fill a full block are discarded.

### 8. Data Collator

Use `DataCollatorForLanguageModeling` with **`mlm=False`** for causal (not masked) language modeling. The collator produces `input_ids`, `labels`, and `attention_mask`.

### 9. DataLoader Creation

- **`BATCH_SIZE = 8`**
- Training: shuffled batches
- Validation: sequential batches

### 10. Batch Verification

Pull one batch from each loader and print keys, shapes, and a short decoded sample.

---

## Generated Output Files

Running the pipeline produces:

| File | Purpose |
|------|---------|
| `dataset_split_sizes.png` | Train/validation sample counts |
| `train_token_length_distribution.png` | Token lengths (train) |
| `validation_token_length_distribution.png` | Token lengths (validation) |
| `lab1_output.txt` | Optional log of stdout (when redirected) |

---

## Dependencies

Install required packages:

```bash
pip install datasets transformers torch matplotlib
```

| Package | Role |
|---------|------|
| `datasets` | Load WikiText-2 |
| `transformers` | GPT-2 tokenizer & collator |
| `torch` | Tensors & DataLoaders |
| `matplotlib` | Plots |

---

## How to Run

```bash
python lab1_pipeline.py
```

Capture logs to a file:

```bash
python lab1_pipeline.py > lab1_output.txt
```

---

## Summary

This lab shows how to build an end-to-end data preparation pipeline for language model training, including:

- Dataset ingestion  
- Cleaning and train/validation splitting  
- Tokenization and fixed-length grouping  
- Visualization and batching  

The resulting dataset is ready for training causal language models such as GPT-2.
