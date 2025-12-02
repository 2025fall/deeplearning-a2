from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from transformers import DataCollatorForSeq2Seq

from constants import MAX_INPUT_LENGTH, MAX_TARGET_LENGTH, SRC_LANG, TGT_LANG


def build_dataset() -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    """
    Build the dataset.

    Returns:
        The dataset.

    NOTE: You can replace this with your own dataset. Make sure to include
    the `validation` split and ensure that it is the same as the test split from the WMT19 dataset,
    Which means that:
        raw_datasets["validation"] = load_dataset('wmt19', 'zh-en', split="validation")
    """
    dataset = load_dataset("wmt19", "zh-en")

    def length_filter(example):
        zh = example["translation"][SRC_LANG]
        en = example["translation"][TGT_LANG]
        # Keep moderately sized pairs to avoid extremely short/long outliers.
        return 4 <= len(zh) <= 400 and 4 <= len(en) <= 400

    train_dataset = dataset["train"].select(range(200_000)).filter(length_filter)
    validation_dataset = dataset["train"].select(range(200_000, 205_000)).filter(length_filter)

    # NOTE: You should not change the test dataset
    test_dataset = dataset["validation"]
    return DatasetDict(
        {
            "train": train_dataset,
            "validation": validation_dataset,
            "test": test_dataset,
        }
    )


def create_data_collator(tokenizer, model):
    """
    Create data collator for sequence-to-sequence tasks.

    Args:
        tokenizer: Tokenizer object.
        model: Model object.

    Returns:
        DataCollatorForSeq2Seq instance.
    """
    return DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


def preprocess_function(examples, prefix, tokenizer, max_input_length, max_target_length):
    """
    Preprocess the data.

    Args:
        examples: Examples.
        prefix: Prefix.
        tokenizer: Tokenizer object.
        max_input_length: Maximum input length.
        max_target_length: Maximum target length.

    Returns:
        Model inputs.
    """
    inputs = [prefix + ex[SRC_LANG] for ex in examples["translation"]]
    targets = [ex[TGT_LANG] for ex in examples["translation"]]

    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def preprocess_data(raw_datasets: DatasetDict, tokenizer) -> DatasetDict:
    """
    Preprocess the data.

    Args:
        raw_datasets: Raw datasets.
        tokenizer: Tokenizer object.

    Returns:
        Tokenized datasets.
    """
    tokenizer.src_lang = SRC_LANG
    tokenizer.tgt_lang = TGT_LANG

    tokenized_datasets: DatasetDict = raw_datasets.map(
        function=lambda examples: preprocess_function(
            examples=examples,
            prefix="",
            tokenizer=tokenizer,
            max_input_length=MAX_INPUT_LENGTH,
            max_target_length=MAX_TARGET_LENGTH,
        ),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )
    return tokenized_datasets
