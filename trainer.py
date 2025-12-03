import inspect

from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, Trainer, TrainingArguments

from constants import MAX_TARGET_LENGTH, OUTPUT_DIR
from evaluation import compute_metrics


def create_training_arguments() -> TrainingArguments:
    """
    Create and return the training arguments for the model.

    Returns:
        Training arguments for the model.

    NOTE: You can change the training arguments as needed.
    # Below is an example of how to create training arguments. You are free to change this.
    # ref: https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments
    """
    kwargs = dict(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.05,
        logging_steps=200,
        save_steps=2000,
        eval_steps=2000,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        max_grad_norm=1.0,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        fp16=True,
        gradient_accumulation_steps=16,
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        label_smoothing_factor=0.1,
    )

    sig_params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    if "evaluation_strategy" in sig_params:
        kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in sig_params:
        kwargs["eval_strategy"] = "steps"

    training_args = Seq2SeqTrainingArguments(**kwargs)

    return training_args


def create_data_collator(tokenizer, model):
    """
    Create data collator for sequence-to-sequence tasks.

    Args:
        tokenizer: Tokenizer object.
        model: Model object.

    Returns:
        DataCollatorForSeq2Seq instance.

    NOTE: You are free to change this. But make sure the data collator is the same as the model.
    """
    base_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    def collate_fn(features):
        batch = base_collator(features)
        # Some environments/models may surface decoder_inputs_embeds; ensure only decoder_input_ids are used.
        batch.pop("decoder_inputs_embeds", None)
        return batch

    return collate_fn


class _SafeSeq2SeqTrainer(Seq2SeqTrainer):
    """Drop decoder_inputs_embeds if it slips into the batch."""

    def prepare_inputs(self, inputs):
        inputs = super().prepare_inputs(inputs)
        # Drop decoder_inputs_embeds defensively to avoid conflicts in m2m100 forward.
        inputs.pop("decoder_inputs_embeds", None)
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Ensure decoder_inputs_embeds is not forwarded alongside decoder_input_ids.
        inputs = dict(inputs)
        inputs.pop("decoder_inputs_embeds", None)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        inputs = dict(inputs)
        inputs.pop("decoder_inputs_embeds", None)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


def build_trainer(model, tokenizer, tokenized_datasets) -> Trainer:
    """
    Build and return the trainer object for training and evaluation.

    Args:
        model: Model for sequence-to-sequence tasks.
        tokenizer: Tokenizer object.
        tokenized_datasets: Tokenized datasets.

    Returns:
        Trainer object for training and evaluation.

    NOTE: You are free to change this. But make sure the trainer is the same as the model.
    """
    data_collator = create_data_collator(tokenizer, model)
    training_args: TrainingArguments = create_training_arguments()

    return _SafeSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),
    )
