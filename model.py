from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from constants import MODEL_CHECKPOINT, SRC_LANG, TGT_LANG


def initialize_tokenizer() -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    """
    Initialize a tokenizer for sequence-to-sequence tasks.

    Returns:
        A tokenizer for sequence-to-sequence tasks.

    NOTE: You are free to change this. But make sure the tokenizer is the same as the model.
    """
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=MODEL_CHECKPOINT, src_lang=SRC_LANG, tgt_lang=TGT_LANG
    )
    tokenizer.src_lang = SRC_LANG
    tokenizer.tgt_lang = TGT_LANG
    return tokenizer


def initialize_model() -> PreTrainedModel:
    """
    Initialize a model for sequence-to-sequence tasks. You are free to change this,
    not only seq2seq models, but also other models like BERT, or even LLMs.

    Returns:
        A model for sequence-to-sequence tasks.

    NOTE: You are free to change this.
    """
    tokenizer_for_lang_id = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_CHECKPOINT)
    model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_CHECKPOINT
    )
    if hasattr(tokenizer_for_lang_id, "get_lang_id"):
        model.config.forced_bos_token_id = tokenizer_for_lang_id.get_lang_id(TGT_LANG)
    model.config.use_cache = False
    return model
