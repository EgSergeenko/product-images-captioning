from omegaconf import DictConfig
from transformers import (GPT2Tokenizer, PreTrainedModel,
                          VisionEncoderDecoderModel, ViTFeatureExtractor)


def get_feature_extractor(
    pretrained_feature_extractor: str,
) -> ViTFeatureExtractor:
    return ViTFeatureExtractor.from_pretrained(
        pretrained_feature_extractor,
    )


def get_tokenizer(
    pretrained_tokenizer: str,
) -> GPT2Tokenizer:
    tokenizer = GPT2Tokenizer.from_pretrained(
        pretrained_tokenizer, use_fast=True,
    )
    tokens_to_add = {
        'pad_token': '[PAD]',
        'bos_token': '[BOS]',
        'eos_token': '[EOS]',
    }
    tokenizer.add_special_tokens(tokens_to_add)
    return tokenizer


def get_model(
    pretrained_encoder: str,
    pretrained_decoder: str,
    tokenizer: GPT2Tokenizer,
    config: DictConfig,
) -> PreTrainedModel:
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        pretrained_encoder, pretrained_decoder,
    )

    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    model.decoder.resize_token_embeddings(len(tokenizer))
    model.config.vocab_size = model.config.decoder.vocab_size

    model.config.max_length = config.max_length
    model.config.num_beams = config.num_beams
    model.config.early_stopping = config.early_stopping
    model.config.no_repeat_ngram_size = config.no_repeat_ngram_size
    model.config.length_penalty = config.length_penalty
    model.config.repetition_penalty = config.repetition_penalty

    return model
