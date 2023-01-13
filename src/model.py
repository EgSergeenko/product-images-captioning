"""Model and feature extraction logic."""
from omegaconf import DictConfig
from transformers import (GPT2Tokenizer, PreTrainedModel, PreTrainedTokenizer,
                          VisionEncoderDecoderModel, ViTFeatureExtractor)


def get_feature_extractor(
    pretrained_feature_extractor: str,
) -> ViTFeatureExtractor:
    """Get a pretrained ViT feature extractor by the given name.

    Args:
        pretrained_feature_extractor: The pretrained ft name.

    Returns:
        The pretrained feature ViT extractor.
    """
    return ViTFeatureExtractor.from_pretrained(
        pretrained_feature_extractor,
    )


def get_tokenizer(
    pretrained_tokenizer: str,
) -> GPT2Tokenizer:
    """Get a pretrained GPT2 tokenizer by the given name.

    Args:
        pretrained_tokenizer: The pretrained tokenizer name.

    Returns:
        The pretrained GPT2 tokenizer.
    """
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
    tokenizer: PreTrainedTokenizer,
    config: DictConfig,
) -> PreTrainedModel:
    """Get and configure a pretrained EncoderDecoder model.

    Args:
        pretrained_encoder: Pretrained encoder name.
        pretrained_decoder: Pretrained decoder name.
        tokenizer: Pretrained tokenizer for encoder.
        config: Model configuration.

    Returns:
        VisionEncoder decoder from pretrained parts with applied configuration.
    """
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
    model.config.num_beam_groups = config.num_beam_groups
    model.config.early_stopping = config.early_stopping
    model.config.no_repeat_ngram_size = config.no_repeat_ngram_size
    model.config.length_penalty = config.length_penalty
    model.config.repetition_penalty = config.repetition_penalty
    model.config.diversity_penalty = config.diversity_penalty

    return model
