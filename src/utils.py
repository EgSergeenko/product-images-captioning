"""Helper functions."""
import logging

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer


def get_max_length(
    texts: list[str],
    tokenizer: PreTrainedTokenizer,
) -> int:
    """Get max length in tokens in the given corpus.

    Args:
        texts: Text corpus.
        tokenizer: The tool to get tokens.

    Returns:
        The max token sequence length.
    """
    max_length = 0
    for text in texts:
        text = add_special_tokens(
            text,
            tokenizer.bos_token,
            tokenizer.eos_token,
        )
        labels = tokenizer(
            text,
            return_tensors='pt',
        ).input_ids
        max_length = max(max_length, labels.size()[-1])
    return max_length


def get_predictions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    data_loader: DataLoader,
    device: torch.device,
) -> list[str]:
    """Get predictions made by the model.

    Args:
        model: The model to use for predictions.
        tokenizer: The tokenizer for the model.
        data_loader: Data to predict.
        device: Device to use (CPU/GPU).

    Returns:
        Predicted sequences.
    """
    predictions = []
    model.eval()
    for batch in data_loader:
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            output_ids = model.generate(
                pixel_values,
                max_length=labels.size()[-1],
                return_dict_in_generate=True,
            ).sequences

        prediction = tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
        )
        predictions.extend(prediction)

    return predictions


def add_special_tokens(
    text: str,
    bos_token: str,
    eos_token: str,
) -> str:
    """Add bos and eos tokens to a sentence.

    Args:
        text: The given sentence.
        bos_token: '[BOS]'
        eos_token: '[EOS]'

    Returns:
        The sentence with special tokens.
    """
    return '{0} {1} {2}'.format(
        bos_token,
        text,
        eos_token,
    )


def get_stream_handler(log_format: str) -> logging.StreamHandler:
    """Get a stream handler.

    Args:
        log_format: Format of log messages.

    Returns:
        The configured stream handler.
    """
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(
        logging.Formatter(log_format, style='{'),
    )
    return stream_handler


def get_logger(log_format: str) -> logging.Logger:
    """Get a logger for tracking experiment.

    Args:
        log_format: Format of the log messages.

    Returns:
        The configured logger.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(get_stream_handler(log_format))
    return logger
