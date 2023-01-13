"""Utils for operating with data."""
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ViTFeatureExtractor

from utils import add_special_tokens


class ProductImageCaptionsDataset(Dataset):
    """A class that produces pairs (image_features, text_features)."""

    def __init__(
        self,
        data: pd.DataFrame,
        image_dir: str,
        feature_extractor: ViTFeatureExtractor,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
    ) -> None:
        """Initialize the dataset instance.

        Args:
            data: Data about samples.
            image_dir: The directory path with dataset images.
            feature_extractor: The tool to extract image features.
            tokenizer: The tool to extract text features.
            max_length: Max token sequence length allowed.
        """
        self.data = data
        self.image_dir = image_dir
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache = {}

    def __len__(self) -> int:
        """Get the dataset length.

        Returns:
            The dataset length.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get the sample by index.

        Args:
            idx: The given index.

        Returns:
            The sample corresponding to this index.
        """
        return {
            'pixel_values': self._get_pixel_values(idx),
            'labels': self._get_labels(idx),
        }

    def _get_pixel_values(self, idx: int) -> torch.Tensor:
        product_id = self.data['article_id'][idx]
        image_filepath = os.path.join(
            self.image_dir,
            product_id[:3],
            '{0}.jpg'.format(
                product_id,
            ),
        )
        image = Image.open(image_filepath).convert('RGB')
        return self.feature_extractor(
            image,
            return_tensors='pt',
        ).pixel_values.squeeze()

    def _get_labels(self, idx: int) -> torch.Tensor:
        if idx in self.cache:
            return self.cache[idx]
        caption = add_special_tokens(
            self.data['detail_desc'][idx],
            self.tokenizer.bos_token,
            self.tokenizer.eos_token,
        )
        labels = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        ).input_ids.squeeze()
        labels = self._labels_mask(labels)
        self.cache[idx] = labels
        return labels

    def _labels_mask(self, labels: torch.Tensor) -> torch.Tensor:
        return torch.where(
            labels != self.tokenizer.pad_token_id,
            labels,
            -100,
        )
