import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import add_special_tokens


class ProductImageCaptionsDataset(Dataset):
    def __init__(
        self,
        data,
        image_dir,
        feature_extractor,
        tokenizer,
        max_length,
    ):
        self.data = data
        self.image_dir = image_dir
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'pixel_values': self._get_pixel_values(idx),
            'labels': self._get_labels(idx),
        }

    def _get_pixel_values(self, idx):
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

    def _get_labels(self, idx):
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

    def _labels_mask(self, labels):
        return torch.where(
            labels != self.tokenizer.pad_token_id,
            labels,
            -100,
        )
