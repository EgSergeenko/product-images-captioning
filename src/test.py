import evaluate
import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import (GPT2Tokenizer, VisionEncoderDecoderModel,
                          ViTFeatureExtractor)

from data import ProductImageCaptionsDataset
from utils import get_logger, get_max_length, get_predictions


@hydra.main(version_base=None, config_path='../configs', config_name='test')
def test(config: DictConfig) -> None:
    logger = get_logger(config.log_format)

    device = torch.device('cpu')
    if config.device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')

    test_data = pd.read_csv(
        config.data.test.csv_filepath,
        dtype={'article_id': str},
    )

    feature_extractor = ViTFeatureExtractor.from_pretrained(
        config.model.encoder.feature_extractor.pretrained_name,
    )
    tokenizer = GPT2Tokenizer.from_pretrained(
        config.model.decoder.tokenizer.pretrained_name,
    )
    model = VisionEncoderDecoderModel.from_pretrained(
        config.model.pretrained_name,
    ).to(device)

    test_max_length = get_max_length(
        test_data['detail_desc'],
        tokenizer,
    )
    references = [[reference] for reference in test_data['detail_desc']]
    test_dataset = ProductImageCaptionsDataset(
        data=test_data,
        image_dir=config.data.image_dir,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        max_length=test_max_length,
    )
    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.data.test.batch_size,
        shuffle=config.data.test.shuffle,
        drop_last=config.data.test.drop_last,
        pin_memory=True,
    )
    predictions = get_predictions(
        model,
        tokenizer,
        test_data_loader,
        device,
    )
    metric = evaluate.load('sacrebleu')
    score = metric.compute(
        predictions=predictions,
        references=references,
    )['score']
    logger.info(
        'SacreBLEU: {0}'.format(score),
    )


if __name__ == '__main__':
    test()
