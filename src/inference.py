import hydra
import torch
from omegaconf import DictConfig
from PIL import Image
from transformers import (GPT2Tokenizer, VisionEncoderDecoderModel,
                          ViTFeatureExtractor)

from utils import get_logger


@hydra.main(version_base=None, config_path='../configs', config_name='inference')
def inference(config: DictConfig) -> None:
    logger = get_logger(config.log_format)

    device = torch.device('cpu')
    if config.device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')

    feature_extractor = ViTFeatureExtractor.from_pretrained(
        config.model.encoder.feature_extractor.pretrained_name,
    )
    tokenizer = GPT2Tokenizer.from_pretrained(
        config.model.decoder.tokenizer.pretrained_name,
    )
    model = VisionEncoderDecoderModel.from_pretrained(
        config.model.pretrained_name,
    ).to(device).eval()

    image = Image.open(config.data.image_filepath).convert('RGB')
    pixel_values = feature_extractor(
        image,
        return_tensors='pt',
    ).pixel_values
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values,
            max_length=config.data.max_length,
            return_dict_in_generate=True,
        ).sequences
    prediction = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True,
    )
    logger.info('Image: {0}'.format(
            config.data.image_filepath,
        ),
    )
    logger.info('Prediction: {0}'.format(
            prediction[0],
        ),
    )


if __name__ == '__main__':
    inference()
