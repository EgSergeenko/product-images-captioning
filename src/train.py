import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (GPT2Tokenizer, VisionEncoderDecoderModel,
                          ViTFeatureExtractor)

from data import ProductImageCaptionsDataset
from utils import get_max_length, get_prediction


def train():
    # TODO Config + CLI

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu',
    )

    data = pd.read_csv(
        'data/data.csv',
        dtype={'article_id': str},
    )

    tokenizer = get_tokenizer()
    feature_extractor = get_feature_extractor()

    max_length = get_max_length(
        data['detail_desc'],
        tokenizer,
    )

    model = get_model(tokenizer).to(device)

    dataset = ProductImageCaptionsDataset(
        data=data,
        image_dir='data/images',
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        max_length=64,
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=1e-4,
    )

    scaler = None
    if device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()

    sample = dataset[0]
    pixel_values, _ = sample['pixel_values'], sample['labels']

    epochs = 1
    for _ in range(epochs):
        train_epoch(
            model,
            data_loader,
            optimizer,
            scaler,
            device,
            25,
        )

        prediction = get_prediction(
            model,
            tokenizer,
            pixel_values.to(device),
        )
        print(prediction)
    model.save_pretrained('vit-distilgpt2')
    tokenizer.save_pretrained('vit-distilgpt2-tokenizer')


def train_epoch(
    model,
    data_loader,
    optimizer,
    scaler,
    device,
    log_step,
):
    model.train()

    for idx, batch in enumerate(data_loader):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        if scaler is None:
            loss = model(
                pixel_values=pixel_values,
                labels=labels,
            ).loss
            loss.backward()
            optimizer.step()
        else:
            with torch.autocast(device_type='cuda'):
                loss = model(
                    pixel_values=pixel_values,
                    labels=labels,
                ).loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if (idx + 1) % log_step == 0:
            print(
                'Step: {0}, Loss: {1:.5f}'.format(
                    idx + 1,
                    loss.item(),
                ),
            )


def get_feature_extractor():
    return ViTFeatureExtractor.from_pretrained(
        'google/vit-base-patch16-224-in21k',
    )


def get_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained(
        'distilgpt2',
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_model(
    tokenizer,
    max_length=64,
    no_repeat_ngram_size=3,
    length_penalty=2.0,
    num_beams=4,
):
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        'google/vit-base-patch16-224-in21k', 'distilgpt2',
    )
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.sep_token_id

    model.decoder.resize_token_embeddings(len(tokenizer))
    model.config.vocab_size = model.config.decoder.vocab_size

    model.config.max_length = max_length
    model.config.no_repeat_ngram_size = no_repeat_ngram_size
    model.config.length_penalty = length_penalty
    model.config.num_beams = num_beams

    return model


if __name__ == '__main__':
    train()
