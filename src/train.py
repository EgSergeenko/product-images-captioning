"""Train model on the train set and evaluate on the val set."""
import evaluate
import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data import ProductImageCaptionsDataset
from model import get_feature_extractor, get_model, get_tokenizer
from utils import get_logger, get_max_length, get_predictions


@hydra.main(version_base=None, config_path='../configs', config_name='train')
def train(config: DictConfig) -> None:
    """Run training.

    Args:
        config: The run configuration.
    """
    logger = get_logger(config.log_format)

    device = torch.device('cpu')
    if config.device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')

    train_data = pd.read_csv(
        config.data.train.csv_filepath,
        dtype={'article_id': str},
    )
    val_data = pd.read_csv(
        config.data.val.csv_filepath,
        dtype={'article_id': str},
    )

    feature_extractor = get_feature_extractor(
        config.model.encoder.feature_extractor.pretrained_name,
    )
    tokenizer = get_tokenizer(
        config.model.decoder.tokenizer.pretrained_name,
    )
    model = get_model(
        config.model.encoder.pretrained_name,
        config.model.decoder.pretrained_name,
        tokenizer,
        config.model,
    ).to(device)

    val_max_length = get_max_length(
        val_data['detail_desc'].tolist(),
        tokenizer,
    )
    references = [[reference] for reference in val_data['detail_desc']]
    train_dataset = ProductImageCaptionsDataset(
        data=train_data,
        image_dir=config.data.image_dir,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        max_length=config.data.max_length,
    )
    val_dataset = ProductImageCaptionsDataset(
        data=val_data,
        image_dir=config.data.image_dir,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        max_length=val_max_length,
    )
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.data.train.batch_size,
        shuffle=config.data.train.shuffle,
        drop_last=config.data.train.drop_last,
        pin_memory=True,
    )
    val_data_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.data.val.batch_size,
        shuffle=config.data.val.shuffle,
        drop_last=config.data.val.drop_last,
        pin_memory=True,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
    )
    scaler = None
    if device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()

    metric = evaluate.load('sacrebleu')

    best_score, current_step = 0, 0
    for _ in range(config.epochs):
        model.train()
        for batch in train_data_loader:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            train_step(
                model,
                pixel_values,
                labels,
                optimizer,
                scaler,
            )
            current_step += 1

            if current_step % config.eval_step == 0:
                predictions = get_predictions(
                    model,
                    tokenizer,
                    val_data_loader,
                    device,
                )
                score = metric.compute(
                    predictions=predictions,
                    references=references,
                )['score']
                logger.info(
                    'Step: {0}, SacreBLEU: {1}'.format(
                        current_step, score,
                    ),
                )
                if score > best_score:
                    logger.info('Updating score...')
                    best_score = score
                    logger.info('Saving model...')
                    feature_extractor.save_pretrained(
                        config.data.checkpoint.feature_extractor,
                    )
                    tokenizer.save_pretrained(
                        config.data.checkpoint.tokenizer,
                    )
                    model.save_pretrained(
                        config.data.checkpoint.model,
                    )


def train_step(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
) -> None:
    """Process one batch.

    Args:
        model: The trained model.
        pixel_values: Image features batch.
        labels: Text features batch.
        optimizer: The model optimizer.
        scaler: The gradient scaler.
    """
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


if __name__ == '__main__':
    train()
