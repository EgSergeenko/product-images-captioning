"""Script that creates the processed dataset from the raw one."""
import os

import click
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command()
@click.option('--input-dataset-filepath', required=True)
@click.option('--output-dir', required=True)
@click.option('--image-dir', required=True)
@click.option('--val-size', default=0.1, type=float)
@click.option('--test-size', default=0.1, type=float)
@click.option('--seed', default=42, type=int)
def create_dataset(
    input_dataset_filepath: str,
    output_dir: str,
    image_dir: str,
    val_size: float,
    test_size: float,
    seed: int,
) -> None:
    """Preprocess the raw dataset and split it into train/val/test.

    Args:
        input_dataset_filepath: A filepath to the raw dataset.
        output_dir: A directory path to save processed datasets in.
        image_dir: A directory path with dataset images.
        val_size: Val set size ratio.
        test_size: Test set size ratio.
        seed: Random seed for reproducibility.
    """
    dataset = pd.read_csv(
        input_dataset_filepath,
        dtype={'article_id': str},
    )
    dataset['image_exists'] = dataset['article_id'].apply(
        image_exists, image_dir=image_dir,
    )
    dataset = dataset[dataset['image_exists']]
    dataset.drop_duplicates(
        subset=['detail_desc'],
        inplace=True,
    )
    dataset.dropna(
        subset=['detail_desc'],
        inplace=True,
    )
    train_dataset, val_dataset, test_dataset = train_test_val_split(
        dataset,
        val_size,
        test_size,
        seed,
    )
    id_column = 'article_id'
    description_column = 'detail_desc'
    train_dataset.to_csv(
        os.path.join(output_dir, 'train.csv'),
        index=False,
        columns=[id_column, description_column],
    )
    val_dataset.to_csv(
        os.path.join(output_dir, 'val.csv'),
        index=False,
        columns=[id_column, description_column],
    )
    test_dataset.to_csv(
        os.path.join(output_dir, 'test.csv'),
        index=False,
        columns=[id_column, description_column],
    )


def image_exists(article_id: str, image_dir: str) -> bool:
    """Check if there is an image with given id.

    Args:
        article_id: Image ID (filepath).
        image_dir:  The directory path with dataset images.

    Returns:
        True if the image exists, False otherwise.
    """
    image_filepath = os.path.join(
        image_dir,
        article_id[:3],
        '{0}.jpg'.format(
            article_id,
        ),
    )
    return os.path.exists(image_filepath)


def train_test_val_split(
    dataset: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset into three parts in the specified proportions.

    Args:
        dataset: A dataset to split.
        val_size: Val set size ratio.
        test_size: Test set size ratio.
        seed: Random seed for reproducibility.

    Returns:
        Three subsets: train dataset, val dataset, test_dataset.
    """
    train_dataset, val_test_dataset = train_test_split(
        dataset,
        test_size=val_size + test_size,
        shuffle=True,
        random_state=seed,
    )
    val_dataset, test_dataset = train_test_split(
        val_test_dataset,
        test_size=test_size / (val_size + test_size),
        shuffle=True,
        random_state=seed,
    )
    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    create_dataset()
