import os

import pandas as pd


def create_dataset(
    raw_dataset_filepath,
    processed_dataset_filepath,
    image_dir,
):
    dataset = pd.read_csv(
        raw_dataset_filepath,
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
    dataset.to_csv(
        processed_dataset_filepath,
        index=False,
        columns=['article_id', 'detail_desc'],
    )


def image_exists(article_id, image_dir):
    image_filepath = os.path.join(
        image_dir,
        article_id[:3],
        '{0}.jpg'.format(
            article_id,
        ),
    )
    return os.path.exists(image_filepath)


if __name__ == '__main__':
    # TODO CLI
    create_dataset(
        'data/articles.csv',
        'data/data.csv',
        'data/images',
    )
