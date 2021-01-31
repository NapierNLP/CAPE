import ast
import os
import pathlib
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import pandas as pd
import logging
from tensorflow.keras.utils import to_categorical
from transformers import BertTokenizer


def clean_file(file):
    """
    Clean up downloaded jsonl file
    :param file: downloaded file
    :return: list of Python dicts
    """
    clean_list = []
    if file.endswith('.jsonl.tmp'):
        with open(file, encoding='utf-8', mode='r') as f:
            for line in f:
                data = ast.literal_eval(line)
                if 'gender' in data.keys() and data['gender']:
                    for review in data['reviews']:
                        if review['rating'] and review['text']:
                            new_data = {
                                'text': review['text'][0],
                                'rating': int(review['rating']),
                                'gender': data['gender']
                            }
                            clean_list.append(new_data)
    return clean_list


def get_task_data(args):
    """
    Return the data and labels loaded for the selected task.
    :param args: program arguments
    :return: pandas Dataframe
    """
    url = "https://bitbucket.org/lowlands/release/raw/fd60e8b4fbb12f0175e0f26153e289bbe2bfd71c/WWW2015/data/united_kingdom.auto-adjusted_gender.NUTS-regions.jsonl.zip"

    data_dir = pathlib.Path.cwd().joinpath(args.cache_dir, 'data')
    data_dir.mkdir(exist_ok=True, parents=True)

    if not data_dir.joinpath('data.json').is_file():
        logging.debug("Downloading and extracting...")
        with urlopen(url) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall(str(data_dir))
        logging.debug("Downloaded.")
        logging.debug("Cleaning and saving data...")
        for filename in os.listdir(data_dir):
            if filename.endswith('.jsonl.tmp'):
                clean = clean_file(os.path.join(data_dir, filename))
                df = pd.DataFrame(clean)
                df['gender'] = df['gender'].astype('category').cat.codes.astype(int)
                df['rating'] = df.rating.astype(int)
                with open(data_dir.joinpath("data.json"), encoding='utf-8', mode='w') as f:
                    df.to_json(f, orient='records', lines=True)
        logging.debug("Saved.")

    df = pd.read_json(data_dir.joinpath('data.json'), orient='records', lines=True)
    if args.data_split:
        df = df.sample(frac=1).reset_index(drop=True)[:args.data_split]

    return df


def prepare_data(df, args):
    """
    Convert dataframe to inputs for Keras model
    :param df: Pandas dataframe with input examples, labels, private information labels
    :param args: program arguments
    :return: x: dict containing token ids, attention mask ids, y: dict containing one-hot encoded labels, binary private variable labels
    """
    df = df.copy()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                              cache_dir=args.cache_dir,
                                              do_lower_case=True)

    encoded = tokenizer(df['text'].to_list(),
                        truncation=True,
                        padding='max_length',
                        max_length=args.max_length,
                        add_special_tokens=True,
                        return_token_type_ids=False,
                        return_tensors='tf')

    df['rating'] = df['rating'].apply(lambda x: x - 1)
    x = {name: encoded[name] for name in ['input_ids', 'attention_mask']}
    y = {"base": to_categorical(df['rating'].to_numpy(), num_classes=None), "attacker": df['gender'].to_numpy()}

    return x, y
