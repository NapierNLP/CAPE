import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
from datasets import load_dataset
from transformers import BertTokenizer
import models
from contextlib import redirect_stdout
import argparse
import csv
import logging
import datetime
import pathlib


def split_dataset(dataset: tf.data.Dataset, train_data_fraction: float):
    """
    Splits a dataset of type tf.data.Dataset into a training and testing dataset using given ratio.
    Fractions are rounded up to two decimal places.
    :param dataset: the input dataset to split.
    :param train_data_fraction: the fraction to use as training data as a float between 0 and 1.
    :return: a tuple of two tf.data.Datasets as (train, test)
    """

    train_data_percent = round(train_data_fraction * 100)
    if not (0 <= train_data_percent <= 100):
        raise ValueError("fraction must be in the range [0,1]")

    dataset = dataset.enumerate()
    test_dataset = dataset.filter(lambda f, data: f % 100 > train_data_percent)
    train_dataset = dataset.filter(lambda f, data: f % 100 <= train_data_percent)

    train_dataset = train_dataset.map(lambda f, data: data)
    test_dataset = test_dataset.map(lambda f, data: data)

    return train_dataset, test_dataset


def get_task_data(args):
    """
    Return a tf.data.Dataset object containing the data and labels loaded for the selected task.
    :param args: command line arguments
    :return: td.data.Dataset, number of labels, number of private variable labels
    """
    data = load_dataset('trust_dataset.py',
                        'uk',
                        cache_dir=args.cache_dir,
                        split=f'train[:{int(args.data_split * 100)}%]')
    n_labels = len(data.unique('label'))
    n_priv_labels = len(data.unique('priv_label'))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=args.cache_dir, do_lower_case=True)
    logging.debug("Applying tokenization...")
    data = data.map(lambda e: tokenizer(e['text'],
                                        truncation=True,
                                        padding='max_length',
                                        max_length=args.max_length,
                                        add_special_tokens=True,
                                        return_token_type_ids=False),
                    batched=True)
    logging.debug("Tokenization complete.")
    logging.debug("Converting to TF tensors...")
    data.set_format(type='tensorflow', columns=['input_ids', 'attention_mask', 'label', 'priv_label'])

    train_features = {x: data[x].to_tensor(default_value=0, shape=[None, args.max_length]) for x in
                      ['input_ids', 'attention_mask']}
    labels = tf.keras.utils.to_categorical(data['label'], num_classes=n_labels)
    priv_labels = tf.keras.utils.to_categorical(data['priv_label'], num_classes=n_priv_labels)
    label_dict = {"base": labels, "attacker": priv_labels}
    logging.debug("Converted.")
    logging.debug("Creating TF dataset...")
    dataset = tf.data.Dataset.from_tensor_slices((train_features, label_dict)).batch(args.batch_size)
    logging.debug("Dataset created.")
    return dataset, n_labels, n_priv_labels


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log_dir",
        default='logs',
        type=str,
        help="The logging directory.",
    )

    parser.add_argument(
        "--cache_dir",
        default='cache',
        type=str,
        help="Cache directory for data and model downloads.",
    )

    parser.add_argument(
        "--max_length",
        default=512,
        type=int,
        help="Maximum input sequence length. Defaults to 512.",
    )

    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size for training. Defaults to 64.",
    )

    parser.add_argument(
        "--epochs",
        default=15,
        type=int,
        help="Total number of training epochs to perform. Defaults to 15.",
    )

    parser.add_argument(
        "--hidden_size",
        default=256,
        type=int,
        help="Number of hidden units to use in linear classifier. Defaults to 256.",
    )

    parser.add_argument(
        "--no_shuffle",
        action='store_true',
        help="Don't shuffle the dataset before training and evaluating. Defaults to False."
    )

    parser.add_argument(
        "--validate",
        action='store_true',
        help="Whether to validate during training. Defaults to False."
    )

    parser.add_argument(
        "--data_split",
        default=1.0,
        type=float,
        help="Proportion of total dataset to use. Defaults to 1.0."
    )

    parser.add_argument(
        "--train_split",
        default=0.7,
        type=float,
        help="Proportion of available data to use in training. Defaults to 0.7."
    )

    parser.add_argument(
        "--val_split",
        default=0.2,
        type=float,
        help="Proportion of training data to use in validation. Defaults to 0.2.",
    )

    parser.add_argument(
        "--no_early_stopping",
        action="store_true",
        help="Turn off early training stopping when loss ceases to fall. Defaults to False."
    )

    parser.add_argument(
        "--checkpoint_every",
        default=100,
        type=int,
        help="Number of training batches to save a checkpoint after. Defaults to 100."
    )

    parser.add_argument(
        "--no_gpu",
        action="store_true",
        help="Force GPU off. Defaults to False."
    )

    parser.add_argument(
        "--embed_length",
        default=768,
        type=int,
        help="Length of embedding. Defaults to 768."
    )

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%m-%y %H:%M:%S', level=logging.DEBUG)

    if args.no_gpu:
        tf.config.set_visible_devices([], 'GPU')
    else:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    dataset, n_labels, n_priv_labels = get_task_data(args)
    train_ds, test_ds = split_dataset(dataset, args.train_split)
    if args.validate:
        val_ds, train_ds = split_dataset(dataset, args.val_split)
    else:
        val_ds = None

    model_builder = models.ClassifierBuilder(args)

    classifier = model_builder.get_classifier(n_labels, n_priv_labels, "base_classifier")

    now = datetime.datetime.now().strftime('%Y%m%d')
    log_dir = pathlib.Path.cwd().joinpath(args.log_dir)
    with open(log_dir.joinpath('base_cls_summary.txt'), 'w') as f:
        with redirect_stdout(f):
            classifier.summary()

    csv_back = CSVLogger(filename=log_dir.joinpath('train.csv'), append=False)
    tb_back = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                             update_freq=10,
                                             histogram_freq=1,
                                             write_graph=False)
    callbacks = [csv_back, tb_back]

    model_history = classifier.fit(train_ds,
                                   epochs=args.epochs,
                                   validation_data=val_ds,
                                   callbacks=callbacks)

    result = classifier.evaluate(test_ds)
    with open(log_dir.joinpath('test.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=classifier.metrics_names)
        result = dict(zip(classifier.metrics_names, result))
        writer.writeheader()
        writer.writerow(result)


if __name__ == '__main__':
    main()
