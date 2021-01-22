import argparse
import csv
import logging
import pathlib
from datetime import datetime
from contextlib import redirect_stdout
import tensorflow as tf
from models import ClassifierBuilder, CheckpointCallback
from utils import get_task_data


def run_model(model: tf.keras.Model,
              args,
              train_ds: tf.data.Dataset,
              test_ds: tf.data.Dataset,
              val_ds: tf.data.Dataset = None):
    """
    Fit and evaluate model on train/validate/test datasets.
    :param model: keras.Model instance to run.
    :param args: program arguments
    :param train_ds: tf.data.Dataset containing training examples
    :param test_ds: tf.data.Dataset containing test examples
    :param val_ds: tf.data.Dataset containing validation examples
    :return: a keras History object
    """

    now = datetime.now().strftime('%y%m%d-%H%M%S')
    log_dir = pathlib.Path.cwd().joinpath(args.log_dir, f"{now}{f'-{args.tag}' if args.tag else ''}")
    log_dir.mkdir()
    chk_dir = log_dir.joinpath('checkpoints')
    chk_dir.mkdir()

    with open(log_dir.joinpath('cls_summary.txt'), 'w') as f:
        with redirect_stdout(f):
            model.summary()
    tf.keras.utils.plot_model(model,
                              to_file=log_dir.joinpath('model.png'),
                              show_shapes=True,
                              show_dtype=True,
                              show_layer_names=True)

    csv_back = tf.keras.callbacks.CSVLogger(filename=log_dir.joinpath('train.csv'), append=False)
    tb_back = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                             update_freq=10,
                                             histogram_freq=0,
                                             profile_batch=0,
                                             write_graph=False)
    chk_back = CheckpointCallback(filepath=chk_dir.joinpath('model.{batch:02d}.weights'),
                                  verbose=1,
                                  monitor='loss',
                                  save_freq=args.checkpoint_every,
                                  save_weights_only=True)

    if args.no_early_stopping:
        callbacks = [csv_back, tb_back, chk_back]
    else:
        early_back = tf.keras.callbacks.EarlyStopping(monitor='val_loss' if args.validate else 'loss',
                                                      min_delta=0,
                                                      patience=5,
                                                      mode='min',
                                                      restore_best_weights=True)

        callbacks = [csv_back, tb_back, early_back, chk_back]

    if args.validate:
        model_history = model.fit(train_ds,
                                  epochs=args.epochs,
                                  validation_data=val_ds,
                                  callbacks=callbacks)
    else:
        model_history = model.fit(train_ds,
                                  epochs=args.epochs,
                                  callbacks=callbacks)

    result = model.evaluate(test_ds)
    with open(log_dir.joinpath('test.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=model.metrics_names)
        result = dict(zip(model.metrics_names, result))
        writer.writeheader()
        writer.writerow(result)

    return model_history


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log_dir",
        default='logs',
        type=str,
        help="The logging directory.",
    )

    parser.add_argument(
        "--tag",
        default=None,
        type=str,
        help="Tag to add to logs for run identification."
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
        default=20,
        type=int,
        help="Total number of training epochs to perform. Defaults to 20.",
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
        "--adversarial",
        action='store_true',
        help="Train with an adversarial objective. Defaults to False."
    )

    parser.add_argument(
        "--dropout_rate",
        default=None,
        type=float,
        help="How much dropout to apply after feature extractor. Expects a float in range [0,1]."
    )

    parser.add_argument(
        "--data_split",
        default=None,
        type=int,
        help="Number of rows from total dataset to use."
    )

    parser.add_argument(
        "--train_split",
        default=0.7,
        type=float,
        help="Proportion of available data to use in training. Expects a float in range [0,1]. Defaults to 0.7."
    )

    parser.add_argument(
        "--val_split",
        default=0.2,
        type=float,
        help="Proportion of training data to use in validation. Expects a float in range [0,1]. Defaults to 0.2.",
    )

    parser.add_argument(
        "--no_early_stopping",
        action="store_true",
        help="Turn off early training stopping when loss ceases to fall. Defaults to False."
    )

    parser.add_argument(
        "--checkpoint_every",
        default=1_000,
        type=int,
        help="Number of training batches to save a checkpoint after. Defaults to 1_000."
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

    parser.add_argument(
        "--hplambda",
        default=1.0,
        type=float,
        help="Regularization parameter for the gradient reversal layer. Defaults to 1.0."
    )

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%m-%y %H:%M:%S', level=logging.DEBUG)

    if args.no_gpu:
        tf.config.set_visible_devices([], 'GPU')
    else:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    train_ds, test_ds, val_ds, n_labels, n_priv_labels = get_task_data(args)

    model = ClassifierBuilder(args).get_classifier(n_labels, n_priv_labels, "combined_classifier")

    history = run_model(model, args, train_ds=train_ds, test_ds=test_ds, val_ds=val_ds)
    logging.debug(history.history)


if __name__ == '__main__':
    main()
