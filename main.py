import argparse
import logging
import pathlib
import csv
from contextlib import redirect_stdout
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from models import ClassifierBuilder
from utils import get_task_data, prepare_data
from numpy import zeros
from datetime import date


def test_model(model, args, test_ds, log_dir=None):
    """
    Evaluate trained model
    :param model: trained keras.Model instance
    :param args: program arguments
    :param test_ds: pandas Dataframe test data
    :param log_dir: pathlib.Path directory to log to
    :return: dictionary of {metric_name: list of metric results}
    """

    test_x, test_y = prepare_data(test_ds, args)
    result = model.evaluate(test_x,
                            test_y,
                            batch_size=args.batch_size,
                            callbacks=[])
    with open(log_dir.joinpath('test.csv'), encoding='utf-8', mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(model.metrics_names)
        writer.writerow(result)

    return dict(zip(model.metrics_names, result))


def run_model(model: tf.keras.Model,
              args,
              train_ds,
              log_dir: pathlib.Path = None):
    """
    Fit model on train/validate/test datasets.
    :param log_dir: pathlib.Path directory to log to
    :param model: keras.Model instance to run
    :param args: program arguments
    :param train_ds: pandas Dataframe training examples
    :return: keras History object
    """

    with open(log_dir.joinpath('cls_summary.txt'), 'w') as f:
        with redirect_stdout(f):
            model.summary()

    tf.keras.utils.plot_model(model,
                              to_file=log_dir.joinpath('model.png'),
                              show_shapes=True,
                              show_dtype=True,
                              show_layer_names=True)

    callbacks = []
    if args.validate:
        train_ds, val_ds = train_test_split(train_ds,
                                            random_state=42,
                                            test_size=args.val_split,
                                            stratify=train_ds[['rating']],
                                            shuffle=not args.no_shuffle)
        val_data = (prepare_data(val_ds, args))

    csv_back = tf.keras.callbacks.CSVLogger(filename=log_dir.joinpath('train.csv'), append=False)
    callbacks.extend([csv_back])

    if not args.no_early_stopping:
        early_back = tf.keras.callbacks.EarlyStopping(monitor='val_base_loss' if args.validate else 'base_loss',
                                                      min_delta=0,
                                                      patience=3,
                                                      mode='min',
                                                      restore_best_weights=True)
        callbacks.append(early_back)

    train_x, train_y = prepare_data(train_ds, args)

    model_history = model.fit(train_x,
                              train_y,
                              batch_size=args.batch_size,
                              epochs=args.epochs,
                              validation_data=val_data if args.validate else None,
                              callbacks=callbacks)
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
        help="Number of hidden units to use in classifiers. Defaults to 256.",
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
        "--dropout",
        default=0.4,
        type=float,
        help="How much dropout to apply. Expects a float in range [0,1]. Defaults to 0.4."
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

    parser.add_argument(
        "--dp",
        action="store_true",
        help="Add Laplace noise to embedding. Defaults to False."
    )

    parser.add_argument(
        "--epsilon",
        default=0.1,
        type=float,
        help="Epsilon parameter for DP-compliant noise generation."
    )

    parser.add_argument(
        "--balance",
        action="store_true",
        help="Balance class weights. Defaults False."
    )

    parser.add_argument(
        "--learning_rate",
        default=0.001,
        type=float,
        help="Learning rate for optimiser. Defaults to 0.001."
    )

    parser.add_argument(
        "--cv",
        default=4,
        type=int,
        help="Number of cross-validation runs to do. Defaults to 4."
    )

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%m-%y %H:%M:%S', level=logging.DEBUG)

    if args.no_gpu:
        tf.config.set_visible_devices([], 'GPU')
    else:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    ds, args.labels, l_labels, b_labels = get_task_data(args)
    skf = StratifiedKFold(n_splits=args.cv, random_state=42, shuffle=True)

    split = 1
    for train_idx, test_idx in skf.split(X=zeros(len(ds)), y=ds.rating):
        train_ds = ds.iloc[train_idx]
        test_ds = ds.iloc[test_idx]

        for identifier in ['birth_year', 'loc', 'gender']:
            if split > 3:
                args.identifier = identifier
                if identifier == 'birth_year':
                    args.priv_labels = b_labels
                elif identifier == 'loc':
                    args.priv_labels = l_labels
                else:
                    args.priv_labels = ds.gender.unique()
                model = ClassifierBuilder(args).get_classifier("combined_classifier")
                log_dir = pathlib.Path.cwd().joinpath(args.log_dir).joinpath(str(date.today())).joinpath(identifier)
                log_dir = log_dir.joinpath(f"{f'{args.tag}' if args.tag else ''}_{split}")
                log_dir.mkdir(parents=True, exist_ok=True)
                logging.debug(f"Training model for {identifier} {f'{args.tag}' if args.tag else ''}_{split}...")
                history = run_model(model,
                                    args,
                                    train_ds=train_ds,
                                    log_dir=log_dir)
                #logging.debug(history.history)
                logging.debug(f"Testing {identifier} {f'{args.tag}' if args.tag else ''}_{split}...")
                result = test_model(model,
                                    args,
                                    test_ds=test_ds,
                                    log_dir=log_dir)
                #logging.debug(result)
        split += 1


if __name__ == '__main__':
    main()
