import argparse
import logging
import pathlib
from datetime import datetime
from contextlib import redirect_stdout
import tensorflow as tf
from models import ClassifierBuilder
from utils import get_task_data
from tensorboard.plugins.hparams import api as hp


def run_model(model: tf.keras.Model,
              args,
              train_ds: tf.data.Dataset,
              val_ds: tf.data.Dataset = None,
              log_dir=None):
    """
    Fit and evaluate model on train/validate/test datasets.
    :param log_dir:
    :param model: keras.Model instance to run.
    :param args: program arguments
    :param train_ds: tf.data.Dataset containing training examples
    :param val_ds: tf.data.Dataset containing validation examples
    :return: a keras History object
    """

    if not args.tuning:
        now = datetime.now().strftime('%y%m%d-%H%M%S')
        log_dir = log_dir.joinpath(f"{now}{f'-{args.tag}' if args.tag else ''}")
        log_dir.mkdir()

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
                                             update_freq='epoch',
                                             histogram_freq=0,
                                             profile_batch=0,
                                             write_graph=False)

    if args.no_early_stopping:
        callbacks = [csv_back, tb_back]
    else:
        early_back = tf.keras.callbacks.EarlyStopping(monitor='val_base_loss' if args.validate else 'base_loss',
                                                      min_delta=0,
                                                      patience=5,
                                                      mode='min',
                                                      restore_best_weights=True)

        callbacks = [csv_back, tb_back, early_back]

    if args.validate:
        model_history = model.fit(train_ds,
                                  epochs=args.epochs,
                                  validation_data=val_ds,
                                  callbacks=callbacks)
    else:
        model_history = model.fit(train_ds,
                                  epochs=args.epochs,
                                  callbacks=callbacks)
    return model_history


def run_tune(model: tf.keras.Model, log_dir, hparams: dict, args, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, test_ds: tf.data.Dataset):
    with tf.summary.create_file_writer(str(log_dir)).as_default():
        hp.hparams(hparams)
        run_model(model,
                  args,
                  train_ds=train_ds,
                  val_ds=val_ds,
                  log_dir=log_dir)
        metrics = model.evaluate(test_ds, verbose=1)
        for idx, metric in enumerate(model.metrics_names):
            if metric in ['base_acc', 'base_prec', 'base_rec', 'attacker_acc', 'attacker_prec', 'attacker_rec']:
                logging.debug(f'{metric}: {metrics[idx]}')
                tf.summary.scalar(metric, metrics[idx], step=1)


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
        "--tuning",
        action="store_true",
        help="Tensorboard Hparam tuning. Will ignore dropout, epsilon, hidden size switches. Defaults to False."
    )

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%m-%y %H:%M:%S', level=logging.DEBUG)

    if args.no_gpu:
        tf.config.set_visible_devices([], 'GPU')
    else:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    train_ds, test_ds, val_ds, n_labels, n_priv_labels = get_task_data(args)
    cls = ClassifierBuilder(args)

    if args.tuning:
        HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([128, 256, 512]))
        HP_EPSILON = hp.HParam('epsilon', hp.Discrete([0.01, 0.1, 0.5, 0.9]))
        HP_LAMBDA = hp.HParam('lambda', hp.Discrete([0.1, 0.5, 1.0]))

        log_dir = pathlib.Path.cwd().joinpath('logs', 'hparam_tuning')
        with tf.summary.create_file_writer(str(log_dir)).as_default():
            hp.hparams_config(
                hparams=[HP_NUM_UNITS, HP_EPSILON, HP_LAMBDA],
                metrics=[hp.Metric('base_acc', display_name='Base Accuracy'),
                         hp.Metric('base_prec', display_name='Base Precision'),
                         hp.Metric('base_rec', display_name='Base Recall'),
                         hp.Metric('attacker_acc', display_name='Attacker Accuracy'),
                         hp.Metric('attacker_prec', display_name='Attacker Precision'),
                         hp.Metric('attacker_rec', display_name='Attacker Recall')]
            )

            session_num = 0
            for num_units in HP_NUM_UNITS.domain.values:
                for epsilon in HP_EPSILON.domain.values:
                    for lbd in HP_LAMBDA.domain.values:
                        hparams = {
                            'HP_NUM_UNITS': num_units,
                            'HP_EPSILON': epsilon,
                            'HP_LAMBDA': lbd
                        }
                        run_name = "run-%d" % session_num
                        logging.debug('--- Starting trial: %s' % run_name)
                        logging.debug({h: hparams[h] for h in hparams})

                        model = cls.get_classifier(n_labels,
                                                   n_priv_labels,
                                                   "combined_classifier",
                                                   hparams)

                        run_tune(model=model,
                                 log_dir=log_dir.joinpath(str(session_num)),
                                 hparams=hparams,
                                 args=args,
                                 train_ds=train_ds,
                                 val_ds=val_ds,
                                 test_ds=test_ds)

                        session_num += 1
                        del model
    else:
        log_dir = args.log_dir
        model = cls.get_classifier(n_labels, n_priv_labels, "combined_classifier")
        history = run_model(model,
                            args,
                            train_ds=train_ds,
                            val_ds=val_ds,
                            log_dir=log_dir)
        logging.debug(history)


if __name__ == '__main__':
    main()
