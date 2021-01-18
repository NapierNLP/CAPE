import models
import data
import numpy as np
from tensorflow.keras.callbacks import CSVLogger
from contextlib import redirect_stdout
import tensorflow as tf
import argparse


N_LABELS = 5
N_PRIVS = 2
EMBED_LENGTH = 768


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .jsonl files for the task.",
    )

    parser.add_argument(
        "--max_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size for training.",
    )

    parser.add_argument(
        "--epochs",
        default=15,
        type=int,
        help="Total number of training epochs to perform.",
    )

    parser.add_argument(
        "--val_split",
        default=0.2,
        type=float,
        help="Percentage of training data to use in validation.",
    )

    args = parser.parse_args()

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    trustpilot_data = data.TrustDataset(args.data_dir, N_LABELS)

    classifier = models.StandardClassifier(N_LABELS, max_length=args.max_length)
    with open('base_cls_summary.txt', 'w') as f:
        with redirect_stdout(f):
            classifier.model.summary()

    x_train = classifier.tokenizer(trustpilot_data.train_data.text.values.tolist(),
                                   add_special_tokens=True,
                                   max_length=args.max_length,
                                   padding='max_length',
                                   return_token_type_ids=False,
                                   truncation=True,
                                   return_tensors='tf')
    y_train = trustpilot_data.train_labels
    x_test = classifier.tokenizer(trustpilot_data.test_data.text.values.tolist(),
                                  add_special_tokens=True,
                                  max_length=args.max_length,
                                  padding='max_length',
                                  return_token_type_ids=False,
                                  truncation=True,
                                  return_tensors='tf')
    y_test = trustpilot_data.test_labels

    cls_log = CSVLogger(filename='base_cls_train.csv')
    cls_test_log = CSVLogger(filename='base_cls_test.csv')

    classifier.model.fit({'input_ids': x_train['input_ids'], 'attention_mask': x_train['attention_mask']},
                         y_train,
                         epochs=args.epochs,
                         batch_size=args.batch_size,
                         validation_split=args.val_split,
                         callbacks=[cls_log])

    classifier.model.evaluate({'input_ids': x_test['input_ids'], 'attention_mask': x_test['attention_mask']},
                              y_test,
                              callbacks=[cls_test_log])

    attacker = models.AttackerClassifier(EMBED_LENGTH, N_PRIVS)
    with open('atk_cls_summary.txt', 'w') as f:
        with redirect_stdout(f):
            attacker.model.summary()

    a_x_train = np.asarray([np.squeeze(classifier.embedder(x, max_length=args.max_length, padding='max_length'))[0, :]
                 for x in trustpilot_data.train_data.text.values.tolist()])
    a_y_train = trustpilot_data.train_privs
    a_x_test = np.asarray([np.squeeze(classifier.embedder(x, max_length=args.max_length, padding='max_length'))[0, :]
                 for x in trustpilot_data.test_data.text.values.tolist()])
    a_y_test = trustpilot_data.test_privs

    a_log = CSVLogger(filename='atk_cls_train.csv')
    a_test_log = CSVLogger(filename='atk_cls_test.csv')

    attacker.model.fit(a_x_train,
                       a_y_train,
                       epochs=args.epochs,
                       batch_size=args.batch_size,
                       callbacks=[a_log])
    attacker.model.evaluate(a_x_test,
                            a_y_test,
                            callbacks=[a_test_log])


if __name__ == '__main__':
    main()